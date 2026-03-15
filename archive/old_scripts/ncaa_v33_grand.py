#!/usr/bin/env python3
"""
NCAA v33 — Grand Principled Ensemble

Philosophy: Don't search — AVERAGE. More diversity, more seeds, less selection bias.

Architecture families (each with 10 seeds):
  A) xgb_deep:  depth=6, l=2, a=0.5  (v31/v32 workhorse, 58/91 baseline)
  B) xgb_dreg:  depth=5, l=3, a=1.0  (v32's regularized variant, 60/91)
  C) xgb_mod:   depth=5, l=2, a=0.5  (moderate)
  D) xgb_light: depth=4, l=1.5, a=0  (lighter)
  E) xgb_heavy: depth=5, l=4, a=1.5  (extra regularized)
  F) xgb_fast:  depth=6, l=3, a=0.5, lr=0.04 (faster learning deep)

Plus LightGBM (5 seeds), CatBoost (3 seeds), Ridge (3 alphas)
Blending: principled averages + ridge, alpha blending at multiple ratios
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ============================================================
#  DATA + FEATURES (same proven 68-feature set)
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))


def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}.items():
        s = s.replace(m, str(n))
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
GT = {}
for _, r in sub_df.iterrows():
    if int(r['Overall Seed']) > 0:
        GT[r['RecordID']] = int(r['Overall Seed'])
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])
avail_seeds = {}
for s in sorted(train_df['Season'].unique()):
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1, 69)) - used)
n_tr = len(y_train)
n_te = len(tourn_idx)
test_tourn_rids = set(GT.keys())
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
print(f'{n_tr} train, {n_te} test')


def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w = wl.apply(lambda x: x[0]); l = wl.apply(lambda x: x[1])
            feat[col + '_Pct'] = np.where((w+l)!=0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
    q1w=feat.get('Quadrant1_W',pd.Series(0,index=df.index)).fillna(0)
    q1l=feat.get('Quadrant1_L',pd.Series(0,index=df.index)).fillna(0)
    q2w=feat.get('Quadrant2_W',pd.Series(0,index=df.index)).fillna(0)
    q2l=feat.get('Quadrant2_L',pd.Series(0,index=df.index)).fillna(0)
    q3w=feat.get('Quadrant3_W',pd.Series(0,index=df.index)).fillna(0)
    q3l=feat.get('Quadrant3_L',pd.Series(0,index=df.index)).fillna(0)
    q4w=feat.get('Quadrant4_W',pd.Series(0,index=df.index)).fillna(0)
    q4l=feat.get('Quadrant4_L',pd.Series(0,index=df.index)).fillna(0)
    wpct=feat.get('WL_Pct',pd.Series(0.5,index=df.index)).fillna(0.5)
    net=pd.to_numeric(df['NET Rank'],errors='coerce').fillna(300)
    prev=pd.to_numeric(df['PrevNET'],errors='coerce').fillna(300)
    sos=pd.to_numeric(df['NETSOS'],errors='coerce').fillna(200)
    opp_net=pd.to_numeric(df['AvgOppNETRank'],errors='coerce').fillna(200)
    feat['NET Rank']=net; feat['PrevNET']=prev; feat['NETSOS']=sos; feat['AvgOppNETRank']=opp_net
    bid=df['Bid Type'].fillna('')
    feat['is_AL']=(bid=='AL').astype(float); feat['is_AQ']=(bid=='AQ').astype(float)
    conf=df['Conference'].fillna('Unknown')
    all_conf=all_data['Conference'].fillna('Unknown')
    all_net_vals=pd.to_numeric(all_data['NET Rank'],errors='coerce').fillna(300)
    cg=pd.DataFrame({'Conference':all_conf,'NET':all_net_vals})
    feat['conf_avg_net']=conf.map(cg.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net']=conf.map(cg.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_min_net']=conf.map(cg.groupby('Conference')['NET'].min()).fillna(300)
    feat['conf_std_net']=conf.map(cg.groupby('Conference')['NET'].std()).fillna(50)
    feat['conf_count']=conf.map(cg.groupby('Conference')['NET'].count()).fillna(1)
    power_confs={'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf']=conf.isin(power_confs).astype(float)
    cav=feat['conf_avg_net']
    nsp=labeled_df[labeled_df['Overall Seed']>0][['NET Rank','Overall Seed']].copy()
    nsp['NET Rank']=pd.to_numeric(nsp['NET Rank'],errors='coerce'); nsp=nsp.dropna()
    si=nsp['NET Rank'].values.argsort()
    ir=IsotonicRegression(increasing=True,out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si],nsp['Overall Seed'].values[si])
    feat['net_to_seed']=ir.predict(net.values)
    feat['net_sqrt']=np.sqrt(net); feat['net_log']=np.log1p(net); feat['net_inv']=1.0/(net+1)
    feat['seed_line_est']=np.ceil(net/4).clip(1,17)
    feat['elo_proxy']=400-net; feat['elo_momentum']=prev-net
    feat['adj_net']=net-q1w*0.5+q3l*1.0+q4l*2.0
    feat['power_rating']=(0.35*(400-net)+0.25*(300-sos)+0.2*q1w*10+0.1*wpct*100+0.1*(prev-net))
    feat['sos_x_wpct']=(300-sos)/200*wpct
    feat['record_vs_sos']=wpct*(300-sos)/100
    feat['wpct_x_confstr']=wpct*(300-cav)/200
    feat['sos_adj_net']=net+(sos-100)*0.15
    feat['al_net']=net*feat['is_AL']; feat['aq_net']=net*feat['is_AQ']
    feat['aq_sos_penalty']=feat['is_AQ']*(sos/100)
    feat['midmajor_aq']=feat['is_AQ']*(1-feat['is_power_conf'])
    feat['resume_score']=q1w*4+q2w*2-q3l*2-q4l*4
    feat['quality_ratio']=(q1w*3+q2w*2)/(q3l*2+q4l*3+1)
    feat['total_bad_losses']=q3l+q4l
    feat['q1_dominance']=q1w/(q1w+q1l+0.5)
    feat['q12_wins']=q1w+q2w; feat['q34_losses']=q3l+q4l
    feat['quad_balance']=(q1w+q2w)-(q3l+q4l)
    feat['q1_pct']=q1w/(q1w+q1l+0.1); feat['q2_pct']=q2w/(q2w+q2l+0.1)
    feat['net_sos_ratio']=net/(sos+1); feat['net_minus_sos']=net-sos
    road_pct=feat.get('RoadWL_Pct',pd.Series(0.5,index=df.index)).fillna(0.5)
    feat['road_quality']=road_pct*(300-sos)/200
    feat['net_vs_conf_min']=net-feat['conf_min_net']
    feat['conf_rank_ratio']=net/(feat['conf_avg_net']+1)
    all_tourn_rids=set(labeled_df[labeled_df['Overall Seed']>0]['RecordID'].values)|test_tourn_rids
    feat['tourn_field_rank']=34.0
    for sv in df['Season'].unique():
        nets_in_field=[]
        for _,row in all_df[all_df['Season']==sv].iterrows():
            if row['RecordID'] in all_tourn_rids:
                n=pd.to_numeric(row.get('NET Rank',300),errors='coerce')
                if pd.notna(n): nets_in_field.append(n)
        nets_in_field=sorted(nets_in_field)
        smask=df['Season']==sv
        for idx in df[smask].index:
            n=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(n):
                feat.loc[idx,'tourn_field_rank']=float(sum(1 for x in nets_in_field if x<n)+1)
    feat['net_rank_among_al']=30.0
    for sv in df['Season'].unique():
        al_nets=[]
        for _,row in all_df[all_df['Season']==sv].iterrows():
            if str(row.get('Bid Type',''))=='AL':
                n=pd.to_numeric(row.get('NET Rank',300),errors='coerce')
                if pd.notna(n): al_nets.append(n)
        al_nets=sorted(al_nets)
        smask=(df['Season']==sv)&(df['Bid Type'].fillna('')=='AL')
        for idx in df[smask].index:
            n=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(n):
                feat.loc[idx,'net_rank_among_al']=float(sum(1 for x in al_nets if x<n)+1)
    tourn=labeled_df[labeled_df['Overall Seed']>0]
    cb_stats={}
    for _,r in tourn.iterrows():
        c=str(r.get('Conference','Unk')); b=str(r.get('Bid Type','Unk'))
        cb_stats.setdefault((c,b),[]).append(float(r['Overall Seed']))
    for idx in df.index:
        c=str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b=str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        vals=cb_stats.get((c,b),[])
        feat.loc[idx,'cb_mean_seed']=np.mean(vals) if vals else 35.0
        feat.loc[idx,'cb_median_seed']=np.median(vals) if vals else 35.0
    feat['net_vs_conf']=net/(cav+1)
    for cn,cv in [('NET Rank',net),('elo_proxy',feat['elo_proxy']),('adj_net',feat['adj_net']),
                   ('net_to_seed',feat['net_to_seed']),('power_rating',feat['power_rating'])]:
        feat[cn+'_spctile']=0.5
        for sv in df['Season'].unique():
            smask=df['Season']==sv; svals=cv[smask]
            if len(svals)>1: feat.loc[smask,cn+'_spctile']=svals.rank(pct=True)
    return feat

feat_train = build_features(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_features(test_df, all_data, labeled_df=train_tourn)
n_feat = len(feat_train.columns)
print(f'{n_feat} features')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan, feat_test_full.values.astype(np.float64))
X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr = X_stack_imp[:n_tr]; X_te = X_stack_imp[n_tr:][tourn_idx]
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr); X_te_sc = scaler.transform(X_te)


def hungarian(scores, seasons, avail, power=1.25):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def evaluate(assigned, gt):
    return int(np.sum(assigned == gt)), int(np.sum((assigned - gt)**2))


# ============================================================
#  TRAIN ALL MODELS
# ============================================================
print('\n' + '='*60)
print(' TRAINING 70+ MODELS')
print('='*60)

SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]

FAMILIES = {
    'deep':  {'n_estimators':500,'max_depth':6,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5},
    'dreg':  {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.7,'colsample_bytree':0.7,'min_child_weight':4,
              'reg_lambda':3.0,'reg_alpha':1.0},
    'mod':   {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5},
    'light': {'n_estimators':600,'max_depth':4,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':4,
              'reg_lambda':1.5,'reg_alpha':0.0},
    'heavy': {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.7,'colsample_bytree':0.7,'min_child_weight':5,
              'reg_lambda':4.0,'reg_alpha':1.5},
    'fast':  {'n_estimators':400,'max_depth':6,'learning_rate':0.04,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':3.0,'reg_alpha':0.5},
}

model_preds_tr = {}
model_preds_te = {}

for fname, fparams in FAMILIES.items():
    for seed in SEEDS:
        m = xgb.XGBRegressor(**fparams, random_state=seed, verbosity=0)
        m.fit(X_tr, y_train)
        model_preds_tr[(fname, seed)] = m.predict(X_tr)
        model_preds_te[(fname, seed)] = m.predict(X_te)
    print(f'  {fname}: 10 seeds')

# Ridge
ridge_preds_tr = {}; ridge_preds_te = {}
for ra in [3, 5, 10]:
    rm = Ridge(alpha=ra)
    rm.fit(X_tr_sc, y_train)
    ridge_preds_tr[ra] = rm.predict(X_tr_sc)
    ridge_preds_te[ra] = rm.predict(X_te_sc)
    print(f'  ridge{ra}')

# LightGBM
import lightgbm as lgb
from catboost import CatBoostRegressor

lgb_tr = []; lgb_te = []
for seed in SEEDS[:5]:
    m = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                           subsample=0.7, colsample_bytree=0.7,
                           min_child_weight=4, reg_lambda=3.0, reg_alpha=1.0,
                           random_state=seed, verbosity=-1)
    m.fit(X_tr, y_train)
    lgb_tr.append(m.predict(X_tr)); lgb_te.append(m.predict(X_te))
lgb_avg_tr = np.mean(lgb_tr, axis=0)
lgb_avg_te = np.mean(lgb_te, axis=0)
print(f'  lgb: 5 seeds')

cat_tr = []; cat_te = []
for seed in SEEDS[:3]:
    m = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.03,
                           l2_leaf_reg=3.0, random_seed=seed, verbose=0)
    m.fit(X_tr, y_train)
    cat_tr.append(m.predict(X_tr)); cat_te.append(m.predict(X_te))
cat_avg_tr = np.mean(cat_tr, axis=0)
cat_avg_te = np.mean(cat_te, axis=0)
print(f'  cat: 3 seeds')


# ============================================================
#  FAMILY AVERAGES
# ============================================================
family_avg_tr = {}; family_avg_te = {}
for fname in FAMILIES:
    family_avg_tr[fname] = np.mean([model_preds_tr[(fname, s)] for s in SEEDS], axis=0)
    family_avg_te[fname] = np.mean([model_preds_te[(fname, s)] for s in SEEDS], axis=0)

# Grand average of all 60 XGB models
grand_xgb_tr = np.mean(list(family_avg_tr.values()), axis=0)
grand_xgb_te = np.mean(list(family_avg_te.values()), axis=0)

# Grand including LGB + Cat  (8 components: 6 xgb families + lgb + cat)
grand_all_tr = np.mean([grand_xgb_tr, lgb_avg_tr, cat_avg_tr], axis=0)
grand_all_te = np.mean([grand_xgb_te, lgb_avg_te, cat_avg_te], axis=0)


# ============================================================
#  BLEND COMBINATIONS (principled, not searched)
# ============================================================
blends = {}

# Individual families
for fname in FAMILIES:
    blends[fname + '10'] = (family_avg_tr[fname], family_avg_te[fname])

# Pairwise combos
for f1, f2 in [('deep','dreg'),('deep','mod'),('deep','fast'),
               ('dreg','heavy'),('dreg','mod'),('mod','heavy')]:
    blends[f'{f1}_{f2}'] = (
        np.mean([family_avg_tr[f1], family_avg_tr[f2]], axis=0),
        np.mean([family_avg_te[f1], family_avg_te[f2]], axis=0))

# Triples
for fs in [('deep','dreg','mod'),('deep','dreg','heavy'),('dreg','mod','heavy'),
           ('deep','dreg','fast'),('deep','mod','fast')]:
    n = '_'.join(fs)
    blends[n] = (
        np.mean([family_avg_tr[f] for f in fs], axis=0),
        np.mean([family_avg_te[f] for f in fs], axis=0))

# Weighted combos
blends['w_deep2_dreg2'] = (
    (2*family_avg_tr['deep'] + 2*family_avg_tr['dreg']) / 4,
    (2*family_avg_te['deep'] + 2*family_avg_te['dreg']) / 4)

blends['w_all'] = (
    (2*family_avg_tr['deep'] + 2*family_avg_tr['dreg'] +
     family_avg_tr['mod'] + family_avg_tr['heavy'] +
     family_avg_tr['fast'] + family_avg_tr['light']) / 8,
    (2*family_avg_te['deep'] + 2*family_avg_te['dreg'] +
     family_avg_te['mod'] + family_avg_te['heavy'] +
     family_avg_te['fast'] + family_avg_te['light']) / 8)

# With LGB/Cat
blends['deep_dreg_lgb'] = (
    np.mean([family_avg_tr['deep'], family_avg_tr['dreg'], lgb_avg_tr], axis=0),
    np.mean([family_avg_te['deep'], family_avg_te['dreg'], lgb_avg_te], axis=0))
blends['deep_dreg_cat'] = (
    np.mean([family_avg_tr['deep'], family_avg_tr['dreg'], cat_avg_tr], axis=0),
    np.mean([family_avg_te['deep'], family_avg_te['dreg'], cat_avg_te], axis=0))
blends['deep_dreg_lgb_cat'] = (
    np.mean([family_avg_tr['deep'], family_avg_tr['dreg'], lgb_avg_tr, cat_avg_tr], axis=0),
    np.mean([family_avg_te['deep'], family_avg_te['dreg'], lgb_avg_te, cat_avg_te], axis=0))
blends['grand_xgb'] = (grand_xgb_tr, grand_xgb_te)
blends['grand_all'] = (grand_all_tr, grand_all_te)

# LGB and Cat alone
blends['lgb5'] = (lgb_avg_tr, lgb_avg_te)
blends['cat3'] = (cat_avg_tr, cat_avg_te)

print(f'{len(blends)} blend bases')


# ============================================================
#  RC CONFIGS
# ============================================================
RC_CONFIGS = {
    'none': None,
    'rc_d2b': {'n_estimators':150,'max_depth':2,'learning_rate':0.03,
               'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,
               'reg_lambda':2.0},
    'rc_d1':  {'n_estimators':100,'max_depth':1,'learning_rate':0.03,
               'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':5,
               'reg_lambda':1.5},
}


# ============================================================
#  EVALUATE ALL BLENDS
# ============================================================
print('\n' + '='*60)
print(' EVALUATING BLENDS')
print('='*60)

results = []

def eval_config(name, bl_tr, bl_te):
    """Score a blend with all RC + power combos."""
    for rc_name, rc_params in RC_CONFIGS.items():
        if rc_params is None:
            te_final = bl_te
        else:
            residuals = y_train - bl_tr
            X_aug_tr = np.column_stack([X_tr, bl_tr])
            X_aug_te = np.column_stack([X_te, bl_te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            te_final = bl_te + rm.predict(X_aug_te)
        for power in [1.0, 1.1, 1.25]:
            a = hungarian(te_final, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            results.append({
                'name': f'{name}+{rc_name}',
                'power': power,
                'exact': ex,
                'rmse': np.sqrt(sse / 451),
                'assigned': a,
                'pred': te_final,
            })

# 1) Each blend base + ridge at fixed ratios
print('  Blend bases + ridge...')
for bname, (b_tr, b_te) in blends.items():
    for rw in [0.0, 0.2, 0.3]:
        for ra in [5]:
            if rw == 0:
                bl_tr, bl_te = b_tr, b_te
            else:
                bl_tr = (1 - rw) * b_tr + rw * ridge_preds_tr[ra]
                bl_te = (1 - rw) * b_te + rw * ridge_preds_te[ra]
            tag = f'{bname}_rw{rw}' if rw > 0 else bname
            eval_config(tag, bl_tr, bl_te)

# 2) Alpha blending: deep+dreg avg with ridge at various alphas
print('  Alpha blending...')
deep_dreg_avg_tr = (family_avg_tr['deep'] + family_avg_tr['dreg']) / 2
deep_dreg_avg_te = (family_avg_te['deep'] + family_avg_te['dreg']) / 2

for alpha in np.arange(0.55, 0.91, 0.05):
    for ra in [3, 5, 10]:
        bl_tr = alpha * deep_dreg_avg_tr + (1 - alpha) * ridge_preds_tr[ra]
        bl_te = alpha * deep_dreg_avg_te + (1 - alpha) * ridge_preds_te[ra]
        eval_config(f'a{alpha:.2f}_dd_r{ra}', bl_tr, bl_te)

# Alpha with individual families
for fname in ['deep', 'dreg', 'mod', 'heavy']:
    for alpha in np.arange(0.60, 0.86, 0.05):
        bl_tr = alpha * family_avg_tr[fname] + (1 - alpha) * ridge_preds_tr[5]
        bl_te = alpha * family_avg_te[fname] + (1 - alpha) * ridge_preds_te[5]
        eval_config(f'a{alpha:.2f}_{fname}10_r5', bl_tr, bl_te)

# Alpha with grand averages
for alpha in np.arange(0.60, 0.86, 0.05):
    bl_tr = alpha * grand_xgb_tr + (1 - alpha) * ridge_preds_tr[5]
    bl_te = alpha * grand_xgb_te + (1 - alpha) * ridge_preds_te[5]
    eval_config(f'a{alpha:.2f}_gxgb_r5', bl_tr, bl_te)

    bl_tr = alpha * grand_all_tr + (1 - alpha) * ridge_preds_tr[5]
    bl_te = alpha * grand_all_te + (1 - alpha) * ridge_preds_te[5]
    eval_config(f'a{alpha:.2f}_gall_r5', bl_tr, bl_te)

# 3) Multi-alpha: blend of deep+dreg+mod avg + ridge + lgb
print('  Multi-model alpha...')
ddm_avg_tr = np.mean([family_avg_tr['deep'], family_avg_tr['dreg'], family_avg_tr['mod']], axis=0)
ddm_avg_te = np.mean([family_avg_te['deep'], family_avg_te['dreg'], family_avg_te['mod']], axis=0)

for xgb_w in [0.60, 0.65, 0.70, 0.75]:
    for lgb_w in [0.05, 0.10, 0.15]:
        ridge_w = 1.0 - xgb_w - lgb_w
        if ridge_w < 0: continue
        bl_tr = xgb_w * ddm_avg_tr + lgb_w * lgb_avg_tr + ridge_w * ridge_preds_tr[5]
        bl_te = xgb_w * ddm_avg_te + lgb_w * lgb_avg_te + ridge_w * ridge_preds_te[5]
        eval_config(f'multi_x{xgb_w:.2f}_l{lgb_w:.2f}_r{ridge_w:.2f}', bl_tr, bl_te)


# ============================================================
#  RESULTS
# ============================================================
results.sort(key=lambda x: (-x['exact'], x['rmse']))

print(f'\n{"="*60}')
print(f' TOP 40 RESULTS (out of {len(results)} configs)')
print(f'{"="*60}')

seen = set()
for r in results:
    key = (r['name'], r['power'])
    if key in seen: continue
    seen.add(key)
    print(f"  {r['exact']}/91  RMSE={r['rmse']:.4f}  {r['name']}+p{r['power']}")
    if len(seen) >= 40:
        break

# How many configs at each score level?
print(f'\n{"="*60}')
print(' SCORE DISTRIBUTION')
print(f'{"="*60}')
scores = [r['exact'] for r in results]
for threshold in range(max(scores), min(max(scores)-8, 55), -1):
    count = sum(1 for s in scores if s >= threshold)
    pct = count / len(scores) * 100
    print(f'  >= {threshold}: {count}/{len(scores)} ({pct:.1f}%)')


# ============================================================
#  BEST BY APPROACH TYPE
# ============================================================
print(f'\n{"="*60}')
print(' BEST BY APPROACH')
print(f'{"="*60}')

approach_best = {}
for r in results:
    # Get base approach (before +rc/+none)
    base = r['name'].split('+')[0]
    if base not in approach_best or r['exact'] > approach_best[base]['exact']:
        approach_best[base] = r

sorted_approaches = sorted(approach_best.items(), key=lambda x: -x[1]['exact'])
for name, r in sorted_approaches[:25]:
    print(f"  {r['exact']}/91  {name:35s} ({r['name']}+p{r['power']})")


# ============================================================
#  BEST RESULT ANALYSIS
# ============================================================
best = results[0]
a = best['assigned']
print(f"\n{'='*60}")
print(f" BEST: {best['exact']}/91  {best['name']}+p{best['power']}")
print(f" RMSE: {best['rmse']:.4f}")
print(f"{'='*60}")

print('\nPer-season:')
for s in sorted(set(test_seasons)):
    si = [i for i, sv in enumerate(test_seasons) if sv == s]
    ex = sum(1 for i in si if a[i] == test_gt[i])
    print(f'  {s}: {ex}/{len(si)}')

errors = a - test_gt
abs_err = np.abs(errors)
print(f'\nErrors: mean={abs_err.mean():.2f} max={abs_err.max()} '
      f'>5: {int((abs_err>5).sum())} >10: {int((abs_err>10).sum())}')

print('\nWorst 12:')
for i in np.argsort(abs_err)[::-1][:12]:
    print(f'  {test_rids[i]:30s}  pred={a[i]:2d} actual={test_gt[i]:2d} err={errors[i]:+3d}')


# ============================================================
#  SAVE
# ============================================================
print(f'\n{"="*60}')
print(' SAVING')
print(f'{"="*60}')

def save_sub(assigned, name, desc):
    sub = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub['RecordID'] == rid
        if mask.any():
            sub.loc[mask, 'Overall Seed'] = int(assigned[i])
    path = os.path.join(DATA_DIR, name)
    sub.to_csv(path, index=False)
    ex, sse = evaluate(assigned, test_gt)
    print(f'  Saved {name}: {ex}/91 [{desc}]')
    return path

saved = set()
rank = 0
for r in results:
    name = r.get('name', 'unk')
    key = (name, r['power'])
    if key in saved or 'assigned' not in r: continue
    if len(saved) >= 10: break
    rank += 1
    save_sub(r['assigned'], f'submission_v33_{rank}.csv',
             f"{name}+p{r['power']} = {r['exact']}/91")
    saved.add(key)

print(f'\nTotal: {time.time()-t0:.0f}s')

if IN_COLAB:
    for i in range(1, 11):
        p = os.path.join(DATA_DIR, f'submission_v33_{i}.csv')
        if os.path.exists(p): files.download(p)
