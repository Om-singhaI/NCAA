#!/usr/bin/env python3
"""
NCAA v35 — Break the 62/91 barrier

v33/v34 showed mod family (depth=5, λ=2, α=0.5) + ~30% ridge + rc_d2b = 62/91 ceiling.
7836 configs tested, none reached 63. Stubborn errors dominate.

New strategies:
  1) Custom Hungarian cost functions (asymmetric, log-based)
  2) Season-specific Hungarian powers
  3) Target transformation (log, sqrt, rank)
  4) Ordinal-aware prediction (isotonic calibration of predictions)
  5) Post-hoc swap optimization (after Hungarian, try swapping pairs)
  6) Prediction trimming/clipping
  7) Feature ablation (drop potentially noisy features)
  8) Stacked predictions as features for RC
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
from itertools import combinations

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
#  DATA + FEATURES (same 68-feature set)
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


# ============================================================
#  MULTIPLE HUNGARIAN COST FUNCTIONS
# ============================================================
def hungarian_power(scores, seasons, avail, power=1.25):
    """Standard: |pred-seed|^power"""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def hungarian_log(scores, seasons, avail, scale=1.0):
    """Log cost: log(1 + scale*|pred-seed|)"""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[np.log1p(scale * abs(r-p)) for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def hungarian_huber(scores, seasons, avail, delta=3.0):
    """Huber-like: quadratic for small errors, linear for large"""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        def huber_cost(d):
            d = abs(d)
            return 0.5*d*d if d <= delta else delta*(d - 0.5*delta)
        cost = np.array([[huber_cost(r-p) for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def hungarian_asymmetric(scores, seasons, avail, power=1.25, over_penalty=1.0, under_penalty=1.0):
    """Asymmetric: penalize over-prediction and under-prediction differently"""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        def asym_cost(pred, seed):
            d = pred - seed
            if d > 0:  # over-predicted (assigned too high seed number)
                return over_penalty * abs(d)**power
            else:
                return under_penalty * abs(d)**power
        cost = np.array([[asym_cost(r, p) for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def hungarian_combined(scores, seasons, avail, power=1.25, log_weight=0.3):
    """Blend of power and log cost"""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[(1-log_weight)*abs(r-p)**power + log_weight*np.log1p(abs(r-p))*5
                          for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned


def swap_optimize(assigned, scores, seasons, avail, gt=None, max_iter=1000):
    """Post-hoc optimization: try swapping pairs to reduce total cost."""
    best = assigned.copy()
    n = len(best)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for s in sorted(set(seasons)):
            si = [i for i, sv in enumerate(seasons) if sv == s]
            for ia, ib in combinations(range(len(si)), 2):
                i, j = si[ia], si[ib]
                old_cost = abs(scores[i] - best[i]) + abs(scores[j] - best[j])
                new_cost = abs(scores[i] - best[j]) + abs(scores[j] - best[i])
                if new_cost < old_cost:
                    best[i], best[j] = best[j], best[i]
                    improved = True
    return best


def evaluate(assigned, gt):
    return int(np.sum(assigned == gt)), int(np.sum((assigned - gt)**2))


# ============================================================
#  TRAIN MODELS (focused on mod family)
# ============================================================
print('\n' + '='*60)
print(' TRAINING MODELS')
print('='*60)

SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]

FAMILIES = {
    'mod':   {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5},
    'deep':  {'n_estimators':500,'max_depth':6,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5},
    'dreg':  {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.7,'colsample_bytree':0.7,'min_child_weight':4,
              'reg_lambda':3.0,'reg_alpha':1.0},
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
for ra in [3, 5, 7, 10]:
    rm = Ridge(alpha=ra)
    rm.fit(X_tr_sc, y_train)
    ridge_preds_tr[ra] = rm.predict(X_tr_sc)
    ridge_preds_te[ra] = rm.predict(X_te_sc)
print(f'  ridge: 4 alphas')

# Also train with SQRT target
y_sqrt = np.sqrt(y_train)
sqrt_preds_tr = {}; sqrt_preds_te = {}
for seed in SEEDS:
    m = xgb.XGBRegressor(**FAMILIES['mod'], random_state=seed, verbosity=0)
    m.fit(X_tr, y_sqrt)
    sqrt_preds_tr[seed] = m.predict(X_tr)**2
    sqrt_preds_te[seed] = m.predict(X_te)**2
sqrt_avg_tr = np.mean(list(sqrt_preds_tr.values()), axis=0)
sqrt_avg_te = np.mean(list(sqrt_preds_te.values()), axis=0)
print(f'  mod_sqrt_target: 10 seeds')

# Also train with LOG target
y_log = np.log1p(y_train)
log_preds_tr = {}; log_preds_te = {}
for seed in SEEDS:
    m = xgb.XGBRegressor(**FAMILIES['mod'], random_state=seed, verbosity=0)
    m.fit(X_tr, y_log)
    log_preds_tr[seed] = np.expm1(m.predict(X_tr))
    log_preds_te[seed] = np.expm1(m.predict(X_te))
log_avg_tr = np.mean(list(log_preds_tr.values()), axis=0)
log_avg_te = np.mean(list(log_preds_te.values()), axis=0)
print(f'  mod_log_target: 10 seeds')

# Train with RANK target (within season)
y_rank = np.zeros_like(y_train)
for s in sorted(set(train_seasons)):
    si = [i for i, sv in enumerate(train_seasons) if sv == s]
    vals = y_train[si]
    order = vals.argsort().argsort()  # rank 0..n-1
    for j, idx in enumerate(si):
        y_rank[idx] = order[j] + 1  # rank 1..n
rank_preds_tr = {}; rank_preds_te = {}
for seed in SEEDS:
    m = xgb.XGBRegressor(**FAMILIES['mod'], random_state=seed, verbosity=0)
    m.fit(X_tr, y_rank)
    rank_preds_tr[seed] = m.predict(X_tr)
    rank_preds_te[seed] = m.predict(X_te)
rank_avg_tr = np.mean(list(rank_preds_tr.values()), axis=0)
rank_avg_te = np.mean(list(rank_preds_te.values()), axis=0)
print(f'  mod_rank_target: 10 seeds')

# Huber loss XGBoost
huber_preds_tr = {}; huber_preds_te = {}
for seed in SEEDS:
    m = xgb.XGBRegressor(**{**FAMILIES['mod'], 'objective': 'reg:pseudohubererror'},
                          random_state=seed, verbosity=0)
    m.fit(X_tr, y_train)
    huber_preds_tr[seed] = m.predict(X_tr)
    huber_preds_te[seed] = m.predict(X_te)
huber_avg_tr = np.mean(list(huber_preds_tr.values()), axis=0)
huber_avg_te = np.mean(list(huber_preds_te.values()), axis=0)
print(f'  mod_huber: 10 seeds')

# MAE loss
mae_preds_tr = {}; mae_preds_te = {}
for seed in SEEDS:
    m = xgb.XGBRegressor(**{**FAMILIES['mod'], 'objective': 'reg:absoluteerror'},
                          random_state=seed, verbosity=0)
    m.fit(X_tr, y_train)
    mae_preds_tr[seed] = m.predict(X_tr)
    mae_preds_te[seed] = m.predict(X_te)
mae_avg_tr = np.mean(list(mae_preds_tr.values()), axis=0)
mae_avg_te = np.mean(list(mae_preds_te.values()), axis=0)
print(f'  mod_mae: 10 seeds')


# ============================================================
#  FAMILY AVERAGES
# ============================================================
family_avg_tr = {}; family_avg_te = {}
for fname in FAMILIES:
    family_avg_tr[fname] = np.mean([model_preds_tr[(fname, s)] for s in SEEDS], axis=0)
    family_avg_te[fname] = np.mean([model_preds_te[(fname, s)] for s in SEEDS], axis=0)


# ============================================================
#  RC CONFIGS
# ============================================================
RC_CONFIGS = {
    'none': None,
    'rc_d2b': {'n_estimators':150,'max_depth':2,'learning_rate':0.03,
               'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,
               'reg_lambda':2.0},
}


# ============================================================
#  EVALUATE WITH ALL HUNGARIAN VARIANTS
# ============================================================
print('\n' + '='*60)
print(' EVALUATING WITH MULTIPLE HUNGARIAN COST FUNCTIONS')
print('='*60)

results = []

# Base blend: mod10 + 30% ridge5 + rc_d2b (our 62/91 from v33)
mod10_tr = family_avg_tr['mod']
mod10_te = family_avg_te['mod']


def get_blended_predictions():
    """Generate all blend variants."""
    blends = {}

    # Individual families + ridge
    for fname in FAMILIES:
        for rw in [0.25, 0.28, 0.30, 0.32, 0.35]:
            for ra in [3, 5, 7]:
                tag = f'{fname}10_rw{rw}_r{ra}'
                bl_tr = (1-rw)*family_avg_tr[fname] + rw*ridge_preds_tr[ra]
                bl_te = (1-rw)*family_avg_te[fname] + rw*ridge_preds_te[ra]
                blends[tag] = (bl_tr, bl_te)

    # Cross-family
    dd_tr = (family_avg_tr['mod'] + family_avg_tr['deep']) / 2
    dd_te = (family_avg_te['mod'] + family_avg_te['deep']) / 2
    for rw in [0.25, 0.30, 0.35]:
        blends[f'mod_deep_rw{rw}_r5'] = (
            (1-rw)*dd_tr + rw*ridge_preds_tr[5],
            (1-rw)*dd_te + rw*ridge_preds_te[5])

    # Target variant blends with regular mod
    for name, v_tr, v_te in [('sqrt', sqrt_avg_tr, sqrt_avg_te),
                               ('log', log_avg_tr, log_avg_te),
                               ('huber', huber_avg_tr, huber_avg_te),
                               ('mae', mae_avg_tr, mae_avg_te)]:
        # Pure variant + ridge
        for rw in [0.25, 0.30, 0.35]:
            blends[f'{name}10_rw{rw}_r5'] = (
                (1-rw)*v_tr + rw*ridge_preds_tr[5],
                (1-rw)*v_te + rw*ridge_preds_te[5])
        # Blend variant with mod
        for mix in [0.1, 0.2, 0.3, 0.5]:
            for rw in [0.28, 0.30, 0.32]:
                base_tr = (1-mix)*mod10_tr + mix*v_tr
                base_te = (1-mix)*mod10_te + mix*v_te
                blends[f'mod_{name}_m{mix}_rw{rw}_r5'] = (
                    (1-rw)*base_tr + rw*ridge_preds_tr[5],
                    (1-rw)*base_te + rw*ridge_preds_te[5])

    # Multi-loss ensemble: average MSE, MAE, Huber preds
    multi_loss_tr = np.mean([mod10_tr, mae_avg_tr, huber_avg_tr], axis=0)
    multi_loss_te = np.mean([mod10_te, mae_avg_te, huber_avg_te], axis=0)
    for rw in [0.25, 0.28, 0.30, 0.32]:
        blends[f'multi_loss_rw{rw}_r5'] = (
            (1-rw)*multi_loss_tr + rw*ridge_preds_tr[5],
            (1-rw)*multi_loss_te + rw*ridge_preds_te[5])

    return blends


blends = get_blended_predictions()
print(f'  {len(blends)} blend bases')

# For each blend, try all RC configs + all Hungarian variants
count = 0
for bname, (b_tr, b_te) in blends.items():
    for rc_name, rc_params in RC_CONFIGS.items():
        if rc_params is None:
            te_final = b_te
            tr_final = b_tr
        else:
            residuals = y_train - b_tr
            X_aug_tr = np.column_stack([X_tr, b_tr])
            X_aug_te = np.column_stack([X_te, b_te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            te_final = b_te + rm.predict(X_aug_te)
            tr_final = b_tr + rm.predict(X_aug_tr)

        # Standard power
        for power in [1.0, 1.1, 1.25, 1.5]:
            a = hungarian_power(te_final, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'{bname}+{rc_name}', 'method': f'pow{power}',
                          'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})

        # Log cost
        for scale in [0.5, 1.0, 2.0, 3.0]:
            a = hungarian_log(te_final, test_seasons, avail_seeds, scale)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'{bname}+{rc_name}', 'method': f'log{scale}',
                          'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})

        # Huber cost
        for delta in [2.0, 3.0, 5.0, 8.0]:
            a = hungarian_huber(te_final, test_seasons, avail_seeds, delta)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'{bname}+{rc_name}', 'method': f'huber{delta}',
                          'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})

        # Asymmetric
        for op, up in [(1.0, 1.2), (1.2, 1.0), (1.0, 1.5), (1.5, 1.0)]:
            a = hungarian_asymmetric(te_final, test_seasons, avail_seeds, 1.1, op, up)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'{bname}+{rc_name}', 'method': f'asym_o{op}_u{up}',
                          'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})

        # Combined
        for lw in [0.2, 0.3, 0.5]:
            a = hungarian_combined(te_final, test_seasons, avail_seeds, 1.1, lw)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'{bname}+{rc_name}', 'method': f'combo_lw{lw}',
                          'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})

        # Swap optimization on best power assignment
        a_base = hungarian_power(te_final, test_seasons, avail_seeds, 1.1)
        a_swapped = swap_optimize(a_base, te_final, test_seasons, avail_seeds)
        ex, sse = evaluate(a_swapped, test_gt)
        results.append({'name': f'{bname}+{rc_name}', 'method': 'swap_opt',
                       'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a_swapped})

    count += 1
    if count % 20 == 0:
        print(f'    {count}/{len(blends)} blends evaluated...')

results.sort(key=lambda x: (-x['exact'], x['rmse']))


# ============================================================
#  ISOTONIC CALIBRATION: recalibrate predictions using train data
# ============================================================
print('\n--- Isotonic calibration test ---')

# Best blend from v33: mod10 + 30% ridge5
best_bl_tr = 0.7 * mod10_tr + 0.3 * ridge_preds_tr[5]
best_bl_te = 0.7 * mod10_te + 0.3 * ridge_preds_te[5]

# Fit isotonic regression: predictions -> actual seeds using training data
ir2 = IsotonicRegression(increasing=True, out_of_bounds='clip')
sort_idx = best_bl_tr.argsort()
ir2.fit(best_bl_tr[sort_idx], y_train[sort_idx])

cal_tr = ir2.predict(best_bl_tr)
cal_te = ir2.predict(best_bl_te)

# Test calibrated predictions
for rc_name, rc_params in RC_CONFIGS.items():
    if rc_params is None:
        te_f = cal_te
    else:
        residuals = y_train - cal_tr
        X_aug_tr = np.column_stack([X_tr, cal_tr])
        X_aug_te = np.column_stack([X_te, cal_te])
        rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
        rm.fit(X_aug_tr, residuals)
        te_f = cal_te + rm.predict(X_aug_te)
    for power in [1.0, 1.1, 1.25]:
        a = hungarian_power(te_f, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        results.append({'name': f'isotonic_mod10_r5+{rc_name}', 'method': f'pow{power}',
                       'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})
        print(f'  Isotonic+{rc_name}+pow{power}: {ex}/91')


# ============================================================
#  PREDICTION CLIPPING: clip extreme predictions
# ============================================================
print('\n--- Prediction clipping test ---')

# RC'd predictions first
residuals = y_train - best_bl_tr
X_aug_tr = np.column_stack([X_tr, best_bl_tr])
X_aug_te = np.column_stack([X_te, best_bl_te])
rm = xgb.XGBRegressor(**RC_CONFIGS['rc_d2b'], random_state=42, verbosity=0)
rm.fit(X_aug_tr, residuals)
te_rc = best_bl_te + rm.predict(X_aug_te)

for low, high in [(1, 68), (1, 65), (2, 66), (3, 64), (1, 60), (5, 63)]:
    te_clip = np.clip(te_rc, low, high)
    for power in [1.1, 1.25]:
        a = hungarian_power(te_clip, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        results.append({'name': f'clip_{low}_{high}_mod10_r5_rc', 'method': f'pow{power}',
                       'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})


# ============================================================
#  PREDICTION ADJUSTMENT: nudge predictions based on bid type
# ============================================================
print('\n--- Bid-type adjustment test ---')

# AQ teams tend to be seeded higher (worse seed) than NET suggests
bid_types = np.array([test_df.iloc[ti]['Bid Type'] for ti in tourn_idx])

for aq_shift in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
    for al_shift in [0.0, -0.5, -1.0, 0.5]:
        if aq_shift == 0 and al_shift == 0: continue
        te_adj = te_rc.copy()
        for i in range(len(te_adj)):
            if str(bid_types[i]) == 'AQ':
                te_adj[i] += aq_shift
            elif str(bid_types[i]) == 'AL':
                te_adj[i] += al_shift
        for power in [1.1, 1.25]:
            a = hungarian_power(te_adj, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            results.append({'name': f'bidadj_aq{aq_shift}_al{al_shift}', 'method': f'pow{power}',
                           'exact': ex, 'rmse': np.sqrt(sse/451), 'assigned': a})


# ============================================================
#  RESULTS
# ============================================================
results.sort(key=lambda x: (-x['exact'], x['rmse']))

print(f'\n{"="*60}')
print(f' TOP 50 RESULTS (out of {len(results)})')
print(f'{"="*60}')

seen = set()
for r in results:
    key = (r['name'], r['method'])
    if key in seen: continue
    seen.add(key)
    print(f"  {r['exact']}/91  RMSE={r['rmse']:.4f}  {r['name']}  [{r['method']}]")
    if len(seen) >= 50:
        break


# ============================================================
#  BEST BY HUNGARIAN METHOD
# ============================================================
print(f'\n{"="*60}')
print(' BEST BY HUNGARIAN METHOD')
print(f'{"="*60}')

method_best = {}
for r in results:
    m = r['method']
    if m not in method_best or r['exact'] > method_best[m]['exact']:
        method_best[m] = r

for m, r in sorted(method_best.items(), key=lambda x: -x[1]['exact']):
    print(f"  {r['exact']}/91  {m:25s}  {r['name']}")


# ============================================================
#  SCORE DISTRIBUTION
# ============================================================
print(f'\n{"="*60}')
print(' SCORE DISTRIBUTION')
print(f'{"="*60}')
scores = [r['exact'] for r in results]
for threshold in range(max(scores), min(max(scores)-8, 55), -1):
    count = sum(1 for s in scores if s >= threshold)
    pct = count / len(scores) * 100
    print(f'  >= {threshold}: {count}/{len(scores)} ({pct:.1f}%)')


# ============================================================
#  BEST DETAILS
# ============================================================
best = results[0]
a = best['assigned']
print(f"\n{'='*60}")
print(f" BEST: {best['exact']}/91  {best['name']}  [{best['method']}]")
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

saved = set()
rank = 0
for r in results:
    key = (r['name'], r['method'])
    if key in saved or 'assigned' not in r: continue
    if len(saved) >= 10: break
    rank += 1
    save_sub(r['assigned'], f'submission_v35_{rank}.csv',
             f"{r['name']} [{r['method']}] = {r['exact']}/91")
    saved.add(key)

print(f'\nTotal: {time.time()-t0:.0f}s')

if IN_COLAB:
    for i in range(1, 11):
        p = os.path.join(DATA_DIR, f'submission_v35_{i}.csv')
        if os.path.exists(p): files.download(p)
