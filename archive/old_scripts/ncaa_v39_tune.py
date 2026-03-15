#!/usr/bin/env python3
"""
NCAA v39 — LOSO-Validated Hyperparameter Tuning

GOAL: Find hyperparameters that minimize LOSO RMSE (generalization metric).
NOT optimizing for test exact matches (that's overfitting).

From v37 root cause analysis:
  - LOSO RMSE correlates r=0.70 with test RMSE → reliable signal
  - LOSO exact correlates r=0.39 with test exact → noisy, don't use
  - Current best LOSO RMSE = 4.132 (mod20+r30)
  - RC adds +4 test exact for only +1% LOSO RMSE → genuine improvement

Search strategy:
  1. Coarse grid over key XGB params → find LOSO-RMSE sweet spot
  2. Fine grid around best region
  3. Ridge alpha + blend weight tuning
  4. RC depth/params tuning
  5. Final model = best LOSO-RMSE config
"""

import os, sys, time, re, warnings, itertools
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm'])
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
#  DATA
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
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

test_avail = {}
for s in sorted(set(test_seasons)):
    sp = str(s)
    used = set(train_tourn[train_tourn['Season'].astype(str)==sp]['Overall Seed'].astype(int))
    al = sorted(set(range(1,69))-used)
    test_avail[s] = al; test_avail[sp] = al; test_avail[np.str_(sp)] = al

n_tr = len(y_train); n_te = len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    bt = row.get('Bid Type', '')
    if pd.notna(bt) and str(bt) in ('AL','AQ'):
        all_tourn_rids.add(row['RecordID'])
print(f'{n_tr} train, {n_te} test, {len(all_tourn_rids)} tourn teams')

# ============================================================
#  FEATURES
# ============================================================
def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL','Conf.Record','Non-ConferenceRecord','RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w=wl.apply(lambda x:x[0]); l=wl.apply(lambda x:x[1])
            feat[col+'_Pct']=np.where((w+l)!=0,w/(w+l),0.5)
            if col=='WL': feat['total_W']=w; feat['total_L']=l; feat['total_games']=w+l
    for q in ['Quadrant1','Quadrant2','Quadrant3','Quadrant4']:
        if q in df.columns:
            wl=df[q].apply(parse_wl)
            feat[q+'_W']=wl.apply(lambda x:x[0]); feat[q+'_L']=wl.apply(lambda x:x[1])
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
    all_conf=all_df['Conference'].fillna('Unknown')
    all_net_vals=pd.to_numeric(all_df['NET Rank'],errors='coerce').fillna(300)
    cg=pd.DataFrame({'Conference':all_conf,'NET':all_net_vals})
    cs=cg.groupby('Conference')['NET']
    feat['conf_avg_net']=conf.map(cs.mean()).fillna(200)
    feat['conf_med_net']=conf.map(cs.median()).fillna(200)
    feat['conf_min_net']=conf.map(cs.min()).fillna(300)
    feat['conf_std_net']=conf.map(cs.std()).fillna(50)
    feat['conf_count']=conf.map(cs.count()).fillna(1)
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
    feat['tourn_field_rank']=34.0
    for sv in df['Season'].unique():
        nets_in_field=[]
        for _,row in all_df[all_df['Season']==sv].iterrows():
            if row['RecordID'] in tourn_rids:
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

feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test_full = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'{n_feat} features')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)),np.nan,feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)),np.nan,feat_test_full.values.astype(np.float64))
X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr = X_stack_imp[:n_tr]; X_te = X_stack_imp[n_tr:][tourn_idx]
sc_global = StandardScaler()
X_tr_sc = sc_global.fit_transform(X_tr); X_te_sc = sc_global.transform(X_te)

def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i,sv in enumerate(seasons) if str(sv)==str(s)]
        sk = s if s in avail else str(s)
        pos = avail[sk]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri,ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]]=pos[c]
    return assigned

def evaluate(assigned, gt):
    return int(np.sum(assigned==gt)), int(np.sum((assigned-gt)**2))

SEEDS_5 = [42, 123, 777, 2024, 31415]
SEEDS_10 = SEEDS_5 + [1337, 9999, 54321, 11111, 88888]
SEEDS_20 = SEEDS_10 + [12345, 67890, 24680, 13579, 99999, 7777, 3141, 2718, 1618, 55555]

# ============================================================
#  LOSO EVALUATION (the honest objective)
# ============================================================
def loso_rmse(xgb_params, X, y, seasons, ridge_alpha=5.0, ridge_w=0.3,
              seeds=SEEDS_5, rc_params=None, power=1.1):
    """
    Leave-one-season-out RMSE.
    This is our HONEST generalization metric.
    """
    all_errors = []
    for hold in sorted(set(seasons)):
        tr = seasons != hold; te = seasons == hold
        Xtr = X[tr]; ytr = y[tr]; Xte = X[te]; yte = y[te]
        ste = seasons[te]

        # XGB ensemble
        xgb_tr = []; xgb_te = []
        for seed in seeds:
            m = xgb.XGBRegressor(**xgb_params, random_state=seed, verbosity=0)
            m.fit(Xtr, ytr)
            xgb_tr.append(m.predict(Xtr)); xgb_te.append(m.predict(Xte))
        avg_tr = np.mean(xgb_tr, axis=0); avg_te = np.mean(xgb_te, axis=0)

        # Ridge
        s = StandardScaler()
        Xtr_s = s.fit_transform(Xtr); Xte_s = s.transform(Xte)
        rm = Ridge(alpha=ridge_alpha)
        rm.fit(Xtr_s, ytr)
        r_tr = rm.predict(Xtr_s); r_te = rm.predict(Xte_s)
        bl_tr = (1-ridge_w)*avg_tr + ridge_w*r_tr
        bl_te = (1-ridge_w)*avg_te + ridge_w*r_te

        # RC
        if rc_params is not None:
            res = ytr - bl_tr
            Xa_tr = np.column_stack([Xtr, bl_tr])
            Xa_te = np.column_stack([Xte, bl_te])
            rc = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rc.fit(Xa_tr, res)
            bl_te = bl_te + rc.predict(Xa_te)

        # Hungarian
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(bl_te, ste, avail, power)
        errors = (assigned - yte.astype(int)) ** 2
        all_errors.extend(errors)

    return np.sqrt(np.mean(all_errors))


def test_score(xgb_params, ridge_alpha=5.0, ridge_w=0.3, seeds=SEEDS_5,
               rc_params=None, power=1.1):
    """Test score (for reference only — NOT used for selection)."""
    xgb_tr = []; xgb_te = []
    for seed in seeds:
        m = xgb.XGBRegressor(**xgb_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_train)
        xgb_tr.append(m.predict(X_tr)); xgb_te.append(m.predict(X_te))
    avg_tr = np.mean(xgb_tr, axis=0); avg_te = np.mean(xgb_te, axis=0)

    rm = Ridge(alpha=ridge_alpha)
    rm.fit(X_tr_sc, y_train)
    r_tr = rm.predict(X_tr_sc); r_te = rm.predict(X_te_sc)
    bl_tr = (1-ridge_w)*avg_tr + ridge_w*r_tr
    bl_te = (1-ridge_w)*avg_te + ridge_w*r_te

    if rc_params is not None:
        res = y_train - bl_tr
        Xa_tr = np.column_stack([X_tr, bl_tr])
        Xa_te = np.column_stack([X_te, bl_te])
        rc = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
        rc.fit(Xa_tr, res)
        bl_te = bl_te + rc.predict(Xa_te)

    assigned = hungarian(bl_te, test_seasons, test_avail, power)
    return evaluate(assigned, test_gt)


# ============================================================
#  PHASES 1-3 RESULTS (from full search run)
# ============================================================
# Phase 1: depth=5, lambda=3.0, alpha=1.0, lr=0.05, mcw=3, ss=0.8, cs=0.8, n_est=700
# Phase 2: ridge_alpha=5, ridge_w=0.3
# Phase 3: No RC best for generalization (LOSO-RMSE=4.0889)

BEST_XGB = {
    'n_estimators': 700,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_lambda': 3.0,
    'reg_alpha': 1.0,
}
best_r_alpha = 5
best_rw = 0.3
BEST_RC = None  # No RC is best for generalization

print(f'  TUNED XGB PARAMS: {BEST_XGB}')
print(f'  Ridge: alpha={best_r_alpha}, weight={best_rw}')
print(f'  RC: None (dropped — hurts generalization)')
print(f'  Baseline LOSO-RMSE: 4.0889')


# ============================================================
#  PHASE 4: SEED COUNT + POWER
# ============================================================
print('\n' + '='*70)
print(' PHASE 4: SEED COUNT + HUNGARIAN POWER')
print('='*70)

results_4 = []
for n_seeds_label, seeds in [('5', SEEDS_5), ('10', SEEDS_10), ('20', SEEDS_20)]:
    for power in [1.0, 1.1, 1.5]:
        rmse = loso_rmse(BEST_XGB, X_tr, y_train, train_seasons,
                         ridge_alpha=best_r_alpha, ridge_w=best_rw,
                         seeds=seeds, rc_params=BEST_RC, power=power)
        results_4.append({'n_seeds': n_seeds_label, 'power': power,
                          'loso_rmse': rmse, 'seeds': seeds})

results_4.sort(key=lambda x: x['loso_rmse'])
print(f'  {"seeds":>5s} {"power":>5s} {"LOSO-RMSE":>10s}')
for r in results_4[:10]:
    print(f'  {r["n_seeds"]:>5s} {r["power"]:5.2f} {r["loso_rmse"]:10.4f}')

best_seeds = results_4[0]['seeds']
best_power = results_4[0]['power']
best_n_seeds_label = results_4[0]['n_seeds']
print(f'\n  Best: {best_n_seeds_label} seeds, power={best_power}')


# ============================================================
#  FINAL COMPARISON: TUNED vs ORIGINAL
# ============================================================
print('\n' + '='*70)
print(' FINAL COMPARISON')
print('='*70)

# Original mod params (from v33)
ORIG_XGB = {'n_estimators':500, 'max_depth':5, 'learning_rate':0.03,
             'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':3,
             'reg_lambda':2.0, 'reg_alpha':0.5}
ORIG_RC = {'n_estimators':150, 'max_depth':2, 'learning_rate':0.03,
            'subsample':0.8, 'colsample_bytree':0.7, 'min_child_weight':5,
            'reg_lambda':2.0}

# LOSO RMSE comparison
orig_loso = loso_rmse(ORIG_XGB, X_tr, y_train, train_seasons,
                      ridge_alpha=5, ridge_w=0.3, seeds=SEEDS_10,
                      rc_params=ORIG_RC, power=1.1)
tuned_loso = loso_rmse(BEST_XGB, X_tr, y_train, train_seasons,
                       ridge_alpha=best_r_alpha, ridge_w=best_rw,
                       seeds=best_seeds, rc_params=BEST_RC, power=best_power)

# Test scores (reference only)
orig_test = test_score(ORIG_XGB, ridge_alpha=5, ridge_w=0.3, seeds=SEEDS_10,
                       rc_params=ORIG_RC, power=1.1)
tuned_test = test_score(BEST_XGB, ridge_alpha=best_r_alpha, ridge_w=best_rw,
                        seeds=best_seeds, rc_params=BEST_RC, power=best_power)

# Also: tuned XGB + original RC (in case tuned RC is too aggressive)
hybrid_loso = loso_rmse(BEST_XGB, X_tr, y_train, train_seasons,
                        ridge_alpha=best_r_alpha, ridge_w=best_rw,
                        seeds=best_seeds, rc_params=ORIG_RC, power=best_power)
hybrid_test = test_score(BEST_XGB, ridge_alpha=best_r_alpha, ridge_w=best_rw,
                         seeds=best_seeds, rc_params=ORIG_RC, power=best_power)

# And tuned XGB with no RC
no_rc_loso = loso_rmse(BEST_XGB, X_tr, y_train, train_seasons,
                       ridge_alpha=best_r_alpha, ridge_w=best_rw,
                       seeds=best_seeds, rc_params=None, power=best_power)
no_rc_test = test_score(BEST_XGB, ridge_alpha=best_r_alpha, ridge_w=best_rw,
                        seeds=best_seeds, rc_params=None, power=best_power)

print(f'\n  {"Config":30s} {"LOSO-RMSE":>10s} {"Test":>8s} {"RMSE_test":>10s}')
print(f'  {"Original (mod10+r30+rc)":30s} {orig_loso:10.4f} {orig_test[0]:3d}/91  {np.sqrt(orig_test[1]/n_te):10.4f}')
print(f'  {"Tuned (full)":30s} {tuned_loso:10.4f} {tuned_test[0]:3d}/91  {np.sqrt(tuned_test[1]/n_te):10.4f}')
print(f'  {"Tuned XGB + orig RC":30s} {hybrid_loso:10.4f} {hybrid_test[0]:3d}/91  {np.sqrt(hybrid_test[1]/n_te):10.4f}')
print(f'  {"Tuned XGB, no RC":30s} {no_rc_loso:10.4f} {no_rc_test[0]:3d}/91  {np.sqrt(no_rc_test[1]/n_te):10.4f}')

# Pick the one with best LOSO-RMSE for submission
configs = [
    ('orig_mod10_rc', orig_loso, orig_test, ORIG_XGB, 5, 0.3, SEEDS_10, ORIG_RC, 1.1),
    ('tuned_full', tuned_loso, tuned_test, BEST_XGB, best_r_alpha, best_rw, best_seeds, BEST_RC, best_power),
    ('tuned_xgb_orig_rc', hybrid_loso, hybrid_test, BEST_XGB, best_r_alpha, best_rw, best_seeds, ORIG_RC, best_power),
    ('tuned_no_rc', no_rc_loso, no_rc_test, BEST_XGB, best_r_alpha, best_rw, best_seeds, None, best_power),
]
configs.sort(key=lambda x: x[1])  # sort by LOSO-RMSE

print(f'\n  RECOMMENDATION (by LOSO-RMSE):')
for i, (name, loso, test, _, _, _, _, _, _) in enumerate(configs):
    marker = ' ← BEST' if i == 0 else ''
    print(f'    {i+1}. {name:25s} LOSO={loso:.4f} Test={test[0]}/91{marker}')


# ============================================================
#  PRINT FINAL PARAMS
# ============================================================
print(f'\n{"="*70}')
print(f' FINAL TUNED PARAMETERS')
print(f'{"="*70}')
print(f'\n  XGB params: {BEST_XGB}')
print(f'  Ridge: alpha={best_r_alpha}, weight={best_rw}')
print(f'  RC: {BEST_RC}')
print(f'  Seeds: {best_n_seeds_label}, Power: {best_power}')
print(f'  LOSO-RMSE: {configs[0][1]:.4f}')


# ============================================================
#  SAVE SUBMISSIONS
# ============================================================
print(f'\n{"="*70}')
print(f' SAVING')
print(f'{"="*70}')

def save_sub(assigned, name, desc):
    sub = sub_df.copy()
    for i,ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub['RecordID']==rid
        if mask.any(): sub.loc[mask,'Overall Seed'] = int(assigned[i])
    path = os.path.join(DATA_DIR, name)
    sub.to_csv(path, index=False)
    ex, sse = evaluate(assigned, test_gt)
    print(f'  {name}: {ex}/91  RMSE={np.sqrt(sse/n_te):.3f}  [{desc}]')

for i, (name, loso, test_sc, xp, ra, rw, seeds, rcp, pw) in enumerate(configs):
    # Generate predictions
    xgb_tr=[]; xgb_te=[]
    for seed in seeds:
        m = xgb.XGBRegressor(**xp, random_state=seed, verbosity=0)
        m.fit(X_tr, y_train)
        xgb_tr.append(m.predict(X_tr)); xgb_te.append(m.predict(X_te))
    avg_tr=np.mean(xgb_tr,axis=0); avg_te=np.mean(xgb_te,axis=0)
    rm=Ridge(alpha=ra); rm.fit(X_tr_sc,y_train)
    r_tr=rm.predict(X_tr_sc); r_te=rm.predict(X_te_sc)
    bl_tr=(1-rw)*avg_tr+rw*r_tr; bl_te=(1-rw)*avg_te+rw*r_te
    if rcp is not None:
        res=y_train-bl_tr
        Xa_tr=np.column_stack([X_tr,bl_tr]); Xa_te=np.column_stack([X_te,bl_te])
        rc=xgb.XGBRegressor(**rcp, random_state=42, verbosity=0)
        rc.fit(Xa_tr, res)
        bl_te=bl_te+rc.predict(Xa_te)
    assigned=hungarian(bl_te, test_seasons, test_avail, pw)
    save_sub(assigned, f'submission_v39_{i+1}.csv',
             f'{name} (LOSO={loso:.3f})')

print(f'\nTotal time: {time.time()-t0:.0f}s')
print(f'\nDone! Best generalizable model selected by LOSO-RMSE, not GT-snooped test score.')

if IN_COLAB:
    for i in range(1, len(configs)+1):
        fp = os.path.join(DATA_DIR, f'submission_v39_{i}.csv')
        if os.path.exists(fp): files.download(fp)
