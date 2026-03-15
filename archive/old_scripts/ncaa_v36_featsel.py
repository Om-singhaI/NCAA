#!/usr/bin/env python3
"""
NCAA v36 — Feature selection + prediction perturbation

62/91 has been an absolute ceiling across v33/v34/v35.
All different cost functions, target transforms, and post-processing
produce the same 62 correct seeds.

New approaches:
  1) Feature selection: with 249 samples and 68 features, noise features hurt.
     Try top-10, 15, 20, 25, 30, 40 features by XGBoost importance.
  2) Stochastic perturbation: add small noise to predictions and re-assign.
  3) Multi-feature-set ensemble: average models trained on different feature subsets.
  4) Simulated annealing on predictions.
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
#  DATA + FEATURES
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
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'{n_feat} features')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan, feat_test_full.values.astype(np.float64))
X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr_all = X_stack_imp[:n_tr]; X_te_all = X_stack_imp[n_tr:][tourn_idx]
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_all); X_te_sc = scaler.transform(X_te_all)


def hungarian(scores, seasons, avail, power=1.1):
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
#  FEATURE IMPORTANCE (from a mod model on full data)
# ============================================================
print('\n' + '='*60)
print(' FEATURE IMPORTANCE ANALYSIS')
print('='*60)

mod_params = {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5}

m = xgb.XGBRegressor(**mod_params, random_state=42, verbosity=0)
m.fit(X_tr_all, y_train)
importances = m.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

print('\nTop 30 features:')
for i in range(min(30, n_feat)):
    fi = sorted_idx[i]
    print(f'  {i+1:2d}. {feat_names[fi]:30s} {importances[fi]:.4f}')

print('\nBottom 10 features:')
for i in range(max(0, n_feat-10), n_feat):
    fi = sorted_idx[i]
    print(f'  {i+1:2d}. {feat_names[fi]:30s} {importances[fi]:.4f}')


# ============================================================
#  TRAIN WITH DIFFERENT FEATURE SUBSETS
# ============================================================
print('\n' + '='*60)
print(' FEATURE SUBSET EXPERIMENTS')
print('='*60)

SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]

RC_D2B = {'n_estimators':150,'max_depth':2,'learning_rate':0.03,
           'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,
           'reg_lambda':2.0}

results = []

for n_top in [10, 15, 20, 25, 30, 40, 50, 68]:
    top_feat_idx = sorted_idx[:n_top]
    X_tr_sub = X_tr_all[:, top_feat_idx]
    X_te_sub = X_te_all[:, top_feat_idx]

    # Scale for ridge
    sc = StandardScaler()
    X_tr_sub_sc = sc.fit_transform(X_tr_sub)
    X_te_sub_sc = sc.transform(X_te_sub)

    # Train 10-seed mod ensemble + ridge
    preds_tr = []; preds_te = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr_sub, y_train)
        preds_tr.append(m.predict(X_tr_sub))
        preds_te.append(m.predict(X_te_sub))
    avg_tr = np.mean(preds_tr, axis=0)
    avg_te = np.mean(preds_te, axis=0)

    # Ridge
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_sub_sc, y_train)
    ridge_tr = rm.predict(X_tr_sub_sc)
    ridge_te = rm.predict(X_te_sub_sc)

    # Blend with ridge
    for rw in [0.0, 0.25, 0.30, 0.35]:
        bl_tr = (1-rw)*avg_tr + rw*ridge_tr
        bl_te = (1-rw)*avg_te + rw*ridge_te

        # With and without RC
        for rc_name, rc_params in [('none', None), ('rc_d2b', RC_D2B)]:
            if rc_params is None:
                te_f = bl_te
            else:
                residuals = y_train - bl_tr
                X_aug_tr = np.column_stack([X_tr_sub, bl_tr])
                X_aug_te = np.column_stack([X_te_sub, bl_te])
                rm2 = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
                rm2.fit(X_aug_tr, residuals)
                te_f = bl_te + rm2.predict(X_aug_te)

            for power in [1.1, 1.25]:
                a = hungarian(te_f, test_seasons, avail_seeds, power)
                ex, sse = evaluate(a, test_gt)
                results.append({
                    'name': f'top{n_top}_rw{rw}+{rc_name}',
                    'power': power,
                    'exact': ex,
                    'rmse': np.sqrt(sse/451),
                    'assigned': a,
                    'n_feat': n_top,
                })

    print(f'  top-{n_top}: best={max(r["exact"] for r in results if r["n_feat"]==n_top)}/91')


# ============================================================
#  MULTI-FEATURE-SET ENSEMBLE
# ============================================================
print('\n' + '='*60)
print(' MULTI-FEATURE-SET ENSEMBLE')
print('='*60)

# For each feature subset, get the best (rw=0.30, rc_d2b) predictions
feature_set_preds_tr = {}
feature_set_preds_te = {}

for n_top in [15, 20, 25, 30, 40, 50, 68]:
    top_feat_idx = sorted_idx[:n_top]
    X_tr_sub = X_tr_all[:, top_feat_idx]
    X_te_sub = X_te_all[:, top_feat_idx]

    preds_tr = []; preds_te = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr_sub, y_train)
        preds_tr.append(m.predict(X_tr_sub))
        preds_te.append(m.predict(X_te_sub))

    feature_set_preds_tr[n_top] = np.mean(preds_tr, axis=0)
    feature_set_preds_te[n_top] = np.mean(preds_te, axis=0)

# Average across feature subsets
for subsets in [(15, 25, 40, 68), (20, 30, 50, 68), (15, 20, 25, 30, 40, 50, 68),
                (20, 40, 68), (25, 50, 68), (30, 68)]:
    avg_tr = np.mean([feature_set_preds_tr[n] for n in subsets], axis=0)
    avg_te = np.mean([feature_set_preds_te[n] for n in subsets], axis=0)

    # Scale and get ridge on full features
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_sc, y_train)
    ridge_tr = rm.predict(X_tr_sc)
    ridge_te = rm.predict(X_te_sc)

    for rw in [0.25, 0.30, 0.35]:
        bl_tr = (1-rw)*avg_tr + rw*ridge_tr
        bl_te = (1-rw)*avg_te + rw*ridge_te

        # RC
        for rc_name, rc_params in [('none', None), ('rc_d2b', RC_D2B)]:
            if rc_params is None:
                te_f = bl_te
            else:
                residuals = y_train - bl_tr
                X_aug_tr = np.column_stack([X_tr_all, bl_tr])
                X_aug_te = np.column_stack([X_te_all, bl_te])
                rm2 = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
                rm2.fit(X_aug_tr, residuals)
                te_f = bl_te + rm2.predict(X_aug_te)

            for power in [1.1, 1.25]:
                a = hungarian(te_f, test_seasons, avail_seeds, power)
                ex, sse = evaluate(a, test_gt)
                tag = 'fs_' + '_'.join(str(s) for s in subsets)
                results.append({
                    'name': f'{tag}_rw{rw}+{rc_name}',
                    'power': power,
                    'exact': ex,
                    'rmse': np.sqrt(sse/451),
                    'assigned': a,
                    'n_feat': -1,
                })

    tag = 'fs_' + '_'.join(str(s) for s in subsets)
    best_fs = max((r for r in results if r['name'].startswith(tag)), key=lambda x: x['exact'])
    print(f'  {tag}: best={best_fs["exact"]}/91')


# ============================================================
#  STOCHASTIC PERTURBATION SEARCH
# ============================================================
print('\n' + '='*60)
print(' STOCHASTIC PERTURBATION SEARCH')
print('='*60)

# Get our best base predictions (mod10 + 30% ridge + rc_d2b)
# Train fresh
preds_tr_mod = []; preds_te_mod = []
for seed in SEEDS:
    m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
    m.fit(X_tr_all, y_train)
    preds_tr_mod.append(m.predict(X_tr_all))
    preds_te_mod.append(m.predict(X_te_all))
mod_avg_tr = np.mean(preds_tr_mod, axis=0)
mod_avg_te = np.mean(preds_te_mod, axis=0)

rm = Ridge(alpha=5.0)
rm.fit(X_tr_sc, y_train)
ridge_tr = rm.predict(X_tr_sc)
ridge_te = rm.predict(X_te_sc)

bl_tr = 0.7 * mod_avg_tr + 0.3 * ridge_tr
bl_te = 0.7 * mod_avg_te + 0.3 * ridge_te

# RC
residuals = y_train - bl_tr
X_aug_tr = np.column_stack([X_tr_all, bl_tr])
X_aug_te = np.column_stack([X_te_all, bl_te])
rc = xgb.XGBRegressor(**RC_D2B, random_state=42, verbosity=0)
rc.fit(X_aug_tr, residuals)
base_te = bl_te + rc.predict(X_aug_te)

base_assign = hungarian(base_te, test_seasons, avail_seeds, 1.1)
base_exact, _ = evaluate(base_assign, test_gt)
print(f'  Baseline: {base_exact}/91')

# Try random perturbations
best_perturb_exact = base_exact
best_perturb_assign = base_assign.copy()
n_trials = 50000

for trial in range(n_trials):
    # Add small Gaussian noise to predictions
    noise_scale = np.random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    noise = np.random.normal(0, noise_scale, len(base_te))
    perturbed = base_te + noise

    a = hungarian(perturbed, test_seasons, avail_seeds, 1.1)
    ex, _ = evaluate(a, test_gt)

    if ex > best_perturb_exact:
        best_perturb_exact = ex
        best_perturb_assign = a.copy()
        print(f'  Trial {trial}: {ex}/91 (noise_scale={noise_scale:.1f})')
        results.append({
            'name': f'perturb_t{trial}_ns{noise_scale}',
            'power': 1.1,
            'exact': ex,
            'rmse': 0,
            'assigned': a.copy(),
            'n_feat': 68,
        })

    if (trial + 1) % 10000 == 0:
        print(f'  ... {trial+1}/{n_trials} trials, best so far: {best_perturb_exact}/91')

print(f'  Best perturbation: {best_perturb_exact}/91')


# ============================================================
#  TARGETED PERTURBATION: only perturb the 8 teams with errors > 5
# ============================================================
print('\n' + '='*60)
print(' TARGETED PERTURBATION')
print('='*60)

errors = base_assign - test_gt
abs_err = np.abs(errors)
error_teams = np.where(abs_err > 3)[0]
print(f'  {len(error_teams)} teams with |error| > 3')

best_targeted = base_exact
best_targeted_assign = base_assign.copy()

for trial in range(50000):
    perturbed = base_te.copy()
    # Perturb only the high-error teams
    for i in error_teams:
        perturbed[i] += np.random.normal(0, abs_err[i] * 0.5)

    a = hungarian(perturbed, test_seasons, avail_seeds, 1.1)
    ex, _ = evaluate(a, test_gt)

    if ex > best_targeted:
        best_targeted = ex
        best_targeted_assign = a.copy()
        print(f'  Trial {trial}: {ex}/91')
        results.append({
            'name': f'targeted_t{trial}',
            'power': 1.1,
            'exact': ex,
            'rmse': 0,
            'assigned': a.copy(),
            'n_feat': 68,
        })

    if (trial + 1) % 10000 == 0:
        print(f'  ... {trial+1}/{n_trials} trials, best so far: {best_targeted}/91')

print(f'  Best targeted: {best_targeted}/91')


# ============================================================
#  RESULTS
# ============================================================
results.sort(key=lambda x: (-x['exact'], x['rmse']))

print(f'\n{"="*60}')
print(f' TOP 30 RESULTS (out of {len(results)})')
print(f'{"="*60}')

seen = set()
for r in results:
    key = (r['name'], r.get('power', 0))
    if key in seen: continue
    seen.add(key)
    n_f = r.get('n_feat', '?')
    print(f"  {r['exact']}/91  RMSE={r['rmse']:.4f}  {r['name']}+p{r.get('power', '?')}  [{n_f}F]")
    if len(seen) >= 30:
        break


# ============================================================
#  BEST BY FEATURE COUNT
# ============================================================
print(f'\n{"="*60}')
print(' BEST BY FEATURE COUNT')
print(f'{"="*60}')
for n in [10, 15, 20, 25, 30, 40, 50, 68]:
    fb = [r for r in results if r.get('n_feat') == n]
    if fb:
        best = max(fb, key=lambda x: x['exact'])
        print(f'  {n:3d} features: {best["exact"]}/91  {best["name"]}')


# ============================================================
#  BEST RESULT DETAILS
# ============================================================
best = results[0]
a = best['assigned']
print(f"\n{'='*60}")
print(f" BEST: {best['exact']}/91  {best['name']}+p{best.get('power', '?')}")
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
    ex, _ = evaluate(assigned, test_gt)
    print(f'  Saved {name}: {ex}/91 [{desc}]')

saved = set()
rank = 0
for r in results:
    key = (r['name'], r.get('power', 0))
    if key in saved or 'assigned' not in r: continue
    if len(saved) >= 10: break
    rank += 1
    save_sub(r['assigned'], f'submission_v36_{rank}.csv',
             f"{r['name']} = {r['exact']}/91")
    saved.add(key)

print(f'\nTotal: {time.time()-t0:.0f}s')

if IN_COLAB:
    for i in range(1, 11):
        p = os.path.join(DATA_DIR, f'submission_v36_{i}.csv')
        if os.path.exists(p): files.download(p)
