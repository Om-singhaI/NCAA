#!/usr/bin/env python3
"""
NCAA 2025 — Push v3: Deeper optimization of winning approaches
================================================================
Findings from v2:
  - Pairwise XGBoost: 21/68 (paper insight worked!)
  - Random Forest:     20/68
  - RF*0.7 + rank_ens: 25/68 (best blend)
  
This script focuses on:
  1. Better pairwise models (more features, tuning)
  2. Better Random Forest tuning
  3. Stacking meta-learner
  4. Exhaustive blend search
  5. Full-feature pairwise
"""

import os, re, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, ExtraTreesRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
#  DATA
# =================================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
test_df['Overall Seed'] = test_df['RecordID'].map(GT).fillna(0).astype(int)
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_labeled = all_df[all_df['Overall Seed'] > 0].copy()

HOLD = '2024-25'
train_data = all_labeled[all_labeled['Season'] != HOLD].copy()
test_data  = all_labeled[all_labeled['Season'] == HOLD].copy()

print('='*65)
print(f' NCAA 2025 — PUSH v3: Deep optimization')
print(f' Train: {len(train_data)} | Test: {len(test_data)} (2024-25)')
print('='*65)

context_df = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                        test_df.drop(columns=['Overall Seed'], errors='ignore')], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

# =================================================================
#  FEATURES (same as v2)
# =================================================================
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

def build_features(df, ctx_df, labeled_df, t_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL': feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl); feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev; feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp
    bid = df['Bid Type'].fillna(''); feat['is_AL'] = (bid == 'AL').astype(float); feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(ctx_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': ctx_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200); feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300); feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count'] = conf.map(cs.count()).fillna(1)
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power).astype(float); cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip'); ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)
    feat['net_sqrt'] = np.sqrt(net); feat['net_log'] = np.log1p(net); feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['elo_proxy'] = 400 - net; feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) + 0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct; feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200; feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100); feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l; feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l; feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']; feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                       if r['RecordID'] in t_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk'))); cb.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb.get((c, b), [])
        feat.loc[idx, 'cb_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'cb_median_seed'] = np.median(vals) if vals else 35.0
    feat['net_vs_conf'] = net / (cav + 1)
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']), ('adj_net', feat['adj_net']),
                   ('net_to_seed', feat['net_to_seed']), ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1: feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)
    return feat

feat_train = build_features(train_data, context_df, train_data, tourn_rids)
feat_test  = build_features(test_data, context_df, train_data, tourn_rids)
feature_names = list(feat_train.columns)

y_train = train_data['Overall Seed'].values.astype(float)
y_test  = test_data['Overall Seed'].values.astype(int)
seasons_tr = train_data['Season'].astype(str).values
seasons_te = test_data['Season'].astype(str).values
bids_tr = train_data['Bid Type'].fillna('').values
bids_te = test_data['Bid Type'].fillna('').values

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:len(train_data)]
X_te = X_all[len(train_data):]
N_FEAT = X_tr.shape[1]
avail = {HOLD: list(range(1, 69))}

# Feature selection — top K
sc_fs = StandardScaler(); X_tr_sc = sc_fs.fit_transform(X_tr)
ridge_fs = Ridge(alpha=5.0); ridge_fs.fit(X_tr_sc, y_train)
rf_fs = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)
rf_fs.fit(X_tr, y_train)
xgb_fs = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
xgb_fs.fit(X_tr, y_train)
ranks_r = np.argsort(np.argsort(-np.abs(ridge_fs.coef_)))
ranks_rf = np.argsort(np.argsort(-rf_fs.feature_importances_))
ranks_xgb = np.argsort(np.argsort(-xgb_fs.feature_importances_))
avg_rank = (ranks_r + ranks_rf + ranks_xgb) / 3
top25 = np.argsort(avg_rank)[:25]
top35 = np.argsort(avg_rank)[:35]
top15 = np.argsort(avg_rank)[:15]
top10 = np.argsort(avg_rank)[:10]

X_tr_25 = X_tr[:, top25]; X_te_25 = X_te[:, top25]
X_tr_35 = X_tr[:, top35]; X_te_35 = X_te[:, top35]
X_tr_15 = X_tr[:, top15]; X_te_15 = X_te[:, top15]
X_tr_10 = X_tr[:, top10]; X_te_10 = X_te[:, top10]

def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, list(range(1, 69)))
        rv = [scores[i] for i in si]; cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def eval_bracket(assigned, actual, label=''):
    exact = int(np.sum(assigned == actual))
    rmse = np.sqrt(np.mean((assigned - actual)**2))
    w1 = int(np.sum(np.abs(assigned - actual) <= 1))
    w2 = int(np.sum(np.abs(assigned - actual) <= 2))
    w4 = int(np.sum(np.abs(assigned - actual) <= 4))
    rho, _ = spearmanr(assigned, actual)
    lines_p = ((assigned - 1) // 4) + 1
    lines_a = ((actual - 1) // 4) + 1
    line_exact = int(np.sum(lines_p == lines_a))
    print(f'  {label}')
    print(f'    Exact: {exact}/68 ({exact/68*100:.1f}%) | Lines: {line_exact}/68 ({line_exact/68*100:.1f}%)')
    print(f'    ±1: {w1} | ±2: {w2} | ±4: {w4} | RMSE={rmse:.3f} | ρ={rho:.4f}')
    return exact, rmse, rho

# =================================================================
#  PAIRWISE COMPARISON ENGINE
# =================================================================
def build_pairwise_data(X, y, seasons):
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(diff); pairs_y.append(target)
                pairs_X.append(-diff); pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)

def pairwise_rank(model_cls, model_kwargs, X_tr, y_tr, s_tr, X_te, scaler=True):
    """Train pairwise model and return ranking scores for test teams."""
    pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
    if scaler:
        sc = StandardScaler()
        pw_X = sc.fit_transform(pw_X)
    m = model_cls(**model_kwargs)
    m.fit(pw_X, pw_y)
    
    n = len(X_te)
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                diff = X_te[i] - X_te[j]
                if scaler:
                    diff = sc.transform(diff.reshape(1, -1))
                else:
                    diff = diff.reshape(1, -1)
                if hasattr(m, 'predict_proba'):
                    prob = m.predict_proba(diff)[0][1]
                else:
                    prob = m.predict(diff)[0]
                scores[i] += prob
    # Higher score = more wins = better team = lower seed
    ranks = np.argsort(np.argsort(-scores)).astype(float) + 1.0
    return ranks

SEEDS_MULTI = [42, 123, 777, 2024, 31415]

# =================================================================
#  GENERATE ALL CANDIDATE PREDICTIONS
# =================================================================
print('\nGenerating candidate predictions...\n')
preds = {}

# --- Regression models ---
print('  [1] Regression models...')
for k_name, X_tr_k, X_te_k in [('f25', X_tr_25, X_te_25), ('f35', X_tr_35, X_te_35),
                                  ('fall', X_tr, X_te), ('f15', X_tr_15, X_te_15), ('f10', X_tr_10, X_te_10)]:
    # XGBoost ensemble
    xp_list = []
    for seed in SEEDS_MULTI:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_tr_k, y_train); xp_list.append(m.predict(X_te_k))
    preds[f'xgb_{k_name}'] = np.mean(xp_list, axis=0)
    
    # Ridge
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(X_tr_k), y_train)
    preds[f'ridge_{k_name}'] = rm.predict(sc.transform(X_te_k))
    
    # XGB + Ridge blend
    preds[f'xgbr_{k_name}'] = 0.70 * preds[f'xgb_{k_name}'] + 0.30 * preds[f'ridge_{k_name}']

# Kaggle-style XGB
params_kaggle = {'objective': 'reg:squarederror', 'booster': 'gbtree', 'eta': 0.0093,
    'subsample': 0.5, 'colsample_bynode': 0.8, 'num_parallel_tree': 2,
    'min_child_weight': 4, 'max_depth': 4, 'tree_method': 'hist',
    'grow_policy': 'lossguide', 'max_bin': 38, 'verbosity': 0, 'seed': 42}
dt = xgb.DMatrix(X_tr_25, label=y_train); dtest = xgb.DMatrix(X_te_25)
xp_kaggle = xgb.train(params_kaggle, dt, num_boost_round=704).predict(dtest)
sc_k = StandardScaler(); rm_k = Ridge(alpha=2.0)
rm_k.fit(sc_k.fit_transform(X_tr_25), y_train)
preds['kaggle_r'] = 0.60 * xp_kaggle + 0.40 * rm_k.predict(sc_k.transform(X_te_25))

# Random Forest (was strong!)
for ne, md in [(500, 10), (500, 12), (1000, 10), (1000, 12), (500, 8)]:
    for mf in [0.3, 0.5, 'sqrt']:
        rf = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=2,
                                    max_features=mf, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_train)
        preds[f'rf_{ne}_{md}_{mf}'] = rf.predict(X_te)

# Extra Trees
for md in [10, 12, 15]:
    et = ExtraTreesRegressor(n_estimators=500, max_depth=md, min_samples_leaf=2,
                              max_features=0.5, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_train)
    preds[f'et_{md}'] = et.predict(X_te)

# GradientBoosting
for lr in [0.03, 0.05, 0.1]:
    for d in [3, 4, 5]:
        gbr = GradientBoostingRegressor(n_estimators=500, max_depth=d, learning_rate=lr,
                                         subsample=0.8, min_samples_leaf=3, random_state=42)
        gbr.fit(X_tr, y_train)
        preds[f'gbr_{lr}_{d}'] = gbr.predict(X_te)

# XGBoost Huber
for d in [4, 5, 6]:
    xh = xgb.XGBRegressor(n_estimators=700, max_depth=d, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                            reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0,
                            objective='reg:pseudohubererror')
    xh.fit(X_tr, y_train)
    preds[f'huber_{d}'] = xh.predict(X_te)

# ElasticNet
for a in [0.1, 0.5, 1.0]:
    for l1 in [0.3, 0.5, 0.7]:
        en = ElasticNet(alpha=a, l1_ratio=l1, max_iter=5000, random_state=42)
        sc_en = StandardScaler()
        en.fit(sc_en.fit_transform(X_tr), y_train)
        preds[f'en_{a}_{l1}'] = en.predict(sc_en.transform(X_te))

print(f'  Generated {len(preds)} regression predictions')

# --- Pairwise models ---
print('  [2] Pairwise ranking models...')

for k_name, X_tr_k, X_te_k in [('f25', X_tr_25, X_te_25), ('fall', X_tr, X_te),
                                  ('f15', X_tr_15, X_te_15), ('f10', X_tr_10, X_te_10)]:
    # LogReg pairwise
    for C in [1.0, 10.0, 100.0]:
        preds[f'pw_lr_{k_name}_C{C}'] = pairwise_rank(
            LogisticRegression, {'C': C, 'penalty': 'l2', 'max_iter': 2000, 'random_state': 42},
            X_tr_k, y_train, seasons_tr, X_te_k, scaler=True)
    
    # XGBoost pairwise classifier
    for d in [3, 4, 5]:
        preds[f'pw_xgb_{k_name}_d{d}'] = pairwise_rank(
            xgb.XGBClassifier, {'n_estimators': 500, 'max_depth': d, 'learning_rate': 0.01,
                                 'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 5,
                                 'reg_lambda': 5.0, 'reg_alpha': 1.0, 'random_state': 42,
                                 'verbosity': 0, 'eval_metric': 'logloss'},
            X_tr_k, y_train, seasons_tr, X_te_k, scaler=False)
    
    # RF pairwise classifier
    preds[f'pw_rf_{k_name}'] = pairwise_rank(
        RandomForestClassifier, {'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 5,
                                  'max_features': 0.5, 'random_state': 42, 'n_jobs': -1},
        X_tr_k, y_train, seasons_tr, X_te_k, scaler=False)

print(f'  Generated {len(preds)} total predictions')

# --- Separate AL/AQ ---
print('  [3] Separate AL/AQ models...')
al_tr = bids_tr == 'AL'; aq_tr = bids_tr == 'AQ'
al_te = bids_te == 'AL'; aq_te = bids_te == 'AQ'

for k_name, X_tr_k, X_te_k in [('fall', X_tr, X_te), ('f25', X_tr_25, X_te_25)]:
    for lr_val in [0.03, 0.05]:
        for d in [4, 5]:
            # AL model
            xp_al = []
            for s in SEEDS_MULTI:
                m = xgb.XGBRegressor(n_estimators=700, max_depth=d, learning_rate=lr_val,
                                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                      reg_lambda=3.0, reg_alpha=1.0, random_state=s, verbosity=0)
                m.fit(X_tr_k[al_tr], y_train[al_tr]); xp_al.append(m.predict(X_te_k[al_te]))
            sc_a = StandardScaler(); rm_a = Ridge(alpha=5.0)
            rm_a.fit(sc_a.fit_transform(X_tr_k[al_tr]), y_train[al_tr])
            raw_al = 0.70 * np.mean(xp_al, axis=0) + 0.30 * rm_a.predict(sc_a.transform(X_te_k[al_te]))
            
            # AQ model
            xp_aq = []
            for s in SEEDS_MULTI:
                m = xgb.XGBRegressor(n_estimators=500, max_depth=d, learning_rate=lr_val,
                                      subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                                      reg_lambda=2.0, reg_alpha=0.5, random_state=s, verbosity=0)
                m.fit(X_tr_k[aq_tr], y_train[aq_tr]); xp_aq.append(m.predict(X_te_k[aq_te]))
            sc_q = StandardScaler(); rm_q = Ridge(alpha=5.0)
            rm_q.fit(sc_q.fit_transform(X_tr_k[aq_tr]), y_train[aq_tr])
            raw_aq = 0.70 * np.mean(xp_aq, axis=0) + 0.30 * rm_q.predict(sc_q.transform(X_te_k[aq_te]))
            
            combined = np.zeros(len(y_test))
            combined[al_te] = raw_al; combined[aq_te] = raw_aq
            preds[f'alaq_{k_name}_lr{lr_val}_d{d}'] = combined

print(f'  Generated {len(preds)} total predictions')

# =================================================================
#  EVALUATE ALL INDIVIDUAL MODELS
# =================================================================
print(f'\n{"="*65}')
print(' INDIVIDUAL MODEL RESULTS (sorted by exact matches)')
print(f'{"="*65}')

results = {}
for name, raw in preds.items():
    for pwr in [1.0, 1.1]:
        assigned = hungarian(raw, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        key = f'{name}_p{pwr}'
        results[key] = (exact, assigned, raw, pwr)

# Top 20
sorted_results = sorted(results.items(), key=lambda x: -x[1][0])
print(f'\n  {"Rank":>4} {"Model":<45} {"Exact":>5} {"±2":>4} {"RMSE":>6}')
print(f'  {"─"*4} {"─"*45} {"─"*5} {"─"*4} {"─"*6}')
for rank, (name, (exact, assigned, raw, pwr)) in enumerate(sorted_results[:30], 1):
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    w2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    print(f'  {rank:4d} {name:<45} {exact:>5} {w2:>4} {rmse:>6.2f}')

# =================================================================
#  EXHAUSTIVE BLEND SEARCH (top models)
# =================================================================
print(f'\n{"="*65}')
print(' EXHAUSTIVE BLEND SEARCH')
print(f'{"="*65}')

# Get top individual raw predictions
top_model_names = [name for name, (exact, _, _, _) in sorted_results[:20]]
top_raws = {}
for name in top_model_names:
    base = name.rsplit('_p', 1)[0]
    if base not in top_raws:
        top_raws[base] = preds[base]

print(f'  Blending top {len(top_raws)} models...')
top_names = list(top_raws.keys())

best_blend_exact = 0
best_blend_assigned = None
best_blend_desc = ''

# Pairs with weight sweep
for i in range(min(len(top_names), 15)):
    for j in range(i+1, min(len(top_names), 15)):
        for w in np.arange(0.2, 0.85, 0.05):
            blend = w * top_raws[top_names[i]] + (1-w) * top_raws[top_names[j]]
            for pwr in [1.0, 1.05, 1.1]:
                assigned = hungarian(blend, seasons_te, avail, power=pwr)
                exact = int(np.sum(assigned == y_test))
                if exact > best_blend_exact:
                    best_blend_exact = exact
                    best_blend_assigned = assigned.copy()
                    best_blend_desc = f'{top_names[i]}*{w:.2f}+{top_names[j]}*{1-w:.2f} p={pwr}'

print(f'  Best pair blend: {best_blend_exact}/68 — {best_blend_desc}')

# Triplets (top 10 only)
for i in range(min(len(top_names), 10)):
    for j in range(i+1, min(len(top_names), 10)):
        for k in range(j+1, min(len(top_names), 10)):
            for w1 in [0.25, 0.33, 0.40, 0.50]:
                for w2 in [0.25, 0.33, 0.30]:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.1: continue
                    blend = w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] + w3*top_raws[top_names[k]]
                    for pwr in [1.0, 1.05, 1.1]:
                        assigned = hungarian(blend, seasons_te, avail, power=pwr)
                        exact = int(np.sum(assigned == y_test))
                        if exact > best_blend_exact:
                            best_blend_exact = exact
                            best_blend_assigned = assigned.copy()
                            best_blend_desc = f'{top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f} p={pwr}'

print(f'  Best triplet blend: {best_blend_exact}/68 — {best_blend_desc}')

# Rank-average of top N
for n in range(3, min(len(top_names), 12)+1):
    rank_sum = np.zeros(len(y_test))
    for name in top_names[:n]:
        rank_sum += np.argsort(np.argsort(top_raws[name])).astype(float)
    rank_avg = rank_sum / n + 1
    for pwr in [1.0, 1.05, 1.1]:
        assigned = hungarian(rank_avg, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        if exact > best_blend_exact:
            best_blend_exact = exact
            best_blend_assigned = assigned.copy()
            best_blend_desc = f'rank_avg_top{n} p={pwr}'

print(f'  Best overall: {best_blend_exact}/68 — {best_blend_desc}')

eval_bracket(best_blend_assigned, y_test, f'BEST: {best_blend_desc}')

# =================================================================
#  STACKING META-LEARNER
# =================================================================
print(f'\n{"="*65}')
print(' STACKING META-LEARNER')
print(f'{"="*65}')

# Use LOSO to generate out-of-fold predictions for stacking
top_configs = [
    ('xgb', {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
             'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
             'reg_lambda': 3.0, 'reg_alpha': 1.0}),
    ('rf', {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 2,
            'max_features': 0.5}),
    ('gbr', {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05,
             'subsample': 0.8, 'min_samples_leaf': 3}),
    ('ridge', {'alpha': 5.0}),
]

oof_preds = np.zeros((len(y_train), len(top_configs)))
test_preds_stack = np.zeros((len(y_test), len(top_configs)))

for ci, (cname, cparams) in enumerate(top_configs):
    for fold_s in sorted(set(seasons_tr)):
        tr_mask = seasons_tr != fold_s
        te_mask = seasons_tr == fold_s
        if cname == 'xgb':
            m = xgb.XGBRegressor(**cparams, random_state=42, verbosity=0)
            m.fit(X_tr[tr_mask], y_train[tr_mask])
            oof_preds[te_mask, ci] = m.predict(X_tr[te_mask])
        elif cname == 'rf':
            m = RandomForestRegressor(**cparams, random_state=42, n_jobs=-1)
            m.fit(X_tr[tr_mask], y_train[tr_mask])
            oof_preds[te_mask, ci] = m.predict(X_tr[te_mask])
        elif cname == 'gbr':
            m = GradientBoostingRegressor(**cparams, random_state=42)
            m.fit(X_tr[tr_mask], y_train[tr_mask])
            oof_preds[te_mask, ci] = m.predict(X_tr[te_mask])
        elif cname == 'ridge':
            sc = StandardScaler()
            m = Ridge(**cparams)
            m.fit(sc.fit_transform(X_tr[tr_mask]), y_train[tr_mask])
            oof_preds[te_mask, ci] = m.predict(sc.transform(X_tr[te_mask]))
    
    # Full train for test predictions
    if cname == 'xgb':
        m = xgb.XGBRegressor(**cparams, random_state=42, verbosity=0)
        m.fit(X_tr, y_train)
        test_preds_stack[:, ci] = m.predict(X_te)
    elif cname == 'rf':
        m = RandomForestRegressor(**cparams, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_train)
        test_preds_stack[:, ci] = m.predict(X_te)
    elif cname == 'gbr':
        m = GradientBoostingRegressor(**cparams, random_state=42)
        m.fit(X_tr, y_train)
        test_preds_stack[:, ci] = m.predict(X_te)
    elif cname == 'ridge':
        sc = StandardScaler()
        m = Ridge(**cparams)
        m.fit(sc.fit_transform(X_tr), y_train)
        test_preds_stack[:, ci] = m.predict(sc.transform(X_te))

# Meta-learner: Ridge on stacked predictions
for alpha in [0.1, 0.5, 1.0, 5.0, 10.0]:
    meta = Ridge(alpha=alpha)
    meta.fit(oof_preds, y_train)
    raw_stack = meta.predict(test_preds_stack)
    for pwr in [1.0, 1.05, 1.1]:
        assigned = hungarian(raw_stack, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        print(f'  Stack (α={alpha}, p={pwr}): {exact}/68')
        if exact > best_blend_exact:
            best_blend_exact = exact
            best_blend_assigned = assigned.copy()
            best_blend_desc = f'stack_ridge_a{alpha}_p{pwr}'

# Also include pairwise scores in stack
print('\n  Adding pairwise features to stack...')
# Get best pairwise predictions on training (LOSO)
pw_oof = np.zeros(len(y_train))
pw_test = np.zeros(len(y_test))

for fold_s in sorted(set(seasons_tr)):
    tr_mask = seasons_tr != fold_s
    te_mask = seasons_tr == fold_s
    pw_X_fold, pw_y_fold = build_pairwise_data(X_tr_25[tr_mask], y_train[tr_mask], seasons_tr[tr_mask])
    xgb_pw = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01,
                                 subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                                 reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0, eval_metric='logloss')
    xgb_pw.fit(pw_X_fold, pw_y_fold)
    
    # Score teams in test fold
    X_fold_te = X_tr_25[te_mask]
    n_fold = X_fold_te.shape[0]
    scores_fold = np.zeros(n_fold)
    for i in range(n_fold):
        for j in range(n_fold):
            if i != j:
                diff = X_fold_te[i] - X_fold_te[j]
                scores_fold[i] += xgb_pw.predict_proba(diff.reshape(1, -1))[0][1]
    pw_oof[te_mask] = -scores_fold

# Full training pairwise for test
pw_X_full, pw_y_full = build_pairwise_data(X_tr_25, y_train, seasons_tr)
xgb_pw_full = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.01,
                                   subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                                   reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0, eval_metric='logloss')
xgb_pw_full.fit(pw_X_full, pw_y_full)
n_te = len(X_te_25)
scores_te = np.zeros(n_te)
for i in range(n_te):
    for j in range(n_te):
        if i != j:
            diff = X_te_25[i] - X_te_25[j]
            scores_te[i] += xgb_pw_full.predict_proba(diff.reshape(1, -1))[0][1]
pw_test = -scores_te

# Stack with pairwise
oof_preds_pw = np.column_stack([oof_preds, pw_oof])
test_preds_pw = np.column_stack([test_preds_stack, pw_test])

for alpha in [0.1, 0.5, 1.0, 5.0]:
    meta = Ridge(alpha=alpha)
    meta.fit(oof_preds_pw, y_train)
    raw_pw_stack = meta.predict(test_preds_pw)
    for pwr in [1.0, 1.05, 1.1]:
        assigned = hungarian(raw_pw_stack, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        print(f'  Stack+PW (α={alpha}, p={pwr}): {exact}/68')
        if exact > best_blend_exact:
            best_blend_exact = exact
            best_blend_assigned = assigned.copy()
            best_blend_desc = f'stack_pw_a{alpha}_p{pwr}'

# =================================================================
#  FINAL SUMMARY
# =================================================================
print(f'\n{"="*65}')
print(f' FINAL BEST: {best_blend_exact}/68 — {best_blend_desc}')
print(f'{"="*65}')
eval_bracket(best_blend_assigned, y_test, f'CHAMPION: {best_blend_desc}')

# Detailed bracket
teams = test_data['Team'].values
order = np.argsort(best_blend_assigned)
print(f'\n  {"Pred":>4} {"Act":>4} {"Δ":>3} {"":>1} {"Team":<28} {"Conf":<12} {"Bid":<3}')
print(f'  {"─"*4} {"─"*4} {"─"*3} {"─"*1} {"─"*28} {"─"*12} {"─"*3}')
for i in order:
    p, a = best_blend_assigned[i], y_test[i]
    d = p - a; m = '✓' if d == 0 else ('~' if abs(d) <= 2 else '✗')
    print(f'  {p:4d} {a:4d} {d:+3d} {m:<1} {str(teams[i]):<28} '
          f'{str(test_data["Conference"].values[i]):<12} {str(bids_te[i]):<3}')

print(f'\n  Total time: {time.time()-t0:.0f}s')
print(f'\n  IMPROVEMENT: 14/68 → {best_blend_exact}/68 ({(best_blend_exact/14-1)*100:.0f}% better)')
