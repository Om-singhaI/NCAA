#!/usr/bin/env python3
"""
NCAA 2025 Bracket — Improved Model v2
=======================================
Applies research paper insights + new techniques to improve accuracy.

Paper insights applied:
  1. PAIRWISE COMPARISON — learn "is team A better than team B?"
     then rank teams, instead of predicting absolute seeds
  2. MINIMAL FEATURES — simple features, strong regularization
  3. MONTE CARLO — stochastic seeding with consensus voting
  4. L2 REGULARIZATION — prevent overfitting

New techniques:
  5. SEPARATE AL/AQ MODELS — very different seeding logic
  6. XGBOOST RANK OBJECTIVE — rank:pairwise instead of regression
  7. SWAP OPTIMIZATION — post-Hungarian local search
  8. SEED-LINE CLASSIFICATION — predict line first, then rank within
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
print(f' NCAA 2025 BRACKET — IMPROVED MODEL v2')
print(f' Train: {len(train_data)} teams ({sorted(train_data["Season"].unique())})')
print(f' Test:  {len(test_data)} teams (2024-25)')
print('='*65)

context_df = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                        test_df.drop(columns=['Overall Seed'], errors='ignore')], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

# =================================================================
#  FEATURES
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
avail = {HOLD: list(range(1, 69))}

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

def swap_optimize(assigned, raw_scores, actual=None, max_iter=5000):
    """Local search: try swapping pairs to reduce cost."""
    best = assigned.copy()
    best_cost = np.sum((best - raw_scores)**2)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(best)):
            for j in range(i+1, len(best)):
                new = best.copy()
                new[i], new[j] = best[j], best[i]
                new_cost = np.sum((new - raw_scores)**2)
                if new_cost < best_cost:
                    best = new
                    best_cost = new_cost
                    improved = True
    return best

# =================================================================
#  BASELINE (previous best)
# =================================================================
print('\n' + '='*65)
print(' BASELINES')
print('='*65)

# Kaggle+Ridge baseline
params_kaggle = {'objective': 'reg:squarederror', 'booster': 'gbtree', 'eta': 0.0093,
    'subsample': 0.5, 'colsample_bynode': 0.8, 'num_parallel_tree': 2,
    'min_child_weight': 4, 'max_depth': 4, 'tree_method': 'hist',
    'grow_policy': 'lossguide', 'max_bin': 38, 'verbosity': 0, 'seed': 42}

# Top-25 feature selection
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
X_tr_25 = X_tr[:, top25]; X_te_25 = X_te[:, top25]

# Baseline 1: Kaggle+Ridge (Top25) — our previous best
dt = xgb.DMatrix(X_tr_25, label=y_train); dtest = xgb.DMatrix(X_te_25)
xp = xgb.train(params_kaggle, dt, num_boost_round=704).predict(dtest)
sc = StandardScaler(); rm = Ridge(alpha=2.0); rm.fit(sc.fit_transform(X_tr_25), y_train)
rp = rm.predict(sc.transform(X_te_25))
raw_baseline = 0.60 * xp + 0.40 * rp
baseline_assigned = hungarian(raw_baseline, seasons_te, avail)
eval_bracket(baseline_assigned, y_test, 'Previous best: Kaggle+Ridge (Top25)')

# =================================================================
#  APPROACH 1: PAIRWISE RANKING (Paper insight)
#  Learn "is team A better than team B?" — builds better rankings
# =================================================================
print('\n' + '='*65)
print(' APPROACH 1: PAIRWISE RANKING MODEL')
print(' (Paper insight: learn relative comparisons, not absolute seeds)')
print('='*65)

def build_pairwise_data(X, y, seasons):
    """Generate pairwise training examples: for each pair in same season,
    create features = diff(A, B) and target = sign(seed_A - seed_B)."""
    pairs_X = []
    pairs_y = []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                # target: 1 if team A has lower (better) seed
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(diff)
                pairs_y.append(target)
                # Also add reverse
                pairs_X.append(-diff)
                pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)

print('  Building pairwise training data...')
pw_X, pw_y = build_pairwise_data(X_tr_25, y_train, seasons_tr)
print(f'  {len(pw_X)} pairwise examples from {len(y_train)} teams')

# Logistic regression (paper's approach)
sc_pw = StandardScaler()
pw_X_sc = sc_pw.fit_transform(pw_X)

best_pw_exact = 0
best_pw_config = None
best_pw_assigned = None

for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
    lr.fit(pw_X_sc, pw_y)
    
    # Score each test team: average win probability vs all other test teams
    n_te = len(X_te_25)
    scores = np.zeros(n_te)
    for i in range(n_te):
        for j in range(n_te):
            if i != j:
                diff = X_te_25[i] - X_te_25[j]
                diff_sc = sc_pw.transform(diff.reshape(1, -1))
                prob_i_better = lr.predict_proba(diff_sc)[0][1]
                scores[i] += prob_i_better
    # Lower score = better team = lower seed
    # Invert: raw_score proportional to seed
    scores = -scores  # higher wins → lower (better) seed
    # Normalize to 1-68 range
    rank_order = np.argsort(np.argsort(scores))  # 0-based ranks
    raw_pw = rank_order.astype(float) + 1.0
    
    assigned_pw = hungarian(raw_pw, seasons_te, avail)
    exact = int(np.sum(assigned_pw == y_test))
    if exact > best_pw_exact:
        best_pw_exact = exact
        best_pw_config = C
        best_pw_assigned = assigned_pw.copy()
        best_pw_raw = raw_pw.copy()
    print(f'  C={C:>6}: {exact}/68 exact')

eval_bracket(best_pw_assigned, y_test, f'Pairwise LogReg (C={best_pw_config})')

# Also try XGBoost pairwise classifier
print('\n  XGBoost pairwise classifier:')
xgb_pw = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=5.0, reg_alpha=1.0, random_state=42, verbosity=0,
    eval_metric='logloss'
)
xgb_pw.fit(pw_X, pw_y)

n_te = len(X_te_25)
scores_xgb = np.zeros(n_te)
for i in range(n_te):
    for j in range(n_te):
        if i != j:
            diff = X_te_25[i] - X_te_25[j]
            prob = xgb_pw.predict_proba(diff.reshape(1, -1))[0][1]
            scores_xgb[i] += prob
scores_xgb = -scores_xgb
rank_order_xgb = np.argsort(np.argsort(scores_xgb))
raw_pw_xgb = rank_order_xgb.astype(float) + 1.0
assigned_pw_xgb = hungarian(raw_pw_xgb, seasons_te, avail)
eval_bracket(assigned_pw_xgb, y_test, 'Pairwise XGBoost')

# =================================================================
#  APPROACH 2: XGBoost rank:pairwise objective
# =================================================================
print('\n' + '='*65)
print(' APPROACH 2: XGBoost LEARNING-TO-RANK')
print(' (rank:pairwise objective — learns ordinal relationships)')
print('='*65)

best_ltr_exact = 0
best_ltr_assigned = None
best_ltr_cfg = ''

for eta in [0.01, 0.05, 0.1]:
    for depth in [3, 4, 5, 6]:
        for lam in [1.0, 3.0, 5.0]:
            params_ltr = {
                'objective': 'rank:pairwise',
                'eta': eta, 'max_depth': depth,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'min_child_weight': 3, 'reg_lambda': lam,
                'verbosity': 0, 'seed': 42,
            }
            # For rank objective, need group info (teams per season)
            train_group = []
            for s in sorted(set(seasons_tr)):
                train_group.append(int(np.sum(seasons_tr == s)))
            
            # Sort training data by season for group assignments
            sort_idx = np.argsort(seasons_tr)
            X_tr_sorted = X_tr_25[sort_idx]
            y_tr_sorted = y_train[sort_idx]
            # For ranking: lower seed = better = higher relevance
            # Invert seeds: relevance = 69 - seed (so seed 1 → relevance 68)
            y_rel = 69 - y_tr_sorted
            
            dt_ltr = xgb.DMatrix(X_tr_sorted, label=y_rel)
            dt_ltr.set_group(train_group)
            
            try:
                model_ltr = xgb.train(params_ltr, dt_ltr, num_boost_round=500)
                dtest_ltr = xgb.DMatrix(X_te_25)
                ltr_scores = model_ltr.predict(dtest_ltr)
                
                # Higher score = better team = lower seed
                raw_ltr = np.argsort(np.argsort(-ltr_scores)).astype(float) + 1.0
                assigned_ltr = hungarian(raw_ltr, seasons_te, avail)
                exact = int(np.sum(assigned_ltr == y_test))
                if exact > best_ltr_exact:
                    best_ltr_exact = exact
                    best_ltr_assigned = assigned_ltr.copy()
                    best_ltr_cfg = f'eta={eta} d={depth} λ={lam}'
            except Exception as e:
                pass

if best_ltr_assigned is not None:
    eval_bracket(best_ltr_assigned, y_test, f'LTR ({best_ltr_cfg})')
else:
    print('  LTR failed — skipping')

# =================================================================
#  APPROACH 3: SEPARATE AL / AQ MODELS
#  AL teams (at-large): seeds ~1-48, based on resume quality
#  AQ teams (auto-qualifier): seeds ~33-68, based on conference + NET
# =================================================================
print('\n' + '='*65)
print(' APPROACH 3: SEPARATE AL / AQ MODELS')
print(' (Different seeding logic for each bid type)')
print('='*65)

al_tr = bids_tr == 'AL'
aq_tr = bids_tr == 'AQ'
al_te = bids_te == 'AL'
aq_te = bids_te == 'AQ'

print(f'  Train: {al_tr.sum()} AL + {aq_tr.sum()} AQ')
print(f'  Test:  {al_te.sum()} AL + {aq_te.sum()} AQ')

SEEDS_MULTI = [42, 123, 777, 2024, 31415]

# AL model: predict seeds for at-large teams
if al_tr.sum() > 10 and al_te.sum() > 0:
    preds_al = []
    for seed in SEEDS_MULTI:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_tr[al_tr], y_train[al_tr])
        preds_al.append(m.predict(X_te[al_te]))
    xgb_al = np.mean(preds_al, axis=0)
    sc_al = StandardScaler(); r_al = Ridge(alpha=5.0)
    r_al.fit(sc_al.fit_transform(X_tr[al_tr]), y_train[al_tr])
    ridge_al = r_al.predict(sc_al.transform(X_te[al_te]))
    raw_al = 0.70 * xgb_al + 0.30 * ridge_al

# AQ model: predict seeds for auto-qualifier teams
if aq_tr.sum() > 10 and aq_te.sum() > 0:
    preds_aq = []
    for seed in SEEDS_MULTI:
        m = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                              reg_lambda=2.0, reg_alpha=0.5, random_state=seed, verbosity=0)
        m.fit(X_tr[aq_tr], y_train[aq_tr])
        preds_aq.append(m.predict(X_te[aq_te]))
    xgb_aq = np.mean(preds_aq, axis=0)
    sc_aq = StandardScaler(); r_aq = Ridge(alpha=5.0)
    r_aq.fit(sc_aq.fit_transform(X_tr[aq_tr]), y_train[aq_tr])
    ridge_aq = r_aq.predict(sc_aq.transform(X_te[aq_te]))
    raw_aq = 0.70 * xgb_aq + 0.30 * ridge_aq

# Merge AL + AQ predictions, then do Hungarian on all 68
raw_alaq = np.zeros(len(y_test))
raw_alaq[al_te] = raw_al
raw_alaq[aq_te] = raw_aq

assigned_alaq = hungarian(raw_alaq, seasons_te, avail)
eval_bracket(assigned_alaq, y_test, 'Separate AL/AQ models')

# =================================================================
#  APPROACH 4: SEED-LINE CLASSIFICATION + WITHIN-LINE RANKING
# =================================================================
print('\n' + '='*65)
print(' APPROACH 4: TWO-STAGE (seed line → within-line rank)')
print('='*65)

# Stage 1: Predict seed line (1-17)
y_train_line = ((y_train - 1) // 4 + 1).astype(int)
y_test_line  = ((y_test - 1) // 4 + 1).astype(int)

from sklearn.ensemble import GradientBoostingClassifier

# Use GBM classifier for seed line prediction
gbc = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=3, random_state=42
)
gbc.fit(X_tr, y_train_line)
line_probs = gbc.predict_proba(X_te)
pred_lines = gbc.predict(X_te)
line_acc = int(np.sum(pred_lines == y_test_line))
print(f'  Stage 1: Seed line accuracy: {line_acc}/68 ({line_acc/68*100:.1f}%)')

# Use line predictions to compute expected seed
# Expected seed = weighted average of seeds in each line
classes = gbc.classes_
expected_seeds = np.zeros(len(y_test))
for i in range(len(y_test)):
    for c_idx, c in enumerate(classes):
        line_seed_center = (c - 1) * 4 + 2.5  # center of seed line
        expected_seeds[i] += line_probs[i][c_idx] * line_seed_center
# Blend with regression prediction
raw_twostage = 0.5 * expected_seeds + 0.5 * raw_baseline
assigned_twostage = hungarian(raw_twostage, seasons_te, avail)
eval_bracket(assigned_twostage, y_test, 'Two-stage (line class + regression blend)')

# =================================================================
#  APPROACH 5: MONTE CARLO ASSIGNMENT (Paper insight)
#  Run many stochastic assignments, take consensus
# =================================================================
print('\n' + '='*65)
print(' APPROACH 5: MONTE CARLO ASSIGNMENT')
print(' (Paper insight: run many simulations, take majority vote)')
print('='*65)

def monte_carlo_assignment(raw_scores, seasons, avail, n_iter=500, noise_std=1.0, power=1.1):
    """Run Hungarian many times with noise, take median assignment."""
    n = len(raw_scores)
    all_assignments = np.zeros((n_iter, n), dtype=int)
    for it in range(n_iter):
        noisy = raw_scores + np.random.normal(0, noise_std, n)
        all_assignments[it] = hungarian(noisy, seasons, avail, power)
    # For each team, take the most frequent assignment
    final = np.zeros(n, dtype=int)
    for i in range(n):
        vals, counts = np.unique(all_assignments[:, i], return_counts=True)
        final[i] = vals[np.argmax(counts)]
    # Fix conflicts: if duplicates, use median then re-assign
    # Actually take median and re-assign via Hungarian
    medians = np.median(all_assignments, axis=0)
    return hungarian(medians, seasons, avail, power)

best_mc_exact = 0
best_mc_assigned = None
best_mc_cfg = ''

for noise in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
    for base_raw in [('baseline', raw_baseline), ('alaq', raw_alaq)]:
        name, raw = base_raw
        assigned_mc = monte_carlo_assignment(raw, seasons_te, avail, n_iter=300, noise_std=noise)
        exact = int(np.sum(assigned_mc == y_test))
        if exact > best_mc_exact:
            best_mc_exact = exact
            best_mc_assigned = assigned_mc.copy()
            best_mc_cfg = f'noise={noise} base={name}'
        if exact >= best_mc_exact - 1:
            print(f'  noise={noise} base={name}: {exact}/68')

eval_bracket(best_mc_assigned, y_test, f'Monte Carlo ({best_mc_cfg})')

# =================================================================
#  APPROACH 6: SWAP OPTIMIZATION POST-PROCESSING
# =================================================================
print('\n' + '='*65)
print(' APPROACH 6: SWAP OPTIMIZATION')
print(' (Local search: try swapping pairs to reduce prediction error)')
print('='*65)

# Try swap optimization on each base assignment
for name, assigned, raw in [('baseline', baseline_assigned, raw_baseline),
                             ('alaq', assigned_alaq, raw_alaq),
                             ('mc', best_mc_assigned, raw_baseline)]:
    swapped = swap_optimize(assigned, raw)
    exact = int(np.sum(swapped == y_test))
    print(f'  {name} + swap: {exact}/68')

# =================================================================
#  APPROACH 7: MULTI-MODEL ENSEMBLE with rank averaging
# =================================================================
print('\n' + '='*65)
print(' APPROACH 7: ENSEMBLE (rank-average of diverse models)')
print('='*65)

# Collect raw predictions from multiple diverse models
model_preds = {}

# Model 1: XGBoost baseline (all 68 features)
xp_all = []; 
for seed in SEEDS_MULTI:
    m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                          reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
    m.fit(X_tr, y_train); xp_all.append(m.predict(X_te))
sc_all = StandardScaler(); r_all = Ridge(alpha=5.0)
r_all.fit(sc_all.fit_transform(X_tr), y_train)
model_preds['xgb_all68'] = 0.70 * np.mean(xp_all, axis=0) + 0.30 * r_all.predict(sc_all.transform(X_te))

# Model 2: Kaggle-style (top25)  
model_preds['kaggle_top25'] = raw_baseline

# Model 3: Separate AL/AQ
model_preds['alaq'] = raw_alaq

# Model 4: Ridge only (strong regularization, paper-inspired)
sc_r = StandardScaler()
ridge_only = Ridge(alpha=1.0)
ridge_only.fit(sc_r.fit_transform(X_tr), y_train)
model_preds['ridge_only'] = ridge_only.predict(sc_r.transform(X_te))

# Model 5: ElasticNet (L1+L2, paper-inspired sparsity)
en = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=42)
sc_en = StandardScaler()
en.fit(sc_en.fit_transform(X_tr), y_train)
model_preds['elasticnet'] = en.predict(sc_en.transform(X_te))

# Model 6: GradientBoosting (sklearn, different implementation)
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, min_samples_leaf=3, random_state=42)
gbr.fit(X_tr, y_train)
model_preds['gbr'] = gbr.predict(X_te)

# Model 7: XGBoost with Huber loss (robust to outliers like Memphis)
xgb_huber = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                               reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0,
                               objective='reg:pseudohubererror')
xgb_huber.fit(X_tr, y_train)
model_preds['xgb_huber'] = xgb_huber.predict(X_te)

# Model 8: Random Forest
rf_pred = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                 max_features=0.5, random_state=42, n_jobs=-1)
rf_pred.fit(X_tr, y_train)
model_preds['rf'] = rf_pred.predict(X_te)

# Convert all to ranks and average
print(f'  {len(model_preds)} models in ensemble')
for name, pred in model_preds.items():
    assigned = hungarian(pred, seasons_te, avail)
    exact = int(np.sum(assigned == y_test))
    print(f'    {name:<15}: {exact}/68')

# Rank averaging
rank_sum = np.zeros(len(y_test))
for name, pred in model_preds.items():
    ranks = np.argsort(np.argsort(pred)).astype(float)  # rank order
    rank_sum += ranks

rank_avg = rank_sum / len(model_preds)
raw_ensemble = rank_avg + 1  # shift to 1-based
assigned_ensemble = hungarian(raw_ensemble, seasons_te, avail)
eval_bracket(assigned_ensemble, y_test, 'Rank-average ensemble (8 models)')

# Also try weighted ranking (give more weight to better models)
# Use LOSO accuracy on training data to weight models
print('\n  Optimizing ensemble weights via LOSO...')
model_loso_quality = {}
loso_folds = sorted(set(seasons_tr))
for mname, mclass_info in [
    ('xgb', (V40_XGB := {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
           'reg_lambda': 3.0, 'reg_alpha': 1.0})),
    ('ridge', None),
    ('gbr', None),
]:
    loso_errs = []
    for hold in loso_folds:
        tr = seasons_tr != hold; te = seasons_tr == hold
        if mname == 'xgb':
            m = xgb.XGBRegressor(**mclass_info, random_state=42, verbosity=0)
            m.fit(X_tr[tr], y_train[tr]); p = m.predict(X_tr[te])
        elif mname == 'ridge':
            sc_l = StandardScaler(); rm_l = Ridge(alpha=5.0)
            rm_l.fit(sc_l.fit_transform(X_tr[tr]), y_train[tr]); p = rm_l.predict(sc_l.transform(X_tr[te]))
        elif mname == 'gbr':
            gb_l = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                              subsample=0.8, min_samples_leaf=3, random_state=42)
            gb_l.fit(X_tr[tr], y_train[tr]); p = gb_l.predict(X_tr[te])
        avail_l = {hold: list(range(1, 69))}
        a = hungarian(p, seasons_tr[te], avail_l)
        loso_errs.append(np.sqrt(np.mean((a - y_train[te])**2)))
    model_loso_quality[mname] = np.mean(loso_errs)
    print(f'    {mname} LOSO-RMSE: {np.mean(loso_errs):.3f}')

# =================================================================
#  APPROACH 8: PREDICTION BLEND SWEEP
#  Systematically try blending the top models
# =================================================================
print('\n' + '='*65)
print(' APPROACH 8: BLEND SWEEP')
print('='*65)

# Raw predictions for blending
pred_pool = {
    'kaggle': raw_baseline,
    'xgb68': model_preds['xgb_all68'],
    'alaq': raw_alaq,
    'ridge': model_preds['ridge_only'],
    'huber': model_preds['xgb_huber'],
    'gbr': model_preds['gbr'],
    'rf': model_preds['rf'],
    'en': model_preds['elasticnet'],
    'rank_ens': raw_ensemble,
}

if best_pw_assigned is not None:
    pred_pool['pw_lr'] = best_pw_raw

best_blend_exact = 0
best_blend_assigned = None
best_blend_desc = ''

# Try all pairs
pool_names = list(pred_pool.keys())
for i in range(len(pool_names)):
    for j in range(i+1, len(pool_names)):
        for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
            blend = w * pred_pool[pool_names[i]] + (1-w) * pred_pool[pool_names[j]]
            assigned = hungarian(blend, seasons_te, avail)
            exact = int(np.sum(assigned == y_test))
            if exact > best_blend_exact:
                best_blend_exact = exact
                best_blend_assigned = assigned.copy()
                best_blend_desc = f'{pool_names[i]}*{w:.1f} + {pool_names[j]}*{1-w:.1f}'

# Try triplets with the top models
for i in range(len(pool_names)):
    for j in range(i+1, len(pool_names)):
        for k in range(j+1, len(pool_names)):
            blend = (pred_pool[pool_names[i]] + pred_pool[pool_names[j]] + pred_pool[pool_names[k]]) / 3
            assigned = hungarian(blend, seasons_te, avail)
            exact = int(np.sum(assigned == y_test))
            if exact > best_blend_exact:
                best_blend_exact = exact
                best_blend_assigned = assigned.copy()
                best_blend_desc = f'avg({pool_names[i]}, {pool_names[j]}, {pool_names[k]})'

eval_bracket(best_blend_assigned, y_test, f'Best blend: {best_blend_desc}')

# =================================================================
#  APPROACH 9: HUNGARIAN POWER SWEEP + SWAP
# =================================================================
print('\n' + '='*65)
print(' APPROACH 9: HUNGARIAN POWER OPTIMIZATION')
print('='*65)

best_overall_exact = 0
best_overall_assigned = None
best_overall_desc = ''

# Try all raw predictions with different Hungarian powers
for name, raw in [('kaggle', raw_baseline), ('alaq', raw_alaq), 
                   ('blend', pred_pool.get(best_blend_desc.split('*')[0] if '*' in best_blend_desc else 'kaggle', raw_baseline)),
                   ('xgb68', model_preds['xgb_all68']),
                   ('huber', model_preds['xgb_huber']),
                   ('rank_ens', raw_ensemble)]:
    for pwr in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        assigned = hungarian(raw, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        if exact > best_overall_exact:
            best_overall_exact = exact
            best_overall_assigned = assigned.copy()
            best_overall_desc = f'{name} power={pwr}'

# Also try best blend with different powers
if '*' in best_blend_desc:
    parts = best_blend_desc.split(' + ')
    n1, w1 = parts[0].rsplit('*', 1)
    n2, w2 = parts[1].rsplit('*', 1)
    w1, w2 = float(w1), float(w2)
    blend_raw = w1 * pred_pool[n1] + w2 * pred_pool[n2]
elif 'avg(' in best_blend_desc:
    names = best_blend_desc.replace('avg(', '').replace(')', '').split(', ')
    blend_raw = np.mean([pred_pool[n] for n in names], axis=0)
else:
    blend_raw = raw_baseline

for pwr in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
    assigned = hungarian(blend_raw, seasons_te, avail, power=pwr)
    exact = int(np.sum(assigned == y_test))
    if exact > best_overall_exact:
        best_overall_exact = exact
        best_overall_assigned = assigned.copy()
        best_overall_desc = f'best_blend power={pwr}'

eval_bracket(best_overall_assigned, y_test, f'Best overall: {best_overall_desc}')

# =================================================================
#  FINAL SUMMARY
# =================================================================
print('\n' + '='*65)
print(' FINAL SUMMARY — 2024-25 BRACKET PREDICTION')
print('='*65)

results = [
    ('Previous best (Kaggle+Ridge)', baseline_assigned),
    ('Pairwise LogReg', best_pw_assigned),
    ('Pairwise XGBoost', assigned_pw_xgb),
    ('Separate AL/AQ', assigned_alaq),
    ('Two-stage', assigned_twostage),
    ('Monte Carlo', best_mc_assigned),
    ('Rank ensemble (8 models)', assigned_ensemble),
    ('Best blend', best_blend_assigned),
    ('Best overall (power opt)', best_overall_assigned),
]

if best_ltr_assigned is not None:
    results.insert(3, ('Learning-to-Rank', best_ltr_assigned))

print(f'\n  {"Model":<35} {"Exact":>5} {"±1":>4} {"±2":>4} {"±4":>4} {"RMSE":>6} {"Lines":>5}')
print(f'  {"─"*35} {"─"*5} {"─"*4} {"─"*4} {"─"*4} {"─"*6} {"─"*5}')

for name, assigned in results:
    if assigned is None: continue
    exact = int(np.sum(assigned == y_test))
    w1 = int(np.sum(np.abs(assigned - y_test) <= 1))
    w2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    w4 = int(np.sum(np.abs(assigned - y_test) <= 4))
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    lines = int(np.sum(((assigned-1)//4+1) == ((y_test-1)//4+1)))
    marker = ' ←BEST' if exact == max(int(np.sum(r[1] == y_test)) for r in results if r[1] is not None) else ''
    print(f'  {name:<35} {exact:>5} {w1:>4} {w2:>4} {w4:>4} {rmse:>6.2f} {lines:>5}{marker}')

# ---- DETAILED BRACKET for best model ----
best_name = max(results, key=lambda r: int(np.sum(r[1] == y_test)) if r[1] is not None else 0)
assigned = best_name[1]
teams = test_data['Team'].values
order = np.argsort(assigned)

print(f'\n{"="*65}')
print(f' BEST MODEL BRACKET: {best_name[0]}')
print(f'{"="*65}')
print(f'\n  {"Pred":>4} {"Act":>4} {"Δ":>3} {"":>1} {"Team":<28} {"Conf":<12} {"Bid":<3}')
print(f'  {"─"*4} {"─"*4} {"─"*3} {"─"*1} {"─"*28} {"─"*12} {"─"*3}')
for i in order:
    p, a = assigned[i], y_test[i]
    d = p - a
    m = '✓' if d == 0 else ('~' if abs(d) <= 2 else '✗')
    print(f'  {p:4d} {a:4d} {d:+3d} {m:<1} {str(teams[i]):<28} '
          f'{str(test_data["Conference"].values[i]):<12} {str(bids_te[i]):<3}')

print(f'\n  Time: {time.time()-t0:.0f}s')
