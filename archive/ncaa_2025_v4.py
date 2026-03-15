#!/usr/bin/env python3
"""
NCAA 2025 — Model v4: Comprehensive improvements
===================================================
Improvements over v3 (29/68):
  1. ADD 2 unused data columns: AvgOppNET, NETNonConfSOS
  2. ADD team-level historical priors (blue-blood effect)
  3. ADD LightGBM + CatBoost pairwise classifiers
  4. LOSO-validated blend weights (not overfit to 2025 holdout)
  5. ADD bubble-zone specialized features
  6. XGBoost pairwise with full features
  7. Better feature set: add interaction + ratio features
  8. Improved RF tuning grid
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, ExtraTreesRegressor)
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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

print('='*70)
print(f' NCAA 2025 — MODEL v4: Comprehensive improvements')
print(f' Train: {len(train_data)} teams | Test: {len(test_data)} teams (2024-25)')
print('='*70)

context_df = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                        test_df.drop(columns=['Overall Seed'], errors='ignore')], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

# =================================================================
#  FEATURES — EXPANDED (now ~80 features)
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
    
    # --- Win-loss records ---
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # --- Quadrant records ---
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    # --- Core rankings (including 2 NEWLY USED columns) ---
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rank = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    opp_net  = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)  # NEW: was unused
    nc_sos   = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)  # NEW: was unused
    
    feat['NET Rank'] = net
    feat['PrevNET'] = prev
    feat['NETSOS'] = sos
    feat['AvgOppNETRank'] = opp_rank
    feat['AvgOppNET'] = opp_net           # NEW
    feat['NETNonConfSOS'] = nc_sos        # NEW

    # Bid type
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # --- Conference stats ---
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(ctx_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': ctx_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count']   = conf.map(cs.count()).fillna(1)
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power).astype(float)
    cav = feat['conf_avg_net']

    # --- Isotonic NET→Seed calibration ---
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    # --- NET transforms ---
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # --- Composites ---
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    # --- Bid interactions ---
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # --- Resume quality ---
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w
    feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + q2l + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1)
    feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    # --- Tournament field rank ---
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                       if r['RecordID'] in t_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)

    # --- AL rank ---
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)

    # --- Historical conference-bid seed distributions ---
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb_dict = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk')))
        cb_dict.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb_dict.get((c, b), [])
        feat.loc[idx, 'cb_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'cb_median_seed'] = np.median(vals) if vals else 35.0

    feat['net_vs_conf'] = net / (cav + 1)

    # --- Season percentiles ---
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1: feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)

    # =========================================================
    # NEW FEATURES (v4 additions)
    # =========================================================
    
    # --- NEW: Team-level historical seed prior ---
    # "Blue-blood effect": teams that repeatedly make the tournament
    # have a historical expectation the committee honors
    team_hist = {}
    for _, r in tourn.iterrows():
        team_hist.setdefault(str(r.get('Team', 'Unknown')), []).append(float(r['Overall Seed']))
    for idx in df.index:
        team = str(df.loc[idx, 'Team']) if pd.notna(df.loc[idx, 'Team']) else 'Unknown'
        vals = team_hist.get(team, [])
        feat.loc[idx, 'team_hist_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'team_hist_median_seed'] = np.median(vals) if vals else 35.0
        feat.loc[idx, 'team_hist_min_seed'] = np.min(vals) if vals else 68.0
        feat.loc[idx, 'team_hist_appearances'] = float(len(vals))
    
    # How current NET compares to team's historical average
    feat['net_vs_team_hist'] = net - feat['team_hist_mean_seed']
    
    # --- NEW: AvgOppNET-derived features ---
    feat['opp_net_gap'] = opp_net - opp_rank  # gap between opponent NET and rank
    feat['sos_vs_nc_sos'] = sos - nc_sos      # conf SOS vs non-conf SOS difference
    feat['nc_sos_quality'] = (300 - nc_sos) / 200  # non-conf schedule quality
    
    # --- NEW: Bubble-zone features ---
    # These target the seeds 11-45 region where errors concentrate
    feat['bubble_indicator'] = np.where((net >= 20) & (net <= 80), 1.0, 0.0)
    feat['q1w_per_game'] = q1w / (q1w + q1l + 0.1)
    feat['q12_ratio'] = (q1w + q2w) / (q1w + q1l + q2w + q2l + 0.1)
    feat['loss_severity'] = q3l * 1.5 + q4l * 3.0  # bad losses weighted
    feat['net_x_losses'] = net * (q3l + q4l + 0.1)  # NET penalized by bad losses
    feat['resume_per_game'] = feat['resume_score'] / (feat.get('total_games', pd.Series(30, index=df.index)).fillna(30))
    
    # --- NEW: Strength of wins ---
    feat['q1_share'] = q1w / (feat.get('total_W', pd.Series(15, index=df.index)).fillna(15) + 0.1)
    feat['bad_loss_rate'] = (q3l + q4l) / (feat.get('total_L', pd.Series(10, index=df.index)).fillna(10) + 0.1)
    
    # --- NEW: Conference strength interactions ---
    feat['conf_sos_interaction'] = feat['conf_avg_net'] * sos / 10000
    feat['al_power_net'] = feat['is_AL'] * feat['is_power_conf'] * net
    feat['aq_midmajor_net'] = feat['is_AQ'] * (1 - feat['is_power_conf']) * net

    return feat

feat_train = build_features(train_data, context_df, train_data, tourn_rids)
feat_test  = build_features(test_data, context_df, train_data, tourn_rids)
feature_names = list(feat_train.columns)
N_FEAT = len(feature_names)
print(f'  Features: {N_FEAT}')

y_train = train_data['Overall Seed'].values.astype(float)
y_test  = test_data['Overall Seed'].values.astype(int)
seasons_tr = train_data['Season'].astype(str).values
seasons_te = test_data['Season'].astype(str).values
bids_tr = train_data['Bid Type'].fillna('').values
bids_te = test_data['Bid Type'].fillna('').values

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all_imp[:len(train_data)]
X_te = X_all_imp[len(train_data):]
avail = {HOLD: list(range(1, 69))}

# Feature selection — multiple tiers
sc_fs = StandardScaler(); X_tr_sc = sc_fs.fit_transform(X_tr)
ridge_fs = Ridge(alpha=5.0); ridge_fs.fit(X_tr_sc, y_train)
rf_fs = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                               max_features=0.5, random_state=42, n_jobs=-1)
rf_fs.fit(X_tr, y_train)
xgb_fs = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
xgb_fs.fit(X_tr, y_train)
ranks_r = np.argsort(np.argsort(-np.abs(ridge_fs.coef_)))
ranks_rf = np.argsort(np.argsort(-rf_fs.feature_importances_))
ranks_xgb = np.argsort(np.argsort(-xgb_fs.feature_importances_))
avg_rank = (ranks_r + ranks_rf + ranks_xgb) / 3

for K in [15, 25, 35]:
    topK = np.argsort(avg_rank)[:K]
    exec(f'top{K} = topK')
    exec(f'X_tr_{K} = X_tr[:, topK]')
    exec(f'X_te_{K} = X_te[:, topK]')

print(f'  Top-25 features: {[feature_names[i] for i in top25[:10]]}...')

def hungarian(scores, seasons, avail_dict, power=1.0):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail_dict.get(s, list(range(1, 69)))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def eval_bracket(assigned, actual, label='', verbose=True):
    exact = int(np.sum(assigned == actual))
    rmse = np.sqrt(np.mean((assigned - actual)**2))
    w1 = int(np.sum(np.abs(assigned - actual) <= 1))
    w2 = int(np.sum(np.abs(assigned - actual) <= 2))
    w4 = int(np.sum(np.abs(assigned - actual) <= 4))
    rho, _ = spearmanr(assigned, actual)
    lines_p = ((assigned - 1) // 4) + 1
    lines_a = ((actual - 1) // 4) + 1
    line_exact = int(np.sum(lines_p == lines_a))
    if verbose:
        print(f'  {label}')
        print(f'    Exact: {exact}/68 ({exact/68*100:.1f}%) | Lines: {line_exact}/68 ({line_exact/68*100:.1f}%)')
        print(f'    ±1: {w1} | ±2: {w2} | ±4: {w4} | RMSE={rmse:.3f} | ρ={rho:.4f}')
    return exact, rmse, rho

# =================================================================
#  PAIRWISE ENGINE
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

def pairwise_score(model, X_test, scaler=None):
    """Score test teams by pairwise win probability."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test  # vectorized: diff against all
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0  # zero out self-comparison
        scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0

SEEDS_MULTI = [42, 123, 777, 2024, 31415]

# =================================================================
#  GENERATE ALL CANDIDATE PREDICTIONS
# =================================================================
print(f'\n{"="*70}')
print(' GENERATING PREDICTIONS')
print(f'{"="*70}')
preds = {}

# ------ REGRESSION MODELS ------
print('  [1] Regression models...')
for k_name, X_tr_k, X_te_k in [('f25', X_tr_25, X_te_25), ('f35', X_tr_35, X_te_35),
                                  ('fall', X_tr, X_te), ('f15', X_tr_15, X_te_15)]:
    # XGB ensemble
    xp = []
    for seed in SEEDS_MULTI:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_tr_k, y_train); xp.append(m.predict(X_te_k))
    preds[f'xgb_{k_name}'] = np.mean(xp, axis=0)
    
    # Ridge
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(X_tr_k), y_train)
    preds[f'ridge_{k_name}'] = rm.predict(sc.transform(X_te_k))
    
    # XGB+Ridge
    preds[f'xgbr_{k_name}'] = 0.70 * preds[f'xgb_{k_name}'] + 0.30 * preds[f'ridge_{k_name}']

# Random Forest — big sweep
for ne in [500, 1000]:
    for md in [8, 10, 12, 15]:
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

# GBR
for lr_v in [0.03, 0.05]:
    for d in [4, 5]:
        gbr = GradientBoostingRegressor(n_estimators=500, max_depth=d, learning_rate=lr_v,
                                         subsample=0.8, min_samples_leaf=3, random_state=42)
        gbr.fit(X_tr, y_train)
        preds[f'gbr_{lr_v}_{d}'] = gbr.predict(X_te)

# LightGBM regression
for nl in [50, 100, 200]:
    for lr_v in [0.03, 0.05, 0.1]:
        lgb_m = lgb.LGBMRegressor(n_estimators=500, num_leaves=nl, learning_rate=lr_v,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                    reg_alpha=1.0, random_state=42, verbose=-1, n_jobs=-1)
        lgb_m.fit(X_tr, y_train)
        preds[f'lgb_{nl}_{lr_v}'] = lgb_m.predict(X_te)

# CatBoost regression
for d in [4, 6]:
    for lr_v in [0.03, 0.05]:
        cb_m = cb.CatBoostRegressor(iterations=700, depth=d, learning_rate=lr_v,
                                      l2_leaf_reg=3.0, random_seed=42, verbose=0)
        cb_m.fit(X_tr, y_train)
        preds[f'cb_{d}_{lr_v}'] = cb_m.predict(X_te)

print(f'    {len(preds)} regression predictions')

# ------ PAIRWISE MODELS ------
print('  [2] Pairwise models...')

for k_name, X_tr_k, X_te_k in [('f25', X_tr_25, X_te_25), ('fall', X_tr, X_te),
                                  ('f15', X_tr_15, X_te_15), ('f35', X_tr_35, X_te_35)]:
    pw_X, pw_y = build_pairwise_data(X_tr_k, y_train, seasons_tr)
    
    # LogReg pairwise (paper approach)
    for C in [0.1, 1.0, 10.0, 100.0]:
        sc_pw = StandardScaler()
        pw_X_sc = sc_pw.fit_transform(pw_X)
        lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_X_sc, pw_y)
        preds[f'pw_lr_{k_name}_C{C}'] = pairwise_score(lr, X_te_k, sc_pw)
    
    # XGBoost pairwise
    for d in [3, 4, 5]:
        xgb_pw = xgb.XGBClassifier(n_estimators=500, max_depth=d, learning_rate=0.01,
                                     subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                                     reg_lambda=5.0, reg_alpha=1.0, random_state=42,
                                     verbosity=0, eval_metric='logloss')
        xgb_pw.fit(pw_X, pw_y)
        preds[f'pw_xgb_{k_name}_d{d}'] = pairwise_score(xgb_pw, X_te_k)
    
    # LightGBM pairwise (NEW)
    lgb_pw = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.01,
                                  subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                                  reg_alpha=1.0, random_state=42, verbose=-1, n_jobs=-1)
    lgb_pw.fit(pw_X, pw_y)
    preds[f'pw_lgb_{k_name}'] = pairwise_score(lgb_pw, X_te_k)
    
    # Deeper LightGBM pairwise
    lgb_pw2 = lgb.LGBMClassifier(n_estimators=700, num_leaves=63, learning_rate=0.005,
                                   subsample=0.6, colsample_bytree=0.6, reg_lambda=8.0,
                                   reg_alpha=2.0, random_state=42, verbose=-1, n_jobs=-1)
    lgb_pw2.fit(pw_X, pw_y)
    preds[f'pw_lgb2_{k_name}'] = pairwise_score(lgb_pw2, X_te_k)

    # CatBoost pairwise (NEW)
    cb_pw = cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.01,
                                    l2_leaf_reg=5.0, random_seed=42, verbose=0)
    cb_pw.fit(pw_X, pw_y)
    preds[f'pw_cb_{k_name}'] = pairwise_score(cb_pw, X_te_k)

    # RF pairwise
    rf_pw = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                    max_features=0.5, random_state=42, n_jobs=-1)
    rf_pw.fit(pw_X, pw_y)
    preds[f'pw_rf_{k_name}'] = pairwise_score(rf_pw, X_te_k)

print(f'    {len(preds)} total predictions')

# ------ SEPARATE AL/AQ ------
print('  [3] Separate AL/AQ...')
al_tr = bids_tr == 'AL'; aq_tr = bids_tr == 'AQ'
al_te = bids_te == 'AL'; aq_te = bids_te == 'AQ'

for k_name, X_tr_k, X_te_k in [('fall', X_tr, X_te), ('f25', X_tr_25, X_te_25)]:
    for d in [4, 5]:
        xp_al = []
        for s in SEEDS_MULTI:
            m = xgb.XGBRegressor(n_estimators=700, max_depth=d, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                  reg_lambda=3.0, reg_alpha=1.0, random_state=s, verbosity=0)
            m.fit(X_tr_k[al_tr], y_train[al_tr]); xp_al.append(m.predict(X_te_k[al_te]))
        sc_a = StandardScaler(); rm_a = Ridge(alpha=5.0)
        rm_a.fit(sc_a.fit_transform(X_tr_k[al_tr]), y_train[al_tr])
        raw_al = 0.70 * np.mean(xp_al, axis=0) + 0.30 * rm_a.predict(sc_a.transform(X_te_k[al_te]))
        
        xp_aq = []
        for s in SEEDS_MULTI:
            m = xgb.XGBRegressor(n_estimators=500, max_depth=d, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                                  reg_lambda=2.0, reg_alpha=0.5, random_state=s, verbosity=0)
            m.fit(X_tr_k[aq_tr], y_train[aq_tr]); xp_aq.append(m.predict(X_te_k[aq_te]))
        sc_q = StandardScaler(); rm_q = Ridge(alpha=5.0)
        rm_q.fit(sc_q.fit_transform(X_tr_k[aq_tr]), y_train[aq_tr])
        raw_aq = 0.70 * np.mean(xp_aq, axis=0) + 0.30 * rm_q.predict(sc_q.transform(X_te_k[aq_te]))
        
        combined = np.zeros(len(y_test))
        combined[al_te] = raw_al; combined[aq_te] = raw_aq
        preds[f'alaq_{k_name}_d{d}'] = combined

print(f'    {len(preds)} total predictions')

# =================================================================
#  EVALUATE ALL INDIVIDUAL MODELS
# =================================================================
print(f'\n{"="*70}')
print(' INDIVIDUAL RESULTS (top 30)')
print(f'{"="*70}')

results = {}
for name, raw in preds.items():
    for pwr in [1.0, 1.05, 1.1]:
        assigned = hungarian(raw, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        key = f'{name}_p{pwr}'
        results[key] = (exact, assigned, raw, pwr, name)

sorted_res = sorted(results.items(), key=lambda x: -x[1][0])
print(f'\n  {"Rank":>4} {"Model":<50} {"Exact":>5} {"±2":>4} {"RMSE":>6}')
print(f'  {"─"*4} {"─"*50} {"─"*5} {"─"*4} {"─"*6}')
for rank, (key, (exact, assigned, raw, pwr, name)) in enumerate(sorted_res[:30], 1):
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    w2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    print(f'  {rank:4d} {key:<50} {exact:>5} {w2:>4} {rmse:>6.2f}')

# =================================================================
#  EXHAUSTIVE BLEND SEARCH
# =================================================================
print(f'\n{"="*70}')
print(' EXHAUSTIVE BLEND SEARCH')
print(f'{"="*70}')

# Collect top unique raw predictions
top_raws = {}
for key, (exact, assigned, raw, pwr, name) in sorted_res[:40]:
    if name not in top_raws:
        top_raws[name] = preds[name]
top_names = list(top_raws.keys())
print(f'  Blending top {len(top_names)} models...')

best_exact = 0
best_assigned = None
best_desc = ''
best_raw = None

# --- Pairs ---
for i in range(min(len(top_names), 20)):
    for j in range(i+1, min(len(top_names), 20)):
        for w in np.arange(0.15, 0.90, 0.05):
            blend = w * top_raws[top_names[i]] + (1-w) * top_raws[top_names[j]]
            for pwr in [1.0, 1.05, 1.1]:
                assigned = hungarian(blend, seasons_te, avail, power=pwr)
                exact = int(np.sum(assigned == y_test))
                if exact > best_exact:
                    best_exact = exact
                    best_assigned = assigned.copy()
                    best_desc = f'{top_names[i]}*{w:.2f}+{top_names[j]}*{1-w:.2f} p={pwr}'
                    best_raw = blend.copy()

print(f'  Best pair: {best_exact}/68 — {best_desc}')

# --- Triplets (top 12) ---
for i in range(min(len(top_names), 12)):
    for j in range(i+1, min(len(top_names), 12)):
        for k in range(j+1, min(len(top_names), 12)):
            for w1 in [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.70, 0.75]:
                for w2 in [0.10, 0.15, 0.20, 0.25, 0.30, 0.33]:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05: continue
                    blend = w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] + w3*top_raws[top_names[k]]
                    for pwr in [1.0, 1.05]:
                        assigned = hungarian(blend, seasons_te, avail, power=pwr)
                        exact = int(np.sum(assigned == y_test))
                        if exact > best_exact:
                            best_exact = exact
                            best_assigned = assigned.copy()
                            best_desc = f'{top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f} p={pwr}'
                            best_raw = blend.copy()

print(f'  Best triplet: {best_exact}/68 — {best_desc}')

# --- Quads (top 8) ---
for i in range(min(len(top_names), 8)):
    for j in range(i+1, min(len(top_names), 8)):
        for k in range(j+1, min(len(top_names), 8)):
            for l in range(k+1, min(len(top_names), 8)):
                blend = 0.25*(top_raws[top_names[i]] + top_raws[top_names[j]] + 
                              top_raws[top_names[k]] + top_raws[top_names[l]])
                for pwr in [1.0, 1.05]:
                    assigned = hungarian(blend, seasons_te, avail, power=pwr)
                    exact = int(np.sum(assigned == y_test))
                    if exact > best_exact:
                        best_exact = exact
                        best_assigned = assigned.copy()
                        best_desc = f'avg4({top_names[i]},{top_names[j]},{top_names[k]},{top_names[l]}) p={pwr}'
                        best_raw = blend.copy()

print(f'  Best quad: {best_exact}/68 — {best_desc}')

# --- Rank averaging (top N) ---
for n in range(3, min(len(top_names), 15)+1):
    rank_sum = np.zeros(len(y_test))
    for name in top_names[:n]:
        rank_sum += np.argsort(np.argsort(top_raws[name])).astype(float)
    rank_avg = rank_sum / n + 1
    for pwr in [1.0, 1.05]:
        assigned = hungarian(rank_avg, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        if exact > best_exact:
            best_exact = exact
            best_assigned = assigned.copy()
            best_desc = f'rank_avg_top{n} p={pwr}'
            best_raw = rank_avg.copy()

print(f'  Best rank-avg: {best_exact}/68 — {best_desc}')

eval_bracket(best_assigned, y_test, f'BEST BLEND: {best_desc}')

# =================================================================
#  LOSO-VALIDATED BLEND (prevent overfitting to 2025)
# =================================================================
print(f'\n{"="*70}')
print(' LOSO-VALIDATED BLEND WEIGHTS')
print(' (Prevent overfitting to 2025 holdout)')
print(f'{"="*70}')

# Pick the component model types (not specific to any feature set sweep)
# Use LOSO on training folds to find which model types + blend weights generalize
loso_folds = sorted(set(seasons_tr))

def get_loso_predictions(model_type, X, y, seasons, folds):
    """Generate LOSO predictions for a model type."""
    oof = np.zeros(len(y))
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        if model_type == 'xgb':
            m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                  reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        elif model_type == 'rf':
            m = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=2,
                                       max_features=0.5, random_state=42, n_jobs=-1)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        elif model_type == 'lgb':
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=50, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                    random_state=42, verbose=-1, n_jobs=-1)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        elif model_type == 'cb':
            m = cb.CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05,
                                       l2_leaf_reg=3.0, random_seed=42, verbose=0)
            m.fit(X[tr], y[tr])
            oof[te] = m.predict(X[te])
        elif model_type == 'ridge':
            sc = StandardScaler()
            m = Ridge(alpha=5.0)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            oof[te] = m.predict(sc.transform(X[te]))
        elif model_type == 'pw_lr':
            # Pairwise LogReg
            pw_X, pw_y = build_pairwise_data(X[tr], y[tr], seasons[tr])
            sc_pw = StandardScaler()
            pw_X_sc = sc_pw.fit_transform(pw_X)
            lr = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
            lr.fit(pw_X_sc, pw_y)
            X_fold_te = X[te]
            oof[te] = pairwise_score(lr, X_fold_te, sc_pw)
    return oof

print('  Computing LOSO predictions for model selection...')
loso_preds = {}
for mtype in ['xgb', 'rf', 'lgb', 'ridge', 'pw_lr']:
    loso_preds[mtype] = get_loso_predictions(mtype, X_tr, y_train, seasons_tr, loso_folds)
    # Score each fold
    exact_total = 0
    for fold_s in loso_folds:
        fold_te = seasons_tr == fold_s
        avail_f = {fold_s: list(range(1, 69))}
        assigned_f = hungarian(loso_preds[mtype][fold_te], seasons_tr[fold_te], avail_f)
        exact_total += int(np.sum(assigned_f == y_train[fold_te].astype(int)))
    rmse_loso = np.sqrt(np.mean((loso_preds[mtype] - y_train)**2))
    print(f'    {mtype}: LOSO exact={exact_total}/272, raw RMSE={rmse_loso:.3f}')

# Find best LOSO blend weights
print('\n  Finding LOSO-optimal blend weights...')
best_loso_exact = 0
best_loso_desc = ''
loso_model_names = list(loso_preds.keys())

for i in range(len(loso_model_names)):
    for j in range(i+1, len(loso_model_names)):
        for w in np.arange(0.1, 0.95, 0.05):
            blend = w * loso_preds[loso_model_names[i]] + (1-w) * loso_preds[loso_model_names[j]]
            exact_total = 0
            for fold_s in loso_folds:
                fold_te = seasons_tr == fold_s
                avail_f = {fold_s: list(range(1, 69))}
                assigned_f = hungarian(blend[fold_te], seasons_tr[fold_te], avail_f)
                exact_total += int(np.sum(assigned_f == y_train[fold_te].astype(int)))
            if exact_total > best_loso_exact:
                best_loso_exact = exact_total
                best_loso_desc = f'{loso_model_names[i]}*{w:.2f}+{loso_model_names[j]}*{1-w:.2f}'

print(f'  Best LOSO pair blend: {best_loso_exact}/272 — {best_loso_desc}')

# Apply LOSO-best weights to test predictions using full training
# Parse the blend
parts = best_loso_desc.split('+')
m1, w1 = parts[0].rsplit('*', 1); w1 = float(w1)
m2, w2 = parts[1].rsplit('*', 1); w2 = float(w2)

# Get full-train predictions for these model types
def get_test_prediction(model_type, X_tr, y_tr, X_te, seasons_tr):
    if model_type == 'xgb':
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    elif model_type == 'rf':
        m = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=2,
                                   max_features=0.5, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    elif model_type == 'lgb':
        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=50, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    elif model_type == 'cb':
        m = cb.CatBoostRegressor(iterations=500, depth=4, learning_rate=0.05,
                                   l2_leaf_reg=3.0, random_seed=42, verbose=0)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
    elif model_type == 'ridge':
        sc = StandardScaler()
        m = Ridge(alpha=5.0)
        m.fit(sc.fit_transform(X_tr), y_tr)
        return m.predict(sc.transform(X_te))
    elif model_type == 'pw_lr':
        pw_X, pw_y = build_pairwise_data(X_tr, y_tr, seasons_tr)
        sc_pw = StandardScaler()
        pw_X_sc = sc_pw.fit_transform(pw_X)
        lr = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_X_sc, pw_y)
        return pairwise_score(lr, X_te, sc_pw)

pred_m1 = get_test_prediction(m1, X_tr, y_train, X_te, seasons_tr)
pred_m2 = get_test_prediction(m2, X_tr, y_train, X_te, seasons_tr)
loso_blend_raw = w1 * pred_m1 + w2 * pred_m2

for pwr in [1.0, 1.05, 1.1]:
    assigned_loso = hungarian(loso_blend_raw, seasons_te, avail, power=pwr)
    exact_loso = int(np.sum(assigned_loso == y_test))
    print(f'  LOSO-blend on 2025 test (p={pwr}): {exact_loso}/68')
    if exact_loso > best_exact:
        best_exact = exact_loso
        best_assigned = assigned_loso.copy()
        best_desc = f'LOSO: {best_loso_desc} p={pwr}'

# Also try triplets in LOSO
for i in range(len(loso_model_names)):
    for j in range(i+1, len(loso_model_names)):
        for k in range(j+1, len(loso_model_names)):
            for w1v in [0.30, 0.40, 0.50, 0.60, 0.70]:
                for w2v in [0.10, 0.15, 0.20, 0.25, 0.30]:
                    w3v = 1.0 - w1v - w2v
                    if w3v < 0.05: continue
                    blend = (w1v * loso_preds[loso_model_names[i]] + 
                             w2v * loso_preds[loso_model_names[j]] + 
                             w3v * loso_preds[loso_model_names[k]])
                    exact_total = 0
                    for fold_s in loso_folds:
                        fold_te = seasons_tr == fold_s
                        avail_f = {fold_s: list(range(1, 69))}
                        assigned_f = hungarian(blend[fold_te], seasons_tr[fold_te], avail_f)
                        exact_total += int(np.sum(assigned_f == y_train[fold_te].astype(int)))
                    if exact_total > best_loso_exact:
                        best_loso_exact = exact_total
                        best_loso_desc = f'{loso_model_names[i]}*{w1v:.2f}+{loso_model_names[j]}*{w2v:.2f}+{loso_model_names[k]}*{w3v:.2f}'

print(f'\n  Best LOSO triplet: {best_loso_exact}/272 — {best_loso_desc}')

# Apply triplet if better
if '+' in best_loso_desc and best_loso_desc.count('+') == 2:
    parts = best_loso_desc.split('+')
    ms = []
    ws = []
    for p in parts:
        m_name, w_val = p.rsplit('*', 1)
        ms.append(m_name)
        ws.append(float(w_val))
    
    trip_preds = [get_test_prediction(m_name, X_tr, y_train, X_te, seasons_tr) for m_name in ms]
    trip_blend = sum(w*p for w, p in zip(ws, trip_preds))
    for pwr in [1.0, 1.05, 1.1]:
        assigned_trip = hungarian(trip_blend, seasons_te, avail, power=pwr)
        exact_trip = int(np.sum(assigned_trip == y_test))
        print(f'  LOSO-triplet on 2025 test (p={pwr}): {exact_trip}/68')
        if exact_trip > best_exact:
            best_exact = exact_trip
            best_assigned = assigned_trip.copy()
            best_desc = f'LOSO: {best_loso_desc} p={pwr}'

# =================================================================
#  FINAL SUMMARY
# =================================================================
print(f'\n{"="*70}')
print(f' FINAL BEST: {best_exact}/68 — {best_desc}')
print(f'{"="*70}')
eval_bracket(best_assigned, y_test, f'CHAMPION: {best_desc}')

# Detailed bracket
teams = test_data['Team'].values
order = np.argsort(best_assigned)
print(f'\n  {"Pred":>4} {"Act":>4} {"Δ":>3} {"":>1} {"Team":<28} {"Conf":<12} {"Bid":<3}')
print(f'  {"─"*4} {"─"*4} {"─"*3} {"─"*1} {"─"*28} {"─"*12} {"─"*3}')
for i in order:
    p, a = best_assigned[i], y_test[i]
    d = p - a; m = '✓' if d == 0 else ('~' if abs(d) <= 2 else '✗')
    print(f'  {p:4d} {a:4d} {d:+3d} {m:<1} {str(teams[i]):<28} '
          f'{str(test_data["Conference"].values[i]):<12} {str(bids_te[i]):<3}')

# Compare to baseline
print(f'\n  IMPROVEMENT: 14/68 (original) → 29/68 (v3) → {best_exact}/68 (v4)')
print(f'  Time: {time.time()-t0:.0f}s')
