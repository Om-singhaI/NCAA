#!/usr/bin/env python3
"""
NCAA 2025 Bracket Simulation
==============================
Train on 2020-21 through 2023-24, predict full 2024-25 bracket (68 teams).
Compare against known actual seeds.
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
test_df['Overall Seed'] = test_df['RecordID'].map(GT).fillna(0).astype(int)

# Combine all labeled data
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_labeled = all_df[all_df['Overall Seed'] > 0].copy()

# SPLIT: Train = seasons != 2024-25, Test = 2024-25
HOLD_SEASON = '2024-25'
train_data = all_labeled[all_labeled['Season'] != HOLD_SEASON].copy()
test_data  = all_labeled[all_labeled['Season'] == HOLD_SEASON].copy()

print('='*65)
print(f' NCAA 2025 BRACKET SIMULATION')
print(f' Train: {len(train_data)} teams from {sorted(train_data["Season"].unique())}')
print(f' Test:  {len(test_data)} teams from 2024-25')
print('='*65)

# Context for features = all teams in dataset
context_df = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df.drop(columns=['Overall Seed'], errors='ignore')
], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

# ---- Feature engineering (same as ncaa_2026_model.py) ----
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

# Build features
# For training: use only pre-2024-25 labeled data for calibration (no leakage)
feat_train = build_features(train_data, context_df, train_data, tourn_rids)
feat_test  = build_features(test_data,  context_df, train_data, tourn_rids)
feature_names = list(feat_train.columns)

y_train = train_data['Overall Seed'].values.astype(float)
y_test  = test_data['Overall Seed'].values.astype(int)

# Impute
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:len(train_data)]
X_te = X_all[len(train_data):]

# Feature selection (top-25)
sc_tmp = StandardScaler(); X_tr_sc = sc_tmp.fit_transform(X_tr)
ridge_imp = Ridge(alpha=5.0); ridge_imp.fit(X_tr_sc, y_train); coef_imp = np.abs(ridge_imp.coef_)
rf_imp = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)
rf_imp.fit(X_tr, y_train); rf_importance = rf_imp.feature_importances_
xgb_imp = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
xgb_imp.fit(X_tr, y_train); xgb_importance = xgb_imp.feature_importances_
ranks_r = np.argsort(np.argsort(-coef_imp)); ranks_rf = np.argsort(np.argsort(-rf_importance)); ranks_xgb = np.argsort(np.argsort(-xgb_importance))
avg_rank = (ranks_r + ranks_rf + ranks_xgb) / 3
top25 = np.argsort(avg_rank)[:25]
X_tr_25 = X_tr[:, top25]; X_te_25 = X_te[:, top25]

# ---- PREDICT WITH ALL 3 CONFIGS ----
KAGGLE_XGB = {
    'objective': 'reg:squarederror', 'booster': 'gbtree', 'eta': 0.0093,
    'subsample': 0.5, 'colsample_bynode': 0.8, 'num_parallel_tree': 2,
    'min_child_weight': 4, 'max_depth': 4, 'tree_method': 'hist',
    'grow_policy': 'lossguide', 'max_bin': 38, 'verbosity': 0, 'seed': 42,
}
V40_XGB = {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
           'reg_lambda': 3.0, 'reg_alpha': 1.0}
SEEDS = [42, 123, 777, 2024, 31415]
avail = {HOLD_SEASON: list(range(1, 69))}

def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, list(range(1, 69)))
        rv = [scores[i] for i in si]; cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

seasons_te = test_data['Season'].astype(str).values

configs = {}

# 1. Kaggle+Ridge (Top25) — LOSO best
dtrain = xgb.DMatrix(X_tr_25, label=y_train); dtest = xgb.DMatrix(X_te_25)
xgb_pred = xgb.train(KAGGLE_XGB, dtrain, num_boost_round=704).predict(dtest)
sc = StandardScaler(); rm = Ridge(alpha=2.0); rm.fit(sc.fit_transform(X_tr_25), y_train)
ridge_pred = rm.predict(sc.transform(X_te_25))
raw_kaggle = 0.60 * xgb_pred + 0.40 * ridge_pred
configs['Kaggle+Ridge (Top25)'] = hungarian(raw_kaggle, seasons_te, avail)

# 2. v40 baseline (All 68 features)
xpreds = []
for seed in SEEDS:
    m = xgb.XGBRegressor(**V40_XGB, random_state=seed, verbosity=0); m.fit(X_tr, y_train); xpreds.append(m.predict(X_te))
xgb_avg = np.mean(xpreds, axis=0)
sc2 = StandardScaler(); rm2 = Ridge(alpha=5.0); rm2.fit(sc2.fit_transform(X_tr), y_train)
raw_v40 = 0.70 * xgb_avg + 0.30 * rm2.predict(sc2.transform(X_te))
configs['v40 (All68)'] = hungarian(raw_v40, seasons_te, avail)

# 3. v40 baseline (Top25)
xpreds25 = []
for seed in SEEDS:
    m = xgb.XGBRegressor(**V40_XGB, random_state=seed, verbosity=0); m.fit(X_tr_25, y_train); xpreds25.append(m.predict(X_te_25))
xgb_avg25 = np.mean(xpreds25, axis=0)
sc3 = StandardScaler(); rm3 = Ridge(alpha=5.0); rm3.fit(sc3.fit_transform(X_tr_25), y_train)
raw_v40_25 = 0.70 * xgb_avg25 + 0.30 * rm3.predict(sc3.transform(X_te_25))
configs['v40 (Top25)'] = hungarian(raw_v40_25, seasons_te, avail)

# ---- RESULTS ----
print(f'\n  {len(test_data)} teams in 2024-25 bracket')
print()

for cfg_name, assigned in configs.items():
    exact = int(np.sum(assigned == y_test))
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    rho, _ = spearmanr(assigned, y_test)
    within1 = int(np.sum(np.abs(assigned - y_test) <= 1))
    within2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    within4 = int(np.sum(np.abs(assigned - y_test) <= 4))
    print(f'  {cfg_name}:')
    print(f'    Exact: {exact}/68 ({exact/68*100:.1f}%)')
    print(f'    Within ±1: {within1}/68 | Within ±2: {within2}/68 | Within ±4: {within4}/68')
    print(f'    RMSE={rmse:.3f}, Spearman ρ={rho:.4f}')
    print()

# ---- DETAILED BRACKET (best config) ----
# Use the LOSO-best config for the detailed view
best_name = min(configs, key=lambda k: np.sqrt(np.mean((configs[k] - y_test)**2)))
assigned = configs[best_name]

print('='*65)
print(f' 2024-25 PREDICTED BRACKET ({best_name})')
print('='*65)

teams = test_data['Team'].values
confs = test_data['Conference'].values
bids  = test_data['Bid Type'].fillna('').values
nets  = pd.to_numeric(test_data['NET Rank'], errors='coerce').values

# Sort by predicted seed
order = np.argsort(assigned)

print(f'\n  {"Pred":>4} {"Actual":>6} {"Δ":>3} {"":>3} {"Team":<28} {"Conf":<12} {"Bid":<3} {"NET":>4}')
print(f'  {"─"*4} {"─"*6} {"─"*3} {"─"*3} {"─"*28} {"─"*12} {"─"*3} {"─"*4}')

total_exact = 0
for i in order:
    pred = assigned[i]
    actual = y_test[i]
    delta = pred - actual
    seed_line_p = ((pred - 1) // 4) + 1
    seed_line_a = ((actual - 1) // 4) + 1

    if delta == 0:
        marker = '✓'
        total_exact += 1
    elif abs(delta) <= 2:
        marker = '~'
    else:
        marker = '✗'

    print(f'  {pred:4d} {actual:6d} {delta:+3d}  {marker:<3} {str(teams[i]):<28} '
          f'{str(confs[i]):<12} {str(bids[i]):<3} {nets[i]:4.0f}')

# Seed line accuracy
seed_lines_pred   = ((assigned - 1) // 4) + 1
seed_lines_actual = ((y_test - 1) // 4) + 1
line_exact = int(np.sum(seed_lines_pred == seed_lines_actual))

print(f'\n  ───────────────────────────────────────────────')
print(f'  Exact seed matches:      {total_exact}/68 ({total_exact/68*100:.1f}%)')
print(f'  Correct seed line (1-17): {line_exact}/68 ({line_exact/68*100:.1f}%)')
print(f'  Within ±1 seed:         {int(np.sum(np.abs(assigned - y_test) <= 1))}/68')
print(f'  Within ±4 seeds:        {int(np.sum(np.abs(assigned - y_test) <= 4))}/68')

# ---- ERROR ANALYSIS ----
print(f'\n  BIGGEST MISSES (|Δ| > 5):')
errors = np.abs(assigned - y_test)
big_miss = np.where(errors > 5)[0]
for i in sorted(big_miss, key=lambda x: -errors[x]):
    print(f'    {str(teams[i]):<25} Pred={assigned[i]:2d} Actual={y_test[i]:2d} '
          f'Δ={assigned[i]-y_test[i]:+3d} NET={nets[i]:.0f} {str(bids[i]):<3} {str(confs[i])}')

# ---- SEED LINE BRACKET VIEW ----
print(f'\n{"="*65}')
print(f' SEED LINE VIEW (4 teams per line)')
print(f'{"="*65}')
for line in range(1, 18):
    seed_start = (line - 1) * 4 + 1
    seed_end = line * 4
    pred_in_line = [i for i in range(len(assigned)) if seed_start <= assigned[i] <= seed_end]
    actual_in_line = [i for i in range(len(y_test)) if seed_start <= y_test[i] <= seed_end]
    
    pred_teams = sorted(pred_in_line, key=lambda x: assigned[x])
    actual_teams = sorted(actual_in_line, key=lambda x: y_test[x])
    
    pred_names = [str(teams[i])[:15] for i in pred_teams]
    actual_names = [str(teams[i])[:15] for i in actual_teams]
    
    # How many predicted correctly landed in this line?
    correct = len(set(pred_in_line) & set(actual_in_line))
    
    print(f'\n  Line {line:2d} (seeds {seed_start}-{seed_end}): {correct}/{len(actual_in_line)} correct')
    print(f'    Predicted: {", ".join(pred_names) if pred_names else "(none)"}')
    print(f'    Actual:    {", ".join(actual_names) if actual_names else "(none)"}')

print(f'\n  Time: {time.time()-t0:.0f}s')
