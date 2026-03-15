#!/usr/bin/env python3
"""
NCAA v11 — Kaggle Winner-Inspired Model
=========================================
Adapted from modeh7's "Final Solution NCAA 2025" winning approach.

Key concepts transferred to seed prediction:
  1. XGBoost with num_parallel_tree=2, lossguide, max_bin=38, low eta
  2. Feature selection (top-25 from v10 analysis)
  3. GLM-inspired team quality feature  
  4. ELO-style feature (already have elo_proxy)
  5. Domain-knowledge post-adjustments (bid type constraints)
  6. Leave-one-season-out validation (same as Kaggle's LOSO)

Selection criterion: LOSO-RMSE ONLY. No test-score snooping.
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)

GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_avail = {}
for s in sorted(set(test_seasons)):
    used = set(train_tourn[train_tourn['Season'].astype(str)==s]['Overall Seed'].astype(int))
    test_avail[s] = sorted(set(range(1, 69)) - used)

n_tr, n_te = len(y_train), len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    if pd.notna(row.get('Bid Type', '')) and str(row['Bid Type']) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])

folds = sorted(set(train_seasons))

# =================================================================
#  FEATURES (68 — same base)
# =================================================================
def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL': feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
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
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float); feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs_grp = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs_grp.mean()).fillna(200); feat['conf_med_net'] = conf.map(cs_grp.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs_grp.min()).fillna(300); feat['conf_std_net'] = conf.map(cs_grp.std()).fillna(50)
    feat['conf_count'] = conf.map(cs_grp.count()).fillna(1)
    power_c = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_c).astype(float)
    cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)
    feat['net_sqrt'] = np.sqrt(net); feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1); feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['elo_proxy'] = 400 - net; feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) + 0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct; feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200; feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l; feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']; feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in all_df[all_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n_val) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in all_df[all_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n_val) + 1)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk')))
        cb.setdefault(key, []).append(float(r['Overall Seed']))
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

feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feature_names = list(feat_train.columns)
n_feat = len(feature_names)
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]; X_te = X_all[n_tr:][tourn_idx]

print(f'{n_tr} train, {n_te} test, {n_feat} features')
POWER = 1.1
SEEDS = [42, 123, 777, 2024, 31415]

def hungarian(scores, seasons, avail, power=POWER):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

# =================================================================
#  TOP-25 FEATURE SELECTION (from v10 analysis)
# =================================================================
# v10 found these are the best 25 by combined Ridge/RF/XGB importance:
# Build combined importance ranking
from sklearn.ensemble import RandomForestRegressor

sc_tmp = StandardScaler()
X_tr_sc = sc_tmp.fit_transform(X_tr)
ridge_full = Ridge(alpha=5.0)
ridge_full.fit(X_tr_sc, y_train)
coef_importance = np.abs(ridge_full.coef_)

rf_imp = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                max_features=0.5, random_state=42, n_jobs=-1)
rf_imp.fit(X_tr, y_train)
rf_importance = rf_imp.feature_importances_

xgb_imp = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                             reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
xgb_imp.fit(X_tr, y_train)
xgb_importance = xgb_imp.feature_importances_

ranks_ridge = np.argsort(np.argsort(-coef_importance))
ranks_rf = np.argsort(np.argsort(-rf_importance))
ranks_xgb = np.argsort(np.argsort(-xgb_importance))
avg_rank = (ranks_ridge + ranks_rf + ranks_xgb) / 3
combined_sorted = np.argsort(avg_rank)
top25 = combined_sorted[:25]

print(f'\nTop-25 features selected:')
for rank, idx in enumerate(top25):
    print(f'  {rank+1:2d}. {feature_names[idx]:25s} (avg_rank={avg_rank[idx]:.1f})')

X_tr_25 = X_tr[:, top25]
X_te_25 = X_te[:, top25]

# =================================================================
#  LOSO helper
# =================================================================
def loso_eval(predict_fn, label='', X_train=X_tr, y=y_train):
    loso_assigned = np.zeros(n_tr, dtype=int)
    fold_rmses = []
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_fn(X_train[tr], y[tr], X_train[te])
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail)
        loso_assigned[te] = assigned
        yte = y[te].astype(int)
        rmse = np.sqrt(np.mean((assigned - yte)**2))
        fold_rmses.append((hold, int(te.sum()), int(np.sum(assigned == yte)), rmse))
    overall_exact = int(np.sum(loso_assigned == y.astype(int)))
    overall_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
    rho, _ = spearmanr(loso_assigned, y.astype(int))
    return overall_rmse, overall_exact, rho, fold_rmses

# =================================================================
#  BASELINE: v40 (all 68 features)
# =================================================================
print('\n' + '='*60)
print(' BASELINE: v40 (XGB+Ridge, 68 features)')
print('='*60)

def pred_v40(Xtr, ytr, Xte):
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
    return 0.70 * xgb_avg + 0.30 * rp

rmse_v40, exact_v40, rho_v40, folds_v40 = loso_eval(pred_v40, 'v40')
print(f'  v40 LOSO: {exact_v40}/{n_tr} exact, RMSE={rmse_v40:.4f}, ρ={rho_v40:.4f}')

# =================================================================
#  A. KAGGLE-STYLE XGB PARAMS (num_parallel_tree, lossguide, max_bin)
# =================================================================
print('\n' + '='*60)
print(' A. Kaggle-style XGB params (num_parallel_tree, lossguide)')
print('='*60)

kaggle_configs = []

# The winning notebook's exact params, adapted
for eta in [0.0093, 0.01, 0.02, 0.03, 0.05]:
    for n_parallel in [1, 2, 3]:
        for max_bin in [32, 38, 64, 128, 256]:
            for max_depth in [4, 5, 6]:
                for num_rounds in [500, 704, 1000, 1500]:
                    kaggle_configs.append({
                        'eta': eta, 'n_parallel': n_parallel, 'max_bin': max_bin,
                        'max_depth': max_depth, 'num_rounds': num_rounds
                    })

# Too many — subsample smartly
# First test the notebook's exact config, then vary one param at a time
print(f'  Testing Kaggle-style configs...')

best_kaggle_rmse = 999
best_kaggle_cfg = None
tested = 0

# Core Kaggle config
kaggle_core = {
    'eta': 0.0093, 'subsample': 0.6, 'colsample_bynode': 0.8,
    'num_parallel_tree': 2, 'min_child_weight': 4, 'max_depth': 4,
    'tree_method': 'hist', 'grow_policy': 'lossguide', 'max_bin': 38,
}

# Sweep configs one-at-a-time from core
sweep_params = {
    'eta': [0.005, 0.0093, 0.01, 0.02, 0.03, 0.05, 0.08],
    'num_parallel_tree': [1, 2, 3, 4],
    'max_bin': [16, 32, 38, 64, 128, 256],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bynode': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [2, 3, 4, 5, 7],
}

num_rounds_sweep = [300, 500, 704, 1000, 1500, 2000]

for feat_set, feat_name, Xtr_use, Xte_use in [
    (top25, 'Top25', X_tr_25, X_te_25),
    (None, 'All68', X_tr, X_te),
]:
    print(f'\n  --- Feature set: {feat_name} ---')
    
    for param_name, param_values in sweep_params.items():
        for pval in param_values:
            for n_rounds in [704]:  # Fix rounds during param sweep
                cfg = dict(kaggle_core)
                cfg[param_name] = pval
                
                def pred_kaggle(Xtr, ytr, Xte, c=cfg, nr=n_rounds):
                    xp = {
                        'objective': 'reg:squarederror',
                        'booster': 'gbtree',
                        'eta': c['eta'],
                        'subsample': c['subsample'],
                        'colsample_bynode': c['colsample_bynode'],
                        'num_parallel_tree': c['num_parallel_tree'],
                        'min_child_weight': c['min_child_weight'],
                        'max_depth': c['max_depth'],
                        'tree_method': 'hist',
                        'grow_policy': c['grow_policy'],
                        'max_bin': c['max_bin'],
                        'verbosity': 0,
                        'seed': 42,
                    }
                    dtrain = xgb.DMatrix(Xtr, label=ytr)
                    dtest = xgb.DMatrix(Xte)
                    model = xgb.train(xp, dtrain, num_boost_round=nr)
                    return model.predict(dtest)
                
                rmse, exact, rho, _ = loso_eval(pred_kaggle, X_train=Xtr_use if feat_set is not None else X_tr)
                tested += 1
                if rmse < best_kaggle_rmse:
                    best_kaggle_rmse = rmse
                    best_kaggle_cfg = (cfg.copy(), n_rounds, feat_name)
                    print(f'    NEW BEST [{feat_name}] RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f} | '
                          f'{param_name}={pval} nr={n_rounds}')
    
    # Also sweep num_rounds for best config so far
    if best_kaggle_cfg:
        best_cfg_copy = best_kaggle_cfg[0].copy()
        for nr in num_rounds_sweep:
            def pred_nr(Xtr, ytr, Xte, c=best_cfg_copy, nrounds=nr):
                xp = {
                    'objective': 'reg:squarederror', 'booster': 'gbtree',
                    'eta': c['eta'], 'subsample': c['subsample'],
                    'colsample_bynode': c['colsample_bynode'],
                    'num_parallel_tree': c['num_parallel_tree'],
                    'min_child_weight': c['min_child_weight'],
                    'max_depth': c['max_depth'], 'tree_method': 'hist',
                    'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
                    'verbosity': 0, 'seed': 42,
                }
                dtrain = xgb.DMatrix(Xtr, label=ytr)
                dtest = xgb.DMatrix(Xte)
                model = xgb.train(xp, dtrain, num_boost_round=nrounds)
                return model.predict(dtest)
            
            rmse, exact, rho, _ = loso_eval(pred_nr, X_train=Xtr_use if feat_set is not None else X_tr)
            tested += 1
            if rmse < best_kaggle_rmse:
                best_kaggle_rmse = rmse
                best_kaggle_cfg = (best_cfg_copy.copy(), nr, feat_name)
                print(f'    NEW BEST [{feat_name}] RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f} | '
                      f'num_rounds={nr}')

print(f'\n  Tested {tested} configs')
print(f'  Best Kaggle-style: RMSE={best_kaggle_rmse:.4f}')
if best_kaggle_cfg:
    print(f'  Config: {best_kaggle_cfg[0]}')
    print(f'  Rounds: {best_kaggle_cfg[1]}, Features: {best_kaggle_cfg[2]}')

# =================================================================
#  B. KAGGLE-XGB + Ridge BLEND (as in original v40)
# =================================================================
print('\n' + '='*60)
print(' B. Kaggle-XGB + Ridge blend')
print('='*60)

best_blend_rmse = 999
best_blend_params = None

if best_kaggle_cfg:
    bk_cfg, bk_nr, bk_feat = best_kaggle_cfg
    Xtr_bk = X_tr_25 if bk_feat == 'Top25' else X_tr
    
    for ridge_alpha in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for ridge_w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            def pred_blend(Xtr, ytr, Xte, c=bk_cfg, nr=bk_nr, ra=ridge_alpha, rw=ridge_w):
                xp = {
                    'objective': 'reg:squarederror', 'booster': 'gbtree',
                    'eta': c['eta'], 'subsample': c['subsample'],
                    'colsample_bynode': c['colsample_bynode'],
                    'num_parallel_tree': c['num_parallel_tree'],
                    'min_child_weight': c['min_child_weight'],
                    'max_depth': c['max_depth'], 'tree_method': 'hist',
                    'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
                    'verbosity': 0, 'seed': 42,
                }
                dtrain = xgb.DMatrix(Xtr, label=ytr)
                dtest = xgb.DMatrix(Xte)
                model = xgb.train(xp, dtrain, num_boost_round=nr)
                xgb_p = model.predict(dtest)
                sc = StandardScaler(); rm = Ridge(alpha=ra)
                rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
                return (1 - rw) * xgb_p + rw * rp
            
            rmse, exact, rho, _ = loso_eval(pred_blend, X_train=Xtr_bk)
            if rmse < best_blend_rmse:
                best_blend_rmse = rmse
                best_blend_params = (ridge_alpha, ridge_w)
                print(f'    NEW BEST: RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f} | '
                      f'ra={ridge_alpha} rw={ridge_w}')

    print(f'\n  Best Kaggle+Ridge: RMSE={best_blend_rmse:.4f}')
    if best_blend_params:
        print(f'  Ridge: α={best_blend_params[0]}, w={best_blend_params[1]}')

# =================================================================
#  C. Multi-seed Kaggle-XGB (averaging like the notebook does)
# =================================================================
print('\n' + '='*60)
print(' C. Multi-seed Kaggle-XGB (LOSO model averaging)')
print('='*60)

best_ms_rmse = 999
best_ms_params = None

if best_kaggle_cfg:
    bk_cfg, bk_nr, bk_feat = best_kaggle_cfg
    Xtr_ms = X_tr_25 if bk_feat == 'Top25' else X_tr
    
    seed_sets = [
        [42],
        [42, 123],
        [42, 123, 777],
        [42, 123, 777, 2024, 31415],
    ]
    
    for seeds_use in seed_sets:
        for ra in [5.0, 10.0]:
            for rw in [0.0, 0.20, 0.30]:
                def pred_ms(Xtr, ytr, Xte, c=bk_cfg, nr=bk_nr, ss=seeds_use, r_a=ra, r_w=rw):
                    preds = []
                    for s in ss:
                        xp = {
                            'objective': 'reg:squarederror', 'booster': 'gbtree',
                            'eta': c['eta'], 'subsample': c['subsample'],
                            'colsample_bynode': c['colsample_bynode'],
                            'num_parallel_tree': c['num_parallel_tree'],
                            'min_child_weight': c['min_child_weight'],
                            'max_depth': c['max_depth'], 'tree_method': 'hist',
                            'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
                            'verbosity': 0, 'seed': s,
                        }
                        dtrain = xgb.DMatrix(Xtr, label=ytr)
                        dtest = xgb.DMatrix(Xte)
                        model = xgb.train(xp, dtrain, num_boost_round=nr)
                        preds.append(model.predict(dtest))
                    xgb_p = np.mean(preds, axis=0)
                    if r_w > 0:
                        sc = StandardScaler(); rm = Ridge(alpha=r_a)
                        rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
                        return (1 - r_w) * xgb_p + r_w * rp
                    return xgb_p
                
                rmse, exact, rho, _ = loso_eval(pred_ms, X_train=Xtr_ms)
                if rmse < best_ms_rmse:
                    best_ms_rmse = rmse
                    best_ms_params = (seeds_use, ra, rw)
                    print(f'    NEW BEST: RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f} | '
                          f'{len(seeds_use)} seeds, ra={ra}, rw={rw}')

    print(f'\n  Best multi-seed: RMSE={best_ms_rmse:.4f}')

# =================================================================
#  D. Domain post-adjustment (like notebook's manual overrides)
# =================================================================
print('\n' + '='*60)
print(' D. Domain post-adjustments')
print(' (Kaggle notebook boosted confidence for certain predictions)')
print('='*60)

# Our equivalent: after getting raw predictions, apply domain knowledge:
# - AQ (auto-qualifier) teams from weak conferences tend to get seeds 13-16
# - AL (at-large) teams from strong conferences tend to get seeds 1-12
# How: shift predictions towards expected ranges based on bid type + conference

def pred_v40_post(Xtr, ytr, Xte, shift_aq=2.0, shift_al=-1.0):
    """v40 prediction with domain post-adjustment."""
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
    raw = 0.70 * xgb_avg + 0.30 * rp
    return raw  # Post-adjustment applied externally where we have bid info

# Test post-adjustment in LOSO
# Build bid type arrays for train
train_bids = train_tourn['Bid Type'].fillna('').values
train_confs = train_tourn['Conference'].fillna('Unknown').values
power_confs = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}

best_post_rmse = 999
best_post_params = None

for aq_shift in [0.0, 1.0, 2.0, 3.0, 5.0]:
    for al_shift in [0.0, -0.5, -1.0, -2.0]:
        loso_assigned = np.zeros(n_tr, dtype=int)
        for hold in folds:
            tr = train_seasons != hold
            te = train_seasons == hold
            pred = pred_v40(X_tr[tr], y_train[tr], X_tr[te])
            
            # Apply domain shifts
            te_bids = train_bids[te]
            te_confs = train_confs[te]
            adjusted = pred.copy()
            for i in range(len(adjusted)):
                if te_bids[i] == 'AQ' and te_confs[i] not in power_confs:
                    adjusted[i] += aq_shift  # Push AQ mid-majors toward higher seeds
                elif te_bids[i] == 'AL':
                    adjusted[i] += al_shift  # Push AL teams toward lower seeds
            
            avail = {hold: list(range(1, 69))}
            assigned = hungarian(adjusted, train_seasons[te], avail)
            loso_assigned[te] = assigned
        
        rmse = np.sqrt(np.mean((loso_assigned - y_train.astype(int))**2))
        exact = int(np.sum(loso_assigned == y_train.astype(int)))
        if rmse < best_post_rmse:
            best_post_rmse = rmse
            best_post_params = (aq_shift, al_shift)
            print(f'    NEW BEST: RMSE={rmse:.4f} ({exact}/{n_tr}) | '
                  f'aq_shift={aq_shift} al_shift={al_shift}')

print(f'\n  Best post-adjusted: RMSE={best_post_rmse:.4f}')

# =================================================================
#  FINAL COMPARISON (LOSO-only)
# =================================================================
print('\n' + '='*60)
print(' FINAL COMPARISON — All selected by LOSO')
print('='*60)

results = {
    'v40 baseline (68f)': rmse_v40,
    'Kaggle-style XGB': best_kaggle_rmse,
    'Kaggle+Ridge blend': best_blend_rmse,
    'Multi-seed Kaggle': best_ms_rmse,
    'Domain post-adjust': best_post_rmse,
}

# Add the v10 best (Top-25 XGB+Ridge)
def pred_top25(Xtr, ytr, Xte):
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
    return 0.70 * xgb_avg + 0.30 * rp

rmse_25, exact_25, rho_25, _ = loso_eval(pred_top25, X_train=X_tr_25)
results['Top-25 feat (v10)'] = rmse_25

print(f'\n  {"Model":<25} {"LOSO-RMSE":>10}')
print(f'  {"-"*25} {"-"*10}')
best_model = min(results, key=results.get)
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    marker = ' <-- BEST' if name == best_model else ''
    print(f'  {name:<25} {rmse:>10.4f}{marker}')

# =================================================================
#  SINGLE TEST EVALUATION (LOSO-best only)
# =================================================================
print('\n' + '='*60)
print(' TEST EVALUATION (LOSO-best only — no snooping)')
print('='*60)

best_rmse_overall = results[best_model]

# Build the final predictor
if best_model == 'v40 baseline (68f)':
    pred_final_fn = pred_v40
    Xtr_final, Xte_final = X_tr, X_te
elif best_model == 'Top-25 feat (v10)':
    pred_final_fn = pred_top25
    Xtr_final, Xte_final = X_tr_25, X_te_25
elif best_model == 'Kaggle-style XGB':
    bk_cfg, bk_nr, bk_feat = best_kaggle_cfg
    Xtr_final = X_tr_25 if bk_feat == 'Top25' else X_tr
    Xte_final = X_te_25 if bk_feat == 'Top25' else X_te
    def pred_final_fn(Xtr, ytr, Xte, c=bk_cfg, nr=bk_nr):
        xp = {
            'objective': 'reg:squarederror', 'booster': 'gbtree',
            'eta': c['eta'], 'subsample': c['subsample'],
            'colsample_bynode': c['colsample_bynode'],
            'num_parallel_tree': c['num_parallel_tree'],
            'min_child_weight': c['min_child_weight'],
            'max_depth': c['max_depth'], 'tree_method': 'hist',
            'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
            'verbosity': 0, 'seed': 42,
        }
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dtest = xgb.DMatrix(Xte)
        model = xgb.train(xp, dtrain, num_boost_round=nr)
        return model.predict(dtest)
elif best_model == 'Kaggle+Ridge blend':
    bk_cfg, bk_nr, bk_feat = best_kaggle_cfg
    ra_best, rw_best = best_blend_params
    Xtr_final = X_tr_25 if bk_feat == 'Top25' else X_tr
    Xte_final = X_te_25 if bk_feat == 'Top25' else X_te
    def pred_final_fn(Xtr, ytr, Xte, c=bk_cfg, nr=bk_nr, ra=ra_best, rw=rw_best):
        xp = {
            'objective': 'reg:squarederror', 'booster': 'gbtree',
            'eta': c['eta'], 'subsample': c['subsample'],
            'colsample_bynode': c['colsample_bynode'],
            'num_parallel_tree': c['num_parallel_tree'],
            'min_child_weight': c['min_child_weight'],
            'max_depth': c['max_depth'], 'tree_method': 'hist',
            'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
            'verbosity': 0, 'seed': 42,
        }
        dtrain = xgb.DMatrix(Xtr, label=ytr)
        dtest = xgb.DMatrix(Xte)
        model = xgb.train(xp, dtrain, num_boost_round=nr)
        xgb_p = model.predict(dtest)
        sc = StandardScaler(); rm = Ridge(alpha=ra)
        rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
        return (1 - rw) * xgb_p + rw * rp
elif best_model == 'Multi-seed Kaggle':
    bk_cfg, bk_nr, bk_feat = best_kaggle_cfg
    seeds_best, ra_ms, rw_ms = best_ms_params
    Xtr_final = X_tr_25 if bk_feat == 'Top25' else X_tr
    Xte_final = X_te_25 if bk_feat == 'Top25' else X_te
    def pred_final_fn(Xtr, ytr, Xte, c=bk_cfg, nr=bk_nr, ss=seeds_best, r_a=ra_ms, r_w=rw_ms):
        preds = []
        for s in ss:
            xp = {
                'objective': 'reg:squarederror', 'booster': 'gbtree',
                'eta': c['eta'], 'subsample': c['subsample'],
                'colsample_bynode': c['colsample_bynode'],
                'num_parallel_tree': c['num_parallel_tree'],
                'min_child_weight': c['min_child_weight'],
                'max_depth': c['max_depth'], 'tree_method': 'hist',
                'grow_policy': c['grow_policy'], 'max_bin': c['max_bin'],
                'verbosity': 0, 'seed': s,
            }
            dtrain = xgb.DMatrix(Xtr, label=ytr)
            dtest = xgb.DMatrix(Xte)
            model = xgb.train(xp, dtrain, num_boost_round=nr)
            preds.append(model.predict(dtest))
        xgb_p = np.mean(preds, axis=0)
        if r_w > 0:
            sc = StandardScaler(); rm = Ridge(alpha=r_a)
            rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
            return (1 - r_w) * xgb_p + r_w * rp
        return xgb_p
elif best_model == 'Domain post-adjust':
    pred_final_fn = pred_v40
    Xtr_final, Xte_final = X_tr, X_te
    # Will apply domain adjustment after prediction

# Final test
pred_final = pred_final_fn(Xtr_final, y_train, Xte_final)

# Apply domain adjustment if that was best
if best_model == 'Domain post-adjust' and best_post_params:
    aq_s, al_s = best_post_params
    test_bids = test_df.iloc[tourn_idx]['Bid Type'].fillna('').values
    test_confs = test_df.iloc[tourn_idx]['Conference'].fillna('Unknown').values
    for i in range(len(pred_final)):
        if test_bids[i] == 'AQ' and test_confs[i] not in power_confs:
            pred_final[i] += aq_s
        elif test_bids[i] == 'AL':
            pred_final[i] += al_s

assigned = hungarian(pred_final, test_seasons, test_avail)
test_exact = int(np.sum(assigned == test_gt))
test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))
test_rho, _ = spearmanr(assigned, test_gt)

# Also v40 for comparison
p40 = pred_v40(X_tr, y_train, X_te)
a40 = hungarian(p40, test_seasons, test_avail)
e40 = int(np.sum(a40 == test_gt))
r40 = np.sqrt(np.mean((a40 - test_gt)**2))
rho40, _ = spearmanr(a40, test_gt)

print(f'\n  LOSO-best ({best_model}):')
print(f'    Test: {test_exact}/91, RMSE={test_rmse:.4f}, ρ={test_rho:.4f}')
print(f'    LOSO: {best_rmse_overall:.4f}')
print(f'    Gap:  {abs(best_rmse_overall - test_rmse):.4f}')
print(f'\n  v40 baseline:')
print(f'    Test: {e40}/91, RMSE={r40:.4f}, ρ={rho40:.4f}')
print(f'    LOSO: {rmse_v40:.4f}')

# Only overwrite if LOSO-best beats v40
if best_rmse_overall < rmse_v40:
    print(f'\n  *** LOSO improvement: {best_rmse_overall:.4f} vs {rmse_v40:.4f} ***')
    sub_out = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub_out['RecordID'] == rid
        if mask.any():
            sub_out.loc[mask, 'Overall Seed'] = int(assigned[i])
    sub_out.to_csv(os.path.join(DATA_DIR, 'final_submission.csv'), index=False)
    print('  Saved: final_submission.csv')
else:
    print(f'\n  v40 still best by LOSO. Not saving.')

print(f'\n  Total time: {time.time()-t0:.0f}s')
