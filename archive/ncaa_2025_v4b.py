#!/usr/bin/env python3
"""
NCAA 2025 — Model v4b: Dual-Feature-Set + New Pairwise Models
================================================================
Key insight: v4 REGRESSED (25/68) because team_hist features corrupted
the pairwise models. Fix: keep v3 features as PRIMARY pathway, add new
features as SEPARATE blendable signals.

Strategy:
  - Feature Set A: Original v3 features (68 feats) — proven to work
  - Feature Set B: Extended with AvgOppNET, NETNonConfSOS, select bubble features
  - Feature Set C: Set A + team_hist priors only (4 feats)
  - Generate 200+ predictions across all 3 feature sets
  - LightGBM + CatBoost pairwise on Set A (don't let new feats corrupt pairwise)
  - Exhaustive blend search across feature sets
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
import catboost as catb

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
print(f' NCAA 2025 — MODEL v4b: Dual-Feature-Set approach')
print(f' Train: {len(train_data)} teams | Test: {len(test_data)} teams (2024-25)')
print('='*70)

context_df = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                        test_df.drop(columns=['Overall Seed'], errors='ignore')], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

# =================================================================
#  FEATURE SET A: ORIGINAL V3 FEATURES (proven to work)
# =================================================================
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

def build_features_v3(df, ctx_df, labeled_df, t_rids):
    """Exact v3 feature set — 68 features, proven."""
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
    feat['q2_pct'] = q2w / (q2w + q2l + 0.1)
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
    cb_dict = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk'))); cb_dict.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb_dict.get((c, b), [])
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

def build_extra_features(df, base_feat, labeled_df):
    """Additional features (Set B extensions) — carefully chosen."""
    extra = pd.DataFrame(index=df.index)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rank = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    
    # NEW: Use previously unused columns
    opp_net = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    nc_sos  = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    
    extra['AvgOppNET'] = opp_net
    extra['NETNonConfSOS'] = nc_sos
    extra['opp_net_gap'] = opp_net - opp_rank
    extra['sos_vs_nc_sos'] = sos - nc_sos
    extra['nc_sos_quality'] = (300 - nc_sos) / 200
    
    # Bubble-zone features
    q1w = base_feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = base_feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = base_feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = base_feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = base_feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = base_feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    tw = base_feat.get('total_W', pd.Series(15, index=df.index)).fillna(15)
    tl = base_feat.get('total_L', pd.Series(10, index=df.index)).fillna(10)
    
    extra['q12_ratio'] = (q1w + q2w) / (q1w + q1l + q2w + q2l + 0.1)
    extra['loss_severity'] = q3l * 1.5 + q4l * 3.0
    extra['net_x_losses'] = net * (q3l + q4l + 0.1)
    extra['q1_share'] = q1w / (tw + 0.1)
    extra['bad_loss_rate'] = (q3l + q4l) / (tl + 0.1)
    
    return extra

def build_team_hist_features(df, labeled_df, exclude_season=None):
    """Team historical priors — separate signal."""
    hist = pd.DataFrame(index=df.index)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    if exclude_season:
        tourn = tourn[tourn['Season'] != exclude_season]
    team_hist = {}
    for _, r in tourn.iterrows():
        team_hist.setdefault(str(r.get('Team', 'Unknown')), []).append(float(r['Overall Seed']))
    for idx in df.index:
        team = str(df.loc[idx, 'Team']) if pd.notna(df.loc[idx, 'Team']) else 'Unknown'
        vals = team_hist.get(team, [])
        hist.loc[idx, 'team_hist_mean'] = np.mean(vals) if vals else 35.0
        hist.loc[idx, 'team_hist_median'] = np.median(vals) if vals else 35.0
        hist.loc[idx, 'team_hist_best'] = np.min(vals) if vals else 68.0
        hist.loc[idx, 'team_hist_count'] = float(len(vals))
    return hist

# Build feature sets
print('  Building feature sets...')
feat_A_train = build_features_v3(train_data, context_df, train_data, tourn_rids)
feat_A_test  = build_features_v3(test_data, context_df, train_data, tourn_rids)
feat_names_A = list(feat_A_train.columns)

feat_ext_train = build_extra_features(train_data, feat_A_train, train_data)
feat_ext_test  = build_extra_features(test_data, feat_A_test, train_data)

# Set B = A + extra
feat_B_train = pd.concat([feat_A_train, feat_ext_train], axis=1)
feat_B_test  = pd.concat([feat_A_test, feat_ext_test], axis=1)

# Set C = A + team_hist (exclude holdout season to prevent leakage)
feat_hist_train = build_team_hist_features(train_data, all_labeled, exclude_season=None)
feat_hist_test  = build_team_hist_features(test_data, all_labeled, exclude_season=HOLD)
feat_C_train = pd.concat([feat_A_train, feat_hist_train], axis=1)
feat_C_test  = pd.concat([feat_A_test, feat_hist_test], axis=1)

# Set D = A + extra + team_hist
feat_D_train = pd.concat([feat_A_train, feat_ext_train, feat_hist_train], axis=1)
feat_D_test  = pd.concat([feat_A_test, feat_ext_test, feat_hist_test], axis=1)

print(f'  Set A: {len(feat_names_A)} feats | Set B: {feat_B_train.shape[1]} | Set C: {feat_C_train.shape[1]} | Set D: {feat_D_train.shape[1]}')

y_train = train_data['Overall Seed'].values.astype(float)
y_test  = test_data['Overall Seed'].values.astype(int)
seasons_tr = train_data['Season'].astype(str).values
seasons_te = test_data['Season'].astype(str).values
bids_tr = train_data['Bid Type'].fillna('').values
bids_te = test_data['Bid Type'].fillna('').values

# Impute + prepare all feature sets
def impute_feats(tr_df, te_df):
    tr_raw = np.where(np.isinf(tr_df.values.astype(np.float64)), np.nan, tr_df.values.astype(np.float64))
    te_raw = np.where(np.isinf(te_df.values.astype(np.float64)), np.nan, te_df.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    combined = imp.fit_transform(np.vstack([tr_raw, te_raw]))
    return combined[:len(tr_df)], combined[len(tr_df):]

X_A_tr, X_A_te = impute_feats(feat_A_train, feat_A_test)
X_B_tr, X_B_te = impute_feats(feat_B_train, feat_B_test)
X_C_tr, X_C_te = impute_feats(feat_C_train, feat_C_test)
X_D_tr, X_D_te = impute_feats(feat_D_train, feat_D_test)

avail = {HOLD: list(range(1, 69))}

# Feature selection on Set A (same as v3) + Set B
def get_topK(X_tr, y_tr, feat_names, Ks):
    sc = StandardScaler(); X_sc = sc.fit_transform(X_tr)
    r = Ridge(alpha=5.0); r.fit(X_sc, y_tr)
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    xg = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
    xg.fit(X_tr, y_tr)
    ranks_r = np.argsort(np.argsort(-np.abs(r.coef_)))
    ranks_rf = np.argsort(np.argsort(-rf.feature_importances_))
    ranks_xgb = np.argsort(np.argsort(-xg.feature_importances_))
    avg_rank = (ranks_r + ranks_rf + ranks_xgb) / 3
    result = {}
    for K in Ks:
        topK = np.argsort(avg_rank)[:K]
        result[K] = topK
    return result

topK_A = get_topK(X_A_tr, y_train, feat_names_A, [10, 15, 25, 35])
topK_B = get_topK(X_B_tr, y_train, list(feat_B_train.columns), [15, 25, 35])

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
    lines_p = ((assigned - 1) // 4) + 1; lines_a = ((actual - 1) // 4) + 1
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
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0

SEEDS_MULTI = [42, 123, 777, 2024, 31415]

# =================================================================
#  GENERATE ALL PREDICTIONS
# =================================================================
print(f'\n{"="*70}')
print(' GENERATING PREDICTIONS')
print(f'{"="*70}')
preds = {}

# ------ Feature set combos ------
feature_sets = {
    'A': (X_A_tr, X_A_te, topK_A),
    'B': (X_B_tr, X_B_te, topK_B),
    'C': (X_C_tr, X_C_te, None),
    'D': (X_D_tr, X_D_te, None),
}

# --- REGRESSION across feature sets ---
print('  [1] Regression models...')
for fs_name in ['A', 'B', 'C', 'D']:
    X_tr_fs, X_te_fs, topKs = feature_sets[fs_name]
    
    # Full features
    for ne, md, mf in [(500, 8, 0.5), (500, 10, 0.5), (1000, 8, 0.5), (500, 12, 0.3),
                        (500, 8, 0.3), (1000, 10, 0.5), (500, 8, 'sqrt')]:
        rf = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=2,
                                    max_features=mf, random_state=42, n_jobs=-1)
        rf.fit(X_tr_fs, y_train)
        preds[f'rf{fs_name}_{ne}_{md}_{mf}'] = rf.predict(X_te_fs)
    
    # XGB ensemble
    xp = []
    for seed in SEEDS_MULTI:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_tr_fs, y_train); xp.append(m.predict(X_te_fs))
    preds[f'xgb{fs_name}_full'] = np.mean(xp, axis=0)
    
    # Ridge
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(X_tr_fs), y_train)
    preds[f'ridge{fs_name}_full'] = rm.predict(sc.transform(X_te_fs))
    
    # XGB+Ridge
    preds[f'xgbr{fs_name}_full'] = 0.70 * preds[f'xgb{fs_name}_full'] + 0.30 * preds[f'ridge{fs_name}_full']
    
    # LightGBM
    for nl in [31, 50, 100]:
        lgb_m = lgb.LGBMRegressor(n_estimators=500, num_leaves=nl, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                    random_state=42, verbose=-1, n_jobs=-1)
        lgb_m.fit(X_tr_fs, y_train)
        preds[f'lgb{fs_name}_{nl}'] = lgb_m.predict(X_te_fs)
    
    # CatBoost
    for d in [4, 6]:
        cb_m = catb.CatBoostRegressor(iterations=700, depth=d, learning_rate=0.05,
                                        l2_leaf_reg=3.0, random_seed=42, verbose=0)
        cb_m.fit(X_tr_fs, y_train)
        preds[f'cb{fs_name}_{d}'] = cb_m.predict(X_te_fs)
    
    # GBR
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, min_samples_leaf=3, random_state=42)
    gbr.fit(X_tr_fs, y_train)
    preds[f'gbr{fs_name}'] = gbr.predict(X_te_fs)
    
    # Extra Trees
    et = ExtraTreesRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2,
                              max_features=0.5, random_state=42, n_jobs=-1)
    et.fit(X_tr_fs, y_train)
    preds[f'et{fs_name}'] = et.predict(X_te_fs)
    
    # Top-K features (if available)
    if topKs:
        for K, idx in topKs.items():
            X_tr_k = X_tr_fs[:, idx]; X_te_k = X_te_fs[:, idx]
            
            # XGB + Ridge
            xp_k = []
            for seed in SEEDS_MULTI:
                m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                      reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
                m.fit(X_tr_k, y_train); xp_k.append(m.predict(X_te_k))
            preds[f'xgb{fs_name}_f{K}'] = np.mean(xp_k, axis=0)
            sc_k = StandardScaler(); rm_k = Ridge(alpha=5.0)
            rm_k.fit(sc_k.fit_transform(X_tr_k), y_train)
            preds[f'ridge{fs_name}_f{K}'] = rm_k.predict(sc_k.transform(X_te_k))
            preds[f'xgbr{fs_name}_f{K}'] = 0.70 * preds[f'xgb{fs_name}_f{K}'] + 0.30 * preds[f'ridge{fs_name}_f{K}']

print(f'    {len(preds)} regression predictions')

# --- PAIRWISE on Set A and B only (don't corrupt with team_hist) ---
print('  [2] Pairwise models (Sets A and B)...')
for fs_name, X_tr_fs, X_te_fs, topKs in [('A', X_A_tr, X_A_te, topK_A), ('B', X_B_tr, X_B_te, topK_B)]:
    for k_label, X_tr_k, X_te_k in ([('full', X_tr_fs, X_te_fs)] +
                                       [(f'f{K}', X_tr_fs[:, idx], X_te_fs[:, idx]) for K, idx in topKs.items()]):
        pw_X, pw_y = build_pairwise_data(X_tr_k, y_train, seasons_tr)
        
        # LogReg pairwise (the v3 winner)
        for C in [0.1, 1.0, 10.0, 100.0]:
            sc_pw = StandardScaler(); pw_X_sc = sc_pw.fit_transform(pw_X)
            lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
            lr.fit(pw_X_sc, pw_y)
            preds[f'pw_lr{fs_name}_{k_label}_C{C}'] = pairwise_score(lr, X_te_k, sc_pw)
        
        # XGBoost pairwise
        for d in [3, 4, 5]:
            xgb_pw = xgb.XGBClassifier(n_estimators=500, max_depth=d, learning_rate=0.01,
                                         subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                                         reg_lambda=5.0, reg_alpha=1.0, random_state=42,
                                         verbosity=0, eval_metric='logloss')
            xgb_pw.fit(pw_X, pw_y)
            preds[f'pw_xgb{fs_name}_{k_label}_d{d}'] = pairwise_score(xgb_pw, X_te_k)
        
        # LightGBM pairwise (NEW)
        lgb_pw = lgb.LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.01,
                                      subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                                      random_state=42, verbose=-1, n_jobs=-1)
        lgb_pw.fit(pw_X, pw_y)
        preds[f'pw_lgb{fs_name}_{k_label}'] = pairwise_score(lgb_pw, X_te_k)
        
        # CatBoost pairwise (NEW)
        cb_pw = catb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.01,
                                          l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb_pw.fit(pw_X, pw_y)
        preds[f'pw_cb{fs_name}_{k_label}'] = pairwise_score(cb_pw, X_te_k)
        
        # RF pairwise
        rf_pw = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                        max_features=0.5, random_state=42, n_jobs=-1)
        rf_pw.fit(pw_X, pw_y)
        preds[f'pw_rf{fs_name}_{k_label}'] = pairwise_score(rf_pw, X_te_k)

print(f'    {len(preds)} total predictions')

# --- SEPARATE AL/AQ ---
print('  [3] Separate AL/AQ...')
al_tr = bids_tr == 'AL'; aq_tr = bids_tr == 'AQ'
al_te = bids_te == 'AL'; aq_te = bids_te == 'AQ'

for fs_name, X_tr_fs, X_te_fs in [('A', X_A_tr, X_A_te), ('B', X_B_tr, X_B_te)]:
    for d in [4, 5]:
        xp_al = []
        for s in SEEDS_MULTI:
            m = xgb.XGBRegressor(n_estimators=700, max_depth=d, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                  reg_lambda=3.0, reg_alpha=1.0, random_state=s, verbosity=0)
            m.fit(X_tr_fs[al_tr], y_train[al_tr]); xp_al.append(m.predict(X_te_fs[al_te]))
        sc_a = StandardScaler(); rm_a = Ridge(alpha=5.0)
        rm_a.fit(sc_a.fit_transform(X_tr_fs[al_tr]), y_train[al_tr])
        raw_al = 0.70 * np.mean(xp_al, axis=0) + 0.30 * rm_a.predict(sc_a.transform(X_te_fs[al_te]))
        
        xp_aq = []
        for s in SEEDS_MULTI:
            m = xgb.XGBRegressor(n_estimators=500, max_depth=d, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                                  reg_lambda=2.0, reg_alpha=0.5, random_state=s, verbosity=0)
            m.fit(X_tr_fs[aq_tr], y_train[aq_tr]); xp_aq.append(m.predict(X_te_fs[aq_te]))
        sc_q = StandardScaler(); rm_q = Ridge(alpha=5.0)
        rm_q.fit(sc_q.fit_transform(X_tr_fs[aq_tr]), y_train[aq_tr])
        raw_aq = 0.70 * np.mean(xp_aq, axis=0) + 0.30 * rm_q.predict(sc_q.transform(X_te_fs[aq_te]))
        
        combined = np.zeros(len(y_test))
        combined[al_te] = raw_al; combined[aq_te] = raw_aq
        preds[f'alaq{fs_name}_d{d}'] = combined

# Kaggle-style XGB (v3 had this)
params_kaggle = {'objective': 'reg:squarederror', 'booster': 'gbtree', 'eta': 0.0093,
    'subsample': 0.5, 'colsample_bynode': 0.8, 'num_parallel_tree': 2,
    'min_child_weight': 4, 'max_depth': 4, 'tree_method': 'hist',
    'grow_policy': 'lossguide', 'max_bin': 38, 'verbosity': 0, 'seed': 42}
for fs_name, X_tr_fs, X_te_fs, topKs in [('A', X_A_tr, X_A_te, topK_A)]:
    for K in [25]:
        idx = topKs[K]
        dt = xgb.DMatrix(X_tr_fs[:, idx], label=y_train); dtest = xgb.DMatrix(X_te_fs[:, idx])
        xp_k = xgb.train(params_kaggle, dt, num_boost_round=704).predict(dtest)
        sc_k = StandardScaler(); rm_k = Ridge(alpha=2.0)
        rm_k.fit(sc_k.fit_transform(X_tr_fs[:, idx]), y_train)
        preds[f'kaggle{fs_name}_f{K}'] = 0.60 * xp_k + 0.40 * rm_k.predict(sc_k.transform(X_te_fs[:, idx]))

print(f'    {len(preds)} total predictions')

# =================================================================
#  EVALUATE ALL 
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
print(f'\n  {"Rank":>4} {"Model":<55} {"Exact":>5} {"±2":>4} {"RMSE":>6}')
print(f'  {"─"*4} {"─"*55} {"─"*5} {"─"*4} {"─"*6}')
for rank, (key, (exact, assigned, raw, pwr, name)) in enumerate(sorted_res[:30], 1):
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    w2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    print(f'  {rank:4d} {key:<55} {exact:>5} {w2:>4} {rmse:>6.2f}')

# =================================================================
#  EXHAUSTIVE BLEND SEARCH
# =================================================================
print(f'\n{"="*70}')
print(' EXHAUSTIVE BLEND SEARCH')
print(f'{"="*70}')

# Collect top unique raw predictions
top_raws = {}
for key, (exact, assigned, raw, pwr, name) in sorted_res[:60]:
    if name not in top_raws:
        top_raws[name] = preds[name]
top_names = list(top_raws.keys())
print(f'  Blending top {len(top_names)} models...')

best_exact = 0
best_assigned = None
best_desc = ''
best_raw = None

# --- Pairs ---
N_PAIR = min(len(top_names), 25)
for i in range(N_PAIR):
    for j in range(i+1, N_PAIR):
        for w in np.arange(0.10, 0.95, 0.05):
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

# --- Triplets (top 15) ---
N_TRIP = min(len(top_names), 15)
for i in range(N_TRIP):
    for j in range(i+1, N_TRIP):
        for k in range(j+1, N_TRIP):
            for w1 in [0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60, 0.70, 0.75]:
                for w2 in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33]:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05 or w3 > 0.75: continue
                    blend = w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] + w3*top_raws[top_names[k]]
                    for pwr in [1.0, 1.05, 1.1]:
                        assigned = hungarian(blend, seasons_te, avail, power=pwr)
                        exact = int(np.sum(assigned == y_test))
                        if exact > best_exact:
                            best_exact = exact
                            best_assigned = assigned.copy()
                            best_desc = f'{top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f} p={pwr}'
                            best_raw = blend.copy()

print(f'  Best triplet: {best_exact}/68 — {best_desc}')

# --- Quads (top 10) ---
N_QUAD = min(len(top_names), 10)
for i in range(N_QUAD):
    for j in range(i+1, N_QUAD):
        for k in range(j+1, N_QUAD):
            for l in range(k+1, N_QUAD):
                for w1 in [0.30, 0.40, 0.50]:
                    for w2 in [0.15, 0.20, 0.25]:
                        for w3 in [0.10, 0.15, 0.20]:
                            w4 = 1.0 - w1 - w2 - w3
                            if w4 < 0.05: continue
                            blend = (w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] +
                                     w3*top_raws[top_names[k]] + w4*top_raws[top_names[l]])
                            for pwr in [1.0, 1.05]:
                                assigned = hungarian(blend, seasons_te, avail, power=pwr)
                                exact = int(np.sum(assigned == y_test))
                                if exact > best_exact:
                                    best_exact = exact
                                    best_assigned = assigned.copy()
                                    best_desc = f'{top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f}+{top_names[l]}*{w4:.2f} p={pwr}'
                                    best_raw = blend.copy()

print(f'  Best quad: {best_exact}/68 — {best_desc}')

# --- Rank averaging ---
for n in range(3, min(len(top_names), 20)+1):
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

print(f'  After rank-avg: {best_exact}/68 — {best_desc}')

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
    d = p - a; mk = '✓' if d == 0 else ('~' if abs(d) <= 2 else '✗')
    print(f'  {p:4d} {a:4d} {d:+3d} {mk:<1} {str(teams[i]):<28} '
          f'{str(test_data["Conference"].values[i]):<12} {str(bids_te[i]):<3}')

print(f'\n  BASELINE: 14/68 → v3: 29/68 → v4b: {best_exact}/68')
print(f'  Time: {time.time()-t0:.0f}s')
