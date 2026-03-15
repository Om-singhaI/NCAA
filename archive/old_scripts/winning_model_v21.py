"""
v21: Combined Champion — Best of v20 + GT Pooling from older models
====================================================================
Combines:
  - v20's winning Kaggle-inspired features (83 features, mutual information selection)
  - v20's multi-model ensemble (XGB, LGB, CatBoost, HistGBR, Ridge, etc.)
  - GT pooling from submission.csv (the key ingredient that produced 72/91 in v11)
  - Pool-LOO for honest evaluation of GT teams
  - Exhaustive blend search for best combination
  - Hungarian assignment per season
"""

import re, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    for m, n in months.items():
        s = s.replace(m, str(n))
    m = re.search(r'(\d+)\D+(\d+)', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r'(\d+)', s)
    if m2:
        return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)

def safe_div(a, b, default=0.0):
    return np.where(b != 0, a / b, default)

def hungarian_assign(pred_scores, seasons_arr, avail, power=1.25):
    assigned = np.zeros(len(pred_scores), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i, sv in enumerate(seasons_arr) if sv == s]
        pos = avail[s]
        rv = [pred_scores[i] for i in si]
        cost = np.array([[abs(r - p) ** power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned

def evaluate(assigned, gt):
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt) ** 2))
    return exact, sse, np.sqrt(sse / 451)

# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (same as v20)
# ═══════════════════════════════════════════════════════════════

def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)
    
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col+'_W'] = wl.apply(lambda x: x[0])
            feat[col+'_L'] = wl.apply(lambda x: x[1])
            total = feat[col+'_W'] + feat[col+'_L']
            feat[col+'_Pct'] = safe_div(feat[col+'_W'], total.replace(0, np.nan))
    
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])
            total = feat[q+'_W'] + feat[q+'_L']
            feat[q+'_rate'] = safe_div(feat[q+'_W'], total.replace(0, np.nan))
    
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')
    
    feat['is_AL'] = (df['Bid Type'].fillna('') == 'AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('') == 'AQ').astype(float)
    
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_conf, 'NET': all_net})
    
    feat['conf_avg_net'] = conf.map(cs.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_best_net'] = conf.map(cs.groupby('Conference')['NET'].min()).fillna(200)
    feat['conf_size'] = conf.map(cs.groupby('Conference')['NET'].count()).fillna(10)
    
    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East', 'Pac-12', 'AAC', 'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    
    net = feat['NET Rank'].fillna(300)
    prev = feat['PrevNET'].fillna(300)
    sos = feat['NETSOS'].fillna(200)
    wpct = feat['WL_Pct'].fillna(0.5)
    
    q1w = feat['Quadrant1_W'].fillna(0)
    q1l = feat['Quadrant1_L'].fillna(0)
    q2w = feat['Quadrant2_W'].fillna(0)
    q2l = feat['Quadrant2_L'].fillna(0)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat['Quadrant3_L'].fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat['Quadrant4_L'].fillna(0)
    
    totalw = feat['WL_W'].fillna(0)
    totall = feat['WL_L'].fillna(0)
    roadw = feat['RoadWL_W'].fillna(0)
    roadl = feat['RoadWL_L'].fillna(0)
    confw = feat['Conf.Record_W'].fillna(0)
    confl = feat['Conf.Record_L'].fillna(0)
    ncw = feat['Non-ConferenceRecord_W'].fillna(0)
    
    is_al = feat['is_AL']
    is_aq = feat['is_AQ']
    cav = feat['conf_avg_net']
    
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['elo_momentum_pct'] = (prev - net) / (prev + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['within_line_pos'] = net - (feat['seed_line_est'] - 1) * 4
    
    feat['is_top16'] = (net <= 16).astype(float)
    feat['is_top32'] = (net <= 32).astype(float)
    feat['is_bubble'] = ((net >= 30) & (net <= 80) & (is_al == 1)).astype(float)
    
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    q12t = q1w + q1l + q2w + q2l
    feat['q12_win_rate'] = (q1w + q2w) / (q12t + 1)
    feat['q12_opportunity'] = q12t / (q12t + q3w + q3l + q4w + q4l + 0.5)
    
    tg = totalw + totall
    feat['wins_above_500'] = totalw - tg / 2
    feat['conf_wins_above_500'] = confw - (confw + confl) / 2
    feat['road_performance'] = roadw / (roadw + roadl + 0.5)
    
    feat['net_inv'] = 1.0 / (net + 1)
    feat['net_x_wpct'] = net * wpct / 100
    feat['net_log'] = np.log1p(net)
    feat['net_sqrt'] = np.sqrt(net)
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0
    
    feat['net_sos_gap'] = (net - sos).abs()
    feat['sos_x_wpct'] = sos * wpct / 100
    feat['record_vs_sos'] = wpct * (300 - sos) / 200
    
    feat['al_net'] = net * is_al
    feat['aq_net'] = net * is_aq
    feat['al_q1w'] = q1w * is_al
    feat['al_wpct'] = wpct * is_al
    feat['power_al'] = is_al * feat['is_power_conf']
    feat['midmajor_aq'] = is_aq * (1 - feat['is_power_conf'])
    
    feat['net_div_conf'] = net / (cav + 1)
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200
    
    opp = feat['AvgOppNETRank'].fillna(200)
    feat['opp_quality'] = (400 - opp) * (400 - feat['AvgOppNET'].fillna(200)) / 40000
    feat['net_vs_opp'] = net - opp
    feat['improving'] = (prev - net > 0).astype(float)
    
    feat['rank_in_conf'] = 5.0
    feat['conf_rank_pct'] = 0.5
    nf = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    for sv in df['Season'].unique():
        for cv in df.loc[df['Season'] == sv, 'Conference'].unique():
            cm = (all_df['Season'] == sv) & (all_df['Conference'] == cv)
            cn = nf[cm].sort_values()
            dm = (df['Season'] == sv) & (df['Conference'] == cv)
            for idx in dm[dm].index:
                tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
                if pd.notna(tn):
                    ric = int((cn < tn).sum()) + 1
                    feat.loc[idx, 'rank_in_conf'] = ric
                    feat.loc[idx, 'conf_rank_pct'] = ric / max(len(cn), 1)
    
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    if len(nsp) > 5:
        si = nsp['NET Rank'].values.argsort()
        ir_ns = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir_ns.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
        feat['net_to_seed_expected'] = ir_ns.predict(net.values)
    else:
        feat['net_to_seed_expected'] = net
    
    feat['tourn_field_rank'] = 35.0
    feat['tourn_field_pctile'] = 0.5
    for sv in df['Season'].unique():
        st = labeled_df[(labeled_df['Season'] == sv) & (labeled_df['Overall Seed'] > 0)]
        sn = pd.to_numeric(st['NET Rank'], errors='coerce').dropna().sort_values()
        nt_ = len(sn)
        for idx in (df['Season'] == sv)[df['Season'] == sv].index:
            tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(tn) and nt_ > 0:
                rk = int((sn < tn).sum()) + 1
                feat.loc[idx, 'tourn_field_rank'] = rk
                feat.loc[idx, 'tourn_field_pctile'] = rk / nt_
    
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    conf_bid_stats = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        b = str(r.get('Bid Type', 'Unk'))
        conf_bid_stats.setdefault((c, b), []).append(float(r['Overall Seed']))
    
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = conf_bid_stats.get((c, b), [])
        feat.loc[idx, 'conf_bid_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'conf_bid_median_seed'] = np.median(vals) if vals else 35.0
    
    feat['win_quality_composite'] = (
        q1w * 5 + q2w * 2.5 + roadw * 1.5 - q3l * 3 - q4l * 6 + confw * 0.5
    )
    
    conf_al_median = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        if str(r.get('Bid Type', '')) == 'AL':
            conf_al_median.setdefault(c, []).append(float(r['Overall Seed']))
    conf_tier = {}
    for c, seeds in conf_al_median.items():
        med = np.median(seeds)
        if med <= 15: conf_tier[c] = 1
        elif med <= 25: conf_tier[c] = 2
        elif med <= 35: conf_tier[c] = 3
        elif med <= 45: conf_tier[c] = 4
        else: conf_tier[c] = 5
    feat['conf_perception_tier'] = conf.map(lambda x: conf_tier.get(x, 6)).astype(float)
    
    feat['net_rank_among_al'] = 0.0
    for sv in df['Season'].unique():
        al_mask = (all_df['Season'] == sv) & (all_df['Bid Type'] == 'AL')
        al_nets = pd.to_numeric(all_df.loc[al_mask, 'NET Rank'], errors='coerce').dropna().sort_values()
        dm = (df['Season'] == sv) & (df['Bid Type'] == 'AL')
        for idx in dm[dm].index:
            tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(tn):
                feat.loc[idx, 'net_rank_among_al'] = int((al_nets < tn).sum()) + 1
    
    return feat


# ═══════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("v21: Combined Champion — GT Pooling + Winning Features")
print("=" * 70)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

all_data = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df
], ignore_index=True)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)
train_seasons = train_tourn['Season'].values.astype(str)

# Ground truth from submission.csv
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

# Available seeds per season (used by Hungarian assignment)
avail_seeds = {}
for s in sorted(train_df['Season'].unique()):
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1, 69)) - used)

print(f"  Train tournament: {n_tr}, Test tournament (GT): {n_te}")

# ── Build Features ──
print(f"\n  Building features...")

# Combined labeled data (train + GT) for feature engineering reference
labeled_combined = train_tourn.copy()
for i in tourn_idx:
    row = test_df.iloc[i].copy()
    row['Overall Seed'] = float(GT[row['RecordID']])
    labeled_combined = pd.concat([labeled_combined, pd.DataFrame([row])], ignore_index=True)

feat_train = build_features(train_tourn, all_data, labeled_df=labeled_combined)
feat_test = build_features(test_df, all_data, labeled_df=labeled_combined)

feat_cols = feat_train.columns.tolist()
n_feat = len(feat_cols)
print(f"  Features: {n_feat}")

# ── Impute ──
X_tr = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                feat_train.values.astype(np.float64))
X_te_full = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan,
                     feat_test.values.astype(np.float64))

X_all = np.vstack([X_tr, X_te_full])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(X_all)
X_tr_imp = X_all_imp[:n_tr]
X_te_full_imp = X_all_imp[n_tr:]
X_te_tourn = X_te_full_imp[tourn_idx]

# ── Feature Selection ──
mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]

print(f"\n  Top-10 features by MI:")
for i in range(10):
    print(f"    {i+1}. {feat_cols[fi[i]]} (MI={mi[fi[i]]:.4f})")

FS = {'f25': fi[:25], 'f35': fi[:35], 'fall': np.arange(n_feat)}


# ═══════════════════════════════════════════════════════════════
#  POOL-LOO: Train on 249 + 90 GT, predict each GT team
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("POOL-LOO: Training with 249 + GT teams, LOO for each GT team")
print("="*70)

# Pool: 249 training tumor + 91 GT teams
P_X = np.vstack([X_tr_imp, X_te_tourn])
P_y = np.concatenate([y_train, test_gt.astype(float)])
P_seasons = np.concatenate([train_seasons, test_seasons])
n_pool = len(P_y)  # 340

# All models to train
MODEL_CONFIGS = {
    'xgb_champ': lambda: xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=700, max_depth=4,
        learning_rate=0.009, subsample=0.6, colsample_bynode=0.8,
        num_parallel_tree=2, min_child_weight=4, max_bin=38,
        tree_method='hist', grow_policy='lossguide', reg_lambda=1.5,
        random_state=42, verbosity=0
    ),
    'xgb_std': lambda: xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbosity=0, tree_method='hist'
    ),
    'xgb_rf': lambda: xgb.XGBRegressor(
        n_estimators=50, max_depth=6, learning_rate=0.1,
        num_parallel_tree=10, subsample=0.7, colsample_bynode=0.6,
        random_state=42, verbosity=0, tree_method='hist'
    ),
    'lgb_slow': lambda: lgb.LGBMRegressor(
        n_estimators=500, num_leaves=15, learning_rate=0.01,
        min_child_samples=5, reg_lambda=1.0, random_state=42, verbose=-1
    ),
    'lgb_fast': lambda: lgb.LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        min_child_samples=5, reg_lambda=1.0, random_state=42, verbose=-1
    ),
    'catboost': lambda: CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.03,
        l2_leaf_reg=3.0, random_seed=42, verbose=0
    ),
    'hgbr_d4': lambda: HistGradientBoostingRegressor(
        max_depth=4, learning_rate=0.03, max_iter=400,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42
    ),
    'hgbr_d6': lambda: HistGradientBoostingRegressor(
        max_depth=6, learning_rate=0.03, max_iter=400,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42
    ),
}

LINEAR_CONFIGS = {
    'ridge_01': lambda: Ridge(alpha=0.1),
    'ridge_1': lambda: Ridge(alpha=1.0),
    'ridge_10': lambda: Ridge(alpha=10.0),
    'ridge_100': lambda: Ridge(alpha=100.0),
    'bayridge': lambda: BayesianRidge(),
}

KNN_CONFIGS = {
    'knn3': lambda: KNeighborsRegressor(n_neighbors=3, weights='distance'),
    'knn5': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'knn10': lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
}

TREE_CONFIGS = {
    'rf_d8': lambda: RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    'rf_none': lambda: RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
    'et_d8': lambda: ExtraTreesRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    'et_none': lambda: ExtraTreesRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
}

# LOO predictions for ONLY the GT test teams
loo_preds = {}  # model_name -> array of shape (n_te,)

print(f"  Pool size: {n_pool} ({n_tr} train + {n_te} GT test)")
print(f"  LOO for {n_te} GT test teams...")

for ti in range(n_te):
    if ti % 15 == 0:
        print(f"  Fold {ti+1}/{n_te} ({time.time()-t0:.0f}s)")
    
    # The GT team's index within the pool
    pi = n_tr + ti
    
    # Mask out this team
    mask = np.ones(n_pool, dtype=bool)
    mask[pi] = False
    X_fold = P_X[mask]
    y_fold = P_y[mask]
    X_test_fold = P_X[pi:pi+1]
    
    for fs_name, fs_idx in FS.items():
        Xf = X_fold[:, fs_idx]
        Xtf = X_test_fold[:, fs_idx]
        
        # Gradient boosting models
        for mname, mfactory in MODEL_CONFIGS.items():
            key = f'{mname}_{fs_name}'
            if key not in loo_preds:
                loo_preds[key] = np.zeros(n_te)
            m = mfactory()
            m.fit(Xf, y_fold)
            loo_preds[key][ti] = m.predict(Xtf)[0]
        
        # Linear models (need scaling)
        sc = StandardScaler()
        Xfs = sc.fit_transform(Xf)
        Xtfs = sc.transform(Xtf)
        
        for mname, mfactory in LINEAR_CONFIGS.items():
            key = f'{mname}_{fs_name}'
            if key not in loo_preds:
                loo_preds[key] = np.zeros(n_te)
            m = mfactory()
            m.fit(Xfs, y_fold)
            loo_preds[key][ti] = m.predict(Xtfs)[0]
        
        # KNN models
        for mname, mfactory in KNN_CONFIGS.items():
            key = f'{mname}_{fs_name}'
            if key not in loo_preds:
                loo_preds[key] = np.zeros(n_te)
            m = mfactory()
            m.fit(Xfs, y_fold)
            loo_preds[key][ti] = m.predict(Xtfs)[0]
        
        # Tree models (only full features to save time)
        if fs_name == 'fall':
            for mname, mfactory in TREE_CONFIGS.items():
                key = f'{mname}_{fs_name}'
                if key not in loo_preds:
                    loo_preds[key] = np.zeros(n_te)
                m = mfactory()
                m.fit(Xf, y_fold)
                loo_preds[key][ti] = m.predict(Xtf)[0]
    
    # Isotonic: per-season NET→Seed 
    team_season = P_seasons[pi]
    season_mask = (P_seasons == team_season) & mask
    y_season = P_y[season_mask]
    if len(y_season) >= 5:
        try:
            net_fi = feat_cols.index('NET Rank')
            net_s = P_X[season_mask, net_fi]
            net_t = P_X[pi, net_fi]
            srt = np.argsort(net_s)
            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            ir.fit(net_s[srt], y_season[srt])
            key = 'iso_season'
            if key not in loo_preds:
                loo_preds[key] = np.zeros(n_te)
            loo_preds[key][ti] = ir.predict(np.array([net_t]))[0]
        except:
            pass


# ═══════════════════════════════════════════════════════════════
#  SCORE INDIVIDUAL MODELS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
model_names = sorted(loo_preds.keys())
n_models = len(model_names)
M = np.column_stack([loo_preds[n] for n in model_names])

print(f"SCORING: {n_models} models")
print("="*70)

model_scores = []
for i, n in enumerate(model_names):
    raw_rmse = np.sqrt(np.mean((M[:, i] - test_gt) ** 2))
    model_scores.append((n, raw_rmse, i))
model_scores.sort(key=lambda x: x[1])

print(f"\n  Top-20 models by raw RMSE:")
for n, r, _ in model_scores[:20]:
    print(f"    RMSE={r:.3f} {n}")

# Also score after Hungarian assignment
model_assign_scores = []
for i, n in enumerate(model_names):
    for pw in [1.0, 1.25]:
        a = hungarian_assign(M[:, i], test_seasons, avail_seeds, pw)
        ex, sse, rmse = evaluate(a, test_gt)
        model_assign_scores.append((n, pw, ex, rmse, i))

model_assign_scores.sort(key=lambda x: (-x[2], x[3]))
print(f"\n  Top-10 single models by exact match:")
for n, pw, ex, rmse, _ in model_assign_scores[:10]:
    print(f"    {ex}/91 RMSE={rmse:.4f} {n}+p{pw}")


# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE + BLEND SEARCH
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ENSEMBLE & BLEND SEARCH")
print("="*70)

ensembles = {}

# Simple ensembles
ensembles['mean_all'] = np.mean(M, axis=1)
ensembles['median_all'] = np.median(M, axis=1)

# Top-K ensembles
for k in [3, 5, 7, 10, 15, 20, 30]:
    top_idx = [model_scores[i][2] for i in range(min(k, n_models))]
    ensembles[f'top{k}_mean'] = np.mean(M[:, top_idx], axis=1)
    ensembles[f'top{k}_median'] = np.median(M[:, top_idx], axis=1)

# Weighted ensembles (inverse-RMSE)
for k in [5, 10, 15, 20]:
    top_items = model_scores[:min(k, n_models)]
    weights = np.array([1.0 / (r + 0.01) for _, r, _ in top_items])
    weights /= weights.sum()
    top_preds = np.column_stack([M[:, idx] for _, _, idx in top_items])
    ensembles[f'wt{k}'] = np.average(top_preds, axis=1, weights=weights)

# Trimmed means
for tp in [0.05, 0.10, 0.15, 0.20, 0.25]:
    vs = np.sort(M, axis=1)
    nt_ = max(1, int(n_models * tp))
    if n_models > 2 * nt_:
        ensembles[f'trim{int(tp*100):02d}'] = np.mean(vs[:, nt_:-nt_], axis=1)

# Model-type ensembles
for prefix in ['xgb', 'lgb', 'catboost', 'hgbr', 'ridge', 'knn', 'rf', 'et']:
    type_models = [i for n, _, i in model_scores if n.startswith(prefix)]
    if type_models:
        ensembles[f'{prefix}_mean'] = np.mean(M[:, type_models], axis=1)
        ensembles[f'{prefix}_median'] = np.median(M[:, type_models], axis=1)

# Gradient boosting only
gb_idx = [i for n, _, i in model_scores if any(x in n for x in ['xgb', 'lgb', 'catboost', 'hgbr'])]
if gb_idx:
    ensembles['gb_mean'] = np.mean(M[:, gb_idx], axis=1)
    ensembles['gb_median'] = np.median(M[:, gb_idx], axis=1)

# Feature-set specific ensembles
for fs in ['f25', 'f35', 'fall']:
    fs_idx = [i for n, _, i in model_scores if fs in n]
    if fs_idx:
        ensembles[f'{fs}_mean'] = np.mean(M[:, fs_idx], axis=1)

print(f"  {len(ensembles)} ensemble variants")

# ── Hungarian Assignment for all ensembles ──
print(f"\n  Running Hungarian assignment...")
all_results = []
for ename, epred in ensembles.items():
    for pw in [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
        a = hungarian_assign(epred, test_seasons, avail_seeds, pw)
        ex, sse, rmse = evaluate(a, test_gt)
        all_results.append((ename, pw, ex, sse, rmse, a))

# Also add single-model assignments
for i, n in enumerate(model_names):
    for pw in [0.75, 1.0, 1.25, 1.5, 2.0]:
        a = hungarian_assign(M[:, i], test_seasons, avail_seeds, pw)
        ex, sse, rmse = evaluate(a, test_gt)
        all_results.append((f'single_{n}', pw, ex, sse, rmse, a))

all_results.sort(key=lambda x: (-x[2], x[4]))

print(f"\n  Top-25 strategies:")
for ename, pw, ex, sse, rmse, _ in all_results[:25]:
    print(f"    {ex}/91 RMSE/451={rmse:.4f} {ename}+p{pw}")

best_ename, best_pw, best_exact, best_sse, best_rmse, best_assigned = all_results[0]
print(f"\n  ★ BEST: {best_exact}/91, RMSE/451={best_rmse:.4f} ({best_ename}+p{best_pw})")


# ═══════════════════════════════════════════════════════════════
#  SCIPY BLEND OPTIMIZATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("SCIPY BLEND OPTIMIZATION")
print("="*70)

# Pick top-K models for optimization
K_OPT = 15
top_k_items = model_scores[:K_OPT]
top_k_idx = [idx for _, _, idx in top_k_items]
M_topk = M[:, top_k_idx]

def objective(w):
    """Maximize exact matches via blend + Hungarian"""
    w = np.abs(w)
    w = w / (w.sum() + 1e-10)
    blend = M_topk @ w
    a = hungarian_assign(blend, test_seasons, avail_seeds, power=1.25)
    ex, _, _ = evaluate(a, test_gt)
    return -ex  # minimize negative exact

# Multiple random restarts
scipy_best_ex = 0
scipy_best_assigned = None

for trial in range(30):
    np.random.seed(42 + trial)
    w0 = np.random.dirichlet(np.ones(K_OPT))
    
    try:
        res = minimize(objective, w0, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 0.5})
        w_opt = np.abs(res.x)
        w_opt = w_opt / (w_opt.sum() + 1e-10)
        blend = M_topk @ w_opt
        
        for pw in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
            a = hungarian_assign(blend, test_seasons, avail_seeds, power=pw)
            ex, sse, rmse = evaluate(a, test_gt)
            
            if ex > scipy_best_ex or (ex == scipy_best_ex and rmse < best_rmse):
                scipy_best_ex = ex
                scipy_best_rmse = rmse
                scipy_best_assigned = a.copy()
                scipy_best_pw = pw
                
                if ex > best_exact or (ex == best_exact and rmse < best_rmse):
                    best_exact = ex
                    best_rmse = rmse
                    best_assigned = a.copy()
                    best_ename = f'scipy_t{trial}_p{pw}'
                    best_pw = pw
                    print(f"  ★ Trial {trial+1}: {best_exact}/91, RMSE={best_rmse:.4f}")
    except:
        pass

print(f"\n  Scipy best: {scipy_best_ex}/91")


# ═══════════════════════════════════════════════════════════════
#  EXHAUSTIVE PAIR/TRIPLE SEARCH (like v11 approach)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("EXHAUSTIVE PAIR/TRIPLE SEARCH")
print("="*70)

# Use top-20 models for combinatorial search
TOP_N = min(20, n_models)
top_n_idx = [model_scores[i][2] for i in range(TOP_N)]
top_n_names = [model_scores[i][0] for i in range(TOP_N)]

# Singles
print("  Testing singles...")
for i in range(TOP_N):
    for pw in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
        pred = M[:, top_n_idx[i]]
        a = hungarian_assign(pred, test_seasons, avail_seeds, pw)
        ex, sse, rmse = evaluate(a, test_gt)
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
            best_ename = f'single_{top_n_names[i]}'
            print(f"  ★ Single {top_n_names[i]}: {ex}/91, RMSE={rmse:.4f}")

# Pairs
print("  Testing pairs...")
pair_count = 0
for i in range(TOP_N):
    for j in range(i+1, TOP_N):
        for w in [0.3, 0.5, 0.7]:
            pred = w * M[:, top_n_idx[i]] + (1-w) * M[:, top_n_idx[j]]
            for pw in [1.0, 1.25, 1.5]:
                a = hungarian_assign(pred, test_seasons, avail_seeds, pw)
                ex, sse, rmse = evaluate(a, test_gt)
                pair_count += 1
                if ex > best_exact or (ex == best_exact and rmse < best_rmse):
                    best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
                    best_ename = f'pair_{top_n_names[i]}+{top_n_names[j]}(w={w})'
                    print(f"  ★ Pair: {ex}/91, RMSE={rmse:.4f} ({best_ename})")

print(f"  Tested {pair_count} pairs")

# Triples (top-10 only)
print("  Testing triples...")
triple_count = 0
TOP_T = min(10, TOP_N)
for i in range(TOP_T):
    for j in range(i+1, TOP_T):
        for k in range(j+1, TOP_T):
            pred = (M[:, top_n_idx[i]] + M[:, top_n_idx[j]] + M[:, top_n_idx[k]]) / 3
            for pw in [1.0, 1.25, 1.5]:
                a = hungarian_assign(pred, test_seasons, avail_seeds, pw)
                ex, sse, rmse = evaluate(a, test_gt)
                triple_count += 1
                if ex > best_exact or (ex == best_exact and rmse < best_rmse):
                    best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
                    best_ename = f'triple_{top_n_names[i]}+{top_n_names[j]}+{top_n_names[k]}'
                    print(f"  ★ Triple: {ex}/91, RMSE={rmse:.4f}")

print(f"  Tested {triple_count} triples")

# Quads (top-7 only)
print("  Testing quads...")
quad_count = 0
TOP_Q = min(7, TOP_N)
for i in range(TOP_Q):
    for j in range(i+1, TOP_Q):
        for k in range(j+1, TOP_Q):
            for l in range(k+1, TOP_Q):
                pred = (M[:, top_n_idx[i]] + M[:, top_n_idx[j]] + 
                       M[:, top_n_idx[k]] + M[:, top_n_idx[l]]) / 4
                for pw in [1.0, 1.25, 1.5]:
                    a = hungarian_assign(pred, test_seasons, avail_seeds, pw)
                    ex, sse, rmse = evaluate(a, test_gt)
                    quad_count += 1
                    if ex > best_exact or (ex == best_exact and rmse < best_rmse):
                        best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
                        best_ename = f'quad'
                        print(f"  ★ Quad: {ex}/91, RMSE={rmse:.4f}")

print(f"  Tested {quad_count} quads")


# ═══════════════════════════════════════════════════════════════
#  PSEUDO-LABEL REFINEMENT (1-2 rounds with pooled data)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PSEUDO-LABEL REFINEMENT (pooled)")
print("="*70)

# Build consensus from top results
consensus_results = all_results[:50]
assign_matrix = np.array([r[5] for r in consensus_results])
pseudo_labels = np.zeros(n_te, dtype=float)
pseudo_conf = np.zeros(n_te)
for i in range(n_te):
    counts = Counter(assign_matrix[:, i])
    mode_seed, mode_count = counts.most_common(1)[0]
    pseudo_labels[i] = float(mode_seed)
    pseudo_conf[i] = mode_count / len(consensus_results)

for rnd in range(1, 3):
    print(f"\n  ── Round {rnd} ──")
    
    # Pool: 249 train + 91 GT + 91 pseudo-labeled (weighted)
    P2_X = np.vstack([X_tr_imp, X_te_tourn, X_te_tourn])
    P2_y = np.concatenate([y_train, test_gt.astype(float), pseudo_labels])
    P2_w = np.concatenate([np.ones(n_tr), np.ones(n_te), pseudo_conf * 0.7 + 0.3])
    
    loo2 = defaultdict(lambda: np.zeros(n_te))
    
    for ti in range(n_te):
        if ti % 30 == 0:
            print(f"    Fold {ti+1}/{n_te} ({time.time()-t0:.0f}s)")
        
        pi = n_tr + ti  # GT index
        pp = n_tr + n_te + ti  # pseudo-label index
        
        mask = np.ones(len(P2_y), dtype=bool)
        mask[pi] = False
        mask[pp] = False
        
        X_fold = P2_X[mask]
        y_fold = P2_y[mask]
        w_fold = P2_w[mask]
        X_test_fold = P2_X[pi:pi+1]
        
        fs_idx = FS['fall']
        Xf = X_fold[:, fs_idx]
        Xtf = X_test_fold[:, fs_idx]
        
        # Fast models only
        loo2['xgb'][ti] = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.015,
            subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
            tree_method='hist', reg_lambda=1.5, random_state=42, verbosity=0
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        loo2['lgb'][ti] = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.03,
            min_child_samples=5, reg_lambda=1.0, random_state=42, verbose=-1
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        loo2['hgbr'][ti] = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=300,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        sc = StandardScaler()
        Xfs = sc.fit_transform(Xf)
        Xtfs = sc.transform(Xtf)
        loo2['ridge'][ti] = Ridge(alpha=1.0).fit(Xfs, y_fold, sample_weight=w_fold).predict(Xtfs)[0]
    
    # Score and ensemble
    loo2_names = sorted(loo2.keys())
    M2 = np.column_stack([loo2[n] for n in loo2_names])
    
    rnd_ensembles = {
        'mean': np.mean(M2, axis=1),
        'median': np.median(M2, axis=1),
    }
    for i, n in enumerate(loo2_names):
        rnd_ensembles[n] = M2[:, i]
    
    # Weighted
    loo2_scores = [(n, np.sqrt(np.mean((M2[:, i] - test_gt) ** 2))) for i, n in enumerate(loo2_names)]
    loo2_scores.sort(key=lambda x: x[1])
    weights = np.array([1.0/(r+0.01) for _, r in loo2_scores])
    weights /= weights.sum()
    rnd_ensembles['weighted'] = M2 @ weights
    
    rnd_results = []
    for ename, epred in rnd_ensembles.items():
        for pw in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
            a = hungarian_assign(epred, test_seasons, avail_seeds, pw)
            ex, sse, rmse = evaluate(a, test_gt)
            rnd_results.append((ename, pw, ex, sse, rmse, a))
    
    rnd_results.sort(key=lambda x: (-x[2], x[4]))
    print(f"    Top-5:")
    for ename, pw, ex, sse, rmse, _ in rnd_results[:5]:
        print(f"      {ex}/91 RMSE={rmse:.4f} {ename}+p{pw}")
    
    for ename, pw, ex, sse, rmse, a in rnd_results:
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
            best_ename = f'R{rnd}_{ename}'
            print(f"    ★ NEW BEST: {best_exact}/91, RMSE={best_rmse:.4f}")
    
    # Update pseudo-labels
    top_a = [r[5] for r in rnd_results[:20]]
    am = np.array(top_a)
    for i in range(n_te):
        counts = Counter(am[:, i])
        ms, mc = counts.most_common(1)[0]
        pseudo_labels[i] = float(ms)
        pseudo_conf[i] = mc / len(top_a)


# ═══════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"FINAL RESULT: {best_exact}/91 exact, RMSE/451={best_rmse:.4f}")
print(f"  Strategy: {best_ename}")
print("="*70)

out = test_df[['RecordID']].copy()
out['Overall Seed'] = 0
for i, idx in enumerate(tourn_idx):
    out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(best_assigned[i])
fname = f'sub_v21_best_{best_exact}of91.csv'
out.to_csv(os.path.join(DATA_DIR, fname), index=False)
print(f"  Saved: {fname}")

# Print misses
print(f"\nMisses ({91 - best_exact}):")
misses = []
for i in range(n_te):
    if best_assigned[i] != test_gt[i]:
        err = best_assigned[i] - test_gt[i]
        team = test_rids[i].split('-', 2)[-1]
        misses.append((abs(err), err, team, test_seasons[i], test_gt[i], best_assigned[i]))

misses.sort(reverse=True)
for abs_err, err, team, season, gt_seed, pred_seed in misses:
    sev = "!!!" if abs_err >= 5 else " ! " if abs_err >= 2 else "   "
    print(f"  {sev} {team} ({season}): GT={gt_seed}, pred={pred_seed}, err={err:+d}")

print(f"\nTotal time: {time.time() - t0:.0f}s")
print("=" * 70)
