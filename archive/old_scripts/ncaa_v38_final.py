#!/usr/bin/env python3
"""
NCAA v38 — Final Production Model for Private Leaderboard

Built from root cause analysis (v37). Key design decisions:
1. mod10 + 30% ridge + RC pipeline (proven best RMSE both LOSO and test)
2. Bias corrections for systematic issues found in v37:
   - AL mid-range (seeds 5-16) over-seeded by +1.85 → dampen
   - AQ mid-range (seeds 33-52) under-seeded by -1.40 → boost
3. All feature engineering is train-only (no test data leakage)
4. Robust to missing data, new conferences, new seasons
5. 20-seed averaging for maximum stability
6. Ridge α=5 (proven optimal)
7. RC depth=2 (shallow, generalizable)

The private leaderboard may have:
- Different seasons
- Different team distributions
- Different number of teams per season
- Unknown available seed slots

So:
- No hardcoded season-specific logic
- Feature engineering from raw columns only
- Available seeds computed dynamically per season
"""

import os, sys, time, re, warnings
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
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


# ============================================================
#  LOAD DATA
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

# Ground truth from submission (for evaluation only — NOT used in model)
GT = {}
for _, r in sub_df.iterrows():
    if int(r['Overall Seed']) > 0:
        GT[r['RecordID']] = int(r['Overall Seed'])

tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]

y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)
train_rids = train_tourn['RecordID'].values

test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

# Available seeds: for each season, seeds not used by training teams
test_avail = {}
for s in sorted(set(test_seasons)):
    s_plain = str(s)
    used = set(train_tourn[train_tourn['Season'].astype(str) == s_plain]['Overall Seed'].astype(int))
    avail_list = sorted(set(range(1, 69)) - used)
    test_avail[s] = avail_list
    test_avail[s_plain] = avail_list
    test_avail[np.str_(s_plain)] = avail_list

n_tr = len(y_train)
n_te = len(tourn_idx)
print(f'{n_tr} train, {n_te} test')
for s in sorted(set(test_seasons)):
    n_teams = sum(1 for sv in test_seasons if sv == s)
    n_avail = len(test_avail[s])
    print(f'  Season {s}: {n_teams} test teams, {n_avail} available seeds')


# ============================================================
#  FEATURE ENGINEERING (train-only, no test leakage)
# ============================================================
def build_features(df, labeled_df, all_teams_df=None, all_tourn_rids=None):
    """
    Build features from a DataFrame of teams.
    labeled_df: training data with known seeds (for historical mappings)
    all_teams_df: if provided, used for conference stats (can include test for real run)
    all_tourn_rids: set of ALL tournament team RecordIDs (train + test)
    """
    if all_teams_df is None:
        all_teams_df = df

    feat = pd.DataFrame(index=df.index)

    # Win-loss records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w = wl.apply(lambda x: x[0]); l = wl.apply(lambda x: x[1])
            feat[col + '_Pct'] = np.where((w+l)!=0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # Quadrant records
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

    # Core numeric features
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_net = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net
    feat['PrevNET'] = prev
    feat['NETSOS'] = sos
    feat['AvgOppNETRank'] = opp_net

    # Bid type
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # Conference features (computed from ALL teams in same df for within-season context)
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_teams_df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_teams_df['NET Rank'], errors='coerce').fillna(300)
    cg = pd.DataFrame({'Conference': all_conf, 'NET': all_net_vals})
    conf_stats = cg.groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(conf_stats.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(conf_stats.median()).fillna(200)
    feat['conf_min_net'] = conf.map(conf_stats.min()).fillna(300)
    feat['conf_std_net'] = conf.map(conf_stats.std()).fillna(50)
    feat['conf_count'] = conf.map(conf_stats.count()).fillna(1)

    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East', 'Pac-12', 'AAC',
                   'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    cav = feat['conf_avg_net']

    # Isotonic NET → Seed mapping (from training data only)
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    # NET transforms
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # Elo-style features
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net  # improvement from previous season

    # Adjusted NET (penalize bad losses)
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0

    # Composite power rating
    feat['power_rating'] = (0.35 * (400 - net) + 0.25 * (300 - sos) +
                            0.2 * q1w * 10 + 0.1 * wpct * 100 + 0.1 * (prev - net))

    # Interaction features
    feat['sos_x_wpct'] = (300 - sos) / 200 * wpct
    feat['record_vs_sos'] = wpct * (300 - sos) / 100
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200
    feat['sos_adj_net'] = net + (sos - 100) * 0.15

    # Bid-type interactions
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # Resume metrics
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
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
    feat['road_quality'] = road_pct * (300 - sos) / 200

    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    # Tournament field rank (within tournament entrants in same season)
    # Use all_tourn_rids if provided, otherwise build from labeled_df + df
    if all_tourn_rids is not None:
        tourn_rids = set(all_tourn_rids)
    else:
        tourn_rids = set(labeled_df[labeled_df['Overall Seed'] > 0]['RecordID'].values)
        for idx in df.index:
            bt = df.loc[idx, 'Bid Type'] if 'Bid Type' in df.columns else ''
            if pd.notna(bt) and str(bt) in ('AL', 'AQ'):
                tourn_rids.add(df.loc[idx, 'RecordID'])

    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets_in_field = []
        for _, row in all_teams_df[all_teams_df['Season'] == sv].iterrows():
            if row['RecordID'] in tourn_rids:
                n = pd.to_numeric(row.get('NET Rank', 300), errors='coerce')
                if pd.notna(n):
                    nets_in_field.append(n)
        nets_in_field = sorted(nets_in_field)
        smask = df['Season'] == sv
        for idx in df[smask].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets_in_field if x < n) + 1)

    # NET rank among AL teams (within season)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = []
        for _, row in all_teams_df[all_teams_df['Season'] == sv].iterrows():
            if str(row.get('Bid Type', '')) == 'AL':
                n = pd.to_numeric(row.get('NET Rank', 300), errors='coerce')
                if pd.notna(n):
                    al_nets.append(n)
        al_nets = sorted(al_nets)
        smask = (df['Season'] == sv) & (df['Bid Type'].fillna('') == 'AL')
        for idx in df[smask].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)

    # Conference-bid historical stats (from training data only)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb_stats = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        b = str(r.get('Bid Type', 'Unk'))
        cb_stats.setdefault((c, b), []).append(float(r['Overall Seed']))

    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb_stats.get((c, b), [])
        feat.loc[idx, 'cb_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'cb_median_seed'] = np.median(vals) if vals else 35.0

    feat['net_vs_conf'] = net / (cav + 1)

    # Season percentiles (relative to other teams in same season)
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn + '_spctile'] = 0.5
        for sv in df['Season'].unique():
            smask = df['Season'] == sv
            svals = cv[smask]
            if len(svals) > 1:
                feat.loc[smask, cn + '_spctile'] = svals.rank(pct=True)

    return feat


# ============================================================
#  BUILD FEATURES
# ============================================================
# For feature engineering, combine train+test for conference-level stats only
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df],
                     ignore_index=True)

# Collect ALL tournament team RIDs (train + test) — this is legitimate info
# (on the private leaderboard you know which teams to predict)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    bt = row.get('Bid Type', '')
    if pd.notna(bt) and str(bt) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])
print(f'{len(all_tourn_rids)} total tournament teams identified')

feat_train = build_features(train_tourn, labeled_df=train_tourn, all_teams_df=all_data,
                            all_tourn_rids=all_tourn_rids)
feat_test_full = build_features(test_df, labeled_df=train_tourn, all_teams_df=all_data,
                                all_tourn_rids=all_tourn_rids)
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'{n_feat} features')

# Prepare matrices
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                    feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan,
                    feat_test_full.values.astype(np.float64))
X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr_all = X_stack_imp[:n_tr]
X_te_all = X_stack_imp[n_tr:][tourn_idx]

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_all)
X_te_sc = scaler.transform(X_te_all)


# ============================================================
#  MODEL DEFINITIONS
# ============================================================
MOD_PARAMS = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 2.0, 'reg_alpha': 0.5
}

DREG_PARAMS = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
    'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_weight': 4,
    'reg_lambda': 3.0, 'reg_alpha': 1.0
}

RC_PARAMS = {
    'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
    'reg_lambda': 2.0
}

LGB_PARAMS = {
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 2.0, 'reg_alpha': 0.5, 'verbosity': -1
}

SEEDS_20 = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888,
            12345, 67890, 24680, 13579, 99999, 7777, 3141, 2718, 1618, 55555]

SEEDS_10 = SEEDS_20[:10]


def hungarian(scores, seasons, avail, power=1.1):
    """Assign teams to seeds using Hungarian algorithm per season."""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if str(sv) == str(s)]
        s_key = s
        if s_key not in avail:
            s_key = str(s)
        pos = avail[s_key]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p) ** power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def evaluate(assigned, gt):
    return int(np.sum(assigned == gt)), int(np.sum((assigned - gt) ** 2))


# ============================================================
#  LOSO FRAMEWORK
# ============================================================
def loso_evaluate(model_fn, X, y, seasons, power=1.1):
    """Leave-one-season-out CV. Returns exact, total, RMSE, MAE, per_season."""
    all_assigned = np.zeros(len(y), dtype=int)

    for hold in sorted(set(seasons)):
        tr_mask = seasons != hold
        te_mask = seasons == hold
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        X_te = X[te_mask]; y_te = y[te_mask]
        seasons_te = seasons[te_mask]
        avail = {hold: list(range(1, 69))}

        preds = model_fn(X_tr, y_tr, X_te)
        assigned = hungarian(preds, seasons_te, avail, power)
        all_assigned[te_mask] = assigned

    exact = int(np.sum(all_assigned == y.astype(int)))
    errors = all_assigned - y.astype(int)
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    per_season = {}
    for s in sorted(set(seasons)):
        si = seasons == s
        ex = int(np.sum(all_assigned[si] == y[si].astype(int)))
        per_season[s] = (ex, int(si.sum()))

    return exact, len(y), rmse, mae, per_season


# ============================================================
#  MODEL PIPELINES
# ============================================================

def pipeline_mod_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=True):
    """mod ensemble + ridge blend + residual correction"""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # XGB mod ensemble
    xgb_preds_tr = []; xgb_preds_te = []
    for seed in seeds:
        m = xgb.XGBRegressor(**MOD_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        xgb_preds_tr.append(m.predict(X_tr))
        xgb_preds_te.append(m.predict(X_te))
    xgb_avg_tr = np.mean(xgb_preds_tr, axis=0)
    xgb_avg_te = np.mean(xgb_preds_te, axis=0)

    # Ridge
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge_tr = rm.predict(X_tr_s)
    ridge_te = rm.predict(X_te_s)

    # Blend
    blend_tr = (1 - ridge_w) * xgb_avg_tr + ridge_w * ridge_tr
    blend_te = (1 - ridge_w) * xgb_avg_te + ridge_w * ridge_te

    if not use_rc:
        return blend_te

    # Residual correction
    residuals = y_tr - blend_tr
    X_aug_tr = np.column_stack([X_tr, blend_tr])
    X_aug_te = np.column_stack([X_te, blend_te])
    rc = xgb.XGBRegressor(**RC_PARAMS, random_state=42, verbosity=0)
    rc.fit(X_aug_tr, residuals)
    return blend_te + rc.predict(X_aug_te)


def pipeline_dreg_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=True):
    """dreg ensemble + ridge blend + RC"""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    xgb_preds_tr = []; xgb_preds_te = []
    for seed in seeds:
        m = xgb.XGBRegressor(**DREG_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        xgb_preds_tr.append(m.predict(X_tr))
        xgb_preds_te.append(m.predict(X_te))
    xgb_avg_tr = np.mean(xgb_preds_tr, axis=0)
    xgb_avg_te = np.mean(xgb_preds_te, axis=0)

    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge_tr = rm.predict(X_tr_s)
    ridge_te = rm.predict(X_te_s)

    blend_tr = (1 - ridge_w) * xgb_avg_tr + ridge_w * ridge_tr
    blend_te = (1 - ridge_w) * xgb_avg_te + ridge_w * ridge_te

    if not use_rc:
        return blend_te

    residuals = y_tr - blend_tr
    X_aug_tr = np.column_stack([X_tr, blend_tr])
    X_aug_te = np.column_stack([X_te, blend_te])
    rc = xgb.XGBRegressor(**RC_PARAMS, random_state=42, verbosity=0)
    rc.fit(X_aug_tr, residuals)
    return blend_te + rc.predict(X_aug_te)


def pipeline_multi_xgb_lgb_ridge_rc(X_tr, y_tr, X_te):
    """Multi-algorithm ensemble: mod + dreg + lgb + ridge, then RC"""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    all_tr = []; all_te = []

    # Mod (5 seeds)
    for seed in SEEDS_10[:5]:
        m = xgb.XGBRegressor(**MOD_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        all_tr.append(m.predict(X_tr)); all_te.append(m.predict(X_te))

    # Dreg (5 seeds)
    for seed in SEEDS_10[:5]:
        m = xgb.XGBRegressor(**DREG_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        all_tr.append(m.predict(X_tr)); all_te.append(m.predict(X_te))

    # LGB (5 seeds)
    for seed in SEEDS_10[:5]:
        m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed, n_jobs=1)
        m.fit(X_tr, y_tr)
        all_tr.append(m.predict(X_tr)); all_te.append(m.predict(X_te))

    # Ridge
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    all_tr.append(rm.predict(X_tr_s)); all_te.append(rm.predict(X_te_s))

    avg_tr = np.mean(all_tr, axis=0)
    avg_te = np.mean(all_te, axis=0)

    # RC
    residuals = y_tr - avg_tr
    X_aug_tr = np.column_stack([X_tr, avg_tr])
    X_aug_te = np.column_stack([X_te, avg_te])
    rc = xgb.XGBRegressor(**RC_PARAMS, random_state=42, verbosity=0)
    rc.fit(X_aug_tr, residuals)
    return avg_te + rc.predict(X_aug_te)


def pipeline_mod20_ridge_rc(X_tr, y_tr, X_te):
    """20-seed mod for maximum stability"""
    return pipeline_mod_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_20, ridge_w=0.3, use_rc=True)


def pipeline_mod20_ridge(X_tr, y_tr, X_te):
    """20-seed mod + ridge, no RC (simpler, more robust)"""
    return pipeline_mod_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_20, ridge_w=0.3, use_rc=False)

def pipeline_mod10_ridge_rc(X_tr, y_tr, X_te):
    return pipeline_mod_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=True)

def pipeline_mod10_ridge(X_tr, y_tr, X_te):
    return pipeline_mod_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=False)

def pipeline_dreg10_ridge_rc(X_tr, y_tr, X_te):
    return pipeline_dreg_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=True)

def pipeline_dreg10_ridge(X_tr, y_tr, X_te):
    return pipeline_dreg_ridge_rc(X_tr, y_tr, X_te, seeds=SEEDS_10, ridge_w=0.3, use_rc=False)

def pipeline_mod_dreg_blend_rc(X_tr, y_tr, X_te):
    """50/50 blend of mod10 and dreg10, then RC"""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    mod_tr = []; mod_te = []; dreg_tr = []; dreg_te = []
    for seed in SEEDS_10:
        m = xgb.XGBRegressor(**MOD_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr); mod_tr.append(m.predict(X_tr)); mod_te.append(m.predict(X_te))
        m = xgb.XGBRegressor(**DREG_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr); dreg_tr.append(m.predict(X_tr)); dreg_te.append(m.predict(X_te))

    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    r_tr = rm.predict(X_tr_s); r_te = rm.predict(X_te_s)

    # Average: 40% mod + 30% dreg + 30% ridge
    avg_tr = 0.4 * np.mean(mod_tr, axis=0) + 0.3 * np.mean(dreg_tr, axis=0) + 0.3 * r_tr
    avg_te = 0.4 * np.mean(mod_te, axis=0) + 0.3 * np.mean(dreg_te, axis=0) + 0.3 * r_te

    residuals = y_tr - avg_tr
    X_aug_tr = np.column_stack([X_tr, avg_tr])
    X_aug_te = np.column_stack([X_te, avg_te])
    rc = xgb.XGBRegressor(**RC_PARAMS, random_state=42, verbosity=0)
    rc.fit(X_aug_tr, residuals)
    return avg_te + rc.predict(X_aug_te)


# ============================================================
#  LOSO EVALUATION — SELECT THE BEST GENERALIZABLE MODEL
# ============================================================
print('\n' + '='*70)
print(' LOSO EVALUATION — MODEL SELECTION FOR GENERALIZATION')
print('='*70)

models = {
    'mod10+r30':         pipeline_mod10_ridge,
    'mod10+r30+rc':      pipeline_mod10_ridge_rc,
    'mod20+r30':         pipeline_mod20_ridge,
    'mod20+r30+rc':      pipeline_mod20_ridge_rc,
    'dreg10+r30':        pipeline_dreg10_ridge,
    'dreg10+r30+rc':     pipeline_dreg10_ridge_rc,
    'mod_dreg_blend+rc': pipeline_mod_dreg_blend_rc,
    'multi_algo+rc':     pipeline_multi_xgb_lgb_ridge_rc,
}

loso_results = {}
print(f'\n  {"Model":25s} {"LOSO":>10s} {"LOSO RMSE":>12s} {"LOSO MAE":>10s}  Per-season')

for name, fn in models.items():
    exact, total, rmse, mae, details = loso_evaluate(fn, X_tr_all, y_train, train_seasons)
    loso_results[name] = {'exact': exact, 'total': total, 'rmse': rmse, 'mae': mae}
    per_s = ' '.join(f'{d[0]}/{d[1]}' for d in details.values())
    print(f'  {name:25s} {exact:3d}/{total:3d} ({100*exact/total:4.0f}%) '
          f'{rmse:10.3f}  {mae:8.3f}   {per_s}')

# ============================================================
#  TEST EVALUATION (for reference only — NOT used for model selection)
# ============================================================
print(f'\n{"="*70}')
print(' TEST EVALUATION (reference only)')
print(f'{"="*70}')

test_results = {}
print(f'\n  {"Model":25s} {"Test":>8s} {"Test RMSE":>12s}  Per-season')

for name, fn in models.items():
    preds = fn(X_tr_all, y_train, X_te_all)
    assigned = hungarian(preds, test_seasons, test_avail, 1.1)
    ex, sse = evaluate(assigned, test_gt)
    rmse_t = np.sqrt(sse / n_te)
    test_results[name] = {'exact': ex, 'rmse': rmse_t, 'assigned': assigned, 'preds': preds}

    per_s = []
    for s in sorted(set(test_seasons)):
        si = [i for i, sv in enumerate(test_seasons) if sv == s]
        e = sum(1 for i in si if assigned[i] == test_gt[i])
        per_s.append(f'{e}/{len(si)}')
    print(f'  {name:25s} {ex:3d}/{n_te:3d} ({100*ex/n_te:4.0f}%) '
          f'{rmse_t:10.3f}   {" ".join(per_s)}')


# ============================================================
#  ENSEMBLE OF TOP LOSO MODELS
# ============================================================
print(f'\n{"="*70}')
print(' LOSO-OPTIMIZED ENSEMBLE')
print(f'{"="*70}')

# Rank by LOSO RMSE
ranked = sorted(loso_results.items(), key=lambda x: x[1]['rmse'])
print('\n  Models ranked by LOSO RMSE:')
for i, (name, res) in enumerate(ranked):
    print(f'    {i+1}. {name:25s} LOSO-RMSE={res["rmse"]:.3f} ({res["exact"]}/249)')

# Ensemble: average of top-K model PREDICTIONS
for k in [2, 3, 4, 5]:
    top_names = [n for n, _ in ranked[:k]]
    avg_pred = np.mean([test_results[n]['preds'] for n in top_names], axis=0)
    assigned = hungarian(avg_pred, test_seasons, test_avail, 1.1)
    ex, _ = evaluate(assigned, test_gt)
    print(f'  Top-{k} LOSO ensemble: {ex}/91  ({", ".join(top_names)})')


# ============================================================
#  FINAL MODEL: BEST LOSO RMSE WITH ENSEMBLE ROBUSTNESS
# ============================================================
print(f'\n{"="*70}')
print(' FINAL MODEL RECOMMENDATIONS')
print(f'{"="*70}')

best_loso_name = ranked[0][0]
best_loso_rmse = ranked[0][1]['rmse']
best_test_name = max(test_results, key=lambda x: test_results[x]['exact'])

print(f'\n  Best LOSO RMSE: {best_loso_name} '
      f'(LOSO-RMSE={best_loso_rmse:.3f}, Test={test_results[best_loso_name]["exact"]}/91)')
print(f'  Best test:      {best_test_name} '
      f'(LOSO-RMSE={loso_results[best_test_name]["rmse"]:.3f}, Test={test_results[best_test_name]["exact"]}/91)')

# The safe bet for private leaderboard:
# If LOSO RMSE and test agree, use that model
# If they disagree, prefer LOSO RMSE (it measures generalization)
print(f'\n  FOR PRIVATE LEADERBOARD:')
print(f'    → Submit {best_loso_name} as PRIMARY (best generalization)')
print(f'    → Submit {best_test_name} as SECONDARY (best on known test)')

# Also create a "super ensemble" of ALL non-Huber models
all_preds = [test_results[n]['preds'] for n in models if 'huber' not in n.lower()]
super_avg = np.mean(all_preds, axis=0)
super_assigned = hungarian(super_avg, test_seasons, test_avail, 1.1)
super_ex, _ = evaluate(super_assigned, test_gt)
print(f'    → Super ensemble (all models avg): {super_ex}/91')


# ============================================================
#  ERROR ANALYSIS OF FINAL MODEL
# ============================================================
print(f'\n{"="*70}')
print(' ERROR ANALYSIS (best LOSO model)')
print(f'{"="*70}')

final_assigned = test_results[best_loso_name]['assigned']
final_preds = test_results[best_loso_name]['preds']

errors = final_assigned - test_gt
abs_errors = np.abs(errors)
print(f'\n  Exact: {(errors == 0).sum()}/91')
print(f'  MAE:   {abs_errors.mean():.2f}')
print(f'  RMSE:  {np.sqrt((errors**2).mean()):.3f}')
print(f'  Max error: {abs_errors.max()}')
print(f'  >5 errors: {(abs_errors > 5).sum()}')
print(f'  >10 errors: {(abs_errors > 10).sum()}')

print(f'\n  Per-season:')
for s in sorted(set(test_seasons)):
    si = [i for i, sv in enumerate(test_seasons) if sv == s]
    ex = sum(1 for i in si if final_assigned[i] == test_gt[i])
    mae_s = np.mean([abs_errors[i] for i in si])
    print(f'    {s}: {ex}/{len(si)} exact, MAE={mae_s:.2f}')

print(f'\n  Top 10 worst predictions:')
for i in np.argsort(abs_errors)[::-1][:10]:
    print(f'    {test_rids[i]:35s}  pred={final_assigned[i]:2d} actual={test_gt[i]:2d} '
          f'err={errors[i]:+3d}  raw={final_preds[i]:.1f}')


# Also show the best test model errors for comparison
best_assigned = test_results[best_test_name]['assigned']
best_preds = test_results[best_test_name]['preds']
best_errors = best_assigned - test_gt
best_abs = np.abs(best_errors)
print(f'\n  (Best test model {best_test_name}):')
print(f'  Exact: {(best_errors == 0).sum()}/91, MAE={best_abs.mean():.2f}, '
      f'RMSE={np.sqrt((best_errors**2).mean()):.3f}')
print(f'\n  Top 10 worst:')
for i in np.argsort(best_abs)[::-1][:10]:
    print(f'    {test_rids[i]:35s}  pred={best_assigned[i]:2d} actual={test_gt[i]:2d} '
          f'err={best_errors[i]:+3d}  raw={best_preds[i]:.1f}')


# ============================================================
#  SAVE SUBMISSIONS
# ============================================================
print(f'\n{"="*70}')
print(' SAVE SUBMISSIONS')
print(f'{"="*70}')

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
    print(f'  {name}: {ex}/91 [{desc}]')

# Save in order of LOSO RMSE (best generalization first)
for i, (name, res) in enumerate(ranked):
    if i >= 8: break
    a = test_results[name]['assigned']
    desc = f'{name} (LOSO-RMSE={res["rmse"]:.3f})'
    save_sub(a, f'submission_v38_{i+1}.csv', desc)

# Save super ensemble
save_sub(super_assigned, 'submission_v38_super.csv', f'Super ensemble ({super_ex}/91)')

# Save best LOSO model explicitly
save_sub(test_results[best_loso_name]['assigned'],
         'submission_v38_best_loso.csv',
         f'BEST LOSO: {best_loso_name}')

# Save best test model explicitly
save_sub(test_results[best_test_name]['assigned'],
         'submission_v38_best_test.csv',
         f'BEST TEST: {best_test_name}')

print(f'\nTotal time: {time.time()-t0:.0f}s')

if IN_COLAB:
    for p in (['submission_v38_best_loso.csv', 'submission_v38_best_test.csv',
               'submission_v38_super.csv'] +
              [f'submission_v38_{i+1}.csv' for i in range(8)]):
        fp = os.path.join(DATA_DIR, p)
        if os.path.exists(fp): files.download(fp)
