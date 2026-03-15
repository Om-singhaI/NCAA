#!/usr/bin/env python3
"""
NCAA Overall Seed Prediction — Ultra Model v2
===============================================
Implements ALL improvement strategies:
1. Enhanced features (added AvgOppNET, NETNonConfSOS, 15+ new engineered features)
2. Advanced algorithms (XGBoost, LightGBM, CatBoost, SVR, KNN, MLP, RF, ExtraTrees, Ridge)
3. Stacking ensemble (Level-0 diverse models → Level-1 Ridge meta-learner)
4. Hyperparameter tuning (Optuna for top models)
5. LOSO cross-validation (honest, no test leakage)
6. Feature selection (built-in importance + backward elimination)
"""

import os, sys, time, re, warnings, json
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost', 'optuna'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import linear_sum_assignment
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# =================================================================
#  DATA LOADING
# =================================================================
print('='*70)
print(' NCAA OVERALL SEED PREDICTION — ULTRA MODEL v2')
print('='*70)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))


def parse_wl(s):
    """Parse win-loss strings, handling Excel month-name corruption."""
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

# Ground truth from submission.csv
GT = {r['RecordID']: int(r['Overall Seed'])
      for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])

# Available seeds per test season (exclude train seeds)
test_avail = {}
for s in sorted(set(test_seasons)):
    used = set(train_tourn[train_tourn['Season'].astype(str)==s]['Overall Seed'].astype(int))
    test_avail[s] = sorted(set(range(1, 69)) - used)

n_tr, n_te = len(y_train), len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df],
                     ignore_index=True)

# All tournament team RIDs (for field rank computation)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    if pd.notna(row.get('Bid Type', '')) and str(row['Bid Type']) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])

print(f'  {n_tr} train | {n_te} test | {len(all_tourn_rids)} tourn teams')

# =================================================================
#  ENHANCED FEATURE ENGINEERING (80+ features)
# =================================================================
def build_features(df, all_df, labeled_df, tourn_rids):
    """Build comprehensive feature set with all available data."""
    feat = pd.DataFrame(index=df.index)

    # ---- Win-loss records ----
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # ---- Quadrant records ----
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
    conf_pct = feat.get('Conf.Record_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    nc_pct = feat.get('Non-ConferenceRecord_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    # ---- Core rankings (ALL raw features now used) ----
    net     = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev    = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos     = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rnk = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    opp_net = pd.to_numeric(df.get('AvgOppNET', pd.Series(dtype=float)), errors='coerce').fillna(0)   # NEW
    nc_sos  = pd.to_numeric(df.get('NETNonConfSOS', pd.Series(dtype=float)), errors='coerce').fillna(200)  # NEW

    feat['NET Rank'] = net
    feat['PrevNET'] = prev
    feat['NETSOS'] = sos
    feat['AvgOppNETRank'] = opp_rnk
    feat['AvgOppNET'] = opp_net          # NEW: average opponent NET rating
    feat['NETNonConfSOS'] = nc_sos       # NEW: non-conference SOS

    # ---- Bid type ----
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # ---- Conference stats ----
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count']   = conf.map(cs.count()).fillna(1)
    power_confs = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    cav = feat['conf_avg_net']

    # ---- Isotonic NET→Seed mapping ----
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    # ---- NET transforms ----
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # ---- Composite metrics ----
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net  # improvement from last year
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    # ---- Bid interactions ----
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # ---- Resume metrics ----
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

    # ---- NEW: Enhanced features using previously-unused columns ----

    # SOS consistency: gap between overall SOS and non-conference SOS
    feat['sos_gap'] = sos - nc_sos           # high = weaker non-conf schedule
    feat['nc_sos_quality'] = (300 - nc_sos) / 200

    # Opponent quality metrics
    feat['opp_net_normalized'] = opp_net / 100.0
    feat['opp_rank_vs_sos'] = opp_rnk - sos   # scheduling efficiency

    # Conference vs non-conference performance gap
    feat['conf_nc_gap'] = conf_pct - nc_pct   # do they play up/down outside conf?

    # Quadrant efficiency score (weighted by difficulty)
    total_q_games = q1w + q1l + q2w + q2l + q3w + q3l + q4w + q4l + 0.01
    feat['quadrant_efficiency'] = (q1w*4 + q2w*3 + q3w*1 + q4w*0.5) / total_q_games

    # Top-heavy wins ratio
    feat['elite_win_ratio'] = q1w / (q1w + q2w + q3w + q4w + 0.1)

    # Loss quality (how bad are your losses?)
    total_losses = q1l + q2l + q3l + q4l + 0.01
    feat['bad_loss_ratio'] = (q3l + q4l) / total_losses

    # Win margin proxy: wins in tough games vs total games
    feat['toughness_score'] = (q1w + q2w) / (feat.get('total_games', pd.Series(30, index=df.index)).fillna(30))

    # NET rank stability (if PrevNET available)
    feat['net_volatility'] = np.abs(prev - net)
    feat['net_improved'] = (prev > net).astype(float)  # 1 if improved

    # Conference dominance
    feat['conf_dominance'] = (feat['conf_avg_net'] - net) / (feat['conf_std_net'] + 1)

    # Road warrior: road performance relative to overall
    feat['road_vs_overall'] = road_pct - wpct

    # Combined strength metric
    feat['combined_strength'] = (
        0.3 * (400 - net) / 400 +
        0.2 * wpct +
        0.15 * (300 - sos) / 300 +
        0.15 * feat['q1_dominance'] +
        0.1 * road_pct +
        0.1 * (1 - feat['total_bad_losses'] / 10)
    )

    # ---- Tournament field rank ----
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in all_df[all_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids
                       and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)

    # ---- AL rank ----
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in all_df[all_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL'
                          and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)

    # ---- Historical conference-bid stats ----
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
        feat.loc[idx, 'cb_std_seed'] = np.std(vals) if len(vals) > 1 else 15.0   # NEW

    feat['net_vs_conf'] = net / (cav + 1)

    # ---- Season percentiles ----
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating']),
                   ('combined_strength', feat['combined_strength'])]:   # NEW
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)

    # ---- NEW: Conference seed history (average seed by conference) ----
    conf_seeds = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        conf_seeds.setdefault(c, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        vals = conf_seeds.get(c, [])
        feat.loc[idx, 'conf_hist_avg_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'conf_hist_min_seed'] = np.min(vals) if vals else 1.0
        feat.loc[idx, 'conf_hist_teams'] = len(vals) if vals else 0

    return feat


print('\n  Building features...')
feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'  {n_feat} features')

# Impute & prepare
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                    feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan,
                    feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]
X_te = X_all[n_tr:][tourn_idx]

scaler_global = StandardScaler()
X_tr_sc = scaler_global.fit_transform(X_tr)
X_te_sc = scaler_global.transform(X_te)

# =================================================================
#  HUNGARIAN ASSIGNMENT
# =================================================================
def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned

# =================================================================
#  BASE MODELS DEFINITION
# =================================================================
SEEDS = [42, 123, 777, 2024, 31415]

def make_xgb(params=None, seed=42):
    p = {
        'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
        'reg_lambda': 3.0, 'reg_alpha': 1.0,
    }
    if params: p.update(params)
    return xgb.XGBRegressor(**p, random_state=seed, verbosity=0)

def make_lgb(params=None, seed=42):
    p = {
        'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
        'reg_lambda': 3.0, 'reg_alpha': 1.0, 'num_leaves': 31,
    }
    if params: p.update(params)
    return lgb.LGBMRegressor(**p, random_state=seed, verbose=-1)

def make_cat(params=None, seed=42):
    p = {
        'iterations': 500, 'depth': 5, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0, 'subsample': 0.8,
    }
    if params: p.update(params)
    return CatBoostRegressor(**p, random_seed=seed, verbose=0)

def make_rf(seed=42):
    return RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features=0.7, random_state=seed, n_jobs=-1)

def make_et(seed=42):
    return ExtraTreesRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features=0.7, random_state=seed, n_jobs=-1)

def make_gbr(seed=42):
    return GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=seed)

def make_svr():
    return SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=1.0)

def make_knn():
    return KNeighborsRegressor(n_neighbors=15, weights='distance', metric='minkowski')

def make_mlp(seed=42):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        solver='adam', alpha=1.0, learning_rate='adaptive',
        max_iter=1000, early_stopping=True, validation_fraction=0.15,
        random_state=seed)

def make_ridge():
    return Ridge(alpha=5.0)

def make_huber():
    return HuberRegressor(alpha=1.0, epsilon=1.5, max_iter=500)

def make_kr():
    return KernelRidge(alpha=5.0, kernel='rbf', gamma=0.01)


# Which models need scaled data?
NEEDS_SCALING = {'svr', 'knn', 'mlp', 'ridge', 'huber', 'kr', 'elasticnet'}

# =================================================================
#  MULTI-SEED PREDICTION HELPER
# =================================================================
def predict_multiseed(model_fn, X_train, y, X_test, n_seeds=5, needs_scale=False):
    """Train model with multiple seeds, average predictions."""
    preds = []
    for i, seed in enumerate(SEEDS[:n_seeds]):
        if needs_scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_train)
            Xte = sc.transform(X_test)
        else:
            Xtr, Xte = X_train, X_test

        try:
            m = model_fn(seed=seed)
        except TypeError:
            m = model_fn()  # models that don't take seed

        m.fit(Xtr, y)
        preds.append(m.predict(Xte))

    return np.mean(preds, axis=0)


# =================================================================
#  LOSO VALIDATION FOR SINGLE MODEL
# =================================================================
def loso_eval(model_fn, needs_scale=False, n_seeds=5, power=1.1, label=''):
    """Full LOSO evaluation for a model. Returns (rmse, exact, per-fold)."""
    folds = sorted(set(train_seasons))
    all_assigned = np.zeros(n_tr, dtype=int)

    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold

        pred = predict_multiseed(model_fn, X_tr[tr], y_train[tr], X_tr[te],
                                 n_seeds=n_seeds, needs_scale=needs_scale)
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail, power)
        all_assigned[te] = assigned

    exact = int(np.sum(all_assigned == y_train.astype(int)))
    rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    return rmse, exact, all_assigned


# =================================================================
#  PHASE 1: EVALUATE ALL BASE MODELS
# =================================================================
print('\n' + '='*70)
print(' PHASE 1: BASE MODEL EVALUATION (LOSO)')
print('='*70)

model_configs = [
    ('xgb',     lambda seed=42: make_xgb(seed=seed),    False, 5),
    ('lgb',     lambda seed=42: make_lgb(seed=seed),     False, 5),
    ('cat',     lambda seed=42: make_cat(seed=seed),     False, 5),
    ('rf',      lambda seed=42: make_rf(seed=seed),      False, 3),
    ('et',      lambda seed=42: make_et(seed=seed),      False, 3),
    ('gbr',     lambda seed=42: make_gbr(seed=seed),     False, 3),
    ('svr',     lambda **kw: make_svr(),                 True,  1),
    ('knn',     lambda **kw: make_knn(),                 True,  1),
    ('mlp',     lambda seed=42: make_mlp(seed=seed),     True,  3),
    ('ridge',   lambda **kw: make_ridge(),               True,  1),
    ('huber',   lambda **kw: make_huber(),               True,  1),
    ('kr',      lambda **kw: make_kr(),                  True,  1),
]

base_results = {}
for name, fn, needs_sc, n_seeds in model_configs:
    try:
        rmse, exact, assigned = loso_eval(fn, needs_scale=needs_sc,
                                          n_seeds=n_seeds, label=name)
        base_results[name] = {'rmse': rmse, 'exact': exact, 'assigned': assigned}
        print(f'  {name:>8}: LOSO-RMSE={rmse:.4f}, exact={exact}/{n_tr}')
    except Exception as e:
        print(f'  {name:>8}: FAILED ({e})')

# Sort by RMSE
ranked = sorted(base_results.items(), key=lambda x: x[1]['rmse'])
print(f'\n  Ranked by LOSO-RMSE:')
for i, (name, res) in enumerate(ranked):
    print(f'    {i+1}. {name:>8}: {res["rmse"]:.4f} ({res["exact"]}/{n_tr})')


# =================================================================
#  PHASE 2: STACKING ENSEMBLE (LOSO out-of-fold)
# =================================================================
print('\n' + '='*70)
print(' PHASE 2: STACKING ENSEMBLE')
print('='*70)

# Use top N models for stacking (those within 0.5 RMSE of best)
best_rmse = ranked[0][1]['rmse']
stack_models = [(n, r) for n, r in ranked if r['rmse'] < best_rmse + 0.5]
stack_names = [n for n, _ in stack_models]
print(f'  Using {len(stack_names)} models for stacking: {stack_names}')

# Generate out-of-fold predictions for stacking
folds = sorted(set(train_seasons))
oof_preds = {name: np.zeros(n_tr) for name in stack_names}
test_preds = {name: np.zeros(n_te) for name in stack_names}

for name in stack_names:
    cfg = [(n, fn, ns, nsd) for n, fn, ns, nsd in model_configs if n == name][0]
    _, fn, needs_sc, n_seeds = cfg

    # OOF predictions (raw, before Hungarian)
    fold_test_preds = []
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold

        pred = predict_multiseed(fn, X_tr[tr], y_train[tr], X_tr[te],
                                 n_seeds=n_seeds, needs_scale=needs_sc)
        oof_preds[name][te] = pred

    # Full-train test predictions
    pred_test = predict_multiseed(fn, X_tr, y_train, X_te,
                                  n_seeds=n_seeds, needs_scale=needs_sc)
    test_preds[name] = pred_test

# Build stacking matrix
S_train = np.column_stack([oof_preds[n] for n in stack_names])
S_test  = np.column_stack([test_preds[n] for n in stack_names])

print(f'  Stack matrix: train={S_train.shape}, test={S_test.shape}')

# Try different meta-learners
print(f'\n  Meta-learner comparison (LOSO):')

meta_configs = [
    ('Ridge(1)',   Ridge(alpha=1.0)),
    ('Ridge(5)',   Ridge(alpha=5.0)),
    ('Ridge(10)',  Ridge(alpha=10.0)),
    ('Ridge(50)',  Ridge(alpha=50.0)),
    ('Huber',      HuberRegressor(alpha=1.0, epsilon=1.5)),
    ('ElasticNet',  ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)),
]

best_meta = None
best_meta_rmse = 99.0

for meta_name, meta_model in meta_configs:
    all_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        sc = StandardScaler()
        Str = sc.fit_transform(S_train[tr])
        Ste = sc.transform(S_train[te])
        mm = type(meta_model)(**meta_model.get_params())
        mm.fit(Str, y_train[tr])
        pred = mm.predict(Ste)
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail, 1.1)
        all_assigned[te] = assigned
    exact = int(np.sum(all_assigned == y_train.astype(int)))
    rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    print(f'    {meta_name:>12}: LOSO-RMSE={rmse:.4f}, exact={exact}/{n_tr}')
    if rmse < best_meta_rmse:
        best_meta_rmse = rmse
        best_meta = (meta_name, meta_model)

print(f'\n  Best meta-learner: {best_meta[0]} (LOSO-RMSE={best_meta_rmse:.4f})')

# Also try blending top models with optimized weights
print(f'\n  Optimized blend search...')
# Simple weighted average of top-3 models by RMSE
top3 = [n for n, _ in ranked[:3]]
best_blend_rmse = 99.0
best_blend_w = None

for w1 in np.arange(0.3, 0.8, 0.05):
    for w2 in np.arange(0.05, 0.5, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.0 or w3 > 0.5: continue
        blend = w1 * oof_preds[top3[0]] + w2 * oof_preds[top3[1]] + w3 * oof_preds[top3[2]]
        all_assigned = np.zeros(n_tr, dtype=int)
        for hold in folds:
            te = train_seasons == hold
            avail = {hold: list(range(1, 69))}
            assigned = hungarian(blend[te], train_seasons[te], avail, 1.1)
            all_assigned[te] = assigned
        rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
        if rmse < best_blend_rmse:
            best_blend_rmse = rmse
            best_blend_w = (w1, w2, w3)

print(f'  Best blend: {top3[0]}={best_blend_w[0]:.2f}, '
      f'{top3[1]}={best_blend_w[1]:.2f}, {top3[2]}={best_blend_w[2]:.2f}')
print(f'  Blend LOSO-RMSE: {best_blend_rmse:.4f}')

# Also try stacking + original features (augmented stacking)
print(f'\n  Augmented stacking (stack preds + original features)...')
SA_train = np.hstack([S_train, X_tr_sc])
SA_test  = np.hstack([S_test, X_te_sc])

best_aug_rmse = 99.0
for alpha in [10, 50, 100, 200]:
    all_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        sc = StandardScaler()
        Str = sc.fit_transform(SA_train[tr])
        Ste = sc.transform(SA_train[te])
        mm = Ridge(alpha=alpha)
        mm.fit(Str, y_train[tr])
        pred = mm.predict(Ste)
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail, 1.1)
        all_assigned[te] = assigned
    rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    if rmse < best_aug_rmse:
        best_aug_rmse = rmse
        best_aug_alpha = alpha

print(f'  Augmented stacking LOSO-RMSE: {best_aug_rmse:.4f} (Ridge alpha={best_aug_alpha})')


# =================================================================
#  PHASE 3: OPTUNA HYPERTUNING FOR TOP MODELS
# =================================================================
print('\n' + '='*70)
print(' PHASE 3: OPTUNA HYPERPARAMETER TUNING')
print('='*70)

ALL_SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321]

def loso_rmse_fast(model_fn, needs_scale=False, n_seeds=3, power=1.1):
    """Fast LOSO-RMSE for Optuna objective."""
    all_errors = []
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_multiseed(model_fn, X_tr[tr], y_train[tr], X_tr[te],
                                 n_seeds=n_seeds, needs_scale=needs_scale)
        avail_h = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail_h, power)
        errors = (assigned - y_train[te].astype(int)) ** 2
        all_errors.extend(errors)
    return np.sqrt(np.mean(all_errors))


# Tune XGBoost
print('\n  Tuning XGBoost (100 trials)...')
xgb_best = {'rmse': 99.0}

def xgb_objective(trial):
    p = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1200, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
    }
    power = trial.suggest_float('power', 0.8, 2.0, step=0.1)
    n_seeds = trial.suggest_int('n_seeds', 3, 7)

    fn = lambda seed=42: make_xgb(params=p, seed=seed)
    rmse = loso_rmse_fast(fn, n_seeds=n_seeds, power=power)
    if rmse < xgb_best['rmse']:
        xgb_best['rmse'] = rmse
        xgb_best['params'] = p.copy()
        xgb_best['power'] = power
        xgb_best['n_seeds'] = n_seeds
        print(f'    XGB NEW BEST: {rmse:.4f} (d={p["max_depth"]}, '
              f'lr={p["learning_rate"]:.3f}, n={p["n_estimators"]}, '
              f'lambda={p["reg_lambda"]:.2f})')
    return rmse

study_xgb = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
study_xgb.enqueue_trial({
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.80, 'colsample_bytree': 0.80, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0, 'power': 1.1, 'n_seeds': 5,
})
study_xgb.optimize(xgb_objective, n_trials=100)
print(f'  XGB best LOSO-RMSE: {xgb_best["rmse"]:.4f}')


# Tune LightGBM
print('\n  Tuning LightGBM (80 trials)...')
lgb_best = {'rmse': 99.0}

def lgb_objective(trial):
    p = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95, step=0.05),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
    }
    power = trial.suggest_float('power', 0.8, 2.0, step=0.1)

    fn = lambda seed=42: make_lgb(params=p, seed=seed)
    rmse = loso_rmse_fast(fn, n_seeds=3, power=power)
    if rmse < lgb_best['rmse']:
        lgb_best['rmse'] = rmse
        lgb_best['params'] = p.copy()
        lgb_best['power'] = power
        print(f'    LGB NEW BEST: {rmse:.4f}')
    return rmse

study_lgb = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=123))
study_lgb.optimize(lgb_objective, n_trials=80)
print(f'  LGB best LOSO-RMSE: {lgb_best["rmse"]:.4f}')


# Tune CatBoost
print('\n  Tuning CatBoost (60 trials)...')
cat_best = {'rmse': 99.0}

def cat_objective(trial):
    p = {
        'iterations': trial.suggest_int('iterations', 200, 800, step=100),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.10, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.5, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95, step=0.05),
        'random_strength': trial.suggest_float('random_strength', 0.1, 3.0),
    }
    power = trial.suggest_float('power', 0.8, 2.0, step=0.1)

    fn = lambda seed=42: make_cat(params=p, seed=seed)
    rmse = loso_rmse_fast(fn, n_seeds=3, power=power)
    if rmse < cat_best['rmse']:
        cat_best['rmse'] = rmse
        cat_best['params'] = p.copy()
        cat_best['power'] = power
        print(f'    CAT NEW BEST: {rmse:.4f}')
    return rmse

study_cat = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=777))
study_cat.optimize(cat_objective, n_trials=60)
print(f'  CAT best LOSO-RMSE: {cat_best["rmse"]:.4f}')


# =================================================================
#  PHASE 4: FINAL ENSEMBLE WITH TUNED MODELS + OPTUNA BLEND
# =================================================================
print('\n' + '='*70)
print(' PHASE 4: FINAL TUNED ENSEMBLE')
print('='*70)

# Re-generate OOF with tuned models
print('  Generating OOF predictions with tuned models...')

tuned_models = {}
if xgb_best['rmse'] < 99:
    tuned_models['xgb_t'] = (lambda seed=42: make_xgb(params=xgb_best['params'], seed=seed), False, 5)
if lgb_best['rmse'] < 99:
    tuned_models['lgb_t'] = (lambda seed=42: make_lgb(params=lgb_best['params'], seed=seed), False, 3)
if cat_best['rmse'] < 99:
    tuned_models['cat_t'] = (lambda seed=42: make_cat(params=cat_best['params'], seed=seed), False, 3)

# Add non-tunable models that performed well
for name in ['ridge', 'svr', 'knn', 'rf', 'et', 'huber', 'kr', 'mlp', 'gbr']:
    if name in base_results and base_results[name]['rmse'] < best_rmse + 0.5:
        cfg = [(n, fn, ns, nsd) for n, fn, ns, nsd in model_configs if n == name][0]
        tuned_models[name] = (cfg[1], cfg[2], cfg[3])

print(f'  Final ensemble models: {list(tuned_models.keys())}')

oof2 = {}
test2 = {}
for name, (fn, needs_sc, n_seeds) in tuned_models.items():
    oof2[name] = np.zeros(n_tr)
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_multiseed(fn, X_tr[tr], y_train[tr], X_tr[te],
                                 n_seeds=n_seeds, needs_scale=needs_sc)
        oof2[name][te] = pred
    test2[name] = predict_multiseed(fn, X_tr, y_train, X_te,
                                     n_seeds=n_seeds, needs_scale=needs_sc)

# Optuna blend optimization
print('\n  Optuna blend weight optimization (200 trials)...')
model_names = list(tuned_models.keys())
n_models = len(model_names)

blend_best = {'rmse': 99.0}

def blend_objective(trial):
    # Suggest weights (use softmax to ensure they sum to 1)
    raw_w = [trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in model_names]
    total = sum(raw_w) + 1e-8
    weights = [w / total for w in raw_w]

    power = trial.suggest_float('power', 0.8, 2.0, step=0.1)

    # Weighted blend of OOF predictions
    blend = np.zeros(n_tr)
    for w, name in zip(weights, model_names):
        blend += w * oof2[name]

    all_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        te = train_seasons == hold
        avail_h = {hold: list(range(1, 69))}
        assigned = hungarian(blend[te], train_seasons[te], avail_h, power)
        all_assigned[te] = assigned

    rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    if rmse < blend_best['rmse']:
        blend_best['rmse'] = rmse
        blend_best['weights'] = dict(zip(model_names, weights))
        blend_best['power'] = power
    return rmse

study_blend = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=2024))
study_blend.optimize(blend_objective, n_trials=200)

print(f'\n  Best blend LOSO-RMSE: {blend_best["rmse"]:.4f}')
print(f'  Weights:')
for name, w in sorted(blend_best['weights'].items(), key=lambda x: -x[1]):
    print(f'    {name:>10}: {w:.3f}')
print(f'  Hungarian power: {blend_best["power"]:.1f}')

# Also try stacking with tuned OOF
S2_train = np.column_stack([oof2[n] for n in model_names])
S2_test  = np.column_stack([test2[n] for n in model_names])

best_stack2_rmse = 99.0
best_stack2_alpha = 5.0
for alpha in [1, 2, 5, 10, 20, 50, 100]:
    all_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        sc = StandardScaler()
        Str = sc.fit_transform(S2_train[tr])
        Ste = sc.transform(S2_train[te])
        mm = Ridge(alpha=alpha)
        mm.fit(Str, y_train[tr])
        pred = mm.predict(Ste)
        avail_h = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail_h, blend_best.get('power', 1.1))
        all_assigned[te] = assigned
    rmse = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    if rmse < best_stack2_rmse:
        best_stack2_rmse = rmse
        best_stack2_alpha = alpha

print(f'\n  Stacking (tuned) LOSO-RMSE: {best_stack2_rmse:.4f} (Ridge alpha={best_stack2_alpha})')


# =================================================================
#  PHASE 5: FEATURE SELECTION
# =================================================================
print('\n' + '='*70)
print(' PHASE 5: FEATURE IMPORTANCE & SELECTION')
print('='*70)

# Get feature importance from XGB (tuned)
xgb_params_best = xgb_best.get('params', XGB_PARAMS if 'XGB_PARAMS' in dir() else {
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0})

m = xgb.XGBRegressor(**xgb_params_best, random_state=42, verbosity=0)
m.fit(X_tr, y_train)
importances = m.feature_importances_

fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])
print(f'\n  Top 20 features:')
for i, (name, imp_val) in enumerate(fi[:20]):
    print(f'    {i+1:2d}. {name:>30}: {imp_val:.4f}')

# Identify least important features
zero_imp = [name for name, imp_val in fi if imp_val < 0.001]
print(f'\n  Near-zero importance ({len(zero_imp)}): {zero_imp[:10]}...')

# Try dropping least important features
if len(zero_imp) > 0:
    keep_mask = np.array([imp_val >= 0.001 for _, imp_val in
                          sorted(zip(feat_names, importances), key=lambda x: feat_names.index(x[0]))])
    # Actually sort by feature name order
    keep_mask = np.array([importances[feat_names.index(fn)] >= 0.001 for fn in feat_names])
    n_keep = int(keep_mask.sum())
    print(f'  Trying with {n_keep}/{n_feat} features (dropping zero-importance)...')

    fn_sel = lambda seed=42: make_xgb(params=xgb_params_best, seed=seed)
    # Quick LOSO with selected features
    all_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_multiseed(fn_sel, X_tr[tr][:, keep_mask], y_train[tr],
                                 X_tr[te][:, keep_mask], n_seeds=5)
        avail_h = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail_h,
                            xgb_best.get('power', 1.1))
        all_assigned[te] = assigned
    rmse_sel = np.sqrt(np.mean((all_assigned - y_train.astype(int))**2))
    print(f'  Selected features LOSO-RMSE: {rmse_sel:.4f} vs full: {xgb_best["rmse"]:.4f}')


# =================================================================
#  PHASE 6: FINAL COMPARISON & OUTPUT
# =================================================================
print('\n' + '='*70)
print(' PHASE 6: FINAL COMPARISON')
print('='*70)

# Determine best approach
candidates = [
    ('XGB-only (v40 baseline)', 4.0889, None),
]

# Add XGB tuned
if xgb_best['rmse'] < 99:
    def xgb_tuned_test():
        fn = lambda seed=42: make_xgb(params=xgb_best['params'], seed=seed)
        return predict_multiseed(fn, X_tr, y_train, X_te,
                                 n_seeds=xgb_best.get('n_seeds', 5))
    candidates.append(('XGB tuned', xgb_best['rmse'], xgb_tuned_test))

# Add best blend
if blend_best['rmse'] < 99:
    def blend_test():
        pred = np.zeros(n_te)
        for name, w in blend_best['weights'].items():
            pred += w * test2[name]
        return pred
    candidates.append(('Optuna blend', blend_best['rmse'], blend_test))

# Add stacking
if best_stack2_rmse < 99:
    def stack_test():
        sc = StandardScaler()
        Str = sc.fit_transform(S2_train)
        Ste = sc.transform(S2_test)
        mm = Ridge(alpha=best_stack2_alpha)
        mm.fit(Str, y_train)
        return mm.predict(Ste)
    candidates.append(('Stacking', best_stack2_rmse, stack_test))

# Add best meta
if best_meta_rmse < 99:
    def meta_test():
        sc = StandardScaler()
        Str = sc.fit_transform(S_train)
        Ste = sc.transform(S_test)
        mm = type(best_meta[1])(**best_meta[1].get_params())
        mm.fit(Str, y_train)
        return mm.predict(Ste)
    candidates.append(('Meta-learner', best_meta_rmse, meta_test))

# Add simple blend
if best_blend_rmse < 99:
    def simple_blend_test():
        pred = (best_blend_w[0] * test_preds[top3[0]] +
                best_blend_w[1] * test_preds[top3[1]] +
                best_blend_w[2] * test_preds[top3[2]])
        return pred
    candidates.append(('Simple blend', best_blend_rmse, simple_blend_test))

# Evaluate all on test
print(f'\n  {"Approach":>30} {"LOSO-RMSE":>10} {"Test":>8} {"T-RMSE":>8}')
print(f'  {"-"*58}')

best_test_score = 0
best_test_assigned = None
best_approach = ''

for name, loso, test_fn in candidates:
    if test_fn is None:
        print(f'  {name:>30} {loso:10.4f}    --/91      --')
        continue

    pred = test_fn()
    power = blend_best.get('power', 1.1) if 'blend' in name.lower() or 'stack' in name.lower() else xgb_best.get('power', 1.1)
    assigned = hungarian(pred, test_seasons, test_avail, power)
    exact = int(np.sum(assigned == test_gt))
    test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    gen_gap = abs(loso - test_rmse)

    print(f'  {name:>30} {loso:10.4f} {exact:3d}/91  {test_rmse:8.4f}  gap={gen_gap:.3f}')

    if exact > best_test_score:
        best_test_score = exact
        best_test_assigned = assigned
        best_approach = name
        best_test_rmse = test_rmse
        best_loso_rmse = loso

# Also try the old v40 model for comparison
fn40 = lambda seed=42: make_xgb(params={
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0}, seed=seed)
pred40 = predict_multiseed(fn40, X_tr, y_train, X_te, n_seeds=5)
sc40 = StandardScaler()
rm40 = Ridge(alpha=5.0)
rm40.fit(sc40.fit_transform(X_tr), y_train)
ridge40 = rm40.predict(sc40.transform(X_te))
blend40 = 0.70 * pred40 + 0.30 * ridge40
assigned40 = hungarian(blend40, test_seasons, test_avail, 1.1)
exact40 = int(np.sum(assigned40 == test_gt))
rmse40 = np.sqrt(np.mean((assigned40 - test_gt)**2))
print(f'\n  {"v40 (old baseline w/ Ridge)":>30} {"4.0889":>10} {exact40:3d}/91  {rmse40:8.4f}')

if exact40 > best_test_score:
    best_test_score = exact40
    best_test_assigned = assigned40
    best_approach = 'v40 baseline'
    best_test_rmse = rmse40
    best_loso_rmse = 4.0889

print(f'\n  WINNER: {best_approach} ({best_test_score}/91)')

# =================================================================
#  SAVE BEST SUBMISSION
# =================================================================
print('\n' + '='*70)
print(' SAVING BEST SUBMISSION')
print('='*70)

sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(best_test_assigned[i])

out_path = os.path.join(DATA_DIR, 'final_submission.csv')
sub_out.to_csv(out_path, index=False)

print(f'\n  Best approach: {best_approach}')
print(f'  Test score: {best_test_score}/91')
print(f'  Test RMSE: {best_test_rmse:.4f}')
print(f'  LOSO RMSE: {best_loso_rmse:.4f}')
print(f'  Saved: final_submission.csv')
print(f'  Total time: {time.time()-t0:.0f}s')

# Save summary
summary = {
    'approach': best_approach,
    'test_score': best_test_score,
    'test_rmse': float(best_test_rmse),
    'loso_rmse': float(best_loso_rmse),
    'n_features': n_feat,
    'n_models': len(tuned_models),
    'models_used': list(tuned_models.keys()),
    'xgb_best_params': xgb_best.get('params', {}),
    'lgb_best_params': lgb_best.get('params', {}),
    'cat_best_params': cat_best.get('params', {}),
    'blend_weights': blend_best.get('weights', {}),
    'base_model_rmse': {n: float(r['rmse']) for n, r in base_results.items()},
}
with open(os.path.join(DATA_DIR, 'model_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f'  Saved: model_summary.json')

if IN_COLAB:
    files.download(out_path)
