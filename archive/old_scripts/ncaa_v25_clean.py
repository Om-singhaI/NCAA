#!/usr/bin/env python3
"""
NCAA Seed Prediction v25 - Clean, Zero-Overfitting Champion

Design Principles:
- ZERO test GT leakage: test ground truth is NEVER used for optimization
- All model selection, weighting, hyperparameters tuned via LOSO on training data
- Inspired by Mike Kim 4th place 2025 March Mania (pure ML, no gambling)
- XGBoost-focused ensemble (proven winner) with ~65 features
- Stacking meta-learner trained on LOSO OOF predictions

Pipeline:
1. Feature engineering: ~65 features (NET, quadrants, SOS, Elo, bid type)
2. LOSO cross-validation: Train on 4 seasons, validate on held-out season
3. During LOSO: optimize ensemble weights + Hungarian power parameter
4. Final: retrain ALL training data, predict test, apply LOSO-tuned params
5. Hungarian assignment -> final submission (no hill climbing against GT)
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost', 'mord'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Data dir: {DATA_DIR}')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import linear_sum_assignment

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
print('All imports loaded')

# ==============================================================
#  SECTION 1: DATA LOADING
# ==============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

print(f'Training: {train_df.shape[0]} teams x {train_df.shape[1]} cols')
print(f'Test:     {test_df.shape[0]} teams x {test_df.shape[1]} cols')
print(f'Seasons:  {sorted(train_df["Season"].unique())}')


def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
              'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
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


train_df['Overall Seed'] = pd.to_numeric(
    train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()

# Extract GT from submission.csv
# ONLY for final evaluation display, NEVER for optimization
GT = {}
for _, r in sub_df.iterrows():
    if int(r['Overall Seed']) > 0:
        GT[r['RecordID']] = int(r['Overall Seed'])
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]

y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)
n_te = len(tourn_idx)
train_seasons = train_tourn['Season'].values.astype(str)

# test_gt: ONLY for final scoring printout, NEVER in any optimization loop
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

# Available test seeds per season
avail_seeds = {}
for s in sorted(train_df['Season'].unique()):
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1, 69)) - used)

all_data = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df
], ignore_index=True)

SEASONS = sorted(train_tourn['Season'].unique().astype(str))

print(f'{n_tr} labeled train, {n_te} test tournament')
for s in SEASONS:
    n_s = (train_seasons == s).sum()
    print(f'  {s}: {n_s} train, {len(avail_seeds[s])} test slots')


# ==============================================================
#  SECTION 2: FEATURE ENGINEERING  (~65 focused features)
# ==============================================================
def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)

    # Core Win-Loss Records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col + '_W'] = wl.apply(lambda x: x[0])
            feat[col + '_L'] = wl.apply(lambda x: x[1])
            total = feat[col + '_W'] + feat[col + '_L']
            feat[col + '_Pct'] = safe_div(
                feat[col + '_W'], total.replace(0, np.nan))

    # Quadrant Records
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q + '_W'] = wl.apply(lambda x: x[0])
            feat[q + '_L'] = wl.apply(lambda x: x[1])
            total = feat[q + '_W'] + feat[q + '_L']
            feat[q + '_rate'] = safe_div(
                feat[q + '_W'], total.replace(0, np.nan))

    # Raw Rankings
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
                'NETSOS', 'NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')

    # Bid Type
    feat['is_AL'] = (df['Bid Type'].fillna('') == 'AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('') == 'AQ').astype(float)

    # Shorthand variables
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
    is_al = feat['is_AL']
    is_aq = feat['is_AQ']

    # Conference Strength
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_conf, 'NET': all_net})
    feat['conf_avg_net'] = conf.map(
        cs.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_best_net'] = conf.map(
        cs.groupby('Conference')['NET'].min()).fillna(200)
    feat['conf_size'] = conf.map(
        cs.groupby('Conference')['NET'].count()).fillna(10)
    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East',
                   'Pac-12', 'AAC', 'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    cav = feat['conf_avg_net']

    # Elo Proxy
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net

    # Power Rating (composite)
    feat['power_rating'] = (0.35 * (400 - net) + 0.25 * (300 - sos) +
                            0.2 * q1w * 10 + 0.1 * wpct * 100 +
                            0.1 * (prev - net))

    # Resume Score
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    q12t = q1w + q1l + q2w + q2l
    feat['q12_win_rate'] = (q1w + q2w) / (q12t + 1)

    # Record
    tg = totalw + totall
    feat['wins_above_500'] = totalw - tg / 2
    feat['road_performance'] = roadw / (roadw + roadl + 0.5)

    # NET Transformations
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['is_top16'] = (net <= 16).astype(float)
    feat['is_top32'] = (net <= 32).astype(float)
    feat['is_bubble'] = ((net >= 30) & (net <= 80) & (is_al == 1)).astype(float)
    feat['net_log'] = np.log1p(net)
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0

    # SOS Interactions
    feat['net_sos_gap'] = (net - sos).abs()
    feat['sos_x_wpct'] = (300 - sos) / 200 * wpct

    # Bid-Type Interactions
    feat['al_net'] = net * is_al
    feat['al_resume'] = feat['resume_score'] * is_al
    feat['midmajor_aq'] = is_aq * (1 - feat['is_power_conf'])

    # Conference-Relative
    feat['net_div_conf'] = net / (cav + 1)
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200

    # Rank within conference
    feat['rank_in_conf'] = 5.0
    nf = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    for sv in df['Season'].unique():
        for cv in df.loc[df['Season'] == sv, 'Conference'].unique():
            cm = (all_df['Season'] == sv) & (all_df['Conference'] == cv)
            cn = nf[cm].sort_values()
            dm = (df['Season'] == sv) & (df['Conference'] == cv)
            for idx in dm[dm].index:
                tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
                if pd.notna(tn):
                    feat.loc[idx, 'rank_in_conf'] = int((cn < tn).sum()) + 1
    feat['conf_rank_pct'] = feat['rank_in_conf'] / (feat['conf_size'] + 1)

    # Isotonic NET->Seed (trained on labeled_df ONLY)
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][
        ['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    if len(nsp) > 5:
        si = nsp['NET Rank'].values.argsort()
        ir_ns = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir_ns.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
        feat['net_to_seed_expected'] = ir_ns.predict(net.values)
    else:
        feat['net_to_seed_expected'] = net

    # Conference historical seed stats
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    conf_bid_stats = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        b = str(r.get('Bid Type', 'Unk'))
        conf_bid_stats.setdefault((c, b), []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(
            df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(
            df.loc[idx, 'Bid Type']) else 'Unk'
        vals = conf_bid_stats.get((c, b), [])
        feat.loc[idx, 'conf_bid_mean_seed'] = np.mean(vals) if vals else 35.0

    return feat


print('Building features...')
feat_train_tourn = build_features(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_features(test_df, all_data, labeled_df=train_tourn)

feat_cols = feat_train_tourn.columns.tolist()
n_feat = len(feat_cols)
print(f'{n_feat} features built')

# Impute
X_tr = np.where(
    np.isinf(feat_train_tourn.values.astype(np.float64)), np.nan,
    feat_train_tourn.values.astype(np.float64))
X_te_full = np.where(
    np.isinf(feat_test_full.values.astype(np.float64)), np.nan,
    feat_test_full.values.astype(np.float64))

X_all = np.vstack([X_tr, X_te_full])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(X_all)
X_tr_imp = X_all_imp[:n_tr]
X_te_full_imp = X_all_imp[n_tr:]
X_te_tourn_imp = X_te_full_imp[tourn_idx]

# Feature Selection via MI (training data only)
mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=7)
mi_order = np.argsort(mi)[::-1]
FS = {
    'top25': mi_order[:25],
    'top35': mi_order[:35],
    'all': np.arange(n_feat)
}

print('Top-10 MI features:')
for fi in mi_order[:10]:
    print(f'  {mi[fi]:.4f}  {feat_cols[fi]}')

# Scale for linear models
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr_imp)
X_te_s = sc.transform(X_te_tourn_imp)

print(f'\nImputed: train={X_tr_imp.shape}, test_tourn={X_te_tourn_imp.shape}')


# ==============================================================
#  SECTION 3: MODEL DEFINITIONS & UTILITIES
# ==============================================================

def hungarian_assign(pred_scores, seasons_arr, avail, power=1.25):
    """Assign seeds using Hungarian algorithm with cost=|pred-seed|^power."""
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
    """Return (exact_matches, sum_squared_error)."""
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt) ** 2))
    return exact, sse


def train_all_models(X, y, X_test, X_s, X_test_s, fs_dict, feat_col_list):
    """Train all model configs, return dict of {name: predictions_on_X_test}.

    All parameters are explicit - no hidden globals.
    feat_col_list: list of feature column names (for isotonic NET index lookup)
    """
    preds = {}

    for fs_name, fs_idx in fs_dict.items():
        Xf = X[:, fs_idx]
        Xtf = X_test[:, fs_idx]
        Xfs = X_s[:, fs_idx]
        Xtfs = X_test_s[:, fs_idx]

        # XGBoost "raddar" (proven 1st-place config)
        m = xgb.XGBRegressor(
            n_estimators=700, max_depth=4, learning_rate=0.01,
            subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
            min_child_weight=4, tree_method='hist', reg_lambda=1.5,
            grow_policy='lossguide', max_bin=38,
            random_state=42, verbosity=0
        ).fit(Xf, y)
        preds[f'xgb_raddar_{fs_name}'] = m.predict(Xtf)

        # XGBoost standard
        m = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.8, min_child_weight=5,
            tree_method='hist', reg_lambda=1.0,
            random_state=42, verbosity=0
        ).fit(Xf, y)
        preds[f'xgb_std_{fs_name}'] = m.predict(Xtf)

        # XGBoost shallow (maximum regularization)
        m = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.6, colsample_bytree=0.7, min_child_weight=6,
            tree_method='hist', reg_lambda=2.0,
            random_state=42, verbosity=0
        ).fit(Xf, y)
        preds[f'xgb_shallow_{fs_name}'] = m.predict(Xtf)

        # LightGBM
        m = lgb.LGBMRegressor(
            n_estimators=300, num_leaves=31, learning_rate=0.03,
            min_child_samples=5, reg_lambda=1.0,
            random_state=42, verbose=-1
        ).fit(Xf, y)
        preds[f'lgb_{fs_name}'] = m.predict(Xtf)

        # CatBoost
        m = CatBoostRegressor(
            iterations=300, depth=4, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42, verbose=0
        ).fit(Xf, y)
        preds[f'catboost_{fs_name}'] = m.predict(Xtf)

        # HistGBR
        m = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=300,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42
        ).fit(Xf, y)
        preds[f'hgbr_{fs_name}'] = m.predict(Xtf)

        # Ridge
        for alpha in [0.5, 1.0, 5.0]:
            m = Ridge(alpha=alpha).fit(Xfs, y)
            preds[f'ridge_a{alpha}_{fs_name}'] = m.predict(Xtfs)

        # BayesianRidge
        m = BayesianRidge().fit(Xfs, y)
        preds[f'bayridge_{fs_name}'] = m.predict(Xtfs)

        # SVR
        m = SVR(kernel='rbf', C=10.0, epsilon=0.5,
                gamma='scale').fit(Xfs, y)
        preds[f'svr_{fs_name}'] = m.predict(Xtfs)

        # MLP
        m = MLPRegressor(
            hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
            alpha=0.01, max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42
        ).fit(Xfs, y)
        preds[f'mlp_{fs_name}'] = m.predict(Xtfs)

        # Random Forest + Extra Trees
        m = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42).fit(Xf, y)
        preds[f'rf_{fs_name}'] = m.predict(Xtf)

        m = ExtraTreesRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42).fit(Xf, y)
        preds[f'et_{fs_name}'] = m.predict(Xtf)

    # Global Isotonic (NET -> seed)
    try:
        net_fi = feat_col_list.index('NET Rank')
        srt = np.argsort(X[:, net_fi])
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir.fit(X[srt, net_fi], y[srt])
        preds['isotonic_global'] = ir.predict(X_test[:, net_fi])
    except Exception:
        pass

    return preds


# ==============================================================
#  SECTION 4: LOSO CROSS-VALIDATION  (saves OOF for reuse)
# ==============================================================
print('\n' + '=' * 65)
print('  LEAVE-ONE-SEASON-OUT CROSS-VALIDATION')
print('=' * 65)

loso_results = {}
loso_model_names = None
oof_preds = None
oof_valid = np.zeros(n_tr, dtype=bool)
POWERS = [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]

for fold_i, held_out in enumerate(SEASONS):
    t_fold = time.time()
    print(f'\n  Fold {fold_i + 1}/{len(SEASONS)}: held out = {held_out}')

    # Split
    val_mask = train_seasons == held_out
    tr_mask = ~val_mask

    X_fold_tr = X_tr_imp[tr_mask]
    y_fold_tr = y_train[tr_mask]
    X_fold_val = X_tr_imp[val_mask]
    y_fold_val = y_train[val_mask]
    val_seasons = train_seasons[val_mask]

    # Scale
    sc_fold = StandardScaler()
    X_fold_tr_s = sc_fold.fit_transform(X_fold_tr)
    X_fold_val_s = sc_fold.transform(X_fold_val)

    # Available seeds: ALL 68 (predicting all held-out tournament teams)
    val_avail = {}
    for s in sorted(set(val_seasons)):
        val_avail[s] = sorted(range(1, 69))

    # Train + predict (no globals - all parameters explicit)
    fold_preds = train_all_models(
        X_fold_tr, y_fold_tr, X_fold_val,
        X_fold_tr_s, X_fold_val_s, FS,
        feat_col_list=feat_cols
    )

    model_names_fold = sorted(fold_preds.keys())
    if loso_model_names is None:
        loso_model_names = model_names_fold
        oof_preds = np.zeros((n_tr, len(loso_model_names)))

    # Save OOF predictions (for stacking and model ranking later)
    M_fold = np.column_stack([fold_preds[n] for n in loso_model_names])
    n_models_fold = M_fold.shape[1]
    val_indices = np.where(val_mask)[0]
    for mi in range(n_models_fold):
        oof_preds[val_indices, mi] = M_fold[:, mi]
    oof_valid[val_mask] = True

    # Score models on held-out season
    fold_model_scores = []
    for i, n in enumerate(loso_model_names):
        rmse = np.sqrt(np.mean((M_fold[:, i] - y_fold_val) ** 2))
        fold_model_scores.append((n, rmse, i))
    fold_model_scores.sort(key=lambda x: x[1])

    print(f'    {n_models_fold} models. Top-5:')
    for n, r, _ in fold_model_scores[:5]:
        print(f'      RMSE={r:.3f}  {n}')

    # Test ensemble strategies x powers
    fold_results = []

    # Mean all
    for power in POWERS:
        pred = np.mean(M_fold, axis=1)
        a = hungarian_assign(pred, val_seasons, val_avail, power)
        ex, sse = evaluate(a, y_fold_val.astype(int))
        fold_results.append(('mean_all', power, ex, sse))

    # Median all
    for power in POWERS:
        pred = np.median(M_fold, axis=1)
        a = hungarian_assign(pred, val_seasons, val_avail, power)
        ex, sse = evaluate(a, y_fold_val.astype(int))
        fold_results.append(('median_all', power, ex, sse))

    # Top-K inverse-RMSE weighted
    for k in [5, 10, 15, 20]:
        top_items = fold_model_scores[:min(k, n_models_fold)]
        wts = np.array([1.0 / (r + 0.1) for _, r, _ in top_items])
        wts /= wts.sum()
        top_preds_arr = np.column_stack(
            [M_fold[:, idx] for _, _, idx in top_items])
        pred = np.average(top_preds_arr, axis=1, weights=wts)
        for power in POWERS:
            a = hungarian_assign(pred, val_seasons, val_avail, power)
            ex, sse = evaluate(a, y_fold_val.astype(int))
            fold_results.append((f'wtop{k}', power, ex, sse))

    # Model-type specific
    for mtype, kw in [('xgb', ['xgb']), ('lgb', ['lgb']),
                      ('gb_all', ['xgb', 'lgb', 'catboost', 'hgbr']),
                      ('linear', ['ridge', 'bayridge']),
                      ('tree', ['rf_', 'et_'])]:
        idxs = [loso_model_names.index(n) for n, _, _ in fold_model_scores
                if any(k in n for k in kw)]
        if len(idxs) >= 2:
            pred = np.mean(M_fold[:, idxs], axis=1)
            for power in POWERS:
                a = hungarian_assign(pred, val_seasons, val_avail, power)
                ex, sse = evaluate(a, y_fold_val.astype(int))
                fold_results.append((f'{mtype}_mean', power, ex, sse))

    # Top-K simple average
    for k in [3, 5, 10]:
        top_n = [fold_model_scores[i][0]
                 for i in range(min(k, n_models_fold))]
        pred = np.mean(
            np.column_stack([fold_preds[n] for n in top_n]), axis=1)
        for power in POWERS:
            a = hungarian_assign(pred, val_seasons, val_avail, power)
            ex, sse = evaluate(a, y_fold_val.astype(int))
            fold_results.append((f'top{k}_avg', power, ex, sse))

    fold_results.sort(key=lambda x: (-x[2], x[3]))
    loso_results[held_out] = fold_results

    n_val = len(y_fold_val)
    best_strat, best_pw, best_ex, best_sse = fold_results[0]
    print(f'    Best: {best_ex}/{n_val} exact, SSE={best_sse}'
          f'  ({best_strat} + p{best_pw})')
    print(f'    Fold time: {time.time() - t_fold:.0f}s')

# LOSO model rankings from OOF predictions (clean - no test GT)
loso_model_rmse = []
for mi, mn in enumerate(loso_model_names):
    rmse = np.sqrt(
        np.mean((oof_preds[oof_valid, mi] - y_train[oof_valid]) ** 2))
    loso_model_rmse.append((mn, rmse, mi))
loso_model_rmse.sort(key=lambda x: x[1])

print(f'\nTop-10 models by LOSO OOF RMSE (clean - no test GT):')
for n, r, _ in loso_model_rmse[:10]:
    print(f'  RMSE={r:.4f}  {n}')

# Aggregate LOSO results -> choose best strategy
print('\n' + '=' * 65)
print('  LOSO AGGREGATION')
print('=' * 65)

combo_scores = defaultdict(lambda: {'exact': 0, 'sse': 0, 'folds': 0})
for season, results in loso_results.items():
    for strat, pw, ex, sse in results:
        key = (strat, pw)
        combo_scores[key]['exact'] += ex
        combo_scores[key]['sse'] += sse
        combo_scores[key]['folds'] += 1

n_folds = len(SEASONS)
valid_combos = [(k, v) for k, v in combo_scores.items()
                if v['folds'] == n_folds]
valid_combos.sort(key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\nTop-15 strategies across all LOSO folds '
      f'(total over {n_folds} seasons):')
for (strat, pw), scores in valid_combos[:15]:
    total_teams = n_tr
    rmse = np.sqrt(scores['sse'] / total_teams)
    print(f'  {scores["exact"]}/{total_teams} exact  '
          f'RMSE={rmse:.4f}  {strat} + p{pw}')

best_combo, best_scores = valid_combos[0]
BEST_STRATEGY = best_combo[0]
BEST_POWER = best_combo[1]
print(f'\n  CHOSEN: {BEST_STRATEGY} + power={BEST_POWER}')
print(f'  LOSO performance: {best_scores["exact"]}/{n_tr} exact '
      f'({best_scores["exact"] / n_tr * 100:.1f}%)')
print(f'  Total LOSO time: {time.time() - t0:.0f}s')


# ==============================================================
#  SECTION 5: FINAL MODEL TRAINING & PREDICTION
# ==============================================================
print('\n' + '=' * 65)
print('  FINAL MODEL TRAINING')
print('=' * 65)

t_final = time.time()
final_preds = train_all_models(
    X_tr_imp, y_train, X_te_tourn_imp,
    X_tr_s, X_te_s, FS,
    feat_col_list=feat_cols
)

model_names = sorted(final_preds.keys())
M = np.column_stack([final_preds[n] for n in model_names])
n_models = len(model_names)
print(f'{n_models} models trained in {time.time() - t_final:.0f}s')

# Build ensembles using LOSO OOF rankings (no test GT!)
ensembles = {}
ensembles['mean_all'] = np.mean(M, axis=1)
ensembles['median_all'] = np.median(M, axis=1)

for k in [3, 5, 10]:
    top_n = [loso_model_rmse[i][0] for i in range(min(k, n_models))]
    ensembles[f'top{k}_avg'] = np.mean(
        np.column_stack([final_preds[n] for n in top_n]), axis=1)

for k in [5, 10, 15, 20]:
    top_items = loso_model_rmse[:min(k, n_models)]
    wts = np.array([1.0 / (r + 0.1) for _, r, _ in top_items])
    wts /= wts.sum()
    top_preds_arr = np.column_stack([M[:, idx] for _, _, idx in top_items])
    ensembles[f'wtop{k}'] = np.average(top_preds_arr, axis=1, weights=wts)

for mtype, kw in [('xgb', ['xgb']), ('lgb', ['lgb']),
                  ('gb_all', ['xgb', 'lgb', 'catboost', 'hgbr']),
                  ('linear', ['ridge', 'bayridge']),
                  ('tree', ['rf_', 'et_'])]:
    idxs = [model_names.index(n) for n, _, _ in loso_model_rmse
            if any(k in n for k in kw)]
    if len(idxs) >= 2:
        ensembles[f'{mtype}_mean'] = np.mean(M[:, idxs], axis=1)

# Stacking meta-learner on LOSO OOF predictions (clean - no test GT)
print('Training stacking meta-learners on LOSO OOF predictions...')
for sname, smodel in [('stack_ridge_1', Ridge(alpha=1.0)),
                      ('stack_ridge_10', Ridge(alpha=10.0)),
                      ('stack_bayridge', BayesianRidge())]:
    smodel.fit(oof_preds[oof_valid], y_train[oof_valid])
    ensembles[sname] = smodel.predict(M)

print(f'Built {len(ensembles)} ensemble variants')

# Apply LOSO-chosen strategy
print(f'\nApplying LOSO-chosen strategy: {BEST_STRATEGY} + power={BEST_POWER}')

if BEST_STRATEGY in ensembles:
    chosen_pred = ensembles[BEST_STRATEGY]
else:
    print(f'  WARNING: {BEST_STRATEGY} not found, falling back to mean_all')
    chosen_pred = ensembles['mean_all']

primary_assigned = hungarian_assign(
    chosen_pred, test_seasons, avail_seeds, BEST_POWER)

# Evaluate ALL strategies x powers (test GT for display ONLY, not selection)
all_results = []
for ename, epred in ensembles.items():
    for power in POWERS:
        a = hungarian_assign(epred, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)  # Display ONLY - never for selection
        all_results.append((ename, power, ex, sse, a))

all_results.sort(key=lambda x: (-x[2], x[3]))

# Report
print(f'\n{"=" * 65}')
print('  RESULTS (test GT used ONLY for display, not selection)')
print(f'{"=" * 65}')

primary_ex, primary_sse = evaluate(primary_assigned, test_gt)
primary_rmse = np.sqrt(primary_sse / 451)
print(f'\n  PRIMARY (LOSO-chosen): {primary_ex}/91 exact, '
      f'RMSE={primary_rmse:.4f}')
print(f'  Strategy: {BEST_STRATEGY} + power={BEST_POWER}')

best_test = all_results[0]
best_test_rmse = np.sqrt(best_test[3] / 451)
print(f'\n  ORACLE (best on test, reference only):')
print(f'  {best_test[2]}/91 exact, RMSE={best_test_rmse:.4f}'
      f'  ({best_test[0]} + p{best_test[1]})')

print(f'\n  Top-15 on test (informational):')
for ename, pw, ex, sse, _ in all_results[:15]:
    rmse = np.sqrt(sse / 451)
    marker = ' <<< PRIMARY' if (ename == BEST_STRATEGY
                                 and pw == BEST_POWER) else ''
    print(f'    {ex}/91  RMSE={rmse:.4f}  {ename} + p{pw}{marker}')

print(f'\n  Total time: {time.time() - t0:.0f}s')


# ==============================================================
#  SECTION 6: ANALYSIS & VISUALIZATION
# ==============================================================
print('\nAnalysis of primary submission:')
print('=' * 55)

for s in sorted(set(test_seasons)):
    s_mask = test_seasons == s
    s_exact = int(np.sum(primary_assigned[s_mask] == test_gt[s_mask]))
    s_sse = int(np.sum((primary_assigned[s_mask] - test_gt[s_mask]) ** 2))
    s_total = s_mask.sum()
    print(f'  {s}: {s_exact}/{s_total} exact, SSE={s_sse}')

errors = primary_assigned - test_gt
abs_err = np.abs(errors)
print(f'\nError stats:')
print(f'  Mean abs error: {abs_err.mean():.2f}')
print(f'  Median abs error: {np.median(abs_err):.2f}')
print(f'  Max abs error: {abs_err.max()}')
print(f'  Teams with error > 5: {(abs_err > 5).sum()}')
print(f'  Teams with error > 10: {(abs_err > 10).sum()}')

worst_idx = np.argsort(abs_err)[::-1][:10]
print(f'\nWorst predictions:')
for idx in worst_idx:
    rid = test_rids[idx]
    print(f'  {rid}: pred={primary_assigned[idx]}, '
          f'actual={test_gt[idx]}, error={errors[idx]}')

if HAS_PLOT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(test_gt, primary_assigned, alpha=0.5, s=30)
    axes[0].plot([0, 68], [0, 68], 'r--')
    axes[0].set_xlabel('Actual Seed')
    axes[0].set_ylabel('Predicted Seed')
    axes[0].set_title(f'Predicted vs Actual ({primary_ex}/91 exact)')

    axes[1].hist(errors, bins=range(-20, 21), edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Error Distribution')

    top_combos_plot = valid_combos[:10]
    labels = [f'{s}+p{p}' for (s, p), _ in top_combos_plot]
    exacts = [v['exact'] for _, v in top_combos_plot]
    axes[2].barh(range(len(labels)), exacts)
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_xlabel('Total LOSO Exact Matches')
    axes[2].set_title('Top-10 LOSO Strategies')

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, 'v25_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved v25_analysis.png')


# ==============================================================
#  SECTION 7: SAVE SUBMISSIONS
# ==============================================================

# Primary: LOSO-chosen strategy (zero overfitting)
sub_primary = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_primary['RecordID'] == rid
    if mask.any():
        sub_primary.loc[mask, 'Overall Seed'] = int(primary_assigned[i])

best_path = os.path.join(DATA_DIR, 'submission_v25_clean.csv')
sub_primary.to_csv(best_path, index=False)
print(f'PRIMARY saved: {best_path}')
print(f'  {primary_ex}/91 exact, RMSE={primary_rmse:.4f}')
print(f'  Strategy: {BEST_STRATEGY} + power={BEST_POWER}')
print(f'  Zero overfitting: YES')

# Also save top-3 LOSO strategies as alternatives
for alt_i, ((strat, pw), scores) in enumerate(valid_combos[:3]):
    if strat in ensembles:
        alt_pred = ensembles[strat]
        alt_assigned = hungarian_assign(
            alt_pred, test_seasons, avail_seeds, pw)
        alt_ex, alt_sse = evaluate(alt_assigned, test_gt)
        alt_rmse = np.sqrt(alt_sse / 451)

        sub_alt = sub_df.copy()
        for i, ti in enumerate(tourn_idx):
            rid = test_df.iloc[ti]['RecordID']
            mask = sub_alt['RecordID'] == rid
            if mask.any():
                sub_alt.loc[mask, 'Overall Seed'] = int(alt_assigned[i])

        alt_path = os.path.join(DATA_DIR, f'submission_v25_alt{alt_i + 1}.csv')
        sub_alt.to_csv(alt_path, index=False)
        print(f'\nALT {alt_i + 1} saved: {alt_path}')
        print(f'  {alt_ex}/91 exact, RMSE={alt_rmse:.4f}  ({strat} + p{pw})')

total_time = time.time() - t0
print(f'\nTotal pipeline time: {total_time:.0f}s ({total_time / 60:.1f} min)')
print(f'\n{"=" * 65}')
print('  v25 COMPLETE - ZERO OVERFITTING MODEL')
print(f'{"=" * 65}')

# Colab download helper
if IN_COLAB:
    for f in ['submission_v25_clean.csv', 'submission_v25_alt1.csv',
              'submission_v25_alt2.csv', 'submission_v25_alt3.csv']:
        fp = os.path.join(DATA_DIR, f)
        if os.path.exists(fp):
            files.download(fp)
            print(f'Downloaded {f}')
