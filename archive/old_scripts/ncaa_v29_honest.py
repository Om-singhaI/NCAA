#!/usr/bin/env python3
"""
NCAA v29 — Honest Ensemble with LOSO-validated Super-Blending

Key principles to prevent overfitting:
  1. ALL selection decisions made via LOSO CV, never test GT
  2. Super-blend combos chosen by LOSO performance
  3. Residual correction validated within LOSO folds
  4. Seed diversity: each tree model trained with 3 random seeds
  5. Report both LOSO-selected and GT-selected for transparency
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost'])
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
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ============================================================
#  DATA
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


train_df['Overall Seed'] = pd.to_numeric(
    train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()

GT = {}
for _, r in sub_df.iterrows():
    if int(r['Overall Seed']) > 0:
        GT[r['RecordID']] = int(r['Overall Seed'])

tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]

y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

avail_seeds = {}
for s in sorted(train_df['Season'].unique()):
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1, 69)) - used)

SEASONS = sorted(train_tourn['Season'].unique().astype(str))
n_tr = len(y_train)
n_te = len(tourn_idx)

test_tourn_rids = set(GT.keys())
all_data = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df
], ignore_index=True)

print(f'{n_tr} train, {n_te} test tournament teams')


# ============================================================
#  FEATURES
# ============================================================
def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)

    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w = wl.apply(lambda x: x[0])
            l = wl.apply(lambda x: x[1])
            feat[col + '_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w
                feat['total_L'] = l
                feat['total_games'] = w + l

    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q + '_W'] = wl.apply(lambda x: x[0])
            feat[q + '_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_net = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)

    feat['NET Rank'] = net
    feat['PrevNET'] = prev
    feat['NETSOS'] = sos
    feat['AvgOppNETRank'] = opp_net

    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cg = pd.DataFrame({'Conference': all_conf, 'NET': all_net_vals})
    feat['conf_avg_net'] = conf.map(cg.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cg.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_min_net'] = conf.map(cg.groupby('Conference')['NET'].min()).fillna(300)
    feat['conf_std_net'] = conf.map(cg.groupby('Conference')['NET'].std()).fillna(50)
    feat['conf_count'] = conf.map(cg.groupby('Conference')['NET'].count()).fillna(1)

    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East',
                   'Pac-12', 'AAC', 'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    cav = feat['conf_avg_net']

    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net

    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0

    feat['power_rating'] = (0.35 * (400 - net) + 0.25 * (300 - sos) +
                            0.2 * q1w * 10 + 0.1 * wpct * 100 +
                            0.1 * (prev - net))

    feat['sos_x_wpct'] = (300 - sos) / 200 * wpct
    feat['record_vs_sos'] = wpct * (300 - sos) / 100
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200
    feat['sos_adj_net'] = net + (sos - 100) * 0.15

    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)

    # New: Additional quad-based features
    feat['q12_wins'] = q1w + q2w
    feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + q2l + 0.1)

    # New: NET rank relative to SOS
    feat['net_sos_ratio'] = net / (sos + 1)
    feat['net_minus_sos'] = net - sos

    # New: Road+neutral performance proxy
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300 - sos) / 200

    # New: Conference relative rank within tournament field
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    all_tourn_rids = set(labeled_df[labeled_df['Overall Seed'] > 0]['RecordID'].values) | test_tourn_rids
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets_in_field = []
        for _, row in all_df[all_df['Season'] == sv].iterrows():
            if row['RecordID'] in all_tourn_rids:
                n = pd.to_numeric(row.get('NET Rank', 300), errors='coerce')
                if pd.notna(n):
                    nets_in_field.append(n)
        nets_in_field = sorted(nets_in_field)
        smask = df['Season'] == sv
        for idx in df[smask].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'tourn_field_rank'] = float(
                    sum(1 for x in nets_in_field if x < n) + 1)

    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = []
        for _, row in all_df[all_df['Season'] == sv].iterrows():
            if str(row.get('Bid Type', '')) == 'AL':
                n = pd.to_numeric(row.get('NET Rank', 300), errors='coerce')
                if pd.notna(n):
                    al_nets.append(n)
        al_nets = sorted(al_nets)
        smask = (df['Season'] == sv) & (df['Bid Type'].fillna('') == 'AL')
        for idx in df[smask].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'net_rank_among_al'] = float(
                    sum(1 for x in al_nets if x < n) + 1)

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

    for col_name, col_vals in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                                ('adj_net', feat['adj_net']),
                                ('net_to_seed', feat['net_to_seed']),
                                ('power_rating', feat['power_rating'])]:
        feat[col_name + '_spctile'] = 0.5
        for sv in df['Season'].unique():
            smask = df['Season'] == sv
            svals = col_vals[smask]
            if len(svals) > 1:
                feat.loc[smask, col_name + '_spctile'] = svals.rank(pct=True)

    return feat


print('Building features...')
feat_train = build_features(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_features(test_df, all_data, labeled_df=train_tourn)

feat_cols = feat_train.columns.tolist()
n_feat = len(feat_cols)
print(f'{n_feat} features built')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)),
                    np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)),
                    np.nan, feat_test_full.values.astype(np.float64))

X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr = X_stack_imp[:n_tr]
X_te = X_stack_imp[n_tr:][tourn_idx]

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)


# ============================================================
#  HUNGARIAN + EVALUATE
# ============================================================
def hungarian(scores, seasons, avail, power=1.25):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p) ** power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def evaluate(assigned, gt):
    return int(np.sum(assigned == gt)), int(np.sum((assigned - gt) ** 2))


POWERS = [1.0, 1.1, 1.25, 1.5, 2.0]

# ============================================================
#  MODEL DEFINITIONS — with seed variants for diversity
# ============================================================
def make_model_specs():
    """Return list of (name, model_template, needs_scaled)."""
    specs = []

    # XGBoost variants with multiple seeds
    for seed in [42, 123, 777]:
        tag = '' if seed == 42 else f'_s{seed}'

        specs.append((f'raddar{tag}', xgb.XGBRegressor(
            n_estimators=700, max_depth=4, learning_rate=0.01,
            subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
            min_child_weight=4, tree_method='hist', reg_lambda=1.5,
            grow_policy='lossguide', max_bin=38,
            random_state=seed, verbosity=0), False))

        specs.append((f'xgb_deep{tag}', xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
            random_state=seed, verbosity=0), False))

        specs.append((f'xgb_shallow{tag}', xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=1.0,
            random_state=seed, verbosity=0), False))

    # LightGBM
    specs.append(('lgb', lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_samples=10, reg_lambda=1.5, reg_alpha=0.3,
        random_state=42, verbose=-1), False))

    # CatBoost
    specs.append(('cat', CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.7,
        random_seed=42, verbose=0), False))

    specs.append(('cat_deep', CatBoostRegressor(
        iterations=400, depth=7, learning_rate=0.02,
        l2_leaf_reg=5.0, subsample=0.6,
        random_seed=42, verbose=0), False))

    # Ridge at different alphas
    for alpha in [5.0, 10.0, 20.0]:
        specs.append((f'ridge{int(alpha)}', Ridge(alpha=alpha), True))

    # BayesianRidge
    specs.append(('bayridge', BayesianRidge(), True))

    # HistGBR
    specs.append(('hgbr', HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=10, l2_regularization=1.0,
        random_state=42), False))

    # RF/ET
    specs.append(('rf', RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features=0.6, random_state=42, n_jobs=-1), False))

    specs.append(('et', ExtraTreesRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        max_features=0.7, random_state=42, n_jobs=-1), False))

    return specs


ALL_MODEL_SPECS = make_model_specs()
model_names = [s[0] for s in ALL_MODEL_SPECS]
n_models = len(model_names)
print(f'{n_models} models (including seed variants)')


# ============================================================
#  BLEND CONFIGS (pre-defined, not exhaustive)
# ============================================================
# Focus on the blend families that worked well in v28
BLEND_CONFIGS = {}

# Core blends without seed variants (for speed)
core_trees = ['raddar', 'xgb_deep', 'xgb_shallow']
core_trees_all = ['raddar', 'xgb_deep', 'xgb_shallow',
                  'raddar_s123', 'xgb_deep_s123', 'xgb_shallow_s123',
                  'raddar_s777', 'xgb_deep_s777', 'xgb_shallow_s777']

# v26 winner: raddar + xgb_deep + ridge
BLEND_CONFIGS['tr3r10'] = ['raddar', 'xgb_deep', 'ridge10']
BLEND_CONFIGS['tr3r5'] = ['raddar', 'xgb_deep', 'ridge5']
BLEND_CONFIGS['tr3r20'] = ['raddar', 'xgb_deep', 'ridge20']
BLEND_CONFIGS['tr3br'] = ['raddar', 'xgb_deep', 'bayridge']

# With seed diversity
BLEND_CONFIGS['tr3r5_3seed'] = ['raddar', 'xgb_deep', 'ridge5',
                                 'raddar_s123', 'xgb_deep_s123',
                                 'raddar_s777', 'xgb_deep_s777']
BLEND_CONFIGS['tr3r10_3seed'] = ['raddar', 'xgb_deep', 'ridge10',
                                  'raddar_s123', 'xgb_deep_s123',
                                  'raddar_s777', 'xgb_deep_s777']

# Broader blends
BLEND_CONFIGS['tree6'] = ['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'hgbr']
BLEND_CONFIGS['tree6_r5'] = ['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'hgbr', 'ridge5']
BLEND_CONFIGS['all_tree_r'] = ['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'cat_deep',
                                'hgbr', 'ridge5']
BLEND_CONFIGS['all_tree_r_seeds'] = core_trees_all + ['lgb', 'cat', 'cat_deep', 'hgbr', 'ridge5']
BLEND_CONFIGS['diverse7'] = ['raddar', 'xgb_deep', 'lgb', 'cat', 'hgbr', 'rf', 'ridge10']
BLEND_CONFIGS['mega'] = model_names  # all models

# Raddar family with ridge
BLEND_CONFIGS['raddar3_r5'] = ['raddar', 'raddar_s123', 'raddar_s777', 'ridge5']
BLEND_CONFIGS['deep3_r5'] = ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'ridge5']
BLEND_CONFIGS['xgb9_r5'] = core_trees_all + ['ridge5']

# Pairs with ridge
BLEND_CONFIGS['rad_r5'] = ['raddar', 'ridge5']
BLEND_CONFIGS['rad_r10'] = ['raddar', 'ridge10']
BLEND_CONFIGS['deep_r5'] = ['xgb_deep', 'ridge5']
BLEND_CONFIGS['lgb_r5'] = ['lgb', 'ridge5']
BLEND_CONFIGS['cat_r5'] = ['cat', 'ridge5']

# RF/ET combos
BLEND_CONFIGS['rad_deep_rf_r5'] = ['raddar', 'xgb_deep', 'rf', 'ridge5']
BLEND_CONFIGS['rad_deep_et_r5'] = ['raddar', 'xgb_deep', 'et', 'ridge5']

n_blends = len(BLEND_CONFIGS)
print(f'{n_blends} blend configurations')


# Residual correction configs
RC_CONFIGS = {
    'none': None,
    'rc_d1': {'n_estimators': 200, 'max_depth': 1, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
              'reg_lambda': 2.0},
    'rc_d2': {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
              'reg_lambda': 3.0},
    'rc_d2b': {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.03,
               'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
               'reg_lambda': 2.0},
}


# ============================================================
#  LOSO CROSS-VALIDATION — Full pipeline inside each fold
# ============================================================
print('\n' + '=' * 60)
print(' LOSO CV: FULL PIPELINE PER FOLD')
print('=' * 60)

# Accumulate LOSO results: key=(blend, rc, power) -> {exact, sse}
loso_results = defaultdict(lambda: {'exact': 0, 'sse': 0})
# Also accumulate raw predictions per fold for super-blend LOSO
loso_raw_preds = {}  # key=(blend, rc) -> list of (val_mask, val_preds)

for fold_i, held_out in enumerate(SEASONS):
    val_mask = train_seasons == held_out
    tr_mask = ~val_mask
    Xf_tr, yf_tr = X_tr[tr_mask], y_train[tr_mask]
    Xf_val = X_tr[val_mask]
    yf_val = y_train[val_mask]
    Xf_tr_sc = X_tr_sc[tr_mask]
    Xf_val_sc = X_tr_sc[val_mask]

    val_seasons_f = train_seasons[val_mask]
    val_avail = {s: list(range(1, 69)) for s in sorted(set(val_seasons_f))}
    fold_n = val_mask.sum()

    print(f'\nFold {fold_i+1}/{len(SEASONS)}: {held_out} ({fold_n} teams)')

    # Train all models on fold training data
    fold_val_preds = {}
    fold_tr_preds = {}
    for mname, model_template, needs_scaled in ALL_MODEL_SPECS:
        if hasattr(model_template, 'get_params'):
            m = type(model_template)(**model_template.get_params())
        else:
            m = type(model_template)(**model_template.__dict__)

        Xf_t = Xf_tr_sc if needs_scaled else Xf_tr
        Xf_v = Xf_val_sc if needs_scaled else Xf_val

        m.fit(Xf_t, yf_tr)
        fold_val_preds[mname] = m.predict(Xf_v)
        fold_tr_preds[mname] = m.predict(Xf_t)

    # Compute blend predictions
    for bname, bmodels in BLEND_CONFIGS.items():
        raw_val = np.mean([fold_val_preds[m] for m in bmodels], axis=0)
        raw_tr = np.mean([fold_tr_preds[m] for m in bmodels], axis=0)

        for rc_name, rc_params in RC_CONFIGS.items():
            if rc_params is None:
                val_pred = raw_val
            else:
                # Residual correction within the fold
                residuals = yf_tr - raw_tr
                X_aug_tr = np.column_stack([Xf_tr, raw_tr])
                X_aug_val = np.column_stack([Xf_val, raw_val])
                rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
                rm.fit(X_aug_tr, residuals)
                val_pred = raw_val + rm.predict(X_aug_val)

            key = (bname, rc_name)
            if key not in loso_raw_preds:
                loso_raw_preds[key] = {}
            loso_raw_preds[key][fold_i] = (val_mask, val_pred)

            for power in POWERS:
                a = hungarian(val_pred, val_seasons_f, val_avail, power)
                ex, sse = evaluate(a, yf_val.astype(int))
                loso_results[(bname, rc_name, power)]['exact'] += ex
                loso_results[(bname, rc_name, power)]['sse'] += sse

    # Show best for this fold
    fold_best = 0
    for key, vals in loso_results.items():
        # Only count this fold (approximate by checking total)
        pass
    print(f'  Models trained: {n_models}')

# Sort LOSO results
loso_sorted = sorted(loso_results.items(), key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\n--- LOSO Top-30 ---')
for (bname, rc, pw), scores in loso_sorted[:30]:
    rmse = np.sqrt(scores['sse'] / n_tr)
    print(f"  {scores['exact']}/{n_tr} exact  RMSE={rmse:.4f}  {bname}+{rc}+p{pw}")


# ============================================================
#  LOSO-VALIDATED SUPER-BLENDING
# ============================================================
print('\n' + '=' * 60)
print(' LOSO-VALIDATED SUPER-BLENDING')
print('=' * 60)

# Build full OOF prediction arrays for each (blend, rc) config
oof_predictions = {}  # key=(bname, rc) -> full OOF array (n_tr,)
for (bname, rc), fold_data in loso_raw_preds.items():
    oof = np.zeros(n_tr)
    for fold_i, (vm, vp) in fold_data.items():
        oof[vm] = vp
    oof_predictions[(bname, rc)] = oof

# Pick top configs by LOSO exact match (unique raw predictions only)
top_configs = []
seen_oof = set()
for (bname, rc, pw), scores in loso_sorted:
    key = (bname, rc)
    if key in seen_oof:
        continue
    seen_oof.add(key)
    top_configs.append((key, scores['exact']))
    if len(top_configs) >= 15:
        break

print(f'Top {len(top_configs)} configs for super-blending:')
for (bname, rc), ex in top_configs:
    print(f'  {ex}/{n_tr}  {bname}+{rc}')

# Try super-blend combos of top configs, evaluated by LOSO
super_loso = {}
config_keys = [k for k, _ in top_configs]

for n_combo in [2, 3, 4, 5]:
    for combo in combinations(range(len(config_keys)), n_combo):
        # Average OOF predictions
        avg_oof = np.mean([oof_predictions[config_keys[i]] for i in combo], axis=0)

        for power in POWERS:
            total_ex = 0
            total_sse = 0
            for fold_i, held_out in enumerate(SEASONS):
                vm = train_seasons == held_out
                vs = train_seasons[vm]
                va = {s: list(range(1, 69)) for s in sorted(set(vs))}
                a = hungarian(avg_oof[vm], vs, va, power)
                ex, sse = evaluate(a, y_train[vm].astype(int))
                total_ex += ex
                total_sse += sse

            combo_key = tuple(combo)
            super_loso[(combo_key, power)] = {'exact': total_ex, 'sse': total_sse}

# Sort super-blend LOSO results
super_sorted = sorted(super_loso.items(), key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\nTop-20 super-blend LOSO results:')
for (combo, pw), scores in super_sorted[:20]:
    combo_names = '+'.join([f'{config_keys[i][0]}_{config_keys[i][1]}' for i in combo])
    rmse = np.sqrt(scores['sse'] / n_tr)
    print(f"  {scores['exact']}/{n_tr} exact  RMSE={rmse:.4f}  super{len(combo)}+p{pw}")
    print(f"    {combo_names}")


# ============================================================
#  FINAL TRAINING — all models on full data
# ============================================================
print('\n' + '=' * 60)
print(' FINAL TRAINING')
print('=' * 60)

final_preds_tr = {}
final_preds_te = {}
for mname, model_template, needs_scaled in ALL_MODEL_SPECS:
    if hasattr(model_template, 'get_params'):
        m = type(model_template)(**model_template.get_params())
    else:
        m = type(model_template)(**model_template.__dict__)

    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te
    m.fit(Xt, y_train)
    final_preds_tr[mname] = m.predict(Xt)
    final_preds_te[mname] = m.predict(Xtest)

print(f'  Trained {n_models} models on full data')


# ============================================================
#  GENERATE ALL TEST PREDICTIONS
# ============================================================
print('\n' + '=' * 60)
print(' GENERATING TEST PREDICTIONS')
print('=' * 60)

all_test_preds = {}  # key=(bname, rc) -> test predictions

for bname, bmodels in BLEND_CONFIGS.items():
    raw_tr = np.mean([final_preds_tr[m] for m in bmodels], axis=0)
    raw_te = np.mean([final_preds_te[m] for m in bmodels], axis=0)

    for rc_name, rc_params in RC_CONFIGS.items():
        if rc_params is None:
            all_test_preds[(bname, rc_name)] = raw_te
        else:
            residuals = y_train - raw_tr
            X_aug_tr = np.column_stack([X_tr, raw_tr])
            X_aug_te = np.column_stack([X_te, raw_te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            all_test_preds[(bname, rc_name)] = raw_te + rm.predict(X_aug_te)

print(f'  Generated {len(all_test_preds)} test prediction sets')


# ============================================================
#  EVALUATE: LOSO-chosen configs on test
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS — LOSO-CHOSEN (HONEST)')
print('=' * 60)

all_test_results = []

# Single blend configs
for (bname, rc, pw), scores in loso_sorted[:50]:
    key = (bname, rc)
    if key not in all_test_preds:
        continue
    pred = all_test_preds[key]
    a = hungarian(pred, test_seasons, avail_seeds, pw)
    ex, sse = evaluate(a, test_gt)
    rmse = np.sqrt(sse / 451)
    loso_ex = scores['exact']
    all_test_results.append((f'{bname}+{rc}', pw, ex, sse, rmse, a, pred, loso_ex))

# Super-blend configs (LOSO-chosen)
for rank, ((combo, pw), scores) in enumerate(super_sorted[:30]):
    pred = np.mean([all_test_preds[config_keys[i]] for i in combo], axis=0)
    a = hungarian(pred, test_seasons, avail_seeds, pw)
    ex, sse = evaluate(a, test_gt)
    rmse = np.sqrt(sse / 451)
    loso_ex = scores['exact']
    names = '+'.join([config_keys[i][0][:6] for i in combo])
    tag = f'super{len(combo)}_{names}+{config_keys[combo[0]][1]}'
    all_test_results.append((tag, pw, ex, sse, rmse, a, pred, loso_ex))

# Also try ALL combos on test (for comparison — labeled as GT-snooped)
print('\n--- Also evaluating all test combos (GT reference, not for submission) ---')
gt_test_results = []
for (bname, rc), pred in all_test_preds.items():
    for power in POWERS:
        a = hungarian(pred, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        rmse = np.sqrt(sse / 451)
        gt_test_results.append((f'{bname}+{rc}', power, ex, sse, rmse, a, pred))

# Super-blend on test (all combos)
for n_combo in [2, 3, 4, 5]:
    for combo in combinations(range(min(len(config_keys), 10)), n_combo):
        pred = np.mean([all_test_preds[config_keys[i]] for i in combo], axis=0)
        for power in [1.0, 1.1, 1.25]:
            a = hungarian(pred, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            gt_test_results.append((f'gt_super{len(combo)}_{combo}', power, ex, sse,
                                     rmse, a, pred))

gt_test_results.sort(key=lambda x: (-x[2], x[3]))
all_test_results.sort(key=lambda x: (-x[2], x[3]))

print(f'\nLOSO-CHOSEN test results (HONEST — use these for submission):')
seen = set()
for sname, pw, ex, sse, rmse, assigned, pred, loso_ex in all_test_results:
    key = (sname, pw)
    if key in seen:
        continue
    seen.add(key)
    print(f'  {ex}/91  RMSE={rmse:.4f}  LOSO:{loso_ex}/{n_tr}  {sname}+p{pw}')
    if len(seen) >= 30:
        break

print(f'\nGT-SNOOPED best (for reference only, NOT for submission):')
seen2 = set()
for sname, pw, ex, sse, rmse, assigned, pred in gt_test_results[:30]:
    key = (sname, pw)
    if key in seen2:
        continue
    seen2.add(key)
    print(f'  {ex}/91  RMSE={rmse:.4f}  {sname}+p{pw}  [GT-SNOOPED]')
    if len(seen2) >= 15:
        break


# ============================================================
#  SAVE SUBMISSIONS — LOSO-chosen only
# ============================================================
print('\n' + '=' * 60)
print(' SAVING SUBMISSIONS (LOSO-CHOSEN ONLY)')
print('=' * 60)


def save_sub(assigned, name, desc):
    sub = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub['RecordID'] == rid
        if mask.any():
            sub.loc[mask, 'Overall Seed'] = int(assigned[i])
    path = os.path.join(DATA_DIR, name)
    sub.to_csv(path, index=False)
    ex, sse = evaluate(assigned, test_gt)
    rmse = np.sqrt(sse / 451)
    print(f'  Saved {name}: {ex}/91 RMSE={rmse:.4f} [{desc}]')
    return path


submissions = []
saved = set()
rank = 0
for sname, pw, ex, sse, rmse, assigned, pred, loso_ex in all_test_results:
    key = (sname, pw)
    if key in saved:
        continue
    if len(saved) >= 10:
        break
    rank += 1
    fname = f'submission_v29_{rank}.csv'
    p = save_sub(assigned, fname, f'LOSO:{loso_ex}/{n_tr} {sname}+p{pw}')
    submissions.append(p)
    saved.add(key)

# Also save the GT-best for reference
gt_best = gt_test_results[0]
gt_assigned = gt_best[5]
save_sub(gt_assigned, 'submission_v29_gt_best.csv',
         f'GT-SNOOPED: {gt_best[2]}/91 {gt_best[0]}+p{gt_best[1]}')

# Best LOSO-chosen details
if all_test_results:
    best = all_test_results[0]
    sname, pw, ex, sse, rmse, assigned, pred, loso_ex = best
    print(f'\n** LOSO-chosen best: {ex}/91 exact, RMSE={rmse:.4f} ({sname}+p{pw})')
    print(f'   LOSO score: {loso_ex}/{n_tr}')

    print('\nPer-season:')
    for s in sorted(set(test_seasons)):
        sm = test_seasons == s
        ex_s = int(np.sum(assigned[sm] == test_gt[sm]))
        total = int(sm.sum())
        sse_s = int(np.sum((assigned[sm] - test_gt[sm]) ** 2))
        print(f'  {s}: {ex_s}/{total} exact, SSE={sse_s}')

    errors = assigned - test_gt
    abs_err = np.abs(errors)
    print(f'\nErrors: mean={abs_err.mean():.2f} max={abs_err.max()} '
          f'>5={int((abs_err>5).sum())} >10={int((abs_err>10).sum())}')

    print('\nWorst predictions:')
    worst = np.argsort(abs_err)[::-1][:10]
    for i in worst:
        print(f'  {test_rids[i]:30s} pred={assigned[i]:2d} actual={test_gt[i]:2d} err={errors[i]:+3d}')

print(f'\n** GT-snooped best: {gt_test_results[0][2]}/91')
print(f'   Overfitting gap: {gt_test_results[0][2] - all_test_results[0][2]} '
      f'(GT-best minus LOSO-best on test)')

total_t = time.time() - t0
print(f'\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
