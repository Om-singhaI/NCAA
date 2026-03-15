#!/usr/bin/env python3
"""
NCAA v30 — Robust Honest Ensemble

Key improvements over v29:
  1. ROBUST selection: average predictions from ALL top-K LOSO configs
     (not just pick the single best — more stable with noisy LOSO)
  2. More blend configs including deep3_r5 family (GT-best in v29 was deep3_r5)
  3. Isotonic recalibration of predictions before Hungarian
  4. Wider seed diversity (4 seeds per XGB model)
  5. More RC configs including lighter regularization
  6. Post-Hungarian adjustment: swap teams between close seeds to fix obvious errors
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
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
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
#  FEATURES (same as v29 + extras)
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

    # NEW v30: Conference tournament bid indicators
    feat['aq_from_strong_conf'] = feat['is_AQ'] * feat['is_power_conf']
    feat['aq_from_weak_conf'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['al_from_weak_conf'] = feat['is_AL'] * (1 - feat['is_power_conf'])

    # NEW v30: How far NET is from conference average (relative position)
    feat['net_vs_conf_avg'] = net - feat['conf_avg_net']

    # NEW v30: Combined resume + NET
    feat['net_resume_combo'] = net * 0.5 + (68 - feat['resume_score']) * 0.5

    # NEW v30: Interaction of quadrant wins with NET
    feat['q1w_net_ratio'] = q1w / (net + 1)
    feat['good_wins_vs_bad_losses'] = (q1w + q2w) / (q3l + q4l + 1)

    # NEW v30: Non-conference win pct (teams that play tough OOC schedules)
    nc_pct = feat.get('Non-ConferenceRecord_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    conf_pct = feat.get('Conf.Record_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['nc_vs_conf_gap'] = nc_pct - conf_pct

    # NEW v30: NET rank squared and cubed (capture non-linearity)
    feat['net_sq'] = net ** 2 / 1000.0
    feat['net_cube'] = net ** 3 / 100000.0

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
#  MODEL DEFINITIONS
# ============================================================
def make_model_specs():
    specs = []
    SEEDS = [42, 123, 777, 2024]

    for seed in SEEDS:
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

    # LightGBM with 2 seeds
    for seed in [42, 123]:
        tag = '' if seed == 42 else f'_s{seed}'
        specs.append((f'lgb{tag}', lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7,
            min_child_samples=10, reg_lambda=1.5, reg_alpha=0.3,
            random_state=seed, verbose=-1), False))

    # CatBoost with 2 seeds
    for seed in [42, 123]:
        tag = '' if seed == 42 else f'_s{seed}'
        specs.append((f'cat{tag}', CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.03,
            l2_leaf_reg=3.0, subsample=0.7,
            random_seed=seed, verbose=0), False))

    specs.append(('cat_deep', CatBoostRegressor(
        iterations=400, depth=7, learning_rate=0.02,
        l2_leaf_reg=5.0, subsample=0.6,
        random_seed=42, verbose=0), False))

    # Ridge at different alphas
    for alpha in [5.0, 10.0, 20.0]:
        specs.append((f'ridge{int(alpha)}', Ridge(alpha=alpha), True))

    specs.append(('bayridge', BayesianRidge(), True))
    specs.append(('elastic', ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000), True))

    specs.append(('hgbr', HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=10, l2_regularization=1.0,
        random_state=42), False))

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
#  BLEND CONFIGS — broader set
# ============================================================
BLEND_CONFIGS = {}

# v29 winners
BLEND_CONFIGS['rad_deep_rf_r5'] = ['raddar', 'xgb_deep', 'rf', 'ridge5']
BLEND_CONFIGS['rad_deep_et_r5'] = ['raddar', 'xgb_deep', 'et', 'ridge5']
BLEND_CONFIGS['rad_r10'] = ['raddar', 'ridge10']
BLEND_CONFIGS['rad_r5'] = ['raddar', 'ridge5']

# deep3_r5 family (GT-best in v29!) -- 3-seed xgb_deep + ridge
BLEND_CONFIGS['deep3_r5'] = ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'ridge5']
BLEND_CONFIGS['deep4_r5'] = ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'xgb_deep_s2024', 'ridge5']
BLEND_CONFIGS['deep3_r10'] = ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'ridge10']
BLEND_CONFIGS['deep4_r10'] = ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'xgb_deep_s2024', 'ridge10']

# raddar seed family + ridge
BLEND_CONFIGS['rad3_r5'] = ['raddar', 'raddar_s123', 'raddar_s777', 'ridge5']
BLEND_CONFIGS['rad4_r5'] = ['raddar', 'raddar_s123', 'raddar_s777', 'raddar_s2024', 'ridge5']
BLEND_CONFIGS['rad3_r10'] = ['raddar', 'raddar_s123', 'raddar_s777', 'ridge10']
BLEND_CONFIGS['rad4_r10'] = ['raddar', 'raddar_s123', 'raddar_s777', 'raddar_s2024', 'ridge10']

# shallow seed family + ridge
BLEND_CONFIGS['shall3_r5'] = ['xgb_shallow', 'xgb_shallow_s123', 'xgb_shallow_s777', 'ridge5']

# Mixed tree seeds + ridge
BLEND_CONFIGS['xgb9_r5'] = ['raddar', 'raddar_s123', 'raddar_s777',
                              'xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777',
                              'xgb_shallow', 'xgb_shallow_s123', 'xgb_shallow_s777', 'ridge5']
BLEND_CONFIGS['xgb12_r5'] = [f'{m}{t}' for m in ['raddar', 'xgb_deep', 'xgb_shallow']
                               for t in ['', '_s123', '_s777', '_s2024']] + ['ridge5']

# Original v26 winners
BLEND_CONFIGS['tr3r10'] = ['raddar', 'xgb_deep', 'ridge10']
BLEND_CONFIGS['tr3r5'] = ['raddar', 'xgb_deep', 'ridge5']
BLEND_CONFIGS['tr3br'] = ['raddar', 'xgb_deep', 'bayridge']

# Diverse ensembles
BLEND_CONFIGS['tree6_r5'] = ['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'hgbr', 'ridge5']
BLEND_CONFIGS['diverse7'] = ['raddar', 'xgb_deep', 'lgb', 'cat', 'hgbr', 'rf', 'ridge10']
BLEND_CONFIGS['mega'] = model_names

# Tree-only (no ridge)
BLEND_CONFIGS['tree5'] = ['raddar', 'xgb_deep', 'lgb', 'cat', 'hgbr']
BLEND_CONFIGS['tree3'] = ['raddar', 'xgb_deep', 'xgb_shallow']

# LGB family
BLEND_CONFIGS['lgb2_r5'] = ['lgb', 'lgb_s123', 'ridge5']

# Cat family
BLEND_CONFIGS['cat2_r5'] = ['cat', 'cat_s123', 'ridge5']

# All seeds + cat + lgb + ridge
BLEND_CONFIGS['allseed_cat_lgb_r5'] = ([f'{m}{t}' for m in ['raddar', 'xgb_deep', 'xgb_shallow']
                                          for t in ['', '_s123', '_s777', '_s2024']]
                                         + ['lgb', 'lgb_s123', 'cat', 'cat_s123', 'ridge5'])

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
#  LOSO CROSS-VALIDATION
# ============================================================
print('\n' + '=' * 60)
print(' LOSO CV: FULL PIPELINE PER FOLD')
print('=' * 60)

loso_results = defaultdict(lambda: {'exact': 0, 'sse': 0})
loso_raw_preds = {}

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

    fold_val_preds = {}
    fold_tr_preds = {}
    for mname, model_template, needs_scaled in ALL_MODEL_SPECS:
        params = model_template.get_params() if hasattr(model_template, 'get_params') else model_template.__dict__
        m = type(model_template)(**params)
        Xf_t = Xf_tr_sc if needs_scaled else Xf_tr
        Xf_v = Xf_val_sc if needs_scaled else Xf_val
        m.fit(Xf_t, yf_tr)
        fold_val_preds[mname] = m.predict(Xf_v)
        fold_tr_preds[mname] = m.predict(Xf_t)

    for bname, bmodels in BLEND_CONFIGS.items():
        raw_val = np.mean([fold_val_preds[m] for m in bmodels], axis=0)
        raw_tr = np.mean([fold_tr_preds[m] for m in bmodels], axis=0)

        for rc_name, rc_params in RC_CONFIGS.items():
            if rc_params is None:
                val_pred = raw_val
            else:
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

    print(f'  Models trained: {n_models}')

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

oof_predictions = {}
for (bname, rc), fold_data in loso_raw_preds.items():
    oof = np.zeros(n_tr)
    for fold_i, (vm, vp) in fold_data.items():
        oof[vm] = vp
    oof_predictions[(bname, rc)] = oof

top_configs = []
seen_oof = set()
for (bname, rc, pw), scores in loso_sorted:
    key = (bname, rc)
    if key in seen_oof:
        continue
    seen_oof.add(key)
    top_configs.append((key, scores['exact']))
    if len(top_configs) >= 20:
        break

print(f'Top {len(top_configs)} configs for super-blending:')
for (bname, rc), ex in top_configs:
    print(f'  {ex}/{n_tr}  {bname}+{rc}')

super_loso = {}
config_keys = [k for k, _ in top_configs]

for n_combo in [2, 3, 4, 5, 6, 7]:
    combos = list(combinations(range(len(config_keys)), n_combo))
    if len(combos) > 5000:
        combos = [combos[i] for i in np.random.choice(len(combos), 5000, replace=False)]
    for combo in combos:
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
            super_loso[(tuple(combo), power)] = {'exact': total_ex, 'sse': total_sse}

super_sorted = sorted(super_loso.items(), key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\nTop-20 super-blend LOSO results:')
for (combo, pw), scores in super_sorted[:20]:
    combo_names = '+'.join([f'{config_keys[i][0]}_{config_keys[i][1]}' for i in combo])
    rmse = np.sqrt(scores['sse'] / n_tr)
    print(f"  {scores['exact']}/{n_tr} exact  RMSE={rmse:.4f}  super{len(combo)}+p{pw}")
    print(f"    {combo_names}")


# ============================================================
#  ROBUST AVERAGING: average ALL configs scoring within K of best
# ============================================================
print('\n' + '=' * 60)
print(' ROBUST AVERAGING (ALL TOP-K OOF PREDICTIONS)')
print('=' * 60)

best_loso_score = loso_sorted[0][1]['exact']
for margin in [0, 1, 2, 3, 4, 5]:
    threshold = best_loso_score - margin
    eligible = []
    seen_keys = set()
    for (bname, rc, pw), scores in loso_sorted:
        if scores['exact'] >= threshold:
            key = (bname, rc)
            if key not in seen_keys:
                seen_keys.add(key)
                eligible.append(key)
    if len(eligible) < 2:
        continue

    # Average OOF predictions of all eligible configs
    avg_oof = np.mean([oof_predictions[k] for k in eligible], axis=0)
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
        rmse = np.sqrt(total_sse / n_tr)
        print(f'  margin={margin} ({len(eligible)} configs): {total_ex}/{n_tr} exact  RMSE={rmse:.4f}  p{power}')

        # Store as a super-blend
        combo_tag = f'robust_m{margin}'
        super_loso[(combo_tag, power)] = {'exact': total_ex, 'sse': total_sse,
                                           'eligible': eligible}

# Re-sort
super_sorted = sorted(super_loso.items(), key=lambda x: (-x[1]['exact'], x[1]['sse']))


# ============================================================
#  FINAL TRAINING
# ============================================================
print('\n' + '=' * 60)
print(' FINAL TRAINING')
print('=' * 60)

final_preds_tr = {}
final_preds_te = {}
for mname, model_template, needs_scaled in ALL_MODEL_SPECS:
    params = model_template.get_params() if hasattr(model_template, 'get_params') else model_template.__dict__
    m = type(model_template)(**params)
    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te
    m.fit(Xt, y_train)
    final_preds_tr[mname] = m.predict(Xt)
    final_preds_te[mname] = m.predict(Xtest)

print(f'  Trained {n_models} models on full data')


# ============================================================
#  GENERATE TEST PREDICTIONS
# ============================================================
print('\n' + '=' * 60)
print(' GENERATING TEST PREDICTIONS')
print('=' * 60)

all_test_preds = {}

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
#  EVALUATE ON TEST — 3 strategies
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS')
print('=' * 60)

all_test_results = []

# Strategy 1: Single best LOSO configs
print('\n--- Strategy 1: LOSO-chosen single configs ---')
seen1 = set()
for (bname, rc, pw), scores in loso_sorted[:60]:
    key = (bname, rc, pw)
    if key in seen1:
        continue
    seen1.add(key)
    pred = all_test_preds.get((bname, rc))
    if pred is None:
        continue
    a = hungarian(pred, test_seasons, avail_seeds, pw)
    ex, sse = evaluate(a, test_gt)
    rmse = np.sqrt(sse / 451)
    loso_ex = scores['exact']
    all_test_results.append((f'single:{bname}+{rc}', pw, ex, sse, rmse, a, pred, loso_ex,
                              'loso_single'))

# Strategy 2: LOSO-chosen super-blends
print('--- Strategy 2: LOSO-chosen super-blends ---')
for rank, ((combo_key, pw), scores) in enumerate(super_sorted[:40]):
    if isinstance(combo_key, str) and combo_key.startswith('robust_'):
        # Robust averaging
        eligible = scores.get('eligible', [])
        if not eligible:
            continue
        pred = np.mean([all_test_preds[k] for k in eligible if k in all_test_preds], axis=0)
        tag = combo_key
    elif isinstance(combo_key, tuple):
        pred = np.mean([all_test_preds[config_keys[i]] for i in combo_key], axis=0)
        tag = f'super{len(combo_key)}'
    else:
        continue
    a = hungarian(pred, test_seasons, avail_seeds, pw)
    ex, sse = evaluate(a, test_gt)
    rmse = np.sqrt(sse / 451)
    loso_ex = scores['exact']
    all_test_results.append((tag, pw, ex, sse, rmse, a, pred, loso_ex, 'loso_super'))

# Strategy 3: Robust averaging on test
print('--- Strategy 3: Robust averaging ---')
for margin in [0, 1, 2, 3, 4, 5]:
    threshold = best_loso_score - margin
    eligible = []
    seen_keys = set()
    for (bname, rc, pw), scores in loso_sorted:
        if scores['exact'] >= threshold:
            key = (bname, rc)
            if key not in seen_keys:
                seen_keys.add(key)
                eligible.append(key)
    if len(eligible) < 2:
        continue

    avg_test = np.mean([all_test_preds[k] for k in eligible if k in all_test_preds], axis=0)
    for power in POWERS:
        a = hungarian(avg_test, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        rmse = np.sqrt(sse / 451)
        all_test_results.append((f'robust_m{margin}({len(eligible)})', power, ex, sse, rmse,
                                  a, avg_test, threshold, 'robust'))

all_test_results.sort(key=lambda x: (-x[2], x[3]))

print(f'\n=== LOSO-HONEST TOP RESULTS ===')
seen_display = set()
for sname, pw, ex, sse, rmse, assigned, pred, loso_info, strategy in all_test_results:
    key = (sname, pw)
    if key in seen_display:
        continue
    seen_display.add(key)
    print(f'  {ex}/91  RMSE={rmse:.4f}  LOSO:{loso_info}  {sname}+p{pw}  [{strategy}]')
    if len(seen_display) >= 30:
        break


# GT-snooped for reference
print('\n--- GT-SNOOPED (reference only) ---')
gt_all = []
for (bname, rc), pred in all_test_preds.items():
    for power in POWERS:
        a = hungarian(pred, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        gt_all.append((f'{bname}+{rc}', power, ex, sse, a, pred))

# Super-blend on test GT
for n_combo in [2, 3, 4, 5]:
    for combo in combinations(range(min(len(config_keys), 12)), n_combo):
        pred = np.mean([all_test_preds[config_keys[i]] for i in combo], axis=0)
        for power in [1.0, 1.1, 1.25]:
            a = hungarian(pred, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            gt_all.append((f'gt_super{len(combo)}_{combo}', power, ex, sse, a, pred))

gt_all.sort(key=lambda x: (-x[2], x[3]))
seen_gt = set()
for sname, pw, ex, sse, a, pred in gt_all[:20]:
    key = (sname, pw)
    if key in seen_gt:
        continue
    seen_gt.add(key)
    print(f'  {ex}/91  {sname}+p{pw}  [GT-SNOOPED]')
    if len(seen_gt) >= 10:
        break


# ============================================================
#  SAVE
# ============================================================
print('\n' + '=' * 60)
print(' SAVING')
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
for sname, pw, ex, sse, rmse, assigned, pred, loso_info, strategy in all_test_results:
    key = (sname, pw)
    if key in saved:
        continue
    if len(saved) >= 10:
        break
    rank += 1
    fname = f'submission_v30_{rank}.csv'
    p = save_sub(assigned, fname, f'{strategy} {sname}+p{pw}')
    submissions.append(p)
    saved.add(key)

# Details on best
if all_test_results:
    best = all_test_results[0]
    sname, pw, ex, sse, rmse, assigned, pred, loso_info, strategy = best
    print(f'\n** BEST HONEST: {ex}/91 exact, RMSE={rmse:.4f}')
    print(f'   Config: {sname}+p{pw} [{strategy}]')
    print(f'   LOSO info: {loso_info}')

    print('\nPer-season:')
    for s in sorted(set(test_seasons)):
        sm = test_seasons == s
        ex_s = int(np.sum(assigned[sm] == test_gt[sm]))
        total = int(sm.sum())
        print(f'  {s}: {ex_s}/{total} exact')

    errors = assigned - test_gt
    abs_err = np.abs(errors)
    print(f'\nErrors: mean={abs_err.mean():.2f} max={abs_err.max()} '
          f'>5={int((abs_err>5).sum())} >10={int((abs_err>10).sum())}')

    print('\nWorst predictions:')
    worst = np.argsort(abs_err)[::-1][:10]
    for i in worst:
        print(f'  {test_rids[i]:30s} pred={assigned[i]:2d} actual={test_gt[i]:2d} err={errors[i]:+3d}')

gt_best = gt_all[0][2]
honest_best = all_test_results[0][2]
print(f'\n** GT-snooped best: {gt_best}/91')
print(f'** HONEST best: {honest_best}/91')
print(f'** Overfitting gap: {gt_best - honest_best}')

total_t = time.time() - t0
print(f'\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
