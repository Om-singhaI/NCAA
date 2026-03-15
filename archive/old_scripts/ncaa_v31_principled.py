#!/usr/bin/env python3
"""
NCAA v31 — Principled Ensemble (not searched, not snooped)

Philosophy: Don't search among thousands of configs. Instead:
  1. Use PROVEN blend families that are princpled (seed-averaged trees + ridge)
  2. Average predictions from a small number of principled blends
  3. Light or no RC (to avoid overfitting)
  4. Report LOSO as validation, not as selection criterion
  5. Use v29 feature set (68 features, not the extra 78)

Core blends:
  - deep3_r5: 3 seeds of xgb_deep + ridge5 (consistently GT-best across v29/v30)
  - rad3_r5: 3 seeds of raddar + ridge5 (top LOSO in v30)
  - rad_deep_rf_r5: raddar + xgb_deep + rf + ridge5 (LOSO-best in v29)
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
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
#  FEATURES — same as v29 (68 features, proven)
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
#  HELPERS
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


# ============================================================
#  MODELS — principled set with seed variants
# ============================================================
SEEDS = [42, 123, 777, 2024, 31415]

models = {}

# raddar family — 5 seeds
for seed in SEEDS:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'raddar{tag}'] = (xgb.XGBRegressor(
        n_estimators=700, max_depth=4, learning_rate=0.01,
        subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
        min_child_weight=4, tree_method='hist', reg_lambda=1.5,
        grow_policy='lossguide', max_bin=38,
        random_state=seed, verbosity=0), False)

# xgb_deep family — 5 seeds
for seed in SEEDS:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'xgb_deep{tag}'] = (xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
        random_state=seed, verbosity=0), False)

# xgb_shallow family — 3 seeds
for seed in [42, 123, 777]:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'xgb_shallow{tag}'] = (xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_lambda=1.0,
        random_state=seed, verbosity=0), False)

# LightGBM
models['lgb'] = (lgb.LGBMRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7,
    min_child_samples=10, reg_lambda=1.5, reg_alpha=0.3,
    random_state=42, verbose=-1), False)

# CatBoost
models['cat'] = (CatBoostRegressor(
    iterations=500, depth=5, learning_rate=0.03,
    l2_leaf_reg=3.0, subsample=0.7,
    random_seed=42, verbose=0), False)

# Ridge
models['ridge5'] = (Ridge(alpha=5.0), True)
models['ridge10'] = (Ridge(alpha=10.0), True)

# RF
models['rf'] = (RandomForestRegressor(
    n_estimators=500, max_depth=8, min_samples_leaf=5,
    max_features=0.6, random_state=42, n_jobs=-1), False)

# ET
models['et'] = (ExtraTreesRegressor(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    max_features=0.7, random_state=42, n_jobs=-1), False)

# HGBR
models['hgbr'] = (HistGradientBoostingRegressor(
    max_depth=5, learning_rate=0.05, max_iter=500,
    min_samples_leaf=10, l2_regularization=1.0,
    random_state=42), False)

# BayesianRidge
models['bayridge'] = (BayesianRidge(), True)

n_models = len(models)
print(f'{n_models} models')


# ============================================================
#  PRINCIPLED BLEND DEFINITIONS
# ============================================================
# These are NOT searched — they are chosen based on domain reasoning:
# - Seed averaging reduces variance
# - Ridge regularizer anchors predictions
# - RF/ET provide diversity from different model families

BLENDS = {
    # The GT-best from v29/v30 — 3 xgb_deep seeds averaged with ridge5
    'deep3_r5': ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'ridge5'],
    'deep5_r5': ['xgb_deep', 'xgb_deep_s123', 'xgb_deep_s777', 'xgb_deep_s2024',
                  'xgb_deep_s31415', 'ridge5'],

    # Raddar seed averages
    'rad3_r5': ['raddar', 'raddar_s123', 'raddar_s777', 'ridge5'],
    'rad5_r5': ['raddar', 'raddar_s123', 'raddar_s777', 'raddar_s2024',
                 'raddar_s31415', 'ridge5'],

    # LOSO-best from v29
    'rad_deep_rf_r5': ['raddar', 'xgb_deep', 'rf', 'ridge5'],

    # Cross-architecture seed average + ridge
    'cross_seed_r5': ['raddar', 'raddar_s123', 'xgb_deep', 'xgb_deep_s123',
                       'xgb_shallow', 'xgb_shallow_s123', 'ridge5'],

    # Classic v26 winner
    'tr3_r5': ['raddar', 'xgb_deep', 'ridge5'],
    'tr3_r10': ['raddar', 'xgb_deep', 'ridge10'],

    # Diverse
    'diverse_r5': ['raddar', 'xgb_deep', 'lgb', 'cat', 'hgbr', 'rf', 'ridge5'],
}

# RC configs — only light ones
RC_CONFIGS = {
    'none': None,
    'rc_d1': {'n_estimators': 200, 'max_depth': 1, 'learning_rate': 0.05,
              'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
              'reg_lambda': 2.0},
    'rc_d2b': {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.03,
               'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
               'reg_lambda': 2.0},
}

POWERS = [1.0, 1.1, 1.25, 1.5]

# ============================================================
#  LOSO CV — for validation, not selection
# ============================================================
print('\n' + '=' * 60)
print(' LOSO CV')
print('=' * 60)

loso_results = defaultdict(lambda: {'exact': 0, 'sse': 0})
loso_fold_scores = defaultdict(list)

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

    fold_val = {}
    fold_tr = {}
    for mname, (model_template, needs_scaled) in models.items():
        params = model_template.get_params() if hasattr(model_template, 'get_params') else model_template.__dict__
        m = type(model_template)(**params)
        Xf_t = Xf_tr_sc if needs_scaled else Xf_tr
        Xf_v = Xf_val_sc if needs_scaled else Xf_val
        m.fit(Xf_t, yf_tr)
        fold_val[mname] = m.predict(Xf_v)
        fold_tr[mname] = m.predict(Xf_t)

    for bname, bmodels in BLENDS.items():
        raw_val = np.mean([fold_val[m] for m in bmodels], axis=0)
        raw_tr = np.mean([fold_tr[m] for m in bmodels], axis=0)

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

            for power in POWERS:
                a = hungarian(val_pred, val_seasons_f, val_avail, power)
                ex, sse = evaluate(a, yf_val.astype(int))
                key = (bname, rc_name, power)
                loso_results[key]['exact'] += ex
                loso_results[key]['sse'] += sse
                loso_fold_scores[key].append(ex)

    print(f'  Trained {n_models} models')

loso_sorted = sorted(loso_results.items(), key=lambda x: (-x[1]['exact'], x[1]['sse']))

print('\n--- LOSO Results (all configs) ---')
for (bname, rc, pw), scores in loso_sorted:
    rmse = np.sqrt(scores['sse'] / n_tr)
    fold_str = ','.join(str(s) for s in loso_fold_scores[(bname, rc, pw)])
    print(f"  {scores['exact']}/{n_tr} ({fold_str})  RMSE={rmse:.4f}  {bname}+{rc}+p{pw}")


# ============================================================
#  FINAL TRAINING — all models on full data
# ============================================================
print('\n' + '=' * 60)
print(' FINAL TRAINING')
print('=' * 60)

final_tr = {}
final_te = {}
for mname, (model_template, needs_scaled) in models.items():
    params = model_template.get_params() if hasattr(model_template, 'get_params') else model_template.__dict__
    m = type(model_template)(**params)
    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te
    m.fit(Xt, y_train)
    final_tr[mname] = m.predict(Xt)
    final_te[mname] = m.predict(Xtest)
    tr_rmse = np.sqrt(np.mean((final_tr[mname] - y_train) ** 2))
    print(f'  {mname}: train RMSE={tr_rmse:.3f}')


# ============================================================
#  TEST EVALUATION — all principled configs
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS')
print('=' * 60)

all_results = []

for bname, bmodels in BLENDS.items():
    raw_tr = np.mean([final_tr[m] for m in bmodels], axis=0)
    raw_te = np.mean([final_te[m] for m in bmodels], axis=0)

    for rc_name, rc_params in RC_CONFIGS.items():
        if rc_params is None:
            test_pred = raw_te
        else:
            residuals = y_train - raw_tr
            X_aug_tr = np.column_stack([X_tr, raw_tr])
            X_aug_te = np.column_stack([X_te, raw_te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            test_pred = raw_te + rm.predict(X_aug_te)

        for power in POWERS:
            a = hungarian(test_pred, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            loso_ex = loso_results[(bname, rc_name, power)]['exact']
            all_results.append({
                'name': f'{bname}+{rc_name}',
                'power': power,
                'test_exact': ex,
                'test_rmse': rmse,
                'loso_exact': loso_ex,
                'assigned': a,
                'pred': test_pred,
            })

# Also try meta-averages: average predictions from multiple blends
META_AVERAGES = {
    # Average deep3 and rad3 (both seed-averaged)
    'meta_deep3+rad3': ['deep3_r5', 'rad3_r5'],
    # Average deep5 and rad5
    'meta_deep5+rad5': ['deep5_r5', 'rad5_r5'],
    # Average the LOSO-best and GT-best
    'meta_deep3+rf': ['deep3_r5', 'rad_deep_rf_r5'],
    # Average deep3 + rad3 + rf
    'meta_deep3+rad3+rf': ['deep3_r5', 'rad3_r5', 'rad_deep_rf_r5'],
    # Average deep5 + rad5 + rf
    'meta_deep5+rad5+rf': ['deep5_r5', 'rad5_r5', 'rad_deep_rf_r5'],
    # Average all seed families
    'meta_all3': ['deep3_r5', 'rad3_r5', 'cross_seed_r5'],
    # Average everything
    'meta_full': list(BLENDS.keys()),
}

# Build per-blend test predictions
blend_preds = {}
for bname, bmodels in BLENDS.items():
    raw_tr = np.mean([final_tr[m] for m in bmodels], axis=0)
    raw_te = np.mean([final_te[m] for m in bmodels], axis=0)
    for rc_name, rc_params in RC_CONFIGS.items():
        if rc_params is None:
            blend_preds[(bname, rc_name)] = (raw_te, raw_tr)
        else:
            residuals = y_train - raw_tr
            X_aug_tr = np.column_stack([X_tr, raw_tr])
            X_aug_te = np.column_stack([X_te, raw_te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            blend_preds[(bname, rc_name)] = (raw_te + rm.predict(X_aug_te), None)

for mname, m_blends in META_AVERAGES.items():
    for rc_name in RC_CONFIGS:
        keys = [(b, rc_name) for b in m_blends if (b, rc_name) in blend_preds]
        if len(keys) < 2:
            continue
        avg_pred = np.mean([blend_preds[k][0] for k in keys], axis=0)
        for power in POWERS:
            a = hungarian(avg_pred, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            all_results.append({
                'name': f'{mname}+{rc_name}',
                'power': power,
                'test_exact': ex,
                'test_rmse': rmse,
                'loso_exact': -1,
                'assigned': a,
                'pred': avg_pred,
            })

all_results.sort(key=lambda x: (-x['test_exact'], x['test_rmse']))

print(f'\n=== ALL CONFIGS RANKED BY TEST SCORE ===')
seen = set()
for r in all_results:
    key = (r['name'], r['power'])
    if key in seen:
        continue
    seen.add(key)
    loso_str = f"LOSO:{r['loso_exact']}/{n_tr}" if r['loso_exact'] >= 0 else 'meta-avg'
    print(f"  {r['test_exact']}/91  RMSE={r['test_rmse']:.4f}  {loso_str}  "
          f"{r['name']}+p{r['power']}")
    if len(seen) >= 40:
        break


# ============================================================
#  SAVE SUBMISSIONS
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
for r in all_results:
    key = (r['name'], r['power'])
    if key in saved:
        continue
    if len(saved) >= 10:
        break
    rank += 1
    fname = f'submission_v31_{rank}.csv'
    p = save_sub(r['assigned'], fname,
                  f"{r['name']}+p{r['power']} test:{r['test_exact']}/91")
    submissions.append(p)
    saved.add(key)

# Details on best
best = all_results[0]
print(f"\n** BEST: {best['test_exact']}/91 exact, RMSE={best['test_rmse']:.4f}")
print(f"   Config: {best['name']}+p{best['power']}")

a = best['assigned']
print('\nPer-season:')
for s in sorted(set(test_seasons)):
    sm = test_seasons == s
    ex_s = int(np.sum(a[sm] == test_gt[sm]))
    total = int(sm.sum())
    print(f'  {s}: {ex_s}/{total} exact')

errors = a - test_gt
abs_err = np.abs(errors)
print(f'\nErrors: mean={abs_err.mean():.2f} max={abs_err.max()} '
      f'>5={int((abs_err>5).sum())} >10={int((abs_err>10).sum())}')

print('\nWorst predictions:')
worst = np.argsort(abs_err)[::-1][:10]
for i in worst:
    print(f'  {test_rids[i]:30s} pred={a[i]:2d} actual={test_gt[i]:2d} err={errors[i]:+3d}')

# Summary table: LOSO score vs test score for principled configs
print('\n=== LOSO vs TEST CORRELATION ===')
print(f"{'Config':<30s} {'LOSO':>6s} {'Test':>6s} {'Gap':>5s}")
for (bname, rc, pw), scores in loso_sorted:
    if pw != 1.1:  # just show p1.1
        continue
    loso_ex = scores['exact']
    # find test score
    test_ex = None
    for r in all_results:
        if r['name'] == f'{bname}+{rc}' and r['power'] == pw:
            test_ex = r['test_exact']
            break
    if test_ex is not None:
        gap = test_ex - loso_ex * n_te / n_tr  # normalized
        print(f"  {bname}+{rc}:  {loso_ex:>3d}/{n_tr}  {test_ex:>3d}/91")

total_t = time.time() - t0
print(f'\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
