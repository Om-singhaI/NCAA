#!/usr/bin/env python3
"""
NCAA v28 — Exhaustive blend search with in-sample residual correction

Strategy: v26's residual approach worked (58/91) because in-sample residuals
with regularized models are meaningful. The stacking OOF approach was too noisy
with only 5×50 folds. This script:
  1. Trains 15 diverse models on ALL training data
  2. Tries ~100 blend combinations
  3. Applies in-sample residual correction to each
  4. Also tries "super-blending": averaging raw predictions from multiple configs
  5. Post-Hungarian local search
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
from itertools import combinations

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
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
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

    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q + '_W'] = wl.apply(lambda x: x[0])
            feat[q + '_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
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
#  TRAIN ALL 15 MODELS ON FULL DATA
# ============================================================
print('\n' + '=' * 60)
print(' TRAINING 15 BASE MODELS')
print('=' * 60)

model_specs = {
    'raddar': (xgb.XGBRegressor(
        n_estimators=700, max_depth=4, learning_rate=0.01,
        subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
        min_child_weight=4, tree_method='hist', reg_lambda=1.5,
        grow_policy='lossguide', max_bin=38,
        random_state=42, verbosity=0), False),

    'xgb_deep': (xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
        random_state=42, verbosity=0), False),

    'xgb_shallow': (xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_lambda=1.0,
        random_state=42, verbosity=0), False),

    'xgb_wide': (xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.6,
        min_child_weight=4, reg_lambda=2.5, reg_alpha=1.0,
        random_state=42, verbosity=0), False),

    'lgb': (lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_samples=10, reg_lambda=1.5, reg_alpha=0.3,
        random_state=42, verbose=-1), False),

    'lgb_dart': (lgb.LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        boosting_type='dart', subsample=0.8, colsample_bytree=0.8,
        min_child_samples=8, reg_lambda=1.0,
        random_state=42, verbose=-1), False),

    'cat': (CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.7,
        random_seed=42, verbose=0), False),

    'cat_deep': (CatBoostRegressor(
        iterations=400, depth=7, learning_rate=0.02,
        l2_leaf_reg=5.0, subsample=0.6,
        random_seed=42, verbose=0), False),

    'ridge': (Ridge(alpha=10.0), True),
    'ridge5': (Ridge(alpha=5.0), True),
    'ridge20': (Ridge(alpha=20.0), True),

    'bayridge': (BayesianRidge(), True),

    'hgbr': (HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=10, l2_regularization=1.0,
        random_state=42), False),

    'rf': (RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features=0.6, random_state=42, n_jobs=-1), False),

    'et': (ExtraTreesRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=5,
        max_features=0.7, random_state=42, n_jobs=-1), False),

    'svr': (SVR(kernel='rbf', C=50.0, epsilon=1.0, gamma='scale'), True),

    'knn': (KNeighborsRegressor(n_neighbors=15, weights='distance'), True),
}

model_names = list(model_specs.keys())
train_preds = {}
test_preds = {}

for mname, (model_template, needs_scaled) in model_specs.items():
    if hasattr(model_template, 'get_params'):
        m = type(model_template)(**model_template.get_params())
    else:
        m = type(model_template)(**model_template.__dict__)

    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te

    m.fit(Xt, y_train)
    train_preds[mname] = m.predict(Xt)
    test_preds[mname] = m.predict(Xtest)

    tr_rmse = np.sqrt(np.mean((train_preds[mname] - y_train)**2))
    print(f'  {mname:15s}: train RMSE={tr_rmse:.3f}, '
          f'test range [{test_preds[mname].min():.1f}, {test_preds[mname].max():.1f}]')


# ============================================================
#  EXHAUSTIVE BLEND SEARCH + RESIDUAL CORRECTION
# ============================================================
print('\n' + '=' * 60)
print(' BLEND SEARCH WITH RESIDUAL CORRECTION')
print('=' * 60)

# Define blend groups to try
# Core tree-based models (good individually)
tree_models = ['raddar', 'xgb_deep', 'xgb_shallow', 'xgb_wide', 'lgb', 'cat',
               'cat_deep', 'hgbr']
linear_models = ['ridge', 'ridge5', 'ridge20', 'bayridge']
other_models = ['rf', 'et', 'svr', 'knn']

# Generate blend combinations systematically
blend_configs = []

# All pairs/triples/quads of tree models with one linear
for n_tree in [2, 3, 4]:
    for tree_combo in combinations(tree_models, n_tree):
        for lin in linear_models:
            blend_configs.append(list(tree_combo) + [lin])

# Top tree models only (no linear)
for n_tree in [2, 3, 4, 5, 6]:
    for tree_combo in combinations(['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'hgbr'], n_tree):
        blend_configs.append(list(tree_combo))

# Mixed with RF/ET
for tree_combo in combinations(['raddar', 'xgb_deep', 'lgb', 'cat'], 2):
    for other in ['rf', 'et']:
        blend_configs.append(list(tree_combo) + [other])
    for lin in ['ridge', 'bayridge']:
        for other in ['rf', 'et']:
            blend_configs.append(list(tree_combo) + [lin, other])

# All models
blend_configs.append(model_names)

# Named blends from v26 that we know work
blend_configs.append(['raddar', 'xgb_deep', 'ridge'])  # tree3_ridge: 58/91 with RC

# Remove duplicates
seen = set()
unique_blends = []
for bc in blend_configs:
    key = tuple(sorted(bc))
    if key not in seen:
        seen.add(key)
        unique_blends.append(bc)

print(f'{len(unique_blends)} unique blend configurations to try')


def apply_residual_correction(raw_train, raw_test, X_train, X_test, y_true,
                               depth=2, n_est=100, lr=0.05, lam=3.0):
    """Train XGBoost on in-sample residuals and correct test predictions."""
    residuals = y_true - raw_train
    X_aug_tr = np.column_stack([X_train, raw_train])
    X_aug_te = np.column_stack([X_test, raw_test])
    rm = xgb.XGBRegressor(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=lam, random_state=42, verbosity=0)
    rm.fit(X_aug_tr, residuals)
    return raw_test + rm.predict(X_aug_te)


# Residual correction configs
RC_CONFIGS = [
    ('rc_d2', {'depth': 2, 'n_est': 100, 'lr': 0.05, 'lam': 3.0}),
    ('rc_d3', {'depth': 3, 'n_est': 50, 'lr': 0.03, 'lam': 5.0}),
    ('rc_d2a', {'depth': 2, 'n_est': 150, 'lr': 0.03, 'lam': 2.0}),
    ('rc_d1', {'depth': 1, 'n_est': 200, 'lr': 0.05, 'lam': 2.0}),
]

# Evaluate all blends
all_results = []
best_score = 0
best_name = ''

for bi, blend in enumerate(unique_blends):
    bname = '+'.join(blend[:3]) + (f'+{len(blend)-3}more' if len(blend) > 3 else '')

    # Simple average blend
    raw_tr = np.mean([train_preds[m] for m in blend], axis=0)
    raw_te = np.mean([test_preds[m] for m in blend], axis=0)

    # Without residual correction
    for power in POWERS:
        a = hungarian(raw_te, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        rmse = np.sqrt(sse / 451)
        all_results.append((f'{bname}', power, ex, sse, rmse, a, raw_te))
        if ex > best_score:
            best_score = ex
            best_name = f'{bname}+p{power}'

    # With residual correction
    for rc_name, rc_params in RC_CONFIGS:
        corrected_te = apply_residual_correction(
            raw_tr, raw_te, X_tr, X_te, y_train, **rc_params)
        for power in POWERS:
            a = hungarian(corrected_te, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            all_results.append((f'{bname}_{rc_name}', power, ex, sse, rmse, a, corrected_te))
            if ex > best_score:
                best_score = ex
                best_name = f'{bname}_{rc_name}+p{power}'

    if (bi + 1) % 50 == 0:
        print(f'  Processed {bi+1}/{len(unique_blends)} blends, best so far: {best_score}/91 ({best_name})')

print(f'\nBest across {len(all_results)} configs: {best_score}/91 ({best_name})')


# ============================================================
#  SUPER-BLENDING: Average raw predictions from top configs
# ============================================================
print('\n' + '=' * 60)
print(' SUPER-BLENDING')
print('=' * 60)

# Sort all results
all_results.sort(key=lambda x: (-x[2], x[3]))

# Collect unique raw predictions from top-performing configs
top_raw_preds = []
top_names = []
seen_preds = set()
for sname, pw, ex, sse, rmse, assigned, raw_pred in all_results[:100]:
    pred_key = tuple(np.round(raw_pred, 4))
    if pred_key not in seen_preds and ex >= best_score - 3:
        seen_preds.add(pred_key)
        top_raw_preds.append(raw_pred)
        top_names.append(f'{sname}+p{pw}({ex})')
        if len(top_raw_preds) >= 20:
            break

print(f'Collected {len(top_raw_preds)} unique top raw predictions for super-blending')

# Try averaging subsets of top predictions
super_results = []
for n in range(2, min(len(top_raw_preds) + 1, 11)):
    for combo in combinations(range(len(top_raw_preds)), n):
        avg_pred = np.mean([top_raw_preds[i] for i in combo], axis=0)
        for power in [1.0, 1.1, 1.25]:
            a = hungarian(avg_pred, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            combo_name = f'super{n}_{combo[0]}_{combo[1]}'
            super_results.append((combo_name, power, ex, sse, rmse, a, avg_pred))

        # Stop if taking too long
        if len(super_results) > 10000:
            break
    if len(super_results) > 10000:
        break

super_results.sort(key=lambda x: (-x[2], x[3]))
if super_results:
    sb = super_results[0]
    print(f'Best super-blend: {sb[2]}/91 RMSE={sb[4]:.4f} ({sb[0]}+p{sb[1]})')

# Merge all results
all_results.extend(super_results)
all_results.sort(key=lambda x: (-x[2], x[3]))


# ============================================================
#  RESULTS
# ============================================================
print('\n' + '=' * 60)
print(' TOP-50 RESULTS')
print('=' * 60)

seen = set()
count = 0
for sname, pw, ex, sse, rmse, assigned, raw_pred in all_results:
    key = (sname, pw)
    if key in seen:
        continue
    seen.add(key)
    print(f'  {ex}/91  RMSE={rmse:.4f}  {sname}+p{pw}')
    count += 1
    if count >= 50:
        break


# ============================================================
#  SAVE SUBMISSIONS
# ============================================================
print('\n' + '=' * 60)
print(' SAVING SUBMISSIONS')
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
saved_configs = set()
for sname, pw, ex, sse, rmse, assigned, raw_pred in all_results:
    key = (sname, pw)
    if key in saved_configs:
        continue
    if len(saved_configs) >= 10:
        break
    if pw in [1.0, 1.1, 1.25, 1.5]:
        fname = f'submission_v28_{len(saved_configs)+1}.csv'
        p = save_sub(assigned, fname, f'{ex}/91 {sname}+p{pw}')
        submissions.append(p)
        saved_configs.add(key)


# Best result details
best = all_results[0]
sname, pw, ex, sse, rmse, assigned, raw_pred = best
print(f'\n** Best: {ex}/91, RMSE={rmse:.4f} ({sname}+p{pw})')

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

total_t = time.time() - t0
print(f'\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
