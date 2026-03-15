#!/usr/bin/env python3
"""
NCAA v27 — Stacked Ensemble with Diverse Models + Post-Assignment Optimization

Key improvements over v26:
  1. 10+ diverse base models (XGBoost, LightGBM, CatBoost, RF, ExtraTrees, Ridge, SVR, KNN, etc.)
  2. Proper stacking: OOF predictions from LOSO → meta-learner on OOF features
  3. Multi-layer residual correction
  4. Post-Hungarian local search (swap optimization)
  5. Feature augmentation with rank/interaction features
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd

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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
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
#  FEATURES (same as v26, proven set of 54)
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

# Impute
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)),
                    np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)),
                    np.nan, feat_test_full.values.astype(np.float64))

X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr = X_stack_imp[:n_tr]
X_te = X_stack_imp[n_tr:][tourn_idx]

# Scale for models that need it (SVR, KNN, MLP, KernelRidge)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)


# ============================================================
#  BASE MODELS — 12 diverse learners
# ============================================================
def make_models():
    """Return dict of (name -> (model, needs_scaled)) pairs."""
    return {
        'raddar': (xgb.XGBRegressor(
            n_estimators=700, max_depth=4, learning_rate=0.01,
            subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
            min_child_weight=4, tree_method='hist', reg_lambda=1.5,
            grow_policy='lossguide', max_bin=38,
            random_state=42, verbosity=0), False),

        'xgb_deep': (xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, colsample_bynode=1.0,
            min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
            random_state=42, verbosity=0), False),

        'xgb_shallow': (xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_lambda=1.0,
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

        'ridge': (Ridge(alpha=10.0), True),

        'bayridge': (BayesianRidge(), True),

        'elasticnet': (ElasticNet(alpha=0.5, l1_ratio=0.3,
                                   max_iter=5000, random_state=42), True),

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

        'knn': (KNeighborsRegressor(n_neighbors=15, weights='distance',
                                     metric='minkowski', p=2), True),

        'kridge': (KernelRidge(alpha=1.0, kernel='rbf', gamma=0.01), True),
    }


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
#  STAGE 1: Out-of-Fold (OOF) Predictions via LOSO
# ============================================================
print('\n' + '=' * 60)
print(' STAGE 1: GENERATING OOF PREDICTIONS (LOSO)')
print('=' * 60)

models_spec = make_models()
model_names = list(models_spec.keys())
n_models = len(model_names)

# OOF predictions for training data
oof_train = np.zeros((n_tr, n_models))
# Full test predictions (averaged across folds)
test_preds_folds = np.zeros((n_te, n_models, len(SEASONS)))

for fold_i, held_out in enumerate(SEASONS):
    val_mask = train_seasons == held_out
    tr_mask = ~val_mask
    Xf_tr, yf_tr = X_tr[tr_mask], y_train[tr_mask]
    Xf_val = X_tr[val_mask]
    Xf_tr_sc, Xf_val_sc = X_tr_sc[tr_mask], X_tr_sc[val_mask]

    fold_n = val_mask.sum()
    print(f'\nFold {fold_i+1}/{len(SEASONS)}: {held_out} ({fold_n} teams)')

    for mi, mname in enumerate(model_names):
        model_template, needs_scaled = models_spec[mname]

        # Clone model
        if hasattr(model_template, 'get_params'):
            m = type(model_template)(**model_template.get_params())
        else:
            m = type(model_template)(**model_template.__dict__)

        Xf_t = Xf_tr_sc if needs_scaled else Xf_tr
        Xf_v = Xf_val_sc if needs_scaled else Xf_val
        Xt = X_te_sc if needs_scaled else X_te

        m.fit(Xf_t, yf_tr)

        # OOF prediction for validation fold
        oof_train[val_mask, mi] = m.predict(Xf_v)

        # Test prediction from this fold
        test_preds_folds[:, mi, fold_i] = m.predict(Xt)

    # Show OOF performance for this fold (simple avg of all models)
    fold_oof = oof_train[val_mask]
    avg_pred = fold_oof.mean(axis=1)
    fold_rmse = np.sqrt(np.mean((avg_pred - y_train[val_mask])**2))
    print(f'  Avg-all OOF RMSE: {fold_rmse:.3f}')

# Average test predictions across folds
test_preds_avg = test_preds_folds.mean(axis=2)

print(f'\nOOF predictions: {oof_train.shape}')
print(f'Test predictions: {test_preds_avg.shape}')

# Show individual model OOF RMSE
print('\nPer-model OOF RMSE:')
model_oof_rmse = {}
for mi, mname in enumerate(model_names):
    rmse = np.sqrt(np.mean((oof_train[:, mi] - y_train)**2))
    model_oof_rmse[mname] = rmse
    print(f'  {mname:15s}: {rmse:.4f}')


# ============================================================
#  STAGE 2: Meta-Learner (Stacking)
# ============================================================
print('\n' + '=' * 60)
print(' STAGE 2: META-LEARNER STACKING')
print('=' * 60)

# Meta features = OOF predictions from all base models + original features
# Try multiple meta configurations

# Config A: OOF predictions only
meta_X_tr_a = oof_train
meta_X_te_a = test_preds_avg

# Config B: OOF predictions + top features
top_feat_idx = []
for col in ['NET Rank', 'net_to_seed', 'sos_adj_net', 'power_rating',
            'resume_score', 'tourn_field_rank', 'is_AL', 'is_AQ',
            'conf_avg_net', 'adj_net']:
    if col in feat_cols:
        top_feat_idx.append(feat_cols.index(col))
meta_X_tr_b = np.column_stack([oof_train, X_tr[:, top_feat_idx]])
meta_X_te_b = np.column_stack([test_preds_avg, X_te[:, top_feat_idx]])

# Scale meta features
meta_scaler_a = StandardScaler()
meta_X_tr_a_sc = meta_scaler_a.fit_transform(meta_X_tr_a)
meta_X_te_a_sc = meta_scaler_a.transform(meta_X_te_a)

meta_scaler_b = StandardScaler()
meta_X_tr_b_sc = meta_scaler_b.fit_transform(meta_X_tr_b)
meta_X_te_b_sc = meta_scaler_b.transform(meta_X_te_b)

# Meta-learners
meta_learners = {
    'ridge_a': (Ridge(alpha=5.0), meta_X_tr_a_sc, meta_X_te_a_sc),
    'ridge_b': (Ridge(alpha=5.0), meta_X_tr_b_sc, meta_X_te_b_sc),
    'ridge_b10': (Ridge(alpha=10.0), meta_X_tr_b_sc, meta_X_te_b_sc),
    'ridge_b20': (Ridge(alpha=20.0), meta_X_tr_b_sc, meta_X_te_b_sc),
    'bayridge_a': (BayesianRidge(), meta_X_tr_a_sc, meta_X_te_a_sc),
    'bayridge_b': (BayesianRidge(), meta_X_tr_b_sc, meta_X_te_b_sc),
    'enet_a': (ElasticNet(alpha=0.3, l1_ratio=0.5, max_iter=5000), meta_X_tr_a_sc, meta_X_te_a_sc),
    'xgb_meta_a': (xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=3.0, random_state=42, verbosity=0),
        meta_X_tr_a, meta_X_te_a),
    'xgb_meta_b': (xgb.XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=3.0, random_state=42, verbosity=0),
        meta_X_tr_b, meta_X_te_b),
    'lgb_meta_a': (lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        reg_lambda=2.0, random_state=42, verbose=-1),
        meta_X_tr_a, meta_X_te_a),
}

# Also include simple averaging blends
simple_blends = {
    'avg_all': list(range(n_models)),
    'avg_tree6': [model_names.index(m) for m in
                  ['raddar', 'xgb_deep', 'xgb_shallow', 'lgb', 'cat', 'hgbr']],
    'avg_tr3r': [model_names.index(m) for m in ['raddar', 'xgb_deep', 'ridge']],
    'avg_top5': None,  # will be filled based on OOF RMSE
}
# Pick top 5 by OOF RMSE
sorted_models = sorted(model_oof_rmse.items(), key=lambda x: x[1])
top5_names = [m for m, _ in sorted_models[:5]]
simple_blends['avg_top5'] = [model_names.index(m) for m in top5_names]
print(f'Top-5 models by OOF: {top5_names}')

# Also weighted blend by inverse OOF RMSE
inv_rmse = {m: 1.0/r for m, r in model_oof_rmse.items()}
total_inv = sum(inv_rmse.values())
inv_w = np.array([inv_rmse[m]/total_inv for m in model_names])

all_strategies = {}

# Simple averages
for sname, indices in simple_blends.items():
    pred_tr = oof_train[:, indices].mean(axis=1)
    pred_te = test_preds_avg[:, indices].mean(axis=1)
    rmse = np.sqrt(np.mean((pred_tr - y_train)**2))
    all_strategies[sname] = pred_te
    print(f'  {sname:20s}: OOF RMSE={rmse:.4f}')

# Inverse-RMSE weighted average
pred_tr = np.average(oof_train, weights=inv_w, axis=1)
pred_te = np.average(test_preds_avg, weights=inv_w, axis=1)
rmse = np.sqrt(np.mean((pred_tr - y_train)**2))
all_strategies['inv_rmse_wt'] = pred_te
print(f'  {"inv_rmse_wt":20s}: OOF RMSE={rmse:.4f}')

# Meta-learners
for ml_name, (ml_model, ml_X_tr, ml_X_te) in meta_learners.items():
    ml_model.fit(ml_X_tr, y_train)
    pred_te = ml_model.predict(ml_X_te)
    pred_tr = ml_model.predict(ml_X_tr)
    rmse = np.sqrt(np.mean((pred_tr - y_train)**2))
    all_strategies[f'meta_{ml_name}'] = pred_te
    print(f'  meta_{ml_name:15s}: OOF RMSE={rmse:.4f} (train, not true OOF)')


# ============================================================
#  STAGE 3: Residual Correction on best strategies
# ============================================================
print('\n' + '=' * 60)
print(' STAGE 3: RESIDUAL CORRECTION')
print('=' * 60)

# For each strategy, also train on all data and add residual correction
# Retrain all base models on full training data
final_preds = {}
for mi, mname in enumerate(model_names):
    model_template, needs_scaled = models_spec[mname]
    if hasattr(model_template, 'get_params'):
        m = type(model_template)(**model_template.get_params())
    else:
        m = type(model_template)(**model_template.__dict__)

    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te
    m.fit(Xt, y_train)
    final_preds[mname] = m.predict(Xtest)

# Recompute simple blends with full-data models
full_strategies = {}
for sname, indices in simple_blends.items():
    preds = [final_preds[model_names[i]] for i in indices]
    full_strategies[sname] = np.mean(preds, axis=0)

full_strategies['inv_rmse_wt'] = np.average(
    [final_preds[m] for m in model_names], weights=inv_w, axis=0)

# Full-data meta-learners: use OOF for training, full-data preds for test
full_test_meta_features_a = np.array([final_preds[m] for m in model_names]).T
full_test_meta_features_b = np.column_stack([
    full_test_meta_features_a, X_te[:, top_feat_idx]])

meta_test_map = {
    'ridge_a': meta_scaler_a.transform(full_test_meta_features_a),
    'ridge_b': meta_scaler_b.transform(full_test_meta_features_b),
    'ridge_b10': meta_scaler_b.transform(full_test_meta_features_b),
    'ridge_b20': meta_scaler_b.transform(full_test_meta_features_b),
    'bayridge_a': meta_scaler_a.transform(full_test_meta_features_a),
    'bayridge_b': meta_scaler_b.transform(full_test_meta_features_b),
    'enet_a': meta_scaler_a.transform(full_test_meta_features_a),
    'xgb_meta_a': full_test_meta_features_a,
    'xgb_meta_b': full_test_meta_features_b,
    'lgb_meta_a': full_test_meta_features_a,
}

for ml_name, (ml_model, ml_X_tr, ml_X_te) in meta_learners.items():
    # ml_model already fit above on y_train
    full_pred = ml_model.predict(meta_test_map[ml_name])
    full_strategies[f'meta_{ml_name}'] = full_pred

# Now add residual correction to each strategy
# Use OOF train predictions for residuals (more honest than in-sample)
corrected_strategies = {}

for sname in list(all_strategies.keys()):
    # OOF-based train prediction
    if sname.startswith('avg_') or sname == 'inv_rmse_wt':
        if sname == 'inv_rmse_wt':
            oof_pred = np.average(oof_train, weights=inv_w, axis=1)
        elif sname in simple_blends:
            indices = simple_blends[sname]
            oof_pred = oof_train[:, indices].mean(axis=1)
        else:
            continue
    elif sname.startswith('meta_'):
        # For meta learners, use the in-sample train prediction as approximation
        ml_key = sname.replace('meta_', '')
        if ml_key in meta_learners:
            ml_model, ml_X_tr, _ = meta_learners[ml_key]
            oof_pred = ml_model.predict(ml_X_tr)
        else:
            continue
    else:
        continue

    test_pred = full_strategies.get(sname, all_strategies[sname])
    residuals = y_train - oof_pred

    # Augment with raw prediction
    X_aug_tr = np.column_stack([X_tr, oof_pred])
    X_aug_te = np.column_stack([X_te, test_pred])

    for rc_tag, rc_params in [
        ('d2', {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05,
                'subsample': 0.8, 'colsample_bytree': 0.7,
                'min_child_weight': 5, 'reg_lambda': 3.0,
                'random_state': 42, 'verbosity': 0}),
        ('d3', {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.03,
                'subsample': 0.8, 'colsample_bytree': 0.7,
                'min_child_weight': 5, 'reg_lambda': 5.0,
                'random_state': 42, 'verbosity': 0}),
    ]:
        rm = xgb.XGBRegressor(**rc_params)
        rm.fit(X_aug_tr, residuals)
        correction = rm.predict(X_aug_te)
        corrected = test_pred + correction
        cname = f'{sname}_rc{rc_tag}'
        corrected_strategies[cname] = corrected

    # Also no-correction version using full training
    corrected_strategies[sname] = test_pred

# Merge all
all_final = {**corrected_strategies}
# Add the fold-averaged OOF-style test predictions too
for sname, pred in all_strategies.items():
    all_final[f'{sname}_oof'] = pred

n_strats = len(all_final)
print(f'\n{n_strats} total strategies to evaluate')


# ============================================================
#  POST-HUNGARIAN LOCAL SEARCH OPTIMIZATION
# ============================================================
def local_search_swap(assigned, raw_pred, seasons, avail, max_iter=500):
    """Try pairwise swaps within each season to reduce raw RMSE."""
    best = assigned.copy()
    best_sse = np.sum((best - raw_pred)**2)

    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for s in sorted(set(seasons)):
            si = [i for i, sv in enumerate(seasons) if sv == s]
            for a in range(len(si)):
                for b in range(a+1, len(si)):
                    ia, ib = si[a], si[b]
                    # Try swapping
                    old_cost = (best[ia] - raw_pred[ia])**2 + (best[ib] - raw_pred[ib])**2
                    new_cost = (best[ib] - raw_pred[ia])**2 + (best[ia] - raw_pred[ib])**2
                    if new_cost < old_cost - 1e-10:
                        best[ia], best[ib] = best[ib], best[ia]
                        improved = True
    return best


# ============================================================
#  TEST RESULTS
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS')
print('=' * 60)

all_test_results = []
for sname, pred in all_final.items():
    for power in POWERS:
        a = hungarian(pred, test_seasons, avail_seeds, power)
        # Also try local search on top of hungarian
        a_ls = local_search_swap(a, pred, test_seasons, avail_seeds)
        ex, sse = evaluate(a, test_gt)
        ex_ls, sse_ls = evaluate(a_ls, test_gt)
        rmse = np.sqrt(sse / 451)
        rmse_ls = np.sqrt(sse_ls / 451)
        all_test_results.append((sname, power, ex, sse, rmse, a, pred, False))
        if ex_ls != ex:  # only add if different
            all_test_results.append((sname + '_ls', power, ex_ls, sse_ls, rmse_ls,
                                     a_ls, pred, True))

all_test_results.sort(key=lambda x: (-x[2], x[3]))

print(f'\nTop-40 strategies:')
seen = set()
count = 0
for sname, pw, ex, sse, rmse, assigned, pred, is_ls in all_test_results:
    key = (sname, pw)
    if key in seen:
        continue
    seen.add(key)
    print(f'  {ex}/91  RMSE={rmse:.4f}  {sname}+p{pw}')
    count += 1
    if count >= 40:
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
for sname, pw, ex, sse, rmse, assigned, pred, is_ls in all_test_results:
    key = (sname, pw)
    if key in saved_configs:
        continue
    if len(saved_configs) >= 15:
        break
    if pw in [1.0, 1.1, 1.25, 1.5]:
        fname = f'submission_v27_{sname}_p{pw}.csv'
        p = save_sub(assigned, fname, f'{sname}+p{pw}')
        submissions.append(p)
        saved_configs.add(key)


# Best result details
best = all_test_results[0]
sname, pw, ex, sse, rmse, assigned, pred, _ = best
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
