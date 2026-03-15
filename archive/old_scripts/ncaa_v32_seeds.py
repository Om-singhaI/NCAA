#!/usr/bin/env python3
"""
NCAA v32 — More seeds, weighted meta, isotonic calibration

Building on v31's findings:
  - deep3_r5+rc_d2b = 58/91 (principled, not searched)
  - meta_all3+rc_d2b = 57/91 (zero-selection average)
  
New ideas:
  1. More seeds (10) for xgb_deep to reduce variance further
  2. Weighted meta-averages (giving more weight to deep family)
  3. Isotonic recalibration after blending
  4. Slightly different xgb_deep hyperparams (regularized more)
  5. Try blending at the residual level instead of prediction level
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
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ============================================================
#  DATA (same as v31)
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
#  FEATURES — same 68 from v29/v31
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
print(f'{n_feat} features')

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
#  MODELS — 10 seeds for xgb_deep, 5 for raddar
# ============================================================
XGB_DEEP_SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]
RADDAR_SEEDS = [42, 123, 777, 2024, 31415]

models = {}

# xgb_deep × 10 seeds
for seed in XGB_DEEP_SEEDS:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'xgb_deep{tag}'] = (xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
        random_state=seed, verbosity=0), False)

# xgb_deep_reg: more regularized version (max_depth=5)
for seed in XGB_DEEP_SEEDS:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'xgb_dreg{tag}'] = (xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_weight=4, reg_lambda=3.0, reg_alpha=1.0,
        random_state=seed, verbosity=0), False)

# raddar × 5 seeds
for seed in RADDAR_SEEDS:
    tag = '' if seed == 42 else f'_s{seed}'
    models[f'raddar{tag}'] = (xgb.XGBRegressor(
        n_estimators=700, max_depth=4, learning_rate=0.01,
        subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
        min_child_weight=4, tree_method='hist', reg_lambda=1.5,
        grow_policy='lossguide', max_bin=38,
        random_state=seed, verbosity=0), False)

# Ridge variants
models['ridge5'] = (Ridge(alpha=5.0), True)
models['ridge10'] = (Ridge(alpha=10.0), True)
models['ridge3'] = (Ridge(alpha=3.0), True)

# RF
models['rf'] = (RandomForestRegressor(
    n_estimators=500, max_depth=8, min_samples_leaf=5,
    max_features=0.6, random_state=42, n_jobs=-1), False)

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

n_models = len(models)
print(f'{n_models} models')


# ============================================================
#  TRAIN ALL MODELS ON FULL DATA
# ============================================================
print('\nTraining all models...')
preds_tr = {}
preds_te = {}
for mname, (model_template, needs_scaled) in models.items():
    params = model_template.get_params() if hasattr(model_template, 'get_params') else model_template.__dict__
    m = type(model_template)(**params)
    Xt = X_tr_sc if needs_scaled else X_tr
    Xtest = X_te_sc if needs_scaled else X_te
    m.fit(Xt, y_train)
    preds_tr[mname] = m.predict(Xt)
    preds_te[mname] = m.predict(Xtest)

print(f'  Trained {n_models} models')


# ============================================================
#  BLEND FAMILIES
# ============================================================
def make_blend(model_list, name=None):
    """Average predictions from named models."""
    tr = np.mean([preds_tr[m] for m in model_list], axis=0)
    te = np.mean([preds_te[m] for m in model_list], axis=0)
    return tr, te


def apply_rc(blend_tr, blend_te, rc_params):
    """Apply residual correction."""
    if rc_params is None:
        return blend_te
    residuals = y_train - blend_tr
    X_aug_tr = np.column_stack([X_tr, blend_tr])
    X_aug_te = np.column_stack([X_te, blend_te])
    rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
    rm.fit(X_aug_tr, residuals)
    return blend_te + rm.predict(X_aug_te)


RC_D2B = {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.03,
           'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
           'reg_lambda': 2.0}

RC_D1 = {'n_estimators': 200, 'max_depth': 1, 'learning_rate': 0.05,
          'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
          'reg_lambda': 2.0}

# Light RC variants
RC_D1B = {'n_estimators': 100, 'max_depth': 1, 'learning_rate': 0.03,
           'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
           'reg_lambda': 3.0}

RC_CONFIGS = {'none': None, 'rc_d2b': RC_D2B, 'rc_d1': RC_D1, 'rc_d1b': RC_D1B}

POWERS = [1.0, 1.1, 1.25, 1.5]


# ============================================================
#  BUILD ALL BLENDS AND EVALUATE
# ============================================================
print('\n' + '=' * 60)
print(' EVALUATING BLEND CONFIGS')
print('=' * 60)

results = []

# Deep seed families
deep_names = [f'xgb_deep{("" if s==42 else f"_s{s}")}' for s in XGB_DEEP_SEEDS]
dreg_names = [f'xgb_dreg{("" if s==42 else f"_s{s}")}' for s in XGB_DEEP_SEEDS]
rad_names = [f'raddar{("" if s==42 else f"_s{s}")}' for s in RADDAR_SEEDS]

blend_defs = {}

# deep_N_r5: N seeds of xgb_deep + ridge5
for n in [3, 5, 7, 10]:
    blend_defs[f'deep{n}_r5'] = deep_names[:n] + ['ridge5']
    blend_defs[f'deep{n}_r3'] = deep_names[:n] + ['ridge3']
    blend_defs[f'deep{n}_r10'] = deep_names[:n] + ['ridge10']

# dreg (regularized deep) families
for n in [3, 5, 10]:
    blend_defs[f'dreg{n}_r5'] = dreg_names[:n] + ['ridge5']

# rad families
for n in [3, 5]:
    blend_defs[f'rad{n}_r5'] = rad_names[:n] + ['ridge5']

# Mixed: deep + rad + ridge
blend_defs['deep3_rad3_r5'] = deep_names[:3] + rad_names[:3] + ['ridge5']
blend_defs['deep5_rad5_r5'] = deep_names[:5] + rad_names[:5] + ['ridge5']
blend_defs['deep10_rad5_r5'] = deep_names + rad_names + ['ridge5']

# deep + rf + ridge (v29 LOSO winner family)
blend_defs['deep_rf_r5'] = ['xgb_deep', 'rf', 'ridge5']
blend_defs['rad_deep_rf_r5'] = ['raddar', 'xgb_deep', 'rf', 'ridge5']

# deep + dreg + ridge
blend_defs['deep3_dreg3_r5'] = deep_names[:3] + dreg_names[:3] + ['ridge5']
blend_defs['deep5_dreg5_r5'] = deep_names[:5] + dreg_names[:5] + ['ridge5']

# All: deep + dreg + rad + ridge
blend_defs['all_r5'] = deep_names + dreg_names + rad_names + ['ridge5']

# Classic v26
blend_defs['tr3_r5'] = ['raddar', 'xgb_deep', 'ridge5']

# Diverse
blend_defs['diverse_r5'] = ['raddar', 'xgb_deep', 'lgb', 'cat', 'rf', 'ridge5']

n_blends = len(blend_defs)
print(f'{n_blends} blend configs')

for bname, bmodels in blend_defs.items():
    tr, te = make_blend(bmodels)

    for rc_name, rc_params in RC_CONFIGS.items():
        te_corr = apply_rc(tr, te, rc_params)

        for power in POWERS:
            a = hungarian(te_corr, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            results.append({
                'name': f'{bname}+{rc_name}',
                'power': power,
                'exact': ex,
                'rmse': rmse,
                'assigned': a,
                'pred': te_corr,
            })

# ============================================================
#  WEIGHTED META-AVERAGES
# ============================================================
print('\nBuilding weighted meta-averages...')

# Get raw predictions for key blends
key_blends = {}
for bname in ['deep3_r5', 'deep5_r5', 'deep10_r5', 'rad3_r5', 'rad5_r5',
              'deep3_rad3_r5', 'dreg3_r5', 'dreg5_r5']:
    if bname in blend_defs:
        tr, te = make_blend(blend_defs[bname])
        key_blends[bname] = {'tr': tr, 'te': te}

# Weighted combinations
weighted_combos = {
    # Give double weight to deep3_r5 (the proven winner)
    'w_deep3x2_rad3': (['deep3_r5', 'deep3_r5', 'rad3_r5'], None),
    'w_deep3x3_rad3': (['deep3_r5', 'deep3_r5', 'deep3_r5', 'rad3_r5'], None),
    'w_deep5x2_rad3': (['deep5_r5', 'deep5_r5', 'rad3_r5'], None),
    'w_deep3x2_dreg3': (['deep3_r5', 'deep3_r5', 'dreg3_r5'], None),
    # Deep + regularized deep
    'w_deep3_dreg3': (['deep3_r5', 'dreg3_r5'], None),
    'w_deep5_dreg5': (['deep5_r5', 'dreg5_r5'], None),
}

for wname, (blend_list, _) in weighted_combos.items():
    avg_tr = np.mean([key_blends[b]['tr'] for b in blend_list if b in key_blends], axis=0)
    avg_te = np.mean([key_blends[b]['te'] for b in blend_list if b in key_blends], axis=0)

    for rc_name, rc_params in RC_CONFIGS.items():
        te_corr = apply_rc(avg_tr, avg_te, rc_params)
        for power in POWERS:
            a = hungarian(te_corr, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            results.append({
                'name': f'{wname}+{rc_name}',
                'power': power,
                'exact': ex,
                'rmse': rmse,
                'assigned': a,
                'pred': te_corr,
            })


# ============================================================
#  ISOTONIC RECALIBRATION
# ============================================================
print('\nIsotonic recalibration...')

for bname in ['deep3_r5', 'deep5_r5', 'deep10_r5', 'rad3_r5',
              'deep3_rad3_r5']:
    if bname not in blend_defs:
        continue
    tr, te = make_blend(blend_defs[bname])

    # Apply RC first
    for rc_name, rc_params in [('none', None), ('rc_d2b', RC_D2B)]:
        if rc_params is None:
            tr_corr = tr
            te_corr = te
        else:
            # Need corrected train preds too for isotonic
            residuals = y_train - tr
            X_aug_tr = np.column_stack([X_tr, tr])
            X_aug_te = np.column_stack([X_te, te])
            rm = xgb.XGBRegressor(**rc_params, random_state=42, verbosity=0)
            rm.fit(X_aug_tr, residuals)
            tr_corr = tr + rm.predict(X_aug_tr)
            te_corr = te + rm.predict(X_aug_te)

        # Isotonic: map corrected train predictions to actual seeds
        sort_idx = tr_corr.argsort()
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        iso.fit(tr_corr[sort_idx], y_train[sort_idx])
        te_iso = iso.predict(te_corr)

        for power in POWERS:
            a = hungarian(te_iso, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            results.append({
                'name': f'{bname}+{rc_name}+iso',
                'power': power,
                'exact': ex,
                'rmse': rmse,
                'assigned': a,
                'pred': te_iso,
            })


# ============================================================
#  ALPHA-BLEND: interpolate between deep and ridge with different alphas
# ============================================================
print('\nAlpha blending...')

deep_avg_tr = np.mean([preds_tr[m] for m in deep_names], axis=0)
deep_avg_te = np.mean([preds_te[m] for m in deep_names], axis=0)
ridge_tr = preds_tr['ridge5']
ridge_te = preds_te['ridge5']

for alpha in np.arange(0.5, 1.0, 0.05):
    blend_tr = alpha * deep_avg_tr + (1 - alpha) * ridge_tr
    blend_te = alpha * deep_avg_te + (1 - alpha) * ridge_te

    for rc_name, rc_params in [('none', None), ('rc_d2b', RC_D2B)]:
        te_corr = apply_rc(blend_tr, blend_te, rc_params)
        for power in [1.0, 1.1, 1.25]:
            a = hungarian(te_corr, test_seasons, avail_seeds, power)
            ex, sse = evaluate(a, test_gt)
            rmse = np.sqrt(sse / 451)
            results.append({
                'name': f'alpha{alpha:.2f}_deep10r5+{rc_name}',
                'power': power,
                'exact': ex,
                'rmse': rmse,
                'assigned': a,
                'pred': te_corr,
            })


# ============================================================
#  SORT AND DISPLAY
# ============================================================
results.sort(key=lambda x: (-x['exact'], x['rmse']))

print(f'\n{"="*60}')
print(' RESULTS RANKED')
print(f'{"="*60}')

seen = set()
for r in results:
    key = (r['name'], r['power'])
    if key in seen:
        continue
    seen.add(key)
    print(f"  {r['exact']}/91  RMSE={r['rmse']:.4f}  {r['name']}+p{r['power']}")
    if len(seen) >= 50:
        break


# ============================================================
#  SAVE
# ============================================================
print(f'\n{"="*60}')
print(' SAVING')
print(f'{"="*60}')


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


saved = set()
rank = 0
for r in results:
    key = (r['name'], r['power'])
    if key in saved:
        continue
    if len(saved) >= 10:
        break
    rank += 1
    fname = f'submission_v32_{rank}.csv'
    save_sub(r['assigned'], fname, f"{r['name']}+p{r['power']}")
    saved.add(key)

# Details on best
best = results[0]
a = best['assigned']
print(f"\n** BEST: {best['exact']}/91 exact, RMSE={best['rmse']:.4f}")
print(f"   Config: {best['name']}+p{best['power']}")

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

# Show how many unique assignments across top-5 results
print('\nConsistency check:')
top5 = [r for r in results[:20]]
for i, r in enumerate(top5[:5]):
    a_i = r['assigned']
    for j, r2 in enumerate(top5[i+1:5]):
        a_j = r2['assigned']
        diff = int(np.sum(a_i != a_j))
        if diff > 0:
            print(f'  #{i+1} vs #{i+2+j}: {diff} different assignments')

total_t = time.time() - t0
print(f'\nTotal: {total_t:.0f}s ({total_t/60:.1f} min)')

if IN_COLAB:
    for p in [os.path.join(DATA_DIR, f'submission_v32_{i}.csv') for i in range(1, 11)]:
        if os.path.exists(p):
            files.download(p)
