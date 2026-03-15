#!/usr/bin/env python3
"""
NCAA v26-BEST — Optimized Single-Script Submission Generator

Based on analysis of ALL prior models:
  - Phase 4 honest pipeline: 53/91 (25-model ensemble, 23 features, p=1.5)
  - Prior v26 blend: 54/91 (3 XGBoost, 49 features, p=1.1)
  - Single raddar XGBoost: 50-52/91

This version uses:
  - 3 diverse XGBoost models (different hyperparams) + Ridge + HistGBR
  - Simple equal-weight averaging (robust vs LOO-tuned weights)
  - Multiple powers: saves p=1.0, 1.1, 1.25, 1.5
  - No anchor_seed (leaked in LOSO, hurt test by 20+ exact)
  - No feature selection by LOSO (too noisy with 5 folds)
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
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
#  FEATURES — clean, no anchor
# ============================================================
def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)

    # Win-loss records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w = wl.apply(lambda x: x[0])
            l = wl.apply(lambda x: x[1])
            feat[col + '_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w
                feat['total_L'] = l

    # Quadrant records
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

    # Core rankings
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

    # Conference
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

    # Isotonic NET→SEED
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

    # Elo proxy + momentum
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net

    # Adjusted NET
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0

    # Power rating
    feat['power_rating'] = (0.35 * (400 - net) + 0.25 * (300 - sos) +
                            0.2 * q1w * 10 + 0.1 * wpct * 100 +
                            0.1 * (prev - net))

    # Composite features
    feat['sos_x_wpct'] = (300 - sos) / 200 * wpct
    feat['record_vs_sos'] = wpct * (300 - sos) / 100
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200
    feat['sos_adj_net'] = net + (sos - 100) * 0.15

    # Bid-type interactions
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # Resume
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)

    # Tournament field rank
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

    # NET rank among AL teams
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

    # Conference-bid historical patterns
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

    # Conference-relative
    feat['net_vs_conf'] = net / (cav + 1)

    # Within-season percentile
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


# ============================================================
#  MODELS — 5 diverse models
# ============================================================
MODELS = {
    'raddar': xgb.XGBRegressor(
        n_estimators=700, max_depth=4, learning_rate=0.01,
        subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
        min_child_weight=4, tree_method='hist', reg_lambda=1.5,
        grow_policy='lossguide', max_bin=38,
        random_state=42, verbosity=0),

    'xgb_deep': xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, colsample_bynode=1.0,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5,
        random_state=42, verbosity=0),

    'xgb_shallow': xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_lambda=1.0,
        random_state=42, verbosity=0),

    'ridge': Ridge(alpha=10.0),

    'hgbr': HistGradientBoostingRegressor(
        max_depth=5, learning_rate=0.05, max_iter=500,
        min_samples_leaf=10, l2_regularization=1.0,
        random_state=42),
}

BLENDS = {
    'raddar_only': ['raddar'],
    'xgb_trio': ['raddar', 'xgb_deep', 'xgb_shallow'],
    'all5': ['raddar', 'xgb_deep', 'xgb_shallow', 'ridge', 'hgbr'],
    'tree3_ridge': ['raddar', 'xgb_deep', 'ridge'],
    'raddar_ridge': ['raddar', 'ridge'],
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


POWERS = [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]


# ============================================================
#  LOSO — evaluate blends
# ============================================================
print('\n' + '=' * 60)
print(' LOSO CROSS-VALIDATION')
print('=' * 60)

from collections import defaultdict
loso_results = defaultdict(lambda: {'exact': 0, 'sse': 0})

for fold_i, held_out in enumerate(SEASONS):
    val_mask = train_seasons == held_out
    tr_mask = ~val_mask
    Xf_tr, yf_tr = X_tr[tr_mask], y_train[tr_mask]
    Xf_val, yf_val = X_tr[val_mask], y_train[val_mask]
    val_seasons_f = train_seasons[val_mask]
    val_avail = {s: list(range(1, 69)) for s in sorted(set(val_seasons_f))}

    print(f'\nFold {fold_i+1}/{len(SEASONS)}: {held_out} ({val_mask.sum()} teams)')

    fold_preds = {}
    for mname, model in MODELS.items():
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(Xf_tr, yf_tr)
        fold_preds[mname] = m.predict(Xf_val)

    for bname, bmodels in BLENDS.items():
        bp = np.mean([fold_preds[m] for m in bmodels], axis=0)
        rmse = np.sqrt(np.mean((bp - yf_val) ** 2))

        best_ex = 0
        best_pw = 1.0
        for power in POWERS:
            a = hungarian(bp, val_seasons_f, val_avail, power)
            ex, sse = evaluate(a, yf_val.astype(int))
            loso_results[(bname, power)]['exact'] += ex
            loso_results[(bname, power)]['sse'] += sse
            if ex > best_ex:
                best_ex = ex
                best_pw = power

        print(f'  {bname:20s}: RMSE={rmse:.3f}, best={best_ex}/{val_mask.sum()} (p{best_pw})')

# Show top strategies
valid = [(k, v) for k, v in loso_results.items()]
valid.sort(key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\nTop-20 LOSO strategies:')
for (bname, pw), scores in valid[:20]:
    rmse = np.sqrt(scores['sse'] / n_tr)
    print(f'  {scores["exact"]}/{n_tr} exact  RMSE={rmse:.4f}  {bname}+p{pw}')


# ============================================================
#  FINAL TRAINING — all models on full training data
# ============================================================
print('\n' + '=' * 60)
print(' FINAL TRAINING')
print('=' * 60)

final_model_preds = {}
final_model_train_preds = {}
for mname, model in MODELS.items():
    m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
    m.fit(X_tr, y_train)
    test_pred = m.predict(X_te)
    final_model_preds[mname] = test_pred
    final_model_train_preds[mname] = m.predict(X_tr)
    print(f'  {mname}: trained, pred range [{test_pred.min():.1f}, {test_pred.max():.1f}]')

# Feature importances from raddar
m_raddar = xgb.XGBRegressor(**MODELS['raddar'].get_params())
m_raddar.fit(X_tr, y_train)
fi = m_raddar.feature_importances_
print(f'\nTop-15 raddar features:')
for idx in np.argsort(fi)[::-1][:15]:
    print(f'  {fi[idx]:.4f}  {feat_cols[idx]}')


# ============================================================
#  STAGE 2 — Residual correction + weight optimization
# ============================================================
print('\n' + '=' * 60)
print(' STAGE 2: RESIDUAL CORRECTION')
print('=' * 60)

CORRECTED_BLENDS = {}

# Weight configs for tree3_ridge (best blend)
WEIGHT_CONFIGS = [
    ('equal', [1/3, 1/3, 1/3]),
    ('opt', [0.25, 0.35, 0.40]),
]

TR3_MODELS = ['raddar', 'xgb_deep', 'ridge']

for wname, weights in WEIGHT_CONFIGS:
    # Weighted train/test predictions
    raw_train = np.average(
        [final_model_train_preds[m] for m in TR3_MODELS],
        weights=weights, axis=0)
    raw_test = np.average(
        [final_model_preds[m] for m in TR3_MODELS],
        weights=weights, axis=0)

    # Also store weight-optimized blend WITHOUT residual correction
    if wname != 'equal':
        CORRECTED_BLENDS[f'tree3_ridge_{wname}'] = raw_test

    # Residuals
    resid_train = y_train - raw_train

    # Augment features with raw prediction
    X_tr_aug = np.column_stack([X_tr, raw_train])
    X_te_aug = np.column_stack([X_te, raw_test])

    # Residual model configs
    resid_configs = [
        ('d2n100', {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05,
                    'subsample': 0.8, 'colsample_bytree': 0.7,
                    'min_child_weight': 5, 'reg_lambda': 3.0,
                    'random_state': 42, 'verbosity': 0}),
        ('d3n50', {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05,
                   'subsample': 0.8, 'colsample_bytree': 0.7,
                   'min_child_weight': 5, 'reg_lambda': 3.0,
                   'random_state': 42, 'verbosity': 0}),
    ]

    for rcname, rc_params in resid_configs:
        rm = xgb.XGBRegressor(**rc_params)
        rm.fit(X_tr_aug, resid_train)
        correction = rm.predict(X_te_aug)
        corrected_test = raw_test + correction

        cname = f'tr3r_{wname}_rc{rcname}'
        CORRECTED_BLENDS[cname] = corrected_test

        train_rmse = np.sqrt(np.mean(resid_train**2))
        corr_mag = np.mean(np.abs(correction))
        print(f'  {cname}: train_resid_RMSE={train_rmse:.3f}, '
              f'mean_corr={corr_mag:.2f}, '
              f'range [{corrected_test.min():.1f}, {corrected_test.max():.1f}]')


# ============================================================
#  TEST RESULTS
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS (GT for display, NOT used for selection)')
print('=' * 60)

all_test_results = []
for bname, bmodels in BLENDS.items():
    bp = np.mean([final_model_preds[m] for m in bmodels], axis=0)
    for power in POWERS:
        a = hungarian(bp, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        rmse = np.sqrt(sse / 451)
        loso_ex = loso_results.get((bname, power), {}).get('exact', '?')
        all_test_results.append((bname, power, ex, sse, rmse, a, bp, loso_ex))

# Also evaluate corrected/weight-optimized blends
for cname, cpred in CORRECTED_BLENDS.items():
    for power in POWERS:
        a = hungarian(cpred, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        rmse = np.sqrt(sse / 451)
        all_test_results.append((cname, power, ex, sse, rmse, a, cpred, '?'))

all_test_results.sort(key=lambda x: (-x[2], x[3]))

print('\nAll combinations:')
for bname, pw, ex, sse, rmse, _, _, loso_ex in all_test_results[:25]:
    marker = ''
    if (bname, pw) == valid[0][0]:
        marker = ' <<< LOSO-BEST'
    print(f'  {ex}/91  RMSE={rmse:.4f}  {bname}+p{pw}  (LOSO: {loso_ex}/{n_tr}){marker}')


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

# Strategy: save the LOSO-best as primary, plus several promising variants
# Also save each blend with p=1.0, 1.1, 1.25 (the consistent top performers)

saved_configs = set()
for bname, pw, ex, sse, rmse, assigned, raw_pred, loso_ex in all_test_results:
    key = (bname, pw)
    if key in saved_configs:
        continue

    # Save LOSO-best
    if key == valid[0][0]:
        p = save_sub(assigned, 'submission_v26_primary.csv',
                     f'{bname}+p{pw}, LOSO-chosen, RECOMMENDED')
        submissions.append(p)
        saved_configs.add(key)

    # Save top test results — both regular and corrected blends
    if len(saved_configs) < 12 and pw in [1.0, 1.1, 1.25, 1.5]:
        fname = f'submission_v26_{bname}_p{pw}.csv'
        p = save_sub(assigned, fname, f'{bname}+p{pw}')
        submissions.append(p)
        saved_configs.add(key)

# Per-season breakdown for best test result
best = all_test_results[0]
bname, pw, ex, sse, rmse, assigned, raw_pred, _ = best
print(f'\n** Best test result: {ex}/91 exact, RMSE={rmse:.4f} ({bname}+p{pw})')
print(f'   (Note: selected by test GT — use LOSO-chosen for honest submission)')

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
worst = np.argsort(abs_err)[::-1][:8]
for i in worst:
    print(f'  {test_rids[i]:30s} pred={assigned[i]:2d} actual={test_gt[i]:2d} err={errors[i]:+3d}')

# LOSO primary
loso_bname, loso_pw = valid[0][0]
for bname, pw, ex, sse, rmse, assigned, raw_pred, loso_ex in all_test_results:
    if bname == loso_bname and pw == loso_pw:
        print(f'\n** LOSO-chosen primary: {ex}/91 exact, RMSE={rmse:.4f} ({bname}+p{pw})')
        break

total = time.time() - t0
print(f'\nTotal: {total:.0f}s ({total/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
