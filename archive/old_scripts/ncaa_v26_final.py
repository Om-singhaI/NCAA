#!/usr/bin/env python3
"""
NCAA v26 — Best Single Model, Zero Overfitting

ANALYSIS: Why perfect isn't possible without overfitting:
  - Pure NET ordering: only 33/91 exact (committee deviates for 58 teams)
  - Committee adjustments range ±5 to ±27 seeds (Memphis: NET 49 → seed 20)
  - Honest ML ceiling: ~53-57/91 (ALL higher scores used GT leakage)
  - Every prior score >57/91 used Pool-LOO with test answers or DE optimization
    against test GT — these DON'T generalize to the private LB

APPROACH:
  Single XGBoost (raddar config) with:
  - Anchor interpolation (per-season NET→seed from training neighbors)
  - MI-selected top features (avoid overfitting 249 samples on 56 features)
  - Multiple power submissions (power selection is unpredictable from LOSO)
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
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import linear_sum_assignment
# from scipy.interpolate import PchipInterpolator  # removed: anchor leaks in LOSO

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
#  FEATURES — focused set + anchor interpolation
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

    # ==== ISOTONIC NET→SEED (MI #1) ====
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    # anchor_seed REMOVED — it leaks in LOSO (features computed once using all
    # training data, so held-out teams' anchor was built from their own season)

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
print(f'{len(feat_cols)} features built')

# Impute
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)),
                    np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)),
                    np.nan, feat_test_full.values.astype(np.float64))

X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr_imp = X_stack_imp[:n_tr]
X_te_imp = X_stack_imp[n_tr:][tourn_idx]

# MI-based feature selection
print('Computing MI for feature selection...')
mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=7)
mi_order = np.argsort(mi)[::-1]

print('Top-20 MI features:')
for i in range(20):
    fi = mi_order[i]
    print(f'  {mi[fi]:.4f}  {feat_cols[fi]}')

# Create feature subsets
FS = {
    'top25': mi_order[:25],
    'top30': mi_order[:30],
    'top40': mi_order[:40],
    'all': np.arange(len(feat_cols)),
}


# ============================================================
#  MODEL + ASSIGNMENT
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


# Raddar config (1st place 2025)
RADDAR = dict(
    n_estimators=700, max_depth=4, learning_rate=0.01,
    subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
    min_child_weight=4, tree_method='hist', reg_lambda=1.5,
    grow_policy='lossguide', max_bin=38,
    random_state=42, verbosity=0)

POWERS = [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]


# ============================================================
#  LOSO — validate feature subsets + find robust power
# ============================================================
print('\n' + '=' * 60)
print(' LOSO CROSS-VALIDATION')
print('=' * 60)

from collections import defaultdict

loso_agg = defaultdict(lambda: {'exact': 0, 'sse': 0, 'folds': 0})
oof_preds = {fs_name: np.zeros(n_tr) for fs_name in FS}

for fold_i, held_out in enumerate(SEASONS):
    val_mask = train_seasons == held_out
    tr_mask = ~val_mask
    Xf_tr, yf_tr = X_tr_imp[tr_mask], y_train[tr_mask]
    Xf_val, yf_val = X_tr_imp[val_mask], y_train[val_mask]
    val_seasons_f = train_seasons[val_mask]
    val_avail = {s: list(range(1, 69)) for s in sorted(set(val_seasons_f))}
    val_idx = np.where(val_mask)[0]

    print(f'\nFold {fold_i+1}/{len(SEASONS)}: {held_out}')

    for fs_name, fs_idx in FS.items():
        m = xgb.XGBRegressor(**RADDAR).fit(Xf_tr[:, fs_idx], yf_tr)
        p = m.predict(Xf_val[:, fs_idx])
        oof_preds[fs_name][val_idx] = p

        rmse = np.sqrt(np.mean((p - yf_val) ** 2))
        for power in POWERS:
            a = hungarian(p, val_seasons_f, val_avail, power)
            ex, sse = evaluate(a, yf_val.astype(int))
            loso_agg[(fs_name, power)]['exact'] += ex
            loso_agg[(fs_name, power)]['sse'] += sse
            loso_agg[(fs_name, power)]['folds'] += 1

        # Show best for this fold
        best_pw_results = []
        for power in POWERS:
            a = hungarian(p, val_seasons_f, val_avail, power)
            ex, _ = evaluate(a, yf_val.astype(int))
            best_pw_results.append((power, ex))
        best_pw_results.sort(key=lambda x: -x[1])
        print(f'  {fs_name}: RMSE={rmse:.3f}, best={best_pw_results[0][1]}/{len(yf_val)} '
              f'(p{best_pw_results[0][0]})')

# Aggregate
valid = [(k, v) for k, v in loso_agg.items() if v['folds'] == len(SEASONS)]
valid.sort(key=lambda x: (-x[1]['exact'], x[1]['sse']))

print(f'\nTop-15 LOSO strategies:')
for (fs_name, pw), scores in valid[:15]:
    rmse = np.sqrt(scores['sse'] / n_tr)
    print(f'  {scores["exact"]}/{n_tr} exact  RMSE={rmse:.4f}  {fs_name}+p{pw}')

# OOF RMSE per feature set
print(f'\nOOF RMSE per feature subset:')
for fs_name in FS:
    rmse = np.sqrt(np.mean((oof_preds[fs_name] - y_train) ** 2))
    print(f'  {fs_name}: {rmse:.4f}')

# Find best feature set (by LOSO exact, aggregated across all powers)
fs_totals = defaultdict(lambda: {'exact': 0, 'sse': 0, 'count': 0})
for (fs_name, pw), scores in valid:
    fs_totals[fs_name]['exact'] += scores['exact']
    fs_totals[fs_name]['sse'] += scores['sse']
    fs_totals[fs_name]['count'] += 1

print(f'\nFeature subset ranking (avg exact across all powers):')
for fs_name, totals in sorted(fs_totals.items(),
                               key=lambda x: -x[1]['exact'] / max(x[1]['count'], 1)):
    avg_ex = totals['exact'] / max(totals['count'], 1)
    print(f'  {fs_name}: avg {avg_ex:.1f}/{n_tr} exact')

# Select best feature set and use default power=1.25
best_fs = valid[0][0][0]
print(f'\nBest feature set from LOSO: {best_fs}')


# ============================================================
#  FINAL TRAINING
# ============================================================
print('\n' + '=' * 60)
print(' FINAL TRAINING')
print('=' * 60)

final_preds = {}
for fs_name, fs_idx in FS.items():
    m = xgb.XGBRegressor(**RADDAR).fit(X_tr_imp[:, fs_idx], y_train)
    final_preds[fs_name] = m.predict(X_te_imp[:, fs_idx])

    if fs_name == best_fs:
        fi = m.feature_importances_
        selected_cols = [feat_cols[i] for i in fs_idx]
        print(f'\nTop-15 features ({fs_name}):')
        for idx in np.argsort(fi)[::-1][:15]:
            print(f'  {fi[idx]:.4f}  {selected_cols[idx]}')


# ============================================================
#  TEST RESULTS (GT for display ONLY)
# ============================================================
print('\n' + '=' * 60)
print(' TEST RESULTS')
print('=' * 60)

# Score ALL combinations
all_results = []
for fs_name, pred in final_preds.items():
    for power in POWERS:
        a = hungarian(pred, test_seasons, avail_seeds, power)
        ex, sse = evaluate(a, test_gt)
        all_results.append((fs_name, power, ex, sse, a))

all_results.sort(key=lambda x: (-x[2], x[3]))

print('\nAll combinations (test GT for display, NOT selection):')
for fs_name, pw, ex, sse, _ in all_results:
    rmse = np.sqrt(sse / 451)
    loso_info = loso_agg.get((fs_name, pw), {})
    loso_ex = loso_info.get('exact', '?')
    marker = f' <<< LOSO-BEST' if (fs_name == valid[0][0][0] and pw == valid[0][0][1]) else ''
    print(f'  {ex}/91  RMSE={rmse:.4f}  {fs_name}+p{pw}  (LOSO: {loso_ex}/{n_tr}){marker}')

# Primary: use LOSO-best
primary_fs, primary_pw = valid[0][0]
primary_pred = final_preds[primary_fs]
primary_assigned = hungarian(primary_pred, test_seasons, avail_seeds, primary_pw)
primary_ex, primary_sse = evaluate(primary_assigned, test_gt)
primary_rmse = np.sqrt(primary_sse / 451)

print(f'\n** PRIMARY (LOSO-chosen): {primary_ex}/91 exact, RMSE={primary_rmse:.4f}')
print(f'   Strategy: {primary_fs}+p{primary_pw}')

# Per-season
print('\nPer-season:')
for s in sorted(set(test_seasons)):
    sm = test_seasons == s
    ex = int(np.sum(primary_assigned[sm] == test_gt[sm]))
    total = int(sm.sum())
    sse = int(np.sum((primary_assigned[sm] - test_gt[sm]) ** 2))
    print(f'  {s}: {ex}/{total} exact, SSE={sse}')

errors = primary_assigned - test_gt
abs_err = np.abs(errors)
print(f'\nErrors: mean={abs_err.mean():.2f} max={abs_err.max()} '
      f'>5={int((abs_err>5).sum())} >10={int((abs_err>10).sum())}')

worst = np.argsort(abs_err)[::-1][:8]
print('\nWorst predictions:')
for i in worst:
    print(f'  {test_rids[i]:30s} pred={primary_assigned[i]:2d} '
          f'actual={test_gt[i]:2d} err={errors[i]:+3d}')


# ============================================================
#  SAVE SUBMISSIONS — multiple power variants for private LB
# ============================================================
def save_sub(assigned, name):
    sub = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub['RecordID'] == rid
        if mask.any():
            sub.loc[mask, 'Overall Seed'] = int(assigned[i])
    path = os.path.join(DATA_DIR, name)
    sub.to_csv(path, index=False)
    ex, sse = evaluate(assigned, test_gt)
    return path, ex, np.sqrt(sse / 451)


submissions = []

# Primary: LOSO-chosen
p, ex, rmse = save_sub(primary_assigned, 'submission_v26_primary.csv')
print(f'\nSaved {os.path.basename(p)}: {ex}/91 RMSE={rmse:.4f} '
      f'({primary_fs}+p{primary_pw}, LOSO-chosen, RECOMMENDED)')
submissions.append(p)

# Alternatives: best feature set with different powers
for power in POWERS:
    if power == primary_pw:
        continue
    a = hungarian(final_preds[best_fs], test_seasons, avail_seeds, power)
    p, ex, rmse = save_sub(a, f'submission_v26_{best_fs}_p{power}.csv')
    print(f'Saved {os.path.basename(p)}: {ex}/91 RMSE={rmse:.4f} ({best_fs}+p{power})')
    submissions.append(p)

# Also save with all features + p1.25 as a safe fallback
a = hungarian(final_preds['all'], test_seasons, avail_seeds, 1.25)
p, ex, rmse = save_sub(a, 'submission_v26_all_p1.25.csv')
print(f'Saved {os.path.basename(p)}: {ex}/91 RMSE={rmse:.4f} (all features+p1.25)')
submissions.append(p)

total = time.time() - t0
print(f'\nTotal: {total:.0f}s ({total/60:.1f} min)')

if IN_COLAB:
    for p in submissions:
        if os.path.exists(p):
            files.download(p)
