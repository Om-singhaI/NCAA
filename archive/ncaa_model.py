#!/usr/bin/env python3
"""
NCAA Overall Seed Prediction — Final Model
===========================================

Single clean model. All decisions validated by LOSO CV RMSE.
No test-score snooping for selection.

Architecture (validated via LOSO grid search):
  XGBoost(depth=5, λ=3, α=1, lr=0.05, 700 trees) × 5 seeds
  + 30% Ridge(α=5) blend
  + Hungarian assignment (power=1.1)
  → LOSO-RMSE = 4.089

Tried and rejected (all worse LOSO-RMSE):
  - Multi-algo (XGB+LGB+CatBoost): 4.198 — adds noise, not signal
  - Heavy regularization (depth=4, λ=5): 4.178 — too constrained
  - Slow-learn (1200 trees, lr=0.02): 4.178 — no benefit
  - Residual correction: 4.135+ — overfits to training residuals
  - 10-seed averaging: 4.165 — more seeds add variance
  - XGB-only (no Ridge): 4.326 — massive overfitting
  - Different ridge weights (20%, 40%): 4.148-4.162 — 30% optimal
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
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# =================================================================
#  CONFIGURATION
# =================================================================
SEEDS = [42, 123, 777, 2024, 31415]

XGB_PARAMS = {
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0,
}

RIDGE_ALPHA = 5.0
RIDGE_WEIGHT = 0.30
HUNGARIAN_POWER = 1.1

# =================================================================
#  DATA
# =================================================================
print('='*60)
print(' NCAA OVERALL SEED PREDICTION')
print('='*60)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))


def parse_wl(s):
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

# Test ground truth (from submission.csv)
GT = {r['RecordID']: int(r['Overall Seed'])
      for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])

# Available seeds per test season
test_avail = {}
for s in sorted(set(test_seasons)):
    used = set(train_tourn[train_tourn['Season'].astype(str)==s]['Overall Seed'].astype(int))
    test_avail[s] = sorted(set(range(1, 69)) - used)

n_tr, n_te = len(y_train), len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df],
                     ignore_index=True)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    if pd.notna(row.get('Bid Type', '')) and str(row['Bid Type']) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])

print(f'  {n_tr} train | {n_te} test | {len(all_tourn_rids)} tournament teams')

# =================================================================
#  FEATURES (68)
# =================================================================
def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)

    # Win-loss records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # Quadrant records
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    # Core rankings
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev
    feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp

    # Bid type
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # Conference stats
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count']   = conf.map(cs.count()).fillna(1)
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power).astype(float)
    cav = feat['conf_avg_net']

    # Isotonic NET→Seed
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    # Transforms
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # Composites
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    # Bid interactions
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # Resume
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w
    feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1)
    feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    # Tournament field rank
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

    # AL rank
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

    # Historical conference-bid stats
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

    feat['net_vs_conf'] = net / (cav + 1)

    # Season percentiles
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)

    return feat


feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
n_feat = len(feat_train.columns)
print(f'  {n_feat} features')

# Impute & scale
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                    feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan,
                    feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]
X_te = X_all[n_tr:][tourn_idx]

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

# =================================================================
#  CORE FUNCTIONS
# =================================================================
def hungarian(scores, seasons, avail, power=HUNGARIAN_POWER):
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


def predict(X_train, y, X_test):
    """5-seed XGB (70%) + Ridge (30%) blend."""
    xgb_preds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_train, y)
        xgb_preds.append(m.predict(X_test))
    xgb_avg = np.mean(xgb_preds, axis=0)

    sc = StandardScaler()
    rm = Ridge(alpha=RIDGE_ALPHA)
    rm.fit(sc.fit_transform(X_train), y)
    ridge_pred = rm.predict(sc.transform(X_test))

    return (1 - RIDGE_WEIGHT) * xgb_avg + RIDGE_WEIGHT * ridge_pred

# =================================================================
#  LEAVE-ONE-SEASON-OUT VALIDATION
# =================================================================
print('\n' + '='*60)
print(' LOSO CROSS-VALIDATION')
print('='*60)

folds = sorted(set(train_seasons))
loso_assigned = np.zeros(n_tr, dtype=int)
fold_stats = []

for hold in folds:
    tr = train_seasons != hold
    te = train_seasons == hold
    pred_te = predict(X_tr[tr], y_train[tr], X_tr[te])
    pred_tr = predict(X_tr[tr], y_train[tr], X_tr[tr])

    avail = {hold: list(range(1, 69))}
    assigned = hungarian(pred_te, train_seasons[te], avail)
    loso_assigned[te] = assigned

    yte = y_train[te].astype(int)
    exact = int(np.sum(assigned == yte))
    n = int(te.sum())
    rmse_cv = np.sqrt(np.mean((assigned - yte)**2))
    rmse_tr = np.sqrt(np.mean((pred_tr - y_train[tr])**2))
    fold_stats.append((hold, n, exact, rmse_cv, rmse_tr))

loso_exact = int(np.sum(loso_assigned == y_train.astype(int)))
loso_rmse = np.sqrt(np.mean((loso_assigned - y_train.astype(int))**2))
avg_tr_rmse = np.mean([f[4] for f in fold_stats])

print(f'\n  {"Season":>10} {"N":>3} {"Exact":>5} {"Pct":>6} '
      f'{"CV-RMSE":>8} {"Tr-RMSE":>8} {"Gap":>7}')
for s, n, ex, rmse_cv, rmse_tr in fold_stats:
    gap = rmse_cv - rmse_tr
    flag = ' !' if gap > 3.5 else ''
    print(f'  {s:>10} {n:3d} {ex:5d} {ex/n*100:5.1f}% '
          f'{rmse_cv:8.3f} {rmse_tr:8.3f} {gap:7.3f}{flag}')

print(f'\n  TOTAL: {loso_exact}/{n_tr} exact ({loso_exact/n_tr*100:.1f}%), '
      f'RMSE={loso_rmse:.4f}')
print(f'  Train RMSE (avg): {avg_tr_rmse:.4f}')
print(f'  Overfit gap: {loso_rmse - avg_tr_rmse:.4f}')

fold_rmses = [f[3] for f in fold_stats]
cv_ratio = np.std(fold_rmses) / np.mean(fold_rmses) * 100
print(f'  Fold stability: mean={np.mean(fold_rmses):.3f}, '
      f'std={np.std(fold_rmses):.3f}, CV={cv_ratio:.1f}%')

# =================================================================
#  FINAL PREDICTION
# =================================================================
print('\n' + '='*60)
print(' FINAL PREDICTION')
print('='*60)

pred = predict(X_tr, y_train, X_te)
assigned = hungarian(pred, test_seasons, test_avail)
test_exact = int(np.sum(assigned == test_gt))
test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))

print(f'\n  Test: {test_exact}/91 exact, RMSE={test_rmse:.4f}')
print(f'  LOSO RMSE: {loso_rmse:.4f}')
print(f'  Generalization: LOSO→Test RMSE Δ = {abs(loso_rmse - test_rmse):.4f} '
      f'({"good" if abs(loso_rmse - test_rmse) < 0.3 else "warning"})')

# =================================================================
#  SAVE
# =================================================================
sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(assigned[i])

out_path = os.path.join(DATA_DIR, 'final_submission.csv')
sub_out.to_csv(out_path, index=False)
print(f'\n  Saved: final_submission.csv')
print(f'  Time: {time.time()-t0:.0f}s')

if IN_COLAB:
    files.download(out_path)
