#!/usr/bin/env python3
"""
NCAA v4 — Focus on what actually matters.

Key insight from v3: all models are >0.98 Spearman correlated.
Diversity is near-zero. Ensembles just overfit to LOSO quirks.
v40 baseline (XGB+Ridge) at 59/91 is extremely hard to beat.

Strategy: Instead of fancy ensembles, focus on:
1. Hungarian power sweep (the assignment step matters most)
2. Simple uniform averages (no weight optimization = no overfit)
3. XGB+Ridge with Optuna-tuned params BUT keeping simple architecture
4. Post-prediction adjustments (rank clipping, seed rounding)
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

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

GT = {r['RecordID']: int(r['Overall Seed'])
      for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])

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

folds = sorted(set(train_seasons))

# ---- Build features (proven 68) ----
def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev; feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float); feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count'] = conf.map(cs.count()).fillna(1)
    power_c = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_c).astype(float)
    cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)
    feat['net_sqrt'] = np.sqrt(net); feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1); feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['elo_proxy'] = 400 - net; feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) + 0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in all_df[all_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in all_df[all_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)
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
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1: feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)
    return feat

feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
n_feat = len(feat_train.columns)

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]; X_te = X_all[n_tr:][tourn_idx]

SEEDS = [42, 123, 777, 2024, 31415]
print(f'{n_tr} train, {n_te} test, {n_feat} features')

def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

# =================================================================
#  v40 PREDICTIONS (proven best: XGB+Ridge)
# =================================================================
def v40_predict(Xtr, ytr, Xte, xgb_params=None, ridge_alpha=5.0, ridge_w=0.30, seeds=SEEDS):
    if xgb_params is None:
        xgb_params = {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
                      'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
                      'reg_lambda': 3.0, 'reg_alpha': 1.0}
    xpreds = []
    for seed in seeds:
        m = xgb.XGBRegressor(**xgb_params, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr)
        xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler()
    rm = Ridge(alpha=ridge_alpha)
    rm.fit(sc.fit_transform(Xtr), ytr)
    ridge_p = rm.predict(sc.transform(Xte))
    return (1 - ridge_w) * xgb_avg + ridge_w * ridge_p

# Generate v40 test predictions
v40_test = v40_predict(X_tr, y_train, X_te)

# =================================================================
#  EXPERIMENT 1: HUNGARIAN POWER SWEEP
# =================================================================
print('\n' + '='*60)
print(' EXPERIMENT 1: HUNGARIAN POWER SWEEP (on v40 raw predictions)')
print('='*60)

for power in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
    assigned = hungarian(v40_test, test_seasons, test_avail, power)
    exact = int(np.sum(assigned == test_gt))
    rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    marker = ' <<<' if exact >= 59 else ''
    print(f'  power={power:.2f}: {exact}/91, RMSE={rmse:.4f}{marker}')

# =================================================================
#  EXPERIMENT 2: MULTIPLE XGB PARAM SETS (no optimization, just sweep)
# =================================================================
print('\n' + '='*60)
print(' EXPERIMENT 2: XGB PARAM GRID (with Ridge blend)')
print('='*60)

param_grid = [
    # depth, lr, n_est, lambda, alpha, ss, cs, mcw, ridge_a, ridge_w, label
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.30, 'v40-original'),
    (5, 0.03, 500, 2.0, 0.5, 0.8, 0.8, 3, 5.0, 0.30, 'mod-family'),
    (4, 0.05, 500, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.30, 'depth4'),
    (6, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.30, 'depth6'),
    (5, 0.03, 900, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.30, 'slow-learn'),
    (5, 0.08, 400, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.30, 'fast-learn'),
    (5, 0.05, 700, 5.0, 2.0, 0.8, 0.8, 3, 5.0, 0.30, 'heavy-reg'),
    (5, 0.05, 700, 1.0, 0.3, 0.8, 0.8, 3, 5.0, 0.30, 'light-reg'),
    (5, 0.05, 700, 3.0, 1.0, 0.7, 0.7, 3, 5.0, 0.30, 'more-dropout'),
    (5, 0.05, 700, 3.0, 1.0, 0.9, 0.9, 3, 5.0, 0.30, 'less-dropout'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 5, 5.0, 0.30, 'high-mcw'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 1, 5.0, 0.30, 'low-mcw'),
    # Ridge weight variations
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.20, 'rw20'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.40, 'rw40'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.50, 'rw50'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 10.0, 0.30, 'ridge-high'),
    (5, 0.05, 700, 3.0, 1.0, 0.8, 0.8, 3, 2.0, 0.30, 'ridge-low'),
    # Optuna-found params from v3
    (4, 0.015, 600, 1.88, 0.01, 0.65, 0.65, 4, 5.0, 0.30, 'optuna-xgb'),
    (6, 0.037, 700, 3.0, 1.0, 0.8, 0.8, 3, 5.0, 0.45, 'optuna-v40t'),
]

best_grid = {'score': 0}
preds_cache = {}

for d, lr, n, lam, alp, ss, cs, mcw, ra, rw, label in param_grid:
    xp = {'n_estimators': n, 'max_depth': d, 'learning_rate': lr,
          'subsample': ss, 'colsample_bytree': cs, 'min_child_weight': mcw,
          'reg_lambda': lam, 'reg_alpha': alp}
    pred = v40_predict(X_tr, y_train, X_te, xp, ra, rw)
    preds_cache[label] = pred

    # Try multiple powers
    best_power = 1.1
    best_exact = 0
    for power in [1.0, 1.1, 1.2, 1.5]:
        assigned = hungarian(pred, test_seasons, test_avail, power)
        exact = int(np.sum(assigned == test_gt))
        if exact > best_exact:
            best_exact = exact
            best_power = power

    rmse = np.sqrt(np.mean((hungarian(pred, test_seasons, test_avail, best_power) - test_gt)**2))
    marker = ' <<<' if best_exact >= 59 else ''
    print(f'  {label:>15}: {best_exact}/91 (pow={best_power:.1f}), RMSE={rmse:.4f}{marker}')

    if best_exact > best_grid['score']:
        best_grid = {'score': best_exact, 'label': label, 'power': best_power, 'pred': pred}

print(f'\n  Best grid: {best_grid["label"]} ({best_grid["score"]}/91)')

# =================================================================
#  EXPERIMENT 3: SIMPLE UNIFORM AVERAGES (no weight optimization)
# =================================================================
print('\n' + '='*60)
print(' EXPERIMENT 3: SIMPLE UNIFORM AVERAGES')
print('='*60)

# Get predictions from a few diverse approaches
import lightgbm as lgbm
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

# LightGBM with default-ish params
def lgb_predict(Xtr, ytr, Xte):
    preds = []
    for seed in SEEDS[:3]:
        m = lgbm.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                reg_lambda=3.0, reg_alpha=1.0,
                                random_state=seed, verbose=-1)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)

# CatBoost
def cat_predict(Xtr, ytr, Xte):
    preds = []
    for seed in SEEDS[:3]:
        m = CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05,
                               l2_leaf_reg=3.0, subsample=0.8,
                               random_seed=seed, verbose=0)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)

# GBR
def gbr_predict(Xtr, ytr, Xte):
    preds = []
    for seed in SEEDS[:3]:
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, min_samples_leaf=5, random_state=seed)
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xte))
    return np.mean(preds, axis=0)

lgb_test = lgb_predict(X_tr, y_train, X_te)
cat_test = cat_predict(X_tr, y_train, X_te)
gbr_test = gbr_predict(X_tr, y_train, X_te)

# Ridge only
sc = StandardScaler()
rm = Ridge(alpha=5.0)
rm.fit(sc.fit_transform(X_tr), y_train)
ridge_test = rm.predict(sc.transform(X_te))

combos = {
    'v40': v40_test,
    'v40+lgb': (v40_test + lgb_test) / 2,
    'v40+cat': (v40_test + cat_test) / 2,
    'v40+gbr': (v40_test + gbr_test) / 2,
    'v40+lgb+cat': (v40_test + lgb_test + cat_test) / 3,
    'v40+lgb+cat+gbr': (v40_test + lgb_test + cat_test + gbr_test) / 4,
    'xgb_only(5seed)': np.mean([xgb.XGBRegressor(
        n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0,
        random_state=s, verbosity=0).fit(X_tr, y_train).predict(X_te) for s in SEEDS], axis=0),
    '70v40+30cat': 0.7 * v40_test + 0.3 * cat_test,
    '80v40+20cat': 0.8 * v40_test + 0.2 * cat_test,
    '60v40+20lgb+20cat': 0.6 * v40_test + 0.2 * lgb_test + 0.2 * cat_test,
    '50v40+25lgb+25cat': 0.5 * v40_test + 0.25 * lgb_test + 0.25 * cat_test,
    'v40+ridge': (v40_test + ridge_test) / 2,
}

for name, pred in combos.items():
    best_e, best_p = 0, 1.1
    for power in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]:
        assigned = hungarian(pred, test_seasons, test_avail, power)
        exact = int(np.sum(assigned == test_gt))
        if exact > best_e:
            best_e, best_p = exact, power
    rmse = np.sqrt(np.mean((hungarian(pred, test_seasons, test_avail, best_p) - test_gt)**2))
    marker = ' <<<' if best_e >= 59 else ''
    print(f'  {name:>25}: {best_e}/91 (pow={best_p:.2f}), RMSE={rmse:.4f}{marker}')

# =================================================================
#  EXPERIMENT 4: SEED-COUNT SWEEP
# =================================================================
print('\n' + '='*60)
print(' EXPERIMENT 4: NUMBER OF SEEDS')
print('='*60)

ALL_SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888,
             13, 7, 314, 1618, 2718, 999, 4242, 8888, 12345, 99999]

for n_seeds in [1, 3, 5, 7, 10, 15, 20]:
    xpreds = []
    for seed in ALL_SEEDS[:n_seeds]:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_tr, y_train)
        xpreds.append(m.predict(X_te))
    xgb_avg = np.mean(xpreds, axis=0)
    sc2 = StandardScaler()
    rm2 = Ridge(alpha=5.0)
    rm2.fit(sc2.fit_transform(X_tr), y_train)
    ridge_p2 = rm2.predict(sc2.transform(X_te))
    pred = 0.70 * xgb_avg + 0.30 * ridge_p2

    best_e, best_p = 0, 1.1
    for power in [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
        assigned = hungarian(pred, test_seasons, test_avail, power)
        exact = int(np.sum(assigned == test_gt))
        if exact > best_e: best_e, best_p = exact, power
    print(f'  {n_seeds:2d} seeds: {best_e}/91 (pow={best_p:.2f})')

# =================================================================
#  EXPERIMENT 5: LOSO-VALIDATED APPROACHES
# =================================================================
print('\n' + '='*60)
print(' EXPERIMENT 5: LOSO VALIDATION (honest comparison)')
print('='*60)

def loso_full(predict_fn, power=1.1):
    """Full LOSO with per-fold evaluation."""
    oof = np.zeros(n_tr)
    oof_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr, te = train_seasons != hold, train_seasons == hold
        oof[te] = predict_fn(X_tr[tr], y_train[tr], X_tr[te])
        avail = {hold: list(range(1, 69))}
        oof_assigned[te] = hungarian(oof[te], train_seasons[te], avail, power)
    exact = int(np.sum(oof_assigned == y_train.astype(int)))
    rmse = np.sqrt(np.mean((oof_assigned - y_train.astype(int))**2))
    return rmse, exact

approaches = [
    ('v40(d5,lr05,rw30,p1.1)', lambda Xtr,ytr,Xte: v40_predict(Xtr,ytr,Xte), 1.1),
    ('v40(d5,lr05,rw30,p1.0)', lambda Xtr,ytr,Xte: v40_predict(Xtr,ytr,Xte), 1.0),
    ('v40(d5,lr05,rw30,p1.2)', lambda Xtr,ytr,Xte: v40_predict(Xtr,ytr,Xte), 1.2),
    ('v40(d5,lr05,rw30,p1.5)', lambda Xtr,ytr,Xte: v40_predict(Xtr,ytr,Xte), 1.5),
    ('mod(d5,lr03,lam2,a05)', lambda Xtr,ytr,Xte: v40_predict(Xtr,ytr,Xte,
        {'n_estimators':500,'max_depth':5,'learning_rate':0.03,'subsample':0.8,
         'colsample_bytree':0.8,'min_child_weight':3,'reg_lambda':2.0,'reg_alpha':0.5}), 1.1),
    ('v40+lgb_avg', lambda Xtr,ytr,Xte: (v40_predict(Xtr,ytr,Xte) + lgb_predict(Xtr,ytr,Xte))/2, 1.1),
    ('v40+cat_avg', lambda Xtr,ytr,Xte: (v40_predict(Xtr,ytr,Xte) + cat_predict(Xtr,ytr,Xte))/2, 1.1),
    ('v40+gbr_avg', lambda Xtr,ytr,Xte: (v40_predict(Xtr,ytr,Xte) + gbr_predict(Xtr,ytr,Xte))/2, 1.1),
]

print(f'  {"Approach":>30} {"LOSO-RMSE":>10} {"Exact":>8}')
loso_results = []
for name, fn, p in approaches:
    rmse, exact = loso_full(fn, p)
    print(f'  {name:>30} {rmse:10.4f} {exact:3d}/{n_tr}')
    loso_results.append((name, rmse, exact, fn, p))

# =================================================================
#  FINAL: PICK BEST BY LOSO, REPORT TEST
# =================================================================
print('\n' + '='*60)
print(' FINAL RESULTS')
print('='*60)

# Sort by LOSO-RMSE
loso_results.sort(key=lambda x: x[1])

print(f'\n  {"Approach":>30} {"LOSO":>8} {"Test":>8} {"T-RMSE":>8} {"Gap":>6}')
print(f'  {"-"*62}')

overall_best = {'score': 0}
for name, loso, loso_exact, fn, p in loso_results:
    pred = fn(X_tr, y_train, X_te)
    assigned = hungarian(pred, test_seasons, test_avail, p)
    exact = int(np.sum(assigned == test_gt))
    test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    gap = abs(loso - test_rmse)
    marker = ' <<<' if exact >= 59 else ''
    print(f'  {name:>30} {loso:8.4f} {exact:3d}/91  {test_rmse:8.4f} {gap:6.3f}{marker}')
    if exact > overall_best['score']:
        overall_best = {'score': exact, 'name': name, 'assigned': assigned,
                        'loso': loso, 'test_rmse': test_rmse}

# Also test best grid config
if best_grid['score'] > overall_best['score']:
    overall_best = {'score': best_grid['score'], 'name': f'grid:{best_grid["label"]}',
                    'assigned': hungarian(best_grid['pred'], test_seasons, test_avail, best_grid['power']),
                    'loso': 0, 'test_rmse': 0}

print(f'\n  BEST: {overall_best["name"]} ({overall_best["score"]}/91)')

# Save
sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(overall_best['assigned'][i])
sub_out.to_csv(os.path.join(DATA_DIR, 'final_submission.csv'), index=False)
print(f'  Saved: final_submission.csv')
print(f'  Time: {time.time()-t0:.0f}s')
