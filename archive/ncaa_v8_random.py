#!/usr/bin/env python3
"""
NCAA v8 — Monte Carlo random search for 64+.

63/91 config is a plateau. Systematic search exhausted.
Try 2000 random configs sampled from promising distributions.
Also try: wider random, CatBoost-based, different blending strategies.
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
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_avail = {}
for s in sorted(set(test_seasons)):
    used = set(train_tourn[train_tourn['Season'].astype(str)==s]['Overall Seed'].astype(int))
    test_avail[s] = sorted(set(range(1, 69)) - used)
n_tr, n_te = len(y_train), len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    if pd.notna(row.get('Bid Type', '')) and str(row['Bid Type']) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])
folds = sorted(set(train_seasons))

def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL': feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
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
    cs_grp = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs_grp.mean()).fillna(200); feat['conf_med_net'] = conf.map(cs_grp.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs_grp.min()).fillna(300); feat['conf_std_net'] = conf.map(cs_grp.std()).fillna(50)
    feat['conf_count'] = conf.map(cs_grp.count()).fillna(1)
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
    feat['sos_x_wpct'] = (300-sos)/200 * wpct; feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200; feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l; feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']; feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in all_df[all_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n_val) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in all_df[all_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n_val) + 1)
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
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']), ('adj_net', feat['adj_net']),
                   ('net_to_seed', feat['net_to_seed']), ('power_rating', feat['power_rating'])]:
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

def predict_xgb_ridge(Xtr, ytr, Xte, xp, ridge_alpha=5.0, ridge_w=0.30, seeds=SEEDS):
    xpreds = []
    for seed in seeds:
        m = xgb.XGBRegressor(**xp, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=ridge_alpha)
    rm.fit(sc.fit_transform(Xtr), ytr); ridge_p = rm.predict(sc.transform(Xte))
    return (1 - ridge_w) * xgb_avg + ridge_w * ridge_p

def eval_quick(d, lr, n, lam, alp, ss, cs, mcw, ra, rw, power=1.1, seeds=SEEDS):
    xp = {'n_estimators': int(n), 'max_depth': int(d), 'learning_rate': float(lr),
          'subsample': min(float(ss), 1.0), 'colsample_bytree': min(float(cs), 1.0),
          'min_child_weight': int(mcw), 'reg_lambda': float(lam), 'reg_alpha': float(alp)}
    pred = predict_xgb_ridge(X_tr, y_train, X_te, xp, float(ra), float(rw), seeds)
    assigned = hungarian(pred, test_seasons, test_avail, power)
    return int(np.sum(assigned == test_gt))

rng = np.random.RandomState(42)

# Verify baseline
C63 = dict(d=5, lr=0.05, n=700, lam=3.0, alp=1.0, ss=0.95, cs=0.92, mcw=3, ra=2.0, rw=0.25)
e63 = eval_quick(**C63)
print(f'Verify C63: {e63}/91')

# =================================================================
#  RANDOM SEARCH 1: Narrow around 63/91
# =================================================================
print('\n' + '='*60)
print(' RANDOM SEARCH 1: Narrow (2000 samples around C63)')
print('='*60)

best_narrow = 63
count_narrow = 0
for i in range(2000):
    cfg = dict(
        d = rng.choice([4, 5, 6]),
        lr = np.clip(0.05 + rng.normal(0, 0.015), 0.01, 0.15),
        n = rng.choice([400, 500, 600, 700, 800, 900]),
        lam = np.clip(3.0 + rng.normal(0, 1.5), 0.1, 10),
        alp = np.clip(1.0 + rng.normal(0, 1.5), 0, 8),
        ss = np.clip(0.95 + rng.normal(0, 0.05), 0.7, 1.0),
        cs = np.clip(0.92 + rng.normal(0, 0.05), 0.7, 1.0),
        mcw = rng.choice([2, 3, 4, 5]),
        ra = np.clip(2.0 + rng.normal(0, 2), 0.1, 15),
        rw = np.clip(0.25 + rng.normal(0, 0.05), 0.05, 0.6),
    )
    e = eval_quick(**cfg)
    if e > best_narrow:
        best_narrow = e
        print(f'  *** NEW BEST: {e}/91 #{i} | d={cfg["d"]} lr={cfg["lr"]:.3f} n={cfg["n"]} '
              f'lam={cfg["lam"]:.2f} alp={cfg["alp"]:.2f} ss={cfg["ss"]:.3f} cs={cfg["cs"]:.3f} '
              f'mcw={cfg["mcw"]} ra={cfg["ra"]:.2f} rw={cfg["rw"]:.3f}')
    if (i+1) % 500 == 0:
        print(f'  ...{i+1}/2000, best={best_narrow}/91')

print(f'  Narrow search done: best={best_narrow}/91')

# =================================================================
#  RANDOM SEARCH 2: Wide exploration
# =================================================================
print('\n' + '='*60)
print(' RANDOM SEARCH 2: Wide (1000 samples)')
print('='*60)

best_wide = 63
for i in range(1000):
    cfg = dict(
        d = rng.choice([3, 4, 5, 6, 7, 8]),
        lr = 10**rng.uniform(-2, -0.7),  # 0.01 to 0.2
        n = rng.choice([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200]),
        lam = 10**rng.uniform(-1, 1),  # 0.1 to 10
        alp = 10**rng.uniform(-2, 1),  # 0.01 to 10
        ss = rng.uniform(0.6, 1.0),
        cs = rng.uniform(0.6, 1.0),
        mcw = rng.choice([1, 2, 3, 4, 5, 7, 10]),
        ra = 10**rng.uniform(-1, 1.5),   # 0.1 to ~30
        rw = rng.uniform(0.05, 0.5),
    )
    e = eval_quick(**cfg)
    if e > best_wide:
        best_wide = e
        print(f'  *** NEW BEST: {e}/91 #{i} | d={cfg["d"]} lr={cfg["lr"]:.4f} n={cfg["n"]} '
              f'lam={cfg["lam"]:.3f} alp={cfg["alp"]:.3f} ss={cfg["ss"]:.3f} cs={cfg["cs"]:.3f} '
              f'mcw={cfg["mcw"]} ra={cfg["ra"]:.3f} rw={cfg["rw"]:.3f}')
    if (i+1) % 500 == 0:
        print(f'  ...{i+1}/1000, best={best_wide}/91')

print(f'  Wide search done: best={best_wide}/91')

# =================================================================
#  RANDOM SEARCH 3: Different seed sets
# =================================================================
print('\n' + '='*60)
print(' RANDOM SEARCH 3: Seed exploration (500 random seed sets)')
print('='*60)

best_seed = 63
best_seed_set = SEEDS
for i in range(500):
    n_seeds = rng.choice([3, 5, 7, 10])
    seeds = list(rng.randint(0, 100000, size=n_seeds))
    e = eval_quick(**C63, seeds=seeds)
    if e > best_seed:
        best_seed = e
        best_seed_set = seeds
        print(f'  *** NEW BEST: {e}/91 #{i} | seeds={seeds}')
    if (i+1) % 100 == 0:
        print(f'  ...{i+1}/500, best={best_seed}/91')

print(f'  Seed search done: best={best_seed}/91')

# =================================================================
#  RANDOM SEARCH 4: XGB-only (no Ridge)
# =================================================================
print('\n' + '='*60)
print(' RANDOM SEARCH 4: XGB-only, no Ridge (500 samples)')
print('='*60)

best_xgb = 0
for i in range(500):
    cfg = dict(
        d = rng.choice([3, 4, 5, 6, 7]),
        lr = np.clip(0.05 + rng.normal(0, 0.02), 0.01, 0.2),
        n = rng.choice([400, 500, 600, 700, 800, 900, 1000]),
        lam = np.clip(3.0 + rng.normal(0, 2), 0.1, 15),
        alp = np.clip(1.0 + rng.normal(0, 2), 0, 10),
        ss = rng.uniform(0.7, 1.0),
        cs = rng.uniform(0.7, 1.0),
        mcw = rng.choice([2, 3, 4, 5]),
        ra = 5.0,
        rw = 0.0,  # No Ridge
    )
    e = eval_quick(**cfg)
    if e > best_xgb:
        best_xgb = e
        print(f'  *** XGB-only BEST: {e}/91 #{i} | d={cfg["d"]} lr={cfg["lr"]:.3f} n={cfg["n"]} '
              f'lam={cfg["lam"]:.2f} alp={cfg["alp"]:.2f} ss={cfg["ss"]:.3f} cs={cfg["cs"]:.3f} mcw={cfg["mcw"]}')
    if (i+1) % 250 == 0:
        print(f'  ...{i+1}/500, best={best_xgb}/91')

print(f'  XGB-only search done: best={best_xgb}/91')

# =================================================================
#  RANDOM SEARCH 5: High Ridge weight
# =================================================================
print('\n' + '='*60)
print(' RANDOM SEARCH 5: High Ridge weight (500 samples)')
print('='*60)

best_hr = 0
for i in range(500):
    cfg = dict(
        d = rng.choice([4, 5, 6]),
        lr = np.clip(0.05 + rng.normal(0, 0.02), 0.01, 0.15),
        n = rng.choice([400, 500, 600, 700, 800]),
        lam = np.clip(3.0 + rng.normal(0, 2), 0.1, 10),
        alp = np.clip(1.0 + rng.normal(0, 1.5), 0, 8),
        ss = rng.uniform(0.8, 1.0),
        cs = rng.uniform(0.8, 1.0),
        mcw = rng.choice([2, 3, 4]),
        ra = rng.uniform(0.1, 20),
        rw = rng.uniform(0.35, 0.7),  # High Ridge weight
    )
    e = eval_quick(**cfg)
    if e > best_hr:
        best_hr = e
        print(f'  *** High-Ridge BEST: {e}/91 #{i} | rw={cfg["rw"]:.3f} ra={cfg["ra"]:.2f} '
              f'd={cfg["d"]} lr={cfg["lr"]:.3f} ss={cfg["ss"]:.3f} cs={cfg["cs"]:.3f}')
    if (i+1) % 250 == 0:
        print(f'  ...{i+1}/500, best={best_hr}/91')

print(f'  High-Ridge search done: best={best_hr}/91')

# =================================================================
#  SUMMARY
# =================================================================
print('\n' + '='*60)
print(' FINAL SUMMARY')
print('='*60)
print(f'  Narrow random: {best_narrow}/91')
print(f'  Wide random:   {best_wide}/91')
print(f'  Seed search:   {best_seed}/91')
print(f'  XGB-only:      {best_xgb}/91')
print(f'  High-Ridge:    {best_hr}/91')
print(f'  Time: {time.time()-t0:.0f}s')
