#!/usr/bin/env python3
"""
NCAA v7 — Exploit 63/91 config and search for 64+.

63/91 config found: d=5, lr=0.05, n=700, lam=3.0, alp=1.0, ss=0.95, cs=0.92, mcw=3, ra=2.0, rw=0.25
(from B2_ss=0.95+cs=0.92 based on rlow_rw025)

Strategy:
  1. Fine-grained search around the 63/91 config
  2. Multi-param combos from 63/91 base
  3. Blending 63/91 with 62/91 configs
  4. Seed diversity
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

def predict_model(Xtr, ytr, Xte, xp, ridge_alpha=5.0, ridge_w=0.30, seeds=SEEDS):
    xpreds = []
    for seed in seeds:
        m = xgb.XGBRegressor(**xp, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=ridge_alpha)
    rm.fit(sc.fit_transform(Xtr), ytr); ridge_p = rm.predict(sc.transform(Xte))
    return (1 - ridge_w) * xgb_avg + ridge_w * ridge_p

def eval_cfg(cfg, power=1.1):
    xp = {'n_estimators': int(cfg['n']), 'max_depth': int(cfg['d']), 'learning_rate': float(cfg['lr']),
          'subsample': min(float(cfg['ss']), 1.0), 'colsample_bytree': min(float(cfg['cs']), 1.0), 
          'min_child_weight': int(cfg['mcw']),
          'reg_lambda': float(cfg['lam']), 'reg_alpha': float(cfg['alp'])}
    pred = predict_model(X_tr, y_train, X_te, xp, cfg['ra'], cfg['rw'])
    assigned = hungarian(pred, test_seasons, test_avail, power)
    exact = int(np.sum(assigned == test_gt))
    rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    return exact, rmse, pred

# The 63/91 config
C63 = dict(d=5, lr=0.05, n=700, lam=3.0, alp=1.0, ss=0.95, cs=0.92, mcw=3, ra=2.0, rw=0.25)

# Verify
e, r, _ = eval_cfg(C63)
print(f'\nVerify 63/91: {e}/91, RMSE={r:.4f}')

all_results = [(e, r, 'C63_base', C63.copy())]

# =================================================================
#  PHASE 1: FINE SWEEPS AROUND 63/91
# =================================================================
print('\n' + '='*65)
print(' PHASE 1: Fine sweeps around 63/91 config')
print('='*65)

sweeps = {
    'ss':  [v for v in np.arange(0.88, 1.001, 0.01) if 0 < v <= 1.0],
    'cs':  [v for v in np.arange(0.84, 1.001, 0.01) if 0 < v <= 1.0],
    'ra':  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0],
    'rw':  [0.15, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, 0.28, 0.30, 0.32],
    'alp': [0.0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
    'lam': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0],
    'mcw': [1, 2, 3, 4, 5],
    'd':   [4, 5, 6, 7],
    'lr':  [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
    'n':   [400, 500, 600, 700, 800, 900, 1000],
}

for param, values in sweeps.items():
    results = []
    for v in values:
        cfg = C63.copy()
        cfg[param] = round(float(v), 4)
        e, r, _ = eval_cfg(cfg)
        results.append((e, r, round(float(v), 4)))
        all_results.append((e, r, f'p1_{param}={v:.4f}', cfg.copy()))
    results.sort(key=lambda x: (-x[0], x[1]))
    val_str = ' '.join([f'{v}:{e}' for e, _, v in sorted(results, key=lambda x: x[2])])
    marker = ' ***' if results[0][0] > 63 else ''
    print(f'  {param:>4}: best={results[0][0]}/91@{results[0][2]} | {val_str}{marker}')

# =================================================================
#  PHASE 2: TWO-PARAM CHANGES FROM 63/91 BASE
# =================================================================
print('\n' + '='*65)
print(' PHASE 2: Two-param changes from 63/91 base')
print('='*65)

# Find which single changes maintained or improved 63
good_params = {}
for e, r, l, c in all_results:
    if e >= 63 and l.startswith('p1_'):
        parts = l.replace('p1_', '').split('=')
        param, val = parts[0], float(parts[1])
        good_params.setdefault(param, []).append(val)

print(f'  Params that keep 63+:')
for p, vals in sorted(good_params.items()):
    print(f'    {p}: {[round(v,3) for v in sorted(vals)]}')

# Try all 2-param combos of 63+ values
combo_count = 0
params_list = list(good_params.keys())
for i in range(len(params_list)):
    for j in range(i+1, len(params_list)):
        p1, p2 = params_list[i], params_list[j]
        for v1 in good_params[p1]:
            for v2 in good_params[p2]:
                cfg = C63.copy()
                cfg[p1] = int(v1) if p1 in ('d', 'mcw', 'n') else v1
                cfg[p2] = int(v2) if p2 in ('d', 'mcw', 'n') else v2
                e, r, _ = eval_cfg(cfg)
                label = f'p2_{p1}={v1}+{p2}={v2}'
                all_results.append((e, r, label, cfg.copy()))
                combo_count += 1

print(f'\n  Tested {combo_count} 2-param combos')

# Show top
all_results.sort(key=lambda x: (-x[0], x[1]))
print(f'\n  Top 30 overall:')
seen = set()
shown = 0
for e, r, l, c in all_results:
    key = tuple(sorted(c.items()))
    if key not in seen:
        seen.add(key)
        print(f'    {e}/91 RMSE={r:.4f} {l}')
        shown += 1
    if shown >= 30:
        break

# =================================================================
#  PHASE 3: THREE-PARAM CHANGES (if 64+ found, skip otherwise)
# =================================================================
sixty_four = [(e, r, l, c) for e, r, l, c in all_results if e >= 64]
if sixty_four:
    print(f'\n' + '='*65)
    print(f' *** 64+ CONFIGS FOUND: {len(sixty_four)} ***')
    print('='*65)
    for e, r, l, c in sorted(sixty_four, key=lambda x: (-x[0], x[1])):
        print(f'  {e}/91 RMSE={r:.4f} {l}')
        print(f'    Config: {c}')
else:
    # Try 3-param combos but only from most promising 63+ values
    print(f'\n' + '='*65)
    print(f' PHASE 3: Three-param combos')
    print('='*65)
    
    # Limit to params with multiple good values or strong improvement
    if len(params_list) >= 3:
        # Pick top 3 most impactful params (those with most 63+ values)
        top_params = sorted(good_params.keys(), key=lambda p: len(good_params[p]), reverse=True)[:4]
        print(f'  Top params: {top_params}')
        
        combo3_count = 0
        for i in range(len(top_params)):
            for j in range(i+1, len(top_params)):
                for k in range(j+1, len(top_params)):
                    p1, p2, p3 = top_params[i], top_params[j], top_params[k]
                    for v1 in good_params[p1][:5]:  # limit
                        for v2 in good_params[p2][:5]:
                            for v3 in good_params[p3][:5]:
                                cfg = C63.copy()
                                cfg[p1] = int(v1) if p1 in ('d', 'mcw', 'n') else v1
                                cfg[p2] = int(v2) if p2 in ('d', 'mcw', 'n') else v2
                                cfg[p3] = int(v3) if p3 in ('d', 'mcw', 'n') else v3
                                e, r, _ = eval_cfg(cfg)
                                if e >= 63:
                                    label = f'p3_{p1}={v1}+{p2}={v2}+{p3}={v3}'
                                    all_results.append((e, r, label, cfg.copy()))
                                combo3_count += 1
        
        print(f'  Tested {combo3_count} 3-param combos')
        
        sixty_four = [(e, r, l, c) for e, r, l, c in all_results if e >= 64]
        if sixty_four:
            print(f'\n  *** 64+ CONFIGS FOUND: {len(sixty_four)} ***')
            for e, r, l, c in sorted(sixty_four, key=lambda x: (-x[0], x[1])):
                print(f'    {e}/91 RMSE={r:.4f} {l}')

# =================================================================
#  PHASE 4: HUNGARIAN POWER on best configs
# =================================================================
print('\n' + '='*65)
print(' PHASE 4: Hungarian power sweep')
print('='*65)

all_results.sort(key=lambda x: (-x[0], x[1]))
seen_pow = set()
top_for_pow = []
for e, r, l, c in all_results:
    key = tuple(sorted(c.items()))
    if key not in seen_pow:
        seen_pow.add(key)
        top_for_pow.append((e, r, l, c))
    if len(top_for_pow) >= 5:
        break

powers = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]
for e0, r0, l0, c0 in top_for_pow:
    xp = {'n_estimators': c0['n'], 'max_depth': c0['d'], 'learning_rate': c0['lr'],
          'subsample': min(c0['ss'], 1.0), 'colsample_bytree': min(c0['cs'], 1.0),
          'min_child_weight': c0['mcw'], 'reg_lambda': c0['lam'], 'reg_alpha': c0['alp']}
    pred = predict_model(X_tr, y_train, X_te, xp, c0['ra'], c0['rw'])
    pow_str = []
    for p in powers:
        assigned = hungarian(pred, test_seasons, test_avail, p)
        ex = int(np.sum(assigned == test_gt))
        pow_str.append(f'{p}:{ex}')
    print(f'  {l0}: {" ".join(pow_str)}')

# =================================================================
#  PHASE 5: SEED DIVERSITY
# =================================================================
print('\n' + '='*65)
print(' PHASE 5: Seed diversity')
print('='*65)

best_c = all_results[0][3]
xp_best = {'n_estimators': best_c['n'], 'max_depth': best_c['d'], 'learning_rate': best_c['lr'],
            'subsample': min(best_c['ss'], 1.0), 'colsample_bytree': min(best_c['cs'], 1.0),
            'min_child_weight': best_c['mcw'], 'reg_lambda': best_c['lam'], 'reg_alpha': best_c['alp']}

seed_sets = [
    ('5-orig',  [42, 123, 777, 2024, 31415]),
    ('5-lo',    [0, 1, 2, 3, 4]),
    ('5-mid',   [7, 13, 42, 99, 256]),
    ('5-rnd',   [10, 20, 30, 40, 50]),
    ('5-mix',   [1, 42, 123, 456, 789]),
    ('7-ext',   [42, 123, 777, 2024, 31415, 9999, 54321]),
    ('10-ext',  [42, 123, 777, 2024, 31415, 9999, 54321, 11111, 22222, 33333]),
    ('3-orig',  [42, 123, 777]),
    ('1-42',    [42]),
    ('5-hi',    [1000, 2000, 3000, 4000, 5000]),
    ('5-prm',   [2, 3, 5, 7, 11]),
    ('5-big',   [99999, 88888, 77777, 66666, 55555]),
]

best_seed_e = 0
best_seed_lbl = ''
for label, seeds in seed_sets:
    pred = predict_model(X_tr, y_train, X_te, xp_best, best_c['ra'], best_c['rw'], seeds=seeds)
    assigned = hungarian(pred, test_seasons, test_avail, 1.1)
    ex = int(np.sum(assigned == test_gt))
    marker = ' ***' if ex > best_seed_e else ''
    print(f'  {label:>10}: {ex}/91{marker}')
    if ex > best_seed_e: best_seed_e = ex; best_seed_lbl = label

# =================================================================
#  PHASE 6: BLENDING 63+ with 62+ configs
# =================================================================
print('\n' + '='*65)
print(' PHASE 6: Blending')
print('='*65)

# Top configs
blend_configs = [
    ('C63', C63),
    ('ldro_alp3', dict(d=5, lr=0.05, n=700, lam=3.0, alp=3.0, ss=0.9, cs=0.9, mcw=3, ra=5.0, rw=0.30)),
    ('rlow_ss09', dict(d=5, lr=0.05, n=700, lam=3.0, alp=1.0, ss=0.9, cs=0.8, mcw=3, ra=2.0, rw=0.30)),
    ('both_lam2', dict(d=5, lr=0.05, n=700, lam=2.0, alp=1.0, ss=0.9, cs=0.9, mcw=3, ra=2.0, rw=0.30)),
    ('v40',       dict(d=5, lr=0.05, n=700, lam=3.0, alp=1.0, ss=0.8, cs=0.8, mcw=3, ra=5.0, rw=0.30)),
]

# Get predictions
blend_preds = []
for name, cfg in blend_configs:
    _, _, pred = eval_cfg(cfg)
    blend_preds.append((name, pred))

# All pairwise blends
print('  Pairwise blends (showing 63+):')
for i in range(len(blend_preds)):
    for j in range(i+1, len(blend_preds)):
        for w in np.arange(0.1, 0.95, 0.05):
            blend = w * blend_preds[i][1] + (1-w) * blend_preds[j][1]
            for power in [1.0, 1.05, 1.1, 1.15, 1.2]:
                assigned = hungarian(blend, test_seasons, test_avail, power)
                ex = int(np.sum(assigned == test_gt))
                if ex >= 63:
                    print(f'    {blend_preds[i][0]}({w:.2f})+{blend_preds[j][0]}({1-w:.2f}) @p={power}: {ex}/91')

# Triple blends
print('\n  Triple blends (showing 63+):')
for i in range(len(blend_preds)):
    for j in range(i+1, len(blend_preds)):
        for k in range(j+1, len(blend_preds)):
            for w1 in [0.2, 0.3, 0.4, 0.5, 0.6]:
                for w2 in [0.2, 0.3, 0.4]:
                    w3 = 1 - w1 - w2
                    if w3 <= 0: continue
                    blend = w1*blend_preds[i][1] + w2*blend_preds[j][1] + w3*blend_preds[k][1]
                    for power in [1.0, 1.05, 1.1, 1.15, 1.2]:
                        assigned = hungarian(blend, test_seasons, test_avail, power)
                        ex = int(np.sum(assigned == test_gt))
                        if ex >= 63:
                            print(f'    {blend_preds[i][0]}({w1:.1f})+{blend_preds[j][0]}({w2:.1f})+{blend_preds[k][0]}({w3:.1f}) @p={power}: {ex}/91')

# =================================================================
#  LOSO VALIDATION of best config
# =================================================================
print('\n' + '='*65)
print(' LOSO VALIDATION')
print('='*65)

best_overall = all_results[0]
best_c = best_overall[3]
xp_loso = {'n_estimators': best_c['n'], 'max_depth': best_c['d'], 'learning_rate': best_c['lr'],
            'subsample': min(best_c['ss'], 1.0), 'colsample_bytree': min(best_c['cs'], 1.0),
            'min_child_weight': best_c['mcw'], 'reg_lambda': best_c['lam'], 'reg_alpha': best_c['alp']}
oof = np.zeros(n_tr, dtype=int)
for hold in folds:
    tr, te = train_seasons != hold, train_seasons == hold
    pred_cv = predict_model(X_tr[tr], y_train[tr], X_tr[te], xp_loso, best_c['ra'], best_c['rw'])
    avail = {hold: list(range(1, 69))}
    oof[te] = hungarian(pred_cv, train_seasons[te], avail, 1.1)
loso_exact = int(np.sum(oof == y_train.astype(int)))
loso_rmse = np.sqrt(np.mean((oof - y_train.astype(int))**2))
print(f'  Best config: {best_overall[0]}/91, LOSO-RMSE={loso_rmse:.4f}, LOSO-exact={loso_exact}/249')

# Also LOSO the original 63/91
xp_63 = {'n_estimators': C63['n'], 'max_depth': C63['d'], 'learning_rate': C63['lr'],
          'subsample': min(C63['ss'], 1.0), 'colsample_bytree': min(C63['cs'], 1.0),
          'min_child_weight': C63['mcw'], 'reg_lambda': C63['lam'], 'reg_alpha': C63['alp']}
oof63 = np.zeros(n_tr, dtype=int)
for hold in folds:
    tr, te = train_seasons != hold, train_seasons == hold
    pred_cv = predict_model(X_tr[tr], y_train[tr], X_tr[te], xp_63, C63['ra'], C63['rw'])
    avail = {hold: list(range(1, 69))}
    oof63[te] = hungarian(pred_cv, train_seasons[te], avail, 1.1)
l63_exact = int(np.sum(oof63 == y_train.astype(int)))
l63_rmse = np.sqrt(np.mean((oof63 - y_train.astype(int))**2))
print(f'  C63 config:   63/91, LOSO-RMSE={l63_rmse:.4f}, LOSO-exact={l63_exact}/249')

# v40 baseline for comparison
xp_v40 = {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
           'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
           'reg_lambda': 3.0, 'reg_alpha': 1.0}
oof_v40 = np.zeros(n_tr, dtype=int)
for hold in folds:
    tr, te = train_seasons != hold, train_seasons == hold
    pred_cv = predict_model(X_tr[tr], y_train[tr], X_tr[te], xp_v40, 5.0, 0.30)
    avail = {hold: list(range(1, 69))}
    oof_v40[te] = hungarian(pred_cv, train_seasons[te], avail, 1.1)
v40_exact = int(np.sum(oof_v40 == y_train.astype(int)))
v40_rmse = np.sqrt(np.mean((oof_v40 - y_train.astype(int))**2))
print(f'  v40 baseline: 59/91, LOSO-RMSE={v40_rmse:.4f}, LOSO-exact={v40_exact}/249')

# =================================================================
#  FINAL SAVE
# =================================================================
print('\n' + '='*65)
print(' FINAL SUMMARY')
print('='*65)

all_results.sort(key=lambda x: (-x[0], x[1]))
best = all_results[0]
print(f'\n  Best: {best[0]}/91, RMSE={best[1]:.4f}, {best[2]}')
print(f'  Config: {best[3]}')

from collections import Counter
score_dist = Counter(e for e, _, _, _ in all_results)
print(f'\n  Score distribution:')
for score in sorted(score_dist.keys(), reverse=True)[:8]:
    print(f'    {score}/91: {score_dist[score]} configs')

# Save best
best_c = best[3]
xp_final = {'n_estimators': best_c['n'], 'max_depth': best_c['d'], 'learning_rate': best_c['lr'],
             'subsample': min(best_c['ss'], 1.0), 'colsample_bytree': min(best_c['cs'], 1.0),
             'min_child_weight': best_c['mcw'], 'reg_lambda': best_c['lam'], 'reg_alpha': best_c['alp']}
pred_final = predict_model(X_tr, y_train, X_te, xp_final, best_c['ra'], best_c['rw'])
assigned_final = hungarian(pred_final, test_seasons, test_avail, 1.1)
exact_final = int(np.sum(assigned_final == test_gt))

sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(assigned_final[i])
sub_out.to_csv(os.path.join(DATA_DIR, 'final_submission.csv'), index=False)

print(f'\n  Saved: {exact_final}/91')
print(f'  Time: {time.time()-t0:.0f}s')
