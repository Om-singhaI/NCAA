#!/usr/bin/env python3
"""
NCAA v9 — Random Forest experiment.
====================================
Selection criterion: LOSO-RMSE ONLY. No test-score snooping.
Test score reported once at the end for the LOSO-best config.

Plan:
  1. RF with various hyperparameters → pick best LOSO-RMSE
  2. RF + Ridge blend → pick best LOSO-RMSE
  3. RF + XGB blend → pick best LOSO-RMSE
  4. Best overall → final test evaluation
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
#  DATA (same as ncaa_model.py)
# =================================================================
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

# =================================================================
#  FEATURES (68 — same as ncaa_model.py)
# =================================================================
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
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
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

print(f'{n_tr} train, {n_te} test, {n_feat} features')

POWER = 1.1

def hungarian(scores, seasons, avail, power=POWER):
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
#  LOSO evaluation function
# =================================================================
def loso_eval(predict_fn, label=''):
    """Run LOSO and return (loso_rmse, fold_details).
    predict_fn(X_train, y_train, X_test) -> predictions
    """
    loso_assigned = np.zeros(n_tr, dtype=int)
    fold_details = []
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_fn(X_tr[tr], y_train[tr], X_tr[te])
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, train_seasons[te], avail)
        loso_assigned[te] = assigned
        yte = y_train[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse = np.sqrt(np.mean((assigned - yte)**2))
        fold_details.append((hold, int(te.sum()), exact, rmse))
    
    overall_exact = int(np.sum(loso_assigned == y_train.astype(int)))
    overall_rmse = np.sqrt(np.mean((loso_assigned - y_train.astype(int))**2))
    return overall_rmse, overall_exact, fold_details

# =================================================================
#  BASELINE: Current v40 (XGB + Ridge)
# =================================================================
print('\n' + '='*60)
print(' BASELINE: v40 (XGB + Ridge)')
print('='*60)

XGB_P = {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
         'reg_lambda': 3.0, 'reg_alpha': 1.0}
SEEDS = [42, 123, 777, 2024, 31415]

def pred_v40(Xtr, ytr, Xte):
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(**XGB_P, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
    return 0.70 * xgb_avg + 0.30 * rp

rmse_v40, exact_v40, folds_v40 = loso_eval(pred_v40, 'v40')
print(f'  v40 LOSO: {exact_v40}/{n_tr} exact, RMSE={rmse_v40:.4f}')
for s, n, ex, rm in folds_v40:
    print(f'    {s}: {ex}/{n} ({ex/n*100:.1f}%) RMSE={rm:.3f}')

# =================================================================
#  PHASE 1: Pure Random Forest — hyperparameter sweep
# =================================================================
print('\n' + '='*60)
print(' PHASE 1: Random Forest hyperparameter sweep')
print('='*60)

rf_configs = []
for n_est in [200, 500, 1000]:
    for max_depth in [None, 10, 15, 20, 30]:
        for min_samples_leaf in [1, 2, 3, 5]:
            for max_features in ['sqrt', 0.5, 0.8, 1.0]:
                rf_configs.append({
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                })

print(f'  Testing {len(rf_configs)} RF configs...')
best_rf_rmse = 999
best_rf_cfg = None
rf_results = []

for i, cfg in enumerate(rf_configs):
    def pred_rf(Xtr, ytr, Xte, c=cfg):
        m = RandomForestRegressor(**c, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    
    rmse, exact, _ = loso_eval(pred_rf)
    rf_results.append((rmse, exact, cfg))
    if rmse < best_rf_rmse:
        best_rf_rmse = rmse
        best_rf_cfg = cfg
        print(f'  #{i+1:3d} NEW BEST RF LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
              f'n={cfg["n_estimators"]} d={cfg["max_depth"]} '
              f'msl={cfg["min_samples_leaf"]} mf={cfg["max_features"]}')
    if (i+1) % 60 == 0:
        print(f'  ...{i+1}/{len(rf_configs)}, best RMSE={best_rf_rmse:.4f}')

# Sort by LOSO-RMSE  
rf_results.sort(key=lambda x: x[0])
print(f'\n  Best RF: LOSO-RMSE={best_rf_rmse:.4f}')
print(f'  Config: {best_rf_cfg}')
print(f'\n  Top 5 RF configs:')
for rmse, exact, cfg in rf_results[:5]:
    print(f'    RMSE={rmse:.4f} ({exact}/{n_tr}) | n={cfg["n_estimators"]} '
          f'd={cfg["max_depth"]} msl={cfg["min_samples_leaf"]} mf={cfg["max_features"]}')

# =================================================================
#  PHASE 2: Extra Trees (often better than RF)
# =================================================================
print('\n' + '='*60)
print(' PHASE 2: Extra Trees')
print('='*60)

et_configs = []
for n_est in [200, 500, 1000]:
    for max_depth in [None, 10, 20]:
        for min_samples_leaf in [1, 2, 3, 5]:
            for max_features in ['sqrt', 0.5, 0.8, 1.0]:
                et_configs.append({
                    'n_estimators': n_est,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                })

print(f'  Testing {len(et_configs)} ExtraTrees configs...')
best_et_rmse = 999
best_et_cfg = None

for i, cfg in enumerate(et_configs):
    def pred_et(Xtr, ytr, Xte, c=cfg):
        m = ExtraTreesRegressor(**c, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    
    rmse, exact, _ = loso_eval(pred_et)
    if rmse < best_et_rmse:
        best_et_rmse = rmse
        best_et_cfg = cfg
        print(f'  #{i+1:3d} NEW BEST ET LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
              f'n={cfg["n_estimators"]} d={cfg["max_depth"]} '
              f'msl={cfg["min_samples_leaf"]} mf={cfg["max_features"]}')
    if (i+1) % 48 == 0:
        print(f'  ...{i+1}/{len(et_configs)}, best RMSE={best_et_rmse:.4f}')

print(f'\n  Best ET: LOSO-RMSE={best_et_rmse:.4f}')
print(f'  Config: {best_et_cfg}')

# =================================================================
#  PHASE 3: RF + Ridge blend
# =================================================================
print('\n' + '='*60)
print(' PHASE 3: RF + Ridge blend')
print('='*60)

best_rfr_rmse = 999
best_rfr_params = None

# Use top 3 RF configs
top_rf = rf_results[:3]
for rmse_rf, _, cfg_rf in top_rf:
    for ridge_alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for ridge_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
            def pred_rfr(Xtr, ytr, Xte, c=cfg_rf, ra=ridge_alpha, rw=ridge_w):
                m = RandomForestRegressor(**c, random_state=42, n_jobs=-1)
                m.fit(Xtr, ytr); rf_p = m.predict(Xte)
                sc = StandardScaler(); rm = Ridge(alpha=ra)
                rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
                return (1 - rw) * rf_p + rw * rp
            
            rmse, exact, _ = loso_eval(pred_rfr)
            if rmse < best_rfr_rmse:
                best_rfr_rmse = rmse
                best_rfr_params = (cfg_rf, ridge_alpha, ridge_w)
                print(f'  NEW BEST RF+Ridge LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
                      f'ra={ridge_alpha} rw={ridge_w} | RF: n={cfg_rf["n_estimators"]} '
                      f'd={cfg_rf["max_depth"]} msl={cfg_rf["min_samples_leaf"]}')

print(f'\n  Best RF+Ridge: LOSO-RMSE={best_rfr_rmse:.4f}')

# =================================================================
#  PHASE 4: RF + XGB blend
# =================================================================
print('\n' + '='*60)
print(' PHASE 4: RF + XGB blend')
print('='*60)

best_blend_rmse = 999
best_blend_params = None

for rmse_rf, _, cfg_rf in top_rf[:2]:
    for rf_w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        def pred_blend(Xtr, ytr, Xte, c=cfg_rf, w=rf_w):
            # RF part
            m_rf = RandomForestRegressor(**c, random_state=42, n_jobs=-1)
            m_rf.fit(Xtr, ytr); rf_p = m_rf.predict(Xte)
            # XGB part (v40 config)
            xpreds = []
            for seed in SEEDS:
                m = xgb.XGBRegressor(**XGB_P, random_state=seed, verbosity=0)
                m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
            xgb_p = np.mean(xpreds, axis=0)
            # Ridge part
            sc = StandardScaler(); rm = Ridge(alpha=5.0)
            rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
            xgb_ridge = 0.70 * xgb_p + 0.30 * rp
            return w * rf_p + (1 - w) * xgb_ridge
        
        rmse, exact, _ = loso_eval(pred_blend)
        if rmse < best_blend_rmse:
            best_blend_rmse = rmse
            best_blend_params = (cfg_rf, rf_w)
            print(f'  NEW BEST RF+XGB+Ridge LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
                  f'rf_w={rf_w} | RF: n={cfg_rf["n_estimators"]} d={cfg_rf["max_depth"]}')

print(f'\n  Best RF+XGB blend: LOSO-RMSE={best_blend_rmse:.4f}')

# =================================================================
#  PHASE 5: Multi-seed RF (reduce variance)
# =================================================================
print('\n' + '='*60)
print(' PHASE 5: Multi-seed RF')
print('='*60)

best_ms_rmse = 999
best_ms_cfg = None

# Use top RF config with multiple seeds
top_cfg = rf_results[0][2]
seed_sets = [
    [42],
    [42, 123, 777],
    [42, 123, 777, 2024, 31415],
    [42, 123, 777, 2024, 31415, 7, 99],
    [42, 123, 777, 2024, 31415, 7, 99, 314, 555, 1000],
]
for seeds in seed_sets:
    for msl in [1, 2, 3]:
        def pred_ms(Xtr, ytr, Xte, c=top_cfg, ss=seeds, ml=msl):
            preds = []
            cfg_mod = dict(c)
            cfg_mod['min_samples_leaf'] = ml
            for s in ss:
                m = RandomForestRegressor(**cfg_mod, random_state=s, n_jobs=-1)
                m.fit(Xtr, ytr); preds.append(m.predict(Xte))
            return np.mean(preds, axis=0)
        
        rmse, exact, _ = loso_eval(pred_ms)
        if rmse < best_ms_rmse:
            best_ms_rmse = rmse
            best_ms_cfg = (seeds, msl)
            print(f'  Multi-seed LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
                  f'{len(seeds)} seeds, msl={msl}')

print(f'\n  Best multi-seed RF: LOSO-RMSE={best_ms_rmse:.4f}')

# =================================================================
#  PHASE 6: RF + Ridge (best RF config, fine-tune blend)
# =================================================================
print('\n' + '='*60)
print(' PHASE 6: Fine-tune best RF + Ridge')
print('='*60)

best_fine_rmse = 999
best_fine_params = None
top3_rf_cfgs = [r[2] for r in rf_results[:3]]

for cfg_rf in top3_rf_cfgs:
    for n_seeds in [1, 3, 5]:
        seeds_use = SEEDS[:n_seeds]
        for ra in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
            for rw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
                def pred_fine(Xtr, ytr, Xte, c=cfg_rf, ss=seeds_use, r_a=ra, r_w=rw):
                    rf_preds = []
                    for s in ss:
                        m = RandomForestRegressor(**c, random_state=s, n_jobs=-1)
                        m.fit(Xtr, ytr); rf_preds.append(m.predict(Xte))
                    rf_p = np.mean(rf_preds, axis=0)
                    sc = StandardScaler(); rm = Ridge(alpha=r_a)
                    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
                    return (1 - r_w) * rf_p + r_w * rp
                
                rmse, exact, _ = loso_eval(pred_fine)
                if rmse < best_fine_rmse:
                    best_fine_rmse = rmse
                    best_fine_params = (cfg_rf, seeds_use, ra, rw)
                    print(f'  Fine LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) | '
                          f'seeds={len(seeds_use)} ra={ra} rw={rw} '
                          f'n={cfg_rf["n_estimators"]} d={cfg_rf["max_depth"]} msl={cfg_rf["min_samples_leaf"]}')

print(f'\n  Best fine-tuned RF+Ridge: LOSO-RMSE={best_fine_rmse:.4f}')

# =================================================================
#  SUMMARY & FINAL TEST (ONE evaluation on the LOSO-best)
# =================================================================
print('\n' + '='*60)
print(' FINAL SUMMARY (selected by LOSO only)')
print('='*60)

candidates = [
    ('v40 (XGB+Ridge)', rmse_v40, exact_v40, pred_v40),
    ('Best RF pure', best_rf_rmse, None, None),
    ('Best ET pure', best_et_rmse, None, None),
    ('RF+Ridge', best_rfr_rmse, None, None),
    ('RF+XGB+Ridge', best_blend_rmse, None, None),
    ('Multi-seed RF', best_ms_rmse, None, None),
    ('Fine RF+Ridge', best_fine_rmse, None, None),
]

print(f'\n  {"Model":<20} {"LOSO-RMSE":>10}')
print(f'  {"-"*20} {"-"*10}')
for name, rmse, _, _ in candidates:
    marker = ' <-- baseline' if name == 'v40 (XGB+Ridge)' else ''
    marker = ' <-- BEST' if rmse == min(c[1] for c in candidates) else marker
    print(f'  {name:<20} {rmse:>10.4f}{marker}')

# Find the overall LOSO-best
all_rmses = {
    'v40': rmse_v40,
    'rf_pure': best_rf_rmse,
    'et_pure': best_et_rmse, 
    'rf_ridge': best_rfr_rmse,
    'rf_xgb': best_blend_rmse,
    'ms_rf': best_ms_rmse,
    'fine_rf': best_fine_rmse,
}
best_key = min(all_rmses, key=all_rmses.get)
best_rmse_overall = all_rmses[best_key]

print(f'\n  LOSO-best model: {best_key} (RMSE={best_rmse_overall:.4f})')

# Build the final predictor for the LOSO-best
if best_key == 'v40':
    final_pred_fn = pred_v40
elif best_key == 'rf_pure':
    cfg = best_rf_cfg
    def final_pred_fn(Xtr, ytr, Xte):
        m = RandomForestRegressor(**cfg, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr); return m.predict(Xte)
elif best_key == 'et_pure':
    cfg = best_et_cfg
    def final_pred_fn(Xtr, ytr, Xte):
        m = ExtraTreesRegressor(**cfg, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr); return m.predict(Xte)
elif best_key == 'rf_ridge':
    cfg_rf, ra, rw = best_rfr_params
    def final_pred_fn(Xtr, ytr, Xte):
        m = RandomForestRegressor(**cfg_rf, random_state=42, n_jobs=-1)
        m.fit(Xtr, ytr); rf_p = m.predict(Xte)
        sc = StandardScaler(); rm = Ridge(alpha=ra)
        rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
        return (1 - rw) * rf_p + rw * rp
elif best_key == 'rf_xgb':
    cfg_rf, rf_w = best_blend_params
    def final_pred_fn(Xtr, ytr, Xte):
        m_rf = RandomForestRegressor(**cfg_rf, random_state=42, n_jobs=-1)
        m_rf.fit(Xtr, ytr); rf_p = m_rf.predict(Xte)
        xpreds = []
        for seed in SEEDS:
            m = xgb.XGBRegressor(**XGB_P, random_state=seed, verbosity=0)
            m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
        xgb_p = np.mean(xpreds, axis=0)
        sc = StandardScaler(); rm = Ridge(alpha=5.0)
        rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
        xgb_ridge = 0.70 * xgb_p + 0.30 * rp
        return rf_w * rf_p + (1 - rf_w) * xgb_ridge
elif best_key == 'ms_rf':
    seeds_best, msl_best = best_ms_cfg
    cfg_top = dict(top_cfg); cfg_top['min_samples_leaf'] = msl_best
    def final_pred_fn(Xtr, ytr, Xte):
        preds = []
        for s in seeds_best:
            m = RandomForestRegressor(**cfg_top, random_state=s, n_jobs=-1)
            m.fit(Xtr, ytr); preds.append(m.predict(Xte))
        return np.mean(preds, axis=0)
elif best_key == 'fine_rf':
    cfg_rf, seeds_f, ra_f, rw_f = best_fine_params
    def final_pred_fn(Xtr, ytr, Xte):
        rf_preds = []
        for s in seeds_f:
            m = RandomForestRegressor(**cfg_rf, random_state=s, n_jobs=-1)
            m.fit(Xtr, ytr); rf_preds.append(m.predict(Xte))
        rf_p = np.mean(rf_preds, axis=0)
        sc = StandardScaler(); rm = Ridge(alpha=ra_f)
        rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
        return (1 - rw_f) * rf_p + rw_f * rp

# Single test evaluation
print('\n  --- Final test evaluation (LOSO-best only) ---')
pred_final = final_pred_fn(X_tr, y_train, X_te)
assigned = hungarian(pred_final, test_seasons, test_avail)
test_exact = int(np.sum(assigned == test_gt))
test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))

print(f'  Test: {test_exact}/91 exact, RMSE={test_rmse:.4f}')
print(f'  LOSO: {best_rmse_overall:.4f} ({best_key})')
print(f'  Gap:  {abs(best_rmse_overall - test_rmse):.4f}')

if best_rmse_overall < rmse_v40:
    print(f'\n  *** RF BEATS v40! LOSO: {best_rmse_overall:.4f} vs {rmse_v40:.4f} ***')
    # Save
    sub_out = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub_out['RecordID'] == rid
        if mask.any():
            sub_out.loc[mask, 'Overall Seed'] = int(assigned[i])
    sub_out.to_csv(os.path.join(DATA_DIR, 'final_submission.csv'), index=False)
    print('  Saved: final_submission.csv')
else:
    print(f'\n  v40 still best by LOSO. Not saving.')

print(f'\n  Total time: {time.time()-t0:.0f}s')
