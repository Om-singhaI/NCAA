#!/usr/bin/env python3
"""
NCAA Hyperparameter Tuning via Bayesian Optimization (Optuna)

Objective: minimize LOSO-RMSE (honest generalization metric).
Searches over:
  - XGB params (depth, lambda, alpha, lr, subsample, colsample, mcw, n_estimators)
  - Loss function (squared error, Huber, MAE)
  - Booster type (gbtree, dart)
  - Ridge alpha & blend weight
  - Hungarian power
  - Target transform (none, sqrt, log)
  - Number of seeds (3-10)
  - Feature subsets (drop noisy features)
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

print(f'{n_tr} train, {n_te} test')

# =================================================================
#  FEATURES
# =================================================================
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
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])
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
    feat['NET Rank'] = net; feat['PrevNET'] = prev
    feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count']   = conf.map(cs.count()).fillna(1)
    power_c = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_c).astype(float)
    cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
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
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
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
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)
    return feat

feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'{n_feat} features')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                    feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan,
                    feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]
X_te = X_all[n_tr:][tourn_idx]

ALL_SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]

# =================================================================
#  LOSO OBJECTIVE
# =================================================================
def hungarian(scores, seasons, avail, power=1.1):
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


def loso_rmse(xgb_params, ridge_alpha, ridge_w, n_seeds, power,
              target_transform='none', feat_mask=None):
    """LOSO-RMSE: our honest, non-overfitting objective."""
    folds = sorted(set(train_seasons))
    all_errors = []
    seeds = ALL_SEEDS[:n_seeds]

    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        Xtr = X_tr[tr]; Xte = X_tr[te]
        ytr = y_train[tr]; yte = y_train[te]

        if feat_mask is not None:
            Xtr = Xtr[:, feat_mask]; Xte = Xte[:, feat_mask]

        # Target transform
        if target_transform == 'sqrt':
            yt = np.sqrt(ytr)
        elif target_transform == 'log':
            yt = np.log1p(ytr)
        else:
            yt = ytr

        # XGB
        xgb_preds = []
        for seed in seeds:
            m = xgb.XGBRegressor(**xgb_params, random_state=seed, verbosity=0)
            m.fit(Xtr, yt)
            p = m.predict(Xte)
            if target_transform == 'sqrt':
                p = p ** 2
            elif target_transform == 'log':
                p = np.expm1(p)
            xgb_preds.append(p)
        xgb_avg = np.mean(xgb_preds, axis=0)

        # Ridge
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        rm = Ridge(alpha=ridge_alpha)
        rm.fit(Xtr_s, ytr)  # Ridge always on raw target
        ridge_pred = rm.predict(Xte_s)

        blend = (1 - ridge_w) * xgb_avg + ridge_w * ridge_pred

        avail = {hold: list(range(1, 69))}
        assigned = hungarian(blend, train_seasons[te], avail, power)
        errors = (assigned - yte.astype(int)) ** 2
        all_errors.extend(errors)

    return np.sqrt(np.mean(all_errors))


# =================================================================
#  OPTUNA STUDY
# =================================================================
print('\n' + '='*60)
print(' BAYESIAN HYPERPARAMETER OPTIMIZATION')
print('='*60)

best_so_far = {'rmse': 99.0, 'params': {}}
trial_count = [0]

def objective(trial):
    trial_count[0] += 1

    # XGB hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 300, 1200, step=100)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    lr = trial.suggest_float('learning_rate', 0.01, 0.10, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 0.95, step=0.05)
    colsample = trial.suggest_float('colsample_bytree', 0.6, 0.95, step=0.05)
    mcw = trial.suggest_int('min_child_weight', 1, 8)
    reg_lambda = trial.suggest_float('reg_lambda', 0.5, 10.0, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 0.01, 5.0, log=True)

    # Booster
    booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])

    # Loss function
    objective_fn = trial.suggest_categorical('objective',
        ['reg:squarederror', 'reg:pseudohubererror', 'reg:absoluteerror'])

    # Ridge
    ridge_alpha = trial.suggest_float('ridge_alpha', 1.0, 30.0, log=True)
    ridge_w = trial.suggest_float('ridge_w', 0.10, 0.50, step=0.05)

    # Hungarian
    power = trial.suggest_float('power', 0.8, 2.0, step=0.1)

    # Seeds
    n_seeds = trial.suggest_int('n_seeds', 3, 10)

    # Target transform
    target_transform = trial.suggest_categorical('target_transform',
        ['none', 'sqrt', 'log'])

    xgb_params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': lr,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'min_child_weight': mcw,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'booster': booster,
        'objective': objective_fn,
    }

    # DART-specific
    if booster == 'dart':
        xgb_params['rate_drop'] = trial.suggest_float('rate_drop', 0.01, 0.3, log=True)
        xgb_params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.9, step=0.1)
        # Fewer trees for DART (slow)
        xgb_params['n_estimators'] = min(n_estimators, 500)

    # Huber delta
    if objective_fn == 'reg:pseudohubererror':
        xgb_params['huber_slope'] = trial.suggest_float('huber_slope', 1.0, 10.0)

    rmse = loso_rmse(xgb_params, ridge_alpha, ridge_w, n_seeds, power,
                     target_transform)

    if rmse < best_so_far['rmse']:
        best_so_far['rmse'] = rmse
        best_so_far['params'] = {
            'xgb': xgb_params, 'ridge_alpha': ridge_alpha,
            'ridge_w': ridge_w, 'n_seeds': n_seeds,
            'power': power, 'transform': target_transform,
        }
        print(f'  [{trial_count[0]:3d}] NEW BEST LOSO-RMSE={rmse:.4f} '
              f'(d={max_depth}, λ={reg_lambda:.2f}, α={reg_alpha:.2f}, '
              f'lr={lr:.3f}, n={n_estimators}, '
              f'ss={subsample:.2f}, cs={colsample:.2f}, mcw={mcw}, '
              f'ridge_a={ridge_alpha:.1f}, ridge_w={ridge_w:.2f}, '
              f'pow={power:.1f}, seeds={n_seeds}, '
              f'obj={objective_fn}, boost={booster}, '
              f'trans={target_transform})')

    return rmse

# Run 300 trials with Bayesian optimization
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=42))

# Seed with our current best as the first trial
study.enqueue_trial({
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.80, 'colsample_bytree': 0.80, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0,
    'booster': 'gbtree', 'objective': 'reg:squarederror',
    'ridge_alpha': 5.0, 'ridge_w': 0.30, 'power': 1.1,
    'n_seeds': 5, 'target_transform': 'none',
})

# Also seed with previous best (mod params from v33)
study.enqueue_trial({
    'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03,
    'subsample': 0.80, 'colsample_bytree': 0.80, 'min_child_weight': 3,
    'reg_lambda': 2.0, 'reg_alpha': 0.5,
    'booster': 'gbtree', 'objective': 'reg:squarederror',
    'ridge_alpha': 5.0, 'ridge_w': 0.30, 'power': 1.1,
    'n_seeds': 5, 'target_transform': 'none',
})

print(f'\n  Running 300 Optuna trials (Bayesian search)...')
print(f'  Current best: LOSO-RMSE=4.0889\n')

study.optimize(objective, n_trials=300, show_progress_bar=False)

print(f'\n  Study complete: {len(study.trials)} trials')
print(f'  Best LOSO-RMSE: {study.best_value:.4f}')

# =================================================================
#  FEATURE SELECTION ON BEST PARAMS
# =================================================================
print('\n' + '='*60)
print(' FEATURE SELECTION (backward elimination)')
print('='*60)

bp = best_so_far['params']
# Try dropping each feature and see if LOSO improves
base_rmse = best_so_far['rmse']
print(f'  Baseline: {n_feat} features, LOSO-RMSE={base_rmse:.4f}')

drop_results = []
for fi in range(n_feat):
    mask = np.ones(n_feat, dtype=bool)
    mask[fi] = False
    rmse = loso_rmse(bp['xgb'], bp['ridge_alpha'], bp['ridge_w'],
                     bp['n_seeds'], bp['power'], bp['transform'], mask)
    delta = rmse - base_rmse
    drop_results.append((fi, feat_names[fi], rmse, delta))

drop_results.sort(key=lambda x: x[2])
print(f'\n  Top features to DROP (lower RMSE = better without them):')
print(f'  {"#":>3} {"Feature":>25} {"RMSE":>8} {"Delta":>8}')
for fi, fname, rmse, delta in drop_results[:15]:
    marker = ' ✓' if delta < -0.01 else ''
    print(f'  {fi:3d} {fname:>25} {rmse:8.4f} {delta:+8.4f}{marker}')

# Greedily drop features that improve LOSO
dropped = set()
current_mask = np.ones(n_feat, dtype=bool)
current_rmse = base_rmse

for iteration in range(10):  # max 10 features to drop
    best_drop = None
    best_drop_rmse = current_rmse
    
    for fi in range(n_feat):
        if fi in dropped:
            continue
        trial_mask = current_mask.copy()
        trial_mask[fi] = False
        rmse = loso_rmse(bp['xgb'], bp['ridge_alpha'], bp['ridge_w'],
                         bp['n_seeds'], bp['power'], bp['transform'], trial_mask)
        if rmse < best_drop_rmse - 0.001:  # require meaningful improvement
            best_drop = fi
            best_drop_rmse = rmse
    
    if best_drop is not None:
        dropped.add(best_drop)
        current_mask[best_drop] = False
        current_rmse = best_drop_rmse
        remaining = int(current_mask.sum())
        print(f'  Drop #{iteration+1}: {feat_names[best_drop]:>25} → '
              f'RMSE={current_rmse:.4f} ({remaining} features)')
    else:
        print(f'  No more features worth dropping.')
        break

feat_selected = current_rmse < base_rmse
if feat_selected:
    print(f'\n  Feature selection improved: {base_rmse:.4f} → {current_rmse:.4f} '
          f'({n_feat}→{int(current_mask.sum())} features)')
    final_mask = current_mask
    final_rmse = current_rmse
else:
    print(f'\n  Feature selection did not help. Keeping all {n_feat} features.')
    final_mask = None
    final_rmse = base_rmse


# =================================================================
#  FINAL COMPARISON
# =================================================================
print('\n' + '='*60)
print(' FINAL RESULTS')
print('='*60)

# Test with best params
def test_score(xgb_params, ridge_alpha, ridge_w, n_seeds, power,
               target_transform='none', feat_mask=None):
    seeds = ALL_SEEDS[:n_seeds]
    Xtr = X_tr; Xte = X_te
    if feat_mask is not None:
        Xtr = Xtr[:, feat_mask]; Xte = Xte[:, feat_mask]
    
    if target_transform == 'sqrt':
        yt = np.sqrt(y_train)
    elif target_transform == 'log':
        yt = np.log1p(y_train)
    else:
        yt = y_train

    xgb_preds = []
    for seed in seeds:
        m = xgb.XGBRegressor(**xgb_params, random_state=seed, verbosity=0)
        m.fit(Xtr, yt)
        p = m.predict(Xte)
        if target_transform == 'sqrt': p = p ** 2
        elif target_transform == 'log': p = np.expm1(p)
        xgb_preds.append(p)
    xgb_avg = np.mean(xgb_preds, axis=0)

    sc = StandardScaler()
    rm = Ridge(alpha=ridge_alpha)
    rm.fit(sc.fit_transform(Xtr), y_train)
    ridge_pred = rm.predict(sc.transform(Xte))

    blend = (1 - ridge_w) * xgb_avg + ridge_w * ridge_pred
    assigned = hungarian(blend, test_seasons, test_avail, power)
    exact = int(np.sum(assigned == test_gt))
    rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    return assigned, exact, rmse

# Current best (from ncaa_model.py)
old_xgb = {'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
            'reg_lambda': 3.0, 'reg_alpha': 1.0}
_, old_exact, old_test_rmse = test_score(old_xgb, 5.0, 0.30, 5, 1.1)
old_loso = 4.0889

# New best from Optuna
_, new_exact, new_test_rmse = test_score(
    bp['xgb'], bp['ridge_alpha'], bp['ridge_w'],
    bp['n_seeds'], bp['power'], bp['transform'])

# With feature selection (if it helped)
if feat_selected:
    assigned_fs, fs_exact, fs_test_rmse = test_score(
        bp['xgb'], bp['ridge_alpha'], bp['ridge_w'],
        bp['n_seeds'], bp['power'], bp['transform'], final_mask)
else:
    fs_exact, fs_test_rmse, final_rmse = new_exact, new_test_rmse, best_so_far['rmse']

print(f'\n  {"Config":>30} {"LOSO-RMSE":>10} {"Test":>8} {"T-RMSE":>8}')
print(f'  {"Current (v40)":>30} {old_loso:10.4f} {old_exact:3d}/91  {old_test_rmse:8.4f}')
print(f'  {"Optuna best":>30} {best_so_far["rmse"]:10.4f} {new_exact:3d}/91  {new_test_rmse:8.4f}')
if feat_selected:
    print(f'  {"Optuna + feat selection":>30} {final_rmse:10.4f} {fs_exact:3d}/91  {fs_test_rmse:8.4f}')

# =================================================================
#  PRINT BEST PARAMS FOR ncaa_model.py
# =================================================================
print(f'\n' + '='*60)
print(' BEST PARAMETERS (copy to ncaa_model.py)')
print('='*60)

xp = bp['xgb']
print(f'''
XGB_PARAMS = {{
    'n_estimators': {xp['n_estimators']},
    'max_depth': {xp['max_depth']},
    'learning_rate': {xp['learning_rate']:.4f},
    'subsample': {xp['subsample']:.2f},
    'colsample_bytree': {xp['colsample_bytree']:.2f},
    'min_child_weight': {xp['min_child_weight']},
    'reg_lambda': {xp['reg_lambda']:.4f},
    'reg_alpha': {xp['reg_alpha']:.4f},
''')
if xp.get('booster') == 'dart':
    print(f"    'booster': 'dart',")
    print(f"    'rate_drop': {xp.get('rate_drop', 0.1):.4f},")
    print(f"    'skip_drop': {xp.get('skip_drop', 0.5):.1f},")
if xp.get('objective') != 'reg:squarederror':
    print(f"    'objective': '{xp['objective']}',")
    if xp.get('huber_slope'):
        print(f"    'huber_slope': {xp['huber_slope']:.1f},")
print(f'}}')
print(f"RIDGE_ALPHA = {bp['ridge_alpha']:.2f}")
print(f"RIDGE_WEIGHT = {bp['ridge_w']:.2f}")
print(f"HUNGARIAN_POWER = {bp['power']:.1f}")
print(f"N_SEEDS = {bp['n_seeds']}")
print(f"TARGET_TRANSFORM = '{bp['transform']}'")

if feat_selected:
    dropped_names = [feat_names[i] for i in range(n_feat) if not final_mask[i]]
    print(f"\nDROP_FEATURES = {dropped_names}")

# =================================================================
#  SAVE BEST SUBMISSION
# =================================================================
if feat_selected:
    best_assigned = assigned_fs
    best_exact = fs_exact
    best_loso = final_rmse
else:
    best_assigned, best_exact, _ = test_score(
        bp['xgb'], bp['ridge_alpha'], bp['ridge_w'],
        bp['n_seeds'], bp['power'], bp['transform'])
    best_loso = best_so_far['rmse']

sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(best_assigned[i])

out_path = os.path.join(DATA_DIR, 'final_submission.csv')
sub_out.to_csv(out_path, index=False)
print(f'\n  Saved: final_submission.csv ({best_exact}/91, LOSO={best_loso:.4f})')
print(f'  Total time: {time.time()-t0:.0f}s')
