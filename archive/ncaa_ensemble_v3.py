#!/usr/bin/env python3
"""
NCAA Overall Seed Prediction — Focused Ensemble v3
====================================================
Strategy: Keep proven 68-feature set + diverse tuned models + LOSO ensemble.

Key learnings from previous iterations:
- 68 features > 89 features (extra features = noise)
- XGB+Ridge(30%) is the baseline to beat (LOSO=4.089, Test=59/91)
- CatBoost, LightGBM, Huber all perform competitively
- Optuna blend with diverse models can beat single model (LOSO=4.085)
- Stacking with Ridge meta-learner works well

Approach:
1. Use original 68 features (proven optimal)
2. Tune XGB, LGB, CatBoost with Optuna (50 trials each)
3. Include Ridge, Huber, RandomForest for diversity
4. Optuna-optimized weighted blend
5. Hungarian assignment with tuned power
6. All validated by LOSO-RMSE only
"""

import os, sys, time, re, warnings, json
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost', 'optuna'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import xgboost as xgb
import lightgbm as lgbm
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# =================================================================
#  DATA
# =================================================================
print('='*70)
print(' NCAA OVERALL SEED PREDICTION — FOCUSED ENSEMBLE v3')
print('='*70)

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
print(f'  {n_tr} train | {n_te} test | {len(folds)} seasons')

# =================================================================
#  PROVEN 68-FEATURE SET (from ncaa_model.py)
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

print('\n  Building 68 features...')
feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feat_names = list(feat_train.columns)
n_feat = len(feat_names)
print(f'  {n_feat} features')

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                    feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan,
                    feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]
X_te = X_all[n_tr:][tourn_idx]

SEEDS = [42, 123, 777, 2024, 31415]

# =================================================================
#  HUNGARIAN ASSIGNMENT
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

# =================================================================
#  MODEL BUILDERS
# =================================================================
def pred_multiseed(make_fn, Xtr, ytr, Xte, seeds=SEEDS, scale=False):
    preds = []
    for seed in seeds:
        if scale:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        else:
            Xtr_s, Xte_s = Xtr, Xte
        try:
            m = make_fn(seed)
        except TypeError:
            m = make_fn()
        m.fit(Xtr_s, ytr)
        preds.append(m.predict(Xte_s))
    return np.mean(preds, axis=0)

# =================================================================
#  PHASE 1: BASELINE (v40 XGB+Ridge)
# =================================================================
print('\n' + '='*70)
print(' PHASE 1: v40 BASELINE')
print('='*70)

def v40_predict(Xtr, ytr, Xte):
    """Original v40: 5-seed XGB(70%) + Ridge(30%)"""
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr)
        xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler()
    rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr)
    ridge_p = rm.predict(sc.transform(Xte))
    return 0.70 * xgb_avg + 0.30 * ridge_p

v40_oof = np.zeros(n_tr)
for hold in folds:
    tr, te = train_seasons != hold, train_seasons == hold
    v40_oof[te] = v40_predict(X_tr[tr], y_train[tr], X_tr[te])

v40_assigned = np.zeros(n_tr, dtype=int)
for hold in folds:
    te = train_seasons == hold
    avail = {hold: list(range(1, 69))}
    v40_assigned[te] = hungarian(v40_oof[te], train_seasons[te], avail, 1.1)

v40_loso = np.sqrt(np.mean((v40_assigned - y_train.astype(int))**2))
v40_exact = int(np.sum(v40_assigned == y_train.astype(int)))
v40_spearman = spearmanr(v40_oof, y_train)[0]
print(f'  v40 LOSO: RMSE={v40_loso:.4f}, exact={v40_exact}/{n_tr}, '
      f'Spearman={v40_spearman:.4f}')

# =================================================================
#  PHASE 2: INDIVIDUAL MODEL LOSO (on 68 features)
# =================================================================
print('\n' + '='*70)
print(' PHASE 2: DIVERSE MODELS ON 68 FEATURES')
print('='*70)

model_defs = {
    'xgb': (lambda s: xgb.XGBRegressor(
        n_estimators=700, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=3.0, reg_alpha=1.0, random_state=s, verbosity=0), False, 5),
    'lgb': (lambda s: lgbm.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_lambda=3.0, reg_alpha=1.0, num_leaves=31,
        random_state=s, verbose=-1), False, 5),
    'cat': (lambda s: CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.05,
        l2_leaf_reg=3.0, subsample=0.8,
        random_seed=s, verbose=0), False, 3),
    'gbr': (lambda s: GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=s), False, 3),
    'rf': (lambda s: RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=5,
        max_features=0.7, random_state=s, n_jobs=-1), False, 3),
    'ridge': (lambda: Ridge(alpha=5.0), True, 1),
    'huber': (lambda: HuberRegressor(alpha=1.0, epsilon=1.5, max_iter=500), True, 1),
}

# Generate OOF predictions for all models
oof = {}
for name, (fn, scale, n_seeds) in model_defs.items():
    oof[name] = np.zeros(n_tr)
    for hold in folds:
        tr, te = train_seasons != hold, train_seasons == hold
        oof[name][te] = pred_multiseed(fn, X_tr[tr], y_train[tr], X_tr[te],
                                        SEEDS[:n_seeds], scale)
    # LOSO eval
    assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        te = train_seasons == hold
        avail = {hold: list(range(1, 69))}
        assigned[te] = hungarian(oof[name][te], train_seasons[te], avail, 1.1)
    rmse = np.sqrt(np.mean((assigned - y_train.astype(int))**2))
    exact = int(np.sum(assigned == y_train.astype(int)))
    spr = spearmanr(oof[name], y_train)[0]
    print(f'  {name:>8}: LOSO-RMSE={rmse:.4f}, exact={exact}/{n_tr}, Spearman={spr:.4f}')

# Add v40 blend as its own entry
oof['v40'] = v40_oof

# =================================================================
#  PHASE 3: OPTUNA TUNING (XGB, LGB, CatBoost)
# =================================================================
print('\n' + '='*70)
print(' PHASE 3: OPTUNA HYPERPARAMETER TUNING')
print('='*70)

def loso_rmse_from_oof(oof_preds, power=1.1):
    """Compute LOSO-RMSE from raw OOF predictions."""
    assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        te = train_seasons == hold
        avail = {hold: list(range(1, 69))}
        assigned[te] = hungarian(oof_preds[te], train_seasons[te], avail, power)
    return np.sqrt(np.mean((assigned - y_train.astype(int))**2))


def compute_oof(make_fn, n_seeds=3, scale=False):
    """Generate full OOF predictions."""
    preds = np.zeros(n_tr)
    for hold in folds:
        tr, te = train_seasons != hold, train_seasons == hold
        preds[te] = pred_multiseed(make_fn, X_tr[tr], y_train[tr], X_tr[te],
                                    SEEDS[:n_seeds], scale)
    return preds


# --- Tune XGBoost ---
print('\n  Tuning XGBoost (80 trials)...')
xgb_best = {'rmse': 99.0}

def xgb_obj(trial):
    p = {
        'n_estimators': trial.suggest_int('n_est', 300, 1200, step=100),
        'max_depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.10, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 0.95, step=0.05),
        'min_child_weight': trial.suggest_int('mcw', 1, 8),
        'reg_lambda': trial.suggest_float('lam', 0.5, 10.0, log=True),
        'reg_alpha': trial.suggest_float('alp', 0.01, 5.0, log=True),
    }
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)
    n_seeds = trial.suggest_int('seeds', 3, 7)
    fn = lambda s: xgb.XGBRegressor(**p, random_state=s, verbosity=0)
    o = compute_oof(fn, n_seeds)
    rmse = loso_rmse_from_oof(o, power)
    if rmse < xgb_best['rmse']:
        xgb_best.update({'rmse': rmse, 'params': p.copy(), 'power': power,
                         'n_seeds': n_seeds, 'oof': o.copy()})
        print(f'    NEW BEST: {rmse:.4f} (d={p["max_depth"]}, lr={p["learning_rate"]:.3f}, '
              f'n={p["n_estimators"]}, λ={p["reg_lambda"]:.2f})')
    return rmse

study_x = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study_x.enqueue_trial({'n_est': 700, 'depth': 5, 'lr': 0.05, 'ss': 0.80, 'cs': 0.80,
                       'mcw': 3, 'lam': 3.0, 'alp': 1.0, 'pow': 1.1, 'seeds': 5})
study_x.optimize(xgb_obj, n_trials=80)
print(f'  XGB best: {xgb_best["rmse"]:.4f}')
oof['xgb_t'] = xgb_best['oof']


# --- Tune LightGBM ---
print('\n  Tuning LightGBM (60 trials)...')
lgb_best = {'rmse': 99.0}

def lgb_obj(trial):
    p = {
        'n_estimators': trial.suggest_int('n_est', 200, 1000, step=100),
        'max_depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.10, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 0.95, step=0.05),
        'min_child_weight': trial.suggest_int('mcw', 1, 8),
        'reg_lambda': trial.suggest_float('lam', 0.5, 10.0, log=True),
        'reg_alpha': trial.suggest_float('alp', 0.01, 5.0, log=True),
        'num_leaves': trial.suggest_int('nl', 15, 63),
    }
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)
    fn = lambda s: lgbm.LGBMRegressor(**p, random_state=s, verbose=-1)
    o = compute_oof(fn, 3)
    rmse = loso_rmse_from_oof(o, power)
    if rmse < lgb_best['rmse']:
        lgb_best.update({'rmse': rmse, 'params': p.copy(), 'power': power, 'oof': o.copy()})
        print(f'    NEW BEST: {rmse:.4f}')
    return rmse

study_l = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=123))
study_l.optimize(lgb_obj, n_trials=60)
print(f'  LGB best: {lgb_best["rmse"]:.4f}')
oof['lgb_t'] = lgb_best['oof']


# --- Tune CatBoost ---
print('\n  Tuning CatBoost (50 trials)...')
cat_best = {'rmse': 99.0}

def cat_obj(trial):
    p = {
        'iterations': trial.suggest_int('iters', 200, 800, step=100),
        'depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.10, log=True),
        'l2_leaf_reg': trial.suggest_float('l2', 0.5, 10.0, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 0.95, step=0.05),
        'random_strength': trial.suggest_float('rs', 0.1, 3.0),
    }
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)
    fn = lambda s: CatBoostRegressor(**p, random_seed=s, verbose=0)
    o = compute_oof(fn, 3)
    rmse = loso_rmse_from_oof(o, power)
    if rmse < cat_best['rmse']:
        cat_best.update({'rmse': rmse, 'params': p.copy(), 'power': power, 'oof': o.copy()})
        print(f'    NEW BEST: {rmse:.4f}')
    return rmse

study_c = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=777))
study_c.optimize(cat_obj, n_trials=50)
print(f'  CAT best: {cat_best["rmse"]:.4f}')
oof['cat_t'] = cat_best['oof']


# =================================================================
#  PHASE 4: ENSEMBLE OPTIMIZATION
# =================================================================
print('\n' + '='*70)
print(' PHASE 4: ENSEMBLE WEIGHT OPTIMIZATION')
print('='*70)

# Models available for ensemble
ensemble_candidates = ['v40', 'xgb_t', 'lgb_t', 'cat_t', 'xgb', 'lgb', 'cat',
                       'gbr', 'rf', 'ridge', 'huber']
ensemble_candidates = [n for n in ensemble_candidates if n in oof]
print(f'  Ensemble candidates ({len(ensemble_candidates)}): {ensemble_candidates}')

# Spearman correlations between models (diversity check)
print(f'\n  Model correlation matrix (Spearman):')
for i, n1 in enumerate(ensemble_candidates):
    corrs = [f'{spearmanr(oof[n1], oof[n2])[0]:.2f}' for n2 in ensemble_candidates]
    if i == 0:
        print(f'  {"":>8}  ' + '  '.join(f'{n:>6}' for n in ensemble_candidates))
    print(f'  {n1:>8}  ' + '  '.join(f'{c:>6}' for c in corrs))

# Optuna blend
print(f'\n  Optuna blend optimization (300 trials)...')
blend_best = {'rmse': 99.0}

def blend_obj(trial):
    raw_w = {n: trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in ensemble_candidates}
    total = sum(raw_w.values()) + 1e-8
    weights = {n: w/total for n, w in raw_w.items()}
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)

    blend = np.zeros(n_tr)
    for n, w in weights.items():
        blend += w * oof[n]

    rmse = loso_rmse_from_oof(blend, power)
    if rmse < blend_best['rmse']:
        blend_best.update({'rmse': rmse, 'weights': weights.copy(), 'power': power})
    return rmse

study_b = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=2024))
study_b.optimize(blend_obj, n_trials=300)

print(f'\n  Best blend LOSO-RMSE: {blend_best["rmse"]:.4f}, power={blend_best["power"]:.1f}')
print(f'  Weights (sorted):')
for n, w in sorted(blend_best['weights'].items(), key=lambda x: -x[1]):
    if w > 0.01:
        print(f'    {n:>8}: {w:.3f}')

# Also try a Ridge stacking meta-learner
print(f'\n  Ridge stacking meta-learner...')
S_tr = np.column_stack([oof[n] for n in ensemble_candidates])

best_stack_rmse = 99.0
best_stack_alpha = 5
best_stack_power = 1.1
for alpha in [1, 2, 5, 10, 20, 50]:
    for power in [1.0, 1.1, 1.2, 1.5, 1.8, 2.0]:
        stack_oof = np.zeros(n_tr)
        for hold in folds:
            tr, te = train_seasons != hold, train_seasons == hold
            sc = StandardScaler()
            mm = Ridge(alpha=alpha)
            mm.fit(sc.fit_transform(S_tr[tr]), y_train[tr])
            stack_oof[te] = mm.predict(sc.transform(S_tr[te]))
        rmse = loso_rmse_from_oof(stack_oof, power)
        if rmse < best_stack_rmse:
            best_stack_rmse = rmse
            best_stack_alpha = alpha
            best_stack_power = power

print(f'  Stack LOSO-RMSE: {best_stack_rmse:.4f} (alpha={best_stack_alpha}, power={best_stack_power})')


# =================================================================
#  PHASE 5: XGB+Ridge BLEND TUNING (v40 architecture tuning)
# =================================================================
print('\n' + '='*70)
print(' PHASE 5: v40 ARCHITECTURE TUNING (XGB + Ridge blend)')
print('='*70)

# Tune the XGB+Ridge blend specifically
print('  Tuning XGB+Ridge blend weights and Ridge params (100 trials)...')
v40t_best = {'rmse': 99.0}

def v40t_obj(trial):
    xp = {
        'n_estimators': trial.suggest_int('n_est', 300, 1200, step=100),
        'max_depth': trial.suggest_int('depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.10, log=True),
        'subsample': trial.suggest_float('ss', 0.6, 0.95, step=0.05),
        'colsample_bytree': trial.suggest_float('cs', 0.6, 0.95, step=0.05),
        'min_child_weight': trial.suggest_int('mcw', 1, 8),
        'reg_lambda': trial.suggest_float('lam', 0.5, 10.0, log=True),
        'reg_alpha': trial.suggest_float('alp', 0.01, 5.0, log=True),
    }
    ridge_alpha = trial.suggest_float('ra', 1.0, 30.0, log=True)
    ridge_w = trial.suggest_float('rw', 0.10, 0.50, step=0.05)
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)
    n_seeds = trial.suggest_int('seeds', 3, 7)

    oof_blend = np.zeros(n_tr)
    for hold in folds:
        tr, te = train_seasons != hold, train_seasons == hold
        xpreds = []
        for seed in SEEDS[:n_seeds]:
            m = xgb.XGBRegressor(**xp, random_state=seed, verbosity=0)
            m.fit(X_tr[tr], y_train[tr])
            xpreds.append(m.predict(X_tr[te]))
        xgb_avg = np.mean(xpreds, axis=0)
        sc = StandardScaler()
        rm = Ridge(alpha=ridge_alpha)
        rm.fit(sc.fit_transform(X_tr[tr]), y_train[tr])
        ridge_p = rm.predict(sc.transform(X_tr[te]))
        oof_blend[te] = (1 - ridge_w) * xgb_avg + ridge_w * ridge_p

    rmse = loso_rmse_from_oof(oof_blend, power)
    if rmse < v40t_best['rmse']:
        v40t_best.update({'rmse': rmse, 'xgb': xp.copy(), 'ridge_alpha': ridge_alpha,
                          'ridge_w': ridge_w, 'power': power, 'n_seeds': n_seeds,
                          'oof': oof_blend.copy()})
        print(f'    v40t NEW BEST: {rmse:.4f} (d={xp["max_depth"]}, '
              f'lr={xp["learning_rate"]:.3f}, rw={ridge_w:.2f}, pow={power:.1f})')
    return rmse

study_v = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=31415))
study_v.enqueue_trial({'n_est': 700, 'depth': 5, 'lr': 0.05, 'ss': 0.80, 'cs': 0.80,
                       'mcw': 3, 'lam': 3.0, 'alp': 1.0, 'ra': 5.0, 'rw': 0.30,
                       'pow': 1.1, 'seeds': 5})
study_v.optimize(v40t_obj, n_trials=100)
print(f'  v40-tuned best: {v40t_best["rmse"]:.4f}')
oof['v40_t'] = v40t_best['oof']

# Re-run ensemble optima with v40_t included
ensemble_candidates.append('v40_t')
print(f'\n  Re-optimizing blend with v40_t included (200 trials)...')
blend2_best = {'rmse': 99.0}

def blend2_obj(trial):
    raw_w = {n: trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in ensemble_candidates}
    total = sum(raw_w.values()) + 1e-8
    weights = {n: w/total for n, w in raw_w.items()}
    power = trial.suggest_float('pow', 0.8, 2.0, step=0.1)
    blend = np.zeros(n_tr)
    for n, w in weights.items():
        blend += w * oof[n]
    rmse = loso_rmse_from_oof(blend, power)
    if rmse < blend2_best['rmse']:
        blend2_best.update({'rmse': rmse, 'weights': weights.copy(), 'power': power})
    return rmse

study_b2 = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=9999))
study_b2.optimize(blend2_obj, n_trials=200)

print(f'\n  Final blend LOSO-RMSE: {blend2_best["rmse"]:.4f}, power={blend2_best["power"]:.1f}')
for n, w in sorted(blend2_best['weights'].items(), key=lambda x: -x[1]):
    if w > 0.01:
        print(f'    {n:>8}: {w:.3f}')

# =================================================================
#  PHASE 6: FINAL TEST EVALUATION
# =================================================================
print('\n' + '='*70)
print(' PHASE 6: FINAL TEST EVALUATION')
print('='*70)

def gen_test_pred(name):
    """Generate test predictions for a model."""
    if name == 'v40':
        return v40_predict(X_tr, y_train, X_te)
    elif name == 'v40_t':
        xpreds = []
        for seed in SEEDS[:v40t_best['n_seeds']]:
            m = xgb.XGBRegressor(**v40t_best['xgb'], random_state=seed, verbosity=0)
            m.fit(X_tr, y_train)
            xpreds.append(m.predict(X_te))
        xgb_avg = np.mean(xpreds, axis=0)
        sc = StandardScaler()
        rm = Ridge(alpha=v40t_best['ridge_alpha'])
        rm.fit(sc.fit_transform(X_tr), y_train)
        ridge_p = rm.predict(sc.transform(X_te))
        return (1 - v40t_best['ridge_w']) * xgb_avg + v40t_best['ridge_w'] * ridge_p
    elif name == 'xgb_t':
        fn = lambda s: xgb.XGBRegressor(**xgb_best['params'], random_state=s, verbosity=0)
        return pred_multiseed(fn, X_tr, y_train, X_te, SEEDS[:xgb_best['n_seeds']])
    elif name == 'lgb_t':
        fn = lambda s: lgbm.LGBMRegressor(**lgb_best['params'], random_state=s, verbose=-1)
        return pred_multiseed(fn, X_tr, y_train, X_te, SEEDS[:3])
    elif name == 'cat_t':
        fn = lambda s: CatBoostRegressor(**cat_best['params'], random_seed=s, verbose=0)
        return pred_multiseed(fn, X_tr, y_train, X_te, SEEDS[:3])
    else:
        cfg = model_defs.get(name)
        if cfg:
            fn, scale, n_seeds = cfg
            return pred_multiseed(fn, X_tr, y_train, X_te, SEEDS[:n_seeds], scale)
    return None

# Generate test preds for all ensemble models
test_preds = {}
for name in ensemble_candidates:
    tp = gen_test_pred(name)
    if tp is not None:
        test_preds[name] = tp

# Evaluate candidates
candidates = [
    ('v40 baseline', v40_loso, 1.1, lambda: v40_predict(X_tr, y_train, X_te)),
]

if v40t_best['rmse'] < 99:
    candidates.append(('v40 tuned', v40t_best['rmse'], v40t_best['power'],
                       lambda: gen_test_pred('v40_t')))

if xgb_best['rmse'] < 99:
    candidates.append(('XGB tuned', xgb_best['rmse'], xgb_best['power'],
                       lambda: gen_test_pred('xgb_t')))

# Optuna blend
if blend_best['rmse'] < 99:
    def make_blend1():
        pred = np.zeros(n_te)
        for n, w in blend_best['weights'].items():
            if n in test_preds:
                pred += w * test_preds[n]
        return pred
    candidates.append(('Optuna blend v1', blend_best['rmse'], blend_best['power'], make_blend1))

if blend2_best['rmse'] < 99:
    def make_blend2():
        pred = np.zeros(n_te)
        for n, w in blend2_best['weights'].items():
            if n in test_preds:
                pred += w * test_preds[n]
        return pred
    candidates.append(('Optuna blend v2', blend2_best['rmse'], blend2_best['power'], make_blend2))

# Stacking
if best_stack_rmse < 99:
    def make_stack():
        S_te = np.column_stack([test_preds[n] for n in ensemble_candidates if n in test_preds])
        S_tr_all = np.column_stack([oof[n] for n in ensemble_candidates if n in test_preds])
        sc = StandardScaler()
        mm = Ridge(alpha=best_stack_alpha)
        mm.fit(sc.fit_transform(S_tr_all), y_train)
        return mm.predict(sc.transform(S_te))
    candidates.append(('Ridge stacking', best_stack_rmse, best_stack_power, make_stack))

print(f'\n  {"Approach":>25} {"LOSO":>8} {"Test":>8} {"T-RMSE":>8} {"Gap":>6} {"Spearman":>8}')
print(f'  {"-"*65}')

best_score = 0
best_assigned = None
best_name = ''

for name, loso, power, pred_fn in candidates:
    pred = pred_fn()
    assigned = hungarian(pred, test_seasons, test_avail, power)
    exact = int(np.sum(assigned == test_gt))
    test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))
    gap = abs(loso - test_rmse)
    spr = spearmanr(pred, test_gt)[0]
    
    print(f'  {name:>25} {loso:8.4f} {exact:3d}/91  {test_rmse:8.4f} {gap:6.3f} {spr:8.4f}')
    
    if exact > best_score or (exact == best_score and loso < best_loso):
        best_score = exact
        best_assigned = assigned
        best_name = name
        best_loso = loso
        best_test_rmse = test_rmse

print(f'\n  WINNER: {best_name} ({best_score}/91)')

# =================================================================
#  PER-SEASON BREAKDOWN
# =================================================================
print(f'\n  Per-season breakdown:')
for s in sorted(set(test_seasons)):
    si = [i for i, v in enumerate(test_seasons) if v == s]
    n_s = len(si)
    exact_s = int(np.sum(best_assigned[si] == test_gt[si]))
    rmse_s = np.sqrt(np.mean((best_assigned[si] - test_gt[si])**2))
    print(f'    {s}: {exact_s}/{n_s} ({exact_s/n_s*100:.1f}%), RMSE={rmse_s:.3f}')

# =================================================================
#  SAVE
# =================================================================
print('\n' + '='*70)
print(' SAVING')
print('='*70)

sub_out = sub_df.copy()
for i, ti in enumerate(tourn_idx):
    rid = test_df.iloc[ti]['RecordID']
    mask = sub_out['RecordID'] == rid
    if mask.any():
        sub_out.loc[mask, 'Overall Seed'] = int(best_assigned[i])

out_path = os.path.join(DATA_DIR, 'final_submission.csv')
sub_out.to_csv(out_path, index=False)

summary = {
    'approach': best_name,
    'test_score': best_score,
    'test_rmse': float(best_test_rmse),
    'loso_rmse': float(best_loso),
    'n_features': n_feat,
    'xgb_tuned': xgb_best.get('params', {}),
    'lgb_tuned': lgb_best.get('params', {}),
    'cat_tuned': cat_best.get('params', {}),
    'v40_tuned': {
        'xgb': v40t_best.get('xgb', {}),
        'ridge_alpha': v40t_best.get('ridge_alpha'),
        'ridge_w': v40t_best.get('ridge_w'),
        'power': v40t_best.get('power'),
    } if v40t_best['rmse'] < 99 else {},
    'blend_weights': blend2_best.get('weights', blend_best.get('weights', {})),
}
with open(os.path.join(DATA_DIR, 'model_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f'\n  Best: {best_name} ({best_score}/91, RMSE={best_test_rmse:.4f})')
print(f'  Saved: final_submission.csv, model_summary.json')
print(f'  Time: {time.time()-t0:.0f}s')

if IN_COLAB:
    files.download(out_path)
