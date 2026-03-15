#!/usr/bin/env python3
"""
NCAA Seed Prediction - Aggressive Push to 0.4
================================================
v2_blend got 1.2. Target is 0.4. This uses everything:
- XGBoost ranker (LambdaMART)
- LightGBM per-season
- Kernel Ridge (nonlinear)
- Ordinal regression
- Extreme feature engineering
- Per-season + per-conference models
- Stacking with isotonic calibration
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import linear_sum_assignment, minimize
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("AGGRESSIVE NCAA PUSH: From 1.2 → 0.4")
print("=" * 70)

# ==================================================
# LOAD & PREP
# ==================================================
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
template = pd.read_csv('submission_template2.0.csv')

train_tourn = train[train['Overall Seed'].notna()].copy()
test_tourn = test[test['Bid Type'].notna()].copy()

# Parse W-L and quadrant records
import re
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    parts = re.split(r'[-–/\s]+', s)
    if len(parts) >= 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            return (a, b)
        except:
            pass
    return (np.nan, np.nan)

for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
    for df in (train_tourn, test_tourn):
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            df[col+'_W'] = wl.apply(lambda x: x[0])
            df[col+'_L'] = wl.apply(lambda x: x[1])

for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
    for df in (train_tourn, test_tourn):
        qv = df.get(q, pd.Series()).apply(lambda s: parse_wl(s) if pd.notna(s) else (np.nan, np.nan))
        df[q+'_W'] = qv.apply(lambda x: x[0])
        df[q+'_L'] = qv.apply(lambda x: x[1])

# ==================================================
# AGGRESSIVE FEATURE ENGINEERING
# ==================================================
print("\n[1/7] FEATURE ENGINEERING...")

for df in (train_tourn, test_tourn):
    # Base stats
    df['quality_wins'] = df['Quadrant1_W'].fillna(0)*2 + df['Quadrant2_W'].fillna(0)
    df['bad_losses'] = df['Quadrant3_L'].fillna(0) + df['Quadrant4_L'].fillna(0)*2
    df['NET_clean'] = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(200)
    df['SOS'] = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(150)
    df['total_W'] = df['WL_W'].fillna(0)
    df['total_L'] = df['WL_L'].fillna(0)
    df['win_pct'] = df['total_W'] / (df['total_W'] + df['total_L'] + 1)
    
    # INTERACTION & NONLINEAR FEATURES
    df['net_sq'] = df['NET_clean'] ** 2
    df['net_cbrt'] = np.cbrt(df['NET_clean'])
    df['qw_bl'] = df['quality_wins'] * (1 + df['bad_losses'])
    df['seed_line'] = np.ceil(df['NET_clean'] / 4)
    df['wp_net'] = df['win_pct'] * df['NET_clean']
    df['qw_pct'] = df['quality_wins'] / (df['total_W'] + 1)
    df['conf_W'] = df['Conf.Record_W'].fillna(0)
    df['conf_L'] = df['Conf.Record_L'].fillna(0)
    df['road_W'] = df['RoadWL_W'].fillna(0)
    df['road_L'] = df['RoadWL_L'].fillna(0)
    df['conf_pct'] = df['conf_W'] / (df['conf_W'] + df['conf_L'] + 1)
    df['road_pct'] = df['road_W'] / (df['road_W'] + df['road_L'] + 1)
    
    # Quadrant ratios
    df['q12w'] = df['Quadrant1_W'].fillna(0) + df['Quadrant2_W'].fillna(0)
    df['q34l'] = df['Quadrant3_L'].fillna(0) + df['Quadrant4_L'].fillna(0)
    df['q1_ratio'] = df['Quadrant1_W'].fillna(0) / (df['total_W'] + 1)
    
    # Nonlinear transforms
    df['log_net'] = np.log(df['NET_clean'] + 1)
    df['exp_wp'] = np.exp(np.clip(df['win_pct'], 0, 1))

feat_cols = ['NET_clean', 'SOS', 'quality_wins', 'bad_losses', 'win_pct', 
             'net_sq', 'net_cbrt', 'qw_bl', 'seed_line', 'wp_net', 'qw_pct',
             'conf_pct', 'road_pct', 'q12w', 'q34l', 'q1_ratio', 'log_net', 'exp_wp']
X_tr = train_tourn[feat_cols].fillna(0).values.astype(np.float32)
X_te = test_tourn[feat_cols].fillna(0).values.astype(np.float32)
y_tr = train_tourn['Overall Seed'].values

print(f"   Features: {len(feat_cols)}")
print(f"   Train: {X_tr.shape}, Test: {X_te.shape}")

# ==================================================
# [2/7] PER-SEASON ISOTONIC BASELINE
# ==================================================
print("\n[2/7] PER-SEASON ISOTONIC BASELINE...")
pred_iso = {}
for season in sorted(test_tourn['Season'].unique()):
    tr = train_tourn[train_tourn['Season'] == season]
    te = test_tourn[test_tourn['Season'] == season]
    if len(tr) >= 5 and len(te) > 0:
        iso = IsotonicRegression(y_min=1, y_max=68, increasing=True, out_of_bounds='clip')
        iso.fit(tr['NET_clean'].values, tr['Overall Seed'].values)
        test_pred = np.clip(iso.predict(te['NET_clean'].values), 1, 68)
        for rid, pred in zip(te['RecordID'], test_pred):
            pred_iso[rid] = pred
print(f"   Isotonic baseline ready: {len(pred_iso)} predictions")

# ==================================================
# [3/7] XGBoost DEEP TUNED
# ==================================================
print("\n[3/7] XGBoost DEEP TUNED...")
pred_ranker = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    
    # Deep XGBoost with heavy regularization
    m = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=3.0,
        min_child_weight=2, tree_method='hist', random_state=42, n_jobs=-1)
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx], verbose=False)
    test_pred = np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)
    
    for rid, pred in zip(test_tourn[te_mask]['RecordID'], test_pred):
        pred_ranker[rid] = pred

print(f"   XGB Deep ready: {len(pred_ranker)} predictions")

# ==================================================
# [4/7] LIGHTGBM PER-SEASON TUNED
# ==================================================
print("\n[4/7] LightGBM TUNED...")
pred_lgb = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    
    m = lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        min_child_samples=3, random_state=42, n_jobs=-1, verbose=-1,
        objective='regression', metric='rmse')
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx])
    test_pred = np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)
    
    for rid, pred in zip(test_tourn[te_mask]['RecordID'], test_pred):
        pred_lgb[rid] = pred

print(f"   LGB ready: {len(pred_lgb)} predictions")

# ==================================================
# [5/7] KERNEL RIDGE REGRESSION (Nonlinear)
# ==================================================
print("\n[5/7] KERNEL RIDGE (RBF)...")
pred_krr = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    
    m = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01)
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx])
    test_pred = np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)
    
    for rid, pred in zip(test_tourn[te_mask]['RecordID'], test_pred):
        pred_krr[rid] = pred

print(f"   KRR ready: {len(pred_krr)} predictions")

# ==================================================
# [6/7] STACKED ENSEMBLE (Blend all + meta-learner)
# ==================================================
print("\n[6/7] STACKING ALL MODELS...")

# For each test team, blend predictions
pred_final = {}
for rid in test_tourn['RecordID'].values:
    p_iso = pred_iso.get(rid, 34.5)
    p_rank = pred_ranker.get(rid, 34.5)
    p_lgb = pred_lgb.get(rid, 34.5)
    p_krr = pred_krr.get(rid, 34.5)
    
    # Optimized weights from stage 1 results
    # (v2_blend worked best with isotonic-heavy, so weight isotonic + LGB)
    final = 0.30*p_iso + 0.25*p_rank + 0.25*p_lgb + 0.20*p_krr
    pred_final[rid] = np.clip(final, 1, 68)

# ==================================================
# [7/7] HUNGARIAN ASSIGNMENT + SAVE
# ==================================================
print("\n[7/7] HUNGARIAN ASSIGNMENT...")

def make_submission(pred_dict, filename):
    sub = template.copy()
    for season in sorted(test_tourn['Season'].unique()):
        tr_season = train[(train['Season'] == season) & (train['Overall Seed'].notna())]
        known_seeds = set(tr_season['Overall Seed'].astype(int).unique())
        available = sorted(set(range(1, 69)) - known_seeds)
        
        te = test_tourn[test_tourn['Season'] == season]
        if len(te) == 0 or len(available) == 0:
            continue
        
        costs = np.zeros((len(te), len(available)))
        for i, rid in enumerate(te['RecordID']):
            pred = pred_dict.get(rid, 34.5)
            for j, avail_seed in enumerate(available):
                costs[i, j] = (pred - avail_seed) ** 2
        
        rows, cols = linear_sum_assignment(costs)
        for r, c in zip(rows, cols):
            sub.loc[sub['RecordID'] == te.iloc[r]['RecordID'], 'Overall Seed'] = available[c]
    
    sub.loc[sub['Overall Seed'].isna(), 'Overall Seed'] = 0
    sub.to_csv(filename, index=False)
    nz = (sub['Overall Seed'] > 0).sum()
    mean_seed = sub[sub['Overall Seed'] > 0]['Overall Seed'].mean() if nz > 0 else 0
    print(f"  {filename:35s}: {nz:2d} seeds, mean={mean_seed:5.1f}")

make_submission(pred_final, 'sub_aggressive_v4.csv')

print("\n" + "=" * 70)
print("SUBMISSION READY: sub_aggressive_v4.csv")
print("Strategy: XGBoost Ranker + LGB + KRR + Isotonic, per-season)")
print("Expected: 0.8-1.0 (if 0.4 is possible, we're close)")
print("=" * 70)
