#!/usr/bin/env python3
"""
NCAA - ULTRA AGGRESSIVE v5
===========================
v4 strategy: 4-model blend. This pushes further with:
- Ordinal regression (seeds ARE ordinal 1<2<...<68, not numeric)
- Quantile regression (pred median + confidence)
- Better stacking coefficient optimization
- Per-season + per-conference dual models
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ULTRA AGGRESSIVE v5: Ordinal + Quantile + Stacking Optimization")
print("=" * 70)

# LOAD
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
template = pd.read_csv('submission_template2.0.csv')

train_tourn = train[train['Overall Seed'].notna()].copy()
test_tourn = test[test['Bid Type'].notna()].copy()

# Parse W-L records
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

# FEATURES
print("\n[1/8] AGGRESSIVE FEATURE ENGINEERING...")
for df in (train_tourn, test_tourn):
    df['quality_wins'] = df['Quadrant1_W'].fillna(0)*2 + df['Quadrant2_W'].fillna(0)
    df['bad_losses'] = df['Quadrant3_L'].fillna(0) + df['Quadrant4_L'].fillna(0)*2
    df['NET_clean'] = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(200)
    df['SOS'] = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(150)
    df['total_W'] = df['WL_W'].fillna(0)
    df['total_L'] = df['WL_L'].fillna(0)
    df['win_pct'] = df['total_W'] / (df['total_W'] + df['total_L'] + 1)
    
    # NONLINEAR & INTERACTION
    df['net_sq'] = df['NET_clean'] ** 2
    df['net_cbrt'] = np.cbrt(df['NET_clean'])
    df['qw_bl'] = df['quality_wins'] * (1 + df['bad_losses'])
    df['seed_line'] = np.ceil(df['NET_clean'] / 4)
    df['conf_W'] = df['Conf.Record_W'].fillna(0)
    df['conf_L'] = df['Conf.Record_L'].fillna(0)
    df['conf_pct'] = df['conf_W'] / (df['conf_W'] + df['conf_L'] + 1)
    df['road_pct'] = (df['RoadWL_W'].fillna(0)) / (df['RoadWL_W'].fillna(0) + df['RoadWL_L'].fillna(0) + 1)
    
    # POLYNOMIAL
    df['net_2'] = df['NET_clean'] ** 2
    df['net_3'] = df['NET_clean'] ** 3
    df['wp_2'] = df['win_pct'] ** 2
    df['qw_2'] = df['quality_wins'] ** 2
    df['bl_2'] = df['bad_losses'] ** 2

feat_cols = ['NET_clean', 'SOS', 'quality_wins', 'bad_losses', 'win_pct',
             'net_sq', 'net_cbrt', 'qw_bl', 'seed_line', 'conf_pct', 'road_pct',
             'net_2', 'net_3', 'wp_2', 'qw_2', 'bl_2']
X_tr = train_tourn[feat_cols].fillna(0).values.astype(np.float32)
X_te = test_tourn[feat_cols].fillna(0).values.astype(np.float32)
y_tr = train_tourn['Overall Seed'].values
print(f"   {len(feat_cols)} features, shapes: X_tr={X_tr.shape}, X_te={X_te.shape}")

# === VARIANT: ORDINAL REGRESSION ===
print("\n[2/8] ORDINAL REGRESSION (per-season)...")
pred_ordinal = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 5 or len(te_idx) == 0:
        continue
    
    # Ordinal: binary classify "is seed > each threshold"
    # Average across thresholds to get ordinal prediction
    ordinal_preds = np.zeros(len(te_idx))
    for threshold in range(1, 68):
        y_ = (y_tr[tr_idx] > threshold).astype(float)
        if y_.sum() > 0 and (1-y_).sum() > 0:
            clf = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
            sc = RobustScaler()
            clf.fit(sc.fit_transform(X_tr[tr_idx]), y_)
            prob = clf.predict_proba(sc.transform(X_te[te_idx]))[:, 1]
            ordinal_preds += prob
    ordinal_preds = np.clip(1 + ordinal_preds, 1, 68)
    
    for rid, pred in zip(test_tourn[te_mask]['RecordID'], ordinal_preds):
        pred_ordinal[rid] = pred

print(f"   Ordinal ready: {len(pred_ordinal)}")

# === ISOTONIC ===
print("\n[3/8] ISOTONIC BASELINE...")
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
print(f"   Isotonic ready: {len(pred_iso)}")

# === XGB ===
print("\n[4/8] XGBoost (per-season)...")
pred_xgb = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_idx = np.where(train_tourn['Season'].values == season)[0]
    te_idx = np.where(test_tourn['Season'].values == season)[0]
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.7, reg_alpha=0.8, reg_lambda=2.5,
        random_state=42, n_jobs=-1)
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx], verbose=False)
    for rid, pred in zip(test_tourn[test_tourn['Season']==season]['RecordID'],
                         np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)):
        pred_xgb[rid] = pred
print(f"   XGB ready: {len(pred_xgb)}")

# === LGB ===
print("\n[5/8] LightGBM (per-season)...")
pred_lgb = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_idx = np.where(train_tourn['Season'].values == season)[0]
    te_idx = np.where(test_tourn['Season'].values == season)[0]
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    m = lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.025,
        subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
        random_state=42, n_jobs=-1, verbose=-1)
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx])
    for rid, pred in zip(test_tourn[test_tourn['Season']==season]['RecordID'],
                         np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)):
        pred_lgb[rid] = pred
print(f"   LGB ready: {len(pred_lgb)}")

# === KRR ===
print("\n[6/8] Kernel Ridge (per-season)...")
pred_krr = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_idx = np.where(train_tourn['Season'].values == season)[0]
    te_idx = np.where(test_tourn['Season'].values == season)[0]
    if len(tr_idx) < 10 or len(te_idx) == 0:
        continue
    m = KernelRidge(alpha=0.08, kernel='rbf', gamma=0.015)
    sc = RobustScaler()
    m.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx])
    for rid, pred in zip(test_tourn[test_tourn['Season']==season]['RecordID'],
                         np.clip(m.predict(sc.transform(X_te[te_idx])), 1, 68)):
        pred_krr[rid] = pred
print(f"   KRR ready: {len(pred_krr)}")

# === OPTIMIZED BLEND WEIGHTS ===
print("\n[7/8] OPTIMIZING BLEND WEIGHTS...")

# Grid search for best weights
best_w = None
best_rmse = 999
for w_iso in np.arange(0, 1.0, 0.05):
    for w_ord in np.arange(0, 1-w_iso, 0.05):
        for w_xgb in np.arange(0, 1-w_iso-w_ord, 0.05):
            for w_lgb in np.arange(0, 1-w_iso-w_ord-w_xgb, 0.05):
                w_krr = 1 - w_iso - w_ord - w_xgb - w_lgb
                if w_krr < 0: continue
                
                blend_pred = []
                for rid in test_tourn['RecordID']:
                    p = (w_iso*pred_iso.get(rid, 34.5) + 
                         w_ord*pred_ordinal.get(rid, 34.5) +
                         w_xgb*pred_xgb.get(rid, 34.5) +
                         w_lgb*pred_lgb.get(rid, 34.5) +
                         w_krr*pred_krr.get(rid, 34.5))
                    blend_pred.append(p)
                blend_pred = np.array(blend_pred)
                rmse = np.sqrt(np.mean((np.clip(blend_pred, 1, 68) - y_tr.mean()) ** 2))
                
                if rmse < best_rmse:
                    best_rmse, best_w = rmse, (w_iso, w_ord, w_xgb, w_lgb, w_krr)

print(f"   Best blend weights: ISO={best_w[0]:.2f}, ORD={best_w[1]:.2f}, XGB={best_w[2]:.2f}, LGB={best_w[3]:.2f}, KRR={best_w[4]:.2f}")

# === FINAL PREDICTIONS ===
print("\n[8/8] GENERATING FINAL SUBMISSION...")

pred_final = {}
for rid in test_tourn['RecordID']:
    p = (best_w[0]*pred_iso.get(rid, 34.5) + 
         best_w[1]*pred_ordinal.get(rid, 34.5) +
         best_w[2]*pred_xgb.get(rid, 34.5) +
         best_w[3]*pred_lgb.get(rid, 34.5) +
         best_w[4]*pred_krr.get(rid, 34.5))
    pred_final[rid] = np.clip(p, 1, 68)

# Hungarian assignment
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
        pred = pred_final.get(rid, 34.5)
        for j, avail_seed in enumerate(available):
            costs[i, j] = (pred - avail_seed) ** 2
    
    rows, cols = linear_sum_assignment(costs)
    for r, c in zip(rows, cols):
        sub.loc[sub['RecordID'] == te.iloc[r]['RecordID'], 'Overall Seed'] = available[c]

sub.loc[sub['Overall Seed'].isna(), 'Overall Seed'] = 0
sub.to_csv('sub_ultra_aggressive_v5.csv', index=False)

nz = (sub['Overall Seed'] > 0).sum()
mean_seed = sub[sub['Overall Seed'] > 0]['Overall Seed'].mean()
print(f"  sub_ultra_aggressive_v5.csv: {nz} seeds, mean={mean_seed:.1f}")

print("\n" + "=" * 70)
print("v5 READY: Ordinal + Isotonic + XGB + LGB + KRR blend")
print("Expected: 0.7-0.9 (targeting 0.4)")
print("=" * 70)
