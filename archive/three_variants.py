#!/usr/bin/env python3
"""
NCAA Seed Prediction - Three Submission Variants
==================================================
Test which approach generalizes best to leaderboard.
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy.optimize import linear_sum_assignment, minimize
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# LOAD & PREP
# ==================================================
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
template = pd.read_csv('submission_template2.0.csv')

train_tourn = train[train['Overall Seed'].notna()].copy()
test_tourn = test[test['Bid Type'].notna()].copy()

def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    import re
    parts = re.split(r'[-–/\s]+', s)
    if len(parts) >= 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            return (a, b)
        except:
            pass
    return (np.nan, np.nan)

for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord']:
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

for df in (train_tourn, test_tourn):
    df['quality_wins'] = df['Quadrant1_W'].fillna(0)*2 + df['Quadrant2_W'].fillna(0)
    df['bad_losses'] = df['Quadrant3_L'].fillna(0) + df['Quadrant4_L'].fillna(0)*2
    df['NET_clean'] = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(200)
    df['SOS'] = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(150)

# ==================================================
# VARIANT 1: Pure Isotonic (NET rank only)
# ==================================================
print("=== Variant 1: Isotonic (NET only) ===")
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

# ==================================================
# VARIANT 2: Isotonic + Ridge blend
# ==================================================
print("=== Variant 2: Isotonic + Ridge (3 features + NET) ===")

# Build features for Ridge
features_ridge = ['NET_clean', 'quality_wins', 'bad_losses', 'SOS']
X_tr = train_tourn[features_ridge].fillna(train_tourn[features_ridge].median()).values.astype(np.float32)
X_te = test_tourn[features_ridge].fillna(test_tourn[features_ridge].median()).values.astype(np.float32)
y_tr = train_tourn['Overall Seed'].values

# Per-season isotonic + Ridge
pred_ridge = {}
pred_blend = {}

for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 5 or len(te_idx) == 0:
        continue
    
    # Isotonic (already computed above)
    tr_season = train_tourn[tr_mask]
    te_season = test_tourn[te_mask]
    iso = IsotonicRegression(y_min=1, y_max=68, increasing=True, out_of_bounds='clip')
    iso.fit(tr_season['NET_clean'].values, tr_season['Overall Seed'].values)
    iso_pred = np.clip(iso.predict(te_season['NET_clean'].values), 1, 68)
    
    # Ridge on 4 features
    ridge = Ridge(alpha=10)
    sc = RobustScaler()
    ridge.fit(sc.fit_transform(X_tr[tr_idx]), y_tr[tr_idx])
    ridge_pred = np.clip(ridge.predict(sc.transform(X_te[te_idx])), 1, 68)
    
    # Blend: 70% isotonic, 30% ridge
    blend_pred = 0.7 * iso_pred + 0.3 * ridge_pred
    
    for rid, p_ridge, p_blend in zip(te_season['RecordID'], ridge_pred, blend_pred):
        pred_ridge[rid] = p_ridge
        pred_blend[rid] = p_blend

# ==================================================
# VARIANT 3: Isotonic + Ridge + Linear Features
# ==================================================
print("=== Variant 3: Isotonic + Ridge + seed_line_est ===")

# Add seed_line_est feature
for df in (train_tourn, test_tourn):
    df['seed_line_est'] = np.ceil(df['NET_clean'] / 4)

features_ext = ['NET_clean', 'quality_wins', 'bad_losses', 'SOS', 'seed_line_est']
X_tr_ext = train_tourn[features_ext].fillna(train_tourn[features_ext].median()).values.astype(np.float32)
X_te_ext = test_tourn[features_ext].fillna(test_tourn[features_ext].median()).values.astype(np.float32)

pred_ext = {}
for season in sorted(test_tourn['Season'].unique()):
    tr_mask = (train_tourn['Season'].values == season)
    te_mask = (test_tourn['Season'].values == season)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    
    if len(tr_idx) < 5 or len(te_idx) == 0:
        continue
    
    ridge = Ridge(alpha=8)
    sc = RobustScaler()
    ridge.fit(sc.fit_transform(X_tr_ext[tr_idx]), y_tr[tr_idx])
    pred = np.clip(ridge.predict(sc.transform(X_te_ext[te_idx])), 1, 68)
    
    for rid, p in zip(test_tourn[te_mask]['RecordID'], pred):
        pred_ext[rid] = p

# ==================================================
# HUNGARIAN ASSIGNMENT + SAVE
# ==================================================
def make_submission(pred_dict, filename):
    sub = template.copy()
    for season in sorted(test_tourn['Season'].unique()):
        tr_season = train[( train['Season'] == season) & (train['Overall Seed'].notna())]
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
    print(f"  {filename:30s}: {nz} seeds, mean={mean_seed:.1f}")

print("\n=== Generating Submissions ===")
make_submission(pred_iso,   'sub_v1_isotonic.csv')
make_submission(pred_blend, 'sub_v2_blend.csv')
make_submission(pred_ext,   'sub_v3_ridge_ext.csv')

print("\n=== TESTING STRATEGY ===")
print("1. Try sub_v1_isotonic.csv first (simplest, least overfitting)")
print("2. If score < 0.8, try sub_v2_blend.csv")
print("3. If still > 0.8, try sub_v3_ridge_ext.csv")
print("\nRanking by expected conservativeness:")
print("  Best: v1_isotonic (no overfitting)")
print("  2nd:  v2_blend (isotonic-dominated)")
print("  3rd:  v3_ridge_ext (more risky)")
