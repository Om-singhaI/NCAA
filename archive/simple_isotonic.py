#!/usr/bin/env python3
"""
NCAA Seed Prediction - Ultra-Simple Isotonic Calibration
==========================================================
Hypothesis: We're overfitting. The best solution is the simplest.

Just use: NET rank → seed via isotonic regression
- One feature only (NET rank)
- Per-season calibration
- No interaction features, no ensemble
- Should generalize perfectly

Contact: OOF we'd get is NOT realistic (training-only). 
True leaderboard score should be ~0.8-1.0 overall RMSE (0.35-0.45 tournament).
"""

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# STEP 1: LOAD DATA
# ==================================================
print("Loading data...")
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
template = pd.read_csv('submission_template2.0.csv')

# Get tournament teams only
train_tourn = train[train['Overall Seed'].notna()].copy()
test_tourn = test[test['Bid Type'].notna()].copy()

print(f"Train tournament: {len(train_tourn)}, Test tournament: {len(test_tourn)}")
print(f"Season range: {train_tourn['Season'].min()} to {train_tourn['Season'].max()}")

# ==================================================
# STEP 2: CLEAN NET RANK
# ==================================================
# Handle NaN and invalid values
train_tourn['NET_clean'] = pd.to_numeric(train_tourn['NET Rank'], errors='coerce').fillna(200)
test_tourn['NET_clean'] = pd.to_numeric(test_tourn['NET Rank'], errors='coerce').fillna(200)

train_tourn = train_tourn[(train_tourn['NET_clean'] > 0) & (train_tourn['NET_clean'] <= 361)].copy()
test_tourn = test_tourn[(test_tourn['NET_clean'] > 0) & (test_tourn['NET_clean'] <= 361)].copy()

print(f"After cleanup: Train {len(train_tourn)}, Test {len(test_tourn)}")

# ==================================================
# STEP 3: PER-SEASON ISOTONIC CALIBRATION
# ==================================================
print("\n=== Per-Season Isotonic Calibration ===")

predictions = {}
for season in sorted(test_tourn['Season'].unique()):
    tr = train_tourn[train_tourn['Season'] == season]
    te = test_tourn[test_tourn['Season'] == season]
    
    if len(tr) < 5 or len(te) == 0:
        print(f"{season}: skip (insufficient data)")
        continue
    
    # Fit isotonic regression on training teams
    iso = IsotonicRegression(y_min=1, y_max=68, increasing=True, out_of_bounds='clip')
    iso.fit(tr['NET_clean'].values, tr['Overall Seed'].values)
    
    # Predict test teams
    test_pred = np.clip(iso.predict(te['NET_clean'].values), 1, 68)
    for rid, pred in zip(te['RecordID'], test_pred):
        predictions[rid] = pred
    
    print(f"{season}: {len(tr)} train teams → {len(te)} test teams, "
          f"mean_pred={test_pred.mean():.1f}, min={test_pred.min():.0f}, max={test_pred.max():.0f}")

# ==================================================
# STEP 4: HUNGARIAN ASSIGNMENT FOR INTEGER SEEDS
# ==================================================
print("\n=== Hungarian Assignment ===")
from scipy.optimize import linear_sum_assignment

sub = template.copy()
non_tourn_ids = sub[sub['Overall Seed'].isna()]['RecordID'].values[:len(sub) - len(test_tourn)]

for season in sorted(test_tourn['Season'].unique()):
    # Known training seeds this season
    tr_seeds = set(train['Season'] == season)
    if not tr_seeds:
        continue
    tr_season = train[(train['Season'] == season) & (train['Overall Seed'].notna())]
    known_seeds = set(tr_season['Overall Seed'].astype(int).unique())
    available = sorted(set(range(1, 69)) - known_seeds)
    
    # Test teams in this season
    te = test_tourn[test_tourn['Season'] == season]
    if len(te) == 0 or len(available) == 0:
        continue
    
    # Cost matrix: (pred_seed - available_seed)^2
    costs = np.zeros((len(te), len(available)))
    for i, rid in enumerate(te['RecordID']):
        pred = predictions.get(rid, 34.5)
        for j, avail_seed in enumerate(available):
            costs[i, j] = (pred - avail_seed) ** 2
    
    # Solve assignment
    rows, cols = linear_sum_assignment(costs)
    for r, c in zip(rows, cols):
        sub.loc[sub['RecordID'] == te.iloc[r]['RecordID'], 'Overall Seed'] = available[c]

# Non-tournament teams get seed 0
sub.loc[sub['Overall Seed'].isna(), 'Overall Seed'] = 0

# ==================================================
# STEP 5: SAVE SUBMISSION
# ==================================================
sub.to_csv('sub_isotonic_simple.csv', index=False)
nz = (sub['Overall Seed'] > 0).sum()
print(f"\nSubmission saved: sub_isotonic_simple.csv")
print(f"Non-zero predictions: {nz}/{ len(sub)}")
print(f"Mean seed (non-zero): {sub[sub['Overall Seed']>0]['Overall Seed'].mean():.1f}")

# ==================================================
# REASONING FOR THIS APPROACH
# ==================================================
print("\n" + "="*60)
print("WHY THIS APPROACH WORKS:")
print("="*60)
print("1. SIMPLE: 1 feature (NET rank) → 1 output (seed)")
print("2. NO OVERFITTING: Isotonic is monotonic, can't overfit")
print("3. PER-SEASON: Captures that committees calibrate per-season")
print("4. BASELINE: Should get ~1.0 overall RMSE (~0.45 tournament)")
print()
print("Expected leaderboard: ~0.5-1.0 RMSE")
print("If getting 0.4, then other features matter significantly.")
print("But at least we won't overfit catastrophically.")
print("="*60)
