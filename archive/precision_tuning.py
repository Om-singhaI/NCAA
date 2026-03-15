"""
PRECISION TUNING: Ultra-refined final_push.py variant
Focus: Maximum precision on core models without excessive complexity
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.search(r"(\d+)[^\d]+(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r"(\d+)", str(s))
    if m2:
        return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)

def parse_quad(s):
    if pd.isna(s) or str(s).strip() == '':
        return np.nan
    m = re.search(r"(\d+)[^\d]+(\d+)", str(s))
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", str(s))
    if m2:
        return int(m2.group(1))
    return np.nan

print("=" * 80)
print("PRECISION TUNING: Refined Ensemble (No Warnings)")
print("=" * 80)

train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

# Clean processing
for df in [train_df, test_df]:
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        df[q + '_w'] = df.get(q, '').apply(parse_quad)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            df[col + '_w'] = wl.apply(lambda x: x[0])
            df[col + '_l'] = wl.apply(lambda x: x[1])

# Create features from known good set
for df in [train_df, test_df]:
    df['Q_total'] = df['Quadrant1_w'].fillna(0) + df['Quadrant2_w'].fillna(0) + df['Quadrant3_w'].fillna(0) + df['Quadrant4_w'].fillna(0)
    df['Elite'] = df['Quadrant1_w'].fillna(0) + df['Quadrant2_w'].fillna(0)
    df['Elite_pct'] = df['Elite'] / (df['WL_w'].fillna(1) + 1)
    df['Wins'] = df['WL_w'].fillna(0)
    df['Losses'] = df['WL_l'].fillna(0)
    df['WinPct'] = df['Wins'] / (df['Wins'] + df['Losses'] + 1)

features = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS',
            'Quadrant1_w', 'Quadrant2_w', 'Quadrant3_w', 'Quadrant4_w',
            'Wins', 'Losses', 'WinPct', 'Elite', 'Elite_pct', 'Q_total']

X_train = train_df[features].fillna(0)
y_train = train_df['Overall Seed'].fillna(0)
groups_train = train_df['Season'].values

X_test = test_df[features].fillna(0)

print(f"Features: {len(features)}")
print(f"Training: {len(X_train)} samples, {(y_train > 0).sum()} selected\n")

# Handpicked best models from all previous runs
print("Training 5 refined models (best from all previous experiments)...\n")

models = [
    ('LGB_Conservative_Best', lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.025, max_depth=8,
        num_leaves=31, subsample=0.9, colsample_bytree=0.8, 
        random_state=100, verbose=-1
    )),
    ('CB_Balanced_Best', CatBoostRegressor(
        iterations=800, learning_rate=0.015, depth=10,
        verbose=0, random_state=101
    )),
    ('LGB_Deep_Best', lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.015, max_depth=12,
        num_leaves=50, subsample=0.8, colsample_bytree=0.7,
        random_state=102, verbose=-1
    )),
    ('XGB_Conservative_Best', xgb.XGBRegressor(
        n_estimators=700, learning_rate=0.025, max_depth=8,
        subsample=0.9, colsample_bytree=0.8, random_state=103
    )),
    ('CB_Conservative_Best', CatBoostRegressor(
        iterations=700, learning_rate=0.025, depth=8,
        verbose=0, random_state=104
    )),
]

# Simple blending approach: train on full data
test_predictions = []
model_names = []

for name, model in models:
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred = np.clip(pred, 0, 68)
    test_predictions.append(pred)
    model_names.append(name)
    print(f"  → Mean: {pred.mean():.2f}, Selected: {(pred > 0).sum()}")

# Weighted average with optimal weights from previous run
# From final_push.py: LGB_Conservative had 27.87%, CB_Balanced 26.43%, LGB_Deep 21%
# Adapting to our 5-model setup
weights = np.array([0.25, 0.26, 0.21, 0.14, 0.14])  # Normalized
weights = weights / weights.sum()

final_pred = np.average(test_predictions, axis=0, weights=weights)
final_pred = np.clip(final_pred, 0, 68)

# Verify quality
submission = pd.DataFrame({
    'RecordID': test_df['RecordID'].values,
    'Overall Seed': final_pred
})

submission.to_csv('my_submission_tuned.csv', index=False)

print("\n" + "=" * 80)
print("PRECISION ENSEMBLE COMPLETE")
print("=" * 80)
print(f"Ensemble Weights: {dict(zip(model_names, [f'{w*100:.1f}%' for w in weights]))}")
print(f"\nSubmission Stats:")
print(f"  Mean Seed:    {final_pred.mean():.2f}")
print(f"  Std Dev:      {final_pred.std():.2f}")
print(f"  Range:        {final_pred.min():.2f} - {final_pred.max():.2f}")
print(f"  Selected:     {(final_pred > 0).sum()}/451 teams ({100*(final_pred > 0).sum()/451:.1f}%)")
print(f"  File:         my_submission_tuned.csv")
print(f"\nNext: Compare this against my_submission.csv on Kaggle")
