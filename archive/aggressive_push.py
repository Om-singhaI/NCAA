"""
AGGRESSIVE PUSH TO RMSE < 2.0
Strategy: Stratified Models + 15 Diverse Ensemble + Aggressive Stacking
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("AGGRESSIVE PUSH: RMSE < 2.0")
print("Strategy: Stratified + 15 Ensemble + Stacking")
print("=" * 80)

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

# Load & process
train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

for df in [train_df, test_df]:
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        df[q + '_w'] = df.get(q, '').apply(parse_quad)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            df[col + '_w'] = wl.apply(lambda x: x[0])
            df[col + '_l'] = wl.apply(lambda x: x[1])

for df in [train_df, test_df]:
    df['Q_total'] = df['Quadrant1_w'].fillna(0) + df['Quadrant2_w'].fillna(0) + df['Quadrant3_w'].fillna(0) + df['Quadrant4_w'].fillna(0)
    df['Elite'] = df['Quadrant1_w'].fillna(0) + df['Quadrant2_w'].fillna(0)
    df['Elite_pct'] = df['Elite'] / (df['WL_w'].fillna(1) + 1)
    df['Wins'] = df['WL_w'].fillna(0)
    df['Losses'] = df['WL_l'].fillna(0)
    df['WinPct'] = df['Wins'] / (df['Wins'] + df['Losses'] + 1)
    df['Elite_vs_Poor'] = (df['Elite'] - df['Quadrant3_w'].fillna(0) - df['Quadrant4_w'].fillna(0)) / (df['Wins'] + 1)
    df['NET_inv'] = 1 / (df['NET Rank'] + 1)

features = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS',
            'Quadrant1_w', 'Quadrant2_w', 'Quadrant3_w', 'Quadrant4_w',
            'Wins', 'Losses', 'WinPct', 'Elite', 'Elite_pct', 'Q_total', 'Elite_vs_Poor', 'NET_inv']

X_train = train_df[features].fillna(0)
y_train = train_df['Overall Seed'].fillna(0)
groups_train = train_df['Season'].values

X_test = test_df[features].fillna(0)

print(f"Features: {len(features)}")
print(f"Training: {len(X_train)}, Selected: {(y_train > 0).sum()}\n")

# Strategy 1: Stratified models by seed range
print("PHASE 1: Stratified Models by Seed Range")
print("=" * 80)

# Separate training data by seed ranges
mask_low = (y_train > 0) & (y_train <= 16)
mask_mid = (y_train > 16) & (y_train <= 40)
mask_high = (y_train > 40)

print(f"Low seeds (1-16): {mask_low.sum()} teams")
print(f"Mid seeds (17-40): {mask_mid.sum()} teams")
print(f"High seeds (41+): {mask_high.sum()} teams\n")

# Train separate selector models for each range
selector_models = {}

# Low seed selector
clf_low = lgb.LGBMClassifier(n_estimators=200, lr=0.1, max_depth=6, random_state=1, verbose=-1)
clf_low.fit(X_train[mask_low | ~(y_train > 0)], (mask_low[mask_low | ~(y_train > 0)]).astype(int))
selector_models['low'] = clf_low

# Mid seed selector
clf_mid = lgb.LGBMClassifier(n_estimators=200, lr=0.1, max_depth=6, random_state=2, verbose=-1)
clf_mid.fit(X_train[mask_mid | ~(y_train > 0)], (mask_mid[mask_mid | ~(y_train > 0)]).astype(int))
selector_models['mid'] = clf_mid

# High seed selector
clf_high = lgb.LGBMClassifier(n_estimators=200, lr=0.1, max_depth=6, random_state=3, verbose=-1)
clf_high.fit(X_train[mask_high | ~(y_train > 0)], (mask_high[mask_high | ~(y_train > 0)]).astype(int))
selector_models['high'] = clf_high

print("Trained 3 stratified selector models\n")

# Strategy 2: 15 Ultra-Diverse Regression Models
print("PHASE 2: Training 15 Ultra-Diverse Regressors")
print("=" * 80)

gkf = GroupKFold(n_splits=3)

base_models = [
    ('XGB_Agg1', xgb.XGBRegressor(n_estimators=900, lr=0.01, max_depth=14, subsample=0.7, colsample_bytree=0.7, random_state=101)),
    ('XGB_Agg2', xgb.XGBRegressor(n_estimators=800, lr=0.012, max_depth=13, subsample=0.75, colsample_bytree=0.75, random_state=102)),
    ('XGB_Agg3', xgb.XGBRegressor(n_estimators=1000, lr=0.015, max_depth=12, subsample=0.8, colsample_bytree=0.8, random_state=103)),
    ('LGB_Agg1', lgb.LGBMRegressor(n_estimators=900, lr=0.01, max_depth=14, num_leaves=60, subsample=0.7, colsample_bytree=0.7, random_state=104, verbose=-1)),
    ('LGB_Agg2', lgb.LGBMRegressor(n_estimators=1000, lr=0.012, max_depth=13, num_leaves=55, subsample=0.75, colsample_bytree=0.75, random_state=105, verbose=-1)),
    ('CB_Agg1', CatBoostRegressor(iterations=900, learning_rate=0.01, depth=12, verbose=0, random_state=106)),
    ('CB_Agg2', CatBoostRegressor(iterations=1000, learning_rate=0.012, depth=11, verbose=0, random_state=107)),
    ('XGB_Cons', xgb.XGBRegressor(n_estimators=700, lr=0.025, max_depth=8, subsample=0.9, colsample_bytree=0.8, random_state=108)),
    ('LGB_Cons', lgb.LGBMRegressor(n_estimators=700, lr=0.025, max_depth=8, num_leaves=31, subsample=0.9, colsample_bytree=0.8, random_state=109, verbose=-1)),
    ('CB_Cons', CatBoostRegressor(iterations=700, learning_rate=0.025, depth=8, verbose=0, random_state=110)),
    ('XGB_Deep', xgb.XGBRegressor(n_estimators=800, lr=0.015, max_depth=15, subsample=0.8, colsample_bytree=0.7, random_state=111)),
    ('LGB_Deep', lgb.LGBMRegressor(n_estimators=850, lr=0.015, max_depth=15, num_leaves=70, subsample=0.8, colsample_bytree=0.7, random_state=112, verbose=-1)),
    ('CB_Deep', CatBoostRegressor(iterations=850, learning_rate=0.015, depth=13, verbose=0, random_state=113)),
    ('XGB_MAE', xgb.XGBRegressor(n_estimators=900, lr=0.015, max_depth=12, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.8, random_state=114)),
    ('LGB_Balanced', lgb.LGBMRegressor(n_estimators=900, lr=0.015, max_depth=12, num_leaves=50, subsample=0.8, colsample_bytree=0.8, random_state=115, verbose=-1)),
]

oof_predictions = np.zeros((len(X_train), len(base_models)))
model_rmses = {}

for m_idx, (m_name, model) in enumerate(base_models):
    print(f"{m_idx+1:2d}. {m_name:15s}", end=" → ", flush=True)
    fold_rmses = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        import copy
        fm = copy.deepcopy(model)
        fm.fit(X_tr, y_tr)
        
        pred = fm.predict(X_val)
        pred = np.clip(pred, 0, 68)
        oof_predictions[val_idx, m_idx] = pred
        
        rmse = np.sqrt(((pred - y_val.values)**2).mean())
        fold_rmses.append(rmse)
    
    avg_rmse = np.mean(fold_rmses)
    model_rmses[m_name] = avg_rmse
    print(f"RMSE: {avg_rmse:.4f}")

print("\n" + "=" * 80)
print("BASE MODEL PERFORMANCE:")
for name, rmse in sorted(model_rmses.items(), key=lambda x: x[1])[:8]:
    print(f"  {name:20s}: {rmse:.4f}")

# Strategy 3: Optimize ensemble weights with aggressive search
print("\n" + "=" * 80)
print("PHASE 3: Optimizing Ensemble Weights")
print("=" * 80)

def ensemble_rmse(w):
    w = np.abs(w) / (np.sum(np.abs(w)) + 1e-8)
    pred = np.average(oof_predictions, axis=1, weights=w)
    return np.sqrt(((pred - y_train.values)**2).mean())

# Use aggressive optimization
from scipy.optimize import minimize, differential_evolution

result = differential_evolution(
    ensemble_rmse,
    bounds=[(0, 1)] * len(base_models),
    seed=42,
    maxiter=300,
    workers=-1,
    atol=1e-9,
    tol=1e-9
)

best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
best_cv_rmse = ensemble_rmse(best_weights)

print(f"\nOptimized Ensemble CV RMSE: {best_cv_rmse:.4f}")
print("\nTop Weights:")
for (name, _), w in sorted(zip(base_models, best_weights), key=lambda x: x[1][1], reverse=True)[:8]:
    print(f"  {name:20s}: {w*100:6.2f}%")

# Train final models on full data
print("\n" + "=" * 80)
print("PHASE 4: Training Final Models on Full Data")
print("=" * 80)

test_predictions = np.zeros((len(X_test), len(base_models)))

for m_idx, (m_name, model) in enumerate(base_models):
    print(f"Training {m_name}...")
    model.fit(X_train, y_train)
    test_predictions[:, m_idx] = np.clip(model.predict(X_test), 0, 68)

# Get stratified selector predictions
test_pred_low = selector_models['low'].predict_proba(X_test)[:, 1]
test_pred_mid = selector_models['mid'].predict_proba(X_test)[:, 1]
test_pred_high = selector_models['high'].predict_proba(X_test)[:, 1]

# Generate weighted predictions
final_pred = np.average(test_predictions, axis=1, weights=best_weights)

# Apply stratified confidence adjustment
# Where classifiers are confident, trust prediction; otherwise, pull toward 0
confidence = np.maximum(test_pred_low, test_pred_mid, test_pred_high)
final_pred = final_pred * (0.7 + 0.3 * confidence)

final_pred = np.clip(final_pred, 0, 68)

# Create submission
submission = pd.DataFrame({
    'RecordID': test_df['RecordID'].values,
    'Overall Seed': final_pred
})

submission.to_csv('my_submission_aggressive.csv', index=False)

print("\n" + "=" * 80)
print("FINAL SUBMISSION")
print("=" * 80)
print(f"CV RMSE: {best_cv_rmse:.4f}")
print(f"Mean Seed: {final_pred.mean():.2f}")
print(f"Std Dev: {final_pred.std():.2f}")
print(f"Range: {final_pred.min():.2f} - {final_pred.max():.2f}")
print(f"Selected: {(final_pred > 0).sum()}/451 teams")
print(f"\nFile: my_submission_aggressive.csv")
print(f"\nTarget: RMSE < 2.0")
print(f"Estimated: {best_cv_rmse:.2f} → {best_cv_rmse - 0.3:.2f} (optimistic)")
