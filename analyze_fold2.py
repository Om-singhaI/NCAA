import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANALYZING FOLD 2: Why did it achieve 2.17 RMSE?")
print("=" * 80)

# Load data
train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

print(f"\nData loaded: {len(train_df)} training samples, {len(test_df)} test samples")

# Feature engineering
def load_and_process_v3(df):
    df = df.copy()
    
    # Basic features
    features = [
        'NET_Rank', 'Quadrant_1_Wins', 'Quadrant_2_Wins', 'Quadrant_3_Wins', 'Quadrant_4_Wins',
        'Wins_Against_Top_25', 'Wins_Against_Top_50', 'Wins_Against_Ranked',
        'Strength_of_Record', 'Wins', 'Losses', 'Win_Pct'
    ]
    
    # Ratios and composites
    df['Q123_Wins'] = df['Quadrant_1_Wins'] + df['Quadrant_2_Wins'] + df['Quadrant_3_Wins']
    df['Q_Avg'] = df[['Quadrant_1_Wins', 'Quadrant_2_Wins', 'Quadrant_3_Wins', 'Quadrant_4_Wins']].mean(axis=1)
    df['Quad_Imbalance'] = df[['Quadrant_1_Wins', 'Quadrant_2_Wins', 'Quadrant_3_Wins', 'Quadrant_4_Wins']].std(axis=1)
    df['Top_Win_Ratio'] = df['Wins_Against_Top_25'] / (df['Wins'] + 1)
    df['Ranked_Win_Ratio'] = df['Wins_Against_Ranked'] / (df['Wins'] + 1)
    df['High_Quality_Win_Pct'] = (df['Quadrant_1_Wins'] + df['Quadrant_2_Wins']) / (df['Wins'] + 1)
    df['Strong_Record_Indicator'] = (df['Strength_of_Record'] > 50).astype(int)
    
    # Polynomial and log transforms
    df['NET_sq'] = df['NET_Rank'] ** 2
    df['NET_inv'] = 1 / (df['NET_Rank'] + 1)
    df['log_Wins'] = np.log1p(df['Wins'])
    df['log_Losses'] = np.log1p(df['Losses'])
    
    # Interaction terms
    df['Wins_x_Ranked'] = df['Wins'] * df['Wins_Against_Ranked']
    df['NET_x_SOR'] = df['NET_Rank'] * df['Strength_of_Record']
    df['WinPct_x_Top25'] = df['Win_Pct'] * df['Wins_Against_Top_25']
    df['Quad1_Density'] = df['Quadrant_1_Wins'] / (df['Wins'] + 1)
    df['Consistency'] = 1 - (df['Quad_Imbalance'] / (df['Q_Avg'] + 1))
    
    # Efficiency metrics
    df['Points_Per_Game_Est'] = df['Wins'] * 1.5
    df['Offensive_Strength'] = df['Quadrant_1_Wins'] + df['Quadrant_2_Wins']
    df['Defensive_Strength'] = df['Wins_Against_Ranked']
    
    # Advanced ratios
    df['Elite_Win_Ratio'] = (df['Quadrant_1_Wins'] + df['Quadrant_2_Wins']) / (df['Wins'] + 1)
    df['Close_Game_Indicator'] = (df['Wins'] - df['Losses']) / (df['Wins'] + df['Losses'] + 1)
    
    feature_list = features + [
        'Q123_Wins', 'Q_Avg', 'Quad_Imbalance', 'Top_Win_Ratio', 'Ranked_Win_Ratio',
        'High_Quality_Win_Pct', 'Strong_Record_Indicator', 'NET_sq', 'NET_inv',
        'log_Wins', 'log_Losses', 'Wins_x_Ranked', 'NET_x_SOR', 'WinPct_x_Top25',
        'Quad1_Density', 'Consistency', 'Points_Per_Game_Est', 'Offensive_Strength',
        'Defensive_Strength', 'Elite_Win_Ratio', 'Close_Game_Indicator'
    ]
    
    return df, feature_list

train_processed, feature_list = load_and_process_v3(train_df)
X_train = train_processed[feature_list].fillna(0)
y_train = train_processed['Overall_Seed'].fillna(0)
groups_train = train_processed['Season'].values

print(f"\nFeatures used: {len(feature_list)}")
print(f"Non-selected (seed=0): {(y_train == 0).sum()}")
print(f"Selected (seed>0): {(y_train > 0).sum()}")

# Replicate fold structure
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train, y_train, groups=groups_train))

print("\n" + "=" * 80)
print("FOLD STRUCTURE ANALYSIS")
print("=" * 80)

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    seasons_train = groups_train[train_idx]
    seasons_val = groups_train[val_idx]
    
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train: {len(train_idx)} samples (seasons: {sorted(np.unique(seasons_train))})")
    print(f"  Val:   {len(val_idx)} samples (seasons: {sorted(np.unique(seasons_val))})")
    print(f"  Val Selected: {(y_val > 0).sum()} / {len(y_val)} ({100*(y_val > 0).sum()/len(y_val):.1f}%)")
    print(f"  Val Seed Range: {y_val[y_val > 0].min():.1f} - {y_val[y_val > 0].max():.1f}")
    print(f"  Val Seed Mean: {y_val[y_val > 0].mean():.2f}, Std: {y_val[y_val > 0].std():.2f}")
    
    if fold_idx == 1:  # Fold 2 (0-indexed)
        print("\n  >>> FOLD 2 SPECIFIC ANALYSIS <<<")
        print(f"  Validation seasons: {sorted(np.unique(seasons_val))}")
        print(f"  This fold's test set likely contains similar season patterns")
        
        # Save fold 2 indices for detailed analysis
        fold2_val_idx = val_idx
        fold2_seasons = seasons_val
        fold2_y_val = y_val

# Train a quick model on Fold 2 to see feature importance
print("\n" + "=" * 80)
print("FOLD 2 MODEL - Feature Importance")
print("=" * 80)

train_idx_f2, val_idx_f2 = folds[1]
X_tr_f2, X_val_f2 = X_train.iloc[train_idx_f2], X_train.iloc[val_idx_f2]
y_tr_f2, y_val_f2 = y_train.iloc[train_idx_f2], y_train.iloc[val_idx_f2]

# Train LGB (since it had best weight in final_push)
lgb_f2 = lgb.LGBMRegressor(
    n_estimators=600, learning_rate=0.025, max_depth=8,
    random_state=42, verbose=-1, num_leaves=31
)
lgb_f2.fit(X_tr_f2, y_tr_f2)

# Feature importance
importance_f2 = pd.DataFrame({
    'feature': feature_list,
    'importance': lgb_f2.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 features for Fold 2:")
for idx, row in importance_f2.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:8.4f}")

# Predictions and errors on Fold 2
y_pred_f2 = lgb_f2.predict(X_val_f2)
y_pred_f2 = np.clip(y_pred_f2, 0, 68)
errors_f2 = (y_pred_f2 - y_val_f2) ** 2

rmse_f2 = np.sqrt(errors_f2.mean())
print(f"\nFold 2 LGB Model RMSE: {rmse_f2:.4f}")

# Find which samples had highest error
high_error_idx = np.argsort(errors_f2)[-10:]
print("\nTop 10 highest-error predictions in Fold 2:")
for rank, idx in enumerate(high_error_idx[::-1], 1):
    actual_seed = y_val_f2.iloc[idx]
    pred_seed = y_pred_f2[idx]
    error = errors_f2[idx]
    print(f"  {rank}. Actual: {actual_seed:.0f}, Pred: {pred_seed:.1f}, Error²: {error:.4f}")

print("\n" + "=" * 80)
print("KEY INSIGHTS FOR IMPROVEMENT")
print("=" * 80)
print("""
1. Fold 2's success (2.17 RMSE) suggests certain season patterns are easier
2. Conservative models (LGB, CatBoost) perform better than aggressive XGBoost
3. Feature importance in Fold 2 shows which predictors matter most
4. High-error samples need special attention - possible outliers

NEXT STEPS:
- Create 10+ base models with different seeds/configs
- Use seed-stratified CV to ensure balanced seed distributions
- Add conference/bracket-aware features
- Apply selective confidence calibration on high-uncertainty predictions
- Use Fold 2 insights for post-processing adjustments
""")
