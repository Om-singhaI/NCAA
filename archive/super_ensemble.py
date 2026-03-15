import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("SUPER-AGGRESSIVE ENSEMBLE: 12+ Models + Advanced Features")
print("=" * 80)

# Load data
train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

print(f"\nData loaded: {len(train_df)} training, {len(test_df)} test samples")

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

# Feature engineering with correct column names
def load_and_process_advanced(df):
    df = df.copy()
    
    # Parse quadrant wins
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        df[q + '_wins'] = df.get(q, pd.Series()).apply(parse_quad)
    
    # Parse W-L records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wins_losses = df[col].apply(parse_wl)
            df[col + '_wins'] = wins_losses.apply(lambda x: x[0])
            df[col + '_losses'] = wins_losses.apply(lambda x: x[1])
    
    # Basic computed features
    df['Wins'] = df['WL_wins'].fillna(0)
    df['Losses'] = df['WL_losses'].fillna(0)
    df['Win_Pct'] = df['Wins'] / (df['Wins'] + df['Losses'] + 1)
    
    features = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'Wins', 'Losses', 'Win_Pct'
    ]
    
    # Advanced composite features
    df['Total_Quad_Wins'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0) + df['Quadrant3_wins'].fillna(0) + df['Quadrant4_wins'].fillna(0)
    df['Elite_Wins'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0)
    df['Elite_Win_Pct'] = df['Elite_Wins'] / (df['Wins'] + 1)
    df['Quad_Avg'] = df[['Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins']].fillna(0).mean(axis=1)
    df['Quad_Std'] = df[['Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins']].fillna(0).std(axis=1)
    df['Quad_Consistency'] = 1 - (df['Quad_Std'] / (df['Quad_Avg'] + 1))
    df['Q1_Density'] = df['Quadrant1_wins'].fillna(0) / (df['Wins'] + 1)
    df['Q1_Q2_Wins'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0)
    df['Q3_Q4_Wins'] = df['Quadrant3_wins'].fillna(0) + df['Quadrant4_wins'].fillna(0)
    df['Elite_vs_Poor'] = (df['Q1_Q2_Wins'] - df['Q3_Q4_Wins']) / (df['Wins'] + 1)
    df['Loss_Distribution'] = df['Losses'] / (df['Wins'] + df['Losses'] + 1)
    
    # Polynomial and log transforms
    df['NET_sq'] = df['NET Rank'] ** 2
    df['NET_inv'] = 1 / (df['NET Rank'] + 1)
    df['NET_cube'] = df['NET Rank'] ** 3
    df['log_NET'] = np.log1p(df['NET Rank'].fillna(100))
    df['log_Wins'] = np.log1p(df['Wins'])
    df['log_Losses'] = np.log1p(df['Losses'])
    df['Wins_Losses_Ratio'] = (df['Wins'] - df['Losses']) / (df['Wins'] + df['Losses'] + 1)
    
    # Interaction terms
    df['NET_x_Elite'] = df['NET Rank'] * df['Elite_Win_Pct']
    df['NET_x_WinPct'] = df['NET Rank'] * df['Win_Pct']
    df['Elite_x_Consistency'] = df['Elite_Win_Pct'] * df['Quad_Consistency']
    df['SOR_x_WinPct'] = df['NETSOS'] * df['Win_Pct']
    df['Q1_x_NET'] = df['Quadrant1_wins'].fillna(0) * df['NET Rank']
    
    # Efficiency-like metrics
    df['WIN_Efficiency'] = (df['Wins'] * df['Win_Pct']) / (df['NET Rank'] + 1)
    df['Schedule_Difficulty'] = (df['AvgOppNETRank'] + df['NETSOS']) / 2
    df['Resume_Balance'] = df['Elite_Win_Pct'] + df['Win_Pct']
    df['Non_Conf_Success'] = df['NETNonConfSOS'] / (df['NETSOS'] + 1)
    
    # Category indicators
    df['Top_25_NET'] = (df['NET Rank'] <= 25).astype(int)
    df['Top_50_NET'] = (df['NET Rank'] <= 50).astype(int)
    df['Top_100_NET'] = (df['NET Rank'] <= 100).astype(int)
    df['High_Elite_Wins'] = (df['Elite_Wins'] >= 8).astype(int)
    df['Strong_Record'] = (df['NETSOS'] > 50).astype(int)
    
    feature_list = features + [
        'Total_Quad_Wins', 'Elite_Wins', 'Elite_Win_Pct', 'Quad_Avg', 'Quad_Std',
        'Quad_Consistency', 'Q1_Density', 'Q1_Q2_Wins', 'Q3_Q4_Wins', 'Elite_vs_Poor',
        'Loss_Distribution', 'NET_sq', 'NET_inv', 'NET_cube', 'log_NET', 'log_Wins',
        'log_Losses', 'Wins_Losses_Ratio', 'NET_x_Elite', 'NET_x_WinPct',
        'Elite_x_Consistency', 'SOR_x_WinPct', 'Q1_x_NET', 'WIN_Efficiency',
        'Schedule_Difficulty', 'Resume_Balance', 'Non_Conf_Success',
        'Top_25_NET', 'Top_50_NET', 'Top_100_NET', 'High_Elite_Wins', 'Strong_Record'
    ]
    
    return df, feature_list

train_processed, feature_list = load_and_process_advanced(train_df)
test_processed, _ = load_and_process_advanced(test_df)

X_train = train_processed[feature_list].fillna(0)
y_train = train_processed['Overall_Seed'].fillna(0)
groups_train = train_processed['Season'].values

X_test = test_processed[feature_list].fillna(0)

print(f"Features: {len(feature_list)}")
print(f"Selected teams: {(y_train > 0).sum()} / {len(y_train)}")

# Create 12+ diverse base models
def create_diverse_models():
    models = {}
    
    # XGBoost variants (5 configs)
    models['XGB_Aggressive_v1'] = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.01, max_depth=12,
        subsample=0.8, colsample_bytree=0.7, random_state=42
    )
    
    models['XGB_Aggressive_v2'] = xgb.XGBRegressor(
        n_estimators=900, learning_rate=0.012, max_depth=11,
        subsample=0.75, colsample_bytree=0.75, random_state=43
    )
    
    models['XGB_Conservative'] = xgb.XGBRegressor(
        n_estimators=700, learning_rate=0.025, max_depth=8,
        subsample=0.9, colsample_bytree=0.8, random_state=44
    )
    
    models['XGB_Deep'] = xgb.XGBRegressor(
        n_estimators=750, learning_rate=0.015, max_depth=13,
        subsample=0.8, colsample_bytree=0.7, random_state=45
    )
    
    models['XGB_Shallow'] = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6,
        subsample=0.85, colsample_bytree=0.85, random_state=46
    )
    
    # LightGBM variants (5 configs)
    models['LGB_Aggressive'] = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.012, max_depth=11,
        num_leaves=40, subsample=0.7, colsample_bytree=0.7, random_state=47
    )
    
    models['LGB_Conservative'] = lgb.LGBMRegressor(
        n_estimators=700, learning_rate=0.025, max_depth=8,
        num_leaves=31, subsample=0.9, colsample_bytree=0.8, random_state=48
    )
    
    models['LGB_Deep'] = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.015, max_depth=12,
        num_leaves=50, subsample=0.8, colsample_bytree=0.7, random_state=49
    )
    
    models['LGB_Balanced'] = lgb.LGBMRegressor(
        n_estimators=750, learning_rate=0.02, max_depth=9,
        num_leaves=35, subsample=0.8, colsample_bytree=0.8, random_state=50
    )
    
    models['LGB_Shallow'] = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6,
        num_leaves=20, subsample=0.85, colsample_bytree=0.85, random_state=51
    )
    
    # CatBoost variants (2+ configs)
    models['CB_Balanced'] = CatBoostRegressor(
        iterations=800, learning_rate=0.015, depth=10,
        random_state=52, verbose=0
    )
    
    models['CB_Conservative'] = CatBoostRegressor(
        iterations=700, learning_rate=0.025, depth=8,
        random_state=53, verbose=0
    )
    
    return models

print("\nTraining 12 diverse base models...")
models = create_diverse_models()

# Train on full training set and get OOF predictions
gkf = GroupKFold(n_splits=5)
oof_predictions = np.zeros((len(X_train), len(models)))
model_rmses = {}

for model_idx, (model_name, model) in enumerate(models.items()):
    print(f"\n{model_idx + 1}. Training {model_name}...")
    
    fold_predictions = np.zeros(len(X_train))
    fold_rmses = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Clone model for this fold
        import copy
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_tr, y_tr)
        
        # Predict on validation set
        val_preds = fold_model.predict(X_val)
        val_preds = np.clip(val_preds, 0, 68)
        fold_predictions[val_idx] = val_preds
        
        # Calculate RMSE
        fold_rmse = np.sqrt(((val_preds - y_val.values) ** 2).mean())
        fold_rmses.append(fold_rmse)
    
    oof_predictions[:, model_idx] = fold_predictions
    avg_rmse = np.mean(fold_rmses)
    model_rmses[model_name] = avg_rmse
    print(f"   → Fold RMSEs: {[f'{r:.4f}' for r in fold_rmses]}")
    print(f"   → Average RMSE: {avg_rmse:.4f}")

print("\n" + "=" * 80)
print("BASE MODEL PERFORMANCE")
print("=" * 80)
for model_name, rmse in sorted(model_rmses.items(), key=lambda x: x[1]):
    print(f"  {model_name:30s}: {rmse:.4f}")

# Optimize ensemble weights using differential evolution (more robust than Nelder-Mead)
print("\n" + "=" * 80)
print("OPTIMIZING ENSEMBLE WEIGHTS...")
print("=" * 80)

def ensemble_rmse(weights, X, y):
    weights = np.abs(weights) / np.sum(np.abs(weights))
    pred = np.average(X, axis=1, weights=weights)
    return np.sqrt(((pred - y.values) ** 2).mean())

# Use differential evolution for global optimization
bounds = [(0, 1)] * len(models)
result = differential_evolution(
    ensemble_rmse, 
    bounds, 
    args=(oof_predictions, y_train),
    seed=42,
    maxiter=500,
    workers=1,
    atol=1e-8,
    tol=1e-8
)

optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))
ensemble_rmse_cv = ensemble_rmse(optimal_weights, oof_predictions, y_train)

print(f"\nOptimal Ensemble RMSE: {ensemble_rmse_cv:.4f}")
print(f"\nOptimal Weights:")
for model_name, weight in zip(models.keys(), optimal_weights):
    if weight > 0.01:
        print(f"  {model_name:30s}: {weight*100:6.2f}%")

# Train final models on full data
print("\n" + "=" * 80)
print("TRAINING FINAL MODELS ON FULL DATA...")
print("=" * 80)

final_test_predictions = np.zeros((len(X_test), len(models)))

for model_idx, (model_name, model) in enumerate(models.items()):
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    final_test_predictions[:, model_idx] = np.clip(test_pred, 0, 68)

# Generate final ensemble predictions
final_ensemble_pred = np.average(final_test_predictions, axis=1, weights=optimal_weights)
final_ensemble_pred = np.clip(final_ensemble_pred, 0, 68)

# Create submission
submission = pd.DataFrame({
    'RecordID': test_processed['RecordID'].values,
    'Overall Seed': final_ensemble_pred
})

submission.to_csv('my_submission_super.csv', index=False)

print("\n" + "=" * 80)
print("SUBMISSION SUMMARY")
print("=" * 80)
print(f"Mean Seed: {final_ensemble_pred.mean():.2f}")
print(f"Std Dev:   {final_ensemble_pred.std():.2f}")
print(f"Min/Max:   {final_ensemble_pred.min():.2f} / {final_ensemble_pred.max():.2f}")
print(f"Selected:  {(final_ensemble_pred > 0).sum()} teams")
print(f"\nSaved to: my_submission_super.csv")
print(f"\nEstimated Performance:")
print(f"  CV RMSE: {ensemble_rmse_cv:.4f}")
print(f"  Kaggle estimate: {ensemble_rmse_cv - 0.3:.4f} (optimistic)")
print(f"  Kaggle estimate: {ensemble_rmse_cv + 0.1:.4f} (conservative)")
