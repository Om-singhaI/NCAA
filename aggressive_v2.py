"""
AGGRESSIVE ENSEMBLE v2: 8 models + Bayesian Optimization
Faster alternative while super_ensemble runs
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy.optimize import differential_evolution
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

def load_and_process(df):
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
    
    df['Wins'] = df['WL_wins'].fillna(0)
    df['Losses'] = df['WL_losses'].fillna(0)
    df['Win_Pct'] = df['Wins'] / (df['Wins'] + df['Losses'] + 1)
    
    df['Q_Total'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0) + df['Quadrant3_wins'].fillna(0) + df['Quadrant4_wins'].fillna(0)
    df['Elite_Wins'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0)
    df['Elite_Pct'] = df['Elite_Wins'] / (df['Wins'] + 1)
    
    features = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'Wins', 'Losses', 'Win_Pct', 'Elite_Wins', 'Elite_Pct', 'Q_Total',
        'PrevNET', 'AvgOppNET', 'Conf.Record_wins', 'RoadWL_wins'
    ]
    
    features_full = features + [
        'NET Rank',  # Polynomial
    ]
    
    for f in features:
        if f not in df.columns:
            features_full.remove(f)
    
    # Remove duplicates and keep only valid
    features_full = list(set(features_full))
    features_final = [f for f in features_full if f in df.columns]
    
    return df, features_final

print("=" * 80)
print("AGGRESSIVE ENSEMBLE v2: Quick-Train with 8 Optimized Models")
print("=" * 80)

train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

train_proc, feature_list = load_and_process(train_df)
test_proc, _ = load_and_process(test_df)

X_train = train_proc[feature_list].fillna(0)
y_train = train_proc['Overall Seed'].fillna(0)
groups_train = train_proc['Season'].values

X_test = test_proc[feature_list].fillna(0)

print(f"Features: {len(feature_list)}")
print(f"Training: {len(X_train)} samples, {(y_train > 0).sum()} selected")

# Create models
models_config = [
    ('XGB_Turbo1', xgb.XGBRegressor(n_estimators=600, lr=0.02, max_depth=9, random_state=1)),
    ('XGB_Turbo2', xgb.XGBRegressor(n_estimators=700, lr=0.018, max_depth=10, random_state=2)),
    ('LGB_Turbo1', lgb.LGBMRegressor(n_estimators=650, learning_rate=0.02, max_depth=9, random_state=3)),
    ('LGB_Turbo2', lgb.LGBMRegressor(n_estimators=700, learning_rate=0.018, max_depth=10, random_state=4)),
    ('CB_Turbo', CatBoostRegressor(iterations=700, learning_rate=0.02, depth=9, verbose=0, random_state=5)),
    ('XGB_Balanced', xgb.XGBRegressor(n_estimators=650, lr=0.025, max_depth=8, random_state=6)),
    ('LGB_Balanced', lgb.LGBMRegressor(n_estimators=650, learning_rate=0.025, max_depth=8, random_state=7)),
    ('CB_Balanced', CatBoostRegressor(iterations=650, learning_rate=0.025, depth=8, verbose=0, random_state=8)),
]

# Quick 3-fold CV for OOF
gkf = GroupKFold(n_splits=3)
oof_preds = np.zeros((len(X_train), len(models_config)))

print(f"\nTraining {len(models_config)} models with 3-fold CV...")
for m_idx, (m_name, model) in enumerate(models_config):
    print(f"\n{m_idx+1}. {m_name}...")
    fold_rmses = []
    
    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        import copy
        fm = copy.deepcopy(model)
        fm.fit(X_tr, y_tr)
        
        pred = fm.predict(X_val)
        pred = np.clip(pred, 0, 68)
        oof_preds[val_idx, m_idx] = pred
        
        rmse = np.sqrt(((pred - y_val.values)**2).mean())
        fold_rmses.append(rmse)
    
    avg_rmse = np.mean(fold_rmses)
    print(f"   Fold RMSE: {fold_rmses} → Avg: {avg_rmse:.4f}")

# Optimize weights
print("\n" + "=" * 80)
print("OPTIMIZING WEIGHTS WITH DIFFERENTIAL EVOLUTION...")
print("=" * 80)

def ensemble_loss(w, X, y):
    w = np.abs(w) / np.sum(np.abs(w))
    pred = np.average(X, axis=1, weights=w)
    return np.sqrt(((pred - y.values)**2).mean())

result = differential_evolution(
    ensemble_loss,
    bounds=[(0, 1)] * len(models_config),
    args=(oof_preds, y_train),
    seed=42,
    maxiter=200,
    atol=1e-6,
    tol=1e-6,
    workers=-1
)

best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
best_cv_rmse = ensemble_loss(best_weights, oof_preds, y_train)

print(f"\nOptimal CV RMSE: {best_cv_rmse:.4f}")
print(f"\nOptimal Weights:")
for (m_name, _), w in zip(models_config, best_weights):
    if w > 0.01:
        print(f"  {m_name:20s}: {w*100:6.2f}%")

# Train final on full data
print("\n" + "=" * 80)
print("TRAINING FINAL MODELS...")
print("=" * 80)

test_preds = np.zeros((len(X_test), len(models_config)))
for m_idx, (m_name, model) in enumerate(models_config):
    print(f"Training {m_name}...")
    model.fit(X_train, y_train)
    test_preds[:, m_idx] = np.clip(model.predict(X_test), 0, 68)

final_pred = np.average(test_preds, axis=1, weights=best_weights)
final_pred = np.clip(final_pred, 0, 68)

# Create submission
submission = pd.DataFrame({
    'RecordID': test_proc['RecordID'].values,
    'Overall Seed': final_pred
})
submission.to_csv('my_submission_v2.csv', index=False)

print("\n" + "=" * 80)
print("SUBMISSION COMPLETE")
print("=" * 80)
print(f"Mean Seed: {final_pred.mean():.2f}")
print(f"Selected:  {(final_pred > 0).sum()} teams")
print(f"CV RMSE:   {best_cv_rmse:.4f}")
print(f"File:      my_submission_v2.csv")
print(f"\nEstimated Kaggle: {best_cv_rmse - 0.25:.4f} to {best_cv_rmse + 0.1:.4f}")
