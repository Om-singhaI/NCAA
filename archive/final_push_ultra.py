"""
FINAL_PUSH_ULTRA: Same proven structure as final_push.py
But with AGGRESSIVE hyperparameters to push toward RMSE < 2.0
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
print("FINAL_PUSH_ULTRA: Proven structure + Aggressive hyperparameters")
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

def load_and_process(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce')
    df_train['Overall Seed'] = df_train['Overall Seed'].fillna(0)

    numeric_cols = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
                    'NETSOS', 'NETNonConfSOS']
    for col in numeric_cols:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        df_train[q + '_wins'] = df_train.get(q, pd.Series()).apply(parse_quad)
        df_test[q + '_wins'] = df_test.get(q, pd.Series()).apply(parse_quad)

    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        for df in (df_train, df_test):
            if col in df.columns:
                wins_losses = df[col].apply(parse_wl)
                df[col + '_wins'] = wins_losses.apply(lambda x: x[0])
                df[col + '_losses'] = wins_losses.apply(lambda x: x[1])

    for df in (df_train, df_test):
        df['WL_ratio'] = df['WL_wins'] / (df['WL_losses'] + 1)
        df['Conf_ratio'] = df['Conf.Record_wins'] / (df['Conf.Record_losses'] + 1)
        df['Road_ratio'] = df['RoadWL_wins'] / (df['RoadWL_losses'] + 1)
        
        df['total_wins'] = df['WL_wins'].fillna(0) + df['Conf.Record_wins'].fillna(0)
        df['total_losses'] = df['WL_losses'].fillna(0) + df['Conf.Record_losses'].fillna(0)
        df['win_rate'] = df['total_wins'] / (df['total_wins'] + df['total_losses'] + 1)
        
        df['quad_wins_total'] = (df['Quadrant1_wins'].fillna(0) +
                                  df['Quadrant2_wins'].fillna(0) +
                                  df['Quadrant3_wins'].fillna(0) +
                                  df['Quadrant4_wins'].fillna(0))
        df['quad1_pct'] = df['Quadrant1_wins'].fillna(0) / (df['quad_wins_total'] + 1)
        df['quad_high_wins'] = (df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0))
        
        df['NET_valid'] = (~df['NET Rank'].isna()).astype(int)
        df['has_quad_wins'] = (df['quad_wins_total'] > 0).astype(int)
        
        df['NET_x_wins'] = df['NET Rank'] * df['total_wins']
        df['NET_x_winrate'] = df['NET Rank'] * df['win_rate']
        df['NET_squared'] = df['NET Rank'] ** 2
        df['NET_inv'] = 1 / (df['NET Rank'] + 1)
        df['quad_balance'] = df['quad_wins_total'] / (df['total_wins'] + 1)
        df['strength_composite'] = df['Strength_of_Record'] * df['win_rate'] if 'Strength_of_Record' in df.columns else 0

    feature_list = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'WL_ratio', 'Conf_ratio', 'Road_ratio',
        'total_wins', 'total_losses', 'win_rate',
        'quad_wins_total', 'quad1_pct', 'quad_high_wins',
        'NET_valid', 'has_quad_wins',
        'NET_x_wins', 'NET_x_winrate', 'NET_squared', 'NET_inv',
        'quad_balance'
    ]
    
    return df_train, df_test, feature_list

train_df, test_df, feature_list = load_and_process('NCAA_Seed_Training_Set2.0.csv', 'NCAA_Seed_Test_Set2.0.csv')

X_train = train_df[feature_list].fillna(0)
y_train = train_df['Overall Seed'].fillna(0)
groups_train = train_df['Season'].values

X_test = test_df[feature_list].fillna(0)

print(f"Training: {len(X_train)} samples, Features: {len(feature_list)}\n")

# 8 AGGRESSIVE MODELS - Pushed to maximum
print("Training 8 ULTRA-AGGRESSIVE models with 5-fold CV...\n")

models = {
    'XGB_Agg': xgb.XGBRegressor(
        n_estimators=1200, learning_rate=0.008, max_depth=16,
        subsample=0.65, colsample_bytree=0.65, random_state=42, gamma=0.1
    ),
    'LGB_Agg': lgb.LGBMRegressor(
        n_estimators=1200, learning_rate=0.008, max_depth=16,
        num_leaves=120, subsample=0.65, colsample_bytree=0.65, random_state=43, verbose=-1
    ),
    'CB_Agg': CatBoostRegressor(
        iterations=1200, learning_rate=0.008, depth=14,
        random_state=44, verbose=0, border_count=254
    ),
    'XGB_Bal': xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.012, max_depth=13,
        subsample=0.75, colsample_bytree=0.75, random_state=45
    ),
    'LGB_Bal': lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.012, max_depth=13,
        num_leaves=80, subsample=0.75, colsample_bytree=0.75, random_state=46, verbose=-1
    ),
    'CB_Bal': CatBoostRegressor(
        iterations=1000, learning_rate=0.012, depth=12,
        random_state=47, verbose=0, border_count=254
    ),
    'XGB_Cons': xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.020, max_depth=9,
        subsample=0.85, colsample_bytree=0.85, random_state=48
    ),
    'LGB_Cons': lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.020, max_depth=9,
        num_leaves=40, subsample=0.85, colsample_bytree=0.85, random_state=49, verbose=-1
    ),
}

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros((len(X_train), len(models)))
model_rmses = {}

for m_idx, (m_name, model) in enumerate(models.items()):
    print(f"{m_idx+1}. {m_name:15s}", end=" → ", flush=True)
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
    model_rmses[m_name] = avg_rmse
    print(f"RMSE: {avg_rmse:.4f}")

print("\n" + "=" * 80)
print("BASE MODEL CV RMSE:")
for name, rmse in sorted(model_rmses.items(), key=lambda x: x[1]):
    print(f"  {name:20s}: {rmse:.4f}")

# Learn optimal weights
print("\n" + "=" * 80)
print("Learning Ensemble Weights...")

def ensemble_rmse(w):
    w_norm = np.abs(w) / (np.sum(np.abs(w)) + 1e-8)
    pred = np.average(oof_preds, axis=1, weights=w_norm)
    return np.sqrt(((pred - y_train.values)**2).mean())

result = minimize(ensemble_rmse, x0=np.ones(len(models)), method='Nelder-Mead',
                  options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})

best_weights = np.abs(result.x) / np.sum(np.abs(result.x))
best_cv_rmse = ensemble_rmse(best_weights)

print(f"Ensemble CV RMSE: {best_cv_rmse:.4f}")
print(f"\nOptimal Weights:")
for (name, _), w in zip(models.items(), best_weights):
    if w > 0.01:
        print(f"  {name:20s}: {w*100:6.2f}%")

# Train final on full data
print("\n" + "=" * 80)
print("Training Final Models on Full Data...")
print("=" * 80)

test_preds = np.zeros((len(X_test), len(models)))

for m_idx, (m_name, model) in enumerate(models.items()):
    print(f"Training {m_name}...")
    model.fit(X_train, y_train)
    test_preds[:, m_idx] = np.clip(model.predict(X_test), 0, 68)

# Apply weights
final_pred = np.average(test_preds, axis=1, weights=best_weights)
final_pred = np.clip(final_pred, 0, 68)

# Create submission
submission = pd.DataFrame({
    'RecordID': test_df['RecordID'].values,
    'Overall Seed': final_pred
})

submission.to_csv('my_submission_ultra.csv', index=False)

print("\n" + "=" * 80)
print("ULTRA SUBMISSION READY")
print("=" * 80)
print(f"CV RMSE: {best_cv_rmse:.4f}")
print(f"Mean Seed: {final_pred.mean():.2f}")
print(f"Std Dev: {final_pred.std():.2f}")
print(f"Range: {final_pred.min():.2f} - {final_pred.max():.2f}")
print(f"Selected: {(final_pred > 0).sum()}/451")
print(f"\nFile: my_submission_ultra.csv")
print(f"Target: RMSE < 2.0")
