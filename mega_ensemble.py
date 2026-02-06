"""
HYBRID MEGA-ENSEMBLE: Average predictions from both ultra_aggressive and final_push models
"""
import pandas as pd
import numpy as np

print("🏆 MEGA-ENSEMBLE: Combining ultra_aggressive + final_push\n")

# Load both submissions
ultra_agg = pd.read_csv('my_submission.csv')
ultra_agg.columns = ['RecordID', 'seed_ultra']

# We need to run final_push to get its predictions, but for now generate using the final_push logic
# Actually, let me just load the current my_submission which is from final_push

# For mega-ensemble, let's also train ONE MORE variant using average of ultra models
print("Training mega-ensemble variant...")

import re
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

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

# Load data
df_train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
df_test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce')
df_train['Overall Seed'] = df_train['Overall Seed'].fillna(0)

numeric_cols = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']
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

for col in ['Conference', 'Bid Type']:
    if col in df_train.columns:
        df_train[col] = df_train[col].fillna('NA')
        df_test[col] = df_test[col].fillna('NA')
        cats = pd.concat([df_train[col], df_test[col]]).astype('category')
        mapping = {c: i for i, c in enumerate(cats.cat.categories)}
        df_train[col + '_enc'] = df_train[col].map(mapping).fillna(-1)
        df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1)

for df in (df_train, df_test):
    df['WL_ratio'] = df['WL_wins'] / (df['WL_losses'] + 1)
    df['Conf_ratio'] = df['Conf.Record_wins'] / (df['Conf.Record_losses'] + 1)
    df['Road_ratio'] = df['RoadWL_wins'] / (df['RoadWL_losses'] + 1)
    df['total_wins'] = df['WL_wins'].fillna(0) + df['Conf.Record_wins'].fillna(0)
    df['total_losses'] = df['WL_losses'].fillna(0) + df['Conf.Record_losses'].fillna(0)
    df['win_rate'] = df['total_wins'] / (df['total_wins'] + df['total_losses'] + 1)
    df['quad_wins_total'] = (df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0) +
                              df['Quadrant3_wins'].fillna(0) + df['Quadrant4_wins'].fillna(0))
    df['quad1_pct'] = df['Quadrant1_wins'].fillna(0) / (df['quad_wins_total'] + 1)
    df['quad_high_wins'] = df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0)
    df['NET_valid'] = (~df['NET Rank'].isna()).astype(int)
    df['has_quad_wins'] = (df['quad_wins_total'] > 0).astype(int)
    df['NET_x_wins'] = df['NET Rank'] * df['total_wins']
    df['NET_x_winrate'] = df['NET Rank'] * df['win_rate']
    df['SOS_x_quad'] = df['NETSOS'] * df['quad_wins_total']
    df['log_NET'] = np.log1p(df['NET Rank'])
    df['log_opp_NET'] = np.log1p(df['AvgOppNETRank'])
    df['NET_sq'] = df['NET Rank'] ** 2
    df['NET_inv'] = 1.0 / (df['NET Rank'] + 1)

features = [
    'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS',
    'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
    'WL_wins', 'WL_losses', 'Conf.Record_wins', 'Conf.Record_losses',
    'Non-ConferenceRecord_wins', 'Non-ConferenceRecord_losses', 'RoadWL_wins', 'RoadWL_losses',
    'Conference_enc', 'Bid Type_enc',
    'total_wins', 'total_losses', 'win_rate', 'quad_wins_total', 'quad_high_wins', 'quad1_pct',
    'NET_valid', 'has_quad_wins',
    'WL_ratio', 'Conf_ratio', 'Road_ratio',
    'NET_x_wins', 'NET_x_winrate', 'SOS_x_quad',
    'log_NET', 'log_opp_NET', 'NET_sq', 'NET_inv'
]

features = [f for f in features if f in df_train.columns]

for col in features:
    if col in df_train.columns:
        med = df_train[col].median()
        df_train[col] = df_train[col].fillna(med)
        if col in df_test.columns:
            df_test[col] = df_test[col].fillna(med)

X_train = df_train[features].values
y_train = df_train['Overall Seed'].values
groups_train = df_train['Season'].values
X_test = df_test[features].values

print("Training mega-variant with ultra-tuned XGBoost...")

# Train 3 XGBoost variants with extremely high learning capacity
gkf = GroupKFold(n_splits=5)

mega_preds = np.zeros(len(X_test))

fold_rmses = []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    
    # Variant 1: Maximum depth
    xgb1 = xgb.XGBRegressor(n_estimators=900, learning_rate=0.01, max_depth=13,
                            subsample=0.95, colsample_bytree=0.5, reg_alpha=0.05, reg_lambda=0.2,
                            random_state=42+fold)
    xgb1.fit(X_tr, y_tr, verbose=False)
    val_p1 = xgb1.predict(X_val)
    test_p1 = xgb1.predict(X_test) / 5
    
    # Variant 2: Low learning rate
    xgb2 = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.008, max_depth=10,
                            subsample=0.97, colsample_bytree=0.6, reg_alpha=0.08, reg_lambda=0.25,
                            random_state=100+fold)
    xgb2.fit(X_tr, y_tr, verbose=False)
    val_p2 = xgb2.predict(X_val)
    test_p2 = xgb2.predict(X_test) / 5
    
    # Variant 3: Balanced
    xgb3 = xgb.XGBRegressor(n_estimators=850, learning_rate=0.012, max_depth=11,
                            subsample=0.93, colsample_bytree=0.58, reg_alpha=0.06, reg_lambda=0.22,
                            random_state=200+fold)
    xgb3.fit(X_tr, y_tr, verbose=False)
    val_p3 = xgb3.predict(X_val)
    test_p3 = xgb3.predict(X_test) / 5
    
    # Average
    val_avg = (val_p1 + val_p2 + val_p3) / 3.0
    val_avg = np.clip(val_avg, 0, 68)
    rmse = np.sqrt(mean_squared_error(y_val, val_avg))
    fold_rmses.append(rmse)
    
    mega_preds += test_p1 + test_p2 + test_p3

print(f"Mega-variant CV RMSE: {np.mean(fold_rmses):.4f}\n")

mega_preds = np.clip(mega_preds, 0, 68)

# Blend with current my_submission
template = pd.read_csv('submission_template2.0.csv')
current_sub = pd.read_csv('my_submission.csv')

# Final ensemble: average current + mega variant
final_blend = (current_sub['Overall Seed'].values + mega_preds) / 2.0
final_blend = np.clip(final_blend, 0, 68)

template['Overall Seed'] = final_blend
template.to_csv('my_submission.csv', index=False)

print(f"✅ FINAL MEGA-ENSEMBLE")
print(f"Mean seed: {final_blend.mean():.2f}, Std: {final_blend.std():.2f}")
print(f"Min/Max: {final_blend.min():.2f} / {final_blend.max():.2f}")
print(f"Teams selected: {(final_blend > 0).sum()} / {len(final_blend)}")
print("\nWrote final my_submission.csv")
