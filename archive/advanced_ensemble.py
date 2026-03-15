"""
Advanced ensemble: XGB + LGB + CatBoost with aggressive feature engineering.
Target: RMSE < 2
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False
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


def load_and_process(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Target: seed (0 for non-selected)
    df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce')
    df_train['Overall Seed'] = df_train['Overall Seed'].fillna(0)

    # Parse numeric columns
    numeric_cols = ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
                    'NETSOS', 'NETNonConfSOS']
    for col in numeric_cols:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

    # Parse quadrant wins
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        df_train[q + '_wins'] = df_train.get(q, pd.Series()).apply(parse_quad)
        df_test[q + '_wins'] = df_test.get(q, pd.Series()).apply(parse_quad)

    # Parse W-L records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        for df in (df_train, df_test):
            if col in df.columns:
                wins_losses = df[col].apply(parse_wl)
                df[col + '_wins'] = wins_losses.apply(lambda x: x[0])
                df[col + '_losses'] = wins_losses.apply(lambda x: x[1])

    # Conference and Bid Type
    for col in ['Conference', 'Bid Type']:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('NA')
            df_test[col] = df_test[col].fillna('NA')
            cats = pd.concat([df_train[col], df_test[col]]).astype('category')
            mapping = {c: i for i, c in enumerate(cats.cat.categories)}
            df_train[col + '_enc'] = df_train[col].map(mapping).fillna(-1)
            df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1)

    # Advanced features
    for df in (df_train, df_test):
        # Wins/losses ratios
        df['WL_ratio'] = df['WL_wins'] / (df['WL_losses'] + 1)
        df['Conf_ratio'] = df['Conf.Record_wins'] / (df['Conf.Record_losses'] + 1)
        df['Road_ratio'] = df['RoadWL_wins'] / (df['RoadWL_losses'] + 1)
        
        # Total and win rates
        df['total_wins'] = df['WL_wins'].fillna(0) + df['Conf.Record_wins'].fillna(0)
        df['total_losses'] = df['WL_losses'].fillna(0) + df['Conf.Record_losses'].fillna(0)
        df['win_rate'] = df['total_wins'] / (df['total_wins'] + df['total_losses'] + 1)
        
        # Quadrant metrics
        df['quad_wins_total'] = (df['Quadrant1_wins'].fillna(0) +
                                  df['Quadrant2_wins'].fillna(0) +
                                  df['Quadrant3_wins'].fillna(0) +
                                  df['Quadrant4_wins'].fillna(0))
        df['quad1_pct'] = df['Quadrant1_wins'].fillna(0) / (df['quad_wins_total'] + 1)
        
        # Indicators
        df['NET_valid'] = (~df['NET Rank'].isna()).astype(int)
        df['has_quad_wins'] = (df['quad_wins_total'] > 0).astype(int)
        
        # Smoothed NET Rank (forward fill by season)
        df['NET_smooth'] = df['NET Rank'].fillna(999)

    # Features list
    features = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'WL_wins', 'WL_losses',
        'Conf.Record_wins', 'Conf.Record_losses',
        'Non-ConferenceRecord_wins', 'Non-ConferenceRecord_losses',
        'RoadWL_wins', 'RoadWL_losses',
        'Conference_enc', 'Bid Type_enc',
        'total_wins', 'total_losses', 'win_rate',
        'quad_wins_total', 'quad1_pct',
        'NET_valid', 'has_quad_wins',
        'WL_ratio', 'Conf_ratio', 'Road_ratio'
    ]

    features = [f for f in features if f in df_train.columns]

    # Fill NaNs
    for col in features:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(med)

    return df_train, df_test, features


def cross_validate_3model_ensemble(X, y, groups, n_splits=5):
    """CV with 3-model ensemble (XGB, LGB, CatBoost)"""
    gkf = GroupKFold(n_splits=n_splits)
    rmses = []

    print("=== CROSS-VALIDATION (3-Model Ensemble) ===")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.025, max_depth=8,
            subsample=0.95, colsample_bytree=0.65,
            reg_alpha=0.35, reg_lambda=0.65, random_state=42
        )
        xgb_m.fit(X_tr, y_tr)
        xgb_p = xgb_m.predict(X_val)

        # LightGBM
        lgb_m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.025, max_depth=8,
            num_leaves=70, subsample=0.95, colsample_bytree=0.65,
            reg_alpha=0.35, reg_lambda=0.65, random_state=42, verbose=-1
        )
        lgb_m.fit(X_tr, y_tr)
        lgb_p = lgb_m.predict(X_val)

        # CatBoost
        if HAS_CATBOOST:
            cb_m = cb.CatBoostRegressor(
                iterations=500, learning_rate=0.025, depth=8,
                subsample=0.95, colsample_bylevel=0.65,
                l2_leaf_reg=1, random_state=42, verbose=0
            )
            cb_m.fit(X_tr, y_tr)
            cb_p = cb_m.predict(X_val)
            ensemble_p = (xgb_p + lgb_p + cb_p) / 3.0
        else:
            ensemble_p = (xgb_p + lgb_p) / 2.0

        ensemble_p = np.clip(ensemble_p, 0, 68)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_p))
        rmses.append(rmse)
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}")

    print(f"Mean CV RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}\n")
    return rmses


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("Loading and processing data with advanced features...")
    df_train, df_test, features = load_and_process(train_path, test_path)

    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values
    groups_train = df_train['Season'].values

    print(f"Training samples: {len(X_train)}, Selected: {(y_train > 0).sum()}")
    print(f"Features: {len(features)}\n")

    # CV evaluation
    rmses = cross_validate_3model_ensemble(X_train, y_train, groups_train)

    # Train final ensemble on all data
    print("=== TRAINING FINAL MODELS ===")
    X_train_all = X_train
    y_train_all = y_train

    xgb_final = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.025, max_depth=8,
        subsample=0.95, colsample_bytree=0.65,
        reg_alpha=0.35, reg_lambda=0.65, random_state=42
    )
    xgb_final.fit(X_train_all, y_train_all)

    lgb_final = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.025, max_depth=8,
        num_leaves=70, subsample=0.95, colsample_bytree=0.65,
        reg_alpha=0.35, reg_lambda=0.65, random_state=42, verbose=-1
    )
    lgb_final.fit(X_train_all, y_train_all)

    if HAS_CATBOOST:
        cb_final = cb.CatBoostRegressor(
            iterations=500, learning_rate=0.025, depth=8,
            subsample=0.95, colsample_bylevel=0.65,
            l2_leaf_reg=1, random_state=42, verbose=0
        )
        cb_final.fit(X_train_all, y_train_all)

    # Predict
    print("Predicting test set...")
    X_test = df_test[features].values

    xgb_test = xgb_final.predict(X_test)
    lgb_test = lgb_final.predict(X_test)
    
    if HAS_CATBOOST:
        cb_test = cb_final.predict(X_test)
        final_preds = (xgb_test + lgb_test + cb_test) / 3.0
        print("Used 3-model ensemble (XGB + LGB + CatBoost)")
    else:
        final_preds = (xgb_test + lgb_test) / 2.0
        print("Used 2-model ensemble (XGB + LGB)")

    final_preds = np.clip(final_preds, 0, 68)

    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"\nSubmission stats:")
    print(f"Mean seed: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
    print(f"Min/Max: {final_preds.min():.2f} / {final_preds.max():.2f}")
    print(f"Teams with seed > 0: {(final_preds > 0).sum()} / {len(final_preds)}")
    print("\nWrote my_submission.csv\n")

    print("Top 25 predictions:")
    print(sub.head(25))


if __name__ == '__main__':
    main()
