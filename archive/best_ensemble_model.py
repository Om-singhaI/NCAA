"""
Final model: Predict seeds (0 for non-selected) using ensemble methods.
Key insight: Use 0 to indicate non-selected teams, not NaN.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def parse_wl(s):
    """Parse W-L strings"""
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
    """Parse quadrant strings"""
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

    # Target: seed (0 for non-selected, 1-68 for selected)
    df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce')
    df_train['Overall Seed'] = df_train['Overall Seed'].fillna(0)  # 0 = not selected

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

    # Conference and Bid Type encoding
    for col in ['Conference', 'Bid Type']:
        if col in df_train.columns:
            df_train[col] = df_train[col].fillna('NA')
            df_test[col] = df_test[col].fillna('NA')
            cats = pd.concat([df_train[col], df_test[col]]).astype('category')
            mapping = {c: i for i, c in enumerate(cats.cat.categories)}
            df_train[col + '_enc'] = df_train[col].map(mapping).fillna(-1)
            df_test[col + '_enc'] = df_test[col].map(mapping).fillna(-1)

    # Create composite features
    df_train['total_wins'] = df_train['WL_wins'].fillna(0) + df_train['Conf.Record_wins'].fillna(0)
    df_train['NET_is_valid'] = (~df_train['NET Rank'].isna()).astype(int)
    df_train['quad_wins_total'] = (df_train['Quadrant1_wins'].fillna(0) +
                                    df_train['Quadrant2_wins'].fillna(0) +
                                    df_train['Quadrant3_wins'].fillna(0) +
                                    df_train['Quadrant4_wins'].fillna(0))

    df_test['total_wins'] = df_test['WL_wins'].fillna(0) + df_test['Conf.Record_wins'].fillna(0)
    df_test['NET_is_valid'] = (~df_test['NET Rank'].isna()).astype(int)
    df_test['quad_wins_total'] = (df_test['Quadrant1_wins'].fillna(0) +
                                   df_test['Quadrant2_wins'].fillna(0) +
                                   df_test['Quadrant3_wins'].fillna(0) +
                                   df_test['Quadrant4_wins'].fillna(0))

    # Feature list
    features = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'WL_wins', 'WL_losses',
        'Conf.Record_wins', 'Conf.Record_losses',
        'Non-ConferenceRecord_wins', 'Non-ConferenceRecord_losses',
        'RoadWL_wins', 'RoadWL_losses',
        'Conference_enc', 'Bid Type_enc',
        'total_wins', 'NET_is_valid', 'quad_wins_total'
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


def cross_validate_ensemble(X, y, groups, n_splits=5):
    """CV with ensemble of models"""
    gkf = GroupKFold(n_splits=n_splits)
    rmses = []
    
    xgb_models = []
    lgb_models = []

    print("=== CROSS-VALIDATION ===")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # XGBoost with optimized params from tuning
        xgb_model = xgb.XGBRegressor(
            n_estimators=420,
            learning_rate=0.027,
            max_depth=9,
            subsample=0.99,
            colsample_bytree=0.61,
            reg_alpha=0.40,
            reg_lambda=0.71,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_preds = xgb_model.predict(X_val)
        xgb_models.append(xgb_model)

        # LightGBM with tuned params
        lgb_model = lgb.LGBMRegressor(
            n_estimators=420,
            learning_rate=0.027,
            max_depth=9,
            num_leaves=73,
            subsample=0.99,
            colsample_bytree=0.61,
            reg_alpha=0.40,
            reg_lambda=0.71,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_tr, y_tr)
        lgb_preds = lgb_model.predict(X_val)
        lgb_models.append(lgb_model)

        # Ensemble: average XGB + LGB
        ensemble_preds = (xgb_preds + lgb_preds) / 2.0
        ensemble_preds = np.clip(ensemble_preds, 0, 68)
        
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
        rmses.append(rmse)
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}")

    print(f"Mean CV RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}\n")
    return xgb_models, lgb_models


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("Loading and processing data...")
    df_train, df_test, features = load_and_process(train_path, test_path)

    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values  # 0 for non-selected, 1-68 for selected
    groups_train = df_train['Season'].values

    print(f"Training samples: {len(X_train)}")
    print(f"Selected: {(y_train > 0).sum()}")
    print(f"Features: {len(features)}\n")

    # Cross-validate
    xgb_models, lgb_models = cross_validate_ensemble(X_train, y_train, groups_train)

    # Train final models on all data
    print("=== TRAINING FINAL MODELS ===")
    final_xgb = xgb.XGBRegressor(
        n_estimators=420,
        learning_rate=0.027,
        max_depth=9,
        subsample=0.99,
        colsample_bytree=0.61,
        reg_alpha=0.40,
        reg_lambda=0.71,
        random_state=42
    )
    final_xgb.fit(X_train, y_train)

    final_lgb = lgb.LGBMRegressor(
        n_estimators=420,
        learning_rate=0.027,
        max_depth=9,
        num_leaves=73,
        subsample=0.99,
        colsample_bytree=0.61,
        reg_alpha=0.40,
        reg_lambda=0.71,
        random_state=42,
        verbose=-1
    )
    final_lgb.fit(X_train, y_train)

    # Predict on test
    print("Predicting test set...")
    X_test = df_test[features].values
    
    xgb_test_preds = final_xgb.predict(X_test)
    lgb_test_preds = final_lgb.predict(X_test)
    
    # Ensemble average
    final_preds = (xgb_test_preds + lgb_test_preds) / 2.0
    final_preds = np.clip(final_preds, 0, 68)

    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"\nSubmission stats:")
    print(f"Mean seed: {final_preds.mean():.2f}")
    print(f"Std seed: {final_preds.std():.2f}")
    print(f"Min/Max: {final_preds.min():.2f} / {final_preds.max():.2f}")
    print(f"Teams with seed > 0: {(final_preds > 0).sum()} / {len(final_preds)}")
    print("Wrote my_submission.csv\n")

    print("Sample predictions:")
    print(sub.head(25))


if __name__ == '__main__':
    main()
