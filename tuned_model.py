import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')


def parse_wl(s):
    """Parse W-L strings like '22-2' into (wins, losses)"""
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
    """Parse quadrant W-L strings"""
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

    # Target: seed (NaN for non-selected)
    df_train['is_selected'] = df_train['Overall Seed'].notna().astype(int)
    df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce')

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


def evaluate_cv(X, y, groups, model_type='lgb', params=None):
    """Cross-validate and return mean RMSE (only on selected=non-NaN teams)"""
    if params is None:
        params = {}

    gkf = GroupKFold(n_splits=5)
    rmses = []

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Keep only selected for training, but all for validation RMSE
        mask_tr = ~np.isnan(y_tr)
        mask_val = ~np.isnan(y_val)

        if mask_tr.sum() == 0 or mask_val.sum() == 0:
            continue

        X_tr_sel = X_tr[mask_tr]
        y_tr_sel = y_tr[mask_tr]
        X_val_sel = X_val[mask_val]
        y_val_sel = y_val[mask_val]

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        else:
            model = xgb.XGBRegressor(**params, random_state=42)

        model.fit(X_tr_sel, y_tr_sel)
        preds = model.predict(X_val_sel)
        rmse = np.sqrt(mean_squared_error(y_val_sel, preds))
        rmses.append(rmse)

    return np.mean(rmses) if rmses else float('inf')


def objective(trial, X, y, groups):
    """Objective function for hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    rmse = evaluate_cv(X, y, groups, model_type='lgb', params=params)
    return rmse


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("Loading and processing data...")
    df_train, df_test, features = load_and_process(train_path, test_path)

    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values  # NaN for non-selected
    groups_train = df_train['Season'].values

    print(f"Training samples: {len(X_train)}")
    print(f"Selected: {np.sum(~np.isnan(y_train))}")
    print(f"Features: {len(features)}\n")

    # Quick baseline CV
    print("=== BASELINE CV (LightGBM, default params) ===")
    baseline_rmse = evaluate_cv(X_train, y_train, groups_train, model_type='lgb')
    print(f"Baseline RMSE: {baseline_rmse:.4f}\n")

    # Hyperparameter tuning
    print("=== HYPERPARAMETER TUNING (20 trials) ===")
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    study.optimize(lambda trial: objective(trial, X_train, y_train, groups_train), n_trials=20, show_progress_bar=True)

    best_params = study.best_params
    best_rmse = study.best_value
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}\n")

    # Train final model with best params on all selected data
    print("=== TRAINING FINAL MODEL ===")
    mask = ~np.isnan(y_train)
    X_sel = X_train[mask]
    y_sel = y_train[mask]

    final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1)
    final_model.fit(X_sel, y_sel)

    # Predict
    print("Predicting test set...")
    X_test = df_test[features].values
    preds = final_model.predict(X_test)

    # Clip to valid range
    preds = np.clip(preds, 1, 68)

    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"Mean predicted seed: {np.mean(preds):.2f}")
    print(f"Min/Max predicted: {np.min(preds):.2f} / {np.max(preds):.2f}")
    print("Wrote my_submission.csv\n")

    print("Sample predictions:")
    print(sub.head(20))


if __name__ == '__main__':
    main()
