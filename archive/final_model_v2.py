import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
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
        return int(m.group(1))  # wins in quadrant
    m2 = re.search(r"(\d+)", str(s))
    if m2:
        return int(m2.group(1))
    return np.nan


def load_and_process(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Create binary target: is_selected
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

    # Feature list
    features = [
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1_wins', 'Quadrant2_wins', 'Quadrant3_wins', 'Quadrant4_wins',
        'WL_wins', 'WL_losses',
        'Conf.Record_wins', 'Conf.Record_losses',
        'Non-ConferenceRecord_wins', 'Non-ConferenceRecord_losses',
        'RoadWL_wins', 'RoadWL_losses',
        'Conference_enc', 'Bid Type_enc'
    ]

    # Keep only features that exist
    features = [f for f in features if f in df_train.columns]

    # Fill NaNs with median
    for col in features:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(med)

    return df_train, df_test, features


def train_selection_model(X, y_sel, groups):
    """Train model to predict tournament selection"""
    print("\n=== SELECTION MODEL (Stage 1) ===")
    gkf = StratifiedGroupKFold(n_splits=5)
    aucs = []
    models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_sel, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_sel[tr_idx], y_sel[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=sum(y_tr == 0) / sum(y_tr == 1)
        )
        model.fit(X_tr, y_tr)
        preds_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds_prob)
        print(f"Fold {fold+1} - AUC: {auc:.4f}")
        aucs.append(auc)
        models.append(model)

    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return models


def train_seed_model(X, y_seed, groups):
    """Train model to predict seed (only on selected teams)"""
    print("\n=== SEED MODEL (Stage 2) ===")
    gkf = GroupKFold(n_splits=5)
    rmses_xgb = []
    rmses_lgb = []
    xgb_models = []
    lgb_models = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y_seed, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_seed[tr_idx], y_seed[val_idx]

        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        preds_xgb = xgb_model.predict(X_val)
        rmse_xgb = np.sqrt(mean_squared_error(y_val, preds_xgb))
        rmses_xgb.append(rmse_xgb)
        xgb_models.append(xgb_model)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_tr, y_tr)
        preds_lgb = lgb_model.predict(X_val)
        rmse_lgb = np.sqrt(mean_squared_error(y_val, preds_lgb))
        rmses_lgb.append(rmse_lgb)
        lgb_models.append(lgb_model)

        print(f"Fold {fold+1} - XGBoost RMSE: {rmse_xgb:.4f}, LightGBM RMSE: {rmse_lgb:.4f}")

    print(f"XGBoost - Mean RMSE: {np.mean(rmses_xgb):.4f} ± {np.std(rmses_xgb):.4f}")
    print(f"LightGBM - Mean RMSE: {np.mean(rmses_lgb):.4f} ± {np.std(rmses_lgb):.4f}")

    return xgb_models, lgb_models


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("Loading and processing data...")
    df_train, df_test, features = load_and_process(train_path, test_path)

    X_train = df_train[features].values
    y_selection = df_train['is_selected'].values
    y_seed = df_train['Overall Seed'].values
    groups_train = df_train['Season'].values

    print(f"Training samples: {len(X_train)}")
    print(f"Selected: {y_selection.sum()} ({100*y_selection.mean():.1f}%)")
    print(f"Features: {len(features)}")

    # Train selection model
    sel_models = train_selection_model(X_train, y_selection, groups_train)

    # Train seed model (only on selected teams)
    X_selected = X_train[y_selection == 1]
    y_selected = y_seed[y_selection == 1]
    groups_selected = groups_train[y_selection == 1]
    xgb_models, lgb_models = train_seed_model(X_selected, y_selected, groups_selected)

    # Predict on test set
    print("\n=== PREDICTION ON TEST SET ===")
    X_test = df_test[features].values

    # Selection predictions (ensemble)
    sel_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in sel_models], axis=0)

    # Seed predictions (ensemble of XGB and LGB)
    seed_preds_xgb = np.mean([m.predict(X_test) for m in xgb_models], axis=0)
    seed_preds_lgb = np.mean([m.predict(X_test) for m in lgb_models], axis=0)
    seed_preds = (seed_preds_xgb + seed_preds_lgb) / 2.0

    # Combine: only output seed for selected teams, else NaN
    final_preds = np.where(sel_probs > 0.5, seed_preds, np.nan)

    # For submission: ensure valid seeds (1-68 or NaN)
    final_preds = np.clip(final_preds, 1, 68)

    # Load template and write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"Predictions: {np.sum(~np.isnan(final_preds))} selected teams")
    print(f"Mean predicted seed: {np.nanmean(final_preds):.2f}")
    print("Wrote my_submission.csv")

    # Show sample predictions
    print("\nSample predictions:")
    print(sub.head(20))


if __name__ == '__main__':
    main()
