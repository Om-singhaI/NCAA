"""
FINAL MODEL: Stacking with meta-learner
Target: RMSE < 2
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
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
        df['quad_wins_total'] = (df['Quadrant1_wins'].fillna(0) +
                                  df['Quadrant2_wins'].fillna(0) +
                                  df['Quadrant3_wins'].fillna(0) +
                                  df['Quadrant4_wins'].fillna(0))
        df['quad1_pct'] = df['Quadrant1_wins'].fillna(0) / (df['quad_wins_total'] + 1)
        df['NET_valid'] = (~df['NET Rank'].isna()).astype(int)
        df['has_quad_wins'] = (df['quad_wins_total'] > 0).astype(int)

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

    for col in features:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(med)

    return df_train, df_test, features


def train_base_models(X, y, groups, n_splits=5):
    """Generate out-of-fold predictions from base models"""
    gkf = GroupKFold(n_splits=n_splits)
    
    # Storage for OOF predictions and test predictions
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_cb = np.zeros(len(X))
    
    test_xgb_preds = []
    test_lgb_preds = []
    test_cb_preds = []
    
    X_test = np.zeros_like(X)  # Placeholder
    
    print("=== STACKING: Training Base Models ===")
    fold_rmses = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # XGBoost
        xgb_m = xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=9,
            subsample=0.95, colsample_bytree=0.65,
            reg_alpha=0.30, reg_lambda=0.60, random_state=42
        )
        xgb_m.fit(X_tr, y_tr, verbose=False)
        oof_xgb[val_idx] = xgb_m.predict(X_val)

        # LightGBM
        lgb_m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.02, max_depth=9,
            num_leaves=80, subsample=0.95, colsample_bytree=0.65,
            reg_alpha=0.30, reg_lambda=0.60, random_state=42, verbose=-1
        )
        lgb_m.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_m.predict(X_val)

        # CatBoost
        cb_m = cb.CatBoostRegressor(
            iterations=600, learning_rate=0.02, depth=9,
            subsample=0.95, l2_leaf_reg=0.5, random_state=42, verbose=0
        )
        cb_m.fit(X_tr, y_tr)
        oof_cb[val_idx] = cb_m.predict(X_val)
        
        ensemble_pred = (oof_xgb[val_idx] + oof_lgb[val_idx] + oof_cb[val_idx]) / 3.0
        ensemble_pred = np.clip(ensemble_pred, 0, 68)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        fold_rmses.append(rmse)
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}")
    
    print(f"Mean CV RMSE: {np.mean(fold_rmses):.4f} ± {np.std(fold_rmses):.4f}\n")
    
    # Create meta-features
    meta_features = np.column_stack([
        oof_xgb,
        oof_lgb,
        oof_cb,
        (oof_xgb + oof_lgb + oof_cb) / 3.0,  # Ensemble mean
        np.abs(oof_xgb - oof_lgb),  # Disagreement
        np.abs(oof_lgb - oof_cb),
        np.abs(oof_xgb - oof_cb),
    ])
    
    return meta_features, fold_rmses


def train_meta_learner(meta_X, y, groups):
    """Train meta-learner on base model predictions"""
    print("=== STACKING: Training Meta-Learner ===")
    
    gkf = GroupKFold(n_splits=3)
    rmses = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(meta_X, y, groups)):
        X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        meta_m = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            num_leaves=32, subsample=0.9, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        meta_m.fit(X_tr, y_tr)
        preds = meta_m.predict(X_val)
        preds = np.clip(preds, 0, 68)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
        print(f"Fold {fold+1} - Meta RMSE: {rmse:.4f}")
    
    print(f"Mean Meta RMSE: {np.mean(rmses):.4f}\n")
    return rmses


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("Loading and processing data...")
    df_train, df_test, features = load_and_process(train_path, test_path)

    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values
    groups_train = df_train['Season'].values
    X_test = df_test[features].values

    print(f"Training: {len(X_train)} samples, {len(features)} features")
    print(f"Selected teams: {(y_train > 0).sum()}\n")

    # Stage 1: Generate OOF predictions
    meta_X, fold_rmses = train_base_models(X_train, y_train, groups_train)

    # Stage 2: Train meta-learner
    meta_rmses = train_meta_learner(meta_X, y_train, groups_train)

    # Stage 3: Final prediction
    print("=== FINAL PREDICTION ===\n")
    
    # Retrain base models on all data
    xgb_final = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=9,
        subsample=0.95, colsample_bytree=0.65,
        reg_alpha=0.30, reg_lambda=0.60, random_state=42
    )
    xgb_final.fit(X_train, y_train, verbose=False)
    
    lgb_final = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.02, max_depth=9,
        num_leaves=80, subsample=0.95, colsample_bytree=0.65,
        reg_alpha=0.30, reg_lambda=0.60, random_state=42, verbose=-1
    )
    lgb_final.fit(X_train, y_train)
    
    cb_final = cb.CatBoostRegressor(
        iterations=600, learning_rate=0.02, depth=9,
        subsample=0.95, l2_leaf_reg=0.5, random_state=42, verbose=0
    )
    cb_final.fit(X_train, y_train)
    
    # Base predictions on test
    test_xgb = xgb_final.predict(X_test)
    test_lgb = lgb_final.predict(X_test)
    test_cb = cb_final.predict(X_test)
    
    # Create meta-features for test
    test_meta = np.column_stack([
        test_xgb,
        test_lgb,
        test_cb,
        (test_xgb + test_lgb + test_cb) / 3.0,
        np.abs(test_xgb - test_lgb),
        np.abs(test_lgb - test_cb),
        np.abs(test_xgb - test_cb),
    ])
    
    # Meta-learner final prediction (retrain on all meta_X)
    meta_final = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        num_leaves=32, subsample=0.9, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    meta_final.fit(meta_X, y_train)
    final_preds = meta_final.predict(test_meta)
    final_preds = np.clip(final_preds, 0, 68)
    
    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"Final submission stats:")
    print(f"Mean seed: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
    print(f"Min/Max: {final_preds.min():.2f} / {final_preds.max():.2f}")
    print(f"Teams with seed > 0: {(final_preds > 0).sum()} / {len(final_preds)}\n")
    print("Wrote my_submission.csv\n")
    
    print("Top 25 predictions:")
    print(sub.head(25))


if __name__ == '__main__':
    main()
