"""
FINAL PUSH: Weighted Blending + Stacking + Post-processing
Target: RMSE < 2.0
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.pruners import MedianPruner
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


def load_and_process_v3(train_path, test_path):
    """Maximum feature engineering"""
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
        df['quad_high_wins'] = (df['Quadrant1_wins'].fillna(0) + df['Quadrant2_wins'].fillna(0))
        
        df['NET_valid'] = (~df['NET Rank'].isna()).astype(int)
        df['has_quad_wins'] = (df['quad_wins_total'] > 0).astype(int)
        
        # Interaction features
        df['NET_x_wins'] = df['NET Rank'] * df['total_wins']
        df['NET_x_winrate'] = df['NET Rank'] * df['win_rate']
        df['SOS_x_quad'] = df['NETSOS'] * df['quad_wins_total']
        
        # Log transforms
        df['log_NET'] = np.log1p(df['NET Rank'])
        df['log_opp_NET'] = np.log1p(df['AvgOppNETRank'])
        
        # Polynomial features
        df['NET_sq'] = df['NET Rank'] ** 2
        df['NET_inv'] = 1.0 / (df['NET Rank'] + 1)

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
        'quad_wins_total', 'quad_high_wins', 'quad1_pct',
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

    return df_train, df_test, features


def ensemble_with_weights(predictions_list, y_val, optimize=True):
    """Learn optimal weights for ensemble"""
    if not optimize:
        return np.ones(len(predictions_list)) / len(predictions_list)
    
    def loss(w):
        w = np.abs(w) / np.sum(np.abs(w))
        ensemble_pred = np.average(predictions_list, axis=0, weights=w)
        ensemble_pred = np.clip(ensemble_pred, 0, 68)
        return np.sqrt(mean_squared_error(y_val, ensemble_pred))
    
    x0 = np.ones(len(predictions_list)) / len(predictions_list)
    result = minimize(loss, x0, method='Nelder-Mead')
    w = np.abs(result.x) / np.sum(np.abs(result.x))
    return w


def generate_diverse_base_models(X, y, groups, X_test, n_splits=5):
    """Generate diverse base models with different seeds and params"""
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_list = []
    test_list = []
    model_names = []
    
    print("=== GENERATING 8 DIVERSE BASE MODELS ===\n")
    
    # Model configs: (name, model_fn, seed_offset)
    configs = [
        ('XGB Aggressive', lambda: xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.012, max_depth=11,
            subsample=0.95, colsample_bytree=0.55, reg_alpha=0.1, reg_lambda=0.3
        ), 42),
        ('LGB Aggressive', lambda: lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.012, max_depth=11, num_leaves=120,
            subsample=0.95, colsample_bytree=0.55, reg_alpha=0.1, reg_lambda=0.3, verbose=-1
        ), 42),
        ('CB Balanced', lambda: cb.CatBoostRegressor(
            iterations=800, learning_rate=0.012, depth=10,
            subsample=0.95, l2_leaf_reg=0.2, verbose=0
        ), 42),
        ('XGB Conservative', lambda: xgb.XGBRegressor(
            n_estimators=600, learning_rate=0.025, max_depth=8,
            subsample=0.9, colsample_bytree=0.7, reg_alpha=0.4, reg_lambda=0.8
        ), 100),
        ('LGB Conservative', lambda: lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.025, max_depth=8, num_leaves=80,
            subsample=0.9, colsample_bytree=0.7, reg_alpha=0.4, reg_lambda=0.8, verbose=-1
        ), 100),
        ('XGB Deep', lambda: xgb.XGBRegressor(
            n_estimators=700, learning_rate=0.015, max_depth=12,
            subsample=0.92, colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=0.5
        ), 200),
        ('LGB Deep', lambda: lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.015, max_depth=12, num_leaves=140,
            subsample=0.92, colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=0.5, verbose=-1
        ), 200),
        ('CB Optimized', lambda: cb.CatBoostRegressor(
            iterations=700, learning_rate=0.015, depth=11,
            subsample=0.92, l2_leaf_reg=0.25, verbose=0
        ), 200),
    ]
    
    for name, model_fn, seed_base in configs:
        oof = np.zeros(len(X))
        test = np.zeros(len(X_test))
        fold_rmses = []
        
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = model_fn()
            if hasattr(model, 'random_state'):
                model.random_state = seed_base + fold
            model.fit(X_tr, y_tr)
            
            oof[val_idx] = model.predict(X_val)
            test += model.predict(X_test) / n_splits
            
            rmse = np.sqrt(mean_squared_error(y_val, np.clip(oof[val_idx], 0, 68)))
            fold_rmses.append(rmse)
        
        rmse_mean = np.mean(fold_rmses)
        print(f"{name:20} - RMSE: {rmse_mean:.4f}")
        oof_list.append(oof)
        test_list.append(test)
        model_names.append(name)
    
    print()
    return np.column_stack(oof_list), np.column_stack(test_list), model_names


def learn_stacking_weights(meta_X, y, groups):
    """Learn optimal weights per fold"""
    print("=== LEARNING STACKING WEIGHTS ===")
    
    gkf = GroupKFold(n_splits=3)
    all_weights = []
    cv_rmses = []
    
    for tr_idx, val_idx in gkf.split(meta_X, y, groups):
        X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        # Learn weights on training
        w = ensemble_with_weights(X_tr.T, y_tr, optimize=True)
        all_weights.append(w)
        
        # Evaluate on validation
        ensemble_val = np.average(X_val, axis=1, weights=w)
        ensemble_val = np.clip(ensemble_val, 0, 68)
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
        cv_rmses.append(rmse)
        print(f"Fold RMSE: {rmse:.4f}, Weights: {w}")
    
    print(f"Mean weight CV RMSE: {np.mean(cv_rmses):.4f}\n")
    
    # Final weights: average across folds
    final_w = np.mean(all_weights, axis=0)
    final_w = final_w / np.sum(final_w)
    return final_w


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("🔥 FINAL PUSH: Weighted Ensemble + Advanced Stacking\n")
    
    df_train, df_test, features = load_and_process_v3(train_path, test_path)
    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values
    groups_train = df_train['Season'].values
    X_test = df_test[features].values

    print(f"Training: {len(X_train)} samples, {len(features)} features\n")

    # Stage 1: 8 diverse base models
    meta_X_train, meta_X_test, model_names = generate_diverse_base_models(
        X_train, y_train, groups_train, X_test, n_splits=5
    )

    # Stage 2: Learn optimal weights
    final_weights = learn_stacking_weights(meta_X_train, y_train, groups_train)

    # Stage 3: Apply weighted ensemble
    print("=== APPLYING WEIGHTED ENSEMBLE ===")
    final_preds = np.average(meta_X_test, axis=1, weights=final_weights)
    final_preds = np.clip(final_preds, 0, 68)

    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"\n✅ FINAL SUBMISSION")
    print(f"Mean seed: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
    print(f"Min/Max: {final_preds.min():.2f} / {final_preds.max():.2f}")
    print(f"Teams selected: {(final_preds > 0).sum()} / {len(final_preds)}")
    print(f"\nModel weights:")
    for name, w in zip(model_names, final_weights):
        print(f"  {name:20} {w:.4f}")
    print("\nWrote my_submission.csv")


if __name__ == '__main__':
    main()
