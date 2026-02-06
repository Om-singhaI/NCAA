"""
ULTRA-AGGRESSIVE: Stacking + Optuna tuning + Weighted ensemble
Target: RMSE < 2.0
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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


def load_and_process_v2(train_path, test_path):
    """Enhanced feature engineering"""
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

    # Advanced features
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
        
        # Log transforms for better scaling
        df['log_NET'] = np.log1p(df['NET Rank'])
        df['log_opp_NET'] = np.log1p(df['AvgOppNETRank'])

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
        'log_NET', 'log_opp_NET'
    ]

    features = [f for f in features if f in df_train.columns]

    for col in features:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            if col in df_test.columns:
                df_test[col] = df_test[col].fillna(med)

    return df_train, df_test, features


def generate_base_predictions(X, y, groups, X_test, n_splits=5):
    """Generate OOF + test predictions from 5 diverse base models"""
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_preds = []
    test_preds = []
    rmses = []
    
    print("=== BASE MODELS (5 diverse models) ===\n")
    
    # Model 1: XGBoost aggressive
    print("Model 1: XGBoost (aggressive)")
    oof_xgb = np.zeros(len(X))
    test_xgb = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        m = xgb.XGBRegressor(
            n_estimators=700, learning_rate=0.015, max_depth=10,
            subsample=0.95, colsample_bytree=0.6, colsample_bylevel=0.8,
            reg_alpha=0.2, reg_lambda=0.5, random_state=42+fold
        )
        m.fit(X_tr, y_tr, verbose=False)
        oof_xgb[val_idx] = m.predict(X_val)
        test_xgb += m.predict(X_test) / n_splits
    
    rmse = np.sqrt(mean_squared_error(y, np.clip(oof_xgb, 0, 68)))
    rmses.append(rmse)
    oof_preds.append(oof_xgb)
    test_preds.append(test_xgb)
    print(f"  OOF RMSE: {rmse:.4f}\n")
    
    # Model 2: LightGBM aggressive
    print("Model 2: LightGBM (aggressive)")
    oof_lgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        m = lgb.LGBMRegressor(
            n_estimators=700, learning_rate=0.015, max_depth=10,
            num_leaves=100, subsample=0.95, colsample_bytree=0.6,
            reg_alpha=0.2, reg_lambda=0.5, random_state=42+fold, verbose=-1
        )
        m.fit(X_tr, y_tr)
        oof_lgb[val_idx] = m.predict(X_val)
        test_lgb += m.predict(X_test) / n_splits
    
    rmse = np.sqrt(mean_squared_error(y, np.clip(oof_lgb, 0, 68)))
    rmses.append(rmse)
    oof_preds.append(oof_lgb)
    test_preds.append(test_lgb)
    print(f"  OOF RMSE: {rmse:.4f}\n")
    
    # Model 3: CatBoost
    print("Model 3: CatBoost")
    oof_cb = np.zeros(len(X))
    test_cb = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        m = cb.CatBoostRegressor(
            iterations=700, learning_rate=0.015, depth=10,
            subsample=0.95, l2_leaf_reg=0.3, random_state=42+fold, verbose=0
        )
        m.fit(X_tr, y_tr)
        oof_cb[val_idx] = m.predict(X_val)
        test_cb += m.predict(X_test) / n_splits
    
    rmse = np.sqrt(mean_squared_error(y, np.clip(oof_cb, 0, 68)))
    rmses.append(rmse)
    oof_preds.append(oof_cb)
    test_preds.append(test_cb)
    print(f"  OOF RMSE: {rmse:.4f}\n")
    
    # Model 4: XGBoost conservative
    print("Model 4: XGBoost (conservative)")
    oof_xgb2 = np.zeros(len(X))
    test_xgb2 = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=7,
            subsample=0.9, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.0, random_state=100+fold
        )
        m.fit(X_tr, y_tr, verbose=False)
        oof_xgb2[val_idx] = m.predict(X_val)
        test_xgb2 += m.predict(X_test) / n_splits
    
    rmse = np.sqrt(mean_squared_error(y, np.clip(oof_xgb2, 0, 68)))
    rmses.append(rmse)
    oof_preds.append(oof_xgb2)
    test_preds.append(test_xgb2)
    print(f"  OOF RMSE: {rmse:.4f}\n")
    
    # Model 5: LightGBM conservative
    print("Model 5: LightGBM (conservative)")
    oof_lgb2 = np.zeros(len(X))
    test_lgb2 = np.zeros(len(X_test))
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        m = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=7,
            num_leaves=60, subsample=0.9, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=1.0, random_state=100+fold, verbose=-1
        )
        m.fit(X_tr, y_tr)
        oof_lgb2[val_idx] = m.predict(X_val)
        test_lgb2 += m.predict(X_test) / n_splits
    
    rmse = np.sqrt(mean_squared_error(y, np.clip(oof_lgb2, 0, 68)))
    rmses.append(rmse)
    oof_preds.append(oof_lgb2)
    test_preds.append(test_lgb2)
    print(f"  OOF RMSE: {rmse:.4f}\n")
    print(f"Mean base model RMSE: {np.mean(rmses):.4f}\n")
    
    return np.column_stack(oof_preds), np.column_stack(test_preds)


def tune_metalearner(meta_X, y, groups):
    """Optuna-tuned meta-learner"""
    print("=== TUNING META-LEARNER WITH OPTUNA ===")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        }
        
        gkf = GroupKFold(n_splits=3)
        rmses = []
        
        for tr_idx, val_idx in gkf.split(meta_X, y, groups):
            X_tr, X_val = meta_X[tr_idx], meta_X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            m = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_val)
            preds = np.clip(preds, 0, 68)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmses.append(rmse)
        
        return np.mean(rmses)
    
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=3)
    )
    study.optimize(objective, n_trials=15, show_progress_bar=True)
    
    best_params = study.best_params
    best_rmse = study.best_value
    print(f"Best meta RMSE: {best_rmse:.4f}")
    print(f"Best params: {best_params}\n")
    
    return best_params


def main():
    train_path = 'NCAA_Seed_Training_Set2.0.csv'
    test_path = 'NCAA_Seed_Test_Set2.0.csv'
    submission_template = 'submission_template2.0.csv'

    print("🚀 ULTRA-AGGRESSIVE: Stacking + Optuna\n")
    print("Loading data...")
    df_train, df_test, features = load_and_process_v2(train_path, test_path)

    X_train = df_train[features].values
    y_train = df_train['Overall Seed'].values
    groups_train = df_train['Season'].values
    X_test = df_test[features].values

    print(f"Training: {len(X_train)} samples, {len(features)} features")
    print(f"Selected teams: {(y_train > 0).sum()}\n")

    # Stage 1: 5 base models
    meta_X_train, meta_X_test = generate_base_predictions(
        X_train, y_train, groups_train, X_test, n_splits=5
    )

    # Stage 2: Tune meta-learner
    best_meta_params = tune_metalearner(meta_X_train, y_train, groups_train)

    # Stage 3: Train final meta-learner
    print("=== TRAINING FINAL META-LEARNER ===")
    meta_final = lgb.LGBMRegressor(**best_meta_params, random_state=42, verbose=-1)
    meta_final.fit(meta_X_train, y_train)
    
    final_preds = meta_final.predict(meta_X_test)
    final_preds = np.clip(final_preds, 0, 68)

    # Write submission
    sub = pd.read_csv(submission_template)
    sub['Overall Seed'] = final_preds
    sub.to_csv('my_submission.csv', index=False)

    print(f"\n✅ FINAL SUBMISSION")
    print(f"Mean seed: {final_preds.mean():.2f}, Std: {final_preds.std():.2f}")
    print(f"Min/Max: {final_preds.min():.2f} / {final_preds.max():.2f}")
    print(f"Teams selected: {(final_preds > 0).sum()} / {len(final_preds)}")
    print("\nWrote my_submission.csv\n")
    
    print("Top predictions:")
    print(sub.head(20))


if __name__ == '__main__':
    main()
