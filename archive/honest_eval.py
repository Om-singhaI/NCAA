"""
Evaluate model accuracy when trained ONLY on original training data.
No test labels leaked — this is the true generalization performance.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from model_perfect import parse_wl, extract_features, ACTUAL_SEEDS


def main():
    print("=" * 60)
    print("HONEST MODEL EVALUATION (train-only, no test labels)")
    print("=" * 60)

    train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
    test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

    # ONLY use training tournament teams
    train_tournament = train_df[train_df['Overall Seed'].notna() & (train_df['Overall Seed'] > 0)].copy()
    print(f"Training on: {len(train_tournament)} tournament teams (train only)")
    print(f"Predicting on: {len(test_df)} test rows")

    # Features
    # We need consistent encoding, so fit on combined then split
    combined_df = pd.concat([train_tournament, test_df], ignore_index=True)
    all_features = extract_features(combined_df)
    feature_cols = all_features.columns.tolist()

    n_train = len(train_tournament)
    X_train = all_features.iloc[:n_train][feature_cols].values
    y_train = train_tournament['Overall Seed'].values.astype(float)
    X_test = all_features.iloc[n_train:][feature_cols].values

    tournament_mask = test_df['Bid Type'].notna().values

    # Ground truth for test tournament teams
    perfect = pd.read_csv('sub_perfect_actual.csv')
    perfect_seeds = perfect['Overall Seed'].values

    models_preds = {}

    # --- XGBoost ---
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=3, random_state=42, verbosity=0
        )
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        models_preds['XGBoost'] = pred
    except ImportError:
        pass

    # --- LightGBM ---
    try:
        from lightgbm import LGBMRegressor
        lgb = LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
            min_child_samples=5, random_state=42, verbose=-1
        )
        lgb.fit(X_train, y_train)
        pred = lgb.predict(X_test)
        models_preds['LightGBM'] = pred
    except ImportError:
        pass

    # --- Ridge ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train_s, y_train)
    pred = ridge.predict(X_test_s)
    models_preds['Ridge'] = pred

    # Evaluate each model individually
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL MODEL RESULTS")
    print(f"{'='*60}")

    for name, pred in models_preds.items():
        pred_final = np.where(tournament_mask, np.round(pred).astype(int), 0)
        pred_final = np.clip(pred_final, 0, 68)

        diff = pred_final - perfect_seeds
        t_diff = diff[tournament_mask]
        exact = np.sum(t_diff == 0)
        within1 = np.sum(np.abs(t_diff) <= 1)
        within2 = np.sum(np.abs(t_diff) <= 2)
        within5 = np.sum(np.abs(t_diff) <= 5)
        rmse = np.sqrt(mean_squared_error(perfect_seeds, pred_final))
        mae = np.mean(np.abs(t_diff))

        print(f"\n{name}:")
        print(f"  RMSE (full 451): {rmse:.4f}")
        print(f"  MAE (tournament): {mae:.2f}")
        print(f"  Exact: {exact}/91 ({100*exact/91:.0f}%)")
        print(f"  Within 1: {within1}/91 ({100*within1/91:.0f}%)")
        print(f"  Within 2: {within2}/91 ({100*within2/91:.0f}%)")
        print(f"  Within 5: {within5}/91 ({100*within5/91:.0f}%)")
        print(f"  Max error: {np.max(np.abs(t_diff))}")

    # Ensemble blend
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*60}")

    blend = np.zeros(len(test_df))
    for pred in models_preds.values():
        blend += pred
    blend /= len(models_preds)

    # Raw rounded
    raw_pred = np.where(tournament_mask, np.round(blend).astype(int), 0)
    raw_pred = np.clip(raw_pred, 0, 68)
    diff = raw_pred - perfect_seeds
    t_diff = diff[tournament_mask]
    rmse_raw = np.sqrt(mean_squared_error(perfect_seeds, raw_pred))
    exact_raw = np.sum(t_diff == 0)
    mae_raw = np.mean(np.abs(t_diff))

    print(f"\nRaw ensemble (rounded):")
    print(f"  RMSE: {rmse_raw:.4f}")
    print(f"  MAE: {mae_raw:.2f}")
    print(f"  Exact: {exact_raw}/91")

    # Hungarian assignment per season
    print(f"\nWith Hungarian optimal assignment:")
    from perfect_submission import ACTUAL_SEEDS as PERFECT_MAP

    hungarian_pred = np.zeros(len(test_df))
    for season in sorted(test_df['Season'].unique()):
        season_t_mask = (test_df['Season'] == season).values & tournament_mask
        season_indices = np.where(season_t_mask)[0]

        season_positions = sorted([s for (se, t), s in PERFECT_MAP.items() if se == season])

        raw_vals = [(i, blend[i]) for i in season_indices]
        n = len(raw_vals)
        cost = np.zeros((n, n))
        for i, (idx, rv) in enumerate(raw_vals):
            for j, pos in enumerate(season_positions):
                cost[i][j] = abs(rv - pos)

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            hungarian_pred[raw_vals[i][0]] = season_positions[j]

        correct = sum(1 for i, j in zip(row_ind, col_ind)
                      if season_positions[j] == int(perfect_seeds[raw_vals[i][0]]))
        print(f"  {season}: {correct}/{n} exact")

    diff_h = hungarian_pred - perfect_seeds
    t_diff_h = diff_h[tournament_mask]
    rmse_h = np.sqrt(mean_squared_error(perfect_seeds, hungarian_pred.astype(int)))
    exact_h = np.sum(t_diff_h == 0)
    mae_h = np.mean(np.abs(t_diff_h))

    print(f"\n  Total exact: {exact_h}/91 ({100*exact_h/91:.0f}%)")
    print(f"  RMSE: {rmse_h:.4f}")
    print(f"  MAE: {mae_h:.2f}")
    print(f"  Max error: {np.max(np.abs(t_diff_h))}")

    # Show biggest misses
    print(f"\n{'='*60}")
    print(f"BIGGEST MISSES (sorted by error)")
    print(f"{'='*60}")
    errors = []
    for i in range(len(test_df)):
        if tournament_mask[i]:
            err = abs(hungarian_pred[i] - perfect_seeds[i])
            if err > 0:
                errors.append((err, test_df.iloc[i]['Season'], test_df.iloc[i]['Team'],
                               int(hungarian_pred[i]), int(perfect_seeds[i]), blend[i]))
    errors.sort(reverse=True)
    for err, season, team, predicted, actual, raw in errors[:20]:
        print(f"  {season} {team}: predicted={predicted}, actual={actual}, "
              f"error={int(err)}, raw={raw:.1f}")


if __name__ == "__main__":
    main()
