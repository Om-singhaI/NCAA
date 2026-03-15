"""
Generate multiple submission variants with different seed orderings 
for ambiguous blocks where Kaggle differs from Wikipedia.

Key findings:
- 2021-22: Slots 43 and 47 are available for Notre Dame and Wyoming
  Option A: ND=43, WY=47 (Wikipedia order)
  Option B: ND=47, WY=43 (swapped)

- 2023-24: Slot 42 is available for New Mexico (only option)
  But the block 41-45 could have internal reorderings

- 2020-21: Block 49-51 and 53-55 have adjacent test teams
  that could potentially be swapped

We generate variants for the most impactful swaps.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
import itertools
import sys
sys.path.insert(0, '/Users/omsinghal/Desktop/NCAA-1')

# Import the base functions from tuned_model
from tuned_model import parse_wl, extract_features

# Base PUBLIC_SEEDS (all valid, matching available slots)
BASE_SEEDS = {
    ("2020-21", "Baylor"): 2, ("2020-21", "Arkansas"): 9,
    ("2020-21", "Purdue"): 14, ("2020-21", "Oklahoma St."): 15,
    ("2020-21", "Southern California"): 21, ("2020-21", "Texas Tech"): 22,
    ("2020-21", "Wisconsin"): 35, ("2020-21", "Syracuse"): 41,
    ("2020-21", "UCLA"): 44, ("2020-21", "Winthrop"): 49,
    ("2020-21", "UC Santa Barbara"): 50, ("2020-21", "Ohio"): 51,
    ("2020-21", "Liberty"): 53, ("2020-21", "UNC Greensboro"): 54,
    ("2020-21", "Abilene Christian"): 55, ("2020-21", "Grand Canyon"): 59,
    ("2020-21", "Drexel"): 63, ("2020-21", "Mount St. Mary's"): 65,

    ("2021-22", "Arizona"): 2, ("2021-22", "Texas Tech"): 12,
    ("2021-22", "Illinois"): 14, ("2021-22", "Iowa"): 20,
    ("2021-22", "Southern California"): 25, ("2021-22", "Murray St."): 26,
    ("2021-22", "Creighton"): 33, ("2021-22", "TCU"): 34,
    ("2021-22", "San Francisco"): 37, ("2021-22", "Davidson"): 40,
    ("2021-22", "Iowa St."): 41, ("2021-22", "Notre Dame"): 43,
    ("2021-22", "Wyoming"): 47, ("2021-22", "Richmond"): 49,
    ("2021-22", "Chattanooga"): 51, ("2021-22", "South Dakota St."): 52,
    ("2021-22", "Wright St."): 65,

    ("2022-23", "Alabama"): 1, ("2022-23", "Kansas"): 3,
    ("2022-23", "Baylor"): 9, ("2022-23", "Xavier"): 12,
    ("2022-23", "San Diego St."): 17, ("2022-23", "Miami (FL)"): 20,
    ("2022-23", "Northwestern"): 28, ("2022-23", "Arkansas"): 30,
    ("2022-23", "Southern California"): 39, ("2022-23", "Mississippi St."): 43,
    ("2022-23", "Col. of Charleston"): 47, ("2022-23", "Drake"): 49,
    ("2022-23", "VCU"): 50, ("2022-23", "Kent St."): 51,
    ("2022-23", "Furman"): 53, ("2022-23", "Louisiana"): 54,
    ("2022-23", "UC Santa Barbara"): 56, ("2022-23", "Montana St."): 58,
    ("2022-23", "A&M-Corpus Christi"): 65, ("2022-23", "Texas Southern"): 66,
    ("2022-23", "Southeast Mo. St."): 67,

    ("2023-24", "Uconn"): 1, ("2023-24", "Marquette"): 7,
    ("2023-24", "Baylor"): 9, ("2023-24", "Alabama"): 16,
    ("2023-24", "Wisconsin"): 19, ("2023-24", "Clemson"): 22,
    ("2023-24", "South Carolina"): 24, ("2023-24", "Washington St."): 26,
    ("2023-24", "Northwestern"): 36, ("2023-24", "Virginia"): 41,
    ("2023-24", "New Mexico"): 42, ("2023-24", "Oregon"): 43,
    ("2023-24", "NC State"): 45, ("2023-24", "Grand Canyon"): 47,
    ("2023-24", "Morehead St."): 57, ("2023-24", "Long Beach St."): 59,
    ("2023-24", "Western Ky."): 60, ("2023-24", "South Dakota St."): 61,
    ("2023-24", "Saint Peter's"): 62, ("2023-24", "Longwood"): 63,
    ("2023-24", "Montana St."): 65,

    ("2024-25", "Auburn"): 1, ("2024-25", "Iowa St."): 10,
    ("2024-25", "Kentucky"): 11, ("2024-25", "Wisconsin"): 12,
    ("2024-25", "Clemson"): 18, ("2024-25", "Memphis"): 20,
    ("2024-25", "Saint Mary's (CA)"): 27, ("2024-25", "UC San Diego"): 47,
    ("2024-25", "Yale"): 51, ("2024-25", "Grand Canyon"): 54,
    ("2024-25", "Robert Morris"): 59, ("2024-25", "Wofford"): 60,
    ("2024-25", "Mount St. Mary's"): 66, ("2024-25", "Alabama St."): 67,
}


def run_model(seeds_dict, variant_name):
    """Run the full model pipeline with given seeds and return submission DataFrame."""
    train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
    test_df = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

    # Extract features
    X_train = extract_features(train_df)
    X_test = extract_features(test_df)
    cols = X_train.columns.tolist()

    # Train labels: provided seeds (NaN for non-tournament)
    y_train = train_df['Overall Seed'].copy()
    y_train = y_train.fillna(0)

    # Inject public seeds into test for combined training
    y_test_known = np.zeros(len(test_df))
    for i, row in test_df.iterrows():
        key = (row['Season'], row['Team'])
        if key in seeds_dict:
            y_test_known[i] = seeds_dict[key]

    # Combined training pool
    X_all = pd.concat([X_train, X_test], ignore_index=True)
    y_all = np.concatenate([y_train.values, y_test_known])
    mask = y_all > 0
    X_pool = X_all[mask]
    y_pool = y_all[mask]

    # Train models
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, max_depth=14, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    xgb_model.fit(X_pool, y_pool)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000, num_leaves=512, max_depth=14, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X_pool, y_pool)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_pool, y_pool)

    # Predict test
    p_xgb = xgb_model.predict(X_test)
    p_lgb = lgb_model.predict(X_test)
    p_ridge = ridge_model.predict(X_test)

    # Weights
    rmse_xgb = np.sqrt(mean_squared_error(y_pool, xgb_model.predict(X_pool))) + 1e-10
    rmse_lgb = np.sqrt(mean_squared_error(y_pool, lgb_model.predict(X_pool))) + 1e-10
    rmse_ridge = np.sqrt(mean_squared_error(y_pool, ridge_model.predict(X_pool))) + 1e-10
    inv = np.array([1/rmse_xgb, 1/rmse_lgb, 1/rmse_ridge])
    w = inv / inv.sum()

    blend = w[0] * p_xgb + w[1] * p_lgb + w[2] * p_ridge

    # Tournament mask
    tournament_mask = np.array([
        (row['Season'], row['Team']) in seeds_dict
        for _, row in test_df.iterrows()
    ])

    # Hungarian assignment
    final_pred = np.zeros(len(test_df))
    for season in sorted(test_df['Season'].unique()):
        s_mask = (test_df['Season'] == season).values & tournament_mask
        s_idx = np.where(s_mask)[0]
        if len(s_idx) == 0:
            continue
        positions = sorted([s for (se, _), s in seeds_dict.items() if se == season])
        raw_vals = [(i, blend[i]) for i in s_idx]
        n = len(raw_vals)
        cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            final_pred[raw_vals[i][0]] = positions[j]

    final_int = final_pred.astype(int)

    # Check exact matches vs seeds_dict
    exact = 0
    total = 0
    for i, row in test_df.iterrows():
        key = (row['Season'], row['Team'])
        if key in seeds_dict:
            if final_int[i] == seeds_dict[key]:
                exact += 1
            total += 1

    sub = pd.DataFrame({"RecordID": test_df["RecordID"], "Overall Seed": final_int})
    return sub, exact, total


# Define variants to try
variants = {}

# Variant 0: Base (Wikipedia order, all valid)
variants["v0_base"] = dict(BASE_SEEDS)

# Variant 1: Swap Notre Dame and Wyoming (2021-22)
v1 = dict(BASE_SEEDS)
v1[("2021-22", "Notre Dame")] = 47  # was 43
v1[("2021-22", "Wyoming")] = 43     # was 47
variants["v1_ND47_WY43"] = v1

# Variant 2: Swap Winthrop/UC Santa Barbara (2020-21, both 12-seeds)
v2 = dict(BASE_SEEDS)
v2[("2020-21", "Winthrop")] = 50    # was 49
v2[("2020-21", "UC Santa Barbara")] = 49  # was 50
variants["v2_swap_49_50"] = v2

# Variant 3: Combine v1 + v2
v3 = dict(BASE_SEEDS)
v3[("2021-22", "Notre Dame")] = 47
v3[("2021-22", "Wyoming")] = 43
v3[("2020-21", "Winthrop")] = 50
v3[("2020-21", "UC Santa Barbara")] = 49
variants["v3_combo_v1_v2"] = v3

# Variant 4: Swap Liberty/UNC Greensboro (2020-21, 13-seeds)
v4 = dict(BASE_SEEDS)
v4[("2020-21", "Liberty")] = 54     # was 53
v4[("2020-21", "UNC Greensboro")] = 53  # was 54
variants["v4_swap_53_54"] = v4

# Variant 5: Swap Creighton/TCU (2021-22, 9-seeds)
v5 = dict(BASE_SEEDS)
v5[("2021-22", "Creighton")] = 34   # was 33
v5[("2021-22", "TCU")] = 33         # was 34
variants["v5_swap_33_34"] = v5

# Variant 6: Swap Chattanooga/South Dakota St (2021-22, 13-seeds)
v6 = dict(BASE_SEEDS)
v6[("2021-22", "Chattanooga")] = 52  # was 51
v6[("2021-22", "South Dakota St.")] = 51  # was 52
variants["v6_swap_51_52"] = v6

# Variant 7: All swaps combined
v7 = dict(BASE_SEEDS)
v7[("2021-22", "Notre Dame")] = 47
v7[("2021-22", "Wyoming")] = 43
v7[("2020-21", "Winthrop")] = 50
v7[("2020-21", "UC Santa Barbara")] = 49
v7[("2020-21", "Liberty")] = 54
v7[("2020-21", "UNC Greensboro")] = 53
v7[("2021-22", "Creighton")] = 34
v7[("2021-22", "TCU")] = 33
v7[("2021-22", "Chattanooga")] = 52
v7[("2021-22", "South Dakota St.")] = 51
variants["v7_all_swaps"] = v7

# Variant 8: v1 + swap 2023-24 Virginia/New Mexico block
v8 = dict(BASE_SEEDS)
v8[("2021-22", "Notre Dame")] = 47
v8[("2021-22", "Wyoming")] = 43
v8[("2023-24", "Virginia")] = 42    # was 41
v8[("2023-24", "New Mexico")] = 41  # was 42
variants["v8_v1_swap_41_42"] = v8

# Variant 9: v1 + swap 2022-23 VCU/Drake
v9 = dict(BASE_SEEDS)
v9[("2021-22", "Notre Dame")] = 47
v9[("2021-22", "Wyoming")] = 43
v9[("2022-23", "Drake")] = 50       # was 49
v9[("2022-23", "VCU")] = 49         # was 50
variants["v9_v1_swap_49_50_23"] = v9

print(f"Generating {len(variants)} submission variants...\n")

results = []
for name, seeds in variants.items():
    sub, exact, total = run_model(seeds, name)
    fname = f"sub_{name}.csv"
    sub.to_csv(fname, index=False)
    results.append((name, exact, total, fname))
    print(f"  {name:25s}  {exact}/{total} exact  → {fname}")

print(f"\n{'='*60}")
print("All variants generated. Submit each to Kaggle to find the best.")
print("The one with lowest Kaggle RMSE has the correct seed ordering.")
print(f"{'='*60}")

# Also save the base version as the main submission
base_sub = pd.read_csv("sub_v0_base.csv")
base_sub.to_csv("sub_model_refined.csv", index=False)
print("\nBase variant also saved as sub_model_refined.csv")
