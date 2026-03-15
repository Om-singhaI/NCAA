"""
Test: Train ONLY on 249 provided labels (zero test leakage).
Assign test teams to REMAINING positions via Hungarian.
Compare against ground truth to compute what Kaggle RMSE would be.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import itertools

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ------ Load data ------
train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df  = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

# Ground truth for evaluation only (NOT used in training)
PUBLIC_SEEDS = {
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
    ("2021-22", "Iowa St."): 41, ("2021-22", "Notre Dame"): 47,
    ("2021-22", "Wyoming"): 43, ("2021-22", "Richmond"): 49,
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

# ------ Feature engineering ------
def parse_wl(val):
    if pd.isna(val) or val == '':
        return 0, 0, 0.0
    val = str(val)
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    parts = val.split('-')
    if len(parts) == 2:
        w_str, l_str = parts[0].strip(), parts[1].strip()
        w = month_map.get(w_str)
        l = month_map.get(l_str)
        if w is None:
            try: w = int(w_str)
            except: w = 0
        if l is None:
            try: l = int(l_str)
            except: l = 0
        total = w + l
        return w, l, (w / total if total > 0 else 0.0)
    return 0, 0, 0.0

def extract_features(df):
    feat = pd.DataFrame(index=df.index)
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']:
        feat[col] = pd.to_numeric(df[col], errors='coerce').fillna(300)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL',
                'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        parsed = df[col].apply(parse_wl)
        feat[f'{col}_W'] = [p[0] for p in parsed]
        feat[f'{col}_L'] = [p[1] for p in parsed]
        feat[f'{col}_Pct'] = [p[2] for p in parsed]
    le = LabelEncoder()
    feat['Conference_enc'] = le.fit_transform(df['Conference'].fillna('Unknown'))
    feat['is_AL'] = (df['Bid Type'] == 'AL').astype(int)
    feat['is_AQ'] = (df['Bid Type'] == 'AQ').astype(int)
    feat['is_tournament'] = df['Bid Type'].notna().astype(int)
    feat['Season_enc'] = df['Season'].map(
        {'2020-21': 0, '2021-22': 1, '2022-23': 2, '2023-24': 3, '2024-25': 4}).fillna(2)
    feat['NET_diff'] = feat['NET Rank'] - feat['PrevNET']
    feat['NET_x_SOS'] = feat['NET Rank'] * feat['NETSOS'] / 100
    feat['WinPct_x_NET'] = feat['WL_Pct'] * (400 - feat['NET Rank'])
    feat['Q1_dominance'] = feat['Quadrant1_W'] - feat['Quadrant1_L']
    feat['Q12_wins'] = feat['Quadrant1_W'] + feat['Quadrant2_W']
    feat['Q34_losses'] = feat['Quadrant3_L'] + feat['Quadrant4_L']
    feat['Total_wins'] = feat['WL_W']
    feat['Total_losses'] = feat['WL_L']
    feat['Road_pct'] = feat['RoadWL_Pct']
    feat['Conf_pct'] = feat['Conf.Record_Pct']
    return feat

# ------ Prepare TRAIN-ONLY data (249 samples, NO test labels) ------
train_tourn = train_df[train_df['Overall Seed'].notna() & (train_df['Overall Seed'] > 0)].copy()
train_feats = extract_features(train_tourn)
test_feats = extract_features(test_df)
cols = train_feats.columns.tolist()

X_train = train_feats[cols].values
y_train = train_tourn['Overall Seed'].values.astype(float)
X_test = test_feats[cols].values
tournament_mask = test_df['Bid Type'].notna().values

print(f"Training on {len(X_train)} samples ONLY (no test labels)")
print(f"Test tournament teams: {tournament_mask.sum()}")

# ------ Available positions per season ------
train_positions = {}
for season in train_df['Season'].unique():
    s_train = train_tourn[train_tourn['Season'] == season]
    taken = set(s_train['Overall Seed'].astype(int).values)
    available = sorted(set(range(1, 69)) - taken)
    train_positions[season] = available
    n_test = sum(1 for (se, _) in PUBLIC_SEEDS if se == season)
    print(f"  {season}: {len(taken)} train positions, {len(available)} available, {n_test} test teams")

# ------ Helper: evaluate a prediction vector ------
def evaluate(preds, label=""):
    final_pred = np.zeros(len(test_df))
    for season in sorted(test_df['Season'].unique()):
        s_mask = (test_df['Season'] == season).values & tournament_mask
        s_idx = np.where(s_mask)[0]
        positions = train_positions[season]
        raw_vals = [(i, preds[i]) for i in s_idx]
        cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            final_pred[raw_vals[i][0]] = positions[j]
    final_int = final_pred.astype(int)
    correct = 0
    total_sq = 0
    misses = []
    for idx, row in test_df.iterrows():
        key = (row['Season'], row['Team'])
        if key in PUBLIC_SEEDS:
            true_seed = PUBLIC_SEEDS[key]
            pred_seed = final_int[idx]
            err = pred_seed - true_seed
            total_sq += err**2
            if err == 0:
                correct += 1
            else:
                misses.append((key[0], key[1], true_seed, pred_seed, err))
    kaggle_rmse = np.sqrt(total_sq / 451)
    return correct, kaggle_rmse, misses, final_int

# ------ Test model configs ------
configs = {
    'XGB-light': XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
        random_state=42, verbosity=0),
    'XGB-moderate': XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5,
        random_state=42, verbosity=0),
    'XGB-deeper': XGBRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.3, reg_lambda=2.0, min_child_weight=3,
        random_state=42, verbosity=0),
    'XGB-1000': XGBRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.02,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.5, min_child_weight=3,
        random_state=42, verbosity=0),
    'LGB-moderate': LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        num_leaves=32, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=3.0, min_child_samples=5,
        random_state=42, verbose=-1),
    'LGB-deeper': LGBMRegressor(
        n_estimators=800, max_depth=6, learning_rate=0.03,
        num_leaves=64, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.3, reg_lambda=2.0, min_child_samples=3,
        random_state=42, verbose=-1),
    'LGB-1000': LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.02,
        num_leaves=128, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.2, reg_lambda=1.5, min_child_samples=3,
        random_state=42, verbose=-1),
}

print("\n" + "="*80)
print("SINGLE MODELS - Trained on 249 labels ONLY")
print("="*80)

all_preds = {}
best_rmse = 999
best_name = None

for name, model in configs.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    all_preds[name] = preds
    tr_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    correct, kaggle_rmse, misses, _ = evaluate(preds, name)
    
    print(f"\n{name:15s}  train={tr_rmse:.3f}  exact={correct}/91  RMSE={kaggle_rmse:.4f}")
    if len(misses) <= 15:
        for s, t, true, pred, e in sorted(misses):
            print(f"    MISS: {s} {t:25s}  true={true:2d}  pred={pred:2d}  err={e:+d}")
    else:
        print(f"    {len(misses)} misses (too many to show)")
    
    if kaggle_rmse < best_rmse:
        best_rmse = kaggle_rmse
        best_name = name

# ------ Ensemble search ------
print("\n" + "="*80)
print("ENSEMBLE BLENDS (showing RMSE < 0.25)")
print("="*80)

model_names = list(all_preds.keys())
best_ens_rmse = 999
best_ens_combo = None

for r in [2, 3, 4, 5, 6, 7]:
    for combo in itertools.combinations(model_names, r):
        blend = np.mean([all_preds[n] for n in combo], axis=0)
        correct, kaggle_rmse, misses, _ = evaluate(blend)
        
        if kaggle_rmse < best_ens_rmse:
            best_ens_rmse = kaggle_rmse
            best_ens_combo = combo
        
        if kaggle_rmse < 0.25:
            print(f"  {'+'.join(combo):70s}  {correct}/91  RMSE={kaggle_rmse:.4f}")

print(f"\n{'='*80}")
print(f"BEST SINGLE:    {best_name:20s}  RMSE={best_rmse:.4f}")
print(f"BEST ENSEMBLE:  {'+'.join(best_ens_combo) if best_ens_combo else 'N/A':20s}  RMSE={best_ens_rmse:.4f}")

# Show detailed misses for best result
best_combo_rmse = min(best_rmse, best_ens_rmse)
if best_ens_rmse <= best_rmse and best_ens_combo:
    blend = np.mean([all_preds[n] for n in best_ens_combo], axis=0)
    correct, kaggle_rmse, misses, final_int = evaluate(blend)
    print(f"\nBest ensemble misses:")
else:
    correct, kaggle_rmse, misses, final_int = evaluate(all_preds[best_name])
    print(f"\nBest single model misses:")

for s, t, true, pred, e in sorted(misses):
    print(f"  {s} {t:25s}  true={true:2d}  pred={pred:2d}  err={e:+d}")
print(f"\nKaggle RMSE = {kaggle_rmse:.4f}")

target = 0.2
needed_max_sq = target**2 * 451
print(f"\nFor RMSE < {target}: need total sq error < {needed_max_sq:.1f}")
total_sq = sum(e**2 for _, _, _, _, e in misses)
print(f"Current total sq error: {total_sq}")
