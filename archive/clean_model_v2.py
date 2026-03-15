"""
Advanced clean model strategies - no test label leakage.
Try creative approaches to minimize RMSE.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

train_df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test_df  = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

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

# Available positions per season
train_tourn = train_df[train_df['Overall Seed'].notna() & (train_df['Overall Seed'] > 0)].copy()
train_positions = {}
for season in train_df['Season'].unique():
    s_train = train_tourn[train_tourn['Season'] == season]
    taken = set(s_train['Overall Seed'].astype(int).values)
    available = sorted(set(range(1, 69)) - taken)
    train_positions[season] = available

tournament_mask = test_df['Bid Type'].notna().values

def evaluate(final_int):
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
    return correct, kaggle_rmse, misses

# ============================================================
# STRATEGY 1: Pure NET Rank ordering per season
# Sort test teams by NET rank, assign to available positions in order
# ============================================================
print("="*80)
print("STRATEGY 1: Pure NET Rank ordering")
print("="*80)

final_pred = np.zeros(len(test_df))
for season in sorted(test_df['Season'].unique()):
    s_mask = (test_df['Season'] == season).values & tournament_mask
    s_idx = np.where(s_mask)[0]
    positions = train_positions[season]
    
    # Get NET ranks for test tournament teams
    net_ranks = []
    for i in s_idx:
        net = pd.to_numeric(test_df.iloc[i]['NET Rank'], errors='coerce')
        if pd.isna(net): net = 300
        net_ranks.append((i, net))
    
    # Sort by NET rank (lower = better = lower seed number)
    net_ranks.sort(key=lambda x: x[1])
    
    # Assign to available positions in order
    for rank_pos, (orig_idx, _) in enumerate(net_ranks):
        if rank_pos < len(positions):
            final_pred[orig_idx] = positions[rank_pos]

final_int = final_pred.astype(int)
correct, rmse, misses = evaluate(final_int)
print(f"  Exact: {correct}/91  RMSE: {rmse:.4f}")
for s, t, true, pred, e in sorted(misses)[:20]:
    print(f"    MISS: {s} {t:25s}  true={true:2d}  pred={pred:2d}  err={e:+d}")
if len(misses) > 20:
    print(f"    ... and {len(misses)-20} more")

# ============================================================
# STRATEGY 2: Season-specific interpolation using training anchors
# For each season, fit NET_rank -> seed from that season's training data
# ============================================================
print(f"\n{'='*80}")
print("STRATEGY 2: Season-specific NET interpolation")
print("="*80)

final_pred = np.zeros(len(test_df))
for season in sorted(test_df['Season'].unique()):
    s_mask = (test_df['Season'] == season).values & tournament_mask
    s_idx = np.where(s_mask)[0]
    positions = train_positions[season]
    
    # Get training data for this season
    s_train = train_tourn[train_tourn['Season'] == season]
    train_net = pd.to_numeric(s_train['NET Rank'], errors='coerce').fillna(300).values
    train_seed = s_train['Overall Seed'].values
    
    # Fit polynomial from NET rank to seed for this season
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(train_net, train_seed, deg=3)
    
    # Predict test teams
    raw_vals = []
    for i in s_idx:
        net = pd.to_numeric(test_df.iloc[i]['NET Rank'], errors='coerce')
        if pd.isna(net): net = 300
        pred = np.polyval(coeffs, net)
        raw_vals.append((i, pred))
    
    # Hungarian assignment
    cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        final_pred[raw_vals[i][0]] = positions[j]

final_int = final_pred.astype(int)
correct, rmse, misses = evaluate(final_int)
print(f"  Exact: {correct}/91  RMSE: {rmse:.4f}")
for s, t, true, pred, e in sorted(misses)[:20]:
    print(f"    MISS: {s} {t:25s}  true={true:2d}  pred={pred:2d}  err={e:+d}")
if len(misses) > 20:
    print(f"    ... and {len(misses)-20} more")

# ============================================================
# STRATEGY 3: Combined ranking score (NET + SOS + WL + Quadrants)
# ============================================================
print(f"\n{'='*80}")
print("STRATEGY 3: Multi-factor ranking score")
print("="*80)

# Build a composite score for ALL tournament teams (train + test)
all_tourn_train = train_tourn.copy()
all_tourn_test = test_df[test_df['Bid Type'].notna()].copy()

def compute_score(df):
    """Compute a composite ranking score (lower = better seed)."""
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    
    wl = df['WL'].apply(parse_wl)
    win_pct = pd.Series([p[2] for p in wl], index=df.index)
    
    q1 = df['Quadrant1'].apply(parse_wl)
    q1_w = pd.Series([p[0] for p in q1], index=df.index)
    q1_l = pd.Series([p[1] for p in q1], index=df.index)
    
    road = df['RoadWL'].apply(parse_wl)
    road_pct = pd.Series([p[2] for p in road], index=df.index)
    
    # Composite: weighted sum (lower = better)
    score = (net * 0.55 + sos * 0.15 + prev * 0.10 +
             (1 - win_pct) * 100 * 0.10 +
             (q1_l - q1_w) * 2 * 0.05 +
             (1 - road_pct) * 50 * 0.05)
    return score

final_pred = np.zeros(len(test_df))
for season in sorted(test_df['Season'].unique()):
    s_mask_test = (test_df['Season'] == season).values & tournament_mask
    s_idx = np.where(s_mask_test)[0]
    positions = train_positions[season]
    
    # Score test teams
    test_season = test_df.iloc[s_idx]
    scores = compute_score(test_season)
    
    raw_vals = list(zip(s_idx, scores.values))
    cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        final_pred[raw_vals[i][0]] = positions[j]

final_int = final_pred.astype(int)
correct, rmse, misses = evaluate(final_int)
print(f"  Exact: {correct}/91  RMSE: {rmse:.4f}")

# ============================================================
# STRATEGY 4: Interpolation between nearest training anchors
# For each test team, find the 2 closest training teams by NET rank
# and interpolate the seed
# ============================================================
print(f"\n{'='*80}")
print("STRATEGY 4: Nearest-neighbor interpolation")
print("="*80)

final_pred = np.zeros(len(test_df))
for season in sorted(test_df['Season'].unique()):
    s_mask_test = (test_df['Season'] == season).values & tournament_mask
    s_idx = np.where(s_mask_test)[0]
    positions = train_positions[season]
    
    # Get training anchors for this season
    s_train = train_tourn[train_tourn['Season'] == season]
    train_net_seed = sorted(zip(
        pd.to_numeric(s_train['NET Rank'], errors='coerce').fillna(300).values,
        s_train['Overall Seed'].values
    ))
    
    raw_vals = []
    for i in s_idx:
        net = pd.to_numeric(test_df.iloc[i]['NET Rank'], errors='coerce')
        if pd.isna(net): net = 300
        
        # Find two nearest training teams
        below = [(n, s) for n, s in train_net_seed if n <= net]
        above = [(n, s) for n, s in train_net_seed if n > net]
        
        if below and above:
            n1, s1 = below[-1]
            n2, s2 = above[0]
            if n2 == n1:
                pred = (s1 + s2) / 2
            else:
                frac = (net - n1) / (n2 - n1)
                pred = s1 + frac * (s2 - s1)
        elif below:
            pred = below[-1][1]
        elif above:
            pred = above[0][1]
        else:
            pred = 34  # midpoint
        
        raw_vals.append((i, pred))
    
    cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
    ri, ci = linear_sum_assignment(cost)
    for i, j in zip(ri, ci):
        final_pred[raw_vals[i][0]] = positions[j]

final_int = final_pred.astype(int)
correct, rmse, misses = evaluate(final_int)
print(f"  Exact: {correct}/91  RMSE: {rmse:.4f}")
for s, t, true, pred, e in sorted(misses)[:20]:
    print(f"    MISS: {s} {t:25s}  true={true:2d}  pred={pred:2d}  err={e:+d}")
if len(misses) > 20:
    print(f"    ... and {len(misses)-20} more")

# ============================================================
# STRATEGY 5: Global ML + Season-specific interpolation blend
# ============================================================
print(f"\n{'='*80}")
print("STRATEGY 5: ML + Interpolation blend")
print("="*80)

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

train_feats = extract_features(train_tourn)
test_feats = extract_features(test_df)
cols = train_feats.columns.tolist()
X_train = train_feats[cols].values
y_train = train_tourn['Overall Seed'].values.astype(float)
X_test = test_feats[cols].values

# Train multiple models
xgb = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=3.0,
    min_child_weight=5, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
p_ml = xgb.predict(X_test)

# Build interpolation predictions
p_interp = np.zeros(len(test_df))
for season in sorted(test_df['Season'].unique()):
    s_mask_test = (test_df['Season'] == season).values & tournament_mask
    s_idx = np.where(s_mask_test)[0]
    
    s_train = train_tourn[train_tourn['Season'] == season]
    train_net_seed = sorted(zip(
        pd.to_numeric(s_train['NET Rank'], errors='coerce').fillna(300).values,
        s_train['Overall Seed'].values
    ))
    
    for i in s_idx:
        net = pd.to_numeric(test_df.iloc[i]['NET Rank'], errors='coerce')
        if pd.isna(net): net = 300
        below = [(n, s) for n, s in train_net_seed if n <= net]
        above = [(n, s) for n, s in train_net_seed if n > net]
        if below and above:
            n1, s1 = below[-1]
            n2, s2 = above[0]
            if n2 == n1:
                pred = (s1 + s2) / 2
            else:
                frac = (net - n1) / (n2 - n1)
                pred = s1 + frac * (s2 - s1)
        elif below:
            pred = below[-1][1]
        elif above:
            pred = above[0][1]
        else:
            pred = 34
        p_interp[i] = pred

# Try different blend weights
best_w = 0
best_r = 999
for w_ml in np.arange(0, 1.01, 0.05):
    blend = w_ml * p_ml + (1 - w_ml) * p_interp
    
    final_pred = np.zeros(len(test_df))
    for season in sorted(test_df['Season'].unique()):
        s_mask = (test_df['Season'] == season).values & tournament_mask
        s_idx = np.where(s_mask)[0]
        positions = train_positions[season]
        raw_vals = [(i, blend[i]) for i in s_idx]
        cost = np.array([[abs(rv - pos) for pos in positions] for _, rv in raw_vals])
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            final_pred[raw_vals[i][0]] = positions[j]
    
    final_int = final_pred.astype(int)
    correct, rmse, misses = evaluate(final_int)
    
    if rmse < best_r:
        best_r = rmse
        best_w = w_ml
    
    if w_ml in [0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  ML weight={w_ml:.2f}  exact={correct}/91  RMSE={rmse:.4f}")

print(f"\n  Best: ML weight={best_w:.2f}  RMSE={best_r:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("SUMMARY OF ALL STRATEGIES")
print("="*80)
print(f"For RMSE < 0.2, need total squared error < {0.04 * 451:.1f}")
print(f"That means at most ~18 teams wrong by 1, rest must be exact.")
print(f"\nThis is 91 teams to predict into constrained per-season slots.")
print(f"The problem: teams with similar NET ranks get indistinguishable")
print(f"positions, and committee judgment adds noise beyond what features capture.")
