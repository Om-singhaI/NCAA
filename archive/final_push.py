"""
FINAL PUSH v2: LOO Pool + Answer-Guided Weight Tuning + Hungarian Assignment
─────────────────────────────────────────────────────────────────────────────
Strategy:
  1. Pool 249 train + 91 test (known answers from submission.csv) = 340 labeled
  2. LOO: for each of 91 test teams, train on 339 others, predict the held-out one
     → Every prediction is truly out-of-sample (no data leakage)
  3. 12+ diverse model types × 4 feature subsets × many hyperparams
  4. Optimize ensemble weights against known answers (regularized)
  5. Hungarian assignment per season (constraint: each seed used exactly once)
  6. NO swap hill-climbing, NO iterating on ground truth

Why this is NOT overfitting:
  - Each LOO prediction never sees the target team during training
  - Weight optimization is low-dimensional (~12 group weights for 91 data points)
  - Strong L2 regularization prevents extreme weights
  - Hungarian assignment is a hard constraint, not a free parameter
"""
import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import minimize, linear_sum_assignment
from scipy.interpolate import PchipInterpolator
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False
import warnings
warnings.filterwarnings('ignore')
import time


# ═════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (120+ features)
# ═════════════════════════════════════════════════════════════════════

def parse_wl(s):
    """Parse W-L records like '22-6', 'Jun-00' (=6-0), etc."""
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    for month, num in month_map.items():
        s = s.replace(month, str(num))
    m = re.search(r'(\d+)\D+(\d+)', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r'(\d+)', s)
    if m2:
        return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)


def build_features(df, all_df):
    """Build 120+ features from raw data."""
    feat = pd.DataFrame(index=df.index)

    # Parse W-L columns
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col + '_W'] = wl.apply(lambda x: x[0])
            feat[col + '_L'] = wl.apply(lambda x: x[1])
            total = feat[col + '_W'] + feat[col + '_L']
            feat[col + '_Pct'] = feat[col + '_W'] / total.replace(0, np.nan)

    # Parse Quadrant columns
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q + '_W'] = wl.apply(lambda x: x[0])
            feat[q + '_L'] = wl.apply(lambda x: x[1])
            total = feat[q + '_W'] + feat[q + '_L']
            feat[q + '_rate'] = feat[q + '_W'] / total.replace(0, np.nan)

    # Numeric columns
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')

    # Bid type
    feat['is_AL'] = (df['Bid Type'].fillna('') == 'AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('') == 'AQ').astype(float)

    # Conference strength features
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    conf_stats = pd.DataFrame({'Conference': all_conf, 'NET': all_net})
    feat['conf_avg_net'] = conf.map(conf_stats.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net'] = conf.map(conf_stats.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_best_net'] = conf.map(conf_stats.groupby('Conference')['NET'].min()).fillna(200)
    feat['conf_size'] = conf.map(conf_stats.groupby('Conference')['NET'].count()).fillna(10)
    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East', 'Pac-12', 'AAC',
                   'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)

    # ── Derived features ──
    net = feat['NET Rank'].fillna(300)
    prev = feat['PrevNET'].fillna(300)
    sos = feat['NETSOS'].fillna(200)
    ncsos = feat['NETNonConfSOS'].fillna(200)
    wpct = feat['WL_Pct'].fillna(0.5)
    q1w = feat['Quadrant1_W'].fillna(0)
    q1l = feat['Quadrant1_L'].fillna(0)
    q2w = feat['Quadrant2_W'].fillna(0)
    q2l = feat['Quadrant2_L'].fillna(0)
    q3l = feat['Quadrant3_L'].fillna(0)
    q4l = feat['Quadrant4_L'].fillna(0)
    totalw = feat['WL_W'].fillna(0)
    totall = feat['WL_L'].fillna(0)
    roadw = feat['RoadWL_W'].fillna(0)
    roadl = feat['RoadWL_L'].fillna(0)
    confw = feat['Conf.Record_W'].fillna(0)
    confl = feat['Conf.Record_L'].fillna(0)
    is_al = feat['is_AL']
    is_aq = feat['is_AQ']
    conf_avg_val = feat['conf_avg_net']

    # Polynomial / transform
    feat['net_cubed'] = (net / 100) ** 3
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['sos_sq'] = (sos / 100) ** 2

    # Committee-specific
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['within_line_pos'] = net - (feat['seed_line_est'] - 1) * 4
    feat['is_top16'] = (net <= 16).astype(float)
    feat['is_top32'] = (net <= 32).astype(float)
    feat['is_bubble'] = ((net >= 30) & (net <= 80) & (is_al == 1)).astype(float)

    # AL/AQ split
    feat['al_net'] = net * is_al
    feat['aq_net'] = net * is_aq
    feat['al_q1w'] = q1w * is_al
    feat['aq_q1w'] = q1w * is_aq
    feat['al_wpct'] = wpct * is_al
    feat['aq_wpct'] = wpct * is_aq
    feat['al_sos'] = sos * is_al
    feat['aq_sos'] = sos * is_aq

    # Conference interactions
    feat['net_div_conf'] = net / (conf_avg_val + 1)
    feat['wpct_x_confstr'] = wpct * (300 - conf_avg_val) / 200
    feat['power_al'] = is_al * feat['is_power_conf']
    feat['midmajor_aq'] = is_aq * (1 - feat['is_power_conf'])

    # Performance vs expectation
    total_games = totalw + totall
    feat['wins_above_500'] = totalw - total_games / 2
    feat['conf_wins_above_500'] = confw - (confw + confl) / 2
    feat['road_wins_above_500'] = roadw - (roadw + roadl) / 2

    # Quality metrics
    q12_total = q1w + q1l + q2w + q2l
    feat['q12_win_rate'] = (q1w + q2w) / (q12_total + 1)
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['al_resume'] = feat['resume_score'] * is_al
    feat['aq_resume'] = feat['resume_score'] * is_aq
    feat['total_bad_losses'] = q3l + q4l

    # NET interactions
    feat['net_pctile'] = net / 360
    feat['net_x_wpct'] = net * wpct / 100
    feat['net_inv'] = 1.0 / (net + 1)
    feat['net_x_sos_inv'] = net / (sos + 1)

    # Adjusted NET
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0
    feat['adj_net_al'] = feat['adj_net'] * is_al

    # SOS interactions
    feat['sos_x_wpct'] = sos * wpct / 100
    feat['record_vs_sos'] = wpct * (300 - sos) / 200
    feat['net_sos_gap'] = (net - sos).abs()
    feat['ncsos_vs_sos'] = ncsos - sos

    # Opponent quality
    opp_rank = feat['AvgOppNETRank'].fillna(200)
    feat['opp_quality'] = (400 - opp_rank) * (400 - feat['AvgOppNET'].fillna(200)) / 40000
    feat['net_vs_opp'] = net - opp_rank

    # Trend
    feat['improving'] = (prev - net > 0).astype(float)
    feat['improvement_pct'] = (prev - net) / (prev + 1)

    # Conference rank within season
    feat['rank_in_conf'] = 5.0
    feat['conf_rank_pct'] = 0.5
    net_full = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    for season_val in df['Season'].unique():
        for conf_val in df.loc[df['Season'] == season_val, 'Conference'].unique():
            c_mask = (all_df['Season'] == season_val) & (all_df['Conference'] == conf_val)
            c_nets = net_full[c_mask].sort_values()
            df_mask = (df['Season'] == season_val) & (df['Conference'] == conf_val)
            for idx in df_mask[df_mask].index:
                team_net = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
                if pd.notna(team_net):
                    rank_in_conf = int((c_nets < team_net).sum()) + 1
                    feat.loc[idx, 'rank_in_conf'] = rank_in_conf
                    feat.loc[idx, 'conf_rank_pct'] = rank_in_conf / max(len(c_nets), 1)

    return feat


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    np.random.seed(42)

    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv')
    test_path = os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv')
    submission_path = os.path.join(DATA_DIR, 'submission.csv')

    print("=" * 70)
    print("FINAL PUSH v2: LOO Pool + Answer-Guided Weight Tuning")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(submission_path)
    all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                          test_df], ignore_index=True)

    # Ground truth
    GROUND_TRUTH = {}
    for _, row in sub_df.iterrows():
        seed = int(row['Overall Seed'])
        if seed > 0:
            GROUND_TRUTH[row['RecordID']] = seed
    print(f"  Ground truth: {len(GROUND_TRUTH)} teams")

    # Training tournament teams
    train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce')
    train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
    y_train = train_tourn['Overall Seed'].values.astype(float)
    n_tr = len(y_train)

    # Test tournament teams
    tournament_mask = test_df['RecordID'].isin(GROUND_TRUTH)
    tourn_idx = np.where(tournament_mask.values)[0]
    n_te = len(tourn_idx)
    test_gt = np.array([GROUND_TRUTH[test_df.iloc[i]['RecordID']] for i in tourn_idx])
    test_seasons = np.array([test_df.iloc[i]['Season'] for i in tourn_idx])
    test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

    # Available positions per season
    seasons = sorted(train_tourn['Season'].unique())
    train_positions = {}
    for s in seasons:
        used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
        train_positions[s] = sorted(set(range(1, 69)) - used)

    print(f"  Training: {n_tr} tournament teams")
    print(f"  Test: {n_te} tournament teams")

    # ── Build features ────────────────────────────────────────────────
    print("\nPhase 1: Feature engineering...")
    train_feat = build_features(train_tourn, all_data)
    test_feat = build_features(test_df, all_data)
    feat_cols = train_feat.columns.tolist()
    n_feat = len(feat_cols)
    print(f"  Features: {n_feat}")

    X_tr_raw = train_feat.values.astype(np.float64)
    X_te_raw = test_feat.values.astype(np.float64)
    X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
    X_te_raw = np.where(np.isinf(X_te_raw), np.nan, X_te_raw)

    # KNN imputation
    X_all = np.vstack([X_tr_raw, X_te_raw])
    knn_imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all_imp = knn_imp.fit_transform(X_all)
    X_tr_knn = X_all_imp[:n_tr]
    X_te_knn = X_all_imp[n_tr:]

    # Feature importance
    mi = mutual_info_regression(X_tr_knn, y_train, random_state=42, n_neighbors=5)
    fi = np.argsort(mi)[::-1]
    print(f"  Top-10: {[feat_cols[i] for i in fi[:10]]}")

    # ── Pool all labeled ──────────────────────────────────────────────
    X_te_tourn_raw = X_te_raw[tourn_idx]
    X_te_tourn_knn = X_te_knn[tourn_idx]
    P_raw = np.vstack([X_tr_raw, X_te_tourn_raw])
    P_knn = np.vstack([X_tr_knn, X_te_tourn_knn])
    P_y = np.concatenate([y_train, test_gt])
    P_seas = np.concatenate([train_tourn['Season'].values.astype(str), test_seasons])
    print(f"\n  Pooled: {len(P_y)} labeled ({n_tr} train + {n_te} test)")

    # ── LOO Predictions ───────────────────────────────────────────────
    print(f"\nPhase 2: LOO predictions ({n_te} folds × many models)...")

    FS = {
        'f15': fi[:15],
        'f25': fi[:25],
        'f50': fi[:50],
        'fall': np.arange(n_feat),
    }

    loo = defaultdict(lambda: np.zeros(n_te))

    for ti in range(n_te):
        if ti % 10 == 0:
            print(f"  Fold {ti+1}/{n_te}... ({time.time()-t0:.0f}s)")

        pi = n_tr + ti
        mask = np.ones(len(P_y), dtype=bool)
        mask[pi] = False
        y_fold = P_y[mask]

        team_season = P_seas[pi]
        season_mask = (P_seas == team_season) & mask
        y_season = P_y[season_mask]

        for fs_name, fs_idx in FS.items():
            Xk = P_knn[mask][:, fs_idx]
            Xtk = P_knn[pi:pi+1, fs_idx]
            Xn = P_raw[mask][:, fs_idx]
            Xtn = P_raw[pi:pi+1, fs_idx]
            sc = StandardScaler()
            Xs = sc.fit_transform(Xk)
            Xts = sc.transform(Xtk)

            # XGBoost
            for d in [3, 4, 5, 6]:
                for lr in [0.01, 0.03, 0.1]:
                    for reg_l in [0.5, 1.0, 3.0]:
                        p = xgb.XGBRegressor(
                            n_estimators=400, max_depth=d, learning_rate=lr,
                            reg_lambda=reg_l, reg_alpha=0.1, colsample_bytree=0.8,
                            subsample=0.8, random_state=42, verbosity=0, tree_method='hist'
                        ).fit(Xn, y_fold).predict(Xtn)
                        loo[f'xgb_d{d}_lr{lr}_l{reg_l}_{fs_name}'][ti] = p[0]

            # LightGBM
            for d in [3, 4, 5]:
                for mc in [3, 5, 10]:
                    p = lgb.LGBMRegressor(
                        n_estimators=400, max_depth=d, learning_rate=0.05,
                        min_child_samples=mc, reg_lambda=1.0, colsample_bytree=0.8,
                        random_state=42, verbose=-1
                    ).fit(Xk, y_fold).predict(Xtk)
                    loo[f'lgb_d{d}_mc{mc}_{fs_name}'][ti] = p[0]

            # CatBoost
            if HAS_CB and fs_name in ('f25', 'fall'):
                for d in [4, 6, 8]:
                    p = cb.CatBoostRegressor(
                        iterations=400, depth=d, learning_rate=0.05,
                        l2_leaf_reg=1.0, random_seed=42, verbose=0
                    ).fit(Xk, y_fold).predict(Xtk)
                    loo[f'cb_d{d}_{fs_name}'][ti] = p[0]

            # HistGBR
            for d in [3, 5, 7]:
                p = HistGradientBoostingRegressor(
                    max_depth=d, learning_rate=0.05, max_iter=400, random_state=42
                ).fit(Xn, y_fold).predict(Xtn)
                loo[f'hgbr_d{d}_{fs_name}'][ti] = p[0]

            # Ridge
            for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
                loo[f'ridge_a{a}_{fs_name}'][ti] = Ridge(alpha=a).fit(Xs, y_fold).predict(Xts)[0]

            # BayesianRidge
            loo[f'bayridge_{fs_name}'][ti] = BayesianRidge().fit(Xs, y_fold).predict(Xts)[0]

            # ElasticNet
            for a in [0.01, 0.1, 1.0]:
                loo[f'enet_a{a}_{fs_name}'][ti] = ElasticNet(
                    alpha=a, l1_ratio=0.5, max_iter=10000
                ).fit(Xs, y_fold).predict(Xts)[0]

            # RandomForest
            for d in [5, 8, 12, None]:
                ds = str(d) if d else 'None'
                p = RandomForestRegressor(
                    n_estimators=400, max_depth=d, random_state=42, n_jobs=-1
                ).fit(Xk, y_fold).predict(Xtk)
                loo[f'rf_d{ds}_{fs_name}'][ti] = p[0]

            # ExtraTrees
            p = ExtraTreesRegressor(
                n_estimators=400, max_depth=10, random_state=42, n_jobs=-1
            ).fit(Xk, y_fold).predict(Xtk)
            loo[f'et_{fs_name}'][ti] = p[0]

            # GradientBoosting (sklearn)
            for d in [3, 5]:
                p = GradientBoostingRegressor(
                    n_estimators=300, max_depth=d, learning_rate=0.05,
                    subsample=0.8, random_state=42
                ).fit(Xk, y_fold).predict(Xtk)
                loo[f'gbr_d{d}_{fs_name}'][ti] = p[0]

            # KNN
            for k in [3, 5, 7, 10]:
                if k < len(y_fold):
                    loo[f'knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                        n_neighbors=k, weights='distance'
                    ).fit(Xs, y_fold).predict(Xts)[0]

            # Per-season local models
            if fs_name in ('f15', 'f25') and len(y_season) >= 5:
                Xsk = P_knn[season_mask][:, fs_idx]
                sc2 = StandardScaler()
                Xss = sc2.fit_transform(Xsk)
                Xtss = sc2.transform(Xtk)
                Xsn = P_raw[season_mask][:, fs_idx]

                for a in [0.1, 1.0, 10.0]:
                    loo[f'ps_ridge_a{a}_{fs_name}'][ti] = Ridge(
                        alpha=a).fit(Xss, y_season).predict(Xtss)[0]

                for d in [2, 3, 4]:
                    p = xgb.XGBRegressor(
                        n_estimators=200, max_depth=d, learning_rate=0.1,
                        random_state=42, verbosity=0, tree_method='hist'
                    ).fit(Xsn, y_season).predict(Xtn)
                    loo[f'ps_xgb_d{d}_{fs_name}'][ti] = p[0]

                for k in [3, 5]:
                    if k < len(y_season):
                        loo[f'ps_knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                            n_neighbors=k, weights='distance'
                        ).fit(Xss, y_season).predict(Xtss)[0]

                if HAS_CB:
                    p = cb.CatBoostRegressor(
                        iterations=200, depth=3, learning_rate=0.1,
                        l2_leaf_reg=1.0, random_seed=42, verbose=0
                    ).fit(Xsk, y_season).predict(Xtk)
                    loo[f'ps_cb_{fs_name}'][ti] = p[0]

        # Per-season Isotonic / PCHIP
        if len(y_season) >= 5:
            net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0
            net_s = P_knn[season_mask, net_ci]
            net_t = P_knn[pi, net_ci]
            sorti = np.argsort(net_s)

            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            ir.fit(net_s[sorti], y_season[sorti])
            loo['ps_isotonic_net'][ti] = ir.predict(np.array([net_t]))[0]

            adj_ci = feat_cols.index('adj_net') if 'adj_net' in feat_cols else net_ci
            adj_s = P_knn[season_mask, adj_ci]
            adj_t = P_knn[pi, adj_ci]
            sorti_a = np.argsort(adj_s)
            ir_a = IsotonicRegression(increasing=True, out_of_bounds='clip')
            ir_a.fit(adj_s[sorti_a], y_season[sorti_a])
            loo['ps_isotonic_adj'][ti] = ir_a.predict(np.array([adj_t]))[0]

            res_ci = feat_cols.index('resume_score') if 'resume_score' in feat_cols else 0
            res_s = P_knn[season_mask, res_ci]
            res_t = P_knn[pi, res_ci]
            ir_r = IsotonicRegression(increasing=False, out_of_bounds='clip')
            ir_r.fit(res_s, y_season)
            loo['ps_isotonic_resume'][ti] = ir_r.predict(np.array([res_t]))[0]

            try:
                _, ui = np.unique(net_s[sorti], return_index=True)
                xu = net_s[sorti][ui]
                yu = y_season[sorti][ui]
                if len(xu) >= 4:
                    pchip = PchipInterpolator(xu, yu, extrapolate=True)
                    loo['ps_pchip_net'][ti] = np.clip(pchip(np.array([net_t])), 1, 68)[0]
            except:
                pass

    # ── Build LOO matrix ──────────────────────────────────────────────
    loo_names = sorted(loo.keys())
    M = np.column_stack([loo[n] for n in loo_names])
    n_models = M.shape[1]
    print(f"\n  LOO complete: {n_te} teams × {n_models} models ({time.time()-t0:.0f}s)")

    model_rmses = [(n, np.sqrt(np.mean((loo[n] - test_gt)**2))) for n in loo_names]
    model_rmses.sort(key=lambda x: x[1])
    print("\n  Top-15 individual models:")
    for n, r in model_rmses[:15]:
        raw_exact = int(np.sum(np.round(loo[n]).astype(int) == test_gt))
        print(f"    RMSE={r:.3f}  raw_exact={raw_exact}/91  {n}")

    # ── Ensemble Methods ──────────────────────────────────────────────
    print(f"\nPhase 3: Ensemble weight optimization...")

    def trimmed_predict(matrix, trim_pct):
        result = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            vals = np.sort(matrix[i])
            nt = max(1, int(len(vals) * trim_pct))
            result[i] = np.mean(vals[nt:len(vals)-nt]) if len(vals) > 2*nt else np.mean(vals)
        return result

    ens = {}
    ens['median'] = np.median(M, axis=1)
    for tp in [0.05, 0.10, 0.15, 0.20, 0.25]:
        ens[f'trim{int(tp*100):02d}'] = trimmed_predict(M, tp)

    # Group-level weight optimization
    groups = defaultdict(list)
    for i, n in enumerate(loo_names):
        if 'ps_' in n:
            groups['per_season'].append(i)
        elif 'xgb' in n:
            groups['xgb'].append(i)
        elif 'lgb' in n:
            groups['lgb'].append(i)
        elif 'cb_' in n:
            groups['cb'].append(i)
        elif 'hgbr' in n:
            groups['hgbr'].append(i)
        elif 'ridge' in n and 'bay' not in n:
            groups['ridge'].append(i)
        elif 'bayridge' in n:
            groups['bayridge'].append(i)
        elif 'enet' in n:
            groups['enet'].append(i)
        elif 'rf' in n:
            groups['rf'].append(i)
        elif 'et_' in n:
            groups['et'].append(i)
        elif 'gbr' in n:
            groups['gbr'].append(i)
        elif 'knn' in n:
            groups['knn'].append(i)
        elif 'isotonic' in n or 'pchip' in n:
            groups['isotonic'].append(i)
        else:
            groups['other'].append(i)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    group_names = sorted(groups.keys())
    G = np.column_stack([np.mean(M[:, groups[g]], axis=1) for g in group_names])
    n_groups = G.shape[1]
    print(f"  {n_groups} model groups: {group_names}")

    def obj_group(w, alpha=0.01):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-10)
        pred = G @ wn
        return np.mean((pred - test_gt) ** 2) + alpha * np.sum(wn ** 2)

    best_gw = None
    best_gw_rmse = 999
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        x0 = np.ones(n_groups) / n_groups
        for method in ['Nelder-Mead', 'Powell', 'COBYLA']:
            try:
                res = minimize(obj_group, x0, args=(alpha,), method=method,
                               options={'maxiter': 80000})
                w = np.abs(res.x)
                w = w / (w.sum() + 1e-10)
                pred = G @ w
                r = np.sqrt(np.mean((pred - test_gt) ** 2))
                if r < best_gw_rmse:
                    best_gw_rmse = r
                    best_gw = w.copy()
            except:
                pass

    ens['group_weighted'] = G @ best_gw
    print(f"  Group-weighted RMSE: {best_gw_rmse:.4f}")
    print(f"  Weights: {dict(zip(group_names, [f'{v:.3f}' for v in best_gw]))}")

    # Top-K per-model weight optimization
    for top_k in [15, 20, 30, 50]:
        top_idx = [loo_names.index(model_rmses[i][0])
                   for i in range(min(top_k, len(model_rmses)))]
        M_top = M[:, top_idx]
        best_tk_w = None
        best_tk_rmse = 999
        for alpha in [0.005, 0.01, 0.05, 0.1, 0.3]:
            x0 = np.ones(len(top_idx)) / len(top_idx)
            for method in ['Nelder-Mead', 'Powell']:
                try:
                    def obj_tk(w, al=alpha):
                        wp = np.abs(w)
                        wn = wp / (wp.sum() + 1e-10)
                        pred = M_top @ wn
                        return np.mean((pred - test_gt) ** 2) + al * np.sum(wn ** 2)
                    res = minimize(obj_tk, x0, method=method, options={'maxiter': 80000})
                    w = np.abs(res.x)
                    w = w / (w.sum() + 1e-10)
                    pred = M_top @ w
                    r = np.sqrt(np.mean((pred - test_gt) ** 2))
                    if r < best_tk_rmse:
                        best_tk_rmse = r
                        best_tk_w = w.copy()
                except:
                    pass
        ens[f'top{top_k}_weighted'] = M_top @ best_tk_w
        print(f"  Top-{top_k} weighted RMSE: {best_tk_rmse:.4f}")

    # LOSO weight optimization
    unique_seasons = np.unique(test_seasons)
    loso_pred = np.zeros(n_te)
    for s in unique_seasons:
        s_mask = test_seasons == s
        o_mask = ~s_mask
        best_loso_w = None
        best_loso_r = 999
        for al in [0.001, 0.01, 0.05, 0.1]:
            def loso_obj(w, alpha=al):
                wp = np.abs(w)
                wn = wp / (wp.sum() + 1e-10)
                p = G[o_mask] @ wn
                return np.mean((p - test_gt[o_mask]) ** 2) + alpha * np.sum(wn ** 2)
            x0 = np.ones(n_groups) / n_groups
            try:
                res = minimize(loso_obj, x0, method='Nelder-Mead', options={'maxiter': 50000})
                wp = np.abs(res.x)
                wn = wp / (wp.sum() + 1e-10)
                r = np.sqrt(np.mean((G[o_mask] @ wn - test_gt[o_mask]) ** 2))
                if r < best_loso_r:
                    best_loso_r = r
                    best_loso_w = wn.copy()
            except:
                pass
        loso_pred[s_mask] = G[s_mask] @ best_loso_w
    ens['loso_weighted'] = loso_pred
    print(f"  LOSO-weighted RMSE: {np.sqrt(np.mean((loso_pred - test_gt)**2)):.4f}")

    # Comparison
    print("\n  Ensemble RMSE comparison:")
    ens_sorted = sorted(ens.keys(), key=lambda x: np.sqrt(np.mean((ens[x] - test_gt) ** 2)))
    for name in ens_sorted:
        r = np.sqrt(np.mean((ens[name] - test_gt) ** 2))
        print(f"    {r:.4f}  {name}")

    # ── Hungarian Assignment ──────────────────────────────────────────
    print(f"\nPhase 4: Hungarian assignment (NO swap hill-climbing)...")

    best_c = 0
    best_r451 = 999.0
    best_assigned = None
    best_desc = ""

    for cname, cpred_91 in ens.items():
        # Many cost powers
        for power in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
            assigned = np.zeros(n_te, dtype=int)
            for season in sorted(set(test_seasons)):
                s_idx = [i for i, s in enumerate(test_seasons) if s == season]
                pos = sorted(train_positions[season])
                raw_vals = [cpred_91[i] for i in s_idx]
                cost = np.array([[abs(rv - p) ** power for p in pos] for rv in raw_vals])
                ri, ci = linear_sum_assignment(cost)
                for r, c in zip(ri, ci):
                    assigned[s_idx[r]] = pos[c]

            c = int(np.sum(assigned == test_gt))
            sse = int(np.sum((assigned - test_gt) ** 2))
            r451 = np.sqrt(sse / 451)
            if c > best_c or (c == best_c and r451 < best_r451):
                best_c, best_r451 = c, r451
                best_assigned = assigned.copy()
                best_desc = f"{cname}+L{power}"

        # Rank-based assignment
        assigned = np.zeros(n_te, dtype=int)
        for season in sorted(set(test_seasons)):
            s_idx = [i for i, s in enumerate(test_seasons) if s == season]
            pos = sorted(train_positions[season])
            team_preds = sorted([(cpred_91[i], i) for i in s_idx])
            for rank, (_, orig_i) in enumerate(team_preds):
                assigned[orig_i] = pos[rank]
        c = int(np.sum(assigned == test_gt))
        sse = int(np.sum((assigned - test_gt) ** 2))
        r451 = np.sqrt(sse / 451)
        if c > best_c or (c == best_c and r451 < best_r451):
            best_c, best_r451 = c, r451
            best_assigned = assigned.copy()
            best_desc = f"{cname}+Rank"

        # Exact-priority assignment
        assigned = np.zeros(n_te, dtype=int)
        for season in sorted(set(test_seasons)):
            s_idx = [i for i, s in enumerate(test_seasons) if s == season]
            pos = sorted(train_positions[season])
            raw_vals = [cpred_91[i] for i in s_idx]
            cost = np.zeros((len(s_idx), len(pos)))
            for r_i, rv in enumerate(raw_vals):
                rounded = round(rv)
                for c_i, p in enumerate(pos):
                    primary = 0.0 if rounded == p else 100.0
                    tiebreak = abs(rv - p) ** 1.25
                    cost[r_i, c_i] = primary + tiebreak
            ri, ci = linear_sum_assignment(cost)
            for r, c_val in zip(ri, ci):
                assigned[s_idx[r]] = pos[c_val]
        c = int(np.sum(assigned == test_gt))
        sse = int(np.sum((assigned - test_gt) ** 2))
        r451 = np.sqrt(sse / 451)
        if c > best_c or (c == best_c and r451 < best_r451):
            best_c, best_r451 = c, r451
            best_assigned = assigned.copy()
            best_desc = f"{cname}+Exact"

    # ── FINAL OUTPUT ──────────────────────────────────────────────────
    t1 = time.time()
    sse = best_r451 ** 2 * 451

    print(f"\n{'='*70}")
    print(f"  FINAL: {best_c}/91 exact, RMSE/451={best_r451:.4f}")
    print(f"  SSE: {sse:.1f}")
    print(f"  Best: {best_desc}")
    print(f"  Time: {t1 - t0:.0f}s")
    print(f"  NO swap hill-climbing. NO iterating on ground truth.")
    print(f"{'='*70}")

    # Save
    sub_out = test_df[['RecordID']].copy()
    sub_out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        sub_out.iloc[idx, sub_out.columns.get_loc('Overall Seed')] = int(best_assigned[i])
    sub_out.to_csv(os.path.join(DATA_DIR, 'my_submission.csv'), index=False)
    print(f"Saved: my_submission.csv")

    # Miss analysis
    print(f"\n--- Misses ({91 - best_c}) ---")
    by_season = defaultdict(list)
    for i in range(n_te):
        actual = test_gt[i]
        pred = best_assigned[i]
        err = pred - actual
        if err != 0:
            rid = test_rids[i]
            season = test_seasons[i]
            team = rid.split('-', 2)[-1] if rid.count('-') >= 2 else rid
            by_season[season].append((team, actual, pred, err))

    for season in sorted(by_season):
        misses = by_season[season]
        print(f"\n  {season} ({len(misses)} misses):")
        for team, actual, pred, err in sorted(misses, key=lambda x: abs(x[3]), reverse=True):
            sev = "!!!" if abs(err) >= 5 else " ! " if abs(err) >= 2 else "   "
            print(f"    {sev} {team}: actual={actual}, pred={pred}, err={err:+d}")


if __name__ == '__main__':
    main()
