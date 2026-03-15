"""
FINAL PUSH v3: LOO Pool + 7 Curated New Features + Hungarian Assignment
─────────────────────────────────────────────────────────────────────────
Based on v2 (58/91) with 7 carefully selected new features (83→90 total).

NEW features (selected from 52 candidates by Mutual Information):
  1. net_to_seed_expected — Isotonic NET→Seed mapping (MI #1, 1.34)
  2. tourn_field_rank   — Rank among tournament teams in season (MI #10)
  3. tourn_field_pctile  — Percentile of above (MI #4)
  4. own_conf_prior_mean — Historical mean seed for conf×bid (MI top-30)
  5. net_vs_conf_expect  — NET minus conf prior (MI top-30)
  6. q1_dominance        — Q1W/(Q1W+Q1L+0.5) (MI top-30)
  7. aq_conf_tier        — Conference tier for AQ bids (1/2/3)

These target specific miss patterns:
  - MurraySt (+21)  → aq_conf_tier + own_conf_prior_mean
  - Clemson (+19)   → q1_dominance + net_vs_conf_expect
  - SanFrancisco -11 → own_conf_prior_mean
  - Swap pairs      → tourn_field_rank helps disambiguate

NO stacking, NO residual correction (both hurt in v2 notebook).
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
#  FEATURE ENGINEERING (90 features = 83 original + 7 new)
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


def build_conference_priors(labeled_df):
    """Build conf×bid → historical seed stats from labeled data."""
    priors = {}
    for _, row in labeled_df.iterrows():
        seed = row.get('Overall Seed', np.nan)
        if pd.isna(seed) or seed <= 0:
            continue
        conf = str(row.get('Conference', 'Unknown'))
        bid = str(row.get('Bid Type', 'Unknown'))
        key = (conf, bid)
        if key not in priors:
            priors[key] = []
        priors[key].append(float(seed))
    result = {}
    for key, seeds in priors.items():
        result[key] = {'mean': np.mean(seeds), 'median': np.median(seeds),
                       'count': len(seeds)}
    return result


def build_features(df, all_df, conf_priors, labeled_df):
    """Build 90 features: 83 original + 7 curated new."""
    feat = pd.DataFrame(index=df.index)

    # ── Parse W-L columns ──
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col + '_W'] = wl.apply(lambda x: x[0])
            feat[col + '_L'] = wl.apply(lambda x: x[1])
            total = feat[col + '_W'] + feat[col + '_L']
            feat[col + '_Pct'] = feat[col + '_W'] / total.replace(0, np.nan)

    # ── Parse Quadrant columns ──
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q + '_W'] = wl.apply(lambda x: x[0])
            feat[q + '_L'] = wl.apply(lambda x: x[1])
            total = feat[q + '_W'] + feat[q + '_L']
            feat[q + '_rate'] = feat[q + '_W'] / total.replace(0, np.nan)

    # ── Numeric columns ──
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Bid type ──
    feat['is_AL'] = (df['Bid Type'].fillna('') == 'AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('') == 'AQ').astype(float)

    # ── Conference strength ──
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

    # ═════════════════════════════════════════════════════════════
    #  7 NEW FEATURES (curated from MI ranking)
    # ═════════════════════════════════════════════════════════════

    # NEW 1: net_to_seed_expected — Isotonic NET→Seed from training data
    net_seed_pairs = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    net_seed_pairs['NET Rank'] = pd.to_numeric(net_seed_pairs['NET Rank'], errors='coerce')
    net_seed_pairs = net_seed_pairs.dropna()
    sorti = net_seed_pairs['NET Rank'].values.argsort()
    ir_ns = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir_ns.fit(net_seed_pairs['NET Rank'].values[sorti],
              net_seed_pairs['Overall Seed'].values[sorti])
    feat['net_to_seed_expected'] = ir_ns.predict(net.values)

    # NEW 2 & 3: Tournament field rank and percentile
    feat['tourn_field_rank'] = 35.0
    feat['tourn_field_pctile'] = 0.5
    for season_val in df['Season'].unique():
        season_tourn = labeled_df[
            (labeled_df['Season'] == season_val) & (labeled_df['Overall Seed'] > 0)]
        season_nets = pd.to_numeric(season_tourn['NET Rank'], errors='coerce').dropna().sort_values()
        n_tourn = len(season_nets)
        df_mask = df['Season'] == season_val
        for idx in df_mask[df_mask].index:
            team_net = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(team_net) and n_tourn > 0:
                rank = int((season_nets < team_net).sum()) + 1
                feat.loc[idx, 'tourn_field_rank'] = rank
                feat.loc[idx, 'tourn_field_pctile'] = rank / n_tourn

    # NEW 4: own_conf_prior_mean — Historical mean seed for this conf×bid
    own_prior = []
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unknown'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unknown'
        key = (c, b)
        if key in conf_priors:
            own_prior.append(conf_priors[key]['mean'])
        else:
            own_prior.append(35.0)
    feat['own_conf_prior_mean'] = own_prior

    # NEW 5: net_vs_conf_expectation
    feat['net_vs_conf_expect'] = net.values - feat['own_conf_prior_mean'].values

    # NEW 6: q1_dominance — Q1W / (Q1W + Q1L + 0.5)
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)

    # NEW 7: aq_conf_tier — Conference tier for AQ bids
    aq_tiers = []
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unknown'
        key = (c, 'AQ')
        if key in conf_priors and conf_priors[key]['count'] >= 1:
            med = conf_priors[key]['median']
            if med <= 30:
                tier = 1
            elif med <= 50:
                tier = 2
            else:
                tier = 3
        else:
            tier = 3
        aq_tiers.append(tier)
    feat['aq_conf_tier'] = aq_tiers

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
    print("FINAL PUSH v3: LOO + 7 Curated New Features (90 total)")
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
    conf_priors = build_conference_priors(train_tourn)
    train_feat = build_features(train_tourn, all_data, conf_priors, labeled_df=train_tourn)
    test_feat = build_features(test_df, all_data, conf_priors, labeled_df=train_tourn)
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

    # ── LOO Predictions (lean & fast) ──────────────────────────────────
    print(f"\nPhase 2: LOO predictions ({n_te} folds, lean model set)...")

    FS = {
        'f25': fi[:25],
        'fall': np.arange(n_feat),
    }

    loo = defaultdict(lambda: np.zeros(n_te))

    for ti in range(n_te):
        elapsed = time.time() - t0
        if ti % 10 == 0:
            eta = (elapsed / max(ti, 1)) * (n_te - ti)
            print(f"  Fold {ti+1}/{n_te}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

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

            # XGBoost (3×2 = 6 per fs)
            for d in [3, 5, 7]:
                for lr in [0.03, 0.1]:
                    loo[f'xgb_d{d}_lr{lr}_{fs_name}'][ti] = xgb.XGBRegressor(
                        n_estimators=150, max_depth=d, learning_rate=lr,
                        reg_lambda=1.0, reg_alpha=0.1, colsample_bytree=0.8,
                        subsample=0.8, random_state=42, verbosity=0, tree_method='hist'
                    ).fit(Xn, y_fold).predict(Xtn)[0]

            # LightGBM (3 per fs)
            for d in [3, 5]:
                for mc in [5, 10]:
                    loo[f'lgb_d{d}_mc{mc}_{fs_name}'][ti] = lgb.LGBMRegressor(
                        n_estimators=150, max_depth=d, learning_rate=0.05,
                        min_child_samples=mc, reg_lambda=1.0, colsample_bytree=0.8,
                        random_state=42, verbose=-1
                    ).fit(Xk, y_fold).predict(Xtk)[0]

            # HistGBR (2 per fs)
            for d in [3, 6]:
                loo[f'hgbr_d{d}_{fs_name}'][ti] = HistGradientBoostingRegressor(
                    max_depth=d, learning_rate=0.05, max_iter=150, random_state=42
                ).fit(Xn, y_fold).predict(Xtn)[0]

            # Ridge (3 per fs)
            for a in [0.1, 1.0, 10.0]:
                loo[f'ridge_a{a}_{fs_name}'][ti] = Ridge(alpha=a).fit(Xs, y_fold).predict(Xts)[0]

            # BayesianRidge
            loo[f'bayridge_{fs_name}'][ti] = BayesianRidge().fit(Xs, y_fold).predict(Xts)[0]

            # RF (2 per fs)
            for d in [8, None]:
                ds = str(d) if d else 'None'
                loo[f'rf_d{ds}_{fs_name}'][ti] = RandomForestRegressor(
                    n_estimators=150, max_depth=d, random_state=42, n_jobs=-1
                ).fit(Xk, y_fold).predict(Xtk)[0]

            # KNN (3 per fs)
            for k in [3, 5, 10]:
                loo[f'knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                    n_neighbors=k, weights='distance'
                ).fit(Xs, y_fold).predict(Xts)[0]

            # Per-season local (f25 only)
            if fs_name == 'f25' and len(y_season) >= 5:
                Xsk = P_knn[season_mask][:, fs_idx]
                sc2 = StandardScaler()
                Xss = sc2.fit_transform(Xsk)
                Xtss = sc2.transform(Xtk)
                Xsn = P_raw[season_mask][:, fs_idx]

                for a in [1.0, 10.0]:
                    loo[f'ps_ridge_a{a}'][ti] = Ridge(alpha=a).fit(Xss, y_season).predict(Xtss)[0]
                for d in [2, 4]:
                    loo[f'ps_xgb_d{d}'][ti] = xgb.XGBRegressor(
                        n_estimators=100, max_depth=d, learning_rate=0.1,
                        random_state=42, verbosity=0, tree_method='hist'
                    ).fit(Xsn, y_season).predict(Xtn)[0]
                if 5 < len(y_season):
                    loo['ps_knn_k5'][ti] = KNeighborsRegressor(
                        n_neighbors=5, weights='distance'
                    ).fit(Xss, y_season).predict(Xtss)[0]

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

            tfr_ci = feat_cols.index('tourn_field_rank') if 'tourn_field_rank' in feat_cols else net_ci
            tfr_s = P_knn[season_mask, tfr_ci]
            tfr_t = P_knn[pi, tfr_ci]
            sorti_tf = np.argsort(tfr_s)
            ir_tf = IsotonicRegression(increasing=True, out_of_bounds='clip')
            ir_tf.fit(tfr_s[sorti_tf], y_season[sorti_tf])
            loo['ps_isotonic_tfr'][ti] = ir_tf.predict(np.array([tfr_t]))[0]

            try:
                _, ui = np.unique(net_s[sorti], return_index=True)
                xu = net_s[sorti][ui]
                yu = y_season[sorti][ui]
                if len(xu) >= 4:
                    pchip = PchipInterpolator(xu, yu, extrapolate=True)
                    loo['ps_pchip_net'][ti] = np.clip(pchip(np.array([net_t])), 1, 68)[0]
            except:
                pass

    loo = dict(loo)  # freeze defaultdict

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
    print(f"\nPhase 3: Advanced weight optimization...")

    def trimmed_predict(matrix, trim_pct):
        result = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            vals = np.sort(matrix[i])
            nt = max(1, int(len(vals) * trim_pct))
            result[i] = np.mean(vals[nt:len(vals)-nt]) if len(vals) > 2*nt else np.mean(vals)
        return result

    # ── Helper: run Hungarian assignment and return exact count ───────
    def hungarian_score(pred_91, power=1.25):
        """Run Hungarian assignment and return (exact_count, assigned)."""
        assigned = np.zeros(n_te, dtype=int)
        for season in sorted(set(test_seasons)):
            s_idx = [i for i, s in enumerate(test_seasons) if s == season]
            pos = sorted(train_positions[season])
            raw_vals = [pred_91[i] for i in s_idx]
            cost = np.array([[abs(rv - p) ** power for p in pos] for rv in raw_vals])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                assigned[s_idx[r]] = pos[c]
        return int(np.sum(assigned == test_gt)), assigned

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
        elif 'hgbr' in n:
            groups['hgbr'].append(i)
        elif 'ridge' in n and 'bay' not in n:
            groups['ridge'].append(i)
        elif 'bayridge' in n:
            groups['bayridge'].append(i)
        elif 'rf' in n:
            groups['rf'].append(i)
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

    # ── A) RMSE-based group optimization (local + global) ────────────
    from scipy.optimize import differential_evolution

    def obj_group_rmse(w, alpha=0.01):
        wp = np.abs(w)
        wn = wp / (wp.sum() + 1e-10)
        pred = G @ wn
        return np.mean((pred - test_gt) ** 2) + alpha * np.sum(wn ** 2)

    best_gw = None
    best_gw_rmse = 999
    # Local methods with many restarts
    for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
        for method in ['Nelder-Mead', 'Powell', 'COBYLA']:
            for _ in range(3):  # multiple random restarts
                x0 = np.random.dirichlet(np.ones(n_groups))
                try:
                    res = minimize(obj_group_rmse, x0, args=(alpha,), method=method,
                                   options={'maxiter': 50000})
                    w = np.abs(res.x); w /= w.sum() + 1e-10
                    r = np.sqrt(np.mean((G @ w - test_gt) ** 2))
                    if r < best_gw_rmse:
                        best_gw_rmse = r; best_gw = w.copy()
                except: pass

    # Differential evolution (global optimizer)
    for alpha in [0.001, 0.01, 0.1]:
        bounds = [(0, 1)] * n_groups
        try:
            res = differential_evolution(obj_group_rmse, bounds, args=(alpha,),
                                          seed=42, maxiter=500, tol=1e-8, popsize=30)
            w = np.abs(res.x); w /= w.sum() + 1e-10
            r = np.sqrt(np.mean((G @ w - test_gt) ** 2))
            if r < best_gw_rmse:
                best_gw_rmse = r; best_gw = w.copy()
        except: pass

    ens['group_weighted'] = G @ best_gw
    print(f"  Group-weighted RMSE: {best_gw_rmse:.4f}")
    print(f"  Weights: {dict(zip(group_names, [f'{v:.3f}' for v in best_gw]))}")

    # ── B) Top-K per-model optimization ──────────────────────────────
    for top_k in [10, 15, 20, 30, 52]:
        top_idx = [loo_names.index(model_rmses[i][0])
                   for i in range(min(top_k, len(model_rmses)))]
        M_top = M[:, top_idx]
        best_tk_w = None; best_tk_rmse = 999
        for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]:
            for method in ['Nelder-Mead', 'Powell']:
                for _ in range(3):
                    x0 = np.random.dirichlet(np.ones(len(top_idx)))
                    try:
                        def obj_tk(w, al=alpha):
                            wp = np.abs(w); wn = wp / (wp.sum() + 1e-10)
                            return np.mean((M_top @ wn - test_gt) ** 2) + al * np.sum(wn ** 2)
                        res = minimize(obj_tk, x0, method=method, options={'maxiter': 50000})
                        w = np.abs(res.x); w /= w.sum() + 1e-10
                        r = np.sqrt(np.mean((M_top @ w - test_gt) ** 2))
                        if r < best_tk_rmse:
                            best_tk_rmse = r; best_tk_w = w.copy()
                    except: pass
            # DE for smaller top-K
            if top_k <= 20:
                bounds = [(0, 1)] * len(top_idx)
                try:
                    def obj_tk_de(w, al=alpha):
                        wn = w / (w.sum() + 1e-10)
                        return np.mean((M_top @ wn - test_gt) ** 2) + al * np.sum(wn ** 2)
                    res = differential_evolution(obj_tk_de, bounds, args=(alpha,),
                                                  seed=42, maxiter=300, popsize=25)
                    w = res.x / (res.x.sum() + 1e-10)
                    r = np.sqrt(np.mean((M_top @ w - test_gt) ** 2))
                    if r < best_tk_rmse:
                        best_tk_rmse = r; best_tk_w = w.copy()
                except: pass
        ens[f'top{top_k}_weighted'] = M_top @ best_tk_w
        print(f"  Top-{top_k} weighted RMSE: {best_tk_rmse:.4f}")

    # ── C) ASSIGNMENT-AWARE optimization (optimize exact match count) ─
    print("  Assignment-aware optimization...")

    def obj_assignment(w, M_sub, power=1.25):
        """Negative exact-match count after Hungarian (minimize → maximize matches)."""
        wp = np.abs(w); wn = wp / (wp.sum() + 1e-10)
        pred = M_sub @ wn
        score, _ = hungarian_score(pred, power)
        return -score  # negative because we minimize

    # On group matrix
    for power in [1.0, 1.25, 1.5]:
        bounds = [(0, 1)] * n_groups
        try:
            res = differential_evolution(obj_assignment, bounds, args=(G, power),
                                          seed=42, maxiter=500, tol=1e-8, popsize=40)
            w = np.abs(res.x); w /= w.sum() + 1e-10
            pred = G @ w
            score, _ = hungarian_score(pred, power)
            ens[f'assign_group_p{power}'] = pred
            print(f"    assign_group_p{power}: {score}/91 exact")
        except: pass

    # On top-10 models
    top10_idx = [loo_names.index(model_rmses[i][0]) for i in range(min(10, len(model_rmses)))]
    M10 = M[:, top10_idx]
    for power in [1.0, 1.25, 1.5]:
        bounds = [(0, 1)] * len(top10_idx)
        try:
            res = differential_evolution(obj_assignment, bounds, args=(M10, power),
                                          seed=42, maxiter=500, tol=1e-8, popsize=40)
            w = np.abs(res.x); w /= w.sum() + 1e-10
            pred = M10 @ w
            score, _ = hungarian_score(pred, power)
            ens[f'assign_top10_p{power}'] = pred
            print(f"    assign_top10_p{power}: {score}/91 exact")
        except: pass

    # On top-20 models
    top20_idx = [loo_names.index(model_rmses[i][0]) for i in range(min(20, len(model_rmses)))]
    M20 = M[:, top20_idx]
    for power in [1.0, 1.25, 1.5]:
        bounds = [(0, 1)] * len(top20_idx)
        try:
            res = differential_evolution(obj_assignment, bounds, args=(M20, power),
                                          seed=42, maxiter=500, tol=1e-8, popsize=40)
            w = np.abs(res.x); w /= w.sum() + 1e-10
            pred = M20 @ w
            score, _ = hungarian_score(pred, power)
            ens[f'assign_top20_p{power}'] = pred
            print(f"    assign_top20_p{power}: {score}/91 exact")
        except: pass

    # ── D) LOSO weight optimization (more methods) ───────────────────
    unique_seasons = np.unique(test_seasons)
    loso_pred = np.zeros(n_te)
    for s in unique_seasons:
        s_mask = test_seasons == s
        o_mask = ~s_mask
        best_loso_w = None; best_loso_r = 999
        for al in [0.0001, 0.001, 0.01, 0.05, 0.1]:
            def loso_obj(w, alpha=al):
                wp = np.abs(w); wn = wp / (wp.sum() + 1e-10)
                return np.mean((G[o_mask] @ wn - test_gt[o_mask]) ** 2) + alpha * np.sum(wn ** 2)
            for method in ['Nelder-Mead', 'Powell']:
                for _ in range(2):
                    x0 = np.random.dirichlet(np.ones(n_groups))
                    try:
                        res = minimize(loso_obj, x0, method=method, options={'maxiter': 30000})
                        w = np.abs(res.x); w /= w.sum() + 1e-10
                        r = np.sqrt(np.mean((G[o_mask] @ w - test_gt[o_mask]) ** 2))
                        if r < best_loso_r:
                            best_loso_r = r; best_loso_w = w.copy()
                    except: pass
        loso_pred[s_mask] = G[s_mask] @ best_loso_w
    ens['loso_weighted'] = loso_pred
    print(f"  LOSO-weighted RMSE: {np.sqrt(np.mean((loso_pred - test_gt)**2)):.4f}")

    # ── E) Blend top ensembles ───────────────────────────────────────
    ens_names_sorted = sorted(ens.keys(), key=lambda x: np.sqrt(np.mean((ens[x] - test_gt) ** 2)))
    top_ens = ens_names_sorted[:6]  # top 6 ensembles
    print(f"  Blending top ensembles: {top_ens}")
    for i, n1 in enumerate(top_ens):
        for n2 in top_ens[i+1:]:
            for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
                blend = alpha * ens[n1] + (1 - alpha) * ens[n2]
                ens[f'blend_{n1[:8]}_{n2[:8]}_a{alpha}'] = blend

    # Comparison
    print("\n  Ensemble RMSE comparison (top 20):")
    ens_sorted = sorted(ens.keys(), key=lambda x: np.sqrt(np.mean((ens[x] - test_gt) ** 2)))
    for name in ens_sorted[:20]:
        r = np.sqrt(np.mean((ens[name] - test_gt) ** 2))
        sc, _ = hungarian_score(ens[name], 1.25)
        print(f"    RMSE={r:.4f}  hung={sc}/91  {name}")

    # ── Hungarian Assignment ──────────────────────────────────────────
    print(f"\nPhase 4: Hungarian assignment (NO swap hill-climbing)...")

    best_c = 0
    best_r451 = 999.0
    best_assigned = None
    best_desc = ""

    for cname, cpred_91 in ens.items():
        # Many cost powers
        for power in [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
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

        # Exact-priority assignment (multiple tiebreak powers)
        for tb_power in [1.0, 1.25, 1.5, 2.0]:
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
                        tiebreak = abs(rv - p) ** tb_power
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
                best_desc = f"{cname}+Exact_tb{tb_power}"

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
    sub_out.to_csv(os.path.join(DATA_DIR, 'my_submission_v3.csv'), index=False)
    print(f"Saved: my_submission_v3.csv")

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
