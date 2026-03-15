#!/usr/bin/env python3
"""
NCAA v10 — Committee-Style Feature Engineering
================================================
The selection committee uses specific criteria that our current 68 features
only partially capture. This version adds committee-mirroring features using
TWO UNUSED DATA COLUMNS (AvgOppNET, NETNonConfSOS) plus better composites.

Key additions:
  1. AvgOppNET (raw column, never used before)
  2. NETNonConfSOS (raw column, never used before)
  3. Strength of Record proxy (wins weighted by opponent quality)
  4. Non-conference schedule quality metrics
  5. Conference tournament density (how many teams make the field)
  6. Committee-weighted quadrant score (mirroring actual committee formula)
  7. NET efficiency gap (NET vs AvgOppNET)
  8. Q4 dominance (margin proxy — committee expects ~100% Q4 wins)
  9. Resume consistency (variance of performance across quadrants)
 10. Conference rank within tournament teams (where is team in its conf peers)

Evaluation: both Kaggle-style (locked seeds) and LOSO.
v6 baseline: local=56/91 RMSE=2.474, LOSO=3.678
"""

import os, sys, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA_DIR)
from ncaa_2026_model import (
    load_data, parse_wl, build_pairwise_data, pairwise_score,
    hungarian, select_top_k_features
)


# =================================================================
#  ENHANCED FEATURE ENGINEERING (68 original + ~20 new committee features)
# =================================================================
def build_committee_features(df, context_df, labeled_df, tourn_rids):
    """
    Build enhanced feature set that mirrors committee evaluation criteria.
    Includes all 68 original features PLUS new committee-specific ones.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Win-loss records (same as v6) ──
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # ── Quadrant records (same as v6) ──
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    # ── Core rankings (same as v6) ──
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev
    feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp

    # ══════════════════════════════════════════════════════
    #  NEW: USE PREVIOUSLY UNUSED RAW COLUMNS
    # ══════════════════════════════════════════════════════

    # AvgOppNET — average opponent NET ranking (different from AvgOppNETRank!)
    avg_opp_net = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    feat['AvgOppNET'] = avg_opp_net

    # NETNonConfSOS — non-conference strength of schedule
    nc_sos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    feat['NETNonConfSOS'] = nc_sos

    # ── Bid type (same as v6) ──
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # ── Conference stats (same as v6) ──
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(context_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': context_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300)
    feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count']   = conf.map(cs.count()).fillna(1)
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power).astype(float)
    cav = feat['conf_avg_net']

    # ── NET → Seed estimate (same as v6) ──
    net_rank = net.rank(method='min')
    n_teams = len(net)
    feat['net_to_seed'] = 1 + (net_rank - 1) * (67 / max(n_teams - 1, 1))

    # ── Transforms (same as v6) ──
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # ── Composites (same as v6) ──
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    # ── Bid interactions (same as v6) ──
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # ── Resume quality (same as v6) ──
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w
    feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + q2l + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1)
    feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    # ── Tournament field rank (same as v6) ──
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in context_df[context_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids
                       and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)

    # ── AL rank (same as v6) ──
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in context_df[context_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL'
                          and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n):
                feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)

    # ── Historical conference-bid seed distributions (same as v6) ──
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk')))
        cb.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb.get((c, b), [])
        feat.loc[idx, 'cb_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'cb_median_seed'] = np.median(vals) if vals else 35.0

    feat['net_vs_conf'] = net / (cav + 1)

    # ── Season percentiles (same as v6) ──
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)

    # ══════════════════════════════════════════════════════
    #  NEW COMMITTEE-STYLE FEATURES
    # ══════════════════════════════════════════════════════

    # --- 1. NET Efficiency Gap ---
    # How much better is a team than its average opponent?
    # Strong teams have NET << AvgOppNET (they're ranked much higher than their opponents)
    feat['net_efficiency_gap'] = avg_opp_net - net  # positive = better than opponents
    feat['net_opp_ratio'] = net / (avg_opp_net + 1)  # < 1 means better than opponents

    # --- 2. Non-Conference Schedule Quality ---
    # Committee values strong non-conference scheduling
    feat['nonconf_sos_diff'] = nc_sos - sos  # positive = weaker non-conf than overall
    feat['nonconf_sos_ratio'] = nc_sos / (sos + 1)  # >1 = weaker non-conf schedule
    feat['nonconf_quality'] = (300 - nc_sos) / 300  # normalized quality (higher = better)
    nc_pct = feat.get('Non-ConferenceRecord_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['nonconf_performance'] = nc_pct * (300 - nc_sos) / 200  # W% adjusted by SOS

    # --- 3. Committee-Weighted Quadrant Score ---
    # The committee formula (approximation):
    # Heavy premium on Q1 wins, slight credit for Q2, severe penalty for Q3/Q4 losses
    feat['committee_score'] = (q1w * 5.0 + q2w * 2.5 + q3w * 0.5 + q4w * 0.1
                                - q1l * 0.5 - q2l * 1.5 - q3l * 4.0 - q4l * 6.0)

    # --- 4. Q4 Dominance (margin proxy) ---
    # Top seeds should beat all Q4 teams. Losses here are catastrophic.
    q4_total = q4w + q4l
    feat['q4_domination'] = np.where(q4_total > 0, q4w / q4_total, 1.0)

    # Q3 win rate (committee also expects no Q3 losses)
    q3_total = q3w + q3l
    feat['q3_win_rate'] = np.where(q3_total > 0, q3w / q3_total, 1.0)

    # --- 5. Strength of Record Proxy ---
    # Value of wins weighted by opponent quality
    # More Q1/Q2 wins with lower SOS → higher strength of record
    total_w = feat.get('total_W', pd.Series(0, index=df.index)).fillna(0)
    feat['strength_of_record'] = (q1w * 4 + q2w * 3 + q3w * 1.5 + q4w * 0.5) / (total_w + 1)

    # Win% vs expected (based on SOS): if SOS is hard, same W% is more impressive
    expected_wpct = 1.0 - (sos / 400)  # rough estimate: harder SOS → lower expected W%
    feat['wins_above_expected'] = wpct - expected_wpct

    # --- 6. Conference Tournament Density ---
    # How many teams from this conference are in the tournament field?
    # More tournament teams = stronger conference = committee bump
    conf_tourn = {}
    for sv in df['Season'].unique():
        for c in df[df['Season']==sv]['Conference'].unique():
            if pd.notna(c):
                count = sum(1 for _, r in labeled_df.iterrows()
                          if r['Season'] == sv and str(r.get('Conference', '')) == str(c)
                          and r.get('Overall Seed', 0) > 0)
                conf_tourn[(sv, str(c))] = count
    feat['conf_tourn_teams'] = 0.0
    for idx in df.index:
        sv, c = df.loc[idx, 'Season'], str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        feat.loc[idx, 'conf_tourn_teams'] = float(conf_tourn.get((sv, c), 0))

    # Team's rank within its conference's tournament teams
    feat['conf_internal_rank'] = 0.5
    for sv in df['Season'].unique():
        for c in df[df['Season']==sv]['Conference'].dropna().unique():
            mask = (df['Season'] == sv) & (df['Conference'] == c)
            if mask.sum() > 1:
                feat.loc[mask, 'conf_internal_rank'] = net[mask].rank(pct=True)

    # --- 7. Resume Consistency ---
    # Teams with balanced performance across quadrants are viewed more favorably
    q_pcts = []
    q_pcts.append(np.where(q1w + q1l > 0, q1w / (q1w + q1l), 0.5))
    q_pcts.append(np.where(q2w + q2l > 0, q2w / (q2w + q2l), 0.5))
    q_pcts.append(np.where(q3w + q3l > 0, q3w / (q3w + q3l), 1.0))
    q_pcts.append(np.where(q4w + q4l > 0, q4w / (q4w + q4l), 1.0))
    q_array = np.column_stack(q_pcts)  # (n_teams, 4)
    feat['quad_consistency'] = 1.0 - np.std(q_array, axis=1)  # higher = more consistent

    # --- 8. NET Trajectory ---
    # Improvement from previous NET (momentum matters to committee)
    feat['net_improvement_pct'] = (prev - net) / (prev + 1)  # positive = improved

    # --- 9. Total Quality Games ---
    # How many Q1+Q2 games did they play? (more = harder schedule)
    feat['quality_game_count'] = q1w + q1l + q2w + q2l

    # --- 10. Road Performance Index ---
    # Committee heavily values road performance
    road_w_l = df['RoadWL'].apply(parse_wl) if 'RoadWL' in df.columns else pd.Series([(0,0)]*len(df))
    road_w = road_w_l.apply(lambda x: x[0]).fillna(0)
    road_l = road_w_l.apply(lambda x: x[1]).fillna(0)
    feat['road_win_count'] = road_w
    feat['road_game_pct'] = (road_w + road_l) / (feat.get('total_games', pd.Series(30, index=df.index)).fillna(30) + 0.1)

    # --- 11. Bad Loss Severity ---
    # Not just count but how bad: Q4 losses are worse than Q3
    feat['bad_loss_severity'] = q3l * 1.0 + q4l * 3.0  # weighted severity

    # --- 12. Conference-Adjusted NET ---
    # Team's NET adjusted for how strong its conference is
    feat['conf_adjusted_net'] = net - (200 - feat['conf_avg_net']) * 0.3

    # --- 13. AvgOppNET interactions ---
    feat['opp_net_vs_rank'] = avg_opp_net - opp  # gap between two opponent metrics
    feat['schedule_difficulty'] = (300 - avg_opp_net) * wpct  # difficulty × results

    return feat


# =================================================================
#  EVALUATION FUNCTIONS
# =================================================================
def compute_loso_score(fold_rmses):
    return np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)


def kaggle_eval(predict_fn, X_all, y, seasons, folds, test_mask, power=0.15):
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        smask = (seasons == hold)
        test_in_season = test_mask & smask
        if test_in_season.sum() == 0:
            continue
        train_mask = ~test_in_season
        X_season = X_all[smask]
        scores = predict_fn(X_all[train_mask], y[train_mask], X_season, seasons[train_mask])
        season_idx = np.where(smask)[0]
        for i, gi in enumerate(season_idx):
            if not test_mask[gi]:
                scores[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[smask], avail, power=power)
        for i, gi in enumerate(season_idx):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    exact = int((pred == gt).sum())
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return exact, rmse


def loso_eval(predict_fn, X_all, y, seasons, folds, power=0.15):
    fold_rmses = []
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        scores = predict_fn(X_all[tr], y[tr], X_all[te], seasons[tr])
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[te], avail, power=power)
        rmse = np.sqrt(np.mean((assigned - y[te].astype(int)) ** 2))
        fold_rmses.append(rmse)
    return compute_loso_score(fold_rmses), fold_rmses


# =================================================================
#  PREDICTION FUNCTIONS
# =================================================================
def make_v6_predict(feature_names):
    """Exact v6 architecture with whatever feature set is provided."""
    def v6_predict(X_tr, y_tr, X_te, s_tr):
        top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]

        pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        s1 = pairwise_score(lr1, X_te, sc)

        X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sck = StandardScaler()
        pw_Xk_sc = sck.fit_transform(pw_Xk)
        lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_Xk_sc, pw_yk)
        s3 = pairwise_score(lr3, X_te_k, sck)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            reg_alpha=1.0, min_child_weight=5, random_state=42,
            verbosity=0, use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        s4 = pairwise_score(xgb_clf, X_te, sc)

        return 0.64 * s1 + 0.28 * s3 + 0.08 * s4
    return v6_predict


def make_committee_predict(feature_names, top_k=25, c1=5.0, c3=0.5,
                            w1=0.64, w3=0.28, w4=0.08):
    """v6-architecture with configurable params, for use with enhanced features."""
    def predict(X_tr, y_tr, X_te, s_tr):
        top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=top_k)[0]

        pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        s1 = pairwise_score(lr1, X_te, sc)

        X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sck = StandardScaler()
        pw_Xk_sc = sck.fit_transform(pw_Xk)
        lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_Xk_sc, pw_yk)
        s3 = pairwise_score(lr3, X_te_k, sck)

        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            reg_alpha=1.0, min_child_weight=5, random_state=42,
            verbosity=0, use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        s4 = pairwise_score(xgb_clf, X_te, sc)

        return w1 * s1 + w3 * s3 + w4 * s4
    return predict


def main():
    t0 = time.time()
    print('=' * 60)
    print(' NCAA v10 — COMMITTEE-STYLE FEATURE ENGINEERING')
    print('=' * 60)

    # ── Load data ──
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    # Test mask (Kaggle GT teams)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # ── Build ORIGINAL features (v6 baseline) ──
    from ncaa_2026_model import build_features as build_features_v6
    feat_orig = build_features_v6(labeled, context_df, labeled, tourn_rids)
    feat_orig_names = list(feat_orig.columns)
    X_orig_raw = np.where(np.isinf(feat_orig.values.astype(np.float64)), np.nan,
                          feat_orig.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_orig = imp.fit_transform(X_orig_raw)

    print(f'  Original features: {len(feat_orig_names)}')

    # ── Build ENHANCED features ──
    feat_new = build_committee_features(labeled, context_df, labeled, tourn_rids)
    feat_new_names = list(feat_new.columns)
    X_new_raw = np.where(np.isinf(feat_new.values.astype(np.float64)), np.nan,
                         feat_new.values.astype(np.float64))
    imp2 = KNNImputer(n_neighbors=10, weights='distance')
    X_new = imp2.fit_transform(X_new_raw)

    new_features = [f for f in feat_new_names if f not in feat_orig_names]
    print(f'  Enhanced features: {len(feat_new_names)}')
    print(f'  New committee features ({len(new_features)}):')
    for f in new_features:
        print(f'    + {f}')

    print(f'\n  {len(labeled)} teams, {len(folds)} folds')
    print(f'  Test teams: {test_mask.sum()}')

    results = []

    # ════════════════════════════════════════════════════
    #  BASELINE: v6 with original 68 features
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' BASELINE: v6 (68 original features)')
    print('=' * 60)

    pred_v6 = make_v6_predict(feat_orig_names)
    v6_exact, v6_rmse = kaggle_eval(pred_v6, X_orig, y, seasons, folds, test_mask)
    v6_loso, v6_fr = loso_eval(pred_v6, X_orig, y, seasons, folds)
    print(f'  Kaggle: {v6_exact}/91 exact, RMSE={v6_rmse:.4f}')
    print(f'  LOSO:   {v6_loso:.4f}')
    results.append(('v6-baseline-68feat', v6_exact, v6_rmse, v6_loso))

    # ════════════════════════════════════════════════════
    #  TEST 1: v6 architecture with enhanced features
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' TEST 1: v6 architecture + enhanced features (same weights/C)')
    print('=' * 60)

    for top_k in [25, 30, 35, 40]:
        pred_fn = make_committee_predict(feat_new_names, top_k=top_k)
        ex, rmse = kaggle_eval(pred_fn, X_new, y, seasons, folds, test_mask)
        loso, fr = loso_eval(pred_fn, X_new, y, seasons, folds)
        tag = f'v6+committee-k{top_k}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  TEST 2: Try different C values with enhanced features
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' TEST 2: Enhanced features with different C values')
    print('=' * 60)

    for c1, c3 in [(5.0, 0.5), (3.0, 0.3), (10.0, 1.0), (5.0, 1.0), (2.0, 0.5)]:
        pred_fn = make_committee_predict(feat_new_names, top_k=25, c1=c1, c3=c3)
        ex, rmse = kaggle_eval(pred_fn, X_new, y, seasons, folds, test_mask)
        loso, fr = loso_eval(pred_fn, X_new, y, seasons, folds)
        tag = f'comm-C{c1}/C{c3}-k25'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  TEST 3: Which new features matter most?
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' TEST 3: Feature importance analysis (new features)')
    print('=' * 60)

    # Train a Ridge model on ALL data and check which new features rank highly
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_new)
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_sc, y)
    ridge_imp = np.abs(ridge.coef_)

    rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                min_samples_leaf=2, max_features=0.5,
                                random_state=42, n_jobs=-1)
    rf.fit(X_new, y)
    rf_imp = rf.feature_importances_

    xgb_m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=3, reg_lambda=3.0,
                               reg_alpha=1.0, random_state=42, verbosity=0)
    xgb_m.fit(X_new, y)
    xgb_imp = xgb_m.feature_importances_

    from scipy.stats import rankdata
    avg_rank = (rankdata(-ridge_imp) + rankdata(-rf_imp) + rankdata(-xgb_imp)) / 3

    print(f'\n  Top-30 features by combined importance:')
    sorted_idx = np.argsort(avg_rank)
    for rank, fi in enumerate(sorted_idx[:30]):
        is_new = feat_new_names[fi] in new_features
        marker = " ★NEW" if is_new else ""
        print(f'    {rank+1:2d}. {feat_new_names[fi]:30s} '
              f'Ridge={ridge_imp[fi]:7.3f}  RF={rf_imp[fi]:6.4f}  '
              f'XGB={xgb_imp[fi]:6.4f}  AvgRank={avg_rank[fi]:5.1f}{marker}')

    print(f'\n  New features in top-25: ', end='')
    top25_new = [feat_new_names[fi] for fi in sorted_idx[:25] if feat_new_names[fi] in new_features]
    print(f'{len(top25_new)} → {top25_new}')

    # ════════════════════════════════════════════════════
    #  TEST 4: Only add the most committee-relevant features
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' TEST 4: Selective committee feature addition')
    print('=' * 60)

    # Try adding only subsets of new features to the original 68
    feature_groups = {
        'unused_raw': ['AvgOppNET', 'NETNonConfSOS'],
        'efficiency': ['net_efficiency_gap', 'net_opp_ratio'],
        'nonconf': ['nonconf_sos_diff', 'nonconf_sos_ratio', 'nonconf_quality', 'nonconf_performance'],
        'committee_score': ['committee_score'],
        'dominance': ['q4_domination', 'q3_win_rate'],
        'sor': ['strength_of_record', 'wins_above_expected'],
        'conf_density': ['conf_tourn_teams', 'conf_internal_rank'],
        'consistency': ['quad_consistency'],
        'road': ['road_win_count', 'road_game_pct'],
        'severity': ['bad_loss_severity'],
        'schedule': ['schedule_difficulty', 'opp_net_vs_rank', 'conf_adjusted_net'],
    }

    # Test each group individually
    for group_name, group_feats in feature_groups.items():
        # Build feature set: original 68 + this group
        selected_feats = feat_orig_names + [f for f in group_feats if f in feat_new_names]
        selected_idx = [feat_new_names.index(f) for f in selected_feats if f in feat_new_names]
        X_sel = X_new[:, selected_idx]
        sel_names = [feat_new_names[i] for i in selected_idx]

        pred_fn = make_v6_predict(sel_names)
        ex, rmse = kaggle_eval(pred_fn, X_sel, y, seasons, folds, test_mask)
        loso, fr = loso_eval(pred_fn, X_sel, y, seasons, folds)
        tag = f'+{group_name}({len(group_feats)})'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  TEST 5: Best combinations of new feature groups
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' TEST 5: Combined groups (top performers from TEST 4)')
    print('=' * 60)

    # Try promising combinations
    combos = [
        ('raw+eff', ['unused_raw', 'efficiency']),
        ('raw+eff+nonconf', ['unused_raw', 'efficiency', 'nonconf']),
        ('raw+comm+dom', ['unused_raw', 'committee_score', 'dominance']),
        ('raw+sor+sev', ['unused_raw', 'sor', 'severity']),
        ('raw+eff+comm+dom', ['unused_raw', 'efficiency', 'committee_score', 'dominance']),
        ('all_new', list(feature_groups.keys())),
        ('raw+eff+nonconf+comm', ['unused_raw', 'efficiency', 'nonconf', 'committee_score']),
        ('raw+eff+dom+sor', ['unused_raw', 'efficiency', 'dominance', 'sor']),
        ('raw+eff+conf+road', ['unused_raw', 'efficiency', 'conf_density', 'road']),
    ]

    for combo_name, groups in combos:
        added = []
        for g in groups:
            added.extend(feature_groups[g])
        selected_feats = feat_orig_names + [f for f in added if f in feat_new_names]
        selected_idx = [feat_new_names.index(f) for f in selected_feats if f in feat_new_names]
        X_sel = X_new[:, selected_idx]
        sel_names = [feat_new_names[i] for i in selected_idx]

        for top_k in [25, 30]:
            pred_fn = make_committee_predict(sel_names, top_k=top_k)
            ex, rmse = kaggle_eval(pred_fn, X_sel, y, seasons, folds, test_mask)
            loso, fr = loso_eval(pred_fn, X_sel, y, seasons, folds)
            tag = f'{combo_name}-k{top_k}({len(selected_feats)}f)'
            results.append((tag, ex, rmse, loso))
            d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
            d_l = "↑" if loso < v6_loso else "↓"
            marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
            print(f'  {tag:40s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY')
    print('=' * 60)

    print(f'\n  {"Approach":<42} {"Kaggle":>10} {"RMSE":>8} {"LOSO":>8} {"Status":>6}')
    print(f'  {"─"*42} {"─"*10} {"─"*8} {"─"*8} {"─"*6}')

    for tag, ex, rmse, loso in results:
        kg_up = ex > v6_exact or (ex == v6_exact and rmse < v6_rmse)
        lo_up = loso < v6_loso
        if kg_up and lo_up:
            status = '★★'
        elif kg_up or lo_up:
            status = '★'
        else:
            status = ''
        bg = '→' if tag == 'v6-baseline-68feat' else ' '
        print(f' {bg}{tag:<42} {ex:2d}/91  {rmse:8.4f} {loso:8.4f} {status:>6}')

    # Winners
    both_better = [(t, e, r, l) for t, e, r, l in results
                   if (e > v6_exact or (e == v6_exact and r < v6_rmse)) and l < v6_loso]
    print(f'\n  Approaches beating v6 on BOTH metrics:')
    if both_better:
        for t, e, r, l in sorted(both_better, key=lambda x: x[2]):
            print(f'    ★★ {t}: Kaggle={e}/91 RMSE={r:.4f} LOSO={l:.4f}')
    else:
        print('    NONE')
        # Show top 5 closest
        print(f'\n  Top single-metric improvements:')
        kg_better = [(t, e, r, l) for t, e, r, l in results
                     if e > v6_exact or (e == v6_exact and r < v6_rmse)]
        lo_better = [(t, e, r, l) for t, e, r, l in results if l < v6_loso]
        if kg_better:
            print(f'    Kaggle improvements:')
            for t, e, r, l in sorted(kg_better, key=lambda x: x[2])[:3]:
                print(f'      {t}: {e}/91 RMSE={r:.4f} (LOSO={l:.4f})')
        if lo_better:
            print(f'    LOSO improvements:')
            for t, e, r, l in sorted(lo_better, key=lambda x: x[3])[:3]:
                print(f'      {t}: LOSO={l:.4f} (Kaggle={e}/91 RMSE={r:.4f})')

    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
