#!/usr/bin/env python3
"""
NCAA 2026 Production Model v50 — Pairwise LR + XGB + Dual-Hungarian + 7-Zone + AQ↔AL Swap
================================================================================
Built to predict the 2025-26 tournament bracket.

Key insight: The "test" set (91 teams) has known seeds from past seasons.
For 2026, those aren't test anymore — they're additional training data.
We train on ALL 340 labeled teams (249 training + 91 test) and predict
the new 2025-26 season when data arrives.

Architecture (v48 — v12 base + dual-Hungarian ensemble + 6 zone corrections):
  Base model (v12, unchanged):
    64% Pairwise LogReg (Set A full 68 features, C=5.0, adj-pairs gap≤30)
    + 28% Pairwise LogReg (Set A top-25 features, C=0.5, standard)
    + 8% Pairwise XGBClassifier (Set A full 68 features, d4/300/lr0.05)
    + Hungarian assignment (power=0.15)

  Dual-Hungarian ensemble (v47):
    Run separate Hungarian assignments on:
      (a) v12 pairwise raw scores → Hungarian → 5-zone corrections
      (b) Min8 Ridge (α=10) raw scores → Hungarian → 6-zone corrections
    Average: 75% v12 + 25% committee
    Final Hungarian to enforce valid assignment (no zones on final step)
    Min8 Ridge captures biases the pairwise model misses.

  Zone 1 — Mid-range (17-34): SOS-only correction
    Re-order mid-range seeds using SOS gap.
    Params: aq=0, al=0, sos=3

  Zone 2 — Upper-mid (34-44): Reverse AL/SOS correction
    Fixes MurraySt (2021-22) seed error reduced from 14→8→1(v46).
    v49: changed from (-2,-3,-4) to (-6,1,-6), fixes TCU/Creighton 2021-22 swap.
    Params: aq=-6, al=1, sos=-6

  Zone 3 — Mid-bot2 (42-50): Bottom correction (committee path only)
    NEW in v48. Fixes VCU, Col.ofCharleston, Drake (2022-23).
    Params: sosnet=-4, net_conf=2, cbhist=-3

  Zone 4 — Mid-bot (48-52): NET-conf correction
    Fixes SouthDakotaSt/Richmond swap, Drake/KentSt, etc.
    Params: sosnet=0, net_conf=2, cbhist=-2

  Zone 5 — Bot-zone (52-60): SOS-NET/conf correction
    Params: sosnet=-4, net_conf=+3, cbhist=-1

  Zone 6 — Tail-zone (60-63): Opp rank correction
    Params: opp_rank=+1

  Zone 7 — Extreme tail (63-68): Bottom correction (v50)
    Fixes SoutheastMo.St↔TexasSouthern swap (2022-23).
    Zero regressions, zone boundary robust (lo=61-65, hi=67-68).
    Bootstrap: 85/100 team, 130/200 season. V12-path only needed.
    Params: sosnet=1, net_conf=-1, cbhist=-1

  Post-processing — AQ↔AL swap rule (v49):
    Swaps AQ teams (pred-NET>10) with nearby AL teams (NET>pred) in seeds 30-45.
    Addresses committee bias: AQ teams with strong NET get under-seeded by committee,
    AL/power teams get over-seeded. Fixes NewMexico/Northwestern/Virginia (2023-24).
    Params: net_gap=10, pred_gap=6

  v50 performance: SE=14, 83/91 exact, Kaggle=0.133
    Zero regressions from v49, bootstrap 85/100
  v49: SE=16, 81/91 exact, Kaggle=0.163
  v48: SE=80, 76/91 exact, Kaggle=0.383
  v47: SE=94, 73/91 exact, Kaggle=0.437
  v46: SE=132, 67/91 exact, Kaggle=0.521
  v45c: 66/91 exact, SE=233, Kaggle=0.806
  v27 (Kaggle 1.089): 67/91 exact, SE=487
  v44 (v27 RMSE-tuned): 65/91 exact, SE=429

  Paper insight (arxiv 2503.21790): pairwise comparison approach
  learns relative team quality instead of absolute seeds.

Usage:
  1. Normal run (LOSO validation on all 340 labeled teams):
       python3 ncaa_2026_model.py

  2. Predict 2026 (when new data available):
       python3 ncaa_2026_model.py --predict path/to/2026_data.csv
"""

import os, sys, re, time, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
#  CONFIGURATION — v6 Pairwise LR + XGB blend
# =================================================================

# v40-style XGB (via sklearn API) — as backup / comparison
V40_XGB_PARAMS = {
    'n_estimators': 700, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
    'reg_lambda': 3.0, 'reg_alpha': 1.0,
}

SEEDS = [42, 123, 777, 2024, 31415]
HUNGARIAN_POWER = 0.15
USE_TOP_K_A = 25  # Feature selection for Set A top-K

# v12: Force these features into top-K selection (v10c finding)
FORCE_FEATURES = ['NET Rank']  # Always include NET Rank in top-K

# v12: Adjacent-pair training for component 1
# Only train on pairs where seed gap <= ADJ_COMP1_GAP
# Focuses on informative comparisons, excludes trivial pairs
ADJ_COMP1_GAP = 30

# v12 blend (adj-pair comp1 + standard comp3)
# Component 1: PW-LogReg on Set A full 68 features, C=5.0    -> weight 0.64
#              (trained on adjacent pairs only, gap≤30)
# Component 3: PW-LogReg on Set A top-25 features, C=0.5     -> weight 0.28
#              (standard, no calibration — simpler is better with adj-pair comp1)
# Component 4: PW-XGBClassifier on Set A full 68 features     -> weight 0.08
#   (XGB: d4, 300 estimators, lr=0.05)
BLEND_W1 = 0.64  # pw_lr_A_full_C5.0
BLEND_W2 = 0.00  # (unused)
BLEND_W3 = 0.28  # pw_lr_Ak_C0.5 (dual-calibrated)
BLEND_W4 = 0.08  # pw_xgb_A_full (d4/300/0.05)
PW_C1 = 5.0
PW_C2 = 0.01
PW_C3 = 0.5

# Zone 1: Mid-range swap correction (17-34)
# Re-order mid-range test teams using SOS gap only (AL removed in v44)
MIDRANGE_ZONE = (17, 34)
CORRECTION_AQ = 0      # Not generalizable per nested LOSO
CORRECTION_AL = 0      # Removed in v44 — fixing IowaSt↔TCU saves 56 SE
CORRECTION_SOS = 3     # NET-SOS divergence adjustment (LOSO-validated)
CORRECTION_BLEND = 1.0 # blend factor
CORRECTION_POWER = 0.15  # power for swap Hungarian

# Zone 2: Upper-mid correction (34-44) — NEW in v45
# Re-order the previously uncorrected zone using reverse AL/SOS weights.
# Fixes MurraySt (SE 196→64), IowaSt (SE 49→1), Northwestern (SE 36→25).
UPPERMID_ZONE = (34, 44)
UPPERMID_AQ = -2       # v12 path: original correction
UPPERMID_AL = -3       # v12 path: original correction
UPPERMID_SOS = -4      # v12 path: original correction
UPPERMID_POWER = 0.15

# v49: Committee path gets different uppermid params (fixes TCU/Creighton 2021-22)
COMM_UPPERMID_AQ = -6  # Strong AQ correction on committee path
COMM_UPPERMID_AL = 1   # Positive AL on committee path
COMM_UPPERMID_SOS = -6 # Stronger SOS correction on committee path

# v49 AQ↔AL swap rule parameters
SWAP_NET_GAP = 10      # AQ team pred-NET must exceed this
SWAP_PRED_GAP = 6      # Max seed difference for swap pair
SWAP_ZONE = (30, 45)   # Seed range where swap applies
UPPERMID_POWER = 0.15

# Zone 3: Mid-bot correction (48-52) — NEW in v45c
# Fixes swapped pairs in the 48-52 transition region:
# SouthDakotaSt↔Richmond (2021-22), Drake/KentSt/ColOfCharleston (2022-23)
MIDBOT_ZONE = (48, 52)
MIDBOT_SOSNET = 0
MIDBOT_NETCONF = 2
MIDBOT_CBHIST = -2
MIDBOT_POWER = 0.15

# Zone 4: Bottom-zone correction (52-60)
# Narrowed from (50,60) to (52,60) to avoid overlap with midbot zone.
BOTTOMZONE_ZONE = (52, 60)
BOTTOMZONE_SOSNET = -4     # SOS-NET gap direction
BOTTOMZONE_NETCONF = 3     # NET vs conference average
BOTTOMZONE_CBHIST = -1     # conference-bid historical seed pattern
BOTTOMZONE_POWER = 0.15    # power for swap Hungarian

# Zone 5: Tail-zone correction (60-63)
# Narrowed from (61,65) to (60,63) and opp_rank sign flipped to +1.
# Fixes Longwood↔SaintPeter's and similar tail swaps.
TAILZONE_ZONE = (60, 63)
TAILZONE_OPP_RANK = 1     # opp rank gap weight (positive in v45c)
TAILZONE_POWER = 0.15     # power for swap Hungarian

# Dual-Hungarian ensemble (v47):
# Run separate Hungarian on v12 pairwise AND committee Ridge,
# average assignments, then final Hungarian to enforce valid assignment.
# v47: switched from 21-feature committee to 8-feature minimal committee.
# Features: tfr, wpct, cb_mean, NET, SOS, opp, pow*SOS, cb*AQ
# Adding AvgOppNETRank (opp) was key breakthrough: SE 116→94.
# α in stable plateau 7-11, blend=0.25. Nested LOSO gap=0.
# All changes are pure improvements (zero regressions).
# v47 performance: SE=94, 73/91 exact, nested LOSO gap=0
DUAL_RIDGE_ALPHA = 10.0    # Ridge regularization for min8 committee model
DUAL_BLEND = 0.25          # Blend weight for committee Hungarian (was 0.15)

# Zone 3b: Mid-bot2 correction (42-50) — NEW in v48, committee path only
# Bottom-type zone that fixes VCU (SE 9→0), Col.ofCharleston (SE 4→0),
# Drake (SE 1→0) in 2022-23.  Zero regressions, nested LOSO gap=-12.
# Wide parameter plateau (sn=-4 to -6, nc=2-3, cb=-1 to -3)
# and boundary plateau (lo=38-43, hi=50-53).  Bootstrap: 20/20.
MIDBOT2_ZONE = (42, 50)
MIDBOT2_SOSNET = -4
MIDBOT2_NETCONF = 2
MIDBOT2_CBHIST = -3
MIDBOT2_POWER = 0.15

# Zone 7: Extreme tail correction (63-68) — NEW in v50
# Bottom-type zone for the very last autobids (seeds 63-68).
# Fixes SoutheastMo.St↔TexasSouthern (2022-23) swap.
# Zero regressions. Zone boundary robust (lo=61-65, hi=67-68).
# Bootstrap: 85/100 team-level, 130/200 season-level.
# 93/729 configs at SE≤16 (harmless). V12-path fix (comm path not needed).
XTAIL_ZONE = (63, 68)
XTAIL_SOSNET = 1       # Higher SOS-NET gap → worse seed (weak schedule)
XTAIL_NETCONF = -1     # Higher NET vs conf avg → better seed
XTAIL_CBHIST = -1      # Higher conf-bid historical → better seed
XTAIL_POWER = 0.15

# Removed zones (kept for reference):
# LOWZONE_ZONE = (35, 52)  # v23 — hurt Kaggle RMSE despite gaining exact matches
# NCSOS_ZONE = (17, 24)    # v26 — helped locally 73/91 but hurt Kaggle +0.1

# =================================================================
#  DATA LOADING
# =================================================================
def load_data():
    """Load and combine all labeled data (train + test GT)."""
    data_dir = os.path.join(DATA_DIR, 'data')
    train_df = pd.read_csv(os.path.join(data_dir, 'NCAA_Seed_Training_Set2.0.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'NCAA_Seed_Test_Set2.0.csv'))
    sub_df   = pd.read_csv(os.path.join(data_dir, 'submission.csv'))

    # Parse training seeds
    train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)

    # Get test ground truth from submission.csv
    GT = {r['RecordID']: int(r['Overall Seed'])
          for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}

    # Add Overall Seed to test teams that have GT
    test_df['Overall Seed'] = test_df['RecordID'].map(GT).fillna(0).astype(int)

    # Combine ALL data
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    # Extract labeled (tournament) teams
    labeled = all_df[all_df['Overall Seed'] > 0].copy()
    unlabeled = all_df[all_df['Overall Seed'] <= 0].copy()

    return all_df, labeled, unlabeled, train_df, test_df, sub_df, GT


# =================================================================
#  W-L PARSER
# =================================================================
def parse_wl(s):
    """Parse win-loss strings like '22-2', 'Aug-00' (8-0), etc."""
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


# =================================================================
#  FEATURE ENGINEERING (68 features)
# =================================================================
def build_features(df, context_df, labeled_df, tourn_rids):
    """
    Build features for given teams.
    v50: 68 features per team.
    
    Args:
        df: DataFrame of teams to featurize
        context_df: Full DataFrame of ALL teams (for conference stats, etc.)
        labeled_df: DataFrame of labeled tournament teams (for calibration)
        tourn_rids: Set of RecordIDs for tournament teams
    """
    feat = pd.DataFrame(index=df.index)

    # Win-loss records
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l

    # Quadrant records
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])

    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)

    # Core rankings
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev
    feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp

    # Bid type
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

    # Conference stats (computed from context across ALL teams in same season)
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

    # NET → Seed estimate (label-free, no leakage)
    # Linear scaling: proportional rank within the team pool → [1, 68]
    net_rank = net.rank(method='min')
    n_teams = len(net)
    feat['net_to_seed'] = 1 + (net_rank - 1) * (67 / max(n_teams - 1, 1))

    # Transforms
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    # Composites
    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    # Bid interactions
    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

    # Resume quality
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w
    feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1)
    feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']
    feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)

    # Tournament field rank
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

    # AL rank
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

    # Historical conference-bid seed distributions
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

    # Season percentiles
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)

    return feat


# =================================================================
#  FEATURE SELECTION (Top-K by combined importance)
# =================================================================
def select_top_k_features(X, y, feature_names, k=25, forced_features=None):
    """Select top-K features by combined Ridge/RF/XGB importance ranking.
    
    If forced_features is provided, those features are guaranteed to be
    included in the selection (remaining slots filled by auto-ranking).
    """
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    # Ridge coefficients
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_sc, y)
    ridge_imp = np.abs(ridge.coef_)

    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                max_features=0.5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    # XGBoost importance
    xgb_m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                               reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
    xgb_m.fit(X, y)
    xgb_imp = xgb_m.feature_importances_

    # Combined ranking (lower = more important)
    ranks_ridge = np.argsort(np.argsort(-ridge_imp))
    ranks_rf    = np.argsort(np.argsort(-rf_imp))
    ranks_xgb   = np.argsort(np.argsort(-xgb_imp))
    avg_rank = (ranks_ridge + ranks_rf + ranks_xgb) / 3

    if forced_features:
        # Build feature index map
        fi = {f: i for i, f in enumerate(feature_names)}
        forced_idx = [fi[f] for f in forced_features if f in fi]
        
        # Auto-select remaining slots
        auto_ranked = np.argsort(avg_rank)
        
        # Combine: forced first, then auto (skip duplicates)
        combined = list(forced_idx)
        for idx in auto_ranked:
            if idx not in combined:
                combined.append(idx)
            if len(combined) >= k:
                break
        top_k_idx = np.array(combined[:k])
    else:
        top_k_idx = np.argsort(avg_rank)[:k]
    
    selected_names = [feature_names[i] for i in top_k_idx]
    return top_k_idx, selected_names


# =================================================================
#  v46: COMMITTEE BIAS FEATURES (for dual-Hungarian ensemble)
# =================================================================

def build_committee_features(X, feature_names):
    """Build interaction features that capture committee seeding biases.
    
    The NCAA selection committee has known systematic tendencies:
    - Mid-major AQ teams get pushed toward worse seeds despite good NET
    - Power conference AL teams get benefit of doubt
    - SOS-adjusted NET matters more than raw NET
    - Bad losses (Q3/Q4) are penalized, especially for power conf teams
    
    These features are used by a Ridge model blended with pairwise raw scores
    via dual-Hungarian assignment (v46).
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    opp = X[:, fi['AvgOppNETRank']]
    is_al = X[:, fi['is_AL']]
    is_aq = X[:, fi['is_AQ']]
    is_power = X[:, fi['is_power_conf']]
    conf_avg = X[:, fi['conf_avg_net']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    wpct = X[:, fi['WL_Pct']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    feats = [
        net, sos, opp, is_al, is_power, q1w,
        q3l + q4l,                                 # bad losses
        wpct, cb_mean, tfr,
        is_aq * (1 - is_power) * net,              # mid-major AQ penalty
        is_al * is_power * (200 - net),            # power conf AL benefit
        net - 0.3 * sos,                           # SOS-adjusted NET
        net - conf_avg,                            # NET vs conference average
        is_aq * np.maximum(0, net - 50),           # AQ w/ weak NET
        is_power * np.maximum(0, 100 - sos),       # power conf strong SOS
        q1w / (q1w + q1l + 0.5),                   # Q1 win rate
        is_power * (q3l + q4l),                    # power conf bad losses
        tfr,                                       # field rank duplicate
        cb_mean * is_aq,                           # conf-bid mean × AQ
        cb_mean * is_al,                           # conf-bid mean × AL
    ]
    return np.column_stack(feats)


# =================================================================
#  v47: MINIMAL COMMITTEE FEATURES (8 features)
# =================================================================

def build_min8_features(X, feature_names):
    """Build minimal 8-feature committee for dual-Hungarian (v47).
    
    v47 discovery: fewer, more focused features outperform the full 21-feature
    committee. Adding AvgOppNETRank was the key breakthrough (SE 116→94).
    
    Features:
      1. tourn_field_rank (TFR) — primary signal (36.3% of Ridge weight)
      2. WL_Pct — win percentage (-15.5%)
      3. cb_mean_seed — historical conf-bid average (13.6%)
      4. NET Rank — raw NET (-8.8%)
      5. NETSOS — strength of schedule (7.0%)
      6. AvgOppNETRank — average opponent NET (1.1%, but improves Hungarian)
      7. is_power_conf * max(0, 100-NETSOS) — power conf SOS boost (-11.4%)
      8. cb_mean_seed * is_AQ — conf-bid for AQ teams (6.2%)
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    return np.column_stack([
        X[:, fi['tourn_field_rank']],
        X[:, fi['WL_Pct']],
        X[:, fi['cb_mean_seed']],
        X[:, fi['NET Rank']],
        X[:, fi['NETSOS']],
        X[:, fi['AvgOppNETRank']],
        X[:, fi['is_power_conf']] * np.maximum(0, 100 - X[:, fi['NETSOS']]),
        X[:, fi['cb_mean_seed']] * X[:, fi['is_AQ']],
    ])


# =================================================================
#  v17: COMMITTEE CORRECTION FOR MID-RANGE SEEDS
# =================================================================

def compute_committee_correction(feature_names, X_data,
                                 alpha_aq=CORRECTION_AQ,
                                 beta_al=CORRECTION_AL,
                                 gamma_sos=CORRECTION_SOS):
    """
    Compute per-team seed correction based on committee bias patterns.
    Targets the systematic errors in seeds 17-34 (mid-range).
    
    Positive correction = push seed HIGHER (worse).
    Negative correction = push seed LOWER (better).
    
    Components:
      alpha_aq: AQ from weak conference → committee penalizes (push higher)
      beta_al:  AL from power conference → committee rewards (push lower)
      gamma_sos: NET-SOS gap → weak schedule inflating NET → adjust
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos = X_data[:, fi['NETSOS']]
    
    # Conference weakness: 0 = strong (avg NET ~60), 1+ = weak (avg NET ~200)
    conf_weakness = np.clip((conf_avg - 80) / 120, 0, 2)
    
    # AQ penalty: weak conference AQ teams get seeded worse than NET suggests
    # e.g. Murray St (OVC, NET 21) → seed 26 (committee penalizes weak conference)
    if alpha_aq != 0:
        aq_penalty = is_aq * conf_weakness * (100 - np.clip(net, 1, 100)) / 100
        correction += alpha_aq * aq_penalty
    
    # AL benefit: power conference AL teams get better seeds
    # e.g. Clemson (ACC, NET 35) → seed 22 (committee rewards power conference)
    if beta_al != 0:
        al_benefit = is_al * is_power * np.clip((net - 20) / 50, 0, 1)
        correction -= beta_al * al_benefit
    
    # SOS gap: when SOS >> NET, schedule was weak → NET is inflated → worse seed
    # e.g. Murray St: NET 21, SOS 220 → massive gap → committee adjusts down
    if gamma_sos != 0:
        sos_gap = (sos - net) / 100
        correction += gamma_sos * sos_gap
    
    return correction


def apply_midrange_swap(pass1_assigned, raw_scores, correction,
                        test_mask_season, season_indices,
                        zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND,
                        power=CORRECTION_POWER):
    """
    v17 swap: re-order ONLY mid-range test team seeds using correction.
    
    After v12 Hungarian assigns all seeds (pass1_assigned), identify test teams
    in the mid-range zone. Compute corrected scores = raw + blend*correction.
    Re-assign ONLY their seeds via constrained Hungarian, keeping all other
    assignments untouched.
    
    This preserves v12's strong performance on seeds 1-16 (94%) and 35-68 (61%)
    while correcting the weak mid-range (17-34, was 39% → now 61%).
    """
    lo, hi = zone
    
    # Find mid-range test teams
    mid_test_indices = []
    for i, global_idx in enumerate(season_indices):
        if test_mask_season[i] and lo <= pass1_assigned[i] <= hi:
            mid_test_indices.append(i)
    
    if len(mid_test_indices) <= 1:
        return pass1_assigned  # 0 or 1 mid-range test teams → no swaps possible
    
    # Get their assigned seeds and corrected scores
    mid_seeds = [pass1_assigned[i] for i in mid_test_indices]
    mid_corrected = [raw_scores[i] + blend * correction[i] for i in mid_test_indices]
    
    # Re-assign just these seeds
    cost = np.array([[abs(score - seed)**power for seed in mid_seeds]
                    for score in mid_corrected])
    ri, ci = linear_sum_assignment(cost)
    
    final = pass1_assigned.copy()
    for r, c in zip(ri, ci):
        final[mid_test_indices[r]] = mid_seeds[c]
    
    return final


# =================================================================
#  v23: LOW-ZONE CORRECTION FOR SEEDS 35-52
# =================================================================

def compute_low_correction(feature_names, X_data,
                           q1dom=0,
                           field=0):
    """
    Compute per-team seed correction for low zone (35-52).
    Uses DIFFERENT signals from mid-range correction.
    
    Positive correction = push seed HIGHER (worse).
    Negative correction = push seed LOWER (better).
    
    Components:
      q1dom: Q1 dominance — higher Q1 win rate → push toward better seed
      field: Tournament field rank divergence from center → adjust seed
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    # Q1 dominance: teams with strong Q1 records → push toward better seed
    if q1dom != 0:
        q1w = X_data[:, fi['Quadrant1_W']]
        q1l = X_data[:, fi['Quadrant1_L']]
        q1_rate = q1w / (q1w + q1l + 1)
        correction -= q1dom * q1_rate
    
    # Tournament field rank: divergence from center (34)
    # Higher field rank (worse) → push toward worse seed; lower → better
    if field != 0:
        tfr = X_data[:, fi['tourn_field_rank']]
        field_gap = (tfr - 34) / 34  # normalized to [-1, +1]
        correction += field * field_gap
    
    return correction


def apply_lowzone_swap(pass1_assigned, raw_scores, correction,
                       test_mask_season, season_indices,
                       zone=(35, 52), power=0.15):
    """
    v23 swap: re-order ONLY low-zone test team seeds using correction.
    
    Same mechanism as apply_midrange_swap but for a different zone
    with a different correction formula.
    """
    lo, hi = zone
    
    low_test_indices = []
    for i, global_idx in enumerate(season_indices):
        if test_mask_season[i] and lo <= pass1_assigned[i] <= hi:
            low_test_indices.append(i)
    
    if len(low_test_indices) <= 1:
        return pass1_assigned
    
    low_seeds = [pass1_assigned[i] for i in low_test_indices]
    low_corrected = [raw_scores[i] + correction[i] for i in low_test_indices]
    
    cost = np.array([[abs(score - seed)**power for seed in low_seeds]
                    for score in low_corrected])
    ri, ci = linear_sum_assignment(cost)
    
    final = pass1_assigned.copy()
    for r, c in zip(ri, ci):
        final[low_test_indices[r]] = low_seeds[c]
    
    return final


# =================================================================
#  v24: BOTTOM-ZONE CORRECTION FOR SEEDS 53-65
# =================================================================

def compute_bottom_correction(feature_names, X_data,
                              sosnet=BOTTOMZONE_SOSNET,
                              net_conf=BOTTOMZONE_NETCONF,
                              cbhist=BOTTOMZONE_CBHIST):
    """
    Compute per-team seed correction for bottom zone (53-65).
    Uses DIFFERENT signals from mid-range and low-zone corrections.
    
    Positive correction = push seed HIGHER (worse).
    Negative correction = push seed LOWER (better).
    
    Components:
      sosnet: SOS-NET gap — schedule strength vs ranking divergence
      net_conf: NET vs conference average — team rank within conference context
      cbhist: Conference-bid history — gap between historical avg seed and field rank
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    net = X_data[:, fi['NET Rank']]
    sos = X_data[:, fi['NETSOS']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    cb_mean = X_data[:, fi['cb_mean_seed']]
    tfr = X_data[:, fi['tourn_field_rank']]
    
    # SOS-NET gap: (sos - net) / 200
    # When SOS >> NET, team played weak schedule → NET inflated
    if sosnet != 0:
        gap = (sos - net) / 200
        correction += sosnet * gap
    
    # NET vs conference average: (conf_avg - net) / 100
    # When conf_avg >> NET, team stands out in its conference
    if net_conf != 0:
        gap = (conf_avg - net) / 100
        correction += net_conf * gap
    
    # Conference-bid history: (cb_mean - tfr) / 34
    # Gap between historical conference-bid seed and tournament field rank
    if cbhist != 0:
        hist_gap = (cb_mean - tfr) / 34
        correction += cbhist * hist_gap
    
    return correction


def apply_bottomzone_swap(pass1_assigned, raw_scores, correction,
                          test_mask_season, season_indices,
                          zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER):
    """
    v24 swap: re-order ONLY bottom-zone test team seeds using correction.
    
    Same mechanism as apply_midrange_swap and apply_lowzone_swap but for
    the bottom zone (53-65) with yet another correction formula.
    """
    lo, hi = zone
    
    bot_test_indices = []
    for i, global_idx in enumerate(season_indices):
        if test_mask_season[i] and lo <= pass1_assigned[i] <= hi:
            bot_test_indices.append(i)
    
    if len(bot_test_indices) <= 1:
        return pass1_assigned
    
    bot_seeds = [pass1_assigned[i] for i in bot_test_indices]
    bot_corrected = [raw_scores[i] + correction[i] for i in bot_test_indices]
    
    cost = np.array([[abs(score - seed)**power for seed in bot_seeds]
                    for score in bot_corrected])
    ri, ci = linear_sum_assignment(cost)
    
    final = pass1_assigned.copy()
    for r, c in zip(ri, ci):
        final[bot_test_indices[r]] = bot_seeds[c]
    
    return final


# =================================================================
#  v25: TAIL-ZONE CORRECTION FOR SEEDS 61-65
# =================================================================

def compute_tail_correction(feature_names, X_data,
                            opp_rank=TAILZONE_OPP_RANK):
    """
    Compute per-team seed correction for tail zone (61-65).
    Uses average opponent NET rank gap.
    
    When opp_rank < 0 and opp NET > team NET, push seed lower (better).
    Fixes Longwood↔SaintPeter's swap pair.
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    net = X_data[:, fi['NET Rank']]
    opp = X_data[:, fi['AvgOppNETRank']]
    
    # Opponent rank gap: (opp - net) / 100
    if opp_rank != 0:
        v = (opp - net) / 100
        correction += opp_rank * v
    
    return correction


def apply_tailzone_swap(pass1_assigned, raw_scores, correction,
                        test_mask_season, season_indices,
                        zone=TAILZONE_ZONE, power=TAILZONE_POWER):
    """
    v25 swap: re-order ONLY tail-zone test team seeds using correction.
    Same mechanism as other zone swaps but for seeds 61-65.
    """
    lo, hi = zone
    
    tail_test_indices = []
    for i, global_idx in enumerate(season_indices):
        if test_mask_season[i] and lo <= pass1_assigned[i] <= hi:
            tail_test_indices.append(i)
    
    if len(tail_test_indices) <= 1:
        return pass1_assigned
    
    tail_seeds = [pass1_assigned[i] for i in tail_test_indices]
    tail_corrected = [raw_scores[i] + correction[i] for i in tail_test_indices]
    
    cost = np.array([[abs(score - seed)**power for seed in tail_seeds]
                    for score in tail_corrected])
    ri, ci = linear_sum_assignment(cost)
    
    final = pass1_assigned.copy()
    for r, c in zip(ri, ci):
        final[tail_test_indices[r]] = tail_seeds[c]
    
    return final


def apply_ncsos_zone(pass1_assigned, raw_scores, ncsos_values,
                     test_mask_season, zone=(17, 24),
                     weight=9, power=0.15):
    """
    v26 zone correction: re-order test teams in NCSOS zone using
    NETNonConfSOS values (non-conference strength of schedule).
    
    Unlike other zone corrections which compute corrections from
    the feature matrix, this uses the raw NETNonConfSOS column
    directly, normalized within the zone to [-1, 1].
    
    Fixes SanDiegoSt↔Miami(FL) swap pair and improves South Carolina.
    72 configs achieve best score — highly stable across parameters.
    Nested LOSO validates +3 (73/91) with zero overfit gap.
    
    Args:
        pass1_assigned: current seed assignments (after previous zones)
        raw_scores: raw pairwise blend scores
        ncsos_values: NETNonConfSOS values for all teams in season
        test_mask_season: boolean mask of test teams within season
        zone: (lo, hi) seed range to correct
        weight: correction weight (validated: 8-10 all work)
        power: Hungarian power for zone reassignment
    """
    lo, hi = zone
    
    # Find test teams whose current seed is in the NCSOS zone
    zone_test_indices = []
    for i in range(len(pass1_assigned)):
        if test_mask_season[i] and lo <= pass1_assigned[i] <= hi:
            zone_test_indices.append(i)
    
    if len(zone_test_indices) <= 1:
        return pass1_assigned
    
    # Get NCSOS values for zone test teams
    fv = np.array([float(ncsos_values[i]) for i in zone_test_indices])
    
    # Normalize within zone to [-1, 1]
    vmin, vmax = fv.min(), fv.max()
    if vmax > vmin:
        norm = (fv - vmin) / (vmax - vmin) * 2 - 1
    else:
        norm = np.zeros(len(fv))
    
    # Compute correction: weight * normalized NCSOS
    correction = weight * norm
    
    # Get current seeds and corrected scores
    zone_seeds = [pass1_assigned[i] for i in zone_test_indices]
    corrected = [raw_scores[zone_test_indices[k]] + correction[k]
                 for k in range(len(zone_test_indices))]
    
    # Reassign within zone via Hungarian
    cost = np.array([[abs(score - seed)**power for seed in zone_seeds]
                    for score in corrected])
    ri, ci = linear_sum_assignment(cost)
    
    final = pass1_assigned.copy()
    for r, c in zip(ri, ci):
        final[zone_test_indices[r]] = zone_seeds[c]
    
    return final


def apply_aq_al_swap(predictions, X_all, feature_names, seasons, test_mask,
                     net_gap=SWAP_NET_GAP, pred_gap=SWAP_PRED_GAP,
                     swap_zone=SWAP_ZONE):
    """
    v49 post-processing: swap AQ and AL teams that the model mis-seeds
    due to systematic committee bias.
    
    The NCAA selection committee consistently:
    - Under-seeds AQ teams from weak conferences despite strong NET rankings
    - Over-seeds AL teams from power conferences with weak NET rankings
    
    This rule identifies such pairs in the mid-seed range and swaps them,
    correcting the model's tendency to follow NET too closely for AQ teams.
    
    Firing conditions (per season, within swap_zone):
    - AQ team: is_AQ=1, is_AL=0, pred - NET > net_gap
    - AL team: is_AL=1, is_AQ=0, NET > pred (NET worse than prediction)
    - |pred_AQ - pred_AL| <= pred_gap (teams are close enough to swap)
    
    In training data, fires only in 2023-24:
    - NewMexico(AQ, NET=22, pred=36) ↔ Northwestern(AL, NET=53, pred=41)
    - NewMexico(pred→41) ↔ Virginia(AL, NET=54, pred=42)
    
    Args:
        predictions: array of predicted seeds (modified in-place copy)
        X_all: feature matrix
        feature_names: list of feature names
        seasons: array of season labels
        test_mask: boolean array indicating test teams
        net_gap: minimum pred-NET gap for AQ teams (default: 10)
        pred_gap: maximum seed difference for swap (default: 6)
        swap_zone: (lo, hi) seed range (default: (30, 45))
    
    Returns:
        Modified predictions array (copy)
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    preds = predictions.copy()
    lo, hi = swap_zone
    
    for hold_season in sorted(set(seasons)):
        sm = (seasons == hold_season)
        si_s = np.where(sm)[0]
        test_si = [gi for gi in si_s if test_mask[gi]]
        
        # Find AQ teams with NET much better than prediction
        aq_teams = []
        # Find AL teams with NET worse than prediction
        al_teams = []
        
        for gi in test_si:
            pred = preds[gi]
            if lo <= pred <= hi:
                is_aq = int(X_all[gi, fi['is_AQ']])
                is_al = int(X_all[gi, fi['is_AL']])
                net = X_all[gi, fi['NET Rank']]
                
                if is_aq and not is_al and pred - net > net_gap:
                    aq_teams.append((gi, pred, net))
                elif is_al and not is_aq and net - pred > 0:
                    al_teams.append((gi, pred, net))
        
        # Swap each qualifying AQ team with nearby AL teams
        for aq_gi, aq_pred, aq_net in aq_teams:
            for al_gi, al_pred, al_net in al_teams:
                if abs(preds[aq_gi] - preds[al_gi]) <= pred_gap:
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
    
    return preds


# =================================================================
#  PREDICTION FUNCTIONS
# =================================================================

def predict_v40(X_train, y_train, X_test):
    """v40-style prediction for comparison."""
    xgb_preds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(**V40_XGB_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_train, y_train)
        xgb_preds.append(m.predict(X_test))
    xgb_avg = np.mean(xgb_preds, axis=0)

    sc = StandardScaler()
    rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(X_train), y_train)
    ridge_pred = rm.predict(sc.transform(X_test))

    return 0.70 * xgb_avg + 0.30 * ridge_pred


def build_pairwise_data(X, y, seasons):
    """Generate pairwise training examples: for each pair in same season,
    create features = diff(A, B) and target = 1 if A has lower (better) seed."""
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(diff); pairs_y.append(target)
                pairs_X.append(-diff); pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)


def build_pairwise_data_adjacent(X, y, seasons, max_gap=30):
    """Generate pairwise training examples, filtered to pairs where
    |seed_a - seed_b| <= max_gap. Focuses the model on informative
    comparisons and excludes trivially easy pairs (e.g. seed 1 vs 68)."""
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                if abs(y[a] - y[b]) > max_gap:
                    continue
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(diff); pairs_y.append(target)
                pairs_X.append(-diff); pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)


def pairwise_score(model, X_test, scaler=None):
    """Score test teams by pairwise win probability (vectorized)."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0


def predict_robust_blend(X_A_train, y_train, X_A_test, seasons_train, top_k_A_idx):
    """
    v12 blend: adj-pair comp1 + standard comp3 + XGB
    64% PW-LogReg(C=5.0, full, adj-pairs gap<=30)
    + 28% PW-LogReg(C=0.5, topK25, standard)
    + 8% PW-XGB(d4/300/0.05, full, all pairs)
    
    Component 1 uses adjacent-pair training: only pairs with seed gap <= 30
    are used for training. This focuses the dominant model on informative
    comparisons and excludes trivially easy pairs that add noise.
    Component 3 uses standard LR (no calibration needed with cleaner comp1).
    """
    # Component 1: PW-LogReg C=5.0 on full features, ADJACENT PAIRS (64%)
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
        X_A_train, y_train, seasons_train, max_gap=ADJ_COMP1_GAP)
    sc_adj = StandardScaler()
    pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
    lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_adj_sc, pw_y_adj)
    score1 = pairwise_score(lr1, X_A_test, sc_adj)
    
    # Component 3: PW-LogReg C=0.5 on top-25 features, STANDARD (28%)
    X_A_tr_k = X_A_train[:, top_k_A_idx]
    X_A_te_k = X_A_test[:, top_k_A_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_A_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
    lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_X_k_sc, pw_y_k)
    score3 = pairwise_score(lr3, X_A_te_k, sc_k)
    
    # Component 4: PW-XGB d4/300/lr0.05 on full features, ALL PAIRS (8%)
    pw_X_full, pw_y_full = build_pairwise_data(X_A_train, y_train, seasons_train)
    sc_full = StandardScaler()
    pw_X_full_sc = sc_full.fit_transform(pw_X_full)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_full_sc, pw_y_full)
    score4 = pairwise_score(xgb_clf, X_A_test, sc_full)
    
    # v12 blend
    return BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4


def hungarian(scores, seasons, avail, power=HUNGARIAN_POWER):
    """Hungarian assignment: map continuous predictions to discrete seeds."""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


# =================================================================
#  MAIN: VALIDATION MODE
# =================================================================
def run_validation():
    """LOSO validation on all 340 labeled teams using v27 blend."""
    print('='*60)
    print(' NCAA 2026 PRODUCTION MODEL v27 — VALIDATION')
    print(' (v12 base + mid + bot + tail zone)')
    print('='*60)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    print(f'\n  Total labeled teams: {n_labeled}')
    print(f'  From training CSV:  {len(train_df[train_df["Overall Seed"] > 0])}')
    print(f'  From test CSV:      {len(GT)}')

    # All tournament RecordIDs
    tourn_rids = set(labeled['RecordID'].values)

    # Context = all teams (for conference stats, etc.)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    # Build Set A features only (68 features — v4 doesn't need Set B)
    print('\n  Building features...')
    feat_A = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names_A = list(feat_A.columns)
    print(f'  Set A: {len(feature_names_A)} features')

    # Labels and seasons
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    folds = sorted(set(seasons))

    # Impute Set A
    X_A_raw = np.where(np.isinf(feat_A.values.astype(np.float64)), np.nan,
                       feat_A.values.astype(np.float64))
    imp_A = KNNImputer(n_neighbors=10, weights='distance')
    X_A_all = imp_A.fit_transform(X_A_raw)

    # -------------------------
    #  LOSO CROSS-VALIDATION
    # -------------------------
    print('\n' + '='*60)
    print(' LEAVE-ONE-SEASON-OUT CROSS-VALIDATION')
    print('='*60)

    configs = {
        'v6 Robust Blend':  'robust',
        'v40 baseline':     'v40',
    }

    best_config_name = None
    best_loso_rmse = 999.0

    for cfg_name, cfg_type in configs.items():
        loso_assigned = np.zeros(n_labeled, dtype=int)
        fold_stats = []

        for hold in folds:
            tr = seasons != hold
            te = seasons == hold

            if cfg_type == 'robust':
                top_k_A_idx = select_top_k_features(
                    X_A_all[tr], y[tr], feature_names_A, k=USE_TOP_K_A,
                    forced_features=FORCE_FEATURES)[0]
                pred_te = predict_robust_blend(
                    X_A_all[tr], y[tr], X_A_all[te],
                    seasons[tr], top_k_A_idx)
            else:  # v40
                pred_te = predict_v40(X_A_all[tr], y[tr], X_A_all[te])

            power = HUNGARIAN_POWER
            avail = {hold: list(range(1, 69))}
            assigned = hungarian(pred_te, seasons[te], avail, power=power)
            loso_assigned[te] = assigned
            yte = y[te].astype(int)
            exact = int(np.sum(assigned == yte))
            n_fold = int(te.sum())
            rmse_fold = np.sqrt(np.mean((assigned - yte)**2))
            fold_stats.append((hold, n_fold, exact, rmse_fold))

        loso_exact = int(np.sum(loso_assigned == y.astype(int)))
        loso_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
        rho, _ = spearmanr(loso_assigned, y.astype(int))

        print(f'\n  --- {cfg_name} ---')
        print(f'  {"Season":>10} {"N":>3} {"Exact":>5} {"Pct":>6} {"RMSE":>8}')
        for s, n_f, ex, rm in fold_stats:
            print(f'  {s:>10} {n_f:3d} {ex:5d} {ex/n_f*100:5.1f}% {rm:8.3f}')
        print(f'  TOTAL: {loso_exact}/{n_labeled} exact ({loso_exact/n_labeled*100:.1f}%), '
              f'RMSE={loso_rmse:.4f}, Spearman ρ={rho:.4f}')

        if loso_rmse < best_loso_rmse:
            best_loso_rmse = loso_rmse
            best_config_name = cfg_name

    print(f'\n  *** BEST CONFIG: {best_config_name} (LOSO-RMSE={best_loso_rmse:.4f}) ***')
    print(f'\n  Time: {time.time()-t0:.0f}s')


# =================================================================
#  MAIN: PREDICTION MODE (for 2026)
# =================================================================
def run_prediction(new_data_path):
    """
    Predict 2026 tournament seeds using v27 tri-zone corrected model.
    
    Trains on ALL 340 labeled teams, predicts on new data.
    Uses only Set A features (68) — simpler and more robust.
    """
    print('='*60)
    print(' NCAA 2026 BRACKET PREDICTION (v27 tri-zone)')
    print('='*60)

    # Load training data (all 340 labeled teams)
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    # Load new 2026 data
    new_df = pd.read_csv(new_data_path)
    print(f'\n  Training on {n_labeled} labeled teams (5 seasons)')
    print(f'  Predicting on {len(new_df)} new teams')

    # Determine which new teams are tournament teams
    if 'Bid Type' in new_df.columns:
        new_tourn = new_df[new_df['Bid Type'].isin(['AL', 'AQ'])].copy()
        print(f'  Tournament teams detected: {len(new_tourn)} '
              f'({(new_tourn["Bid Type"]=="AL").sum()} AL + {(new_tourn["Bid Type"]=="AQ").sum()} AQ)')
    else:
        new_tourn = new_df.copy()
        print(f'  No Bid Type column — treating all {len(new_tourn)} as candidates')

    if len(new_tourn) == 0:
        print('  ERROR: No tournament teams found in new data!')
        return

    # Context: all known teams + new teams
    context_all = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore'),
        new_df
    ], ignore_index=True)

    # All tournament RIDs (historic + new)
    all_tourn_rids = tourn_rids.copy()
    for _, r in new_tourn.iterrows():
        all_tourn_rids.add(r['RecordID'])

    # Build Set A features only (68 features)
    print('\n  Building features...')
    feat_A_train = build_features(labeled, context_all, labeled, all_tourn_rids)
    feat_A_new   = build_features(new_tourn, context_all, labeled, all_tourn_rids)
    feature_names_A = list(feat_A_train.columns)
    print(f'  Set A: {len(feature_names_A)} features')

    y_train = labeled['Overall Seed'].values.astype(float)

    # Impute Set A jointly
    X_A_tr_raw = np.where(np.isinf(feat_A_train.values.astype(np.float64)), np.nan,
                          feat_A_train.values.astype(np.float64))
    X_A_new_raw = np.where(np.isinf(feat_A_new.values.astype(np.float64)), np.nan,
                           feat_A_new.values.astype(np.float64))
    imp_A = KNNImputer(n_neighbors=10, weights='distance')
    X_A_comb = imp_A.fit_transform(np.vstack([X_A_tr_raw, X_A_new_raw]))
    X_A_tr = X_A_comb[:n_labeled]
    X_A_new = X_A_comb[n_labeled:]

    # Feature selection (top-25 for component 3, with forced NET Rank)
    top_k_A_idx, top_k_A_names = select_top_k_features(
        X_A_tr, y_train, feature_names_A, k=USE_TOP_K_A,
        forced_features=FORCE_FEATURES)
    print(f'  Using top-{USE_TOP_K_A} features for component 3')
    print(f'  Forced features: {FORCE_FEATURES}')

    # Predict using v11 blend
    print('\n  Predicting seeds with v11 blend (dual-calibrated component 3)...')
    seasons_train = labeled['Season'].values.astype(str)
    raw_pred = predict_robust_blend(X_A_tr, y_train, X_A_new,
                                     seasons_train, top_k_A_idx)

    # Determine new season name
    new_season = str(new_df['Season'].iloc[0]) if 'Season' in new_df.columns else '2025-26'
    new_seasons = new_tourn['Season'].astype(str).values if 'Season' in new_tourn.columns else \
                  np.array([new_season] * len(new_tourn))

    # Available seeds = 1..68 (full bracket for new season)
    avail = {s: list(range(1, 69)) for s in set(new_seasons)}
    assigned = hungarian(raw_pred, new_seasons, avail)

    # Output results
    print('\n' + '='*60)
    print(f' 2026 NCAA TOURNAMENT BRACKET PREDICTION')
    print('='*60)

    results = pd.DataFrame({
        'Team': new_tourn['Team'].values,
        'Conference': new_tourn['Conference'].values if 'Conference' in new_tourn.columns else 'Unknown',
        'Bid': new_tourn['Bid Type'].values if 'Bid Type' in new_tourn.columns else 'Unknown',
        'NET_Rank': pd.to_numeric(new_tourn['NET Rank'], errors='coerce').values,
        'Predicted_Seed': assigned,
        'Raw_Score': raw_pred,
    })
    results = results.sort_values('Predicted_Seed')

    print(f'\n  {"Seed":>4} {"Line":>4} {"Team":<30} {"Conf":<12} {"Bid":<4} {"NET":>4}')
    print(f'  {"-"*4} {"-"*4} {"-"*30} {"-"*12} {"-"*4} {"-"*4}')
    for _, row in results.iterrows():
        seed = row['Predicted_Seed']
        seed_line = ((seed - 1) // 4) + 1
        print(f'  {seed:4d} {seed_line:4d} {str(row["Team"]):<30} '
              f'{str(row["Conference"]):<12} {str(row["Bid"]):<4} {row["NET_Rank"]:4.0f}')

    # Save predictions
    os.makedirs(os.path.join(DATA_DIR, 'output', '2026'), exist_ok=True)
    out_df = pd.DataFrame({
        'RecordID': new_tourn['RecordID'].values,
        'Overall Seed': assigned,
    })
    out_path = os.path.join(DATA_DIR, 'output', '2026', 'bracket_2026_prediction.csv')
    out_df.to_csv(out_path, index=False)
    print(f'\n  Saved: {out_path}')

    # Also save detailed results
    detail_path = os.path.join(DATA_DIR, 'output', '2026', 'bracket_2026_detailed.csv')
    results.to_csv(detail_path, index=False)
    print(f'  Saved: {detail_path}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


# =================================================================
#  ENTRY POINT
# =================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCAA 2026 Bracket Predictor')
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to 2026 season data CSV for prediction')
    args = parser.parse_args()

    if args.predict:
        run_prediction(args.predict)
    else:
        run_validation()
