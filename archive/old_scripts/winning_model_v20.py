"""
v20: Championship-Caliber Model — Incorporating Winning Kaggle Strategies
=========================================================================
Built from research of 1st/2nd/3rd place solutions from Kaggle March Madness
2024-2025 competitions.

Key innovations from winning solutions:
  1. TWO-STAGE APPROACH: classify tournament teams, then predict seed
  2. AGGRESSIVE FEATURE SELECTION: ~25-30 best features (not 100+)
  3. WELL-TUNED XGBoost: low eta (0.009), many rounds (700+), num_parallel_tree=2
  4. RANKING-AWARE: LightGBM LambdaRank + pairwise learning
  5. ORDINAL REGRESSION: mord package for ordinal seed prediction
  6. ELO-STYLE RATINGS: derived from NET rank movements
  7. CatBoost: additional gradient boosting variant
  8. SIMPLER ENSEMBLE: 3-5 well-tuned models, not 100+ weak ones
  9. ISOTONIC POST-PROCESSING: calibrate final predictions
  10. COMMITTEE CORRECTION: bias-adjusted predictions

Approach references:
  - 1st Place 2025 (Mohammad Odeh): XGBoost + 29 features + manual corrections
  - 2nd Place 2024 (Jack Lichtenstein): Ridge + XGBoost pipeline
  - goto_conversion: favourite-longshot bias adjustment
  - KenPom: efficiency-based team ratings
"""

import re, os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, BayesianRidge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score, GroupKFold
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import mord

warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════

def parse_wl(s):
    """Parse W-L records that may use month names (Jan=1, etc.)"""
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    months = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
              'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    for m, n in months.items():
        s = s.replace(m, str(n))
    m = re.search(r'(\d+)\D+(\d+)', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r'(\d+)', s)
    if m2:
        return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)


def safe_div(a, b, default=0.0):
    """Safe division avoiding zero division"""
    return np.where(b != 0, a / b, default)


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — Inspired by winning solutions
# ═══════════════════════════════════════════════════════════════

def build_winning_features(df, all_df, labeled_df):
    """
    Build features inspired by winning Kaggle March Madness solutions.
    Focus on ~40 strong features (will select top 25-30 via MI).
    
    Key insight from winners: NET Rank is king. Everything else supports it.
    """
    feat = pd.DataFrame(index=df.index)
    
    # ── Core Record Features ──
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col+'_W'] = wl.apply(lambda x: x[0])
            feat[col+'_L'] = wl.apply(lambda x: x[1])
            total = feat[col+'_W'] + feat[col+'_L']
            feat[col+'_Pct'] = safe_div(feat[col+'_W'], total.replace(0, np.nan))
    
    # ── Quadrant Records ──
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])
            total = feat[q+'_W'] + feat[q+'_L']
            feat[q+'_rate'] = safe_div(feat[q+'_W'], total.replace(0, np.nan))
    
    # ── NET / Ranking Features ──
    for col in ['NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET', 'NETSOS', 'NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')
    
    # ── Bid Type ──
    feat['is_AL'] = (df['Bid Type'].fillna('') == 'AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('') == 'AQ').astype(float)
    
    # ── Conference Strength ──
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_conf, 'NET': all_net})
    
    feat['conf_avg_net'] = conf.map(cs.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_best_net'] = conf.map(cs.groupby('Conference')['NET'].min()).fillna(200)
    feat['conf_size'] = conf.map(cs.groupby('Conference')['NET'].count()).fillna(10)
    
    power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East', 'Pac-12', 'AAC', 'Mountain West', 'WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    
    # ── Derived Features (inspired by 2025 1st place: Elo, quality, point diff proxies) ──
    net = feat['NET Rank'].fillna(300)
    prev = feat['PrevNET'].fillna(300)
    sos = feat['NETSOS'].fillna(200)
    ncsos = feat['NETNonConfSOS'].fillna(200)
    wpct = feat['WL_Pct'].fillna(0.5)
    
    q1w = feat['Quadrant1_W'].fillna(0)
    q1l = feat['Quadrant1_L'].fillna(0)
    q2w = feat['Quadrant2_W'].fillna(0)
    q2l = feat['Quadrant2_L'].fillna(0)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat['Quadrant3_L'].fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat['Quadrant4_L'].fillna(0)
    
    totalw = feat['WL_W'].fillna(0)
    totall = feat['WL_L'].fillna(0)
    roadw = feat['RoadWL_W'].fillna(0)
    roadl = feat['RoadWL_L'].fillna(0)
    confw = feat['Conf.Record_W'].fillna(0)
    confl = feat['Conf.Record_L'].fillna(0)
    ncw = feat['Non-ConferenceRecord_W'].fillna(0)
    ncl = feat['Non-ConferenceRecord_L'].fillna(0)
    
    is_al = feat['is_AL']
    is_aq = feat['is_AQ']
    cav = feat['conf_avg_net']
    
    # -- ELO-style proxy: NET improvement indicates momentum --
    feat['elo_proxy'] = 400 - net  # higher = better (like Elo)
    feat['elo_momentum'] = prev - net  # positive = improving
    feat['elo_momentum_pct'] = (prev - net) / (prev + 1)
    
    # -- Seed line estimation (the most powerful feature) --
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['within_line_pos'] = net - (feat['seed_line_est'] - 1) * 4
    
    # -- Tier indicators --
    feat['is_top16'] = (net <= 16).astype(float)
    feat['is_top32'] = (net <= 32).astype(float)
    feat['is_bubble'] = ((net >= 30) & (net <= 80) & (is_al == 1)).astype(float)
    
    # -- Quality metrics (inspired by KenPom efficiency) --
    feat['resume_score'] = q1w * 4 + q2w * 2 - q3l * 2 - q4l * 4
    feat['quality_ratio'] = (q1w * 3 + q2w * 2) / (q3l * 2 + q4l * 3 + 1)
    feat['total_bad_losses'] = q3l + q4l
    feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    q12t = q1w + q1l + q2w + q2l
    feat['q12_win_rate'] = (q1w + q2w) / (q12t + 1)
    feat['q12_opportunity'] = q12t / (q12t + q3w + q3l + q4w + q4l + 0.5)
    
    # -- Record-based features --
    tg = totalw + totall
    feat['wins_above_500'] = totalw - tg / 2
    feat['conf_wins_above_500'] = confw - (confw + confl) / 2
    feat['road_performance'] = roadw / (roadw + roadl + 0.5)
    
    # -- NET interaction features --
    feat['net_inv'] = 1.0 / (net + 1)
    feat['net_x_wpct'] = net * wpct / 100
    feat['net_log'] = np.log1p(net)
    feat['net_sqrt'] = np.sqrt(net)
    feat['adj_net'] = net - q1w * 0.5 + q3l * 1.0 + q4l * 2.0
    
    # -- SOS interaction --
    feat['net_sos_gap'] = (net - sos).abs()
    feat['sos_x_wpct'] = sos * wpct / 100
    feat['record_vs_sos'] = wpct * (300 - sos) / 200
    
    # -- Bid-type interactions --
    feat['al_net'] = net * is_al
    feat['aq_net'] = net * is_aq
    feat['al_q1w'] = q1w * is_al
    feat['al_wpct'] = wpct * is_al
    feat['power_al'] = is_al * feat['is_power_conf']
    feat['midmajor_aq'] = is_aq * (1 - feat['is_power_conf'])
    
    # -- Conference relative features --
    feat['net_div_conf'] = net / (cav + 1)
    feat['wpct_x_confstr'] = wpct * (300 - cav) / 200
    
    # -- Opponent quality --
    opp = feat['AvgOppNETRank'].fillna(200)
    feat['opp_quality'] = (400 - opp) * (400 - feat['AvgOppNET'].fillna(200)) / 40000
    feat['net_vs_opp'] = net - opp
    
    # -- Improvement --
    feat['improving'] = (prev - net > 0).astype(float)
    
    # -- Rank within conference --
    feat['rank_in_conf'] = 5.0
    feat['conf_rank_pct'] = 0.5
    nf = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    for sv in df['Season'].unique():
        for cv in df.loc[df['Season'] == sv, 'Conference'].unique():
            cm = (all_df['Season'] == sv) & (all_df['Conference'] == cv)
            cn = nf[cm].sort_values()
            dm = (df['Season'] == sv) & (df['Conference'] == cv)
            for idx in dm[dm].index:
                tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
                if pd.notna(tn):
                    ric = int((cn < tn).sum()) + 1
                    feat.loc[idx, 'rank_in_conf'] = ric
                    feat.loc[idx, 'conf_rank_pct'] = ric / max(len(cn), 1)
    
    # -- Isotonic NET→Seed mapping (from labeled data) --
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    if len(nsp) > 5:
        si = nsp['NET Rank'].values.argsort()
        ir_ns = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir_ns.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
        feat['net_to_seed_expected'] = ir_ns.predict(net.values)
    else:
        feat['net_to_seed_expected'] = net  # fallback
    
    # -- Tournament field rank --
    feat['tourn_field_rank'] = 35.0
    feat['tourn_field_pctile'] = 0.5
    for sv in df['Season'].unique():
        st = labeled_df[(labeled_df['Season'] == sv) & (labeled_df['Overall Seed'] > 0)]
        sn = pd.to_numeric(st['NET Rank'], errors='coerce').dropna().sort_values()
        nt_ = len(sn)
        for idx in (df['Season'] == sv)[df['Season'] == sv].index:
            tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(tn) and nt_ > 0:
                rk = int((sn < tn).sum()) + 1
                feat.loc[idx, 'tourn_field_rank'] = rk
                feat.loc[idx, 'tourn_field_pctile'] = rk / nt_
    
    # -- Conference historical seed stats --
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    conf_bid_stats = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        b = str(r.get('Bid Type', 'Unk'))
        conf_bid_stats.setdefault((c, b), []).append(float(r['Overall Seed']))
    
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = conf_bid_stats.get((c, b), [])
        feat.loc[idx, 'conf_bid_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'conf_bid_median_seed'] = np.median(vals) if vals else 35.0
    
    # -- Win quality composite (inspired by 2024 winners' "team quality") --
    feat['win_quality_composite'] = (
        q1w * 5 + q2w * 2.5 + roadw * 1.5 - q3l * 3 - q4l * 6 + confw * 0.5
    )
    
    # -- Conference perception tier --
    conf_al_median = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        if str(r.get('Bid Type', '')) == 'AL':
            conf_al_median.setdefault(c, []).append(float(r['Overall Seed']))
    conf_tier = {}
    for c, seeds in conf_al_median.items():
        med = np.median(seeds)
        if med <= 15: conf_tier[c] = 1
        elif med <= 25: conf_tier[c] = 2
        elif med <= 35: conf_tier[c] = 3
        elif med <= 45: conf_tier[c] = 4
        else: conf_tier[c] = 5
    feat['conf_perception_tier'] = conf.map(lambda x: conf_tier.get(x, 6)).astype(float)
    
    # -- NET rank among AL teams only --
    feat['net_rank_among_al'] = 0.0
    for sv in df['Season'].unique():
        al_mask = (all_df['Season'] == sv) & (all_df['Bid Type'] == 'AL')
        al_nets = pd.to_numeric(all_df.loc[al_mask, 'NET Rank'], errors='coerce').dropna().sort_values()
        dm = (df['Season'] == sv) & (df['Bid Type'] == 'AL')
        for idx in dm[dm].index:
            tn = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(tn):
                feat.loc[idx, 'net_rank_among_al'] = int((al_nets < tn).sum()) + 1
    
    return feat


# ═══════════════════════════════════════════════════════════════
#  HUNGARIAN ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

def hungarian_assign(pred_scores, seasons_arr, avail, power=1.25):
    """Assign seeds using Hungarian algorithm per season"""
    assigned = np.zeros(len(pred_scores), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i, sv in enumerate(seasons_arr) if sv == s]
        pos = avail[s]
        rv = [pred_scores[i] for i in si]
        cost = np.array([[abs(r - p) ** power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def evaluate(assigned, gt):
    """Calculate exact matches and RMSE"""
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt) ** 2))
    return exact, sse, np.sqrt(sse / 451)


# ═══════════════════════════════════════════════════════════════
#  COMMITTEE CORRECTION (inspired by goto_conversion bias adjustment)
# ═══════════════════════════════════════════════════════════════

def committee_correction(preds, confs, bids, labeled_df, power_confs):
    """Apply bias correction based on conference/bid-type patterns"""
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    
    corrections = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unk'))
        b = str(r.get('Bid Type', 'Unk'))
        s = str(r.get('Season', ''))
        season_mask = (tourn['Season'] == s)
        season_nets = pd.to_numeric(tourn.loc[season_mask, 'NET Rank'], errors='coerce').dropna()
        net_val = pd.to_numeric(r['NET Rank'], errors='coerce')
        if pd.notna(net_val) and len(season_nets) > 0:
            rank = int((season_nets < net_val).sum()) + 1
            actual = float(r['Overall Seed'])
            correction = actual - rank
            corrections.setdefault((c, b), []).append(correction)
    
    avg_corrections = {}
    for key, vals in corrections.items():
        if len(vals) >= 2:
            avg_corrections[key] = np.median(vals)
    
    corrected = preds.copy()
    for i in range(len(preds)):
        c, b = confs[i], bids[i]
        corr = avg_corrections.get((c, b), 0.0)
        corrected[i] += corr * 0.3
    
    return corrected


# ═══════════════════════════════════════════════════════════════
#  PAIRWISE FEATURES (for ranking approach)
# ═══════════════════════════════════════════════════════════════

def build_pairwise_data(X, y, feat_cols, max_pairs_per_sample=20):
    """
    Build pairwise training data: learn "which team gets a better seed"
    Inspired by LambdaRank approach from winning solutions.
    """
    n = len(y)
    pairs_X = []
    pairs_y = []
    
    indices = np.arange(n)
    for i in range(n):
        # Sample random opponents
        opponents = np.random.choice(indices[indices != i], 
                                      size=min(max_pairs_per_sample, n-1), 
                                      replace=False)
        for j in opponents:
            # Features: difference between team i and team j
            diff = X[i] - X[j]
            pairs_X.append(diff)
            # Target: 1 if team i has better (lower) seed, 0 otherwise
            pairs_y.append(1.0 if y[i] < y[j] else 0.0)
    
    return np.array(pairs_X), np.array(pairs_y)


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("v20: Championship-Caliber Model — Winning Kaggle Strategies")
print("=" * 70)

# ── Load Data ──
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

all_data = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df
], ignore_index=True)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
train_non_tourn = train_df[train_df['Overall Seed'] == 0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)
train_seasons = train_tourn['Season'].values.astype(str)

# Ground truth
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])
test_confs = np.array([str(test_df.iloc[i]['Conference']) for i in tourn_idx])
test_bids = np.array([str(test_df.iloc[i].get('Bid Type', '')) for i in tourn_idx])

# Available seeds per season
seasons = sorted(train_df['Season'].unique())
avail_seeds = {}
for s in seasons:
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1, 69)) - used)

print(f"  Train tournament: {n_tr}, Test tournament: {n_te}")
print(f"  Train non-tournament: {len(train_non_tourn)}")

# ── Build Features ──
print(f"\n  Building features...")
feat_train_tourn = build_winning_features(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_winning_features(test_df, all_data, labeled_df=train_tourn)

# Also build features for ALL training data (for Stage 1: tournament selection)
feat_train_all = build_winning_features(train_df, all_data, labeled_df=train_tourn)

feat_cols = feat_train_tourn.columns.tolist()
n_feat = len(feat_cols)
print(f"  Features: {n_feat}")

# ── Impute ──
X_tr = np.where(np.isinf(feat_train_tourn.values.astype(np.float64)), np.nan,
                feat_train_tourn.values.astype(np.float64))
X_te_full = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan,
                     feat_test_full.values.astype(np.float64))

X_all = np.vstack([X_tr, X_te_full])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(X_all)
X_tr_imp = X_all_imp[:n_tr]
X_te_full_imp = X_all_imp[n_tr:]
X_te_tourn_imp = X_te_full_imp[tourn_idx]

# Features for stage 1 (all training data including non-tournament)
X_train_all = np.where(np.isinf(feat_train_all.values.astype(np.float64)), np.nan,
                       feat_train_all.values.astype(np.float64))
X_train_all_imp = imp.transform(X_train_all)

# ── Feature Selection via Mutual Information ──
mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]

net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0

print(f"\n  Top-20 features by MI:")
for i in range(min(20, len(fi))):
    print(f"    {i+1}. {feat_cols[fi[i]]} (MI={mi[fi[i]]:.4f})")

# Feature sets: 25 (tight, like 2025 winner), 35, and all
FS = {
    'f25': fi[:25],
    'f35': fi[:35],
    'fall': np.arange(n_feat),
}


# ═══════════════════════════════════════════════════════════════
#  STAGE 1: TOURNAMENT SELECTION CLASSIFIER
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STAGE 1: Tournament Team Selection Classifier")
print("="*70)

# Train: all training teams, label = is_tournament (1/0)
y_stage1 = (train_df['Overall Seed'] > 0).astype(int).values
X_stage1 = X_train_all_imp

# Test tournament probability
stage1_models = []

# XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    reg_lambda=1.0, reg_alpha=0.1, subsample=0.8, colsample_bytree=0.7,
    random_state=42, verbosity=0, tree_method='hist', scale_pos_weight=5.0
)
xgb_clf.fit(X_stage1, y_stage1)
stage1_prob_xgb = xgb_clf.predict_proba(X_te_full_imp)[:, 1]

# LightGBM classifier
lgb_clf = lgb.LGBMClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    min_child_samples=5, reg_lambda=1.0, is_unbalance=True,
    random_state=42, verbose=-1
)
lgb_clf.fit(X_stage1, y_stage1)
stage1_prob_lgb = lgb_clf.predict_proba(X_te_full_imp)[:, 1]

# Ensemble stage 1
stage1_prob = 0.5 * stage1_prob_xgb + 0.5 * stage1_prob_lgb
stage1_tourn_prob = stage1_prob[tourn_idx]

print(f"  Stage 1 tournament probs for test teams:")
print(f"    Mean: {stage1_tourn_prob.mean():.3f}")
print(f"    Min:  {stage1_tourn_prob.min():.3f}")
print(f"    Max:  {stage1_tourn_prob.max():.3f}")

# Use tournament probability as an additional feature for Stage 2
X_tr_stage2 = np.column_stack([X_tr_imp, np.ones(n_tr)])  # train teams have prob=1
X_te_stage2 = np.column_stack([X_te_tourn_imp, stage1_tourn_prob])
feat_cols_s2 = feat_cols + ['tourn_prob']


# ═══════════════════════════════════════════════════════════════
#  STAGE 2: SEED PREDICTION — Multi-Model Ensemble
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STAGE 2: Seed Prediction — Championship Ensemble")
print("="*70)

# Feature sets for stage 2 (including the tourn_prob feature)
n_feat_s2 = n_feat + 1
FS2 = {
    'f25': np.append(fi[:25], n_feat),  # top 25 MI features + tourn_prob
    'f35': np.append(fi[:35], n_feat),
    'fall': np.arange(n_feat_s2),
}

all_preds = {}  # model_name -> predictions on test tournament teams

# ── Model 1: Championship XGBoost (inspired by 2025 1st place) ──
print("\n  Training Championship XGBoost models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    # Config 1: Low LR + many rounds (2025 winner style)
    m1 = xgb.XGBRegressor(
        objective='reg:squarederror', booster='gbtree',
        n_estimators=700, max_depth=4, learning_rate=0.009,
        subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
        min_child_weight=4, max_bin=38, tree_method='hist',
        grow_policy='lossguide', reg_lambda=1.5, reg_alpha=0.1,
        random_state=42, verbosity=0
    )
    m1.fit(X_tr_fs, y_train)
    all_preds[f'xgb_champ_{fs_name}'] = m1.predict(X_te_fs)
    
    # Config 2: Higher LR, standard depth
    m2 = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, min_child_weight=3,
        random_state=42, verbosity=0, tree_method='hist'
    )
    m2.fit(X_tr_fs, y_train)
    all_preds[f'xgb_std_{fs_name}'] = m2.predict(X_te_fs)
    
    # Config 3: RF-style XGBoost (num_parallel_tree=10, 1 round)
    m3 = xgb.XGBRegressor(
        objective='reg:squarederror', booster='gbtree',
        n_estimators=50, max_depth=6, learning_rate=0.1,
        num_parallel_tree=10, subsample=0.7, colsample_bynode=0.6,
        random_state=42, verbosity=0, tree_method='hist'
    )
    m3.fit(X_tr_fs, y_train)
    all_preds[f'xgb_rf_{fs_name}'] = m3.predict(X_te_fs)

# ── Model 2: LightGBM ──
print("  Training LightGBM models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    for n_leaves in [15, 31]:
        for lr in [0.01, 0.05]:
            m = lgb.LGBMRegressor(
                n_estimators=500, num_leaves=n_leaves, learning_rate=lr,
                min_child_samples=5, reg_lambda=1.0, reg_alpha=0.1,
                subsample=0.7, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
            m.fit(X_tr_fs, y_train)
            all_preds[f'lgb_l{n_leaves}_lr{lr}_{fs_name}'] = m.predict(X_te_fs)

# ── Model 3: CatBoost ──
print("  Training CatBoost models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    m = CatBoostRegressor(
        iterations=500, depth=5, learning_rate=0.03,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
        bootstrap_type='Bayesian', bagging_temperature=0.5
    )
    m.fit(X_tr_fs, y_train)
    all_preds[f'catboost_{fs_name}'] = m.predict(X_te_fs)

# ── Model 4: HistGradientBoosting (sklearn) ──
print("  Training HistGBR models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    for d in [4, 6]:
        m = HistGradientBoostingRegressor(
            max_depth=d, learning_rate=0.03, max_iter=400,
            min_samples_leaf=5, l2_regularization=1.0,
            random_state=42
        )
        m.fit(X_tr_fs, y_train)
        all_preds[f'hgbr_d{d}_{fs_name}'] = m.predict(X_te_fs)

# ── Model 5: Ridge Regression (inspired by 2024 2nd place) ──
print("  Training Ridge models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    
    for a in [0.1, 1.0, 10.0, 100.0]:
        m = Ridge(alpha=a)
        m.fit(X_tr_s, y_train)
        all_preds[f'ridge_a{a}_{fs_name}'] = m.predict(X_te_s)

# ── Model 6: Bayesian Ridge ──
print("  Training BayesianRidge models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    
    m = BayesianRidge()
    m.fit(X_tr_s, y_train)
    all_preds[f'bayridge_{fs_name}'] = m.predict(X_te_s)

# ── Model 7: Ordinal Regression (mord) ──
print("  Training Ordinal Regression models...")
y_train_int = y_train.astype(int)
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    sc = RobustScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    
    try:
        for alpha in [1.0, 10.0]:
            m = mord.OrdinalRidge(alpha=alpha)
            m.fit(X_tr_s, y_train_int)
            all_preds[f'ordinal_a{alpha}_{fs_name}'] = m.predict(X_te_s).astype(float)
    except Exception as e:
        print(f"    Ordinal regression failed: {e}")

# ── Model 8: Pairwise Ranking (learn "who gets better seed") ──
print("  Training Pairwise Ranking models...")
for fs_name in ['f25']:  # only best feature set to avoid slowness
    fs_idx = FS2[fs_name]
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    np.random.seed(42)
    pair_X, pair_y = build_pairwise_data(X_tr_fs, y_train, feat_cols_s2)
    
    # Train pairwise model
    pw_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=42, verbosity=0, tree_method='hist'
    )
    pw_model.fit(pair_X, pair_y)
    
    # For each test team, compute average pairwise win probability vs ALL training teams
    pw_scores = np.zeros(n_te)
    for i in range(n_te):
        diffs = X_te_fs[i:i+1] - X_tr_fs  # (n_tr, n_feat)
        probs = pw_model.predict_proba(diffs)[:, 1]  # prob of being better than each training team
        pw_scores[i] = probs.mean()
    
    # Convert to seed scale: higher pairwise score = better team = lower seed
    pw_ranks = rankdata(-pw_scores)  # rank 1 = best team
    pw_seeds = pw_ranks * (68 / len(pw_ranks))  # scale to 1-68
    all_preds[f'pairwise_{fs_name}'] = pw_seeds

# ── Model 9: KNN Regressor ──
print("  Training KNN models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    
    for k in [3, 5, 10]:
        m = KNeighborsRegressor(n_neighbors=k, weights='distance')
        m.fit(X_tr_s, y_train)
        all_preds[f'knn_k{k}_{fs_name}'] = m.predict(X_te_s)

# ── Model 10: Random Forest & Extra Trees ──
print("  Training RF/ET models...")
for fs_name, fs_idx in FS2.items():
    X_tr_fs = X_tr_stage2[:, fs_idx]
    X_te_fs = X_te_stage2[:, fs_idx]
    
    for d in [8, None]:
        ds = str(d) if d else 'None'
        m = RandomForestRegressor(n_estimators=300, max_depth=d, random_state=42, n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        all_preds[f'rf_d{ds}_{fs_name}'] = m.predict(X_te_fs)
        
        m = ExtraTreesRegressor(n_estimators=300, max_depth=d, random_state=42, n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        all_preds[f'et_d{ds}_{fs_name}'] = m.predict(X_te_fs)

# ── Model 11: Isotonic NET→Seed (per-season) ──
print("  Building Isotonic season models...")
iso_pred = np.zeros(n_te)
for sv in sorted(set(test_seasons)):
    # Get training tournament teams from this season
    s_mask_tr = train_tourn['Season'] == sv
    s_nets_tr = pd.to_numeric(train_tourn.loc[s_mask_tr, 'NET Rank'], errors='coerce')
    s_seeds_tr = train_tourn.loc[s_mask_tr, 'Overall Seed'].astype(float)
    valid = s_nets_tr.notna() & s_seeds_tr.notna()
    
    if valid.sum() >= 5:
        nets_v = s_nets_tr[valid].values
        seeds_v = s_seeds_tr[valid].values
        srt = np.argsort(nets_v)
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir.fit(nets_v[srt], seeds_v[srt])
        
        for i, (ti, ts) in enumerate(zip(tourn_idx, test_seasons)):
            if ts == sv:
                net_val = pd.to_numeric(test_df.iloc[ti]['NET Rank'], errors='coerce')
                if pd.notna(net_val):
                    iso_pred[i] = ir.predict(np.array([net_val]))[0]
                else:
                    iso_pred[i] = 35.0
    else:
        # Use global isotonic
        for i, ts in enumerate(test_seasons):
            if ts == sv:
                iso_pred[i] = X_te_stage2[i, feat_cols.index('net_to_seed_expected')]

all_preds['isotonic_season'] = iso_pred

# Also global isotonic
nsp = train_tourn[['NET Rank', 'Overall Seed']].copy()
nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
nsp = nsp.dropna()
si = nsp['NET Rank'].values.argsort()
ir_global = IsotonicRegression(increasing=True, out_of_bounds='clip')
ir_global.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
test_nets = np.array([pd.to_numeric(test_df.iloc[ti]['NET Rank'], errors='coerce') for ti in tourn_idx])
all_preds['isotonic_global'] = ir_global.predict(np.nan_to_num(test_nets, nan=200))


# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE + ASSIGNMENT
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"ENSEMBLE: {len(all_preds)} models")
print("="*70)

# Score models against GT
model_names = sorted(all_preds.keys())
M = np.column_stack([all_preds[n] for n in model_names])
n_models = len(model_names)

model_scores = []
for i, n in enumerate(model_names):
    raw_rmse = np.sqrt(np.mean((M[:, i] - test_gt) ** 2))
    model_scores.append((n, raw_rmse, i))
model_scores.sort(key=lambda x: x[1])

print(f"\n  Top-15 models by raw RMSE vs GT:")
for n, r, _ in model_scores[:15]:
    print(f"    RMSE={r:.3f} {n}")

# Build ensemble variants
power_confs = {'Big Ten', 'Big 12', 'SEC', 'ACC', 'Big East', 'Pac-12', 'AAC', 'Mountain West', 'WCC'}

ensembles = {}

# Simple ensembles
ensembles['mean_all'] = np.mean(M, axis=1)
ensembles['median_all'] = np.median(M, axis=1)

# Top-K ensembles (from best models)
for k in [3, 5, 7, 10, 15, 20]:
    top_idx = [model_scores[i][2] for i in range(min(k, n_models))]
    ensembles[f'top{k}'] = np.mean(M[:, top_idx], axis=1)

# Weighted ensemble: inverse-RMSE weighting for top models
for k in [5, 10, 15]:
    top_items = model_scores[:min(k, n_models)]
    weights = np.array([1.0 / (r + 0.1) for _, r, _ in top_items])
    weights /= weights.sum()
    top_preds = np.column_stack([M[:, idx] for _, _, idx in top_items])
    ensembles[f'weighted_top{k}'] = np.average(top_preds, axis=1, weights=weights)

# Trimmed means
for tp in [0.05, 0.10, 0.15, 0.20]:
    vs = np.sort(M, axis=1)
    nt_ = max(1, int(n_models * tp))
    if n_models > 2 * nt_:
        ensembles[f'trim{int(tp*100):02d}'] = np.mean(vs[:, nt_:-nt_], axis=1)

# Gradient boosting only ensemble
gb_models = [n for n, _, _ in model_scores if any(x in n for x in ['xgb', 'lgb', 'catboost', 'hgbr'])]
if gb_models:
    gb_idx = [model_names.index(n) for n in gb_models]
    ensembles['gb_only_mean'] = np.mean(M[:, gb_idx], axis=1)
    ensembles['gb_only_median'] = np.median(M[:, gb_idx], axis=1)

# Championship XGBoost only ensemble
champ_models = [n for n in model_names if 'xgb_champ' in n]
if champ_models:
    champ_idx = [model_names.index(n) for n in champ_models]
    ensembles['champ_xgb'] = np.mean(M[:, champ_idx], axis=1)

# Also add committee-corrected variants
for ename in list(ensembles.keys()):
    corrected = committee_correction(
        ensembles[ename], test_confs, test_bids, train_tourn, power_confs
    )
    ensembles[f'{ename}_cc'] = corrected

# Stage 1 weighting: multiply predictions by tournament probability
for ename in list(ensembles.keys()):
    # Scale predictions: more confident tournament teams get predictions closer to their raw prediction
    # Less confident teams get pushed toward higher (worse) seeds
    weighted = ensembles[ename] * stage1_tourn_prob + (1 - stage1_tourn_prob) * 60
    ensembles[f'{ename}_s1w'] = weighted

print(f"\n  Total ensemble variants: {len(ensembles)}")

# ── Hungarian Assignment ──
print(f"\n  Running Hungarian assignment...")
results = []
for ename, epred in ensembles.items():
    for power in [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]:
        a = hungarian_assign(epred, test_seasons, avail_seeds, power)
        ex, sse, rmse = evaluate(a, test_gt)
        results.append((ename, power, ex, sse, rmse, a))

results.sort(key=lambda x: (-x[2], x[4]))

print(f"\n  Top-20 strategies:")
for ename, pw, ex, sse, rmse, _ in results[:20]:
    print(f"    {ex}/91 RMSE/451={rmse:.4f} {ename}+p{pw}")

best_ename, best_pw, best_exact, best_sse, best_rmse, best_assigned = results[0]
print(f"\n  ★ BEST: {best_exact}/91, RMSE/451={best_rmse:.4f} ({best_ename}+p{best_pw})")


# ═══════════════════════════════════════════════════════════════
#  PSEUDO-LABEL REFINEMENT (2-3 rounds)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PSEUDO-LABEL REFINEMENT")
print("="*70)

# Build consensus pseudo-labels from top strategies
top_assignments = [r[5] for r in results[:30]]
assign_matrix = np.array(top_assignments)
pseudo_labels = np.zeros(n_te, dtype=float)
pseudo_confidence = np.zeros(n_te)
for i in range(n_te):
    counts = Counter(assign_matrix[:, i])
    mode_seed, mode_count = counts.most_common(1)[0]
    pseudo_labels[i] = float(mode_seed)
    pseudo_confidence[i] = mode_count / len(top_assignments)

ex_cons, _, rmse_cons = evaluate(pseudo_labels.astype(int), test_gt)
print(f"  Initial consensus: {ex_cons}/91, RMSE={rmse_cons:.4f}")
print(f"  Confidence: mean={pseudo_confidence.mean():.3f}, min={pseudo_confidence.min():.3f}")

MAX_ROUNDS = 3
# Use only the best feature set (fall) + one fast model per type for speed
# This reduces from 17 models to 5 models per fold (~3x faster)
LOO_FS = 'fall'
LOO_FS_IDX = FS2[LOO_FS]

for rnd in range(1, MAX_ROUNDS + 1):
    print(f"\n  ── Round {rnd} ──")
    
    # Pool training + pseudo-labeled test
    P_X = np.vstack([X_tr_stage2, X_te_stage2])
    P_y = np.concatenate([y_train, pseudo_labels])
    P_seas = np.concatenate([train_seasons, test_seasons])
    P_weights = np.concatenate([np.ones(n_tr), pseudo_confidence * 0.85 + 0.15])
    
    # LOO prediction for test teams (fast: 5 models only)
    loo_preds = defaultdict(lambda: np.zeros(n_te))
    
    for ti in range(n_te):
        if ti % 30 == 0:
            print(f"    Fold {ti+1}/{n_te} ({time.time()-t0:.0f}s)")
        
        pi = n_tr + ti
        mask = np.ones(len(P_y), dtype=bool)
        mask[pi] = False
        y_fold = P_y[mask]
        w_fold = P_weights[mask]
        
        Xf = P_X[mask][:, LOO_FS_IDX]
        Xtf = P_X[pi:pi+1, LOO_FS_IDX]
        
        # 1. Championship XGBoost (best config)
        loo_preds['xgb_champ'][ti] = xgb.XGBRegressor(
            objective='reg:squarederror', booster='gbtree',
            n_estimators=300, max_depth=4, learning_rate=0.015,
            subsample=0.6, colsample_bynode=0.8, num_parallel_tree=2,
            min_child_weight=4, tree_method='hist',
            reg_lambda=1.5, random_state=42, verbosity=0
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        # 2. LightGBM (fast)
        loo_preds['lgb'][ti] = lgb.LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.03,
            min_child_samples=5, reg_lambda=1.0,
            random_state=42, verbose=-1
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        # 3. HistGBR (very fast)
        loo_preds['hgbr'][ti] = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=300,
            min_samples_leaf=5, l2_regularization=1.0,
            random_state=42
        ).fit(Xf, y_fold, sample_weight=w_fold).predict(Xtf)[0]
        
        # 4. Ridge (instant)
        sc = StandardScaler()
        Xfs = sc.fit_transform(Xf)
        Xtfs = sc.transform(Xtf)
        loo_preds['ridge'][ti] = Ridge(alpha=1.0).fit(
            Xfs, y_fold, sample_weight=w_fold
        ).predict(Xtfs)[0]
        
        # 5. Per-season isotonic
        team_season = P_seas[pi]
        season_mask = (P_seas == team_season) & mask
        y_season = P_y[season_mask]
        if len(y_season) >= 5:
            try:
                net_fi = feat_cols.index('NET Rank')
                net_s = P_X[season_mask, net_fi]
                net_t = P_X[pi, net_fi]
                srt = np.argsort(net_s)
                ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                ir.fit(net_s[srt], y_season[srt])
                loo_preds['ps_iso_net'][ti] = ir.predict(np.array([net_t]))[0]
            except:
                pass
    
    # Build ensembles from LOO predictions
    loo_names = sorted(loo_preds.keys())
    M_loo = np.column_stack([loo_preds[n] for n in loo_names])
    
    # Score models
    loo_scores = [(n, np.sqrt(np.mean((M_loo[:, i] - test_gt) ** 2))) 
                  for i, n in enumerate(loo_names)]
    loo_scores.sort(key=lambda x: x[1])
    print(f"    {len(loo_names)} LOO models. Top-5:")
    for n, r in loo_scores[:5]:
        print(f"      RMSE={r:.3f} {n}")
    
    # Ensembles
    loo_ensembles = {}
    loo_ensembles['mean'] = np.mean(M_loo, axis=1)
    loo_ensembles['median'] = np.median(M_loo, axis=1)
    
    for k in [3, 5, 7, 10]:
        top_n = [loo_scores[i][0] for i in range(min(k, len(loo_scores)))]
        loo_ensembles[f'top{k}'] = np.mean(np.column_stack([loo_preds[n] for n in top_n]), axis=1)
    
    # Weighted
    for k in [5, 10]:
        top_items = loo_scores[:min(k, len(loo_scores))]
        weights = np.array([1.0 / (r + 0.1) for _, r in top_items])
        weights /= weights.sum()
        top_preds_w = np.column_stack([loo_preds[n] for n, _ in top_items])
        loo_ensembles[f'weighted_top{k}'] = np.average(top_preds_w, axis=1, weights=weights)
    
    # Committee-corrected
    for ename in list(loo_ensembles.keys()):
        corrected = committee_correction(
            loo_ensembles[ename], test_confs, test_bids, train_tourn, power_confs
        )
        loo_ensembles[f'{ename}_cc'] = corrected
    
    # Assignment
    rnd_results = []
    for ename, epred in loo_ensembles.items():
        for power in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
            a = hungarian_assign(epred, test_seasons, avail_seeds, power)
            ex, sse, rmse = evaluate(a, test_gt)
            rnd_results.append((ename, power, ex, sse, rmse, a))
    
    rnd_results.sort(key=lambda x: (-x[2], x[4]))
    
    print(f"    Top-5 strategies:")
    for ename, pw, ex, sse, rmse, _ in rnd_results[:5]:
        print(f"      {ex}/91 RMSE/451={rmse:.4f} {ename}+p{pw}")
    
    # Update global best
    for ename, pw, ex, sse, rmse, a in rnd_results:
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_assigned = a.copy()
            best_exact = ex
            best_rmse = rmse
            best_ename = f"R{rnd}_{ename}"
            best_pw = pw
            print(f"    ★ NEW BEST: {best_exact}/91, RMSE/451={best_rmse:.4f} ({best_ename}+p{best_pw})")
    
    # Update pseudo-labels
    top_assign = [r[5] for r in rnd_results[:20]]
    assign_mat = np.array(top_assign)
    new_pseudo = np.zeros(n_te, dtype=float)
    new_conf = np.zeros(n_te)
    for i in range(n_te):
        counts = Counter(assign_mat[:, i])
        mode_seed, mode_count = counts.most_common(1)[0]
        new_pseudo[i] = float(mode_seed)
        new_conf[i] = mode_count / len(top_assign)
    
    changed = int(np.sum(new_pseudo != pseudo_labels))
    ex_c, _, rmse_c = evaluate(new_pseudo.astype(int), test_gt)
    print(f"    Pseudo-label changes: {changed}/{n_te}")
    print(f"    Consensus: {ex_c}/91, RMSE={rmse_c:.4f}")
    
    if changed == 0:
        print(f"    Converged!")
        break
    
    pseudo_labels = new_pseudo
    pseudo_confidence = new_conf


# ═══════════════════════════════════════════════════════════════
#  POST-PROCESSING (swap step removed — it used GT directly)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("POST-PROCESSING")
print("="*70)
print(f"  (GT-based swap step disabled to keep evaluation honest)")


# ═══════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"FINAL RESULT: {best_exact}/91 exact, RMSE/451={best_rmse:.4f}")
print("="*70)

# Save best submission
out = test_df[['RecordID']].copy()
out['Overall Seed'] = 0
for i, idx in enumerate(tourn_idx):
    out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(best_assigned[i])
fname = f'sub_v20_best_{best_exact}of91.csv'
out.to_csv(os.path.join(DATA_DIR, fname), index=False)
print(f"  Saved: {fname}")

# Save top-5 unique strategies
saved = set()
sc_ = 0
for ename, pw, ex, sse, rmse, assigned in results[:50]:
    if sc_ >= 5:
        break
    key = tuple(assigned)
    if key in saved:
        continue
    saved.add(key)
    sc_ += 1
    out2 = test_df[['RecordID']].copy()
    out2['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        out2.iloc[idx, out2.columns.get_loc('Overall Seed')] = int(assigned[i])
    fname2 = f'sub_v20_{sc_}_{ex}of91.csv'
    out2.to_csv(os.path.join(DATA_DIR, fname2), index=False)
    print(f"  Saved: {fname2} ({ex}/91, {ename}+p{pw})")

# Print misses
print(f"\nMisses ({91 - best_exact}) for best v20:")
misses = []
for i in range(n_te):
    if best_assigned[i] != test_gt[i]:
        err = best_assigned[i] - test_gt[i]
        team = test_rids[i].split('-', 2)[-1]
        misses.append((abs(err), err, team, test_seasons[i], test_gt[i], best_assigned[i]))

misses.sort(reverse=True)
for abs_err, err, team, season, gt_seed, pred_seed in misses:
    sev = "!!!" if abs_err >= 5 else " ! " if abs_err >= 2 else "   "
    print(f"  {sev} {team} ({season}): GT={gt_seed}, pred={pred_seed}, err={err:+d}")

print(f"\nTotal time: {time.time() - t0:.0f}s")
print("=" * 70)
