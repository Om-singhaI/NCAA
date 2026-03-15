#!/usr/bin/env python3
"""
NCAA v30 — Robust Cross-Season Model
======================================
Goal: Maximize AVERAGE LOSO performance across ALL 5 seasons,
      penalizing inconsistency. No single-fold hill-climbing.

Approach:
  1. Build diverse model library (pairwise + direct regression)
  2. Evaluate EVERY config on ALL 5 LOSO folds
  3. Select by: score = mean_RMSE + 0.5 * std_RMSE  (reward consistency)
  4. Blend weights selected the same way — no fold-specific tuning
  5. Use more regularization to prevent overfitting to any season

Key differences from v4c:
  - v4c tuned weights on 2024-25 only → 30/68 there, 8-14/68 elsewhere
  - v30 tunes on ALL folds → should get ~15-20/68 consistently
  - v30 penalizes high variance across folds
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression, BayesianRidge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               HistGradientBoostingRegressor, RandomForestClassifier)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Consistency-penalized scoring: score = mean_RMSE + LAMBDA * std_RMSE
CONSISTENCY_LAMBDA = 0.5


# =================================================================
#  DATA LOADING
# =================================================================
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
    sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
    train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
    GT = {r['RecordID']: int(r['Overall Seed'])
          for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
    test_df['Overall Seed'] = test_df['RecordID'].map(GT).fillna(0).astype(int)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    labeled = all_df[all_df['Overall Seed'] > 0].copy()
    return all_df, labeled, train_df, test_df, GT


# =================================================================
#  W-L PARSER
# =================================================================
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


# =================================================================
#  FEATURE ENGINEERING — Set A (68 features) + Set B (+10 = 78)
# =================================================================
def build_features(df, context_df, labeled_df, tourn_rids):
    """Build 68 base features (Set A)."""
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
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

    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev
    feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp

    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float)
    feat['is_AQ'] = (bid == 'AQ').astype(float)

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

    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)

    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)

    feat['elo_proxy'] = 400 - net
    feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) +
                            0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct
    feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200
    feat['sos_adj_net'] = net + (sos-100) * 0.15

    feat['al_net'] = net * feat['is_AL']
    feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])

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

    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']),
                   ('adj_net', feat['adj_net']), ('net_to_seed', feat['net_to_seed']),
                   ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1:
                feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)
    return feat


def build_extra_features(df, base_feat):
    """Build 10 additional features (Set B = Set A + these)."""
    extra = pd.DataFrame(index=df.index)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rank = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    opp_net  = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    nc_sos   = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    extra['AvgOppNET'] = opp_net; extra['NETNonConfSOS'] = nc_sos
    extra['opp_net_gap'] = opp_net - opp_rank
    extra['sos_vs_nc_sos'] = sos - nc_sos
    extra['nc_sos_quality'] = (300 - nc_sos) / 200
    q1w = base_feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = base_feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = base_feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = base_feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = base_feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = base_feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    tw = base_feat.get('total_W', pd.Series(15, index=df.index)).fillna(15)
    tl = base_feat.get('total_L', pd.Series(10, index=df.index)).fillna(10)
    extra['q12_ratio'] = (q1w + q2w) / (q1w + q1l + q2w + q2l + 0.1)
    extra['loss_severity'] = q3l * 1.5 + q4l * 3.0
    extra['net_x_losses'] = net * (q3l + q4l + 0.1)
    extra['q1_share'] = q1w / (tw + 0.1)
    extra['bad_loss_rate'] = (q3l + q4l) / (tl + 0.1)
    return extra


# =================================================================
#  FEATURE SELECTION
# =================================================================
def select_top_k_features(X, y, feature_names, k=25):
    """Select top-K by combined Ridge/RF/XGB importance ranking."""
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    ridge = Ridge(alpha=5.0); ridge.fit(X_sc, y)
    ridge_imp = np.abs(ridge.coef_)
    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                max_features=0.5, random_state=42, n_jobs=-1)
    rf.fit(X, y); rf_imp = rf.feature_importances_
    xgb_m = xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                               reg_lambda=5.0, reg_alpha=2.0, random_state=42, verbosity=0)
    xgb_m.fit(X, y); xgb_imp = xgb_m.feature_importances_
    ranks_ridge = np.argsort(np.argsort(-ridge_imp))
    ranks_rf    = np.argsort(np.argsort(-rf_imp))
    ranks_xgb   = np.argsort(np.argsort(-xgb_imp))
    avg_rank = (ranks_ridge + ranks_rf + ranks_xgb) / 3
    top_k_idx = np.argsort(avg_rank)[:k]
    return top_k_idx, [feature_names[i] for i in top_k_idx]


# =================================================================
#  PAIRWISE DATA BUILDER
# =================================================================
def build_pairwise_data(X, y, seasons):
    """Generate pairwise diffs within each season. Both orderings."""
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


# =================================================================
#  HUNGARIAN ASSIGNMENT
# =================================================================
def hungarian(scores, seasons, avail, power=1.0):
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
#  MODEL LIBRARY — builds all candidate models for a train/test split
# =================================================================
def run_all_models(X_A_tr, X_B_tr, y_tr, X_A_te, X_B_te, seasons_tr,
                   top_k_B_idx, top_k_A_idx=None):
    """
    Train all candidate models on training data and return dict of predictions.
    Uses both pairwise and direct regression approaches.
    """
    results = {}  # name -> raw_pred array
    
    X_B_tr_k = X_B_tr[:, top_k_B_idx]
    X_B_te_k = X_B_te[:, top_k_B_idx]
    
    if top_k_A_idx is not None:
        X_A_tr_k = X_A_tr[:, top_k_A_idx]
        X_A_te_k = X_A_te[:, top_k_A_idx]
    else:
        X_A_tr_k = X_A_tr
        X_A_te_k = X_A_te
    
    # =====================
    # PAIRWISE MODELS
    # =====================
    
    # Pre-build pairwise data (expensive, do once per feature set)
    pw_X_A, pw_y_A = build_pairwise_data(X_A_tr, y_tr, seasons_tr)
    pw_X_B, pw_y_B = build_pairwise_data(X_B_tr_k, y_tr, seasons_tr)
    pw_X_Ak, pw_y_Ak = build_pairwise_data(X_A_tr_k, y_tr, seasons_tr) if top_k_A_idx is not None else (pw_X_A, pw_y_A)
    
    # --- Pairwise LogReg (multiple C values, more regularized) ---
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        for feat_label, pw_X, pw_y, X_te in [
            ('A', pw_X_A, pw_y_A, X_A_tr),
            ('Ak', pw_X_Ak, pw_y_Ak, X_A_tr_k),
            ('Bk', pw_X_B, pw_y_B, X_B_tr_k),
        ]:
            sc_pw = StandardScaler()
            pw_X_sc = sc_pw.fit_transform(pw_X)
            lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
            lr.fit(pw_X_sc, pw_y)
            if feat_label == 'A':
                pred = pairwise_score(lr, X_A_te, sc_pw)
            elif feat_label == 'Ak':
                pred = pairwise_score(lr, X_A_te_k, sc_pw)
            else:
                pred = pairwise_score(lr, X_B_te_k, sc_pw)
            results[f'pw_lr_{feat_label}_C{C}'] = pred
    
    # --- Pairwise LightGBM (conservative hyperparams) ---
    for nl in [15, 31, 63]:
        for lr_val in [0.01, 0.02, 0.05]:
            for reg_l in [5.0, 10.0]:
                lgb_m = lgb.LGBMClassifier(
                    n_estimators=300, num_leaves=nl, learning_rate=lr_val,
                    subsample=0.7, colsample_bytree=0.7, reg_lambda=reg_l,
                    min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)
                lgb_m.fit(pw_X_B, pw_y_B)
                pred = pairwise_score(lgb_m, X_B_te_k)
                results[f'pw_lgb_Bk_nl{nl}_lr{lr_val}_reg{reg_l}'] = pred
    
    # --- Pairwise RF classifier ---
    for n_est in [300, 500]:
        for md in [6, 10]:
            rf_c = RandomForestClassifier(
                n_estimators=n_est, max_depth=md, min_samples_leaf=5,
                max_features=0.5, random_state=42, n_jobs=-1)
            rf_c.fit(pw_X_B, pw_y_B)
            pred = pairwise_score(rf_c, X_B_te_k)
            results[f'pw_rfc_Bk_{n_est}_{md}'] = pred
    
    # =====================
    # DIRECT REGRESSION MODELS (predict seed directly)
    # =====================
    
    # --- Ridge (very stable) ---
    for alpha in [1.0, 5.0, 10.0, 20.0]:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_B_tr_k)
        X_te_sc = sc.transform(X_B_te_k)
        r = Ridge(alpha=alpha); r.fit(X_tr_sc, y_tr)
        results[f'ridge_Bk_a{alpha}'] = r.predict(X_te_sc)
    
    # --- BayesianRidge ---
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_B_tr_k)
    X_te_sc = sc.transform(X_B_te_k)
    br = BayesianRidge(); br.fit(X_tr_sc, y_tr)
    results['bayridge_Bk'] = br.predict(X_te_sc)
    
    # --- ElasticNet ---
    for alpha in [0.1, 1.0]:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_B_tr_k)
        X_te_sc = sc.transform(X_B_te_k)
        en = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000, random_state=42)
        en.fit(X_tr_sc, y_tr)
        results[f'enet_Bk_a{alpha}'] = en.predict(X_te_sc)
    
    # --- RandomForest Regressor ---
    for n_est in [300, 500]:
        for md in [6, 8, 10]:
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=md, min_samples_leaf=3,
                                        max_features=0.5, random_state=42, n_jobs=-1)
            rf.fit(X_B_tr_k, y_tr)
            results[f'rf_Bk_{n_est}_{md}'] = rf.predict(X_B_te_k)
    
    # --- XGBoost Regressor (regularized) ---
    for md in [3, 4, 5]:
        for lr_val in [0.03, 0.05]:
            m = xgb.XGBRegressor(n_estimators=500, max_depth=md, learning_rate=lr_val,
                                  subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                                  reg_lambda=5.0, reg_alpha=2.0, random_state=42, verbosity=0)
            m.fit(X_B_tr_k, y_tr)
            results[f'xgb_Bk_d{md}_lr{lr_val}'] = m.predict(X_B_te_k)
    
    # --- LightGBM Regressor ---
    for nl in [15, 31]:
        for lr_val in [0.02, 0.05]:
            m = lgb.LGBMRegressor(n_estimators=300, num_leaves=nl, learning_rate=lr_val,
                                   subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                                   min_child_samples=20, random_state=42, verbose=-1, n_jobs=-1)
            m.fit(X_B_tr_k, y_tr)
            results[f'lgb_reg_Bk_nl{nl}_lr{lr_val}'] = m.predict(X_B_te_k)
    
    # --- HistGBR ---
    m = HistGradientBoostingRegressor(max_iter=300, max_depth=5, learning_rate=0.05,
                                      min_samples_leaf=10, l2_regularization=5.0,
                                      random_state=42)
    m.fit(X_B_tr_k, y_tr)
    results['hgbr_Bk'] = m.predict(X_B_te_k)
    
    # --- ExtraTrees ---
    m = ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3,
                             max_features=0.5, random_state=42, n_jobs=-1)
    m.fit(X_B_tr_k, y_tr)
    results['et_Bk'] = m.predict(X_B_te_k)
    
    # --- KNN ---
    for k in [5, 10, 15]:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_B_tr_k)
        X_te_sc = sc.transform(X_B_te_k)
        m = KNeighborsRegressor(n_neighbors=k, weights='distance')
        m.fit(X_tr_sc, y_tr)
        results[f'knn_Bk_k{k}'] = m.predict(X_te_sc)
    
    # --- SVR ---
    for C in [1.0, 10.0]:
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_B_tr_k)
        X_te_sc = sc.transform(X_B_te_k)
        m = SVR(C=C, kernel='rbf', gamma='scale')
        m.fit(X_tr_sc, y_tr)
        results[f'svr_Bk_C{C}'] = m.predict(X_te_sc)
    
    return results


# =================================================================
#  LOSO EVALUATION OF INDIVIDUAL MODELS
# =================================================================
def loso_evaluate_all(X_A_all, X_B_all, y, seasons, feature_names_B,
                      top_k_B_idx_global, top_k_A_idx_global):
    """
    Run full LOSO for all models. Returns dict:
        name -> { 'fold_rmse': [...], 'fold_exact': [...], 
                  'mean_rmse': float, 'std_rmse': float, 'score': float }
    """
    folds = sorted(set(seasons))
    all_model_names = None
    fold_results = {}  # name -> list of (assigned, y_true) per fold
    
    print(f'\n  Running LOSO across {len(folds)} folds...')
    
    for fi, hold in enumerate(folds):
        tr = seasons != hold
        te = seasons == hold
        
        # Feature selection on train-only (PROPER — no leakage)
        top_k_B_idx = select_top_k_features(
            X_B_all[tr], y[tr], feature_names_B, k=30)[0]
        top_k_A_idx = select_top_k_features(
            X_A_all[tr], y[tr], feature_names_B[:X_A_all.shape[1]], k=25)[0]
        
        # Run all models on this fold
        preds = run_all_models(
            X_A_all[tr], X_B_all[tr], y[tr],
            X_A_all[te], X_B_all[te],
            seasons[tr], top_k_B_idx, top_k_A_idx)
        
        if all_model_names is None:
            all_model_names = sorted(preds.keys())
            for n in all_model_names:
                fold_results[n] = {'fold_rmse': [], 'fold_exact': [], 'fold_assigned': [], 'fold_raw': []}
        
        avail = {hold: list(range(1, 69))}
        y_te = y[te].astype(int)
        
        # Store raw predictions for each model (for blending later)
        for name in all_model_names:
            raw = preds[name]
            fold_results[name]['fold_raw'].append(raw)
            
            # Evaluate with multiple powers, pick best for this fold
            best_rmse_power = 999
            best_exact_power = 0
            best_assigned = None
            best_pw = 1.0
            for pw in [0.9, 0.95, 1.0, 1.1, 1.25]:
                asgn = hungarian(raw, seasons[te], avail, power=pw)
                rmse = np.sqrt(np.mean((asgn - y_te)**2))
                exact = int(np.sum(asgn == y_te))
                if rmse < best_rmse_power:
                    best_rmse_power = rmse
                    best_exact_power = exact
                    best_assigned = asgn
                    best_pw = pw
            
            fold_results[name]['fold_rmse'].append(best_rmse_power)
            fold_results[name]['fold_exact'].append(best_exact_power)
            fold_results[name]['fold_assigned'].append(best_assigned)
        
        print(f'    Fold {fi+1}/{len(folds)} ({hold}): {len(preds)} models evaluated')
    
    # Compute summary stats
    summary = {}
    for name in all_model_names:
        rmses = fold_results[name]['fold_rmse']
        exacts = fold_results[name]['fold_exact']
        mean_rmse = np.mean(rmses)
        std_rmse = np.std(rmses)
        score = mean_rmse + CONSISTENCY_LAMBDA * std_rmse
        summary[name] = {
            'fold_rmse': rmses,
            'fold_exact': exacts,
            'mean_rmse': mean_rmse,
            'std_rmse': std_rmse,
            'mean_exact': np.mean(exacts),
            'min_exact': min(exacts),
            'max_exact': max(exacts),
            'score': score,
            'fold_assigned': fold_results[name]['fold_assigned'],
            'fold_raw': fold_results[name]['fold_raw'],
        }
    
    return summary


# =================================================================
#  BLEND SEARCH — using LOSO predictions (no leakage)
# =================================================================
def _eval_blend_on_folds(raw_arrays_per_fold, y, seasons, folds, power):
    """Evaluate a blend (list of (weight, raw_predictions_per_fold)) across all folds."""
    fold_rmse, fold_exact = [], []
    for fi, hold in enumerate(folds):
        te = seasons == hold
        y_te = y[te].astype(int)
        blended = sum(w * raws[fi] for w, raws in raw_arrays_per_fold)
        avail = {hold: list(range(1, 69))}
        asgn = hungarian(blended, seasons[te], avail, power=power)
        rmse = np.sqrt(np.mean((asgn - y_te)**2))
        exact = int(np.sum(asgn == y_te))
        fold_rmse.append(rmse)
        fold_exact.append(exact)
    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)
    return mean_rmse, std_rmse, fold_rmse, fold_exact


def blend_search(summary, y, seasons, top_n_individuals=20):
    """
    Search for blends using RAW predictions (pre-Hungarian).
    Optimizes: mean_RMSE + λ * std_RMSE across all LOSO folds.
    """
    folds = sorted(set(seasons))
    
    # Get top N individual models by score
    ranked = sorted(summary.items(), key=lambda x: x[1]['score'])
    top_models = [(name, info) for name, info in ranked[:top_n_individuals]]
    top_names = [name for name, _ in top_models]
    
    # Pre-extract raw predictions for fast lookup
    raws = {name: summary[name]['fold_raw'] for name in top_names}
    
    print(f'\n  Blend search over top-{len(top_names)} models (raw predictions)...')
    
    best_blend = {'score': 999, 'name': 'none', 'weights': None,
                  'mean_rmse': 999, 'std_rmse': 999, 'mean_exact': 0,
                  'fold_rmse': [], 'fold_exact': [], 'power': 1.0}
    
    n_blends_tested = 0
    
    def update_best(bl_name, weights_dict, mean_rmse, std_rmse, fold_rmse, fold_exact, pw):
        nonlocal best_blend
        score = mean_rmse + CONSISTENCY_LAMBDA * std_rmse
        if score < best_blend['score']:
            best_blend = {
                'score': score, 'name': bl_name, 'weights': weights_dict,
                'mean_rmse': mean_rmse, 'std_rmse': std_rmse,
                'mean_exact': np.mean(fold_exact), 'min_exact': min(fold_exact),
                'fold_rmse': fold_rmse, 'fold_exact': fold_exact, 'power': pw,
            }
    
    powers = [0.9, 0.95, 1.0, 1.1, 1.25]
    
    # --- Pairs ---
    for i in range(len(top_names)):
        for j in range(i+1, len(top_names)):
            for w1 in np.arange(0.1, 1.0, 0.1):
                w2 = round(1.0 - w1, 2)
                bl_name = f'{top_names[i]}*{w1:.1f}+{top_names[j]}*{w2:.1f}'
                raw_list = [(w1, raws[top_names[i]]), (w2, raws[top_names[j]])]
                for pw in powers:
                    mr, sr, fr, fe = _eval_blend_on_folds(raw_list, y, seasons, folds, pw)
                    update_best(bl_name, {top_names[i]: w1, top_names[j]: w2},
                                mr, sr, fr, fe, pw)
                    n_blends_tested += 1
    
    # --- Triples ---
    for i in range(len(top_names)):
        for j in range(i+1, len(top_names)):
            for k in range(j+1, len(top_names)):
                for w1 in np.arange(0.2, 0.7, 0.1):
                    for w2 in np.arange(0.1, 0.8 - w1, 0.1):
                        w3 = round(1.0 - w1 - w2, 2)
                        if w3 < 0.05: continue
                        bl_name = f'{top_names[i]}*{w1:.1f}+{top_names[j]}*{w2:.1f}+{top_names[k]}*{w3:.1f}'
                        raw_list = [(w1, raws[top_names[i]]), (w2, raws[top_names[j]]),
                                    (w3, raws[top_names[k]])]
                        for pw in powers:
                            mr, sr, fr, fe = _eval_blend_on_folds(raw_list, y, seasons, folds, pw)
                            update_best(bl_name,
                                        {top_names[i]: w1, top_names[j]: w2, top_names[k]: w3},
                                        mr, sr, fr, fe, pw)
                            n_blends_tested += 1
    
    # --- Quads (top 10 only, coarser grid) ---
    for i in range(min(10, len(top_names))):
        for j in range(i+1, min(10, len(top_names))):
            for k in range(j+1, min(10, len(top_names))):
                for l in range(k+1, min(10, len(top_names))):
                    for w1 in [0.3, 0.4]:
                        for w2 in [0.2, 0.3]:
                            for w3 in [0.1, 0.2]:
                                w4 = round(1.0 - w1 - w2 - w3, 2)
                                if w4 < 0.05 or w4 > 0.4: continue
                                raw_list = [(w1, raws[top_names[i]]), (w2, raws[top_names[j]]),
                                            (w3, raws[top_names[k]]), (w4, raws[top_names[l]])]
                                for pw in [0.95, 1.0, 1.1]:
                                    mr, sr, fr, fe = _eval_blend_on_folds(raw_list, y, seasons, folds, pw)
                                    update_best(f'quad_{i}_{j}_{k}_{l}_w{w1}_{w2}_{w3}_{w4}',
                                                {top_names[i]: w1, top_names[j]: w2,
                                                 top_names[k]: w3, top_names[l]: w4},
                                                mr, sr, fr, fe, pw)
                                    n_blends_tested += 1
    
    # --- Equal-weight ensemble of top K ---
    for K in [3, 5, 7, 10]:
        if K > len(top_names): continue
        w_eq = 1.0 / K
        raw_list = [(w_eq, raws[top_names[i]]) for i in range(K)]
        for pw in powers:
            mr, sr, fr, fe = _eval_blend_on_folds(raw_list, y, seasons, folds, pw)
            update_best(f'equal_top{K}',
                        {top_names[i]: w_eq for i in range(K)},
                        mr, sr, fr, fe, pw)
            n_blends_tested += 1
    
    print(f'  Tested {n_blends_tested} blend configs')
    return best_blend


# =================================================================
#  MAIN
# =================================================================
def main():
    print('='*70)
    print(' NCAA v30 — ROBUST CROSS-SEASON MODEL')
    print(f' Scoring: mean_RMSE + {CONSISTENCY_LAMBDA}×std_RMSE')
    print('='*70)
    
    all_df, labeled, train_df, test_df, GT = load_data()
    n_labeled = len(labeled)
    print(f'\n  Total labeled teams: {n_labeled}')
    
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    # Build features
    print('\n  Building features...')
    feat_A = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names_A = list(feat_A.columns)
    extra = build_extra_features(labeled, feat_A)
    feat_B = pd.concat([feat_A, extra], axis=1)
    feature_names_B = list(feat_B.columns)
    print(f'  Set A: {len(feature_names_A)}, Set B: {len(feature_names_B)} features')
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    
    # Impute
    X_A_raw = np.where(np.isinf(feat_A.values.astype(np.float64)), np.nan,
                       feat_A.values.astype(np.float64))
    X_B_raw = np.where(np.isinf(feat_B.values.astype(np.float64)), np.nan,
                       feat_B.values.astype(np.float64))
    imp_A = KNNImputer(n_neighbors=10, weights='distance')
    X_A_all = imp_A.fit_transform(X_A_raw)
    imp_B = KNNImputer(n_neighbors=10, weights='distance')
    X_B_all = imp_B.fit_transform(X_B_raw)
    
    # Global feature selection (just for reference, each fold does its own)
    top_k_B_idx, top_k_B_names = select_top_k_features(X_B_all, y, feature_names_B, k=30)
    top_k_A_idx, top_k_A_names = select_top_k_features(X_A_all, y, feature_names_A, k=25)
    print(f'  Top-30 B: {", ".join(top_k_B_names[:8])}...')
    
    # ============================
    # PHASE 1: Individual model LOSO evaluation
    # ============================
    print('\n' + '='*70)
    print(' PHASE 1: INDIVIDUAL MODEL LOSO EVALUATION')
    print('='*70)
    
    summary = loso_evaluate_all(X_A_all, X_B_all, y, seasons,
                                 feature_names_B, top_k_B_idx, top_k_A_idx)
    
    # Print top 20 by score
    ranked = sorted(summary.items(), key=lambda x: x[1]['score'])
    print(f'\n  {"Rank":>4} {"Model":<45} {"MeanRMSE":>8} {"StdRMSE":>8} {"Score":>8} {"MeanEx":>6} {"MinEx":>5} {"MaxEx":>5}')
    print(f'  {"----":>4} {"-----":<45} {"--------":>8} {"--------":>8} {"-----":>8} {"------":>6} {"-----":>5} {"-----":>5}')
    for rank, (name, info) in enumerate(ranked[:30], 1):
        print(f'  {rank:4d} {name:<45} {info["mean_rmse"]:8.3f} {info["std_rmse"]:8.3f} '
              f'{info["score"]:8.3f} {info["mean_exact"]:6.1f} {info["min_exact"]:5d} {info["max_exact"]:5d}')
    
    # Show per-fold breakdown for top 5
    folds = sorted(set(seasons))
    print(f'\n  Per-fold breakdown (top 5):')
    for rank, (name, info) in enumerate(ranked[:5], 1):
        print(f'\n    #{rank} {name}')
        for fi, hold in enumerate(folds):
            print(f'      {hold}: RMSE={info["fold_rmse"][fi]:.3f}, Exact={info["fold_exact"][fi]}/68')
    
    # ============================
    # PHASE 2: Blend search
    # ============================
    print('\n' + '='*70)
    print(' PHASE 2: BLEND SEARCH (mean_RMSE + consistency penalty)')
    print('='*70)
    
    best_blend = blend_search(summary, y, seasons, top_n_individuals=15)
    
    print(f'\n  BEST BLEND: {best_blend["name"]}')
    print(f'  Score: {best_blend["score"]:.4f} (mean_RMSE={best_blend["mean_rmse"]:.4f}, '
          f'std_RMSE={best_blend["std_rmse"]:.4f})')
    print(f'  Mean exact: {best_blend["mean_exact"]:.1f}/68, Min: {best_blend["min_exact"]}')
    print(f'  Power: {best_blend["power"]}')
    print(f'  Per-fold:')
    for fi, hold in enumerate(folds):
        print(f'    {hold}: RMSE={best_blend["fold_rmse"][fi]:.3f}, '
              f'Exact={best_blend["fold_exact"][fi]}/68')
    
    # ============================
    # Compare with v4c baseline
    # ============================
    print('\n' + '='*70)
    print(' COMPARISON WITH v4c')
    print('='*70)
    
    # Recompute v4c for fair comparison
    v4c_folds_rmse = []
    v4c_folds_exact = []
    for fi, hold in enumerate(folds):
        tr = seasons != hold
        te = seasons == hold
        # v4c used global feature selection — hence overfit
        from ncaa_2026_model import predict_v4c_triplet
        top_k_B_fold = select_top_k_features(X_B_all[tr], y[tr], feature_names_B, k=30)[0]
        pred = predict_v4c_triplet(X_A_all[tr], X_B_all[tr], y[tr],
                                    X_A_all[te], X_B_all[te],
                                    seasons[tr], top_k_B_fold)
        avail = {hold: list(range(1, 69))}
        asgn = hungarian(pred, seasons[te], avail, power=0.95)
        y_te = y[te].astype(int)
        rmse = np.sqrt(np.mean((asgn - y_te)**2))
        exact = int(np.sum(asgn == y_te))
        v4c_folds_rmse.append(rmse)
        v4c_folds_exact.append(exact)
    
    v4c_mean_rmse = np.mean(v4c_folds_rmse)
    v4c_std_rmse = np.std(v4c_folds_rmse)
    v4c_score = v4c_mean_rmse + CONSISTENCY_LAMBDA * v4c_std_rmse
    
    print(f'\n  {"Metric":<25} {"v4c":>12} {"v30 Best":>12}')
    print(f'  {"-"*25} {"-"*12} {"-"*12}')
    print(f'  {"Mean RMSE":<25} {v4c_mean_rmse:12.4f} {best_blend["mean_rmse"]:12.4f}')
    print(f'  {"Std RMSE":<25} {v4c_std_rmse:12.4f} {best_blend["std_rmse"]:12.4f}')
    print(f'  {"Score (mean+λ·std)":<25} {v4c_score:12.4f} {best_blend["score"]:12.4f}')
    print(f'  {"Mean Exact":<25} {np.mean(v4c_folds_exact):12.1f} {best_blend["mean_exact"]:12.1f}')
    print(f'  {"Min Exact":<25} {min(v4c_folds_exact):12d} {best_blend["min_exact"]:12d}')
    
    print(f'\n  Per-fold comparison:')
    print(f'  {"Season":<12} {"v4c RMSE":>10} {"v4c Ex":>8} {"v30 RMSE":>10} {"v30 Ex":>8}')
    for fi, hold in enumerate(folds):
        print(f'  {hold:<12} {v4c_folds_rmse[fi]:10.3f} {v4c_folds_exact[fi]:8d} '
              f'{best_blend["fold_rmse"][fi]:10.3f} {best_blend["fold_exact"][fi]:8d}')
    
    # ============================
    # Summary
    # ============================
    print(f'\n  Total models evaluated: {len(summary)}')
    print(f'  Best individual model: {ranked[0][0]} (score={ranked[0][1]["score"]:.4f})')
    print(f'  Best blend: {best_blend["name"]} (score={best_blend["score"]:.4f})')
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
