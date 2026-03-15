#!/usr/bin/env python3
"""
NCAA v5 Improved Model — Systematic improvements over v4
=========================================================

Improvements:
  1. Extended features: +AvgOppNET, +NETNonConfSOS, +5 derived = 75 total
  2. Pairwise XGBClassifier (non-linear pairwise comparisons)
  3. Direct regression (XGB+Ridge) for absolute seed magnitude
  4. Structured blend search across all 5 scoring methods
  5. Hungarian power search (0.75, 1.0, 1.25, 1.5)

v4 baseline: LOSO RMSE=3.316, std=0.232, 14.0/68 exact per fold
"""

import os, sys, re, time, warnings
import numpy as np
import pandas as pd
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

# Import shared functions from production model
from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
    USE_TOP_K_A, HUNGARIAN_POWER,
    PW_C1, PW_C2, PW_C3, BLEND_W1, BLEND_W2, BLEND_W3,
    V40_XGB_PARAMS, SEEDS
)


# =================================================================
#  EXTENDED FEATURE ENGINEERING (68 → 75 features)
# =================================================================
def build_features_v5(df, context_df, labeled_df, tourn_rids):
    """
    Extended feature set: v4's 68 features + 7 new = 75 features.
    
    New features:
      - AvgOppNET: raw opponent average NET score (not rank)
      - NETNonConfSOS: non-conference strength of schedule
      - nonconf_sos_delta: NETNonConfSOS - NETSOS
      - net_improvement_pct: (PrevNET - NET) / (PrevNET + 1)
      - opp_quality_ratio: AvgOppNETRank / (NET + 1)
      - q1q2_win_pct: combined Q1+Q2 win percentage
      - road_win_freq: road wins per total game
    """
    # Start with v4's 68 features
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    # Parse raw columns
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rank = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    
    # NEW FEATURE 1: AvgOppNET (raw opponent NET score, different from rank)
    feat['AvgOppNET'] = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    
    # NEW FEATURE 2: NETNonConfSOS (non-conference SOS)
    ncsos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    feat['NETNonConfSOS'] = ncsos
    
    # NEW FEATURE 3: Non-conf SOS delta (how much harder/easier non-conf vs overall)
    feat['nonconf_sos_delta'] = ncsos - sos
    
    # NEW FEATURE 4: NET improvement as percentage (captures momentum magnitude)
    feat['net_improvement_pct'] = (prev - net) / (prev + 1)
    
    # NEW FEATURE 5: Schedule quality relative to team quality
    feat['opp_quality_ratio'] = opp_rank / (net + 1)
    
    # NEW FEATURE 6: Combined Q1+Q2 win percentage (committee focus)
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    feat['q1q2_win_pct'] = (q1w + q2w) / (q1w + q1l + q2w + q2l + 0.5)
    
    # NEW FEATURE 7: Road win frequency (wins on the road per total game)
    road_wl = df['RoadWL'].apply(parse_wl)
    road_w = road_wl.apply(lambda x: x[0]).fillna(0)
    total_games = feat.get('total_games', pd.Series(30, index=df.index)).fillna(30)
    feat['road_win_freq'] = road_w / (total_games + 0.5)
    
    return feat


# =================================================================
#  PW-XGB CLASSIFIER (non-linear pairwise model)
# =================================================================
def pairwise_xgb_score(X_train, y_train, X_test, seasons_train):
    """
    Pairwise XGBClassifier: captures non-linear feature interactions
    in team comparison diffs.
    """
    pw_X, pw_y = build_pairwise_data(X_train, y_train, seasons_train)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0,
        min_child_weight=5,  # conservative for ~18K pairs
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_sc, pw_y)
    return pairwise_score(xgb_clf, X_test, sc)


# =================================================================
#  DIRECT REGRESSION (XGB + Ridge)
# =================================================================
def direct_regression_score(X_train, y_train, X_test):
    """
    Direct regression: predict seed values, then convert to ranks.
    Captures absolute magnitude info that pairwise approach misses.
    """
    # Multi-seed XGB for stability
    xgb_preds = []
    for seed in [42, 123, 777]:
        m = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_lambda=5.0, reg_alpha=2.0,
            min_child_weight=5,
            random_state=seed, verbosity=0
        )
        m.fit(X_train, y_train)
        xgb_preds.append(m.predict(X_test))
    xgb_avg = np.mean(xgb_preds, axis=0)
    
    # Ridge with heavier regularization
    sc = StandardScaler()
    ridge = Ridge(alpha=10.0)
    ridge.fit(sc.fit_transform(X_train), y_train)
    ridge_pred = ridge.predict(sc.transform(X_test))
    
    # Blend XGB + Ridge
    direct = 0.7 * xgb_avg + 0.3 * ridge_pred
    
    # Convert to ranks (1 = best/lowest predicted seed)
    return np.argsort(np.argsort(direct)).astype(float) + 1.0


# =================================================================
#  LOSO VALIDATION WITH BLEND SEARCH
# =================================================================
def main():
    print('='*65)
    print(' NCAA v5 IMPROVED MODEL — LOSO VALIDATION')
    print(' (Extended features + XGB pairwise + Direct regression)')
    print('='*65)
    
    # Load data
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    
    # Context for conference stats
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    # Build EXTENDED features (75 features)
    print(f'\n  Building v5 features (extended)...')
    feat = build_features_v5(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat.columns)
    print(f'  Features: {len(feature_names)} (v4 had 68)')
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    folds = sorted(set(seasons))
    
    # Impute
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    # ─── Precompute scores from all 5 methods for all LOSO folds ───
    method_names = [
        'PW-LR(C=5)',        # 0: existing v4 component
        'PW-LR(C=0.01)',     # 1: existing v4 component
        'PW-LR(topK,C=1)',   # 2: existing v4 component
        'PW-XGB',            # 3: NEW — non-linear pairwise
        'Direct-XGB+Ridge',  # 4: NEW — direct regression
    ]
    n_methods = len(method_names)
    all_scores = np.zeros((n_methods, n_labeled))
    
    print(f'\n  Computing {n_methods} scoring methods × {len(folds)} LOSO folds...')
    
    for fi, hold_season in enumerate(folds):
        tr = seasons != hold_season
        te = seasons == hold_season
        te_idx = np.where(te)[0]
        n_te = te.sum()
        
        print(f'    Fold {fi+1}/{len(folds)}: hold={hold_season}, '
              f'train={tr.sum()}, test={n_te}', end='', flush=True)
        
        # Feature selection (per-fold, no leakage)
        top_k_idx = select_top_k_features(
            X_all[tr], y[tr], feature_names, k=USE_TOP_K_A)[0]
        
        # ── Method 0: PW-LogReg full C=5.0 ──
        pw_X, pw_y = build_pairwise_data(X_all[tr], y[tr], seasons[tr])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        all_scores[0, te] = pairwise_score(lr1, X_all[te], sc)
        
        # ── Method 1: PW-LogReg full C=0.01 ──
        lr2 = LogisticRegression(C=PW_C2, penalty='l2', max_iter=2000, random_state=42)
        lr2.fit(pw_X_sc, pw_y)
        all_scores[1, te] = pairwise_score(lr2, X_all[te], sc)
        
        # ── Method 2: PW-LogReg topK C=1.0 ──
        X_k_tr = X_all[tr][:, top_k_idx]
        X_k_te = X_all[te][:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(X_k_tr, y[tr], seasons[tr])
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        all_scores[2, te] = pairwise_score(lr3, X_k_te, sc_k)
        
        # ── Method 3: PW-XGB (NEW — non-linear) ──
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0,
            min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_clf.fit(pw_X_sc, pw_y)
        all_scores[3, te] = pairwise_score(xgb_clf, X_all[te], sc)
        
        # ── Method 4: Direct XGB+Ridge (NEW) ──
        xgb_preds = []
        for seed in [42, 123, 777]:
            m = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=5.0, reg_alpha=2.0,
                min_child_weight=5,
                random_state=seed, verbosity=0
            )
            m.fit(X_all[tr], y[tr])
            xgb_preds.append(m.predict(X_all[te]))
        xgb_avg = np.mean(xgb_preds, axis=0)
        
        sc_r = StandardScaler()
        ridge = Ridge(alpha=10.0)
        ridge.fit(sc_r.fit_transform(X_all[tr]), y[tr])
        ridge_pred = ridge.predict(sc_r.transform(X_all[te]))
        
        direct = 0.7 * xgb_avg + 0.3 * ridge_pred
        all_scores[4, te] = np.argsort(np.argsort(direct)).astype(float) + 1.0
        
        print(' ✓')
    
    # ─── Evaluate each method solo ───
    print('\n' + '='*65)
    print(' INDIVIDUAL METHOD RESULTS')
    print('='*65)
    print(f'  {"Method":<22} {"Exact":>5} {"RMSE":>8} {"ρ":>7}')
    print(f'  {"─"*22} {"─"*5} {"─"*8} {"─"*7}')
    
    for mi in range(n_methods):
        assigned = np.zeros(n_labeled, dtype=int)
        for s in folds:
            mask = seasons == s
            avail = {s: list(range(1, 69))}
            assigned[mask] = hungarian(all_scores[mi, mask], seasons[mask], avail)
        exact = int((assigned == y.astype(int)).sum())
        rmse = np.sqrt(np.mean((assigned - y.astype(int))**2))
        rho, _ = spearmanr(assigned, y.astype(int))
        print(f'  {method_names[mi]:<22} {exact:5d} {rmse:8.4f} {rho:7.4f}')
    
    # ─── Blend search ───
    print('\n' + '='*65)
    print(' BLEND SEARCH (scored by mean_RMSE + 0.5 × std_RMSE)')
    print('='*65)
    
    # Predefined blend configurations
    configs = [
        # Name,                          [pw_lr5, pw_lr001, pw_lrk, pw_xgb, direct]
        ('v4 baseline (68feat)',          [0.60, 0.10, 0.30, 0.00, 0.00]),
        ('PW-XGB only',                  [0.00, 0.00, 0.00, 1.00, 0.00]),
        ('Direct only',                  [0.00, 0.00, 0.00, 0.00, 1.00]),
        
        # Add PW-XGB to v4
        ('v4 + 10% XGB',                 [0.54, 0.09, 0.27, 0.10, 0.00]),
        ('v4 + 20% XGB',                 [0.48, 0.08, 0.24, 0.20, 0.00]),
        ('v4 + 30% XGB',                 [0.42, 0.07, 0.21, 0.30, 0.00]),
        ('v4 + 40% XGB',                 [0.36, 0.06, 0.18, 0.40, 0.00]),
        
        # Add Direct to v4
        ('v4 + 10% Direct',              [0.54, 0.09, 0.27, 0.00, 0.10]),
        ('v4 + 20% Direct',              [0.48, 0.08, 0.24, 0.00, 0.20]),
        ('v4 + 30% Direct',              [0.42, 0.07, 0.21, 0.00, 0.30]),
        ('v4 + 40% Direct',              [0.36, 0.06, 0.18, 0.00, 0.40]),
        
        # Add both
        ('v4 + 10%X + 10%D',             [0.48, 0.08, 0.24, 0.10, 0.10]),
        ('v4 + 15%X + 15%D',             [0.42, 0.07, 0.21, 0.15, 0.15]),
        ('v4 + 20%X + 20%D',             [0.36, 0.06, 0.18, 0.20, 0.20]),
        
        # Heavy on new methods
        ('40%LR5 + 30%XGB + 30%D',       [0.40, 0.00, 0.00, 0.30, 0.30]),
        ('30%LR5 + 40%XGB + 30%D',       [0.30, 0.00, 0.00, 0.40, 0.30]),
        ('30%LR5 + 30%XGB + 40%D',       [0.30, 0.00, 0.00, 0.30, 0.40]),
        ('50%XGB + 50%D',                [0.00, 0.00, 0.00, 0.50, 0.50]),
        
        # Equal blend
        ('Equal all 5',                   [0.20, 0.20, 0.20, 0.20, 0.20]),
        
        # Pairs of existing methods with new
        ('70%LR5 + 30%XGB',              [0.70, 0.00, 0.00, 0.30, 0.00]),
        ('60%LR5 + 40%XGB',              [0.60, 0.00, 0.00, 0.40, 0.00]),
        ('50%LR5 + 50%XGB',              [0.50, 0.00, 0.00, 0.50, 0.00]),
        ('70%LR5 + 30%D',                [0.70, 0.00, 0.00, 0.00, 0.30]),
        ('60%LR5 + 40%D',                [0.60, 0.00, 0.00, 0.00, 0.40]),
        ('50%LR5 + 50%D',                [0.50, 0.00, 0.00, 0.00, 0.50]),
        
        # PW-LR + PW-XGB heavy
        ('40%LR5 + 20%LR001 + 40%XGB',   [0.40, 0.20, 0.00, 0.40, 0.00]),
        ('30%LR5 + 20%LRk + 50%XGB',     [0.30, 0.00, 0.20, 0.50, 0.00]),
        
        # Direct-heavy with LR stabilizer
        ('30%LR5 + 70%D',                [0.30, 0.00, 0.00, 0.00, 0.70]),
        ('20%LR5 + 20%XGB + 60%D',       [0.20, 0.00, 0.00, 0.20, 0.60]),
    ]
    
    # Hungarian power values to test
    powers = [0.75, 1.0, 1.25, 1.5]
    
    best_overall_score = float('inf')
    best_overall_config = None
    best_overall_power = 1.0
    best_overall_assigned = None
    
    results = []
    
    for cfg_name, weights in configs:
        for power in powers:
            # Blend scores
            blended = np.zeros(n_labeled)
            for mi in range(n_methods):
                blended += weights[mi] * all_scores[mi]
            
            # Hungarian per season
            assigned = np.zeros(n_labeled, dtype=int)
            fold_rmses = []
            for s in folds:
                mask = seasons == s
                avail = {s: list(range(1, 69))}
                assigned[mask] = hungarian(blended[mask], seasons[mask], avail, power=power)
                ys = y[mask].astype(int)
                fold_rmses.append(np.sqrt(np.mean((assigned[mask] - ys)**2)))
            
            exact = int((assigned == y.astype(int)).sum())
            rmse = np.sqrt(np.mean((assigned - y.astype(int))**2))
            rho, _ = spearmanr(assigned, y.astype(int))
            mean_rmse = np.mean(fold_rmses)
            std_rmse = np.std(fold_rmses)
            score = mean_rmse + 0.5 * std_rmse
            
            results.append({
                'name': cfg_name, 'power': power, 'weights': weights,
                'exact': exact, 'rmse': rmse, 'rho': rho,
                'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 'score': score,
                'assigned': assigned.copy()
            })
            
            if score < best_overall_score:
                best_overall_score = score
                best_overall_config = cfg_name
                best_overall_power = power
                best_overall_assigned = assigned.copy()
    
    # Sort by score and show top 20
    results.sort(key=lambda r: r['score'])
    
    print(f'\n  {"Rank":>4} {"Config":<32} {"Pwr":>4} {"Exact":>5} {"RMSE":>7} '
          f'{"μ±σ":>12} {"Score":>7}')
    print(f'  {"─"*4} {"─"*32} {"─"*4} {"─"*5} {"─"*7} {"─"*12} {"─"*7}')
    
    for i, r in enumerate(results[:25]):
        marker = ' ★' if i == 0 else ''
        print(f'  {i+1:4d} {r["name"]:<32} {r["power"]:4.2f} {r["exact"]:5d} '
              f'{r["rmse"]:7.4f} {r["mean_rmse"]:.3f}±{r["std_rmse"]:.3f} '
              f'{r["score"]:7.4f}{marker}')
    
    # ─── Best result details ───
    print('\n' + '='*65)
    print(f' BEST CONFIGURATION')
    print('='*65)
    best = results[0]
    print(f'  Config: {best["name"]}')
    print(f'  Power:  {best["power"]}')
    print(f'  Weights: {dict(zip(method_names, best["weights"]))}')
    print(f'  TOTAL: {best["exact"]}/340 exact ({best["exact"]/340*100:.1f}%), '
          f'RMSE={best["rmse"]:.4f}, ρ={best["rho"]:.4f}')
    print(f'  Selection score: {best["score"]:.4f} '
          f'(μ={best["mean_rmse"]:.4f}, σ={best["std_rmse"]:.4f})')
    
    # Per-season breakdown
    print(f'\n  {"Season":<12} {"N":>3} {"Exact":>5} {"Pct":>6} {"RMSE":>8}')
    print(f'  {"─"*12} {"─"*3} {"─"*5} {"─"*6} {"─"*8}')
    for s in folds:
        mask = seasons == s
        ys = y[mask].astype(int)
        ps = best['assigned'][mask]
        ex = int((ps == ys).sum())
        rm = np.sqrt(np.mean((ps - ys)**2))
        print(f'  {s:<12} {mask.sum():3d} {ex:5d} {ex/mask.sum()*100:5.1f}% {rm:8.3f}')
    
    # ─── Compare against v4 baseline ───
    print('\n' + '='*65)
    print(' v4 vs v5 COMPARISON')
    print('='*65)
    
    # Find v4 baseline result (power=1.0)
    v4_result = None
    for r in results:
        if 'v4 baseline' in r['name'] and r['power'] == 1.0:
            v4_result = r
            break
    
    if v4_result:
        print(f'  v4: {v4_result["exact"]}/340 exact, RMSE={v4_result["rmse"]:.4f}, '
              f'score={v4_result["score"]:.4f}')
        print(f'  v5: {best["exact"]}/340 exact, RMSE={best["rmse"]:.4f}, '
              f'score={best["score"]:.4f}')
        delta_exact = best['exact'] - v4_result['exact']
        delta_rmse = best['rmse'] - v4_result['rmse']
        delta_score = best['score'] - v4_result['score']
        print(f'  Δ exact: {delta_exact:+d}, Δ RMSE: {delta_rmse:+.4f}, '
              f'Δ score: {delta_score:+.4f}')
        if delta_score < 0:
            print(f'  → v5 IMPROVES by {-delta_score:.4f} on selection score')
        else:
            print(f'  → v4 still better by {delta_score:.4f}')
    
    # ─── Feature analysis: which new features are most important? ───
    print('\n' + '='*65)
    print(' FEATURE IMPORTANCE (new features)')
    print('='*65)
    
    new_feat_names = ['AvgOppNET', 'NETNonConfSOS', 'nonconf_sos_delta',
                      'net_improvement_pct', 'opp_quality_ratio', 
                      'q1q2_win_pct', 'road_win_freq']
    
    # Quick importance check via Ridge on all data
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_all)
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_sc, y)
    
    all_importances = list(zip(feature_names, np.abs(ridge.coef_)))
    all_importances.sort(key=lambda x: x[1], reverse=True)
    
    print(f'\n  New features rank among all {len(feature_names)}:')
    for fn in new_feat_names:
        for rank, (name, imp) in enumerate(all_importances, 1):
            if name == fn:
                print(f'    {fn:<25} rank {rank:>3}/{len(feature_names)} '
                      f'(|coef|={imp:.4f})')
                break
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    
    # ─── Save best config for production use ───
    print('\n' + '='*65)
    print(' SUMMARY — Copy to production model if improved:')
    print('='*65)
    print(f'  BLEND_W1 = {best["weights"][0]:.2f}  # PW-LR(C=5)')
    print(f'  BLEND_W2 = {best["weights"][1]:.2f}  # PW-LR(C=0.01)')
    print(f'  BLEND_W3 = {best["weights"][2]:.2f}  # PW-LR(topK,C=1)')
    print(f'  BLEND_W4 = {best["weights"][3]:.2f}  # PW-XGB (NEW)')
    print(f'  BLEND_W5 = {best["weights"][4]:.2f}  # Direct-XGB+Ridge (NEW)')
    print(f'  HUNGARIAN_POWER = {best["power"]}')


if __name__ == '__main__':
    main()
