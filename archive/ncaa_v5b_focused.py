#!/usr/bin/env python3
"""
NCAA v5b — Focused improvements with original 68 features
==========================================================

Tests the most promising findings from v5 on clean 68-feature set:
  1. Hungarian power search (0.75, 1.0, 1.25, 1.5)  
  2. PW-XGB classifier on 68 features
  3. Direct regression on 68 features
  4. Selective new features (only the most useful)
  5. Blend search across all scoring methods

Baseline to beat: v4 (68 feat, power=1.0) = 70/340 exact, RMSE≈3.316
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

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
    USE_TOP_K_A, PW_C1, PW_C2, PW_C3
)


def build_features_selective(df, context_df, labeled_df, tourn_rids):
    """Add only the most useful new features (AvgOppNET + nonconf_sos_delta)."""
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    # AvgOppNET: rank 12/75 in v5 analysis — strong signal
    feat['AvgOppNET'] = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    
    # nonconf_sos_delta: rank 20/75 — decent signal
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    ncsos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    feat['nonconf_sos_delta'] = ncsos - sos
    
    return feat


def evaluate_config(all_scores, weights, y, seasons, folds, power=1.0):
    """Evaluate a single blend config across LOSO folds."""
    n = len(y)
    n_methods = all_scores.shape[0]
    
    # Blend scores
    blended = np.zeros(n)
    for mi in range(n_methods):
        blended += weights[mi] * all_scores[mi]
    
    # Hungarian per season
    assigned = np.zeros(n, dtype=int)
    fold_rmses = []
    fold_exacts = []
    for s in folds:
        mask = seasons == s
        avail = {s: list(range(1, 69))}
        assigned[mask] = hungarian(blended[mask], seasons[mask], avail, power=power)
        ys = y[mask].astype(int)
        fold_rmses.append(np.sqrt(np.mean((assigned[mask] - ys)**2)))
        fold_exacts.append(int((assigned[mask] == ys).sum()))
    
    total_exact = int((assigned == y.astype(int)).sum())
    total_rmse = np.sqrt(np.mean((assigned - y.astype(int))**2))
    rho, _ = spearmanr(assigned, y.astype(int))
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    score = mean_rmse + 0.5 * std_rmse
    
    return {
        'exact': total_exact, 'rmse': total_rmse, 'rho': rho,
        'mean_rmse': mean_rmse, 'std_rmse': std_rmse, 'score': score,
        'fold_exacts': fold_exacts, 'fold_rmses': fold_rmses,
        'assigned': assigned.copy()
    }


def compute_all_scores(X_all, y, seasons, folds, feature_names, label=""):
    """Compute all 5 scoring methods for all LOSO folds."""
    n = len(y)
    method_names = [
        'PW-LR(C=5)',
        'PW-LR(C=0.01)',
        'PW-LR(topK,C=1)',
        'PW-XGB',
        'Direct-XGB+Ridge',
    ]
    n_methods = len(method_names)
    all_scores = np.zeros((n_methods, n))
    
    for fi, hold in enumerate(folds):
        tr = seasons != hold
        te = seasons == hold
        print(f'    [{label}] Fold {fi+1}/{len(folds)} ({hold})', end='', flush=True)
        
        top_k_idx = select_top_k_features(
            X_all[tr], y[tr], feature_names, k=USE_TOP_K_A)[0]
        
        # Method 0: PW-LR full C=5.0
        pw_X, pw_y = build_pairwise_data(X_all[tr], y[tr], seasons[tr])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        all_scores[0, te] = pairwise_score(lr1, X_all[te], sc)
        
        # Method 1: PW-LR full C=0.01
        lr2 = LogisticRegression(C=PW_C2, penalty='l2', max_iter=2000, random_state=42)
        lr2.fit(pw_X_sc, pw_y)
        all_scores[1, te] = pairwise_score(lr2, X_all[te], sc)
        
        # Method 2: PW-LR topK C=1.0
        X_k_tr = X_all[tr][:, top_k_idx]
        X_k_te = X_all[te][:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(X_k_tr, y[tr], seasons[tr])
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        all_scores[2, te] = pairwise_score(lr3, X_k_te, sc_k)
        
        # Method 3: PW-XGB
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
        
        # Method 4: Direct XGB+Ridge
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
    
    return all_scores, method_names


def run_blend_search(all_scores, method_names, y, seasons, folds, label=""):
    """Search over blend configs and powers."""
    configs = [
        # v4 variants
        ('v4 original',              [0.60, 0.10, 0.30, 0.00, 0.00]),
        ('PW-LR(C=5) only',         [1.00, 0.00, 0.00, 0.00, 0.00]),
        ('PW-XGB only',             [0.00, 0.00, 0.00, 1.00, 0.00]),
        ('Direct only',             [0.00, 0.00, 0.00, 0.00, 1.00]),
        
        # Add XGB to v4
        ('v4 + 10% XGB',            [0.54, 0.09, 0.27, 0.10, 0.00]),
        ('v4 + 20% XGB',            [0.48, 0.08, 0.24, 0.20, 0.00]),
        ('v4 + 30% XGB',            [0.42, 0.07, 0.21, 0.30, 0.00]),
        
        # Add Direct to v4
        ('v4 + 10% Direct',         [0.54, 0.09, 0.27, 0.00, 0.10]),
        ('v4 + 20% Direct',         [0.48, 0.08, 0.24, 0.00, 0.20]),
        ('v4 + 30% Direct',         [0.42, 0.07, 0.21, 0.00, 0.30]),
        
        # Add both
        ('v4 + 10%X + 10%D',        [0.48, 0.08, 0.24, 0.10, 0.10]),
        ('v4 + 15%X + 15%D',        [0.42, 0.07, 0.21, 0.15, 0.15]),
        ('v4 + 20%X + 10%D',        [0.42, 0.07, 0.21, 0.20, 0.10]),
        ('v4 + 10%X + 20%D',        [0.42, 0.07, 0.21, 0.10, 0.20]),
        
        # Heavy on new methods
        ('40%LR5 + 30%XGB + 30%D',  [0.40, 0.00, 0.00, 0.30, 0.30]),
        ('50%LR5 + 50%XGB',         [0.50, 0.00, 0.00, 0.50, 0.00]),
        ('50%LR5 + 50%D',           [0.50, 0.00, 0.00, 0.00, 0.50]),
        ('60%LR5 + 40%XGB',         [0.60, 0.00, 0.00, 0.40, 0.00]),
        ('70%LR5 + 30%XGB',         [0.70, 0.00, 0.00, 0.30, 0.00]),
        ('80%LR5 + 20%XGB',         [0.80, 0.00, 0.00, 0.20, 0.00]),
        
        # LR+XGB pairs with regularized LR
        ('50%LR5+10%LR001+40%XGB',  [0.50, 0.10, 0.00, 0.40, 0.00]),
        ('40%LR5+20%LR001+40%XGB',  [0.40, 0.20, 0.00, 0.40, 0.00]),
        ('50%LR5+20%LRk+30%XGB',    [0.50, 0.00, 0.20, 0.30, 0.00]),
        
        # Equal blends
        ('Equal all 5',              [0.20, 0.20, 0.20, 0.20, 0.20]),
        ('Equal LR+XGB',            [0.25, 0.25, 0.25, 0.25, 0.00]),
    ]
    
    powers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    results = []
    for cfg_name, weights in configs:
        for power in powers:
            r = evaluate_config(all_scores, weights, y, seasons, folds, power=power)
            r['name'] = cfg_name
            r['power'] = power
            r['weights'] = weights
            results.append(r)
    
    results.sort(key=lambda r: r['score'])
    return results


def main():
    print('='*70)
    print(' NCAA v5b — FOCUSED IMPROVEMENT TESTS')
    print(' (Original 68 features + selective additions + new methods)')
    print('='*70)
    
    # Load data
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    folds = sorted(set(seasons))
    
    # ═══════════════════════════════════════════════════
    # PHASE 1: Original 68 features
    # ═══════════════════════════════════════════════════
    print(f'\n  ── Phase 1: Original 68 features ──')
    feat_68 = build_features(labeled, context_df, labeled, tourn_rids)
    fn_68 = list(feat_68.columns)
    print(f'  Features: {len(fn_68)}')
    
    X_68_raw = np.where(np.isinf(feat_68.values.astype(np.float64)), np.nan,
                        feat_68.values.astype(np.float64))
    imp_68 = KNNImputer(n_neighbors=10, weights='distance')
    X_68 = imp_68.fit_transform(X_68_raw)
    
    scores_68, mnames = compute_all_scores(X_68, y, seasons, folds, fn_68, label="68feat")
    results_68 = run_blend_search(scores_68, mnames, y, seasons, folds, label="68feat")
    
    # ═══════════════════════════════════════════════════
    # PHASE 2: 68 + 2 selective new features (70 features)
    # ═══════════════════════════════════════════════════
    print(f'\n  ── Phase 2: 68 + 2 selective features (70 total) ──')
    feat_70 = build_features_selective(labeled, context_df, labeled, tourn_rids)
    fn_70 = list(feat_70.columns)
    print(f'  Features: {len(fn_70)}')
    
    X_70_raw = np.where(np.isinf(feat_70.values.astype(np.float64)), np.nan,
                        feat_70.values.astype(np.float64))
    imp_70 = KNNImputer(n_neighbors=10, weights='distance')
    X_70 = imp_70.fit_transform(X_70_raw)
    
    scores_70, _ = compute_all_scores(X_70, y, seasons, folds, fn_70, label="70feat")
    results_70 = run_blend_search(scores_70, mnames, y, seasons, folds, label="70feat")
    
    # ═══════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════
    for tag, results in [("68 FEATURES (original)", results_68),
                          ("70 FEATURES (selective new)", results_70)]:
        print('\n' + '='*70)
        print(f' TOP 20 — {tag}')
        print('='*70)
        print(f'  {"Rk":>3} {"Config":<32} {"Pwr":>4} {"Exact":>5} {"RMSE":>7} '
              f'{"μ±σ":>12} {"Score":>7}')
        print(f'  {"─"*3} {"─"*32} {"─"*4} {"─"*5} {"─"*7} {"─"*12} {"─"*7}')
        
        for i, r in enumerate(results[:20]):
            m = ' ★' if i == 0 else ''
            print(f'  {i+1:3d} {r["name"]:<32} {r["power"]:4.2f} {r["exact"]:5d} '
                  f'{r["rmse"]:7.4f} {r["mean_rmse"]:.3f}±{r["std_rmse"]:.3f} '
                  f'{r["score"]:7.4f}{m}')
    
    # ─── Overall best ───
    all_results = [(r, "68feat") for r in results_68] + [(r, "70feat") for r in results_70]
    all_results.sort(key=lambda x: x[0]['score'])
    best, best_tag = all_results[0]
    
    print('\n' + '='*70)
    print(f' OVERALL BEST CONFIGURATION')
    print('='*70)
    print(f'  Feature set: {best_tag}')
    print(f'  Config: {best["name"]}')
    print(f'  Power:  {best["power"]}')
    print(f'  Weights: {dict(zip(mnames, best["weights"]))}')
    print(f'  TOTAL: {best["exact"]}/340 exact ({best["exact"]/340*100:.1f}%), '
          f'RMSE={best["rmse"]:.4f}, ρ={best["rho"]:.4f}')
    print(f'  Score: {best["score"]:.4f} (μ={best["mean_rmse"]:.4f}, σ={best["std_rmse"]:.4f})')
    
    # Per-season breakdown for best
    print(f'\n  {"Season":<12} {"N":>3} {"Exact":>5} {"Pct":>6} {"RMSE":>8}')
    print(f'  {"─"*12} {"─"*3} {"─"*5} {"─"*6} {"─"*8}')
    for fi, s in enumerate(folds):
        mask = seasons == s
        n_s = int(mask.sum())
        print(f'  {s:<12} {n_s:3d} {best["fold_exacts"][fi]:5d} '
              f'{best["fold_exacts"][fi]/n_s*100:5.1f}% {best["fold_rmses"][fi]:8.3f}')
    
    # ─── Compare best vs v4 original at power=1.0 ───
    v4_ref = None
    for r in results_68:
        if 'v4 original' in r['name'] and r['power'] == 1.0:
            v4_ref = r
            break
    
    if v4_ref:
        print(f'\n  v4 (68feat, p=1.0): {v4_ref["exact"]}/340, RMSE={v4_ref["rmse"]:.4f}, '
              f'score={v4_ref["score"]:.4f}')
        print(f'  BEST:               {best["exact"]}/340, RMSE={best["rmse"]:.4f}, '
              f'score={best["score"]:.4f}')
        d_ex = best['exact'] - v4_ref['exact']
        d_rm = best['rmse'] - v4_ref['rmse']
        d_sc = best['score'] - v4_ref['score']
        print(f'  Δ exact: {d_ex:+d}, Δ RMSE: {d_rm:+.4f}, Δ score: {d_sc:+.4f}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
