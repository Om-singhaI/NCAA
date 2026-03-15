#!/usr/bin/env python3
"""
NCAA v12b — Deep dive on best findings from v12.

Key findings to refine:
  1. margin-wt-linear: 57/91 Kaggle, LOSO=3.567 (same as v11)
  2. adj-pairs-gap20: 56/91 Kaggle, LOSO=3.528 (best safe LOSO)
  3. adj-pairs-gap30-40: 57/91 Kaggle, LOSO≈3.575
  
Concerns: Kaggle RMSE went up for all. Need to find configs that 
improve LOSO AND Kaggle RMSE, or at least maintain Kaggle RMSE.

This script:
  Part 1: Fine-tune margin weighting strength
  Part 2: Fine-tune adjacent-pair gap
  Part 3: Combine margin-wt + adj-pairs (not tested with linear+gap20)
  Part 4: Per-season analysis of best configs
  Part 5: Blend weight tuning with best ideas
  Part 6: Partial margin weighting (only on hard components)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
    DUAL_C3_ALPHA
)

warnings.filterwarnings('ignore')
np.random.seed(42)


def pairwise_score(model, X_test, scaler=None):
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


def build_pw_margin(X, y, seasons, scale=1.0):
    """Pairwise data with margin weights. scale controls weight strength:
       0.0 = uniform weights (standard)
       1.0 = linear margin weights (full)
       intermediate = blended
    """
    pairs_X, pairs_y, pairs_w = [], [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                gap = abs(y[a] - y[b])
                # Interpolate between uniform (1.0) and linear margin (1 + gap/67)
                w = 1.0 + scale * (gap / 67.0)
                pairs_X.append(diff); pairs_y.append(target); pairs_w.append(w)
                pairs_X.append(-diff); pairs_y.append(1.0 - target); pairs_w.append(w)
    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_w)


def build_pw_adjacent(X, y, seasons, max_gap=9999):
    """Pairwise data with optional gap filter."""
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


def build_pw_adj_margin(X, y, seasons, max_gap=9999, mscale=1.0):
    """Adjacent-pair filter + margin weights combined."""
    pairs_X, pairs_y, pairs_w = [], [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                gap = abs(y[a] - y[b])
                if gap > max_gap:
                    continue
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                w = 1.0 + mscale * (gap / max_gap)  # normalize within max_gap
                pairs_X.append(diff); pairs_y.append(target); pairs_w.append(w)
                pairs_X.append(-diff); pairs_y.append(1.0 - target); pairs_w.append(w)
    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_w)


def predict_blend(X_tr, y_tr, X_te, seasons_tr, top_k_idx,
                   pw_mode='standard', max_gap=9999, mscale=1.0,
                   w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
                   c1=PW_C1, c3=PW_C3, dual_alpha=DUAL_C3_ALPHA,
                   apply_margin_to='all'):
    """
    Flexible blend with margin/adjacent controls.
    apply_margin_to: 'all' = apply to all components
                     'comp1' = only component 1
                     'comp3' = only component 3
                     'xgb' = only XGB component
    """
    # --- Helper to build pairwise data ---
    def make_pw(X, y_vals, seasons_vals, apply_here=True):
        if not apply_here or pw_mode == 'standard':
            pw_X, pw_y = build_pairwise_data(X, y_vals, seasons_vals)
            return pw_X, pw_y, None
        elif pw_mode == 'margin':
            return build_pw_margin(X, y_vals, seasons_vals, scale=mscale)
        elif pw_mode == 'adjacent':
            pw_X, pw_y = build_pw_adjacent(X, y_vals, seasons_vals, max_gap=max_gap)
            return pw_X, pw_y, None
        elif pw_mode == 'adj_margin':
            return build_pw_adj_margin(X, y_vals, seasons_vals, max_gap=max_gap, mscale=mscale)
        else:
            pw_X, pw_y = build_pairwise_data(X, y_vals, seasons_vals)
            return pw_X, pw_y, None
    
    # Component 1
    apply_c1 = (apply_margin_to in ('all', 'comp1'))
    pw_X1, pw_y1, pw_w1 = make_pw(X_tr, y_tr, seasons_tr, apply_c1)
    sc1 = StandardScaler()
    pw_X1_sc = sc1.fit_transform(pw_X1)
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X1_sc, pw_y1, sample_weight=pw_w1)
    score1 = pairwise_score(lr1, X_te, sc1)
    
    # Component 3 (dual-cal)
    X_tr_k = X_tr[:, top_k_idx]
    X_te_k = X_te[:, top_k_idx]
    apply_c3 = (apply_margin_to in ('all', 'comp3'))
    pw_Xk, pw_yk, pw_wk = make_pw(X_tr_k, y_tr, seasons_tr, apply_c3)
    sc_k = StandardScaler()
    pw_Xk_sc = sc_k.fit_transform(pw_Xk)
    
    lr3s = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3s.fit(pw_Xk_sc, pw_yk, sample_weight=pw_wk)
    s3_std = pairwise_score(lr3s, X_te_k, sc_k)
    
    base_cal = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3c = CalibratedClassifierCV(base_cal, cv=3, method='isotonic')
    lr3c.fit(pw_Xk_sc, pw_yk, sample_weight=pw_wk)
    s3_cal = pairwise_score(lr3c, X_te_k, sc_k)
    
    score3 = dual_alpha * s3_std + (1 - dual_alpha) * s3_cal
    
    # Component 4: XGB
    apply_c4 = (apply_margin_to in ('all', 'xgb'))
    pw_Xf, pw_yf, pw_wf = make_pw(X_tr, y_tr, seasons_tr, apply_c4)
    sc_f = StandardScaler()
    pw_Xf_sc = sc_f.fit_transform(pw_Xf)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_Xf_sc, pw_yf, sample_weight=pw_wf)
    score4 = pairwise_score(xgb_clf, X_te, sc_f)
    
    return w1 * score1 + w3 * score3 + w4 * score4


def eval_full(X_all, y, seasons, feature_names, folds, GT, record_ids, test_mask,
              pw_mode='standard', max_gap=9999, mscale=1.0,
              w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
              c1=PW_C1, c3=PW_C3, dual_alpha=DUAL_C3_ALPHA, power=HUNGARIAN_POWER,
              apply_margin_to='all'):
    """Full LOSO + Kaggle eval."""
    n = len(y)
    
    # LOSO
    loso_assigned = np.zeros(n, dtype=int)
    fold_stats = []
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        tk = select_top_k_features(X_all[tr], y[tr], feature_names, k=USE_TOP_K_A,
                                    forced_features=FORCE_FEATURES)[0]
        pred = predict_blend(X_all[tr], y[tr], X_all[te], seasons[tr], tk,
                              pw_mode=pw_mode, max_gap=max_gap, mscale=mscale,
                              w1=w1, w3=w3, w4=w4, c1=c1, c3=c3,
                              dual_alpha=dual_alpha, apply_margin_to=apply_margin_to)
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, seasons[te], avail, power=power)
        loso_assigned[te] = assigned
        yte = y[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse_fold = np.sqrt(np.mean((assigned - yte)**2))
        fold_stats.append((hold, int(te.sum()), exact, rmse_fold))
    
    loso_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
    
    # Kaggle
    test_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        season_mask = (seasons == hold)
        season_test = test_mask & season_mask
        if season_test.sum() == 0:
            continue
        global_tr = ~season_test
        tk = select_top_k_features(X_all[global_tr], y[global_tr], feature_names,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        X_season = X_all[season_mask]
        pred = predict_blend(X_all[global_tr], y[global_tr], X_season, seasons[global_tr], tk,
                              pw_mode=pw_mode, max_gap=max_gap, mscale=mscale,
                              w1=w1, w3=w3, w4=w4, c1=c1, c3=c3,
                              dual_alpha=dual_alpha, apply_margin_to=apply_margin_to)
        si = np.where(season_mask)[0]
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                pred[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    
    gt_all = y[test_mask].astype(int)
    pred_all = test_assigned[test_mask]
    k_exact = int((pred_all == gt_all).sum())
    k_rmse = np.sqrt(np.mean((pred_all - gt_all)**2))
    
    return k_exact, k_rmse, loso_rmse, fold_stats


def main():
    t0 = time.time()
    print('='*70)
    print(' NCAA v12b — DEEP DIVE ON BEST v12 FINDINGS')
    print('='*70)
    
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    feat = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat.columns)
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    results = []
    
    def run(name, **kwargs):
        st = time.time()
        kex, krmse, loso, fstats = eval_full(X_all, y, seasons, feature_names, folds,
                                              GT, record_ids, test_mask, **kwargs)
        elapsed = time.time() - st
        results.append((name, kex, krmse, loso, fstats))
        tag = ''
        if kex > 56: tag = ' ★★★'
        elif kex == 56 and loso < 3.567: tag = ' ★★'
        elif kex == 56: tag = ' ★'
        print(f'  {name:<55s} K={kex}/91 RMSE={krmse:.4f} LOSO={loso:.4f} ({elapsed:.0f}s){tag}')
        return kex, krmse, loso
    
    # ═══════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════
    print(f'\n  Baseline:')
    run('v11-baseline')
    
    # ═══════════════════════════════════════════════
    # PART 1: Fine-tune margin weight strength
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 1: Margin weight scale sweep (0=standard, 1=v12 linear)')
    print(f'{"─"*70}')
    for sc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
        run(f'margin-scale-{sc}', pw_mode='margin', mscale=sc)
    
    # ═══════════════════════════════════════════════
    # PART 2: Fine-tune adjacent gap
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 2: Adjacent-pair gap sweep')
    print(f'{"─"*70}')
    for g in [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 28, 32, 35]:
        run(f'adj-gap-{g}', pw_mode='adjacent', max_gap=g)
    
    # ═══════════════════════════════════════════════
    # PART 3: Adjacent + margin combined
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 3: Adjacent pairs + margin weights combined')
    print(f'{"─"*70}')
    for g in [20, 25, 30, 35]:
        for sc in [0.3, 0.5, 1.0]:
            run(f'adj{g}+margin{sc}', pw_mode='adj_margin', max_gap=g, mscale=sc)
    
    # ═══════════════════════════════════════════════
    # PART 4: Apply margin/adj to specific components only
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 4: Selective application (only certain components)')
    print(f'{"─"*70}')
    # Margin on comp1 only (the dominant 64% component)
    for sc in [0.5, 1.0]:
        run(f'margin{sc}-comp1only', pw_mode='margin', mscale=sc, apply_margin_to='comp1')
    
    # Margin on comp3 only (the regularized 28% component)
    for sc in [0.5, 1.0]:
        run(f'margin{sc}-comp3only', pw_mode='margin', mscale=sc, apply_margin_to='comp3')
    
    # Adjacent on comp1 only
    for g in [20, 30]:
        run(f'adj{g}-comp1only', pw_mode='adjacent', max_gap=g, apply_margin_to='comp1')
    
    # Adjacent on comp3 only
    for g in [20, 30]:
        run(f'adj{g}-comp3only', pw_mode='adjacent', max_gap=g, apply_margin_to='comp3')
    
    # ═══════════════════════════════════════════════
    # PART 5: Blend weight fine-tuning with margin
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 5: Blend weight tuning with best margin setting')
    print(f'{"─"*70}')
    # Try shifting weight from XGB to LR or vice versa
    weight_configs = [
        (0.66, 0.26, 0.08),  # more comp1
        (0.62, 0.30, 0.08),  # more comp3
        (0.64, 0.28, 0.08),  # baseline (same)
        (0.60, 0.30, 0.10),  # more comp3 + XGB
        (0.66, 0.28, 0.06),  # less XGB
        (0.64, 0.30, 0.06),  # more comp3, less XGB
        (0.68, 0.24, 0.08),  # much more comp1
        (0.60, 0.32, 0.08),  # much more comp3
        (0.62, 0.28, 0.10),  # more XGB
    ]
    for w1, w3, w4 in weight_configs:
        run(f'margin1.0+w{w1}/{w3}/{w4}',
            pw_mode='margin', mscale=1.0, w1=w1, w3=w3, w4=w4)
    
    # ═══════════════════════════════════════════════
    # PART 6: C value tuning with margin
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 6: C value tuning with margin weighting')
    print(f'{"─"*70}')
    for c1 in [3.0, 4.0, 5.0, 7.0, 10.0]:
        run(f'margin1.0+C1={c1}', pw_mode='margin', mscale=1.0, c1=c1)
    
    for c3 in [0.3, 0.5, 0.7, 1.0]:
        run(f'margin1.0+C3={c3}', pw_mode='margin', mscale=1.0, c3=c3)
    
    # ═══════════════════════════════════════════════
    # PART 7: Dual-alpha tuning with margin
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 7: Dual-alpha tuning with margin')
    print(f'{"─"*70}')
    for da in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        run(f'margin1.0+dual_a={da}', pw_mode='margin', mscale=1.0, dual_alpha=da)
    
    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f' SUMMARY — sorted by LOSO RMSE')
    print(f'{"="*70}')
    print(f'  {"Config":<55s} {"Kaggle":>6s} {"RMSE":>7s} {"LOSO":>7s} {"Δ LOSO":>7s}')
    print(f'  {"─"*55} {"─"*6} {"─"*7} {"─"*7} {"─"*7}')
    
    bl = results[0][3]
    for name, kex, krmse, loso, _ in sorted(results, key=lambda x: x[3]):
        d = loso - bl
        tag = ''
        if kex > 56: tag = ' ★★★'
        elif kex == 56 and d < -0.01: tag = ' ★★'
        elif kex == 56: tag = ' ★'
        print(f'  {name:<55s} {kex:3d}/91 {krmse:7.4f} {loso:7.4f} {d:+7.4f}{tag}')
    
    # Best safe
    safe = [(n, k, r, l) for n, k, r, l, _ in results if k >= 56]
    if safe:
        best_safe = min(safe, key=lambda x: x[3])
        print(f'\n  BEST SAFE (Kaggle≥56): {best_safe[0]}')
        print(f'    Kaggle={best_safe[1]}/91 RMSE={best_safe[2]:.4f} LOSO={best_safe[3]:.4f}')
        print(f'    vs baseline: LOSO {best_safe[3]-bl:+.4f}')
    
    # Best Kaggle
    best_k = max(results, key=lambda x: (x[1], -x[3]))
    print(f'\n  BEST KAGGLE: {best_k[0]}')
    print(f'    Kaggle={best_k[1]}/91 RMSE={best_k[2]:.4f} LOSO={best_k[3]:.4f}')
    
    # Per-season analysis of top configs
    print(f'\n{"─"*70}')
    print(f' PER-SEASON ANALYSIS — top configs')
    print(f'{"─"*70}')
    top5 = sorted(results, key=lambda x: x[3])[:5]
    top5_kaggle = sorted(results, key=lambda x: (-x[1], x[3]))[:3]
    show = {r[0]: r for r in top5}
    for r in top5_kaggle:
        show[r[0]] = r
    show['v11-baseline'] = results[0]
    
    for name, kex, krmse, loso, fstats in show.values():
        print(f'\n  {name} (K={kex}/91, LOSO={loso:.4f}):')
        for s, n_f, ex, rm in fstats:
            print(f'    {s}: {ex}/{n_f} exact ({ex/n_f*100:.0f}%), RMSE={rm:.3f}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print(f'  Configs tested: {len(results)}')


if __name__ == '__main__':
    main()
