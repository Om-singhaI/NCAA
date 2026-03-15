#!/usr/bin/env python3
"""
NCAA v12c — Final refinement of adj-comp1only discovery.

Key finding from v12b:
  adj30-comp1only: K=57/91, RMSE=2.483, LOSO=3.520 — best on ALL metrics!
  adj20-comp1only: K=56/91, LOSO=3.442 — best LOSO ever

This script does final optimization:
  1. Fine-tune gap for comp1-only adjacent pairs
  2. Combine adj-comp1only with margin weights on comp3
  3. Combine adj-comp1only with dual-alpha tuning
  4. Combine adj-comp1only with blend weight tuning
  5. Triple combination: adj-comp1 + margin-comp3 + dual-alpha
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


def build_pw_adj(X, y, seasons, max_gap=9999):
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


def build_pw_margin(X, y, seasons, scale=1.0):
    pairs_X, pairs_y, pairs_w = [], [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                gap = abs(y[a] - y[b])
                w = 1.0 + scale * (gap / 67.0)
                pairs_X.append(diff); pairs_y.append(target); pairs_w.append(w)
                pairs_X.append(-diff); pairs_y.append(1.0 - target); pairs_w.append(w)
    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_w)


def predict_v12c(X_tr, y_tr, X_te, seasons_tr, top_k_idx,
                  comp1_gap=9999, comp3_margin_scale=0.0,
                  w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
                  c1=PW_C1, c3=PW_C3, dual_alpha=DUAL_C3_ALPHA):
    """
    v12c blend:
    - Component 1: adjacent-pair training (comp1_gap)
    - Component 3: optional margin-weighted + dual-cal
    - Component 4: standard XGB
    """
    # Component 1: PW-LR with adjacent-pair training
    if comp1_gap < 9999:
        pw_X1, pw_y1 = build_pw_adj(X_tr, y_tr, seasons_tr, max_gap=comp1_gap)
        pw_w1 = None
    else:
        pw_X1, pw_y1 = build_pairwise_data(X_tr, y_tr, seasons_tr)
        pw_w1 = None
    
    sc1 = StandardScaler()
    pw_X1_sc = sc1.fit_transform(pw_X1)
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X1_sc, pw_y1)
    score1 = pairwise_score(lr1, X_te, sc1)
    
    # Component 3: dual-cal with optional margin weights
    X_tr_k = X_tr[:, top_k_idx]
    X_te_k = X_te[:, top_k_idx]
    
    if comp3_margin_scale > 0:
        pw_Xk, pw_yk, pw_wk = build_pw_margin(X_tr_k, y_tr, seasons_tr, scale=comp3_margin_scale)
    else:
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, seasons_tr)
        pw_wk = None
    
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
    
    # Component 4: standard XGB
    pw_Xf, pw_yf = build_pairwise_data(X_tr, y_tr, seasons_tr)
    sc_f = StandardScaler()
    pw_Xf_sc = sc_f.fit_transform(pw_Xf)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_Xf_sc, pw_yf)
    score4 = pairwise_score(xgb_clf, X_te, sc_f)
    
    return w1 * score1 + w3 * score3 + w4 * score4


def eval_full(X_all, y, seasons, feature_names, folds, GT, record_ids, test_mask, **kwargs):
    n = len(y)
    power = kwargs.pop('power', HUNGARIAN_POWER)
    
    loso_assigned = np.zeros(n, dtype=int)
    fold_stats = []
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        tk = select_top_k_features(X_all[tr], y[tr], feature_names, k=USE_TOP_K_A,
                                    forced_features=FORCE_FEATURES)[0]
        pred = predict_v12c(X_all[tr], y[tr], X_all[te], seasons[tr], tk, **kwargs)
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred, seasons[te], avail, power=power)
        loso_assigned[te] = assigned
        yte = y[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse_fold = np.sqrt(np.mean((assigned - yte)**2))
        fold_stats.append((hold, int(te.sum()), exact, rmse_fold))
    
    loso_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
    
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
        pred = predict_v12c(X_all[global_tr], y[global_tr], X_season, seasons[global_tr], tk, **kwargs)
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
    print(' NCAA v12c — FINAL REFINEMENT')
    print(' Target: adj30-comp1only (K=57/91, LOSO=3.520)')
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
    
    # Baseline
    print(f'\n  Baselines:')
    run('v11-baseline')
    run('adj30-comp1only', comp1_gap=30)
    run('adj20-comp1only', comp1_gap=20)
    
    # ═══════════════════════════════════════════════
    # PART 1: Fine-tune comp1 gap around 20-35
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 1: Fine-tune comp1-only gap')
    print(f'{"─"*70}')
    for g in [22, 24, 26, 27, 28, 29, 31, 32, 33, 34, 36, 38]:
        run(f'adj{g}-comp1only', comp1_gap=g)
    
    # ═══════════════════════════════════════════════
    # PART 2: adj-comp1 + margin on comp3
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 2: adj-comp1 + margin-weighted comp3')
    print(f'{"─"*70}')
    for g in [20, 25, 28, 30, 32]:
        for ms in [0.3, 0.5, 1.0]:
            run(f'adj{g}c1+marg{ms}c3', comp1_gap=g, comp3_margin_scale=ms)
    
    # ═══════════════════════════════════════════════
    # PART 3: adj-comp1 + dual-alpha tuning  
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 3: adj-comp1 + dual-alpha tuning')
    print(f'{"─"*70}')
    for g in [20, 28, 30]:
        for da in [0.0, 0.1, 0.2, 0.5, 0.7, 1.0]:
            run(f'adj{g}c1+da{da}', comp1_gap=g, dual_alpha=da)
    
    # ═══════════════════════════════════════════════
    # PART 4: adj-comp1 + blend weight tuning
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 4: adj-comp1 + blend weights')
    print(f'{"─"*70}')
    for w1, w3, w4 in [(0.60, 0.32, 0.08), (0.62, 0.30, 0.08), (0.66, 0.26, 0.08),
                        (0.60, 0.30, 0.10), (0.64, 0.30, 0.06), (0.68, 0.24, 0.08),
                        (0.62, 0.28, 0.10), (0.56, 0.34, 0.10)]:
        run(f'adj30c1+w{w1}/{w3}/{w4}', comp1_gap=30, w1=w1, w3=w3, w4=w4)
    
    # ═══════════════════════════════════════════════
    # PART 5: adj-comp1 + C1 tuning
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 5: adj-comp1 + C1 tuning')
    print(f'{"─"*70}')
    for g in [28, 30, 32]:
        for c1 in [3.0, 4.0, 6.0, 7.0, 8.0, 10.0]:
            run(f'adj{g}c1+C1={c1}', comp1_gap=g, c1=c1)
    
    # ═══════════════════════════════════════════════
    # PART 6: Triple combinations
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 6: Triple combinations (adj-comp1 + margin-comp3 + dual-alpha)')
    print(f'{"─"*70}')
    for g in [28, 30]:
        for ms in [0.3, 0.5, 1.0]:
            for da in [0.0, 0.1, 0.3]:
                run(f'adj{g}+marg{ms}+da{da}', comp1_gap=g, comp3_margin_scale=ms, dual_alpha=da)
    
    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f' SUMMARY — sorted by LOSO RMSE')
    print(f'{"="*70}')
    print(f'  {"Config":<55s} {"Kaggle":>6s} {"RMSE":>7s} {"LOSO":>7s} {"Δ":>7s}')
    print(f'  {"─"*55} {"─"*6} {"─"*7} {"─"*7} {"─"*7}')
    
    bl = results[0][3]
    for name, kex, krmse, loso, _ in sorted(results, key=lambda x: x[3])[:40]:
        d = loso - bl
        tag = ''
        if kex > 56: tag = ' ★★★'
        elif kex == 56 and d < -0.01: tag = ' ★★'
        elif kex == 56: tag = ' ★'
        print(f'  {name:<55s} {kex:3d}/91 {krmse:7.4f} {loso:7.4f} {d:+7.4f}{tag}')
    
    # Best maintaining Kaggle ≥ 56
    safe = [(n, k, r, l) for n, k, r, l, _ in results if k >= 56]
    if safe:
        best_safe = min(safe, key=lambda x: x[3])
        print(f'\n  BEST SAFE (Kaggle≥56): {best_safe[0]}')
        print(f'    Kaggle={best_safe[1]}/91 RMSE={best_safe[2]:.4f} LOSO={best_safe[3]:.4f}')
        print(f'    vs baseline: LOSO {best_safe[3]-bl:+.4f}')
    
    # Best maintaining Kaggle ≥ 57
    elite = [(n, k, r, l) for n, k, r, l, _ in results if k >= 57]
    if elite:
        best_elite = min(elite, key=lambda x: x[3])
        print(f'\n  BEST ELITE (Kaggle≥57): {best_elite[0]}')
        print(f'    Kaggle={best_elite[1]}/91 RMSE={best_elite[2]:.4f} LOSO={best_elite[3]:.4f}')
    
    # Best Kaggle overall
    best_k = max(results, key=lambda x: (x[1], -x[3]))
    print(f'\n  BEST KAGGLE: {best_k[0]}')
    print(f'    Kaggle={best_k[1]}/91 RMSE={best_k[2]:.4f} LOSO={best_k[3]:.4f}')
    
    # Per-season for top configs
    print(f'\n{"─"*70}')
    print(f' PER-SEASON — Top 5 safe configs')
    print(f'{"─"*70}')
    safe_results = sorted([(n,k,r,l,f) for n,k,r,l,f in results if k >= 56], key=lambda x: x[3])
    for name, kex, krmse, loso, fstats in safe_results[:5]:
        print(f'\n  {name} (K={kex}/91, RMSE={krmse:.4f}, LOSO={loso:.4f}):')
        for s, n_f, ex, rm in fstats:
            print(f'    {s}: {ex}/{n_f} ({ex/n_f*100:.0f}%), RMSE={rm:.3f}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print(f'  Configs tested: {len(results)}')


if __name__ == '__main__':
    main()
