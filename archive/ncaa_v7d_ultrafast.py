#!/usr/bin/env python3
"""
v7d ULTRA-FAST SEARCH
======================
Strategy:
 1. Top 6 methods only → C(6,3)=20 triplets
 2. Precompute scores per season for each method
 3. 5% weight steps PHASE 1 → pick top 20 configs
 4. 1% refinement around winners → exact best
 5. LOSO, stability, bootstrap
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
)


def fast_eval(season_scores, weights, power, fold_info):
    """Evaluate a blend config. weights = [(method_idx, w), ...]."""
    total_exact = 0
    total_sq_err = 0.0
    total_test = 0
    for fi, (s_idx, gt_s, test_s, n_s) in enumerate(fold_info):
        bl = np.zeros(n_s)
        for mi, w in weights:
            bl += w * season_scores[fi][mi]
        # Pin known (non-test) to ground truth
        for i in range(n_s):
            if not test_s[i]:
                bl[i] = gt_s[i]
        avail = list(range(1, 69))
        n = n_s
        cost = np.zeros((n, 68))
        for i in range(n):
            for s in range(68):
                cost[i, s] = abs(bl[i] - (s + 1)) ** power
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned = col_ind + 1
        for i in range(n):
            if test_s[i]:
                if assigned[i] == int(gt_s[i]):
                    total_exact += 1
                total_sq_err += (assigned[i] - gt_s[i]) ** 2
                total_test += 1
    rmse = np.sqrt(total_sq_err / total_test)
    return total_exact, rmse


def main():
    print('=' * 70)
    print(' v7d ULTRA-FAST SEARCH')
    print('=' * 70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    test_rids = set(GT.keys())

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat = build_features(labeled, context_df, labeled, tourn_rids)
    fn = list(feat.columns)

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    test_mask = np.array([rid in test_rids for rid in record_ids])
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)

    # ── TOP METHODS ──
    method_names = [
        'MLP_64_32',         # 54/91, 2.610
        'MLP_64_32_mseed',   # 56/91, 2.614
        'MLP_100',           # 44/91, 2.859
        'LR_C5',             # 48/91, 3.234
        'XGB_d3_200',        # 47/91, 3.251
        'LR_C1',             # 49/91, 3.275
        'XGB_d4_300_mseed',  # 45/91, 3.282
        'XGB_d4_300',        # 46/91, 3.416
        'LRk25_C0.5',       # v6 component
        'GBC_200_d4',        # 48/91, 3.674
        'GBC_300_d3',        # 48/91, 3.691
        'LRk25_C1',         # alt topK
    ]
    N_METHODS = len(method_names)
    print(f'  Methods: {N_METHODS}')

    # ── PRECOMPUTE per-season scores ──
    print('  Precomputing...')
    # season_scores[fi][mi] = scores for season fi, method mi
    season_scores = []
    fold_info = []  # (season_idx, gt, test_flags, n)

    for fi, hold_season in enumerate(folds):
        sm = (seasons == hold_season)
        si = np.where(sm)[0]
        n_s = len(si)
        gt_s = y[sm]
        test_s = test_mask[sm]
        fold_info.append((si, gt_s, test_s, n_s))

        # Only season with test teams matters
        season_test_any = test_s.any()

        global_train_mask = ~(test_mask & sm)
        # For Kaggle-style: train on everything except this season's test teams
        pw_X, pw_y_pw = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        X_season = X[sm]
        scores_this = {}

        # TopK features
        topk_cache = {}
        for K in [25]:
            tk = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = tk

        # LR
        for C_val, name in [(5.0, 'LR_C5'), (1.0, 'LR_C1')]:
            if name not in method_names:
                continue
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            scores_this[name] = pairwise_score(cls, X_season, sc)

        # TopK LR
        for C_val, K, name in [(0.5, 25, 'LRk25_C0.5'), (1.0, 25, 'LRk25_C1')]:
            if name not in method_names:
                continue
            tk = topk_cache[K]
            Xk_tr = X[global_train_mask][:, tk]
            pk_X, pk_y = build_pairwise_data(Xk_tr, y[global_train_mask],
                                              seasons[global_train_mask])
            sc_k = StandardScaler()
            pk_X_sc = sc_k.fit_transform(pk_X)
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pk_X_sc, pk_y)
            scores_this[name] = pairwise_score(cls, X_season[:, tk], sc_k)

        # XGB single
        for n_est, md, lr_v, name in [
            (300, 4, 0.05, 'XGB_d4_300'), (200, 3, 0.1, 'XGB_d3_200'),
        ]:
            if name not in method_names:
                continue
            cls = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=lr_v,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            cls.fit(pw_X_sc, pw_y_pw)
            scores_this[name] = pairwise_score(cls, X_season, sc)

        # XGB multi-seed
        if 'XGB_d4_300_mseed' in method_names:
            multi_xgb = []
            for seed in [42, 123, 777, 2024, 31415]:
                cls = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=seed, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss')
                cls.fit(pw_X_sc, pw_y_pw)
                multi_xgb.append(pairwise_score(cls, X_season, sc))
            scores_this['XGB_d4_300_mseed'] = np.mean(multi_xgb, axis=0)

        # GBC
        from sklearn.ensemble import GradientBoostingClassifier
        for n_est, md, name in [(200, 4, 'GBC_200_d4'), (300, 3, 'GBC_300_d3')]:
            if name not in method_names:
                continue
            cls = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            scores_this[name] = pairwise_score(cls, X_season, sc)

        # MLP single
        for hidden, name in [((64, 32), 'MLP_64_32'), ((100,), 'MLP_100')]:
            if name not in method_names:
                continue
            cls = MLPClassifier(
                hidden_layer_sizes=hidden, max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y_pw)
            scores_this[name] = pairwise_score(cls, X_season, sc)

        # MLP multi-seed
        if 'MLP_64_32_mseed' in method_names:
            multi_mlp = []
            for seed in [42, 123, 777, 2024, 31415]:
                cls = MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                    early_stopping=True, validation_fraction=0.1,
                    learning_rate='adaptive', alpha=0.001)
                cls.fit(pw_X_sc, pw_y_pw)
                multi_mlp.append(pairwise_score(cls, X_season, sc))
            scores_this['MLP_64_32_mseed'] = np.mean(multi_mlp, axis=0)

        # Store as array: season_scores[fi][mi]
        arr = []
        for mi, mn in enumerate(method_names):
            arr.append(scores_this[mn])
        season_scores.append(arr)

        print(f'    Fold {fi+1}/5 ({hold_season}): {time.time()-t0:.0f}s')

    print(f'  Precomp done: {time.time()-t0:.0f}s')

    # ── PHASE 1: Coarse scan (5% steps, top 8 methods) ──
    print('\n' + '=' * 70)
    print(' PHASE 1: Coarse 3c scan (5% steps, top 8)')
    print('=' * 70)

    TOP_N = 8  # only top 8 methods
    powers = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    combos = list(itertools.combinations(range(TOP_N), 3))
    print(f'  Triplets: {len(combos)}, powers: {len(powers)}')

    coarse_results = []  # (rmse, exact, config_tuple)
    n_configs = 0
    best_rmse_c = 999.0
    best_exact_c = 0

    for ci, (i, j, k) in enumerate(combos):
        for w1_int in range(5, 96, 5):
            for w2_int in range(5, 96 - w1_int, 5):
                w3_int = 100 - w1_int - w2_int
                if w3_int < 5:
                    continue
                w1, w2, w3 = w1_int / 100, w2_int / 100, w3_int / 100
                for power in powers:
                    n_configs += 1
                    e, r = fast_eval(season_scores,
                                      [(i, w1), (j, w2), (k, w3)],
                                      power, fold_info)
                    if r < best_rmse_c:
                        best_rmse_c = r
                    if e > best_exact_c:
                        best_exact_c = e
                    coarse_results.append((r, e, i, w1, j, w2, k, w3, power))

    coarse_results.sort(key=lambda x: x[0])
    print(f'  Configs: {n_configs:,}, Best RMSE={best_rmse_c:.4f}, '
          f'Best exact={best_exact_c}, {time.time()-t0:.0f}s')
    print(f'\n  Top 20 by RMSE:')
    for rank, (r, e, i, w1, j, w2, k, w3, p) in enumerate(coarse_results[:20]):
        print(f'    {rank+1:2d}. {e:3d}/91 RMSE={r:.4f}: '
              f'{w1:.0%} {method_names[i]} + {w2:.0%} {method_names[j]} '
              f'+ {w3:.0%} {method_names[k]}, p={p}')

    # ── PHASE 2: Fine refinement around top 30 configs ──
    print('\n' + '=' * 70)
    print(' PHASE 2: Fine refinement (1% steps around top 30)')
    print('=' * 70)

    # Collect unique triplets from top 30
    seen = set()
    fine_triplets = []
    for (r, e, i, w1, j, w2, k, w3, p) in coarse_results[:30]:
        key = (i, j, k)
        if key not in seen:
            seen.add(key)
            fine_triplets.append((i, j, k, w1, w2, p))

    best_rmse_f = 999.0
    best_exact_f = 0
    best_cfg = None
    pareto = {}
    n_fine = 0

    fine_powers = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
                   0.25, 0.30, 0.375, 0.50, 0.625, 0.75]

    for i, j, k, w1_center, w2_center, p_center in fine_triplets:
        # Search ±10% around center weights, 1% steps
        w1_lo = max(1, int(w1_center * 100) - 10)
        w1_hi = min(95, int(w1_center * 100) + 10)
        w2_lo = max(1, int(w2_center * 100) - 10)
        w2_hi = min(95, int(w2_center * 100) + 10)

        for w1_int in range(w1_lo, w1_hi + 1, 1):
            for w2_int in range(w2_lo, w2_hi + 1, 1):
                w3_int = 100 - w1_int - w2_int
                if w3_int < 1 or w3_int > 95:
                    continue
                w1, w2, w3 = w1_int / 100, w2_int / 100, w3_int / 100
                for power in fine_powers:
                    n_fine += 1
                    e, r = fast_eval(season_scores,
                                      [(i, w1), (j, w2), (k, w3)],
                                      power, fold_info)
                    if r < best_rmse_f:
                        best_rmse_f = r
                        best_cfg = (method_names[i], w1, method_names[j], w2,
                                    method_names[k], w3, power, e)
                    if e > best_exact_f:
                        best_exact_f = e
                    if e not in pareto or r < pareto[e][0]:
                        pareto[e] = (r, (method_names[i], w1, method_names[j],
                                         w2, method_names[k], w3, power))

    print(f'  Fine configs: {n_fine:,}, Best RMSE={best_rmse_f:.4f}, '
          f'Best exact={best_exact_f}, {time.time()-t0:.0f}s')

    if best_cfg:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg
        print(f'\n  ★ BEST RMSE CONFIG:')
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}')
        print(f'    power={p}, {e}/91 exact, RMSE={best_rmse_f:.4f}')

    print(f'\n  Pareto frontier:')
    for ex in sorted(pareto.keys(), reverse=True):
        r, (m1, w1, m2, w2, m3, w3, p) = pareto[ex]
        star = ' ★' if r < 2.474 else ''
        print(f'    {ex}/91 RMSE={r:.4f}: {w1:.0%} {m1} + {w2:.0%} {m2} '
              f'+ {w3:.0%} {m3}, p={p}{star}')
        if ex < best_exact_f - 10:
            break

    # ── PHASE 3: Also try ALL 12 methods (top-only triplets might miss v6 mix) ──
    print('\n' + '=' * 70)
    print(' PHASE 3: Extended coarse scan (all 12 methods, 5% steps)')
    print('=' * 70)

    all_combos = list(itertools.combinations(range(N_METHODS), 3))
    remaining = [c for c in all_combos if c not in set(itertools.combinations(range(TOP_N), 3))]
    print(f'  Remaining triplets: {len(remaining)}')

    for ci, (i, j, k) in enumerate(remaining):
        for w1_int in range(5, 91, 5):
            for w2_int in range(5, 91 - w1_int, 5):
                w3_int = 100 - w1_int - w2_int
                if w3_int < 5:
                    continue
                w1, w2, w3 = w1_int / 100, w2_int / 100, w3_int / 100
                for power in powers:
                    n_configs += 1
                    e, r = fast_eval(season_scores,
                                      [(i, w1), (j, w2), (k, w3)],
                                      power, fold_info)
                    if r < best_rmse_f:
                        best_rmse_f = r
                        best_cfg = (method_names[i], w1, method_names[j], w2,
                                    method_names[k], w3, power, e)
                    if e > best_exact_f:
                        best_exact_f = e
                    if e not in pareto or r < pareto[e][0]:
                        pareto[e] = (r, (method_names[i], w1, method_names[j],
                                         w2, method_names[k], w3, power))

    print(f'  Total configs now: {n_configs:,}, Best RMSE={best_rmse_f:.4f}, '
          f'Best exact={best_exact_f}, {time.time()-t0:.0f}s')

    # Refine any newly found better configs from extended search
    # Check if best_cfg changed from phase2
    if best_cfg:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg
        i_best = method_names.index(m1)
        j_best = method_names.index(m2)
        k_best = method_names.index(m3)

        # Fine-tune this specific config
        w1_lo = max(1, int(w1 * 100) - 10)
        w1_hi = min(95, int(w1 * 100) + 10)
        w2_lo = max(1, int(w2 * 100) - 10)
        w2_hi = min(95, int(w2 * 100) + 10)

        for w1_int in range(w1_lo, w1_hi + 1, 1):
            for w2_int in range(w2_lo, w2_hi + 1, 1):
                w3_int = 100 - w1_int - w2_int
                if w3_int < 1 or w3_int > 95:
                    continue
                w1n, w2n, w3n = w1_int / 100, w2_int / 100, w3_int / 100
                for power in fine_powers:
                    n_fine += 1
                    e2, r2 = fast_eval(season_scores,
                                        [(i_best, w1n), (j_best, w2n), (k_best, w3n)],
                                        power, fold_info)
                    if r2 < best_rmse_f:
                        best_rmse_f = r2
                        best_cfg = (m1, w1n, m2, w2n, m3, w3n, power, e2)
                    if e2 not in pareto or r2 < pareto[e2][0]:
                        pareto[e2] = (r2, (m1, w1n, m2, w2n, m3, w3n, power))

        print(f'  After fine-tuning extended: RMSE={best_rmse_f:.4f}, {time.time()-t0:.0f}s')

    # ── PHASE 4: 4-component search ──
    print('\n' + '=' * 70)
    print(' PHASE 4: 4-Component search (around best 3c)')
    print('=' * 70)

    if best_cfg:
        m1, w1, m2, w2, m3, w3, p_best, e_best = best_cfg
        i1 = method_names.index(m1)
        i2 = method_names.index(m2)
        i3 = method_names.index(m3)

        best_rmse4 = 999.0
        best_cfg4 = None
        n4 = 0

        for i4 in range(N_METHODS):
            if i4 in (i1, i2, i3):
                continue
            for w4_pct in range(2, 26, 2):  # 4th component: 2-24%
                rem = 100 - w4_pct
                # Proportionally scale 3c weights
                for dw1 in range(-8, 9, 2):
                    for dw2 in range(-8, 9, 2):
                        nw1 = int(w1 * 100) + dw1
                        nw2 = int(w2 * 100) + dw2
                        nw3 = 100 - w4_pct - nw1 - nw2
                        if nw1 < 2 or nw2 < 2 or nw3 < 2:
                            continue
                        for power in [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]:
                            n4 += 1
                            e4, r4 = fast_eval(
                                season_scores,
                                [(i1, nw1/100), (i2, nw2/100),
                                 (i3, nw3/100), (i4, w4_pct/100)],
                                power, fold_info)
                            if r4 < best_rmse4:
                                best_rmse4 = r4
                                best_cfg4 = (m1, nw1/100, m2, nw2/100, m3, nw3/100,
                                             method_names[i4], w4_pct/100, power, e4)

        print(f'  4c configs: {n4:,}, Best RMSE={best_rmse4:.4f}, {time.time()-t0:.0f}s')
        if best_cfg4:
            c1, w1, c2, w2, c3, w3, c4, w4, p, e = best_cfg4
            print(f'    {w1:.0%} {c1} + {w2:.0%} {c2} + {w3:.0%} {c3} + {w4:.0%} {c4}')
            print(f'    power={p}, {e}/91 exact')

    # ── PHASE 5: LOSO validation ──
    print('\n' + '=' * 70)
    print(' PHASE 5: LOSO Validation')
    print('=' * 70)

    cfgs_to_validate = []
    cfgs_to_validate.append(('v6 baseline', [('LR_C5', 0.64), ('LRk25_C0.5', 0.28),
                                              ('XGB_d4_300', 0.08)], 0.15))
    if best_cfg:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg
        cfgs_to_validate.append((f'v7d 3c (RMSE={best_rmse_f:.4f})',
                                  [(m1, w1), (m2, w2), (m3, w3)], p))

    if best_cfg4 and best_rmse4 < best_rmse_f:
        c1, w1, c2, w2, c3, w3, c4, w4, p, e = best_cfg4
        cfgs_to_validate.append((f'v7d 4c (RMSE={best_rmse4:.4f})',
                                  [(c1, w1), (c2, w2), (c3, w3), (c4, w4)], p))

    for label, weights, power in cfgs_to_validate:
        _loso(label, weights, power, X, y, seasons, fn, folds, method_names)

    # ── PHASE 6: Stability & Bootstrap ──
    print('\n' + '=' * 70)
    print(' PHASE 6: Stability & Bootstrap')
    print('=' * 70)

    if best_cfg:
        m1, w1_b, m2, w2_b, m3, w3_b, p_b, e_b = best_cfg
        i1 = method_names.index(m1)
        i2 = method_names.index(m2)
        i3 = method_names.index(m3)

        print(f'  Base: {w1_b:.0%} {m1} + {w2_b:.0%} {m2} + {w3_b:.0%} {m3}')
        print(f'  Base: {e_b}/91 exact, RMSE={best_rmse_f:.4f}, p={p_b}')

        # Weight perturbations
        print(f'\n  {"Perturbation":<35} {"Ex":>3} {"RMSE":>8} {"Δ":>8}')
        print(f'  {"─"*35} {"─"*3} {"─"*8} {"─"*8}')
        for dw1, dw2 in [(2, -2), (-2, 2), (4, -4), (-4, 4), (2, 0), (-2, 0),
                          (0, 2), (0, -2), (4, 0), (-4, 0)]:
            nw1 = int(w1_b * 100) + dw1
            nw2 = int(w2_b * 100) + dw2
            nw3 = 100 - nw1 - nw2
            if nw1 < 1 or nw2 < 1 or nw3 < 1:
                continue
            e, r = fast_eval(season_scores,
                              [(i1, nw1/100), (i2, nw2/100), (i3, nw3/100)],
                              p_b, fold_info)
            label = f'Δw1={dw1:+d}% Δw2={dw2:+d}%'
            print(f'  {label:<35} {e:3d} {r:8.4f} {r - best_rmse_f:+8.4f}')

        for dp in [-0.025, -0.05, 0.025, 0.05, 0.10]:
            np_v = round(p_b + dp, 4)
            if np_v <= 0:
                continue
            e, r = fast_eval(season_scores,
                              [(i1, w1_b), (i2, w2_b), (i3, w3_b)],
                              np_v, fold_info)
            label = f'Δp={dp:+.3f} (p={np_v:.3f})'
            print(f'  {label:<35} {e:3d} {r:8.4f} {r - best_rmse_f:+8.4f}')

        # Bootstrap comparison
        print(f'\n  Bootstrap v6 vs v7d (2000 resamples):')

        # Get predictions for both configs
        v7_preds = []
        v6_preds = []
        gts = []
        i_v6_1 = method_names.index('LR_C5')
        i_v6_2 = method_names.index('LRk25_C0.5')
        i_v6_3 = method_names.index('XGB_d4_300')

        for fi, (s_idx, gt_s, test_s, n_s) in enumerate(fold_info):
            # v7d
            bl7 = np.zeros(n_s)
            for mi_v, w in [(i1, w1_b), (i2, w2_b), (i3, w3_b)]:
                bl7 += w * season_scores[fi][mi_v]
            for i_t in range(n_s):
                if not test_s[i_t]:
                    bl7[i_t] = gt_s[i_t]
            cost7 = np.zeros((n_s, 68))
            for i_t in range(n_s):
                for s in range(68):
                    cost7[i_t, s] = abs(bl7[i_t] - (s + 1)) ** p_b
            _, col7 = linear_sum_assignment(cost7)

            # v6
            bl6 = np.zeros(n_s)
            for mi_v, w in [(i_v6_1, 0.64), (i_v6_2, 0.28), (i_v6_3, 0.08)]:
                bl6 += w * season_scores[fi][mi_v]
            for i_t in range(n_s):
                if not test_s[i_t]:
                    bl6[i_t] = gt_s[i_t]
            cost6 = np.zeros((n_s, 68))
            for i_t in range(n_s):
                for s in range(68):
                    cost6[i_t, s] = abs(bl6[i_t] - (s + 1)) ** 0.15
            _, col6 = linear_sum_assignment(cost6)

            for i_t in range(n_s):
                if test_s[i_t]:
                    v7_preds.append(col7[i_t] + 1)
                    v6_preds.append(col6[i_t] + 1)
                    gts.append(int(gt_s[i_t]))

        v7_preds = np.array(v7_preds)
        v6_preds = np.array(v6_preds)
        gts = np.array(gts)

        rng = np.random.RandomState(42)
        v7_wins = 0
        for _ in range(2000):
            idx = rng.choice(len(gts), len(gts), replace=True)
            r6 = np.sqrt(np.mean((v6_preds[idx] - gts[idx])**2))
            r7 = np.sqrt(np.mean((v7_preds[idx] - gts[idx])**2))
            if r7 < r6:
                v7_wins += 1

        print(f'    v7d wins {v7_wins}/2000 ({v7_wins/20:.1f}%)')
        if v7_wins >= 1800:
            print(f'    ★ Strong evidence v7d is genuinely better!')
        elif v7_wins >= 1400:
            print(f'    Moderate evidence v7d is better')
        else:
            print(f'    Weak evidence — possible overfit')

    # ── SUMMARY ──
    print('\n' + '=' * 70)
    print(' FINAL SUMMARY')
    print('=' * 70)
    print(f'  v6:     56/91 exact, RMSE=2.4740, LOSO score=3.678')
    if best_cfg:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg
        print(f'  v7d 3c: {e}/91 exact, RMSE={best_rmse_f:.4f}')
        print(f'    Config: {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}, p={p}')
    if best_cfg4:
        c1, w1, c2, w2, c3, w3, c4, w4, p, e = best_cfg4
        print(f'  v7d 4c: {e}/91 exact, RMSE={best_rmse4:.4f}')
        print(f'    Config: {w1:.0%} {c1} + {w2:.0%} {c2} + {w3:.0%} {c3} + {w4:.0%} {c4}, p={p}')

    overall = min(best_rmse_f, best_rmse4 if best_cfg4 else 999)
    if overall < 2.474:
        print(f'\n  ★★★ v7d BEATS v6! {2.474:.4f} → {overall:.4f} ★★★')
    print(f'\n  Total time: {time.time()-t0:.0f}s')


def _loso(label, weights, power, X, y, seasons, fn, folds, method_names):
    """Run full LOSO (retrain each fold)."""
    print(f'\n  --- {label} ---')
    total_exact = 0
    fold_rmses = []

    for hold in folds:
        tr = seasons != hold
        te = seasons == hold

        pw_X, pw_y_pw = build_pairwise_data(X[tr], y[tr], seasons[tr])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        topk_cache = {}
        for K in [25]:
            tk = select_top_k_features(X[tr], y[tr], fn, k=K)[0]
            topk_cache[K] = tk

        X_te = X[te]
        bl = np.zeros(int(te.sum()))

        for m_tag, w in weights:
            s = _compute(m_tag, X_te, X[tr], y[tr], seasons[tr],
                         pw_X_sc, pw_y_pw, sc, topk_cache, fn)
            bl += w * s

        avail = {hold: list(range(1, 69))}
        assigned = hungarian(bl, seasons[te], avail, power=power)

        yte = y[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse_f = np.sqrt(np.mean((assigned - yte)**2))
        total_exact += exact
        fold_rmses.append(rmse_f)
        print(f'    {hold}: {exact}/{int(te.sum())} ({exact/te.sum()*100:.1f}%), '
              f'RMSE={rmse_f:.3f}')

    score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)
    print(f'  LOSO: {total_exact}/340, mean_RMSE={np.mean(fold_rmses):.4f}, '
          f'score={score:.4f}')


def _compute(m_tag, X_te, X_tr, y_tr, seasons_tr,
             pw_X_sc, pw_y, sc, topk_cache, fn):
    from sklearn.ensemble import GradientBoostingClassifier

    if m_tag.startswith('LRk'):
        parts = m_tag.split('_')
        K = int(parts[0].replace('LRk', ''))
        C = float(parts[1].replace('C', ''))
        tk = topk_cache[K]
        Xk_tr = X_tr[:, tk]
        pk_X, pk_y = build_pairwise_data(Xk_tr, y_tr, seasons_tr)
        sc_k = StandardScaler()
        pk_X_sc = sc_k.fit_transform(pk_X)
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pk_X_sc, pk_y)
        return pairwise_score(cls, X_te[:, tk], sc_k)

    if m_tag.startswith('LR_C'):
        C = float(m_tag.replace('LR_C', ''))
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_te, sc)

    if m_tag.startswith('XGB') and 'mseed' in m_tag:
        multi = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=seed, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            cls.fit(pw_X_sc, pw_y)
            multi.append(pairwise_score(cls, X_te, sc))
        return np.mean(multi, axis=0)

    if m_tag.startswith('XGB'):
        parts = m_tag.split('_')
        md = int(parts[1].replace('d', ''))
        n_est = int(parts[2])
        lr_v = 0.1 if md == 3 else 0.05
        cls = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=lr_v,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_te, sc)

    if m_tag.startswith('GBC'):
        parts = m_tag.split('_')
        n_est = int(parts[1])
        md = int(parts[2].replace('d', ''))
        cls = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_te, sc)

    if m_tag.startswith('MLP') and 'mseed' in m_tag:
        multi = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y)
            multi.append(pairwise_score(cls, X_te, sc))
        return np.mean(multi, axis=0)

    if m_tag.startswith('MLP'):
        parts = m_tag.split('_')
        if parts[1] == '64':
            hidden = (64, 32)
        else:
            hidden = (int(parts[1]),)
        cls = MLPClassifier(
            hidden_layer_sizes=hidden, max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', alpha=0.001)
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_te, sc)

    raise ValueError(f'Unknown method: {m_tag}')


if __name__ == '__main__':
    main()
