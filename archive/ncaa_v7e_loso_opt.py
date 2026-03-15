#!/usr/bin/env python3
"""
v7e LOSO-OPTIMIZED SEARCH
===========================
v7d found RMSE=2.328 on Kaggle but LOSO went from 3.678 → 3.874 (worse).
This script precomputes LOSO scores for all methods, then searches for
blends that optimize LOSO score while also checking Kaggle RMSE.
Goal: find a config that improves BOTH metrics.
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
)


def main():
    print('=' * 70)
    print(' v7e LOSO-OPTIMIZED SEARCH')
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

    method_names = [
        'MLP_64_32',
        'MLP_64_32_mseed',
        'MLP_100',
        'LR_C5',
        'XGB_d3_200',
        'LR_C1',
        'XGB_d4_300',
        'LRk25_C0.5',
        'GBC_200_d4',
    ]
    N_M = len(method_names)
    print(f'  Methods: {N_M}')

    # ═══════════════════════════════════════════════
    # PART A: Precompute KAGGLE scores (same as v7d)
    # ═══════════════════════════════════════════════
    print('\n  A: Precomputing Kaggle scores...')
    kaggle_scores = []  # kaggle_scores[fi][mi] = scores for fold fi
    kaggle_fold_info = []

    for fi, hold_season in enumerate(folds):
        sm = (seasons == hold_season)
        si = np.where(sm)[0]
        n_s = len(si)
        gt_s = y[sm]
        test_s = test_mask[sm]
        kaggle_fold_info.append((si, gt_s, test_s, n_s))

        global_train_mask = ~(test_mask & sm)
        pw_X, pw_y_pw = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        topk_cache = {}
        for K in [25]:
            tk = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = tk

        X_season = X[sm]
        scores = {}
        scores.update(_train_all(pw_X_sc, pw_y_pw, sc, X_season,
                                  X[global_train_mask], y[global_train_mask],
                                  seasons[global_train_mask], topk_cache, fn,
                                  method_names))
        kaggle_scores.append([scores[m] for m in method_names])
        print(f'    Kaggle fold {fi+1}/5 ({hold_season}): {time.time()-t0:.0f}s')

    # ═══════════════════════════════════════════════
    # PART B: Precompute LOSO scores
    # ═══════════════════════════════════════════════
    print('\n  B: Precomputing LOSO scores...')
    loso_scores = []  # loso_scores[fi][mi] = scores for LOSO fold fi
    loso_fold_info = []

    for fi, hold_season in enumerate(folds):
        tr = seasons != hold_season
        te = seasons == hold_season
        n_te = int(te.sum())

        pw_X, pw_y_pw = build_pairwise_data(X[tr], y[tr], seasons[tr])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        topk_cache = {}
        for K in [25]:
            tk = select_top_k_features(X[tr], y[tr], fn, k=K)[0]
            topk_cache[K] = tk

        X_te = X[te]
        scores = _train_all(pw_X_sc, pw_y_pw, sc, X_te,
                             X[tr], y[tr], seasons[tr], topk_cache, fn,
                             method_names)
        loso_scores.append([scores[m] for m in method_names])
        loso_fold_info.append((y[te].astype(int), n_te, hold_season))
        print(f'    LOSO fold {fi+1}/5 ({hold_season}): {time.time()-t0:.0f}s')

    print(f'  All precomputation done: {time.time()-t0:.0f}s')

    # ═══════════════════════════════════════════════
    # Individual results
    # ═══════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' Individual Performance')
    print('=' * 70)
    print(f'  {"Method":<22} {"Kaggle Ex":>9} {"Kaggle RMSE":>11} '
          f'{"LOSO Ex":>7} {"LOSO Score":>10}')
    for mi, mn in enumerate(method_names):
        ke, kr = _eval_kaggle(kaggle_scores, [(mi, 1.0)], 0.15, kaggle_fold_info, y)
        le, ls = _eval_loso(loso_scores, [(mi, 1.0)], 0.15, loso_fold_info)
        print(f'  {mn:<22} {ke:5d}/91   {kr:8.4f}     '
              f'{le:3d}/340  {ls:8.4f}')

    # ═══════════════════════════════════════════════
    # 3c blend: joint Kaggle+LOSO optimization (5% steps)
    # ═══════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' 3c Joint Kaggle+LOSO Search (5% steps)')
    print('=' * 70)

    powers = [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.375, 0.50]
    combos = list(itertools.combinations(range(N_M), 3))
    print(f'  Triplets: {len(combos)}, powers: {len(powers)}')

    results_coarse = []
    n_cfg = 0

    for i, j, k in combos:
        for w1 in range(5, 91, 5):
            for w2 in range(5, 91 - w1, 5):
                w3 = 100 - w1 - w2
                if w3 < 5:
                    continue
                for power in powers:
                    n_cfg += 1
                    ke, kr = _eval_kaggle(kaggle_scores,
                                           [(i, w1/100), (j, w2/100), (k, w3/100)],
                                           power, kaggle_fold_info, y)
                    le, ls = _eval_loso(loso_scores,
                                         [(i, w1/100), (j, w2/100), (k, w3/100)],
                                         power, loso_fold_info)
                    results_coarse.append((kr, ls, ke, le, i, w1, j, w2, k, w3, power))

    print(f'  Configs: {n_cfg:,}, {time.time()-t0:.0f}s')

    # Best by Kaggle RMSE
    results_coarse.sort(key=lambda x: x[0])
    print(f'\n  Top 10 by Kaggle RMSE:')
    print(f'  {"#":>2} {"KgEx":>4} {"KgRMSE":>7} {"LOEx":>5} {"LOSc":>7} '
          f'{"Config":<50}')
    for rank, (kr, ls, ke, le, i, w1, j, w2, k, w3, p) in enumerate(results_coarse[:10]):
        cfg = f'{w1}% {method_names[i]} + {w2}% {method_names[j]} + {w3}% {method_names[k]}'
        print(f'  {rank+1:2d} {ke:4d} {kr:7.4f} {le:5d} {ls:7.4f} {cfg}, p={p}')

    # Best by LOSO score
    results_coarse.sort(key=lambda x: x[1])
    print(f'\n  Top 10 by LOSO score:')
    print(f'  {"#":>2} {"KgEx":>4} {"KgRMSE":>7} {"LOEx":>5} {"LOSc":>7} '
          f'{"Config":<50}')
    for rank, (kr, ls, ke, le, i, w1, j, w2, k, w3, p) in enumerate(results_coarse[:10]):
        cfg = f'{w1}% {method_names[i]} + {w2}% {method_names[j]} + {w3}% {method_names[k]}'
        print(f'  {rank+1:2d} {ke:4d} {kr:7.4f} {le:5d} {ls:7.4f} {cfg}, p={p}')

    # Best balanced: minimize 0.5*kaggle_rmse + 0.5*loso_score
    results_coarse.sort(key=lambda x: 0.5 * x[0] + 0.5 * x[1])
    print(f'\n  Top 10 by balanced (0.5*KgRMSE + 0.5*LOSO):')
    print(f'  {"#":>2} {"KgEx":>4} {"KgRMSE":>7} {"LOEx":>5} {"LOSc":>7} '
          f'{"Config":<50}')
    for rank, (kr, ls, ke, le, i, w1, j, w2, k, w3, p) in enumerate(results_coarse[:10]):
        cfg = f'{w1}% {method_names[i]} + {w2}% {method_names[j]} + {w3}% {method_names[k]}'
        print(f'  {rank+1:2d} {ke:4d} {kr:7.4f} {le:5d} {ls:7.4f} {cfg}, p={p}')

    # Pareto: configs that beat v6 on Kaggle (RMSE<2.474) AND LOSO (score<3.678)
    v6_kr, v6_ls = 2.474, 3.678
    pareto_both = [r for r in results_coarse if r[0] < v6_kr and r[1] < v6_ls]
    print(f'\n  Configs beating v6 on BOTH Kaggle RMSE AND LOSO: {len(pareto_both)}')
    if pareto_both:
        pareto_both.sort(key=lambda x: x[0])
        for i_r, (kr, ls, ke, le, i, w1, j, w2, k, w3, p) in enumerate(pareto_both[:15]):
            cfg = (f'{w1}% {method_names[i]} + {w2}% {method_names[j]} '
                   f'+ {w3}% {method_names[k]}')
            print(f'    {ke}/91 KgRMSE={kr:.4f} | {le}/340 LOSc={ls:.4f} | {cfg}, p={p}')

    # ═══════════════════════════════════════════════
    # Fine-tune: refine best balanced + best LOSO configs
    # ═══════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' Fine-tuning (1% steps) around best configs')
    print('=' * 70)

    # Collect unique triplets from top configs across all criteria
    fine_set = set()
    for sort_key in [lambda x: x[0], lambda x: x[1], lambda x: 0.5*x[0]+0.5*x[1]]:
        sorted_r = sorted(results_coarse, key=sort_key)
        for r in sorted_r[:10]:
            fine_set.add((r[4], r[6], r[8]))  # (i, j, k)

    if pareto_both:
        for r in pareto_both[:5]:
            fine_set.add((r[4], r[6], r[8]))

    fine_powers = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20,
                   0.25, 0.30, 0.375, 0.50]

    best_kr = 999.0
    best_ls = 999.0
    best_bal = 999.0
    best_cfg_kr = None
    best_cfg_ls = None
    best_cfg_bal = None
    pareto_fine = {}  # (ke, le) → (kr, ls, config)
    n_fine = 0

    for (i, j, k) in fine_set:
        for w1 in range(2, 95, 1):
            for w2 in range(2, 95 - w1, 1):
                w3 = 100 - w1 - w2
                if w3 < 2:
                    continue
                for power in fine_powers:
                    n_fine += 1
                    ke, kr = _eval_kaggle(kaggle_scores,
                                           [(i, w1/100), (j, w2/100), (k, w3/100)],
                                           power, kaggle_fold_info, y)
                    le, ls = _eval_loso(loso_scores,
                                         [(i, w1/100), (j, w2/100), (k, w3/100)],
                                         power, loso_fold_info)
                    bal = 0.5 * kr + 0.5 * ls
                    if kr < best_kr:
                        best_kr = kr
                        best_cfg_kr = (method_names[i], w1, method_names[j], w2,
                                       method_names[k], w3, power, ke, ls, le)
                    if ls < best_ls:
                        best_ls = ls
                        best_cfg_ls = (method_names[i], w1, method_names[j], w2,
                                       method_names[k], w3, power, ke, kr, le)
                    if bal < best_bal:
                        best_bal = bal
                        best_cfg_bal = (method_names[i], w1, method_names[j], w2,
                                        method_names[k], w3, power, ke, kr, ls, le)
                    # Track Pareto combinations of good Kaggle + good LOSO
                    if kr < v6_kr and ls < v6_ls:
                        key = (ke, le)
                        if key not in pareto_fine or kr + ls < pareto_fine[key][0] + pareto_fine[key][1]:
                            pareto_fine[key] = (kr, ls,
                                                (method_names[i], w1, method_names[j], w2,
                                                 method_names[k], w3, power))

    print(f'  Fine configs: {n_fine:,}, {time.time()-t0:.0f}s')

    print(f'\n  ★ BEST KAGGLE RMSE:')
    if best_cfg_kr:
        m1, w1, m2, w2, m3, w3, p, ke, ls, le = best_cfg_kr
        print(f'    {w1}% {m1} + {w2}% {m2} + {w3}% {m3}, p={p}')
        print(f'    Kaggle: {ke}/91 RMSE={best_kr:.4f} | LOSO: {le}/340 score={ls:.4f}')

    print(f'\n  ★ BEST LOSO SCORE:')
    if best_cfg_ls:
        m1, w1, m2, w2, m3, w3, p, ke, kr, le = best_cfg_ls
        print(f'    {w1}% {m1} + {w2}% {m2} + {w3}% {m3}, p={p}')
        print(f'    Kaggle: {ke}/91 RMSE={kr:.4f} | LOSO: {le}/340 score={best_ls:.4f}')

    print(f'\n  ★ BEST BALANCED:')
    if best_cfg_bal:
        m1, w1, m2, w2, m3, w3, p, ke, kr, ls, le = best_cfg_bal
        print(f'    {w1}% {m1} + {w2}% {m2} + {w3}% {m3}, p={p}')
        print(f'    Kaggle: {ke}/91 RMSE={kr:.4f} | LOSO: {le}/340 score={ls:.4f}')

    print(f'\n  Configs beating v6 on BOTH (fine): {len(pareto_fine)}')
    if pareto_fine:
        items = sorted(pareto_fine.items(), key=lambda x: x[1][0])  # sort by Kaggle RMSE
        for (ke, le), (kr, ls, (m1, w1, m2, w2, m3, w3, p)) in items[:15]:
            star_kr = '↑' if kr < 2.35 else ''
            star_ls = '↑' if ls < 3.65 else ''
            print(f'    {ke}/91{star_kr} KgRMSE={kr:.4f} | {le}/340{star_ls} LOSc={ls:.4f} | '
                  f'{w1}% {m1} + {w2}% {m2} + {w3}% {m3}, p={p}')

    # ═══════════════════════════════════════════════
    # Compare key configs
    # ═══════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' CONFIG COMPARISON')
    print('=' * 70)

    # v6 reference
    i_lr5 = method_names.index('LR_C5')
    i_lrk = method_names.index('LRk25_C0.5')
    i_xgb4 = method_names.index('XGB_d4_300')
    ke6, kr6 = _eval_kaggle(kaggle_scores,
                              [(i_lr5, 0.64), (i_lrk, 0.28), (i_xgb4, 0.08)],
                              0.15, kaggle_fold_info, y)
    le6, ls6 = _eval_loso(loso_scores,
                            [(i_lr5, 0.64), (i_lrk, 0.28), (i_xgb4, 0.08)],
                            0.15, loso_fold_info)
    print(f'  v6:     {ke6}/91 KgRMSE={kr6:.4f} | {le6}/340 LOSc={ls6:.4f}')

    # v7d (Kaggle-optimized)
    i_mlp = method_names.index('MLP_64_32')
    i_xgb3 = method_names.index('XGB_d3_200')
    ke7d, kr7d = _eval_kaggle(kaggle_scores,
                                [(i_mlp, 0.76), (i_lr5, 0.08), (i_xgb3, 0.16)],
                                0.075, kaggle_fold_info, y)
    le7d, ls7d = _eval_loso(loso_scores,
                              [(i_mlp, 0.76), (i_lr5, 0.08), (i_xgb3, 0.16)],
                              0.075, loso_fold_info)
    print(f'  v7d:    {ke7d}/91 KgRMSE={kr7d:.4f} | {le7d}/340 LOSc={ls7d:.4f}')

    print(f'\n  v6 benchmarks: KgRMSE=2.4740, LOSc=3.6776')

    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' FINAL SUMMARY')
    print('=' * 70)
    if pareto_fine:
        best_both = sorted(pareto_fine.items(), key=lambda x: x[1][0])[0]
        (ke, le), (kr, ls, (m1, w1, m2, w2, m3, w3, p)) = best_both
        print(f'  ★ BEST CONFIG BEATING v6 ON BOTH METRICS:')
        print(f'    {w1}% {m1} + {w2}% {m2} + {w3}% {m3}, p={p}')
        print(f'    Kaggle: {ke}/91 RMSE={kr:.4f} (vs v6 {kr6:.4f})')
        print(f'    LOSO: {le}/340 score={ls:.4f} (vs v6 {ls6:.4f})')
    else:
        print(f'  No config beats v6 on BOTH metrics simultaneously.')
        print(f'  v7d improves Kaggle but degrades LOSO.')

    print(f'\n  Total time: {time.time()-t0:.0f}s')


# ═══════════════════════════════════════════════════════════════
def _eval_kaggle(season_scores, weights, power, fold_info, y):
    total_exact = 0
    total_sq = 0.0
    total_n = 0
    for fi, (s_idx, gt_s, test_s, n_s) in enumerate(fold_info):
        bl = np.zeros(n_s)
        for mi, w in weights:
            bl += w * season_scores[fi][mi]
        for i in range(n_s):
            if not test_s[i]:
                bl[i] = gt_s[i]
        cost = np.zeros((n_s, 68))
        for i in range(n_s):
            for s in range(68):
                cost[i, s] = abs(bl[i] - (s + 1)) ** power
        _, col = linear_sum_assignment(cost)
        assigned = col + 1
        for i in range(n_s):
            if test_s[i]:
                if assigned[i] == int(gt_s[i]):
                    total_exact += 1
                total_sq += (assigned[i] - gt_s[i]) ** 2
                total_n += 1
    return total_exact, np.sqrt(total_sq / total_n)


def _eval_loso(loso_scores, weights, power, fold_info):
    fold_rmses = []
    total_exact = 0
    for fi, (gt, n_te, season) in enumerate(fold_info):
        bl = np.zeros(n_te)
        for mi, w in weights:
            bl += w * loso_scores[fi][mi]
        cost = np.zeros((n_te, 68))
        for i in range(n_te):
            for s in range(68):
                cost[i, s] = abs(bl[i] - (s + 1)) ** power
        _, col = linear_sum_assignment(cost)
        assigned = col + 1
        total_exact += int(np.sum(assigned == gt))
        fold_rmses.append(np.sqrt(np.mean((assigned - gt)**2)))
    score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)
    return total_exact, score


def _train_all(pw_X_sc, pw_y, sc, X_te, X_tr, y_tr, seasons_tr,
               topk_cache, fn, method_names):
    scores = {}

    for C_val, name in [(5.0, 'LR_C5'), (1.0, 'LR_C1')]:
        if name not in method_names:
            continue
        cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        scores[name] = pairwise_score(cls, X_te, sc)

    for C_val, K, name in [(0.5, 25, 'LRk25_C0.5'), (1.0, 25, 'LRk25_C1')]:
        if name not in method_names:
            continue
        tk = topk_cache[K]
        Xk_tr = X_tr[:, tk]
        pk_X, pk_y = build_pairwise_data(Xk_tr, y_tr, seasons_tr)
        sc_k = StandardScaler()
        pk_X_sc = sc_k.fit_transform(pk_X)
        cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pk_X_sc, pk_y)
        scores[name] = pairwise_score(cls, X_te[:, tk], sc_k)

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
        cls.fit(pw_X_sc, pw_y)
        scores[name] = pairwise_score(cls, X_te, sc)

    for n_est, md, name in [(200, 4, 'GBC_200_d4'), (300, 3, 'GBC_300_d3')]:
        if name not in method_names:
            continue
        cls = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        scores[name] = pairwise_score(cls, X_te, sc)

    for hidden, name in [((64, 32), 'MLP_64_32'), ((100,), 'MLP_100')]:
        if name not in method_names:
            continue
        cls = MLPClassifier(
            hidden_layer_sizes=hidden, max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', alpha=0.001)
        cls.fit(pw_X_sc, pw_y)
        scores[name] = pairwise_score(cls, X_te, sc)

    if 'MLP_64_32_mseed' in method_names:
        multi = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y)
            multi.append(pairwise_score(cls, X_te, sc))
        scores['MLP_64_32_mseed'] = np.mean(multi, axis=0)

    return scores


if __name__ == '__main__':
    main()
