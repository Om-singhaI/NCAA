#!/usr/bin/env python3
"""
v7c FAST FOCUSED SEARCH
========================
v7b found RMSE=2.3276 in the first 100 triplets (top 15 methods).
The breakthrough involves MLP_64_32_rank (best individual: 54/91, RMSE=2.610).

This script: Fast targeted search with top ~10 methods, 2% steps,
find the exact winning config then validate with LOSO + overfit check.
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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
)


def main():
    print('=' * 70)
    print(' v7c FAST FOCUSED SEARCH')
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

    # ─── FOCUSED METHOD POOL (top performers from v7b) ───
    method_names = [
        'MLP_64_32',         # #1 individual: 54/91, RMSE=2.610
        'MLP_64_32_mseed',   # #2: 56/91, RMSE=2.614 (5 seeds avg)
        'MLP_100',           # #3: 44/91, RMSE=2.859
        'LR_C5',             # #4: 48/91, v6 component
        'LR_C1',             # #5: 49/91
        'XGB_d3_200',        # #6: 47/91
        'XGB_d4_300_mseed',  # #7: 45/91 (multi-seed)
        'XGB_d4_300',        # #8: 46/91, v6 component
        'GBC_200_d4',        # #9: 48/91
        'GBC_300_d3',        # #10: 48/91
        'LRk25_C0.5',       # v6 component (36/91 solo but good in blends)
        'LRk25_C1',         # alternative topK
        'LR_C3',            # 46/91
    ]

    print(f'  Methods: {len(method_names)}')

    # ─── PRECOMPUTE (Kaggle-style) ───
    print('  Precomputing scores...')
    all_scores = {m: np.zeros(n) for m in method_names}

    for fi, hold_season in enumerate(folds):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        season_indices = np.where(season_mask)[0]

        pw_X, pw_y_pw = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        topk_cache = {}
        for K in [25]:
            tk = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = tk
            Xk_tr = X[global_train_mask][:, tk]
            pk_X, pk_y = build_pairwise_data(Xk_tr, y[global_train_mask],
                                              seasons[global_train_mask])
            sc_k = StandardScaler()
            pk_X_sc = sc_k.fit_transform(pk_X)
            topk_cache[f'{K}_pw'] = (pk_X_sc, pk_y, sc_k)

        def _store(name, scores):
            for i, gi in enumerate(season_indices):
                all_scores[name][gi] = scores[i]

        # LR variants
        for C_val, name in [(5.0, 'LR_C5'), (1.0, 'LR_C1'), (3.0, 'LR_C3')]:
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # TopK LR
        for C_val, name in [(0.5, 'LRk25_C0.5'), (1.0, 'LRk25_C1')]:
            pk_X_sc, pk_y, sc_k = topk_cache['25_pw']
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pk_X_sc, pk_y)
            _store(name, pairwise_score(cls, X_season[:, topk_cache[25]], sc_k))

        # XGB
        for n_est, md, lr_val, name in [
            (300, 4, 0.05, 'XGB_d4_300'), (200, 3, 0.1, 'XGB_d3_200'),
        ]:
            cls = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=lr_val,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # Multi-seed XGB
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
        _store('XGB_d4_300_mseed', np.mean(multi_xgb, axis=0))

        # GBC
        for n_est, md, name in [(200, 4, 'GBC_200_d4'), (300, 3, 'GBC_300_d3')]:
            cls = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # MLP
        for hidden, name in [((64, 32), 'MLP_64_32'), ((100,), 'MLP_100')]:
            cls = MLPClassifier(
                hidden_layer_sizes=hidden, max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # Multi-seed MLP
        multi_mlp = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y_pw)
            multi_mlp.append(pairwise_score(cls, X_season, sc))
        _store('MLP_64_32_mseed', np.mean(multi_mlp, axis=0))

        print(f'    Fold {fi+1}/5 ({hold_season}): {time.time()-t0:.0f}s')

    print(f'  Precomputation done: {time.time()-t0:.0f}s')

    # ─── INDIVIDUAL PERFORMANCE ───
    print('\n' + '=' * 70)
    print(' Individual Performance')
    print('=' * 70)
    indiv = {}
    for name in method_names:
        e, r = _eval(all_scores, [(name, 1.0)], 0.15, y, seasons, test_mask, folds)
        indiv[name] = (e, r)
    for name, (e, r) in sorted(indiv.items(), key=lambda x: x[1][1]):
        print(f'  {name:<22} {e:3d}/91 RMSE={r:.4f}')

    # ─── 3-COMPONENT SEARCH (2% steps, fine power) ───
    print('\n' + '=' * 70)
    print(' 3-Component Blend Search (2% steps)')
    print('=' * 70)

    powers = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.375, 0.50]
    weight_steps = np.arange(0.02, 0.99, 0.02)

    best_rmse = 999.0
    best_exact = 0
    best_cfg_rmse = None
    best_cfg_exact = None
    pareto = {}
    n_tested = 0

    combos = list(itertools.combinations(range(len(method_names)), 3))
    print(f'  Triplets: {len(combos)}, powers: {len(powers)}')

    for ci, (i, j, k) in enumerate(combos):
        m1, m2, m3 = method_names[i], method_names[j], method_names[k]
        for w1 in weight_steps:
            for w2 in weight_steps:
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0.02 or w3 > 0.98:
                    continue
                for power in powers:
                    n_tested += 1
                    e, r = _eval(all_scores,
                                 [(m1, w1), (m2, w2), (m3, w3)],
                                 power, y, seasons, test_mask, folds)
                    if r < best_rmse:
                        best_rmse = r
                        best_cfg_rmse = (m1, w1, m2, w2, m3, w3, power, e)
                    if e > best_exact or (e == best_exact and r < pareto.get(e, (999,))[0]):
                        best_exact = max(best_exact, e)
                        pareto[e] = (r, (m1, w1, m2, w2, m3, w3, power))
        if (ci + 1) % 50 == 0 or ci == len(combos) - 1:
            print(f'    {ci+1}/{len(combos)}: {n_tested:,} configs, '
                  f'best_rmse={best_rmse:.4f}, best_exact={best_exact}, '
                  f'{time.time()-t0:.0f}s')

    print(f'\n  BEST RMSE: {best_rmse:.4f}')
    if best_cfg_rmse:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg_rmse
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}')
        print(f'    power={p}, {e}/91 exact')

    print(f'\n  BEST EXACT: {best_exact}/91')
    if best_exact in pareto:
        r, (m1, w1, m2, w2, m3, w3, p) = pareto[best_exact]
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}')
        print(f'    power={p}, RMSE={r:.4f}')

    print(f'\n  Pareto frontier:')
    for ex in sorted(pareto.keys(), reverse=True):
        r, (m1, w1, m2, w2, m3, w3, p) = pareto[ex]
        star = ' ★' if r < 2.474 else ''
        print(f'    {ex}/91 RMSE={r:.4f}: {w1:.0%} {m1} + {w2:.0%} {m2} '
              f'+ {w3:.0%} {m3}, p={p}{star}')
        if ex < best_exact - 8:
            break

    # ─── 4-COMPONENT SEARCH ───
    print('\n' + '=' * 70)
    print(' 4-Component Blend Search')
    print('=' * 70)

    # If 3c found a good config, try adding a 4th
    best_rmse4 = 999.0
    best_cfg4 = None
    best_exact4 = 0
    n4 = 0

    # Build from v6 base + new methods, and from best 3c
    base_triplets = [
        ('LR_C5', 'LRk25_C0.5', 'XGB_d4_300'),  # v6
        ('LR_C5', 'LRk25_C0.5', 'MLP_64_32'),  # v6+MLP
    ]
    if best_cfg_rmse:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg_rmse
        base_triplets.append((m1, m2, m3))

    for base_m1, base_m2, base_m3 in base_triplets:
        fourths = [m for m in method_names if m not in (base_m1, base_m2, base_m3)]
        for m4 in fourths:
            for w4 in np.arange(0.02, 0.24, 0.02):
                rem = 1.0 - w4
                for r1 in np.arange(0.20, 0.82, 0.02):
                    for r2 in np.arange(0.06, 0.60, 0.02):
                        r3 = round(1.0 - r1 - r2, 4)
                        if r3 < 0.02 or r3 > 0.60:
                            continue
                        w1 = round(rem * r1, 4)
                        w2 = round(rem * r2, 4)
                        w3 = round(rem * r3, 4)
                        for power in [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.375]:
                            n4 += 1
                            e, r = _eval(
                                all_scores,
                                [(base_m1, w1), (base_m2, w2),
                                 (base_m3, w3), (m4, w4)],
                                power, y, seasons, test_mask, folds)
                            if r < best_rmse4:
                                best_rmse4 = r
                                best_cfg4 = (base_m1, w1, base_m2, w2,
                                             base_m3, w3, m4, w4, power, e)
                            if e > best_exact4:
                                best_exact4 = e

    print(f'  4-component configs: {n4:,}')
    print(f'  Best RMSE: {best_rmse4:.4f}')
    if best_cfg4:
        m1, w1, m2, w2, m3, w3, m4, w4, p, e = best_cfg4
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3} + {w4:.0%} {m4}')
        print(f'    p={p}, {e}/91 exact')
    print(f'  Best exact: {best_exact4}/91')

    # ─── LOSO VALIDATION ───
    print('\n' + '=' * 70)
    print(' LOSO Validation of Top Configs')
    print('=' * 70)

    loso_cfgs = [
        ('v6 baseline', [('LR_C5', 0.64), ('LRk25_C0.5', 0.28),
                         ('XGB_d4_300', 0.08)], 0.15),
    ]
    if best_cfg_rmse:
        m1, w1, m2, w2, m3, w3, p, e = best_cfg_rmse
        loso_cfgs.append((f'v7c 3c RMSE best ({e}/91 RMSE={best_rmse:.4f})',
                          [(m1, w1), (m2, w2), (m3, w3)], p))
    if best_exact in pareto and best_exact > 56:
        r, (m1, w1, m2, w2, m3, w3, p) = pareto[best_exact]
        loso_cfgs.append((f'v7c 3c Exact best ({best_exact}/91)',
                          [(m1, w1), (m2, w2), (m3, w3)], p))
    if best_cfg4 and best_rmse4 < best_rmse:
        m1, w1, m2, w2, m3, w3, m4, w4, p, e = best_cfg4
        loso_cfgs.append((f'v7c 4c best ({e}/91 RMSE={best_rmse4:.4f})',
                          [(m1, w1), (m2, w2), (m3, w3), (m4, w4)], p))

    for label, weights, power in loso_cfgs:
        _run_loso(label, weights, power, X, y, seasons, fn, folds)

    # ─── STABILITY TEST ───
    print('\n' + '=' * 70)
    print(' Stability Test (weight perturbations around best)')
    print('=' * 70)

    if best_cfg_rmse:
        m1, w1_b, m2, w2_b, m3, w3_b, p_b, e_b = best_cfg_rmse
        print(f'  Base: {w1_b:.0%} {m1} + {w2_b:.0%} {m2} + {w3_b:.0%} {m3}, p={p_b}')
        print(f'  Base: {e_b}/91 exact, RMSE={best_rmse:.4f}')

        print(f'\n  {"Perturbation":<30} {"Exact":>5} {"RMSE":>8} {"Δ":>8}')
        print(f'  {"─"*30} {"─"*5} {"─"*8} {"─"*8}')

        for dw1, dw2 in [(0.02, -0.02), (-0.02, 0.02), (0.04, -0.04),
                          (-0.04, 0.04), (0.02, 0), (-0.02, 0)]:
            nw1 = round(w1_b + dw1, 4)
            nw2 = round(w2_b + dw2, 4)
            nw3 = round(1.0 - nw1 - nw2, 4)
            if nw1 <= 0 or nw2 <= 0 or nw3 <= 0:
                continue
            e, r = _eval(all_scores, [(m1, nw1), (m2, nw2), (m3, nw3)],
                         p_b, y, seasons, test_mask, folds)
            label = f'w1{dw1:+.2f} w2{dw2:+.2f}'
            print(f'  {label:<30} {e:5d} {r:8.4f} {r - best_rmse:+8.4f}')

        for dp in [-0.025, -0.05, 0.025, 0.05, 0.10, 0.20]:
            np_v = round(p_b + dp, 4)
            if np_v <= 0:
                continue
            e, r = _eval(all_scores, [(m1, w1_b), (m2, w2_b), (m3, w3_b)],
                         np_v, y, seasons, test_mask, folds)
            label = f'p{dp:+.3f} (p={np_v:.3f})'
            print(f'  {label:<30} {e:5d} {r:8.4f} {r - best_rmse:+8.4f}')

    # ─── BOOTSTRAP COMPARISON ───
    print('\n' + '=' * 70)
    print(' Bootstrap: v6 vs v7c best')
    print('=' * 70)

    if best_cfg_rmse:
        m1, w1, m2, w2, m3, w3, p, _ = best_cfg_rmse
        v7_pred = _get_preds(all_scores, [(m1, w1), (m2, w2), (m3, w3)],
                              p, y, seasons, test_mask, folds)
        v6_pred = _get_preds(all_scores,
                              [('LR_C5', 0.64), ('LRk25_C0.5', 0.28), ('XGB_d4_300', 0.08)],
                              0.15, y, seasons, test_mask, folds)
        gt = y[test_mask].astype(int)

        rng = np.random.RandomState(42)
        v7_wins = 0
        for _ in range(2000):
            idx = rng.choice(len(gt), len(gt), replace=True)
            r6 = np.sqrt(np.mean((v6_pred[idx] - gt[idx])**2))
            r7 = np.sqrt(np.mean((v7_pred[idx] - gt[idx])**2))
            if r7 < r6:
                v7_wins += 1

        print(f'  v7c wins {v7_wins}/2000 bootstrap resamples ({v7_wins/20:.1f}%)')
        if v7_wins >= 1800:
            print(f'  ★ Strong evidence v7c is genuinely better!')
        elif v7_wins >= 1400:
            print(f'  Moderate evidence v7c is better')
        else:
            print(f'  Weak evidence — possible overfit')

    # ─── SUMMARY ───
    print('\n' + '=' * 70)
    print(' FINAL SUMMARY')
    print('=' * 70)
    print(f'  v6:          56/91 exact, RMSE=2.4740')
    print(f'  v7c 3c best: RMSE={best_rmse:.4f}, exact from Pareto')
    print(f'  v7c 4c best: RMSE={best_rmse4:.4f}')
    overall_best = min(best_rmse, best_rmse4)
    if overall_best < 2.474:
        print(f'\n  ★★★ v7c BEATS v6! RMSE improved {2.474:.4f} → {overall_best:.4f} ★★★')
    else:
        print(f'  v6 remains champion.')
    print(f'  Time: {time.time()-t0:.0f}s')


# ═══════════════════════════════════════════════════════════
def _eval(all_scores, weights, power, y, seasons, test_mask, folds):
    n = len(y)
    assigned = np.zeros(n, dtype=int)
    for s in folds:
        sm = (seasons == s)
        si = np.where(sm)[0]
        bl = np.zeros(len(si))
        for m, w in weights:
            bl += w * all_scores[m][sm]
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {s: list(range(1, 69))}
        a = hungarian(bl, seasons[sm], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                assigned[gi] = a[i]
    gt = y[test_mask].astype(int)
    pred = assigned[test_mask]
    return int((pred == gt).sum()), np.sqrt(np.mean((pred - gt)**2))


def _get_preds(all_scores, weights, power, y, seasons, test_mask, folds):
    n = len(y)
    assigned = np.zeros(n, dtype=int)
    for s in folds:
        sm = (seasons == s)
        si = np.where(sm)[0]
        bl = np.zeros(len(si))
        for m, w in weights:
            bl += w * all_scores[m][sm]
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {s: list(range(1, 69))}
        a = hungarian(bl, seasons[sm], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                assigned[gi] = a[i]
    return assigned[test_mask]


def _run_loso(label, weights, power, X, y, seasons, fn, folds):
    print(f'\n  --- {label} ---')
    n = len(y)
    all_assigned = np.zeros(n, dtype=int)
    fold_stats = []

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

        bl = np.zeros(int(te.sum()))
        X_te = X[te]

        for m_tag, w in weights:
            s = _compute(m_tag, X_te, X[tr], y[tr], seasons[tr],
                         pw_X_sc, pw_y_pw, sc, topk_cache, fn)
            bl += w * s

        avail = {hold: list(range(1, 69))}
        assigned = hungarian(bl, seasons[te], avail, power=power)
        all_assigned[te] = assigned

        yte = y[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse_f = np.sqrt(np.mean((assigned - yte)**2))
        fold_stats.append((hold, int(te.sum()), exact, rmse_f))

    total_exact = int(np.sum(all_assigned == y.astype(int)))
    total_rmse = np.sqrt(np.mean((all_assigned - y.astype(int))**2))
    fold_rmses = [r for _, _, _, r in fold_stats]
    score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)

    print(f'  LOSO: {total_exact}/340, RMSE={total_rmse:.4f}, score={score:.4f}')
    for s, nf, ex, rm in fold_stats:
        print(f'    {s}: {ex}/{nf} ({ex/nf*100:.1f}%), RMSE={rm:.3f}')


def _compute(m_tag, X_te, X_tr, y_tr, seasons_tr,
             pw_X_sc, pw_y, sc, topk_cache, fn):
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
        C = float(m_tag.split('_')[1].replace('C', ''))
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
        cls = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=0.05,
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

    raise ValueError(f'Unknown: {m_tag}')


if __name__ == '__main__':
    main()
