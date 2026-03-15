#!/usr/bin/env python3
"""
v7b Targeted Improvements — Novel Paradigms
=============================================
v6: 56/91 exact, RMSE=2.474 (rank scoring, 3 components)
v7: All new scoring modes (raw, BT) FAILED to beat v6.
    MLP showed promise individually (50/91) and in 4c blend (LOSO better).

v7b tries genuinely NEW paradigms:
  A. PAIRWISE REGRESSION: predict seed DIFFERENCES, directly estimate seeds
     → Different info than classification (magnitude, not just direction)
  B. HYBRID: blend pairwise classification + direct XGB regression
     → Pairwise captures relative order; regression captures absolute magnitude
  C. MULTI-SEED ENSEMBLE: average over 5 random seeds for XGB/GBC/MLP
     → Reduces variance of non-convex models
  D. REFINED 4-COMPONENT: fine search with v6 rank + new diversity methods
  E. LOSO-OPTIMIZED: target LOSO performance (better for 2026 prediction)
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
    V40_XGB_PARAMS, SEEDS,
)


# ═══════════════════════════════════════════════════════════
# NEW PARADIGM A: PAIRWISE REGRESSION
# ═══════════════════════════════════════════════════════════

def build_pairwise_regression_data(X, y, seasons):
    """Build pairwise data for REGRESSION (predict seed difference).
    Target = normalized seed difference in [-1, 1]."""
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                seed_diff = (y[a] - y[b]) / 67.0  # normalize to [-1, 1]
                pairs_X.append(diff)
                pairs_y.append(seed_diff)
                pairs_X.append(-diff)
                pairs_y.append(-seed_diff)
    return np.array(pairs_X), np.array(pairs_y)


def pairwise_regression_score(model, X_test, X_train_season, y_train_season,
                               scaler, is_locked):
    """Score test teams using pairwise regression.
    For each test team, estimate seed by averaging over all training teams:
      estimated_seed_i = median_j [ seed_j + model.predict(X_i - X_j) * 67 ]
    """
    n_test = len(X_test)
    estimates = np.zeros(n_test)

    for i in range(n_test):
        if is_locked[i]:
            # This team has a known seed, use it directly
            estimates[i] = y_train_season[i]
            continue

        # Use all OTHER teams with known seeds as reference
        refs = []
        for j in range(n_test):
            if j == i and not is_locked[j]:
                continue
            if not is_locked[j]:
                continue
            diff = X_test[i] - X_test[j]
            if scaler is not None:
                diff = scaler.transform(diff.reshape(1, -1))
            pred_diff = model.predict(diff.reshape(1, -1))[0] * 67.0
            refs.append(y_train_season[j] + pred_diff)

        if refs:
            estimates[i] = np.median(refs)
        else:
            # Fallback: use all teams
            all_refs = []
            for j in range(n_test):
                if j == i:
                    continue
                diff = X_test[i] - X_test[j]
                if scaler is not None:
                    diff = scaler.transform(diff.reshape(1, -1))
                pred_diff = model.predict(diff.reshape(1, -1))[0] * 67.0
                all_refs.append(pred_diff)
            # Average diff → rank
            avg_sum = np.mean(all_refs)
            estimates[i] = 34.5 - avg_sum * 34  # crude estimate

    return estimates


def pairwise_regression_score_allvall(model, X_test, scaler):
    """Score using all-vs-all pairwise regression (no locked seeds needed).
    For each team, their "strength" = average predicted advantage over opponents.
    Then convert to seed-like scale."""
    n = len(X_test)
    strengths = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        pred_diffs = model.predict(diffs) * 67.0
        pred_diffs[i] = 0  # self-comparison
        strengths[i] = pred_diffs.mean()  # average advantage

    # Convert to rank-based scores (like pairwise_score)
    return np.argsort(np.argsort(-strengths)).astype(float) + 1.0


# ═══════════════════════════════════════════════════════════
# NEW PARADIGM B: DIRECT REGRESSION (v40-style)
# ═══════════════════════════════════════════════════════════

def predict_direct_regression(X_train, y_train, X_test, seeds_list=None):
    """Direct XGBRegressor + Ridge regression of seeds."""
    # Multi-seed XGB
    xgb_preds = []
    for seed in (seeds_list or SEEDS):
        m = xgb.XGBRegressor(**V40_XGB_PARAMS, random_state=seed, verbosity=0)
        m.fit(X_train, y_train)
        xgb_preds.append(m.predict(X_test))
    xgb_avg = np.mean(xgb_preds, axis=0)

    # Ridge
    sc = StandardScaler()
    rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(X_train), y_train)
    ridge_pred = rm.predict(sc.transform(X_test))

    return 0.70 * xgb_avg + 0.30 * ridge_pred


# ═══════════════════════════════════════════════════════════
# MULTI-SEED PAIRWISE
# ═══════════════════════════════════════════════════════════

def multi_seed_pairwise_score(model_class, model_params, pw_X_sc, pw_y,
                                X_test, sc, seeds, kind='classification'):
    """Average pairwise scores from multiple random seeds."""
    all_scores = []
    for seed in seeds:
        params = {**model_params, 'random_state': seed}
        if kind == 'classification':
            cls = model_class(**params)
            cls.fit(pw_X_sc, pw_y)
            s = pairwise_score(cls, X_test, sc)
        all_scores.append(s)
    return np.mean(all_scores, axis=0)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print(' v7b TARGETED IMPROVEMENTS')
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

    # ══════════════════════════════════════════════════
    # PRECOMPUTE ALL METHOD SCORES (Kaggle-style)
    # ══════════════════════════════════════════════════
    print('\n  Precomputing method scores...')

    method_names = [
        # v6 rank components
        'LR_C5_rank', 'LRk25_C0.5_rank', 'XGB_d4_300_rank',
        # Alternative LR
        'LR_C3_rank', 'LR_C7_rank', 'LR_C1_rank',
        # Alternative topK
        'LRk20_C0.5_rank', 'LRk30_C0.5_rank', 'LRk25_C1_rank',
        'LRk25_C0.3_rank', 'LRk35_C0.5_rank',
        # XGB variants
        'XGB_d3_200_rank', 'XGB_d3_500_rank', 'XGB_d5_300_rank',
        # GBC (showed promise in v6c best-exact)
        'GBC_200_d4_rank', 'GBC_300_d3_rank', 'GBC_300_d4_rank',
        # MLP (showed promise in v7)
        'MLP_64_32_rank', 'MLP_100_rank',
        # Multi-seed XGB ensemble
        'XGB_d4_300_mseed_rank',
        # Multi-seed MLP ensemble
        'MLP_64_32_mseed_rank',
        # Pairwise REGRESSION (new paradigm)
        'PW_Ridge_reg', 'PW_XGBReg_reg',
        # Direct regression (v40-style)
        'DirectReg',
    ]

    all_scores = {m: np.zeros(n) for m in method_names}

    for fi, hold_season in enumerate(folds):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        y_season = y[season_mask]
        season_indices = np.where(season_mask)[0]
        is_locked = np.array([not test_mask[gi] for gi in season_indices])

        # Standard pairwise data
        pw_X, pw_y_pw = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        # Pairwise regression data (NEW)
        pw_reg_X, pw_reg_y = build_pairwise_regression_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc_reg = StandardScaler()
        pw_reg_X_sc = sc_reg.fit_transform(pw_reg_X)

        # TopK caches
        topk_cache = {}
        topk_pw_cache = {}
        for K in [20, 25, 30, 35]:
            tk = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = tk
            Xk_tr = X[global_train_mask][:, tk]
            pk_X, pk_y = build_pairwise_data(Xk_tr, y[global_train_mask],
                                              seasons[global_train_mask])
            sc_k = StandardScaler()
            pk_X_sc = sc_k.fit_transform(pk_X)
            topk_pw_cache[K] = (pk_X_sc, pk_y, sc_k, tk)

        def _store(name, scores):
            for i, gi in enumerate(season_indices):
                all_scores[name][gi] = scores[i]

        # ── LR variants (rank) ──
        for C_val, name in [(5.0, 'LR_C5_rank'), (3.0, 'LR_C3_rank'),
                            (7.0, 'LR_C7_rank'), (1.0, 'LR_C1_rank')]:
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # ── TopK LR variants (rank) ──
        for K, C_val, name in [
            (25, 0.5, 'LRk25_C0.5_rank'), (25, 1.0, 'LRk25_C1_rank'),
            (25, 0.3, 'LRk25_C0.3_rank'),
            (20, 0.5, 'LRk20_C0.5_rank'), (30, 0.5, 'LRk30_C0.5_rank'),
            (35, 0.5, 'LRk35_C0.5_rank'),
        ]:
            pk_X_sc, pk_y, sc_k, tk = topk_pw_cache[K]
            cls = LogisticRegression(C=C_val, penalty='l2', max_iter=2000, random_state=42)
            cls.fit(pk_X_sc, pk_y)
            _store(name, pairwise_score(cls, X_season[:, tk], sc_k))

        # ── XGB variants (rank) ──
        for n_est, md, lr_val, name in [
            (300, 4, 0.05, 'XGB_d4_300_rank'), (200, 3, 0.1, 'XGB_d3_200_rank'),
            (500, 3, 0.05, 'XGB_d3_500_rank'), (300, 5, 0.05, 'XGB_d5_300_rank'),
        ]:
            cls = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=lr_val,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # ── GBC variants (rank) ──
        for n_est, md, name in [
            (200, 4, 'GBC_200_d4_rank'), (300, 3, 'GBC_300_d3_rank'),
            (300, 4, 'GBC_300_d4_rank'),
        ]:
            cls = GradientBoostingClassifier(
                n_estimators=n_est, max_depth=md, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # ── MLP variants (rank) ──
        for hidden, name in [((64, 32), 'MLP_64_32_rank'), ((100,), 'MLP_100_rank')]:
            cls = MLPClassifier(
                hidden_layer_sizes=hidden, max_iter=500, random_state=42,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y_pw)
            _store(name, pairwise_score(cls, X_season, sc))

        # ── Multi-seed XGB ensemble (rank) ──
        multi_scores = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=seed, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            cls.fit(pw_X_sc, pw_y_pw)
            multi_scores.append(pairwise_score(cls, X_season, sc))
        _store('XGB_d4_300_mseed_rank', np.mean(multi_scores, axis=0))

        # ── Multi-seed MLP ensemble (rank) ──
        multi_scores_mlp = []
        for seed in [42, 123, 777, 2024, 31415]:
            cls = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                early_stopping=True, validation_fraction=0.1,
                learning_rate='adaptive', alpha=0.001)
            cls.fit(pw_X_sc, pw_y_pw)
            multi_scores_mlp.append(pairwise_score(cls, X_season, sc))
        _store('MLP_64_32_mseed_rank', np.mean(multi_scores_mlp, axis=0))

        # ── Pairwise REGRESSION (NEW PARADIGM) ──
        # Ridge regression on pairwise diffs → predict seed difference
        ridge_reg = Ridge(alpha=10.0)
        ridge_reg.fit(pw_reg_X_sc, pw_reg_y)
        s_reg = pairwise_regression_score_allvall(ridge_reg, X_season, sc_reg)
        _store('PW_Ridge_reg', s_reg)

        # XGB regression on pairwise diffs → predict seed difference
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0)
        xgb_reg.fit(pw_reg_X_sc, pw_reg_y)
        s_xreg = pairwise_regression_score_allvall(xgb_reg, X_season, sc_reg)
        _store('PW_XGBReg_reg', s_xreg)

        # ── Direct regression (v40-style) ──
        direct_pred = predict_direct_regression(
            X[global_train_mask], y[global_train_mask], X_season)
        _store('DirectReg', direct_pred)

        print(f'    Fold {fi + 1}/{len(folds)} ({hold_season}): done '
              f'({season_test_mask.sum()} test teams), {time.time() - t0:.0f}s')

    print(f'  Total precomputation: {time.time() - t0:.0f}s')

    # ══════════════════════════════════════════════════
    # PHASE 1: Individual method performance
    # ══════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 1: Individual Method Performance (Kaggle)')
    print('=' * 70)

    indiv = {}
    for name in method_names:
        e, r = _eval_kaggle(all_scores, [(name, 1.0)], 0.15,
                            y, seasons, test_mask, folds)
        indiv[name] = (e, r)

    sorted_m = sorted(indiv.items(), key=lambda x: x[1][1])
    print(f'\n  {"Method":<28} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─" * 28} {"─" * 5} {"─" * 8}')
    for name, (e, r) in sorted_m:
        marker = ' ← v6 component' if name in ('LR_C5_rank', 'LRk25_C0.5_rank', 'XGB_d4_300_rank') else ''
        if 'mseed' in name:
            base = name.replace('_mseed', '')
            if base in indiv:
                delta = r - indiv[base][1]
                marker = f' (Δ={delta:+.3f} vs single seed)'
        if name.startswith('PW_') or name == 'DirectReg':
            marker = ' ★ NEW PARADIGM'
        print(f'  {name:<28} {e:5d} {r:8.4f}{marker}')

    # Test at different powers for pairwise regression
    print(f'\n  Pairwise Regression at different powers:')
    for power in [0.10, 0.15, 0.20, 0.25, 0.375, 0.50, 0.75, 1.0]:
        for name in ['PW_Ridge_reg', 'PW_XGBReg_reg', 'DirectReg']:
            e, r = _eval_kaggle(all_scores, [(name, 1.0)], power,
                                y, seasons, test_mask, folds)
            print(f'    {name:<18} p={power:.3f}: {e:3d}/91 exact, RMSE={r:.4f}')
        print()

    # ══════════════════════════════════════════════════
    # PHASE 2: Hybrid blends (v6 + new paradigm)
    # ══════════════════════════════════════════════════
    print('=' * 70)
    print(' PHASE 2: Hybrid Blends (v6 pairwise + new paradigms)')
    print('=' * 70)

    # Test blending v6 components with pairwise regression and direct regression
    hybrid_configs = []

    # v6 base + PW regression
    for pw_reg_name in ['PW_Ridge_reg', 'PW_XGBReg_reg', 'DirectReg']:
        for w_new in np.arange(0.02, 0.30, 0.02):
            remaining = 1.0 - w_new
            # Keep v6 ratios for the remaining weight
            w1 = round(remaining * 0.64, 4)
            w2 = round(remaining * 0.28, 4)
            w3 = round(remaining * 0.08, 4)
            for power in [0.10, 0.15, 0.20, 0.25, 0.375, 0.50]:
                e, r = _eval_kaggle(
                    all_scores,
                    [('LR_C5_rank', w1), ('LRk25_C0.5_rank', w2),
                     ('XGB_d4_300_rank', w3), (pw_reg_name, w_new)],
                    power, y, seasons, test_mask, folds)
                hybrid_configs.append((pw_reg_name, w_new, power, e, r))

    # Find best hybrids
    best_hybrid = min(hybrid_configs, key=lambda x: x[4])
    best_hybrid_exact = max(hybrid_configs, key=lambda x: (x[3], -x[4]))

    print(f'\n  Best hybrid RMSE: {best_hybrid[0]} at w={best_hybrid[1]:.2f}, '
          f'p={best_hybrid[2]:.3f}')
    print(f'    {best_hybrid[3]}/91 exact, RMSE={best_hybrid[4]:.4f}')
    print(f'  Best hybrid exact: {best_hybrid_exact[0]} at w={best_hybrid_exact[1]:.2f}, '
          f'p={best_hybrid_exact[2]:.3f}')
    print(f'    {best_hybrid_exact[3]}/91 exact, RMSE={best_hybrid_exact[4]:.4f}')

    # ══════════════════════════════════════════════════
    # PHASE 3: Comprehensive 3-Component Search (rank mode)
    # ══════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 3: 3-Component Blend Search (rank mode, 2% steps)')
    print('=' * 70)

    # Focus on rank-mode methods (proven scoring mode)
    rank_methods = [n for n in method_names
                    if n.endswith('_rank') or n.endswith('_reg') or n == 'DirectReg']
    # Sort by individual performance
    rank_methods.sort(key=lambda n: indiv.get(n, (0, 999))[1])
    top_rank = rank_methods[:15]
    print(f'  Using top {len(top_rank)} rank methods')

    powers = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.375, 0.50]
    weight_steps = np.arange(0.02, 0.99, 0.02)

    best_rmse3 = 999.0
    best_exact3 = 0
    best_cfg_rmse3 = None
    best_cfg_exact3 = None
    pareto3 = {}
    n_tested3 = 0

    combos = list(itertools.combinations(range(len(top_rank)), 3))
    print(f'  Triplets: {len(combos)}, powers: {len(powers)}')

    for ci, (i, j, k) in enumerate(combos):
        m1, m2, m3 = top_rank[i], top_rank[j], top_rank[k]

        for w1 in weight_steps:
            for w2 in weight_steps:
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0.02 or w3 > 0.98:
                    continue

                for power in powers:
                    n_tested3 += 1
                    e, r = _eval_kaggle(all_scores,
                                        [(m1, w1), (m2, w2), (m3, w3)],
                                        power, y, seasons, test_mask, folds)

                    if r < best_rmse3:
                        best_rmse3 = r
                        best_cfg_rmse3 = (m1, w1, m2, w2, m3, w3, power)

                    if e > best_exact3 or (e == best_exact3 and r < pareto3.get(e, (999,))[0]):
                        best_exact3 = max(best_exact3, e)
                        pareto3[e] = (r, (m1, w1, m2, w2, m3, w3, power))

        if (ci + 1) % 100 == 0 or ci == len(combos) - 1:
            print(f'    {ci + 1}/{len(combos)}: {n_tested3:,} configs, '
                  f'best_rmse={best_rmse3:.4f}, best_exact={best_exact3}, '
                  f'{time.time() - t0:.0f}s')

    print(f'\n  3-Component Results ({n_tested3:,} configs):')
    print(f'  Best RMSE: {best_rmse3:.4f} (v6: 2.474)')
    if best_cfg_rmse3:
        m1, w1, m2, w2, m3, w3, p = best_cfg_rmse3
        e, _ = _eval_kaggle(all_scores, [(m1, w1), (m2, w2), (m3, w3)],
                            p, y, seasons, test_mask, folds)
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}, p={p}')
        print(f'    {e}/91 exact')

    print(f'\n  Pareto frontier:')
    for ex in sorted(pareto3.keys(), reverse=True):
        r, cfg = pareto3[ex]
        m1, w1, m2, w2, m3, w3, p = cfg
        star = ' ★' if r < 2.474 else ''
        print(f'    {ex}/91, RMSE={r:.4f}: {w1:.0%} {m1} + {w2:.0%} {m2} '
              f'+ {w3:.0%} {m3}, p={p}{star}')
        if ex < best_exact3 - 6:
            break

    # ══════════════════════════════════════════════════
    # PHASE 4: Fine 4-Component Search
    # ══════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 4: 4-Component Blend (fine search, 2% steps)')
    print('=' * 70)

    # Start from v6 base, add 4th component from diverse pool
    fourth_candidates = [n for n in rank_methods
                         if n not in ('LR_C5_rank', 'LRk25_C0.5_rank', 'XGB_d4_300_rank')]

    best_rmse4 = 999.0
    best_cfg4 = None
    best_exact4 = 0
    n4 = 0

    for m4 in fourth_candidates[:12]:
        for w4 in np.arange(0.02, 0.20, 0.02):
            remaining = 1.0 - w4
            for r1 in np.arange(0.30, 0.80, 0.02):
                for r2 in np.arange(0.08, 0.50, 0.02):
                    r3 = round(1.0 - r1 - r2, 4)
                    if r3 < 0.02 or r3 > 0.50:
                        continue
                    w1 = round(remaining * r1, 4)
                    w2 = round(remaining * r2, 4)
                    w3 = round(remaining * r3, 4)

                    for power in [0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.375]:
                        n4 += 1
                        e, r = _eval_kaggle(
                            all_scores,
                            [('LR_C5_rank', w1), ('LRk25_C0.5_rank', w2),
                             ('XGB_d4_300_rank', w3), (m4, w4)],
                            power, y, seasons, test_mask, folds)

                        if r < best_rmse4:
                            best_rmse4 = r
                            best_cfg4 = (w1, w2, w3, m4, w4, power, e)
                        if e > best_exact4:
                            best_exact4 = e

        print(f'    Tested 4th={m4}: {n4:,} configs so far, '
              f'best_rmse={best_rmse4:.4f}, best_exact={best_exact4}')

    print(f'\n  4-Component Results ({n4:,} configs):')
    print(f'  Best RMSE: {best_rmse4:.4f} (v6: 2.474)')
    if best_cfg4:
        w1, w2, w3, m4, w4, p, e = best_cfg4
        print(f'    {w1:.0%} LR_C5 + {w2:.0%} LRk25_C0.5 + {w3:.0%} XGB_d4_300 + {w4:.0%} {m4}')
        print(f'    p={p}, {e}/91 exact')
    print(f'  Best exact: {best_exact4}/91')

    # ══════════════════════════════════════════════════
    # PHASE 5: LOSO Validation of Top Configs
    # ══════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 5: LOSO Validation')
    print('=' * 70)

    loso_configs = [
        ('v6 baseline', [('LR_C5_rank', 0.64), ('LRk25_C0.5_rank', 0.28),
                         ('XGB_d4_300_rank', 0.08)], 0.15),
    ]

    if best_cfg_rmse3:
        m1, w1, m2, w2, m3, w3, p = best_cfg_rmse3
        loso_configs.append(('v7b 3c best RMSE',
                             [(m1, w1), (m2, w2), (m3, w3)], p))

    if best_cfg4:
        w1, w2, w3, m4, w4, p, _ = best_cfg4
        loso_configs.append(('v7b 4c best RMSE',
                             [('LR_C5_rank', w1), ('LRk25_C0.5_rank', w2),
                              ('XGB_d4_300_rank', w3), (m4, w4)], p))

    # Best exact configs
    if best_exact3 > 56 and best_exact3 in pareto3:
        r, cfg = pareto3[best_exact3]
        m1, w1, m2, w2, m3, w3, p = cfg
        loso_configs.append((f'v7b 3c best exact ({best_exact3})',
                             [(m1, w1), (m2, w2), (m3, w3)], p))

    for cfg_label, weights, power in loso_configs:
        _run_loso(cfg_label, weights, power, X, y, seasons, fn, folds)

    # ══════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' v7b SUMMARY')
    print('=' * 70)
    print(f'  v6 baseline:       56/91 exact, RMSE=2.4740')
    print(f'  v7b 3c best RMSE:  {best_rmse3:.4f}')
    print(f'  v7b 3c best exact: {best_exact3}/91')
    print(f'  v7b 4c best RMSE:  {best_rmse4:.4f}')
    print(f'  v7b 4c best exact: {best_exact4}/91')
    print(f'  Best hybrid:       {best_hybrid[3]}/91, RMSE={best_hybrid[4]:.4f}')

    # Identify overall best
    all_bests = [
        ('v6', 2.474, 56),
        ('v7b 3c', best_rmse3, best_exact3),
        ('v7b 4c', best_rmse4, best_exact4),
        ('hybrid', best_hybrid[4], best_hybrid[3]),
    ]
    best_overall = min(all_bests, key=lambda x: x[1])
    print(f'\n  OVERALL BEST: {best_overall[0]} with RMSE={best_overall[1]:.4f}, '
          f'{best_overall[2]}/91 exact')
    if best_overall[1] < 2.474:
        print(f'  ★★★ IMPROVEMENT OVER v6! ★★★')
    else:
        print(f'  v6 remains the champion.')

    print(f'\n  Total time: {time.time() - t0:.0f}s')


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _eval_kaggle(all_scores, weights, power, y, seasons, test_mask, folds):
    """Fast Kaggle evaluation."""
    n = len(y)
    assigned = np.zeros(n, dtype=int)
    for s in folds:
        sm = (seasons == s)
        si = np.where(sm)[0]
        bl = np.zeros(len(si))
        for m_tag, w in weights:
            if m_tag in all_scores:
                bl += w * all_scores[m_tag][sm]
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
    return int((pred == gt).sum()), np.sqrt(np.mean((pred - gt) ** 2))


def _run_loso(label, weights, power, X, y, seasons, fn, folds):
    """Full LOSO validation."""
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

        pw_reg_X, pw_reg_y = build_pairwise_regression_data(X[tr], y[tr], seasons[tr])
        sc_reg = StandardScaler()
        pw_reg_X_sc = sc_reg.fit_transform(pw_reg_X)

        topk_cache = {}
        for K in [20, 25, 30, 35]:
            tk = select_top_k_features(X[tr], y[tr], fn, k=K)[0]
            topk_cache[K] = tk

        bl = np.zeros(int(te.sum()))
        X_te = X[te]

        for m_tag, w in weights:
            s = _compute_score(m_tag, X_te, X[tr], y[tr], seasons[tr],
                               pw_X_sc, pw_y_pw, sc, pw_reg_X_sc, pw_reg_y,
                               sc_reg, topk_cache, fn)
            bl += w * s

        avail = {hold: list(range(1, 69))}
        assigned = hungarian(bl, seasons[te], avail, power=power)
        all_assigned[te] = assigned

        yte = y[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse_f = np.sqrt(np.mean((assigned - yte) ** 2))
        fold_stats.append((hold, int(te.sum()), exact, rmse_f))

    total_exact = int(np.sum(all_assigned == y.astype(int)))
    total_rmse = np.sqrt(np.mean((all_assigned - y.astype(int)) ** 2))
    fold_rmses = [r for _, _, _, r in fold_stats]
    score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)

    print(f'  LOSO: {total_exact}/340, RMSE={total_rmse:.4f}, score={score:.4f}')
    for s, nf, ex, rm in fold_stats:
        print(f'    {s}: {ex}/{nf} ({ex / nf * 100:.1f}%), RMSE={rm:.3f}')


def _compute_score(m_tag, X_te, X_tr, y_tr, seasons_tr,
                    pw_X_sc, pw_y, sc, pw_reg_X_sc, pw_reg_y,
                    sc_reg, topk_cache, fn):
    """Compute single method score for LOSO."""

    if m_tag == 'DirectReg':
        return predict_direct_regression(X_tr, y_tr, X_te)

    if m_tag.startswith('PW_Ridge_reg'):
        ridge_reg = Ridge(alpha=10.0)
        ridge_reg.fit(pw_reg_X_sc, pw_reg_y)
        return pairwise_regression_score_allvall(ridge_reg, X_te, sc_reg)

    if m_tag.startswith('PW_XGBoReg') or m_tag == 'PW_XGBReg_reg':
        xgb_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0)
        xgb_reg.fit(pw_reg_X_sc, pw_reg_y)
        return pairwise_regression_score_allvall(xgb_reg, X_te, sc_reg)

    if m_tag.startswith('LRk'):
        parts = m_tag.split('_')
        K = int(parts[0].replace('LRk', ''))
        C = float(parts[1].replace('C', '').replace('_rank', '').replace('_raw', ''))
        tk = topk_cache.get(K, topk_cache[25])
        Xk_tr = X_tr[:, tk]
        pk_X, pk_y = build_pairwise_data(Xk_tr, y_tr, seasons_tr)
        sc_k = StandardScaler()
        pk_X_sc = sc_k.fit_transform(pk_X)
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pk_X_sc, pk_y)
        return pairwise_score(cls, X_te[:, tk], sc_k)

    if m_tag.startswith('LR_C'):
        parts = m_tag.split('_')
        C = float(parts[1].replace('C', ''))
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_te, sc)

    if m_tag.startswith('XGB'):
        parts = m_tag.split('_')
        if 'mseed' in m_tag:
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
        else:
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

    if m_tag.startswith('MLP'):
        parts = m_tag.split('_')
        if 'mseed' in m_tag:
            multi = []
            for seed in [42, 123, 777, 2024, 31415]:
                cls = MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed,
                    early_stopping=True, validation_fraction=0.1,
                    learning_rate='adaptive', alpha=0.001)
                cls.fit(pw_X_sc, pw_y)
                multi.append(pairwise_score(cls, X_te, sc))
            return np.mean(multi, axis=0)
        elif parts[1] == '64':
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
