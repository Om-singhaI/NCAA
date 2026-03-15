#!/usr/bin/env python3
"""
v7 Deep Exploration — Genuine New Directions
=============================================
v6: 56/91, RMSE=2.474 (validated not overfit)

New ideas NOT explored in v6:
  1. RAW SCORE BLENDING: blend raw win-probability sums instead of ranks
     → preserves magnitude confidence (biggest potential unlock)
  2. BRADLEY-TERRY scoring: iterative opponent-strength-aware scoring
  3. NEW METHODS: GBC, MLP for pairwise (more diversity)
  4. AUGMENTED PAIRWISE FEATURES: [diff, |diff|] doubles feature space
  5. 4-COMPONENT BLENDS (v6 used 3)
  6. LOSO + Kaggle dual evaluation (avoid overfitting)
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
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
    build_pairwise_data, hungarian,
)


# ═══════════════════════════════════════════════════════════
# ENHANCED SCORING FUNCTIONS
# ═══════════════════════════════════════════════════════════

def pairwise_score_rank(model, X_test, scaler=None):
    """Original: rank-based scoring (1 to N)."""
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


def pairwise_score_raw(model, X_test, scaler=None):
    """NEW: raw win-probability sum, linearly scaled to [1, 68].
    Preserves magnitude information (confidence gaps between teams)."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    # Linear rescale to [1, 68] (higher win prob → lower seed number)
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scaled = 1.0 + 67.0 * (smax - scores) / (smax - smin)
    else:
        scaled = np.full(n, 34.5)
    return scaled


def pairwise_score_bt(model, X_test, scaler=None, n_iters=20):
    """NEW: Bradley-Terry iterative scoring.
    Accounts for opponent strength — beating strong teams counts more."""
    n = len(X_test)
    # Build full pairwise probability matrix
    P = np.zeros((n, n))
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        P[i] = model.predict_proba(diffs)[:, 1]
        P[i, i] = 0.5

    # Bradley-Terry MLE via iterative algorithm
    theta = np.ones(n)
    for _ in range(n_iters):
        new_theta = np.ones(n)
        for i in range(n):
            numer = 0.0
            denom = 0.0
            for j in range(n):
                if j == i:
                    continue
                numer += P[i, j]
                denom += (P[i, j] + P[j, i]) / (theta[i] + theta[j])
            new_theta[i] = numer / (denom + 1e-10)
        # Normalize
        new_theta /= new_theta.sum() / n
        theta = new_theta

    # Convert to seed-like scale (higher theta → lower seed)
    smin, smax = theta.min(), theta.max()
    if smax > smin:
        scaled = 1.0 + 67.0 * (smax - theta) / (smax - smin)
    else:
        scaled = np.full(n, 34.5)
    return scaled


def build_pairwise_data_aug(X, y, seasons):
    """AUGMENTED: pairwise features = [diff, |diff|].
    Captures that large magnitude differences are predictive."""
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                aug = np.concatenate([diff, np.abs(diff)])
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(aug)
                pairs_y.append(target)
                pairs_X.append(np.concatenate([-diff, np.abs(diff)]))
                pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)


def pairwise_score_aug(model, X_test, scaler, mode='rank'):
    """Score using augmented features [diff, |diff|]."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        aug_diffs = np.concatenate([diffs, np.abs(diffs)], axis=1)
        if scaler is not None:
            aug_diffs = scaler.transform(aug_diffs)
        probs = model.predict_proba(aug_diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    if mode == 'rank':
        return np.argsort(np.argsort(-scores)).astype(float) + 1.0
    else:
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return 1.0 + 67.0 * (smax - scores) / (smax - smin)
        return np.full(n, 34.5)


# ═══════════════════════════════════════════════════════════
# MAIN EXPLORATION
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print(' v7 DEEP EXPLORATION — New Directions')
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

    # ──────────────────────────────────────────────────
    # METHOD DEFINITIONS
    # ──────────────────────────────────────────────────
    # Each method: (name, type, params, feature_mode, score_mode)
    # feature_mode: 'full', 'topK_25', 'topK_20', 'topK_30', 'topK_15', 'topK_35', 'aug'
    # score_mode: 'rank', 'raw', 'bt'

    method_defs = [
        # ── v6 components in RANK mode (baseline) ──
        ('LR_C5_rank',       'lr',  {'C': 5.0},  'full', 'rank'),
        ('LRk25_C0.5_rank',  'lr',  {'C': 0.5},  'topK_25', 'rank'),
        ('XGB_d4_300_rank',  'xgb', {'n': 300, 'md': 4, 'lr': 0.05}, 'full', 'rank'),

        # ── v6 components in RAW mode (KEY INNOVATION) ──
        ('LR_C5_raw',        'lr',  {'C': 5.0},  'full', 'raw'),
        ('LRk25_C0.5_raw',   'lr',  {'C': 0.5},  'topK_25', 'raw'),
        ('XGB_d4_300_raw',   'xgb', {'n': 300, 'md': 4, 'lr': 0.05}, 'full', 'raw'),

        # ── v6 components in BRADLEY-TERRY mode ──
        ('LR_C5_bt',         'lr',  {'C': 5.0},  'full', 'bt'),
        ('LRk25_C0.5_bt',    'lr',  {'C': 0.5},  'topK_25', 'bt'),

        # ── Additional LR variants (raw) ──
        ('LR_C3_raw',        'lr',  {'C': 3.0},  'full', 'raw'),
        ('LR_C7_raw',        'lr',  {'C': 7.0},  'full', 'raw'),
        ('LR_C1_raw',        'lr',  {'C': 1.0},  'full', 'raw'),
        ('LR_C10_raw',       'lr',  {'C': 10.0}, 'full', 'raw'),
        ('LR_C0.5_raw',      'lr',  {'C': 0.5},  'full', 'raw'),

        # ── Additional topK (raw) ──
        ('LRk20_C0.5_raw',   'lr',  {'C': 0.5},  'topK_20', 'raw'),
        ('LRk30_C0.5_raw',   'lr',  {'C': 0.5},  'topK_30', 'raw'),
        ('LRk15_C0.5_raw',   'lr',  {'C': 0.5},  'topK_15', 'raw'),
        ('LRk35_C0.5_raw',   'lr',  {'C': 0.5},  'topK_35', 'raw'),
        ('LRk25_C0.3_raw',   'lr',  {'C': 0.3},  'topK_25', 'raw'),
        ('LRk25_C1_raw',     'lr',  {'C': 1.0},  'topK_25', 'raw'),

        # ── GBC (NEW) ──
        ('GBC_200_d4_rank',  'gbc', {'n': 200, 'md': 4}, 'full', 'rank'),
        ('GBC_200_d4_raw',   'gbc', {'n': 200, 'md': 4}, 'full', 'raw'),
        ('GBC_300_d3_raw',   'gbc', {'n': 300, 'md': 3}, 'full', 'raw'),
        ('GBC_300_d4_raw',   'gbc', {'n': 300, 'md': 4}, 'full', 'raw'),

        # ── MLP (NEW) ──
        ('MLP_100_raw',  'mlp', {'hidden': (100,)}, 'full', 'raw'),
        ('MLP_64_32_raw','mlp', {'hidden': (64, 32)}, 'full', 'raw'),
        ('MLP_100_rank', 'mlp', {'hidden': (100,)}, 'full', 'rank'),

        # ── XGB variants (raw) ──
        ('XGB_d4_300_raw2',  'xgb', {'n': 300, 'md': 4, 'lr': 0.05}, 'full', 'raw'),
        ('XGB_d3_500_raw',   'xgb', {'n': 500, 'md': 3, 'lr': 0.05}, 'full', 'raw'),
        ('XGB_d5_200_raw',   'xgb', {'n': 200, 'md': 5, 'lr': 0.05}, 'full', 'raw'),

        # ── AUGMENTED features [diff, |diff|] ──
        ('LR_C5_aug_rank',   'lr_aug', {'C': 5.0}, 'full', 'rank'),
        ('LR_C5_aug_raw',    'lr_aug', {'C': 5.0}, 'full', 'raw'),
        ('LR_C3_aug_raw',    'lr_aug', {'C': 3.0}, 'full', 'raw'),
    ]

    print(f'  Methods to evaluate: {len(method_defs)}')
    print(f'  Innovations: raw scoring, Bradley-Terry, GBC, MLP, augmented features')

    # ──────────────────────────────────────────────────
    # PRECOMPUTE ALL SCORES (Kaggle-style: lock train seeds)
    # ──────────────────────────────────────────────────
    print('\n  Precomputing all method scores...')
    all_scores = {name: np.zeros(n) for name, _, _, _, _ in method_defs}

    for fi, hold_season in enumerate(folds):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        season_indices = np.where(season_mask)[0]

        # Build pairwise data (standard)
        pw_X, pw_y = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        # Build augmented pairwise data
        pw_X_aug, pw_y_aug = build_pairwise_data_aug(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc_aug = StandardScaler()
        pw_X_aug_sc = sc_aug.fit_transform(pw_X_aug)

        # Top-K caches
        topk_cache = {}
        topk_pw_cache = {}
        topk_sc_cache = {}
        for K in [15, 20, 25, 30, 35]:
            tk_idx = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = tk_idx
            X_k_tr = X[global_train_mask][:, tk_idx]
            pk_X, pk_y = build_pairwise_data(X_k_tr, y[global_train_mask],
                                              seasons[global_train_mask])
            sc_k = StandardScaler()
            pk_X_sc = sc_k.fit_transform(pk_X)
            topk_pw_cache[K] = (pk_X_sc, pk_y)
            topk_sc_cache[K] = sc_k

        # Trained model cache (avoid re-training same model for different score modes)
        model_cache = {}

        for m_name, m_type, m_params, feat_mode, score_mode in method_defs:
            try:
                # Determine feature subset
                if feat_mode == 'full':
                    X_te = X_season
                elif feat_mode.startswith('topK_'):
                    K = int(feat_mode.split('_')[1])
                    X_te = X_season[:, topk_cache[K]]
                else:
                    X_te = X_season

                # Build cache key for model (same model can serve rank/raw/bt)
                if m_type == 'lr':
                    cache_key = f"lr_C{m_params['C']}_{feat_mode}"
                elif m_type == 'lr_aug':
                    cache_key = f"lr_aug_C{m_params['C']}"
                elif m_type == 'xgb':
                    cache_key = f"xgb_n{m_params['n']}_d{m_params['md']}_lr{m_params['lr']}_{feat_mode}"
                elif m_type == 'gbc':
                    cache_key = f"gbc_n{m_params['n']}_d{m_params['md']}_{feat_mode}"
                elif m_type == 'mlp':
                    cache_key = f"mlp_h{m_params['hidden']}_{feat_mode}"
                else:
                    cache_key = m_name

                if cache_key not in model_cache:
                    # Train the model
                    if m_type == 'lr':
                        cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                                  max_iter=2000, random_state=42)
                        if feat_mode.startswith('topK_'):
                            K = int(feat_mode.split('_')[1])
                            cls.fit(topk_pw_cache[K][0], topk_pw_cache[K][1])
                            model_cache[cache_key] = (cls, topk_sc_cache[K])
                        else:
                            cls.fit(pw_X_sc, pw_y)
                            model_cache[cache_key] = (cls, sc)

                    elif m_type == 'lr_aug':
                        cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                                  max_iter=2000, random_state=42)
                        cls.fit(pw_X_aug_sc, pw_y_aug)
                        model_cache[cache_key] = (cls, sc_aug)

                    elif m_type == 'xgb':
                        cls = xgb.XGBClassifier(
                            n_estimators=m_params['n'], max_depth=m_params['md'],
                            learning_rate=m_params['lr'],
                            subsample=0.8, colsample_bytree=0.8,
                            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                            random_state=42, verbosity=0, use_label_encoder=False,
                            eval_metric='logloss')
                        cls.fit(pw_X_sc, pw_y)
                        model_cache[cache_key] = (cls, sc)

                    elif m_type == 'gbc':
                        cls = GradientBoostingClassifier(
                            n_estimators=m_params['n'], max_depth=m_params['md'],
                            learning_rate=0.05, subsample=0.8,
                            min_samples_leaf=5, random_state=42)
                        cls.fit(pw_X_sc, pw_y)
                        model_cache[cache_key] = (cls, sc)

                    elif m_type == 'mlp':
                        cls = MLPClassifier(
                            hidden_layer_sizes=m_params['hidden'],
                            max_iter=500, random_state=42,
                            early_stopping=True, validation_fraction=0.1,
                            learning_rate='adaptive', alpha=0.001)
                        cls.fit(pw_X_sc, pw_y)
                        model_cache[cache_key] = (cls, sc)

                model, model_sc = model_cache[cache_key]

                # Score
                if m_type == 'lr_aug':
                    s = pairwise_score_aug(model, X_te, model_sc, mode=score_mode)
                elif score_mode == 'rank':
                    s = pairwise_score_rank(model, X_te, model_sc)
                elif score_mode == 'raw':
                    s = pairwise_score_raw(model, X_te, model_sc)
                elif score_mode == 'bt':
                    s = pairwise_score_bt(model, X_te, model_sc)
                else:
                    s = pairwise_score_rank(model, X_te, model_sc)

                for i, gi in enumerate(season_indices):
                    all_scores[m_name][gi] = s[i]

            except Exception as e:
                print(f'    ERROR {m_name}: {e}')

        print(f'    Fold {fi + 1}/{len(folds)} ({hold_season}): {len(model_cache)} models, '
              f'{season_test_mask.sum()} test teams')

    print(f'  Precomputation: {time.time() - t0:.0f}s')

    # ──────────────────────────────────────────────────
    # PHASE 1: Individual method evaluation
    # ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' PHASE 1: Individual Method Performance')
    print('=' * 70)

    indiv_results = {}
    for m_name, _, _, _, _ in method_defs:
        e, r = _eval_kaggle(all_scores, [(m_name, 1.0)], 0.15,
                            y, seasons, test_mask, folds)
        indiv_results[m_name] = (e, r)

    # Sort by RMSE
    sorted_methods = sorted(indiv_results.items(), key=lambda x: x[1][1])
    print(f'\n  {"Method":<25} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─" * 25} {"─" * 5} {"─" * 8}')
    for name, (e, r) in sorted_methods:
        marker = ''
        if 'raw' in name and name.replace('_raw', '_rank') in indiv_results:
            rank_r = indiv_results[name.replace('_raw', '_rank')][1]
            if r < rank_r:
                marker = ' ★ raw wins!'
            elif r > rank_r:
                marker = ' (rank better)'
        if 'bt' in name:
            rank_name = name.replace('_bt', '_rank')
            if rank_name in indiv_results:
                rank_r = indiv_results[rank_name][1]
                if r < rank_r:
                    marker = ' ★ BT wins!'
        print(f'  {name:<25} {e:5d} {r:8.4f}{marker}')

    # ──────────────────────────────────────────────────
    # PHASE 2: v6 in rank vs raw vs BT scoring
    # ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' PHASE 2: v6 Config in Different Scoring Modes')
    print('=' * 70)

    scoring_configs = [
        ('v6 rank (baseline)', [
            ('LR_C5_rank', 0.64), ('LRk25_C0.5_rank', 0.28), ('XGB_d4_300_rank', 0.08)]),
        ('v6 raw', [
            ('LR_C5_raw', 0.64), ('LRk25_C0.5_raw', 0.28), ('XGB_d4_300_raw2', 0.08)]),
        ('v6 BT', [
            ('LR_C5_bt', 0.64), ('LRk25_C0.5_bt', 0.28), ('XGB_d4_300_rank', 0.08)]),
        ('v6 mixed (raw+rank)', [
            ('LR_C5_raw', 0.64), ('LRk25_C0.5_rank', 0.28), ('XGB_d4_300_raw2', 0.08)]),
    ]

    for label, weights in scoring_configs:
        for power in [0.10, 0.15, 0.20, 0.25, 0.375, 0.50]:
            e, r = _eval_kaggle(all_scores, weights, power,
                                y, seasons, test_mask, folds)
            marker = ' ← v6' if power == 0.15 and 'baseline' in label else ''
            if r < 2.474:
                marker = ' ★ BETTER THAN v6!'
            print(f'  {label:<24} p={power:.3f}: {e:3d}/91 exact, RMSE={r:.4f}{marker}')
        print()

    # ──────────────────────────────────────────────────
    # PHASE 3: Systematic 3-component blend search (top methods)
    # ──────────────────────────────────────────────────
    print('=' * 70)
    print(' PHASE 3: 3-Component Blend Search')
    print('=' * 70)

    # Select top methods for blend search (by individual performance)
    top_methods = [n for n, (e, r) in sorted_methods if r < 4.0][:20]
    print(f'  Candidate methods: {len(top_methods)}')
    print(f'  Methods: {top_methods}')

    powers = [0.10, 0.15, 0.20, 0.25, 0.30, 0.375, 0.50]
    weight_steps = np.arange(0.04, 0.97, 0.04)  # 4% steps

    best_rmse3 = 999.0
    best_exact3 = 0
    best_cfg_rmse3 = None
    best_cfg_exact3 = None
    n_tested = 0
    pareto3 = {}

    combos = list(itertools.combinations(range(len(top_methods)), 3))
    print(f'  Triplets: {len(combos)}')
    total_configs_est = len(combos) * 200 * len(powers)
    print(f'  Estimated configs: ~{total_configs_est:,}')

    for ci, (i, j, k) in enumerate(combos):
        m1, m2, m3 = top_methods[i], top_methods[j], top_methods[k]

        for w1 in weight_steps:
            for w2 in weight_steps:
                w3 = round(1.0 - w1 - w2, 4)
                if w3 < 0.04 or w3 > 0.96:
                    continue

                for power in powers:
                    n_tested += 1
                    e, r = _eval_kaggle(all_scores,
                                        [(m1, w1), (m2, w2), (m3, w3)],
                                        power, y, seasons, test_mask, folds)

                    if r < best_rmse3:
                        best_rmse3 = r
                        best_cfg_rmse3 = (m1, w1, m2, w2, m3, w3, power)

                    if e > best_exact3 or (e == best_exact3 and r < pareto3.get(e, (999, None))[0]):
                        best_exact3 = max(best_exact3, e)
                        pareto3[e] = (r, (m1, w1, m2, w2, m3, w3, power))

        if (ci + 1) % 50 == 0 or ci == len(combos) - 1:
            elapsed = time.time() - t0
            print(f'    {ci + 1}/{len(combos)} triplets, {n_tested:,} configs, '
                  f'best={best_rmse3:.4f}, best_exact={best_exact3}, '
                  f'{elapsed:.0f}s')

    print(f'\n  3-COMPONENT RESULTS ({n_tested:,} configs tested):')
    print(f'  Best RMSE: {best_rmse3:.4f}')
    if best_cfg_rmse3:
        m1, w1, m2, w2, m3, w3, p = best_cfg_rmse3
        e, r = _eval_kaggle(all_scores, [(m1, w1), (m2, w2), (m3, w3)],
                            p, y, seasons, test_mask, folds)
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}, p={p}')
        print(f'    {e}/91 exact, RMSE={r:.4f}')

    print(f'\n  Pareto frontier (exact → RMSE):')
    for ex in sorted(pareto3.keys(), reverse=True):
        r, cfg = pareto3[ex]
        m1, w1, m2, w2, m3, w3, p = cfg
        star = ' ★' if r < 2.474 else ''
        print(f'    {ex}/91 exact, RMSE={r:.4f}: '
              f'{w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3}, p={p}{star}')
        if ex <= best_exact3 - 8:  # Don't print too many
            break

    # ──────────────────────────────────────────────────
    # PHASE 4: 4-component blend search (v6 base + 1 new)
    # ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' PHASE 4: 4-Component Blend (v6 base + 1 new method)')
    print('=' * 70)

    # v6 base methods in both scoring modes
    v6_base_sets = [
        ('rank_base', 'LR_C5_rank', 'LRk25_C0.5_rank', 'XGB_d4_300_rank'),
        ('raw_base',  'LR_C5_raw',  'LRk25_C0.5_raw',  'XGB_d4_300_raw2'),
    ]

    # New methods to try adding
    new_methods = [n for n in top_methods
                   if n not in {'LR_C5_rank', 'LRk25_C0.5_rank', 'XGB_d4_300_rank',
                                'LR_C5_raw', 'LRk25_C0.5_raw', 'XGB_d4_300_raw2'}]

    best_rmse4 = 999.0
    best_cfg4 = None
    best_exact4 = 0
    n4 = 0

    for base_label, m1_name, m2_name, m3_name in v6_base_sets:
        for m4_name in new_methods[:15]:  # Top 15 new methods
            # Search weight space: shrink v6 weights to make room for m4
            for w4 in np.arange(0.04, 0.25, 0.04):
                remaining = 1.0 - w4
                for ratio1 in np.arange(0.40, 0.80, 0.04):
                    for ratio2 in np.arange(0.10, 0.50, 0.04):
                        ratio3 = 1.0 - ratio1 - ratio2
                        if ratio3 < 0.02:
                            continue
                        w1 = round(remaining * ratio1, 4)
                        w2 = round(remaining * ratio2, 4)
                        w3 = round(remaining * ratio3, 4)

                        for power in [0.10, 0.15, 0.20, 0.25, 0.375]:
                            n4 += 1
                            e, r = _eval_kaggle(
                                all_scores,
                                [(m1_name, w1), (m2_name, w2),
                                 (m3_name, w3), (m4_name, w4)],
                                power, y, seasons, test_mask, folds)

                            if r < best_rmse4:
                                best_rmse4 = r
                                best_cfg4 = (m1_name, w1, m2_name, w2,
                                             m3_name, w3, m4_name, w4, power)

                            if e > best_exact4:
                                best_exact4 = e

    print(f'  4-component configs tested: {n4:,}')
    print(f'  Best RMSE: {best_rmse4:.4f} (vs v6: 2.4740)')
    if best_cfg4:
        m1, w1, m2, w2, m3, w3, m4, w4, p = best_cfg4
        e, r = _eval_kaggle(all_scores,
                            [(m1, w1), (m2, w2), (m3, w3), (m4, w4)],
                            p, y, seasons, test_mask, folds)
        print(f'    {w1:.0%} {m1} + {w2:.0%} {m2} + {w3:.0%} {m3} + {w4:.0%} {m4}')
        print(f'    p={p}, {e}/91 exact, RMSE={r:.4f}')
    print(f'  Best exact: {best_exact4}/91')

    # ──────────────────────────────────────────────────
    # PHASE 5: LOSO validation of top configs
    # ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' PHASE 5: LOSO Validation of Top Configs')
    print('=' * 70)

    loso_configs = [
        ('v6 baseline', [('LR_C5_rank', 0.64), ('LRk25_C0.5_rank', 0.28),
                         ('XGB_d4_300_rank', 0.08)], 0.15),
    ]

    if best_cfg_rmse3:
        m1, w1, m2, w2, m3, w3, p = best_cfg_rmse3
        loso_configs.append(('v7 3c best RMSE',
                             [(m1, w1), (m2, w2), (m3, w3)], p))

    if best_cfg4:
        m1, w1, m2, w2, m3, w3, m4, w4, p = best_cfg4
        loso_configs.append(('v7 4c best RMSE',
                             [(m1, w1), (m2, w2), (m3, w3), (m4, w4)], p))

    # Also test raw scoring with v6 weights
    loso_configs.append(('v6 raw scoring', [
        ('LR_C5_raw', 0.64), ('LRk25_C0.5_raw', 0.28),
        ('XGB_d4_300_raw2', 0.08)], 0.15))

    for cfg_label, weights, power in loso_configs:
        _run_loso(cfg_label, weights, power, X, y, seasons, fn, folds)

    # ──────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print(' v7 EXPLORATION SUMMARY')
    print('=' * 70)
    print(f'  v6 baseline: 56/91, RMSE=2.4740')
    print(f'  v7 best 3-component: {best_rmse3:.4f}')
    print(f'  v7 best 4-component: {best_rmse4:.4f}')
    if best_rmse3 < 2.474 or best_rmse4 < 2.474:
        print(f'  ★ v7 BEAT v6!')
    else:
        print(f'  v6 remains the best config (or v7 only marginally different)')
    print(f'  Total time: {time.time() - t0:.0f}s')


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def _eval_kaggle(all_scores, weights, power, y, seasons, test_mask, folds):
    """Fast Kaggle test evaluation from precomputed scores."""
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
    """Run full LOSO validation using precomputed-style methods."""
    print(f'\n  --- {label} ---')

    # Need to retrain per LOSO fold (different train set)
    n = len(y)
    all_assigned = np.zeros(n, dtype=int)
    fold_stats = []

    for hold in folds:
        tr = seasons != hold
        te = seasons == hold

        pw_X, pw_y = build_pairwise_data(X[tr], y[tr], seasons[tr])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        # Also build augmented pairwise
        pw_X_aug, pw_y_aug = build_pairwise_data_aug(X[tr], y[tr], seasons[tr])
        sc_aug = StandardScaler()
        pw_X_aug_sc = sc_aug.fit_transform(pw_X_aug)

        topk_cache = {}
        topk_sc_cache = {}
        for K in [15, 20, 25, 30, 35]:
            tk_idx = select_top_k_features(X[tr], y[tr], fn, k=K)[0]
            topk_cache[K] = tk_idx
            X_k_tr = X[tr][:, tk_idx]
            pk_X, pk_y = build_pairwise_data(X_k_tr, y[tr], seasons[tr])
            sc_k = StandardScaler()
            sc_k.fit_transform(pk_X)
            topk_sc_cache[K] = (pk_X, pk_y, sc_k)

        bl = np.zeros(int(te.sum()))
        X_te = X[te]

        for m_tag, w in weights:
            s = _compute_method_score(
                m_tag, X_te, X[tr], y[tr], seasons[tr],
                pw_X_sc, pw_y, sc, pw_X_aug_sc, pw_y_aug, sc_aug,
                topk_cache, topk_sc_cache, fn)
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


def _compute_method_score(m_tag, X_te, X_tr, y_tr, seasons_tr,
                           pw_X_sc, pw_y, sc,
                           pw_X_aug_sc, pw_y_aug, sc_aug,
                           topk_cache, topk_sc_cache, fn):
    """Compute a single method's score for LOSO evaluation."""
    # Parse method tag
    parts = m_tag.split('_')

    # Determine score mode
    score_mode = 'rank'
    if m_tag.endswith('_raw') or m_tag.endswith('_raw2'):
        score_mode = 'raw'
    elif m_tag.endswith('_bt'):
        score_mode = 'bt'

    # Augmented features?
    if 'aug' in m_tag:
        C = float(parts[1].replace('C', ''))
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pw_X_aug_sc, pw_y_aug)
        return pairwise_score_aug(cls, X_te, sc_aug, mode=score_mode)

    # TopK?
    if m_tag.startswith('LRk'):
        K_str = parts[0].replace('LRk', '')
        K = int(K_str)
        C_str = parts[1].replace('C', '')
        C = float(C_str)
        tk_idx = topk_cache[K]
        pk_X, pk_y, sc_k = topk_sc_cache[K]
        pk_X_sc = sc_k.transform(pk_X)
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(sc_k.fit_transform(pk_X), pk_y)
        X_te_k = X_te[:, tk_idx]
        if score_mode == 'rank':
            return pairwise_score_rank(cls, X_te_k, sc_k)
        elif score_mode == 'bt':
            return pairwise_score_bt(cls, X_te_k, sc_k)
        else:
            return pairwise_score_raw(cls, X_te_k, sc_k)

    # LR full
    if m_tag.startswith('LR_C'):
        C = float(parts[1].replace('C', ''))
        cls = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        if score_mode == 'rank':
            return pairwise_score_rank(cls, X_te, sc)
        elif score_mode == 'bt':
            return pairwise_score_bt(cls, X_te, sc)
        else:
            return pairwise_score_raw(cls, X_te, sc)

    # XGB
    if m_tag.startswith('XGB'):
        md = int(parts[1].replace('d', ''))
        n_est = int(parts[2])
        lr_val = 0.05  # default
        cls = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=lr_val,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        cls.fit(pw_X_sc, pw_y)
        if score_mode == 'rank':
            return pairwise_score_rank(cls, X_te, sc)
        else:
            return pairwise_score_raw(cls, X_te, sc)

    # GBC
    if m_tag.startswith('GBC'):
        n_est = int(parts[1])
        md = int(parts[2].replace('d', ''))
        cls = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=md,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=5, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        if score_mode == 'rank':
            return pairwise_score_rank(cls, X_te, sc)
        else:
            return pairwise_score_raw(cls, X_te, sc)

    # MLP
    if m_tag.startswith('MLP'):
        h_str = parts[1]
        if len(parts) > 2 and parts[2] not in ('rank', 'raw', 'bt'):
            hidden = (int(parts[1]), int(parts[2]))
        else:
            hidden = (int(h_str),)
        cls = MLPClassifier(
            hidden_layer_sizes=hidden, max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            learning_rate='adaptive', alpha=0.001)
        cls.fit(pw_X_sc, pw_y)
        if score_mode == 'rank':
            return pairwise_score_rank(cls, X_te, sc)
        else:
            return pairwise_score_raw(cls, X_te, sc)

    raise ValueError(f'Unknown method: {m_tag}')


if __name__ == '__main__':
    main()
