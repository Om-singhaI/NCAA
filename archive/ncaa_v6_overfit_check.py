#!/usr/bin/env python3
"""
v6 Overfitting Diagnostic
=========================
Test whether v6 improvements are genuine or overfit to the 91 Kaggle test teams.

Tests:
  1. LOSO comparison: v4 vs v5 vs v6 (no test teams visible)
  2. Leave-one-season-out on TEST TEAMS ONLY (true held-out)
     - Train on 4 seasons of training+test, predict 5th season's test teams
  3. Weight perturbation stability: small changes to weights/power
  4. Bootstrap: random 50% of test teams, repeat 1000×
  5. Per-season variance analysis
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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
    print('='*70)
    print(' v6 OVERFITTING DIAGNOSTIC')
    print('='*70)

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

    # Define configs to compare
    configs = {
        'v4': {
            'methods': [('lr', {'C': 5.0}, 0.60),
                        ('lr', {'C': 0.01}, 0.10),
                        ('lr_topk', {'C': 1.0, 'K': 25}, 0.30)],
            'power': 1.0,
        },
        'v5': {
            'methods': [('lr', {'C': 5.0}, 0.60),
                        ('lr_topk', {'C': 1.0, 'K': 25}, 0.20),
                        ('xgb', {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}, 0.20)],
            'power': 0.5,
        },
        'v6': {
            'methods': [('lr', {'C': 5.0}, 0.64),
                        ('lr_topk', {'C': 0.5, 'K': 25}, 0.28),
                        ('xgb', {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}, 0.08)],
            'power': 0.15,
        },
    }

    # ══════════════════════════════════════════════════════════════
    # TEST 1: FULL LOSO (no test team info, all 68 teams predicted)
    # This is the gold standard for generalization
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 1: FULL LOSO (all 340 teams, no locked seeds)')
    print(' This tests generalization — no test set info used')
    print('='*70)

    for cfg_name, cfg in configs.items():
        fold_results = []
        all_assigned = np.zeros(n, dtype=int)

        for hold in folds:
            tr = seasons != hold
            te = seasons == hold

            top_k_idx = select_top_k_features(X[tr], y[tr], fn, k=25)[0]

            pw_X, pw_y = build_pairwise_data(X[tr], y[tr], seasons[tr])
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            bl = np.zeros(int(te.sum()))
            for m_type, m_params, w in cfg['methods']:
                s = _compute_method(m_type, m_params, pw_X_sc, pw_y,
                                    X[te], X[tr], y[tr], seasons[tr],
                                    sc, top_k_idx, fn)
                bl += w * s

            avail = {hold: list(range(1, 69))}
            assigned = hungarian(bl, seasons[te], avail, power=cfg['power'])
            all_assigned[te] = assigned

            yte = y[te].astype(int)
            exact = int(np.sum(assigned == yte))
            rmse_f = np.sqrt(np.mean((assigned - yte)**2))
            fold_results.append((hold, int(te.sum()), exact, rmse_f))

        total_exact = int(np.sum(all_assigned == y.astype(int)))
        total_rmse = np.sqrt(np.mean((all_assigned - y.astype(int))**2))
        fold_rmses = [r for _, _, _, r in fold_results]
        score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)

        print(f'\n  {cfg_name}: LOSO RMSE={total_rmse:.4f}, exact={total_exact}/340, '
              f'score={score:.4f}')
        for s, nf, ex, rm in fold_results:
            print(f'    {s}: {ex}/{nf} exact ({ex/nf*100:.1f}%), RMSE={rm:.3f}')

    # ══════════════════════════════════════════════════════════════
    # TEST 2: SEASON-LEVEL HOLDOUT ON TEST TEAMS ONLY
    # Hold out one season's TEST teams entirely (don't even train on them)
    # Train on remaining 4 seasons' training+test teams
    # This tests if improvements transfer across seasons
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 2: SEASON-LEVEL HOLDOUT (test teams only)')
    print(' Hold out entire season of test teams — true generalization')
    print('='*70)

    for cfg_name, cfg in configs.items():
        season_exact = []
        season_rmse = []
        total_e = 0
        total_n = 0
        total_se = 0

        for hold in folds:
            # Hold out this season's test teams
            hold_test_mask = test_mask & (seasons == hold)
            n_hold_test = int(hold_test_mask.sum())
            if n_hold_test == 0:
                continue

            # Train on everything EXCEPT this season's test teams
            train_mask = ~hold_test_mask

            # Build features on the training subset
            top_k_idx = select_top_k_features(
                X[train_mask], y[train_mask], fn, k=25)[0]

            # Build pairwise from training
            pw_X, pw_y = build_pairwise_data(
                X[train_mask], y[train_mask], seasons[train_mask])
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            # Predict ALL teams in hold season (need full season for Hungarian)
            hold_season_mask = (seasons == hold)
            X_hold = X[hold_season_mask]
            hold_indices = np.where(hold_season_mask)[0]

            bl = np.zeros(len(hold_indices))
            for m_type, m_params, w in cfg['methods']:
                s = _compute_method(m_type, m_params, pw_X_sc, pw_y,
                                    X_hold, X[train_mask], y[train_mask],
                                    seasons[train_mask], sc, top_k_idx, fn)
                bl += w * s

            # Lock the TRAINING teams' seeds in the hold season
            for i, gi in enumerate(hold_indices):
                if not test_mask[gi]:  # training team → lock seed
                    bl[i] = y[gi]

            avail = {hold: list(range(1, 69))}
            assigned = hungarian(bl, seasons[hold_season_mask], avail,
                                  power=cfg['power'])

            # Evaluate only on test teams
            gt = []; pred = []
            for i, gi in enumerate(hold_indices):
                if test_mask[gi]:
                    gt.append(int(y[gi]))
                    pred.append(assigned[i])

            gt = np.array(gt); pred = np.array(pred)
            exact = int((pred == gt).sum())
            rmse = np.sqrt(np.mean((pred - gt)**2))

            season_exact.append(exact)
            season_rmse.append(rmse)
            total_e += exact
            total_n += len(gt)
            total_se += np.sum((pred - gt)**2)

        total_rmse = np.sqrt(total_se / total_n)
        avg_rmse = np.mean(season_rmse)
        std_rmse = np.std(season_rmse)
        score = avg_rmse + 0.5 * std_rmse

        print(f'\n  {cfg_name}: {total_e}/{total_n} exact ({total_e/total_n*100:.1f}%), '
              f'RMSE={total_rmse:.4f}, score={score:.4f}')
        for i, hold in enumerate(folds):
            print(f'    {hold}: {season_exact[i]} exact, RMSE={season_rmse[i]:.4f}')

    # ══════════════════════════════════════════════════════════════
    # TEST 3: WEIGHT PERTURBATION STABILITY
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 3: STABILITY — weight/power perturbations')
    print(' If overfit, small changes should degrade significantly')
    print('='*70)

    # Precompute method scores for Kaggle eval
    method_scores = _precompute_kaggle_scores(
        X, y, seasons, test_mask, folds, fn)

    v6_base_e, v6_base_r = _eval_config(
        method_scores, [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)],
        0.15, y, seasons, test_mask, folds)
    print(f'\n  v6 base: {v6_base_e}/91 exact, RMSE={v6_base_r:.4f}')

    # Perturb weights ±2-5%
    perturbations = [
        ('w1±2%', [('lr_C5', 0.62), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.08)]),
        ('w1±4%', [('lr_C5', 0.60), ('lrk25_C0.5', 0.32), ('xgb_d4_300', 0.08)]),
        ('w1+4%', [('lr_C5', 0.68), ('lrk25_C0.5', 0.24), ('xgb_d4_300', 0.08)]),
        ('w3±4%', [('lr_C5', 0.64), ('lrk25_C0.5', 0.24), ('xgb_d4_300', 0.12)]),
        ('w3=0%', [('lr_C5', 0.70), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.00)]),
        ('w3=16%',[('lr_C5', 0.56), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.16)]),
        ('even',  [('lr_C5', 0.50), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.20)]),
    ]

    print(f'\n  {"Perturbation":<15} {"Exact":>5} {"RMSE":>8} {"Δ RMSE":>8}')
    print(f'  {"─"*15} {"─"*5} {"─"*8} {"─"*8}')
    for label, weights in perturbations:
        e, r = _eval_config(method_scores, weights, 0.15, y, seasons, test_mask, folds)
        print(f'  {label:<15} {e:5d} {r:8.4f} {r-v6_base_r:+8.4f}')

    # Perturb power
    print(f'\n  {"Power":<15} {"Exact":>5} {"RMSE":>8} {"Δ RMSE":>8}')
    print(f'  {"─"*15} {"─"*5} {"─"*8} {"─"*8}')
    for power in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.375, 0.50, 0.625, 0.75, 1.0]:
        e, r = _eval_config(
            method_scores,
            [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)],
            power, y, seasons, test_mask, folds)
        marker = ' ← v6' if power == 0.15 else (' ← v5' if power == 0.5 else '')
        print(f'  p={power:<11.3f} {e:5d} {r:8.4f} {r-v6_base_r:+8.4f}{marker}')

    # ══════════════════════════════════════════════════════════════
    # TEST 4: BOOTSTRAP CONFIDENCE INTERVAL
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 4: BOOTSTRAP (1000 resamples of test teams)')
    print('='*70)

    # Get predictions for v5 and v6
    v5_pred = _get_predictions(
        method_scores,
        [('lr_C5', 0.60), ('lrk25_C1', 0.20), ('xgb_d3_200', 0.20)],
        0.5, y, seasons, test_mask, folds)

    v6_pred = _get_predictions(
        method_scores,
        [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)],
        0.15, y, seasons, test_mask, folds)

    gt = y[test_mask].astype(int)
    test_indices = np.where(test_mask)[0]

    v5_rmses = []
    v6_rmses = []
    v6_wins = 0
    rng = np.random.RandomState(42)

    for _ in range(1000):
        idx = rng.choice(len(gt), len(gt), replace=True)
        v5_rmse = np.sqrt(np.mean((v5_pred[idx] - gt[idx])**2))
        v6_rmse = np.sqrt(np.mean((v6_pred[idx] - gt[idx])**2))
        v5_rmses.append(v5_rmse)
        v6_rmses.append(v6_rmse)
        if v6_rmse < v5_rmse:
            v6_wins += 1

    v5_rmses = np.array(v5_rmses)
    v6_rmses = np.array(v6_rmses)
    delta_rmses = v5_rmses - v6_rmses

    print(f'\n  v5 RMSE: {np.mean(v5_rmses):.4f} '
          f'[{np.percentile(v5_rmses, 2.5):.4f}, {np.percentile(v5_rmses, 97.5):.4f}]')
    print(f'  v6 RMSE: {np.mean(v6_rmses):.4f} '
          f'[{np.percentile(v6_rmses, 2.5):.4f}, {np.percentile(v6_rmses, 97.5):.4f}]')
    print(f'  Δ (v5-v6): {np.mean(delta_rmses):.4f} '
          f'[{np.percentile(delta_rmses, 2.5):.4f}, {np.percentile(delta_rmses, 97.5):.4f}]')
    print(f'  v6 wins: {v6_wins}/1000 ({v6_wins/10:.1f}%)')
    print(f'  Probability v6 > v5: {100-v6_wins/10:.1f}% (overfit risk)')

    # ══════════════════════════════════════════════════════════════
    # TEST 5: CROSS-VALIDATED HYPERPARAMETER SELECTION
    # Leave one season's test teams out, find best config on other 4,
    # then evaluate on held-out season
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 5: CV-SELECTED CONFIG (honest model selection)')
    print(' For each season: select best config on other 4 seasons,')
    print(' then evaluate on held-out season. No information leak.')
    print('='*70)

    # Test a range of configs
    test_configs = [
        ('v4', [('lr_C5', 0.60), ('lr_C0.01', 0.10), ('lrk25_C1', 0.30)], 1.0),
        ('v5', [('lr_C5', 0.60), ('lrk25_C1', 0.20), ('xgb_d3_200', 0.20)], 0.5),
        ('v6', [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)], 0.15),
        ('v6_p0.25', [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)], 0.25),
        ('v6_p0.375', [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)], 0.375),
        ('v6_p0.5', [('lr_C5', 0.64), ('lrk25_C0.5', 0.28), ('xgb_d4_300', 0.08)], 0.5),
        ('v6w_p0.25', [('lr_C5', 0.60), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.10)], 0.25),
        ('v6w_p0.375', [('lr_C5', 0.60), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.10)], 0.375),
        ('60/30/10_p0.15', [('lr_C5', 0.60), ('lrk25_C0.5', 0.30), ('xgb_d4_300', 0.10)], 0.15),
    ]

    # For each held-out season, find which config wins on the other 4
    cv_results = {s: {} for s in folds}
    for hold in folds:
        other_seasons = [s for s in folds if s != hold]
        # Evaluate each config on the 4 other seasons' test teams
        for cfg_name, weights, power in test_configs:
            rmse_sum = 0; n_sum = 0
            for eval_season in other_seasons:
                eval_mask = test_mask & (seasons == eval_season)
                if eval_mask.sum() == 0:
                    continue
                season_mask = (seasons == eval_season)
                season_indices = np.where(season_mask)[0]

                bl = np.zeros(len(season_indices))
                for m_tag, w in weights:
                    if m_tag in method_scores:
                        bl += w * method_scores[m_tag][season_mask]

                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        bl[i] = y[gi]

                avail = {eval_season: list(range(1, 69))}
                assigned = hungarian(bl, seasons[season_mask], avail, power=power)

                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        rmse_sum += (assigned[i] - int(y[gi]))**2
                        n_sum += 1

            cv_rmse = np.sqrt(rmse_sum / n_sum) if n_sum > 0 else 999
            cv_results[hold][cfg_name] = cv_rmse

    # For each held-out season, pick the config that won on other 4
    print(f'\n  {"Season":<12} {"Selected":<20} {"CV RMSE":>8}  {"Held-out RMSE":>13}')
    print(f'  {"─"*12} {"─"*20} {"─"*8}  {"─"*13}')

    cv_total_se = 0; cv_total_n = 0
    for hold in folds:
        # Find best config on other 4 seasons
        best_cfg = min(cv_results[hold], key=cv_results[hold].get)
        cv_rmse = cv_results[hold][best_cfg]

        # Get that config's actual weights/power
        for cfg_name, weights, power in test_configs:
            if cfg_name == best_cfg:
                break

        # Evaluate on held-out season
        hold_mask = test_mask & (seasons == hold)
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]

        bl = np.zeros(len(season_indices))
        for m_tag, w in weights:
            if m_tag in method_scores:
                bl += w * method_scores[m_tag][season_mask]

        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]

        avail = {hold: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=power)

        gt_s = []; pred_s = []
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                gt_s.append(int(y[gi]))
                pred_s.append(assigned[i])
        gt_s = np.array(gt_s); pred_s = np.array(pred_s)
        held_rmse = np.sqrt(np.mean((pred_s - gt_s)**2))
        held_exact = int((pred_s == gt_s).sum())

        cv_total_se += np.sum((pred_s - gt_s)**2)
        cv_total_n += len(gt_s)

        print(f'  {hold:<12} {best_cfg:<20} {cv_rmse:8.4f}  '
              f'{held_rmse:8.4f} ({held_exact}/{len(gt_s)})')

    cv_total_rmse = np.sqrt(cv_total_se / cv_total_n)
    print(f'\n  CV-SELECTED TOTAL RMSE: {cv_total_rmse:.4f}')
    print(f'  v5 fixed RMSE:         2.6520')
    print(f'  v6 fixed RMSE:         2.4740')

    # ═════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' OVERFITTING ASSESSMENT SUMMARY')
    print('='*70)
    print(f'  Configs tested on 91 test teams: 610,000+')
    print(f'  Test set size: 91 teams (small!)')
    print(f'  Degrees of freedom: 3 weights + 1 power = 4 params')
    print(f'  Effective search ratio: ~150,000 configs per parameter')
    print(f'\n  v6 bootstrap win rate: {v6_wins/10:.1f}%')
    if v6_wins >= 900:
        print(f'  → Strong evidence v6 is genuinely better')
    elif v6_wins >= 700:
        print(f'  → Moderate evidence v6 is better, some overfit risk')
    elif v6_wins >= 500:
        print(f'  → Weak evidence, substantial overfit risk')
    else:
        print(f'  → v6 is likely overfit — v5 is safer')

    print(f'\n  Time: {time.time()-t0:.0f}s')


# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def _compute_method(m_type, m_params, pw_X_sc, pw_y, X_test, X_train, y_train,
                    seasons_train, sc_full, top_k_idx, fn):
    """Compute pairwise scores for one method."""
    if m_type == 'lr':
        cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                max_iter=2000, random_state=42)
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_test, sc_full)

    elif m_type == 'lr_topk':
        K = m_params['K']
        tk = select_top_k_features(X_train, y_train, fn, k=K)[0]
        pw_Xk, pw_yk = build_pairwise_data(X_train[:, tk], y_train, seasons_train)
        sc_k = StandardScaler()
        pw_Xk_sc = sc_k.fit_transform(pw_Xk)
        cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                max_iter=2000, random_state=42)
        cls.fit(pw_Xk_sc, pw_yk)
        return pairwise_score(cls, X_test[:, tk], sc_k)

    elif m_type == 'xgb':
        cls = xgb.XGBClassifier(
            n_estimators=m_params['n_estimators'],
            max_depth=m_params['max_depth'],
            learning_rate=m_params['learning_rate'],
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        cls.fit(pw_X_sc, pw_y)
        return pairwise_score(cls, X_test, sc_full)


def _precompute_kaggle_scores(X, y, seasons, test_mask, folds, fn):
    """Precompute all method scores for Kaggle-style evaluation."""
    method_defs = [
        ('lr_C5',      'lr',    {'C': 5.0}),
        ('lr_C0.01',   'lr',    {'C': 0.01}),
        ('lrk25_C1',   'lr_topk', {'C': 1.0, 'K': 25}),
        ('lrk25_C0.5', 'lr_topk', {'C': 0.5, 'K': 25}),
        ('xgb_d3_200', 'xgb',   {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}),
        ('xgb_d4_300', 'xgb',   {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}),
    ]

    n = len(y)
    all_scores = {m: np.zeros(n) for m, _, _ in method_defs}

    for hold_season in folds:
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
            topk_cache[K] = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]

        for m_name, m_type, m_params in method_defs:
            if m_type == 'lr':
                cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                        max_iter=2000, random_state=42)
                cls.fit(pw_X_sc, pw_y_pw)
                s = pairwise_score(cls, X_season, sc)

            elif m_type == 'lr_topk':
                K = m_params['K']
                top_idx = topk_cache[K]
                X_k_tr = X[global_train_mask][:, top_idx]
                X_k_te = X_season[:, top_idx]
                pw_Xk, pw_yk = build_pairwise_data(
                    X_k_tr, y[global_train_mask], seasons[global_train_mask])
                sc_k = StandardScaler()
                pw_Xk_sc = sc_k.fit_transform(pw_Xk)
                cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                        max_iter=2000, random_state=42)
                cls.fit(pw_Xk_sc, pw_yk)
                s = pairwise_score(cls, X_k_te, sc_k)

            elif m_type == 'xgb':
                cls = xgb.XGBClassifier(
                    n_estimators=m_params['n_estimators'],
                    max_depth=m_params['max_depth'],
                    learning_rate=m_params['learning_rate'],
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=42, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss')
                cls.fit(pw_X_sc, pw_y_pw)
                s = pairwise_score(cls, X_season, sc)

            for i, gi in enumerate(season_indices):
                all_scores[m_name][gi] = s[i]

    return all_scores


def _eval_config(method_scores, weights, power, y, seasons, test_mask, folds):
    """Evaluate a config on Kaggle test set."""
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = np.zeros(len(season_indices))
        for m_tag, w in weights:
            if m_tag in method_scores:
                bl += w * method_scores[m_tag][season_mask]
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    return int((pred == gt).sum()), np.sqrt(np.mean((pred - gt)**2))


def _get_predictions(method_scores, weights, power, y, seasons, test_mask, folds):
    """Get test team predictions for a config."""
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = np.zeros(len(season_indices))
        for m_tag, w in weights:
            if m_tag in method_scores:
                bl += w * method_scores[m_tag][season_mask]
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    return test_assigned[test_mask]


if __name__ == '__main__':
    main()
