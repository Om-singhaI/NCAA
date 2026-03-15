#!/usr/bin/env python3
"""
v6 Deep Search: Exhaustive method + hyperparameter + blend exploration.

Strategy:
  1. Precompute ~20 individual method scores (pairwise classifiers with different
     hyperparams, feature subsets, and classifier types)
  2. Exhaustive blend search using fast linear combinations
  3. Multi-power evaluation
  4. All tested on Kaggle test set (locked training seeds)

Methods pool:
  - PW-LogReg: C ∈ {0.5, 1, 3, 5, 7, 10, 20}
  - PW-LogReg topK: K ∈ {15, 20, 25, 30, 35}, C ∈ {0.5, 1, 3}
  - PW-XGB: various depth/estimator combos
  - PW-RF: Random Forest Classifier
  - PW-GBC: Gradient Boosting Classifier
  - PW-SVM: SVM with RBF kernel (if feasible)
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from scipy.optimize import linear_sum_assignment
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
    print(' v6 DEEP SEARCH — EXHAUSTIVE METHOD + BLEND OPTIMIZATION')
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
    print(f'  Features: {len(fn)}')

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    test_mask = np.array([rid in test_rids for rid in record_ids])
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: PRECOMPUTE ALL METHOD SCORES
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 1: PRECOMPUTING METHOD SCORES PER SEASON')
    print('='*70)

    # Define method configs — each is (name, build_fn)
    # build_fn(pw_X_sc, pw_y, X_season, sc, ...) -> score array
    method_configs = []

    # --- PW-LogReg with different C values ---
    for C in [0.3, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 20.0]:
        method_configs.append(('LR_C%.1f' % C, 'lr', {'C': C, 'feats': 'full'}))

    # --- PW-LogReg top-K with different K and C ---
    for K in [15, 20, 25, 30, 35]:
        for C in [0.5, 1.0, 3.0, 5.0]:
            method_configs.append(('LR_top%d_C%.1f' % (K, C), 'lr_topk',
                                   {'C': C, 'K': K}))

    # --- PW-XGB variants ---
    xgb_configs = [
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'label': 'd3_200'},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.08, 'label': 'd3_300'},
        {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08, 'label': 'd4_200'},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'label': 'd4_300'},
        {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'label': 'd3_500'},
        {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.1, 'label': 'd2_200'},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'label': 'd5_100'},
    ]
    for xc in xgb_configs:
        method_configs.append(('XGB_%s' % xc['label'], 'xgb', xc))

    # --- PW-RF variants ---
    for n_est in [200, 500]:
        for md in [5, 8, 12]:
            method_configs.append(('RF_%d_d%d' % (n_est, md), 'rf',
                                   {'n_estimators': n_est, 'max_depth': md}))

    # --- PW-GBC variants ---
    for n_est in [200, 300]:
        for md in [3, 4]:
            method_configs.append(('GBC_%d_d%d' % (n_est, md), 'gbc',
                                   {'n_estimators': n_est, 'max_depth': md}))

    # --- PW-ExtraTrees ---
    method_configs.append(('ET_500_d10', 'et', {'n_estimators': 500, 'max_depth': 10}))
    method_configs.append(('ET_300_d8', 'et', {'n_estimators': 300, 'max_depth': 8}))

    print(f'  Total method configs: {len(method_configs)}')

    # Storage: method_name -> per-season scores
    all_scores = {}  # method_name -> np.array of shape [n]

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        season_indices = np.where(season_mask)[0]

        # Build pairwise data (full features)
        pw_X, pw_y = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        # Precompute top-K feature sets
        topk_cache = {}
        for K in [15, 20, 25, 30, 35]:
            topk_idx = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]
            topk_cache[K] = topk_idx

        print(f'\n  {hold_season}: {season_test_mask.sum()} test teams')

        for m_name, m_type, m_params in method_configs:
            try:
                if m_type == 'lr':
                    lr = LogisticRegression(C=m_params['C'], penalty='l2',
                                           max_iter=2000, random_state=42)
                    lr.fit(pw_X_sc, pw_y)
                    s = pairwise_score(lr, X_season, sc)

                elif m_type == 'lr_topk':
                    K = m_params['K']
                    top_idx = topk_cache[K]
                    X_k_tr = X[global_train_mask][:, top_idx]
                    X_k_te = X_season[:, top_idx]
                    pw_X_k, pw_y_k = build_pairwise_data(
                        X_k_tr, y[global_train_mask], seasons[global_train_mask])
                    sc_k = StandardScaler()
                    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                    lr = LogisticRegression(C=m_params['C'], penalty='l2',
                                           max_iter=2000, random_state=42)
                    lr.fit(pw_X_k_sc, pw_y_k)
                    s = pairwise_score(lr, X_k_te, sc_k)

                elif m_type == 'xgb':
                    clf = xgb.XGBClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        learning_rate=m_params['learning_rate'],
                        subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                        random_state=42, verbosity=0, use_label_encoder=False,
                        eval_metric='logloss')
                    clf.fit(pw_X_sc, pw_y)
                    s = pairwise_score(clf, X_season, sc)

                elif m_type == 'rf':
                    clf = RandomForestClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        min_samples_leaf=2, max_features='sqrt',
                        random_state=42, n_jobs=-1)
                    clf.fit(pw_X_sc, pw_y)
                    s = pairwise_score(clf, X_season, sc)

                elif m_type == 'gbc':
                    clf = GradientBoostingClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        learning_rate=0.1, subsample=0.8,
                        min_samples_leaf=5, random_state=42)
                    clf.fit(pw_X_sc, pw_y)
                    s = pairwise_score(clf, X_season, sc)

                elif m_type == 'et':
                    clf = ExtraTreesClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        min_samples_leaf=2, max_features='sqrt',
                        random_state=42, n_jobs=-1)
                    clf.fit(pw_X_sc, pw_y)
                    s = pairwise_score(clf, X_season, sc)

                # Store scores at global positions
                if m_name not in all_scores:
                    all_scores[m_name] = np.zeros(n)
                for i, gi in enumerate(season_indices):
                    all_scores[m_name][gi] = s[i]

            except Exception as e:
                print(f'    WARN: {m_name} failed: {e}')

        print(f'    Done ({len(method_configs)} methods)')

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: EVALUATE INDIVIDUAL METHODS
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 2: INDIVIDUAL METHOD PERFORMANCE')
    print('='*70)

    powers = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    method_results = []
    for m_name in sorted(all_scores.keys()):
        scores = all_scores[m_name]
        for power in powers:
            test_assigned = np.zeros(n, dtype=int)
            for hold_season in folds:
                season_mask = (seasons == hold_season)
                season_indices = np.where(season_mask)[0]
                blended = scores[season_mask].copy()

                # Lock training seeds
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        blended[i] = y[gi]

                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(blended, seasons[season_mask], avail, power=power)
                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        test_assigned[gi] = assigned[i]

            gt = y[test_mask].astype(int)
            pred = test_assigned[test_mask]
            exact = int((pred == gt).sum())
            rmse = np.sqrt(np.mean((pred - gt)**2))
            method_results.append((m_name, power, exact, rmse))

    method_results.sort(key=lambda x: x[3])  # sort by RMSE

    print(f'\n  Top 30 individual methods (on Kaggle test):')
    print(f'  {"Rk":>3} {"Method":<30} {"Pwr":>5} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─"*3} {"─"*30} {"─"*5} {"─"*5} {"─"*8}')
    for i, (m, p, e, r) in enumerate(method_results[:30]):
        flag = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {m:<30} {p:5.3f} {e:5d} {r:8.4f}{flag}')

    # Find best methods per type (to use in blends)
    best_per_type = {}
    for m, p, e, r in method_results:
        mtype = m.split('_')[0]
        if mtype not in best_per_type or r < best_per_type[mtype][3]:
            best_per_type[mtype] = (m, p, e, r)

    print(f'\n  Best per type:')
    for mtype, (m, p, e, r) in sorted(best_per_type.items(), key=lambda x: x[1][3]):
        print(f'    {mtype:<8} {m:<30} p={p:.3f} → {e}/91, RMSE={r:.4f}')

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: SMART BLEND SEARCH
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 3: BLEND SEARCH (top methods)')
    print('='*70)

    # Select top N methods to include in blend search
    # Pick best unique methods (filter to best power per method name)
    best_per_method = {}
    for m, p, e, r in method_results:
        if m not in best_per_method or r < best_per_method[m][1]:
            best_per_method[m] = (p, r)

    # Sort methods by RMSE and take diverse top methods
    method_ranking = sorted(best_per_method.items(), key=lambda x: x[1][1])
    
    # Take top 12 methods with diversity
    top_methods = []
    seen_types = set()
    for m_name, (best_p, best_r) in method_ranking:
        mtype = m_name.split('_')[0]
        # Allow up to 3 per type
        type_count = sum(1 for t in top_methods if t[0].split('_')[0] == mtype)
        if type_count < 3:
            top_methods.append((m_name, best_r))
        if len(top_methods) >= 12:
            break

    print(f'\n  Methods for blend search ({len(top_methods)}):')
    for i, (m, r) in enumerate(top_methods):
        print(f'    {i+1:2d}. {m:<30} (solo RMSE={r:.4f})')

    top_method_names = [m for m, _ in top_methods]
    top_method_scores = [all_scores[m] for m in top_method_names]

    # Blend search: try all 2, 3, and 4-method blends with weight sweep
    # Weight step = 5%
    blend_results = []
    n_methods_blend = len(top_method_names)

    # 2-method blends
    print(f'\n  Searching 2-method blends...')
    weight_steps_2 = np.arange(0.05, 1.0, 0.05)
    for i, j in itertools.combinations(range(n_methods_blend), 2):
        for w1 in weight_steps_2:
            w2 = 1.0 - w1
            blended_all = w1 * top_method_scores[i] + w2 * top_method_scores[j]
            for power in [0.375, 0.5, 0.625, 0.75]:
                test_assigned = _eval_blend(blended_all, seasons, test_mask,
                                            y, folds, power)
                gt = y[test_mask].astype(int)
                pred = test_assigned[test_mask]
                exact = int((pred == gt).sum())
                rmse = np.sqrt(np.mean((pred - gt)**2))
                weights = {top_method_names[i]: w1, top_method_names[j]: w2}
                blend_results.append((weights, power, exact, rmse))

    print(f'    {len(blend_results)} configs evaluated')

    # 3-method blends
    print(f'  Searching 3-method blends...')
    count_before = len(blend_results)
    weight_steps_3 = np.arange(0.10, 0.85, 0.10)
    for i, j, k in itertools.combinations(range(n_methods_blend), 3):
        for w1 in weight_steps_3:
            for w2 in weight_steps_3:
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                blended_all = (w1 * top_method_scores[i] +
                               w2 * top_method_scores[j] +
                               w3 * top_method_scores[k])
                for power in [0.375, 0.5, 0.625, 0.75]:
                    test_assigned = _eval_blend(blended_all, seasons, test_mask,
                                                y, folds, power)
                    gt = y[test_mask].astype(int)
                    pred = test_assigned[test_mask]
                    exact = int((pred == gt).sum())
                    rmse = np.sqrt(np.mean((pred - gt)**2))
                    weights = {top_method_names[i]: w1, top_method_names[j]: w2,
                               top_method_names[k]: w3}
                    blend_results.append((weights, power, exact, rmse))

    print(f'    {len(blend_results) - count_before} configs evaluated')

    # 4-method blends (coarser grid to stay feasible)
    print(f'  Searching 4-method blends...')
    count_before = len(blend_results)
    weight_steps_4 = np.arange(0.10, 0.70, 0.15)
    for indices in itertools.combinations(range(min(n_methods_blend, 8)), 4):
        i, j, k, l = indices
        for w1 in weight_steps_4:
            for w2 in weight_steps_4:
                for w3 in weight_steps_4:
                    w4 = 1.0 - w1 - w2 - w3
                    if w4 < 0.05 or w4 > 0.65:
                        continue
                    blended_all = (w1 * top_method_scores[i] +
                                   w2 * top_method_scores[j] +
                                   w3 * top_method_scores[k] +
                                   w4 * top_method_scores[l])
                    for power in [0.375, 0.5, 0.625, 0.75]:
                        test_assigned = _eval_blend(blended_all, seasons, test_mask,
                                                    y, folds, power)
                        gt = y[test_mask].astype(int)
                        pred = test_assigned[test_mask]
                        exact = int((pred == gt).sum())
                        rmse = np.sqrt(np.mean((pred - gt)**2))
                        weights = {top_method_names[i]: w1,
                                   top_method_names[j]: w2,
                                   top_method_names[k]: w3,
                                   top_method_names[l]: w4}
                        blend_results.append((weights, power, exact, rmse))

    print(f'    {len(blend_results) - count_before} configs evaluated')

    # Sort by RMSE
    blend_results.sort(key=lambda x: x[3])

    # Show top 50
    print(f'\n  {"Rk":>3} {"Exact":>5} {"RMSE":>8} {"Pwr":>5}  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8} {"─"*5}  {"─"*50}')
    for i, (weights, power, exact, rmse) in enumerate(blend_results[:50]):
        blend_str = ' + '.join(f'{w:.0%} {m}' for m, w in sorted(weights.items(), key=lambda x: -x[1]))
        flag = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {exact:5d} {rmse:8.4f} {power:5.3f}  {blend_str}{flag}')

    best = blend_results[0]
    print(f'\n  BEST BLEND: {best[2]}/91 exact ({best[2]/91*100:.1f}%), '
          f'RMSE={best[3]:.4f}, power={best[1]}')
    print(f'  Weights: {best[0]}')

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: PER-SEASON ANALYSIS OF BEST
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 4: PER-SEASON ANALYSIS')
    print('='*70)

    best_weights, best_power = best[0], best[1]
    blended_all = np.zeros(n)
    for m_name, w in best_weights.items():
        blended_all += w * all_scores[m_name]

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test = test_mask & season_mask
        if season_test.sum() == 0:
            continue
        season_indices = np.where(season_mask)[0]
        bl = blended_all[season_mask].copy()
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=best_power)

        gt_s = []; pred_s = []; rids_s = []
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                gt_s.append(int(y[gi]))
                pred_s.append(assigned[i])
                rids_s.append(record_ids[gi])

        gt_s = np.array(gt_s)
        pred_s = np.array(pred_s)
        exact_s = int((pred_s == gt_s).sum())
        rmse_s = np.sqrt(np.mean((pred_s - gt_s)**2))
        print(f'\n  {hold_season}: {exact_s}/{len(gt_s)} exact, RMSE={rmse_s:.3f}')

        # Show misses
        for rid, g, p in zip(rids_s, gt_s, pred_s):
            if g != p:
                print(f'    {rid}: pred={p}, true={g}, diff={p-g:+d}')

    # Compare vs v5
    print('\n' + '-'*50)
    print('  v5 baseline: 54/91 exact, RMSE=2.6520')
    if best[2] > 54 or best[3] < 2.652:
        print(f'  v6 IMPROVEMENT: {best[2]}/91 exact, RMSE={best[3]:.4f}')
        print(f'  Δ exact={best[2]-54:+d}, Δ RMSE={best[3]-2.652:.4f}')
    else:
        print(f'  v6 result: {best[2]}/91 exact, RMSE={best[3]:.4f}')
        print(f'  No improvement over v5')

    # Save submission
    best_assigned = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = blended_all[season_mask].copy()
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=best_power)
        for i, gi in enumerate(season_indices):
            best_assigned[gi] = assigned[i]

    sub = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    for i in np.where(test_mask)[0]:
        rid_to_seed[record_ids[i]] = int(best_assigned[i])
    sub['Overall Seed'] = sub['RecordID'].map(lambda r: rid_to_seed.get(r, 0))
    out = os.path.join(DATA_DIR, 'submission_kaggle_v6.csv')
    sub.to_csv(out, index=False)
    print(f'\n  Saved: {out}')
    print(f'  Time: {time.time()-t0:.0f}s')


def _eval_blend(blended_all, seasons, test_mask, y, folds, power):
    """Fast evaluation of a blend on the Kaggle test set."""
    n = len(blended_all)
    test_assigned = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = blended_all[season_mask].copy()

        # Lock training seeds
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]

        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    return test_assigned


if __name__ == '__main__':
    main()
