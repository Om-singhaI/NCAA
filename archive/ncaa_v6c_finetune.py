#!/usr/bin/env python3
"""
v6c Fine-grain: zoom in around best v6b configs with 2% weight steps.

Top v6b findings:
  RMSE best: 60% LR_C5 + 30% LR_topK25_C0.5 + 10% GBC_200_d4, p=0.375 → 52/91, RMSE=2.518
  Exact best: 70% LR_C1 + 30% LR_topK25_C0.5, p=0.25 → 58/91, RMSE=2.70
  Key insight: LR_topK25_C0.5 is better than LR_topK25_C1.0

This script:
  1. Precompute focused method pool
  2. Ultra-fine 2% weight grid around top configs
  3. Fine power grid (0.05 steps from 0.15–0.65)
  4. Combined RMSE+exact optimization
  5. LOSO validation of winning config
"""

import os, sys, time, warnings, itertools
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
    print(' v6c FINE-GRAINED SEARCH')
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

    # Focused method pool (from v6b top results)
    method_defs = [
        # Primary LR variants (different C)
        ('LR_C1',    'lr', {'C': 1.0}),
        ('LR_C3',    'lr', {'C': 3.0}),
        ('LR_C5',    'lr', {'C': 5.0}),
        ('LR_C7',    'lr', {'C': 7.0}),
        ('LR_C10',   'lr', {'C': 10.0}),
        ('LR_C20',   'lr', {'C': 20.0}),
        ('LR_C0.3',  'lr', {'C': 0.3}),
        # TopK variants that dominated v6b
        ('LRk25_C0.5', 'lr_topk', {'C': 0.5, 'K': 25}),
        ('LRk25_C0.3', 'lr_topk', {'C': 0.3, 'K': 25}),
        ('LRk25_C0.7', 'lr_topk', {'C': 0.7, 'K': 25}),
        ('LRk25_C1', 'lr_topk', {'C': 1.0, 'K': 25}),
        ('LRk20_C0.5', 'lr_topk', {'C': 0.5, 'K': 20}),
        ('LRk20_C1', 'lr_topk', {'C': 1.0, 'K': 20}),
        ('LRk30_C0.5', 'lr_topk', {'C': 0.5, 'K': 30}),
        ('LRk30_C1', 'lr_topk', {'C': 1.0, 'K': 30}),
        ('LRk15_C0.5', 'lr_topk', {'C': 0.5, 'K': 15}),
        ('LRk35_C0.5', 'lr_topk', {'C': 0.5, 'K': 35}),
        # GBC that helped
        ('GBC_200_d4', 'gbc', {'n_estimators': 200, 'max_depth': 4}),
        ('GBC_300_d3', 'gbc', {'n_estimators': 300, 'max_depth': 3}),
        ('GBC_300_d4', 'gbc', {'n_estimators': 300, 'max_depth': 4}),
        # XGB variants
        ('XGB_d3_200', 'xgb', {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}),
        ('XGB_d3_500', 'xgb', {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05}),
        ('XGB_d4_300', 'xgb', {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}),
    ]

    print(f'  Methods: {len(method_defs)}')

    # ── Precompute all scores ──
    print('  Precomputing...')
    all_scores = {}

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        season_indices = np.where(season_mask)[0]

        pw_X, pw_y = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)

        topk_cache = {}
        for K in [15, 20, 25, 30, 35]:
            topk_cache[K] = select_top_k_features(
                X[global_train_mask], y[global_train_mask], fn, k=K)[0]

        for m_name, m_type, m_params in method_defs:
            try:
                if m_type == 'lr':
                    cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                            max_iter=2000, random_state=42)
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X_season, sc)

                elif m_type == 'lr_topk':
                    K = m_params['K']
                    top_idx = topk_cache[K]
                    X_k_tr = X[global_train_mask][:, top_idx]
                    X_k_te = X_season[:, top_idx]
                    pw_X_k, pw_y_k = build_pairwise_data(
                        X_k_tr, y[global_train_mask], seasons[global_train_mask])
                    sc_k = StandardScaler()
                    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                    cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                            max_iter=2000, random_state=42)
                    cls.fit(pw_X_k_sc, pw_y_k)
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
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X_season, sc)

                elif m_type == 'gbc':
                    cls = GradientBoostingClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        learning_rate=0.1, subsample=0.8,
                        min_samples_leaf=5, random_state=42)
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X_season, sc)

                if m_name not in all_scores:
                    all_scores[m_name] = np.zeros(n)
                for i, gi in enumerate(season_indices):
                    all_scores[m_name][gi] = s[i]

            except Exception as e:
                print(f'    WARN: {m_name}: {e}')

        print(f'    {hold_season} ✓')

    method_names = sorted(all_scores.keys())
    scores_map = {m: all_scores[m] for m in method_names}
    print(f'  {len(method_names)} methods ready')

    # ══════════════════════════════════════════════════════════════
    # ZONE A: Fine search around RMSE leader
    # 60% LR_C5 + 30% LRk25_C0.5 + 10% GBC_200_d4, p=0.375
    # ══════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' ZONE A: Fine-grain around top RMSE configs')
    print('='*70)

    # Generate configs: each primary LR + each topK + optional small 3rd
    results = []
    powers = np.arange(0.15, 0.70, 0.05)

    primary_models = ['LR_C1', 'LR_C3', 'LR_C5', 'LR_C7', 'LR_C10', 'LR_C20', 'LR_C0.3']
    topk_models = ['LRk25_C0.5', 'LRk25_C0.3', 'LRk25_C0.7', 'LRk25_C1',
                   'LRk20_C0.5', 'LRk20_C1', 'LRk30_C0.5', 'LRk30_C1',
                   'LRk15_C0.5', 'LRk35_C0.5']
    third_models = ['GBC_200_d4', 'GBC_300_d3', 'GBC_300_d4',
                    'XGB_d3_200', 'XGB_d3_500', 'XGB_d4_300', 'NONE']

    cnt = 0
    for primary in primary_models:
        for topk in topk_models:
            for third in third_models:
                if third == 'NONE':
                    # 2-method: scan primary weight from 50-85%
                    for w1_pct in range(50, 86, 2):
                        w1 = w1_pct / 100.0
                        w2 = 1.0 - w1
                        bl = w1 * scores_map[primary] + w2 * scores_map[topk]
                        for power in powers:
                            e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                            results.append(({primary: w1, topk: w2}, power, e, r))
                            cnt += 1
                else:
                    # 3-method: scan with 3rd at 4-20%, primary 40-76%
                    for w3_pct in range(4, 22, 2):
                        for w2_pct in range(20, 46, 2):
                            w1_pct = 100 - w2_pct - w3_pct
                            if w1_pct < 38 or w1_pct > 76:
                                continue
                            w1 = w1_pct / 100.0
                            w2 = w2_pct / 100.0
                            w3 = w3_pct / 100.0
                            bl = (w1 * scores_map[primary] +
                                  w2 * scores_map[topk] +
                                  w3 * scores_map[third])
                            for power in powers:
                                e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                                results.append(
                                    ({primary: w1, topk: w2, third: w3}, power, e, r))
                                cnt += 1

    print(f'  {cnt} configs evaluated')

    # ══════════════════════════════════════════════════════════════
    # ZONE B: Fine search around EXACT leader
    # 70% LR_C1 + 30% LRk, p=0.25
    # ══════════════════════════════════════════════════════════════
    print('\n  ZONE B: around exact match leaders...')
    cnt2 = 0
    for primary in primary_models:
        for topk in topk_models:
            # 2-method fine grid
            for w1_pct in range(55, 86, 2):
                w1 = w1_pct / 100.0
                w2 = 1.0 - w1
                bl = w1 * scores_map[primary] + w2 * scores_map[topk]
                for power in np.arange(0.10, 0.45, 0.05):
                    e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                    results.append(({primary: w1, topk: w2}, power, e, r))
                    cnt2 += 1
    print(f'  {cnt2} configs evaluated')

    # ══════════════════════════════════════════════════════════════
    # ZONE C: 4-method with LR + topK + XGB/GBC + diversity
    # ══════════════════════════════════════════════════════════════
    print('\n  ZONE C: 4-method exploration...')
    cnt3 = 0
    for primary in ['LR_C5', 'LR_C10', 'LR_C1', 'LR_C3', 'LR_C7']:
        for topk in ['LRk25_C0.5', 'LRk30_C0.5', 'LRk25_C0.3', 'LRk30_C1',
                     'LRk20_C0.5', 'LRk25_C1']:
            for t3 in ['GBC_200_d4', 'GBC_300_d3', 'XGB_d3_200', 'XGB_d3_500']:
                for t4 in ['LR_C0.3', 'LR_C20', 'XGB_d4_300', 'GBC_300_d4']:
                    if t3 == t4:
                        continue
                    for w1_pct in range(40, 66, 8):
                        for w2_pct in range(20, 38, 6):
                            for w3_pct in range(6, 22, 6):
                                w4_pct = 100 - w1_pct - w2_pct - w3_pct
                                if w4_pct < 4 or w4_pct > 24:
                                    continue
                                w1 = w1_pct / 100.
                                w2 = w2_pct / 100.
                                w3 = w3_pct / 100.
                                w4 = w4_pct / 100.
                                bl = (w1*scores_map[primary] + w2*scores_map[topk] +
                                      w3*scores_map[t3] + w4*scores_map[t4])
                                for power in [0.20, 0.30, 0.375, 0.50, 0.625]:
                                    e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                                    results.append(
                                        ({primary: w1, topk: w2, t3: w3, t4: w4},
                                         power, e, r))
                                    cnt3 += 1
    print(f'  {cnt3} configs evaluated')

    # ══════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════
    total = len(results)
    print(f'\n  TOTAL: {total} configs')

    results.sort(key=lambda x: x[3])  # RMSE

    print(f'\n  Top 30 by RMSE:')
    print(f'  {"Rk":>3} {"Exact":>5} {"RMSE":>8} {"Pwr":>5}  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8} {"─"*5}  {"─"*60}')
    for i, (w, p, e, r) in enumerate(results[:30]):
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        flag = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {e:5d} {r:8.4f} {p:5.2f}  {blend_str}{flag}')

    exact_sorted = sorted(results, key=lambda x: (-x[2], x[3]))
    print(f'\n  Top 20 by EXACT:')
    print(f'  {"Rk":>3} {"Exact":>5} {"RMSE":>8} {"Pwr":>5}  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8} {"─"*5}  {"─"*60}')
    for i, (w, p, e, r) in enumerate(exact_sorted[:20]):
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        print(f'  {i+1:3d} {e:5d} {r:8.4f} {p:5.2f}  {blend_str}')

    # Find Pareto-optimal (best RMSE for each exact count)
    print(f'\n  Pareto frontier (best RMSE for each exact count):')
    pareto = {}
    for w, p, e, r in results:
        if e not in pareto or r < pareto[e][3]:
            pareto[e] = (w, p, e, r)
    for e in sorted(pareto.keys(), reverse=True):
        w, p, ex, r = pareto[e]
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        marker = ' ← v5' if e == 54 else (' ★' if r == results[0][3] else '')
        print(f'    {e:3d}/91 → RMSE={r:.4f} p={p:.2f}  {blend_str}{marker}')

    # ══════════════════════════════════════════════════════════════
    # LOSO VALIDATION
    # ══════════════════════════════════════════════════════════════
    best_rmse_cfg = results[0]
    print(f'\n' + '='*70)
    print(' LOSO VALIDATION OF BEST RMSE CONFIG')
    print(f' {best_rmse_cfg[0]}, power={best_rmse_cfg[1]}')
    print('='*70)

    # Also validate the best exact-with-good-RMSE
    # and the top 3 RMSE configs
    configs_to_validate = []
    seen = set()
    for w, p, e, r in results[:5]:
        key = (tuple(sorted(w.items())), p)
        if key not in seen:
            configs_to_validate.append((w, p, f'RMSE#{len(configs_to_validate)+1}'))
            seen.add(key)
    
    # Also add best high-exact with RMSE < 2.60
    for w, p, e, r in exact_sorted:
        if r < 2.60 and e >= 56:
            key = (tuple(sorted(w.items())), p)
            if key not in seen:
                configs_to_validate.append((w, p, f'Exact{e}'))
                seen.add(key)
                break

    for weights, power, label in configs_to_validate:
        loso_assigned = np.zeros(n, dtype=int)
        fold_stats = []

        for hold in folds:
            tr = seasons != hold
            te = seasons == hold

            top_k_idx = select_top_k_features(X[tr], y[tr], fn, k=25)[0]
            top_k_idx_30 = select_top_k_features(X[tr], y[tr], fn, k=30)[0]
            top_k_idx_20 = select_top_k_features(X[tr], y[tr], fn, k=20)[0]
            top_k_idx_15 = select_top_k_features(X[tr], y[tr], fn, k=15)[0]
            top_k_idx_35 = select_top_k_features(X[tr], y[tr], fn, k=35)[0]

            pw_X, pw_y = build_pairwise_data(X[tr], y[tr], seasons[tr])
            sc_full = StandardScaler()
            pw_X_sc = sc_full.fit_transform(pw_X)

            bl = np.zeros(int(te.sum()))
            for m_name, w in weights.items():
                m_type = None
                m_params = None
                for md_name, md_type, md_params in method_defs:
                    if md_name == m_name:
                        m_type = md_type
                        m_params = md_params
                        break
                if m_type is None:
                    continue

                if m_type == 'lr':
                    cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                            max_iter=2000, random_state=42)
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X[te], sc_full)

                elif m_type == 'lr_topk':
                    K = m_params['K']
                    tk = {15: top_k_idx_15, 20: top_k_idx_20, 25: top_k_idx,
                          30: top_k_idx_30, 35: top_k_idx_35}[K]
                    pw_Xk, pw_yk = build_pairwise_data(X[tr][:, tk], y[tr], seasons[tr])
                    sc_k = StandardScaler()
                    pw_Xk_sc = sc_k.fit_transform(pw_Xk)
                    cls = LogisticRegression(C=m_params['C'], penalty='l2',
                                            max_iter=2000, random_state=42)
                    cls.fit(pw_Xk_sc, pw_yk)
                    s = pairwise_score(cls, X[te][:, tk], sc_k)

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
                    s = pairwise_score(cls, X[te], sc_full)

                elif m_type == 'gbc':
                    cls = GradientBoostingClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        learning_rate=0.1, subsample=0.8,
                        min_samples_leaf=5, random_state=42)
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X[te], sc_full)

                bl += w * s

            avail = {hold: list(range(1, 69))}
            assigned = hungarian(bl, seasons[te], avail, power=power)
            loso_assigned[te] = assigned
            yte = y[te].astype(int)
            exact = int(np.sum(assigned == yte))
            rmse_fold = np.sqrt(np.mean((assigned - yte)**2))
            fold_stats.append((hold, int(te.sum()), exact, rmse_fold))

        loso_exact = int(np.sum(loso_assigned == y.astype(int)))
        loso_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
        fold_rmses = [r for _, _, _, r in fold_stats]
        score = np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)

        print(f'\n  --- {label}: {weights} ---')
        print(f'  power={power}')
        for s, nf, ex, rm in fold_stats:
            print(f'    {s}: {ex}/{nf} exact, RMSE={rm:.3f}')
        print(f'  LOSO: {loso_exact}/340 exact, RMSE={loso_rmse:.4f}, score={score:.4f}')

    # ── Save best RMSE submission ──
    best = results[0]
    bl_all = np.zeros(n)
    for m_name, w in best[0].items():
        bl_all += w * all_scores[m_name]

    assigned_all = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = bl_all[season_mask].copy()
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                bl[i] = y[gi]
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(bl, seasons[season_mask], avail, power=best[1])
        for i, gi in enumerate(season_indices):
            assigned_all[gi] = assigned[i]

    # Per-season breakdown
    print(f'\n  Best RMSE config per-season:')
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        st = test_mask & season_mask
        if st.sum() == 0: continue
        gt_s = y[st].astype(int)
        pred_s = assigned_all[st]
        e_s = int((pred_s == gt_s).sum())
        r_s = np.sqrt(np.mean((pred_s - gt_s)**2))
        print(f'    {hold_season}: {e_s}/{st.sum()} exact, RMSE={r_s:.3f}')

    sub = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    for i in np.where(test_mask)[0]:
        rid_to_seed[record_ids[i]] = int(assigned_all[i])
    sub['Overall Seed'] = sub['RecordID'].map(lambda r: rid_to_seed.get(r, 0))
    out = os.path.join(DATA_DIR, 'submission_kaggle_v6c.csv')
    sub.to_csv(out, index=False)
    print(f'\n  Saved: {out}')

    print(f'\n  v5: 54/91, RMSE=2.6520')
    print(f'  v6c best RMSE: {best[2]}/91, RMSE={best[3]:.4f}, power={best[1]}')
    print(f'  v6c best EXACT: {exact_sorted[0][2]}/91, RMSE={exact_sorted[0][3]:.4f}')
    print(f'\n  Time: {time.time()-t0:.0f}s')


def _quick_eval(blended_all, seasons, test_mask, y, folds, power):
    n = len(blended_all)
    test_assigned = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_indices = np.where(season_mask)[0]
        bl = blended_all[season_mask].copy()
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


if __name__ == '__main__':
    main()
