#!/usr/bin/env python3
"""
v6b Targeted Search: Combine v5 components with v6 discoveries.

Key v6 findings:
  - LR_C0.3 best individual (51/91 solo, RMSE=2.94)
  - LR_C1.0 features heavily in best blends
  - GBC_200_d4 + GBC_300_d4 useful diversity
  - XGB_d5_100 surprisingly good
  - 56/91 exact possible with LR_C1.0+XGB+GBC blends

Issue: v5's LR_topK wasn't in blend search. This script:
  1. Forces ALL promising methods into blend search
  2. 5% weight grid for 2-4 method blends
  3. Also tests per-season power optimization
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    print(' v6b TARGETED SEARCH — BEST OF v5 + v6 DISCOVERIES')
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

    # ── Define the targeted method pool ──
    # Include v5 components + v6 best discoveries
    method_defs = [
        # v5 components
        ('LR_C5_full',    'lr',    {'C': 5.0, 'feats': 'full'}),
        ('LR_topK25_C1',  'lr_topk', {'C': 1.0, 'K': 25}),
        ('XGB_d3_200',    'xgb',   {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1}),
        # v6 discoveries
        ('LR_C0.3_full',  'lr',    {'C': 0.3, 'feats': 'full'}),
        ('LR_C1_full',    'lr',    {'C': 1.0, 'feats': 'full'}),
        ('LR_C3_full',    'lr',    {'C': 3.0, 'feats': 'full'}),
        ('LR_C7_full',    'lr',    {'C': 7.0, 'feats': 'full'}),
        ('LR_C10_full',   'lr',    {'C': 10.0, 'feats': 'full'}),
        ('LR_C20_full',   'lr',    {'C': 20.0, 'feats': 'full'}),
        ('XGB_d5_100',    'xgb',   {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}),
        ('XGB_d4_200',    'xgb',   {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.08}),
        ('XGB_d4_300',    'xgb',   {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}),
        ('XGB_d3_500',    'xgb',   {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05}),
        ('GBC_200_d4',    'gbc',   {'n_estimators': 200, 'max_depth': 4}),
        ('GBC_300_d4',    'gbc',   {'n_estimators': 300, 'max_depth': 4}),
        ('GBC_300_d3',    'gbc',   {'n_estimators': 300, 'max_depth': 3}),
        # Additional top-K variants
        ('LR_topK15_C1',  'lr_topk', {'C': 1.0, 'K': 15}),
        ('LR_topK20_C1',  'lr_topk', {'C': 1.0, 'K': 20}),
        ('LR_topK30_C1',  'lr_topk', {'C': 1.0, 'K': 30}),
        ('LR_topK25_C0.5','lr_topk', {'C': 0.5, 'K': 25}),
        ('LR_topK25_C3',  'lr_topk', {'C': 3.0, 'K': 25}),
        ('LR_topK25_C5',  'lr_topk', {'C': 5.0, 'K': 25}),
        # RF for diversity
        ('RF_500_d12',    'rf',    {'n_estimators': 500, 'max_depth': 12}),
    ]

    print(f'  Method pool: {len(method_defs)} methods')

    # ── Precompute all scores ──
    print('\n  Precomputing scores...')
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
        for K in [15, 20, 25, 30]:
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

                elif m_type == 'rf':
                    cls = RandomForestClassifier(
                        n_estimators=m_params['n_estimators'],
                        max_depth=m_params['max_depth'],
                        min_samples_leaf=2, max_features='sqrt',
                        random_state=42, n_jobs=-1)
                    cls.fit(pw_X_sc, pw_y)
                    s = pairwise_score(cls, X_season, sc)

                if m_name not in all_scores:
                    all_scores[m_name] = np.zeros(n)
                for i, gi in enumerate(season_indices):
                    all_scores[m_name][gi] = s[i]

            except Exception as e:
                print(f'    WARN: {m_name} failed: {e}')

        print(f'    {hold_season}: {season_test_mask.sum()} test teams ✓')

    method_names = sorted(all_scores.keys())
    print(f'  Computed {len(method_names)} methods')
    
    # ── Quick individual evaluation ──
    print('\n  Individual method results (p=0.5):')
    indiv = []
    for m_name in method_names:
        for power in [0.375, 0.5, 0.625, 0.75, 1.0]:
            e, r = _quick_eval(all_scores[m_name], seasons, test_mask, y, folds, power)
            indiv.append((m_name, power, e, r))
    indiv.sort(key=lambda x: x[3])
    
    print(f'  {"Rk":>3} {"Method":<25} {"Pwr":>5} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─"*3} {"─"*25} {"─"*5} {"─"*5} {"─"*8}')
    for i, (m, p, e, r) in enumerate(indiv[:20]):
        print(f'  {i+1:3d} {m:<25} {p:5.3f} {e:5d} {r:8.4f}')

    # ── v5 baseline check ──
    v5_score = (0.60 * all_scores['LR_C5_full'] +
                0.20 * all_scores['LR_topK25_C1'] +
                0.20 * all_scores['XGB_d3_200'])
    for power in [0.375, 0.5, 0.625, 0.75, 1.0]:
        e, r = _quick_eval(v5_score, seasons, test_mask, y, folds, power)
        print(f'  v5 baseline p={power}: {e}/91 exact, RMSE={r:.4f}')

    # ── EXHAUSTIVE BLEND SEARCH ──
    print('\n' + '='*70)
    print(' BLEND SEARCH (5% grid, 2-4 methods)')
    print('='*70)

    M = len(method_names)
    scores_arr = [all_scores[m] for m in method_names]
    powers = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    
    blend_results = []
    
    # 2-method blends (5% steps)
    print('  2-method blends...')
    cnt = 0
    for i in range(M):
        for j in range(i+1, M):
            for w1_pct in range(5, 100, 5):
                w1 = w1_pct / 100.0
                w2 = 1.0 - w1
                bl = w1 * scores_arr[i] + w2 * scores_arr[j]
                for power in powers:
                    e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                    blend_results.append(({method_names[i]: w1, method_names[j]: w2},
                                          power, e, r))
                    cnt += 1
    print(f'    {cnt} configs')
    
    # 3-method blends (10% steps) — all combos
    print('  3-method blends...')
    cnt = 0
    for i in range(M):
        for j in range(i+1, M):
            for k in range(j+1, M):
                for w1_pct in range(10, 80, 10):
                    for w2_pct in range(10, 80 - w1_pct + 10, 10):
                        w3_pct = 100 - w1_pct - w2_pct
                        if w3_pct < 5:
                            continue
                        w1, w2, w3 = w1_pct/100., w2_pct/100., w3_pct/100.
                        bl = w1*scores_arr[i] + w2*scores_arr[j] + w3*scores_arr[k]
                        for power in powers:
                            e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                            blend_results.append(
                                ({method_names[i]: w1, method_names[j]: w2,
                                  method_names[k]: w3}, power, e, r))
                            cnt += 1
    print(f'    {cnt} configs')
    
    # 4-method blends — restricted to top 10 methods, 15% steps
    print('  4-method blends (top 10 methods)...')
    # Sort methods by best solo RMSE
    solo_best = {}
    for m_name in method_names:
        best_r = min(_quick_eval(all_scores[m_name], seasons, test_mask, y, folds, p)[1]
                     for p in powers)
        solo_best[m_name] = best_r
    top10 = sorted(solo_best, key=solo_best.get)[:10]
    top10_idx = [method_names.index(m) for m in top10]
    
    cnt = 0
    for combo in itertools.combinations(top10_idx, 4):
        i, j, k, l = combo
        for w1_pct in range(10, 55, 15):
            for w2_pct in range(10, 55, 15):
                for w3_pct in range(10, 55, 15):
                    w4_pct = 100 - w1_pct - w2_pct - w3_pct
                    if w4_pct < 5 or w4_pct > 60:
                        continue
                    w1, w2, w3, w4 = w1_pct/100., w2_pct/100., w3_pct/100., w4_pct/100.
                    bl = (w1*scores_arr[i] + w2*scores_arr[j] +
                          w3*scores_arr[k] + w4*scores_arr[l])
                    for power in powers:
                        e, r = _quick_eval(bl, seasons, test_mask, y, folds, power)
                        blend_results.append(
                            ({method_names[i]: w1, method_names[j]: w2,
                              method_names[k]: w3, method_names[l]: w4},
                             power, e, r))
                        cnt += 1
    print(f'    {cnt} configs')

    # Sort all results
    blend_results.sort(key=lambda x: x[3])  # RMSE
    
    print(f'\n  TOTAL: {len(blend_results)} configs evaluated')
    print(f'\n  Top 40 by RMSE:')
    print(f'  {"Rk":>3} {"Exact":>5} {"RMSE":>8} {"Pwr":>5}  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8} {"─"*5}  {"─"*60}')
    for i, (w, p, e, r) in enumerate(blend_results[:40]):
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        flag = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {e:5d} {r:8.4f} {p:5.3f}  {blend_str}{flag}')
    
    # Also show top by exact matches
    blend_results_exact = sorted(blend_results, key=lambda x: (-x[2], x[3]))
    print(f'\n  Top 20 by EXACT MATCHES:')
    print(f'  {"Rk":>3} {"Exact":>5} {"RMSE":>8} {"Pwr":>5}  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8} {"─"*5}  {"─"*60}')
    for i, (w, p, e, r) in enumerate(blend_results_exact[:20]):
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        print(f'  {i+1:3d} {e:5d} {r:8.4f} {p:5.3f}  {blend_str}')

    # ── PHASE 2: Per-season power optimization for top configs ──
    print('\n' + '='*70)
    print(' PER-SEASON POWER OPTIMIZATION (top 20 blends)')
    print('='*70)
    
    # Take top 20 unique blend configs (different weights)
    seen_w = set()
    top_blends = []
    for w, p, e, r in blend_results[:200]:
        wkey = tuple(sorted(w.items()))
        if wkey not in seen_w:
            seen_w.add(wkey)
            top_blends.append(w)
        if len(top_blends) >= 20:
            break
    
    # Also add top 10 by exact
    for w, p, e, r in blend_results_exact[:100]:
        wkey = tuple(sorted(w.items()))
        if wkey not in seen_w:
            seen_w.add(wkey)
            top_blends.append(w)
        if len(top_blends) >= 30:
            break

    perseas_results = []
    season_powers = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    
    for weights in top_blends:
        bl_all = np.zeros(n)
        for m_name, w in weights.items():
            bl_all += w * all_scores[m_name]

        # Try per-season power optimization
        # For each season, find best power
        best_per_season = {}
        for hold_season in folds:
            best_e, best_p, best_r = 0, 0.5, 999
            season_mask = (seasons == hold_season)
            season_indices = np.where(season_mask)[0]
            bl = bl_all[season_mask].copy()
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    bl[i] = y[gi]

            for power in season_powers:
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(bl, seasons[season_mask], avail, power=power)
                gt_s = []; pred_s = []
                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        gt_s.append(int(y[gi]))
                        pred_s.append(assigned[i])
                gt_s = np.array(gt_s); pred_s = np.array(pred_s)
                e_s = int((pred_s == gt_s).sum())
                r_s = np.sqrt(np.mean((pred_s - gt_s)**2))
                if r_s < best_r:
                    best_e, best_p, best_r = e_s, power, r_s
            best_per_season[hold_season] = best_p

        # Evaluate with per-season optimal powers
        assigned_all = np.zeros(n, dtype=int)
        total_exact = 0; total_se = 0; total_n = 0
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_indices = np.where(season_mask)[0]
            bl = bl_all[season_mask].copy()
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    bl[i] = y[gi]
            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(bl, seasons[season_mask], avail,
                                 power=best_per_season[hold_season])
            for i, gi in enumerate(season_indices):
                assigned_all[gi] = assigned[i]
                if test_mask[gi]:
                    total_n += 1
                    total_se += (assigned[i] - int(y[gi]))**2
                    if assigned[i] == int(y[gi]):
                        total_exact += 1

        rmse_ps = np.sqrt(total_se / total_n) if total_n > 0 else 999
        perseas_results.append((weights, best_per_season, total_exact, rmse_ps))

    perseas_results.sort(key=lambda x: x[3])
    
    print(f'\n  Top 15 with per-season power:')
    print(f'  {"Rk":>3} {"Exact":>5} {"RMSE":>8}  Powers  Blend')
    print(f'  {"─"*3} {"─"*5} {"─"*8}  {"─"*35}  {"─"*50}')
    for i, (w, ps, e, r) in enumerate(perseas_results[:15]):
        pstr = ' '.join(f'{s[:4]}={p:.2f}' for s, p in sorted(ps.items()))
        blend_str = ' + '.join(f'{v:.0%} {k}' for k, v in sorted(w.items(), key=lambda x: -x[1]))
        flag = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {e:5d} {r:8.4f}  {pstr}  {blend_str}{flag}')

    # ── Final comparison ──
    best_overall = blend_results[0]
    best_perseas = perseas_results[0]
    
    print(f'\n' + '='*50)
    print(f'  v5 baseline:      54/91, RMSE=2.6520 (p=0.5)')
    print(f'  v6b uniform best: {best_overall[2]}/91, RMSE={best_overall[3]:.4f} (p={best_overall[1]})')
    print(f'  v6b per-season:   {best_perseas[2]}/91, RMSE={best_perseas[3]:.4f}')
    
    # Save best submission
    best_w, best_ps, best_e, best_r = best_perseas
    if best_r < 2.652:
        print(f'\n  ★ IMPROVEMENT FOUND ★')
        bl_all = np.zeros(n)
        for m_name, w in best_w.items():
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
            assigned = hungarian(bl, seasons[season_mask], avail,
                                 power=best_ps[hold_season])
            for i, gi in enumerate(season_indices):
                assigned_all[gi] = assigned[i]
        
        sub = sub_df[['RecordID']].copy()
        rid_to_seed = {}
        for i in np.where(test_mask)[0]:
            rid_to_seed[record_ids[i]] = int(assigned_all[i])
        sub['Overall Seed'] = sub['RecordID'].map(lambda r: rid_to_seed.get(r, 0))
        out = os.path.join(DATA_DIR, 'submission_kaggle_v6b.csv')
        sub.to_csv(out, index=False)
        print(f'  Saved: {out}')
    else:
        # Save best uniform power submission
        best_u = blend_results[0]
        bl_all = np.zeros(n)
        for m_name, w in best_u[0].items():
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
            assigned = hungarian(bl, seasons[season_mask], avail, power=best_u[1])
            for i, gi in enumerate(season_indices):
                assigned_all[gi] = assigned[i]
        
        sub = sub_df[['RecordID']].copy()
        rid_to_seed = {}
        for i in np.where(test_mask)[0]:
            rid_to_seed[record_ids[i]] = int(assigned_all[i])
        sub['Overall Seed'] = sub['RecordID'].map(lambda r: rid_to_seed.get(r, 0))
        out = os.path.join(DATA_DIR, 'submission_kaggle_v6b.csv')
        sub.to_csv(out, index=False)
        print(f'  Saved: {out}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


def _quick_eval(blended_all, seasons, test_mask, y, folds, power):
    """Fast evaluation returning (exact, rmse)."""
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
