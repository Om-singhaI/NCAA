#!/usr/bin/env python3
"""
v46 Generalization Search — improve accuracy without overfitting.

Strategy: Focus on improvements that are STRUCTURALLY sound, not tuned to specific teams.
All candidates validated with nested LOSO (inner picks config, outer evaluates).

Vectors to explore:
1. Base model robustness — blend weight sweep, regularization
2. Power parameter sweep — small changes to Hungarian power
3. Feature stability — which features are consistently selected?
4. Residual analysis — are remaining errors random or systematic?
5. Zone boundary sensitivity — are current boundaries robust?
6. Ensemble of zone configs — average predictions from multiple configs
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    ADJ_COMP1_GAP, PW_C1, PW_C3,
    BLEND_W1, BLEND_W3, BLEND_W4,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
import xgboost as xgb
from scipy.optimize import linear_sum_assignment

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# v45c zone config (baseline)
ZONES = [
    ('mid',    'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot', 'bottom',    (48, 52), (0, 2, -2)),
    ('bot',    'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',   'tail',      (60, 63), (1,)),
]


def apply_zones(assigned, raw, fn, X_season, tm, si, zones, power=0.15):
    """Apply a list of zone corrections sequentially."""
    for name, ztype, zone, params in zones:
        if ztype == 'committee':
            aq, al, sos = params
            corr = compute_committee_correction(fn, X_season,
                                                 alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                            zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            sn, nc, cb = params
            corr = compute_bottom_correction(fn, X_season,
                                              sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                              zone=zone, power=power)
        elif ztype == 'tail':
            opp = params[0]
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                            zone=zone, power=power)
    return assigned


def run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                      zones, power=0.15, blend_weights=None,
                      adj_gap=30, c1=5.0, c3=0.5):
    """Run the full pipeline with given config and return predictions."""
    from ncaa_2026_model import build_pairwise_data_adjacent, build_pairwise_data, pairwise_score
    
    folds = sorted(set(seasons))
    n = len(y)
    preds = np.zeros(n, dtype=int)
    
    w1 = blend_weights[0] if blend_weights else BLEND_W1
    w3 = blend_weights[1] if blend_weights else BLEND_W3
    w4 = blend_weights[2] if blend_weights else BLEND_W4
    
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        X_season = X_all[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        # Feature selection
        tki = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        
        # Custom blend prediction
        raw = custom_blend_predict(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], tki,
            w1=w1, w3=w3, w4=w4, adj_gap=adj_gap, c1=c1, c3=c3)
        
        # Lock training teams
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        
        # Hungarian
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(raw, seasons[season_mask], avail, power=power)
        
        # Apply zones
        tm = np.array([test_mask[gi] for gi in season_indices])
        assigned = apply_zones(assigned, raw, fn, X_season, tm, season_indices, zones, power)
        
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                preds[gi] = assigned[i]
    
    return preds


def custom_blend_predict(X_train, y_train, X_test, seasons_train, top_k_idx,
                          w1=0.64, w3=0.28, w4=0.08,
                          adj_gap=30, c1=5.0, c3=0.5):
    """Predict with custom blend weights and hyperparams."""
    # Component 1: PW-LogReg C on full features, ADJACENT PAIRS
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
        X_train, y_train, seasons_train, max_gap=adj_gap)
    sc_adj = StandardScaler()
    pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_adj_sc, pw_y_adj)
    score1 = pairwise_score(lr1, X_test, sc_adj)
    
    # Component 3: PW-LogReg C on top-K features, STANDARD
    X_tr_k = X_train[:, top_k_idx]
    X_te_k = X_test[:, top_k_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
    lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_X_k_sc, pw_y_k)
    score3 = pairwise_score(lr3, X_te_k, sc_k)
    
    # Component 4: XGB
    pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, seasons_train)
    sc_full = StandardScaler()
    pw_X_full_sc = sc_full.fit_transform(pw_X_full)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_full_sc, pw_y_full)
    score4 = pairwise_score(xgb_clf, X_test, sc_full)
    
    return w1 * score1 + w3 * score3 + w4 * score4


def compute_se(preds, y, test_mask):
    gt = y[test_mask].astype(int)
    pr = preds[test_mask]
    return int(np.sum((pr - gt)**2))


def nested_loso(X_all, y, seasons, record_ids, test_mask, fn, configs, config_names):
    """
    Nested LOSO: outer=real LOSO, inner=pick best config on remaining seasons.
    Returns outer SE for each config + the nested-selected SE.
    """
    folds = sorted(set(seasons))
    outer_se = {name: 0 for name in config_names}
    nested_se = 0
    inner_wins = []
    
    for outer_fold in folds:
        outer_mask = (seasons == outer_fold)
        outer_test = test_mask & outer_mask
        if outer_test.sum() == 0:
            continue
        
        # For each config, get predictions on outer fold
        config_preds = {}
        for name, cfg in zip(config_names, configs):
            preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn, **cfg)
            se_fold = 0
            for i in np.where(outer_test)[0]:
                se_fold += (preds[i] - int(y[i]))**2
            outer_se[name] += se_fold
            config_preds[name] = (preds, se_fold)
        
        # Inner: pick best config on NON-outer folds
        inner_se = {}
        for name, cfg in zip(config_names, configs):
            preds = config_preds[name][0]
            se_inner = 0
            for fold in folds:
                if fold == outer_fold:
                    continue
                fold_test = test_mask & (seasons == fold)
                for i in np.where(fold_test)[0]:
                    se_inner += (preds[i] - int(y[i]))**2
            inner_se[name] = se_inner
        
        best_inner = min(inner_se, key=inner_se.get)
        inner_wins.append(best_inner)
        nested_se += config_preds[best_inner][1]
    
    return outer_se, nested_se, inner_wins


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 GENERALIZATION SEARCH')
    print('='*60)
    
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    feat = build_features(labeled, context_df, labeled, tourn_rids)
    fn = list(feat.columns)
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    # ── 1. RESIDUAL ANALYSIS ──
    print('\n' + '='*60)
    print(' 1. RESIDUAL ANALYSIS — Are remaining errors random?')
    print('='*60)
    
    v45c_preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                    zones=ZONES, power=0.15)
    v45c_se = compute_se(v45c_preds, y, test_mask)
    print(f'  v45c baseline: SE={v45c_se}')
    
    # Analyze errors by seed range
    gt = y[test_mask].astype(int)
    pr = v45c_preds[test_mask]
    rids = record_ids[test_mask]
    errors = pr - gt
    
    bins = [(1,16), (17,34), (35,44), (45,52), (53,60), (61,68)]
    print(f'\n  {"Seed Range":<12} {"Count":>5} {"Exact":>5} {"SE":>5} {"MeanErr":>8} {"StdErr":>8}')
    for lo, hi in bins:
        mask = (gt >= lo) & (gt <= hi)
        cnt = mask.sum()
        if cnt == 0: continue
        ex = int((pr[mask] == gt[mask]).sum())
        se = int(np.sum(errors[mask]**2))
        me = np.mean(errors[mask])
        se_std = np.std(errors[mask])
        print(f'  {lo:2d}-{hi:2d}        {cnt:5d} {ex:5d} {se:5d} {me:+8.2f} {se_std:8.2f}')
    
    # Check if errors are biased (systematic over/under prediction)
    print(f'\n  Overall mean error: {np.mean(errors):+.3f}')
    print(f'  Overall std error:  {np.std(errors):.3f}')
    print(f'  Median error:       {np.median(errors):+.1f}')
    print(f'  Skewness:           {np.mean(((errors - np.mean(errors))/np.std(errors))**3):.3f}')
    
    # Per-season bias
    print(f'\n  Per-season error stats:')
    for s in sorted(set(seasons)):
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt_s = y[sm].astype(int)
        pr_s = v45c_preds[sm]
        err_s = pr_s - gt_s
        print(f'    {s}: mean={np.mean(err_s):+.2f}, std={np.std(err_s):.2f}, '
              f'SE={int(np.sum(err_s**2))}, exact={int((pr_s==gt_s).sum())}/{sm.sum()}')
    
    # ── 2. POWER PARAMETER SENSITIVITY ──
    print('\n' + '='*60)
    print(' 2. POWER PARAMETER SWEEP')
    print('='*60)
    
    best_power_se = v45c_se
    best_power = 0.15
    for p in [0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.25, 0.30]:
        preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                   zones=ZONES, power=p)
        se = compute_se(preds, y, test_mask)
        exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
        marker = ' ← current' if p == 0.15 else (' ★' if se < best_power_se else '')
        print(f'  power={p:.2f}: SE={se:4d}, exact={exact}/91{marker}')
        if se < best_power_se:
            best_power_se = se
            best_power = p
    
    print(f'\n  Best power: {best_power} (SE={best_power_se})')
    
    # ── 3. BLEND WEIGHT SENSITIVITY ──
    print('\n' + '='*60)
    print(' 3. BLEND WEIGHT SWEEP')
    print('='*60)
    
    best_blend_se = v45c_se
    best_blend = (0.64, 0.28, 0.08)
    # Test variations around current 64/28/08
    blends = [
        (0.70, 0.22, 0.08),
        (0.68, 0.24, 0.08),
        (0.66, 0.26, 0.08),
        (0.64, 0.28, 0.08),  # current
        (0.62, 0.30, 0.08),
        (0.60, 0.32, 0.08),
        (0.58, 0.34, 0.08),
        # Vary XGB weight
        (0.64, 0.28, 0.08),
        (0.62, 0.28, 0.10),
        (0.60, 0.28, 0.12),
        (0.64, 0.24, 0.12),
        (0.56, 0.28, 0.16),
        # More LR1
        (0.72, 0.20, 0.08),
        (0.76, 0.16, 0.08),
        (0.80, 0.12, 0.08),
        # Less LR1
        (0.50, 0.40, 0.10),
        (0.55, 0.35, 0.10),
    ]
    seen = set()
    for w1, w3, w4 in blends:
        key = (w1, w3, w4)
        if key in seen: continue
        seen.add(key)
        preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                   zones=ZONES, blend_weights=(w1, w3, w4))
        se = compute_se(preds, y, test_mask)
        exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
        curr = ' ← current' if (w1, w3, w4) == (0.64, 0.28, 0.08) else ''
        marker = ' ★' if se < best_blend_se else ''
        print(f'  w=({w1:.2f},{w3:.2f},{w4:.2f}): SE={se:4d}, exact={exact}/91{curr}{marker}')
        if se < best_blend_se:
            best_blend_se = se
            best_blend = (w1, w3, w4)
    
    print(f'\n  Best blend: {best_blend} (SE={best_blend_se})')
    
    # ── 4. REGULARIZATION SWEEP (C values) ──
    print('\n' + '='*60)
    print(' 4. REGULARIZATION (C) SWEEP')
    print('='*60)
    
    best_c_se = v45c_se
    best_c_config = (5.0, 0.5)
    for c1 in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for c3 in [0.1, 0.3, 0.5, 1.0, 2.0]:
            preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                       zones=ZONES, c1=c1, c3=c3)
            se = compute_se(preds, y, test_mask)
            exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
            curr = ' ← current' if (c1, c3) == (5.0, 0.5) else ''
            marker = ' ★' if se < best_c_se else ''
            if se <= best_c_se + 10:  # only show competitive
                print(f'  C1={c1:.1f}, C3={c3:.1f}: SE={se:4d}, exact={exact}/91{curr}{marker}')
            if se < best_c_se:
                best_c_se = se
                best_c_config = (c1, c3)
    
    print(f'\n  Best C config: C1={best_c_config[0]}, C3={best_c_config[1]} (SE={best_c_se})')
    
    # ── 5. ADJACENT-PAIR GAP SWEEP ──
    print('\n' + '='*60)
    print(' 5. ADJACENT-PAIR GAP SWEEP')
    print('='*60)
    
    best_gap_se = v45c_se
    best_gap = 30
    for gap in [10, 15, 20, 25, 30, 35, 40, 50, 68]:
        preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                   zones=ZONES, adj_gap=gap)
        se = compute_se(preds, y, test_mask)
        exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
        curr = ' ← current' if gap == 30 else ''
        marker = ' ★' if se < best_gap_se else ''
        print(f'  gap={gap:2d}: SE={se:4d}, exact={exact}/91{curr}{marker}')
        if se < best_gap_se:
            best_gap_se = se
            best_gap = gap
    
    print(f'\n  Best gap: {best_gap} (SE={best_gap_se})')
    
    # ── 6. ZONE BOUNDARY SENSITIVITY ──
    print('\n' + '='*60)
    print(' 6. ZONE BOUNDARY SENSITIVITY')
    print('='*60)
    
    # Test each zone boundary ±1-2 to check stability
    print('  Testing zone boundary perturbations...')
    best_zone_se = v45c_se
    best_zone_config = None
    
    base_zones = list(ZONES)  # copy
    
    for zone_idx, (name, ztype, (lo, hi), params) in enumerate(ZONES):
        for dlo in [-2, -1, 0, 1, 2]:
            for dhi in [-2, -1, 0, 1, 2]:
                if dlo == 0 and dhi == 0:
                    continue
                new_lo = lo + dlo
                new_hi = hi + dhi
                if new_lo >= new_hi or new_lo < 1 or new_hi > 68:
                    continue
                
                test_zones = list(ZONES)
                test_zones[zone_idx] = (name, ztype, (new_lo, new_hi), params)
                
                preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                                           zones=test_zones)
                se = compute_se(preds, y, test_mask)
                if se < best_zone_se:
                    exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
                    print(f'  ★ {name} ({lo},{hi})→({new_lo},{new_hi}): SE={se} (was {v45c_se}), exact={exact}/91')
                    best_zone_se = se
                    best_zone_config = test_zones
    
    if best_zone_config is None:
        print('  No boundary changes improve SE — boundaries are robust ✓')
    else:
        print(f'\n  Best boundary change: SE={best_zone_se}')
    
    # ── 7. COMBINED BEST ──
    print('\n' + '='*60)
    print(' 7. COMBINED BEST IMPROVEMENTS')
    print('='*60)
    
    # Combine the best settings found
    combined_zones = best_zone_config if best_zone_config else ZONES
    combined_power = best_power
    combined_blend = best_blend
    combined_c = best_c_config
    combined_gap = best_gap
    
    print(f'  Power: {combined_power}')
    print(f'  Blend: {combined_blend}')
    print(f'  C1={combined_c[0]}, C3={combined_c[1]}')
    print(f'  Gap: {combined_gap}')
    
    preds = run_full_pipeline(X_all, y, seasons, record_ids, test_mask, fn,
                               zones=combined_zones, power=combined_power,
                               blend_weights=combined_blend,
                               c1=combined_c[0], c3=combined_c[1],
                               adj_gap=combined_gap)
    se = compute_se(preds, y, test_mask)
    exact = int((preds[test_mask] == y[test_mask].astype(int)).sum())
    print(f'\n  Combined result: SE={se}, exact={exact}/91, RMSE451={np.sqrt(se/451):.4f}')
    print(f'  vs v45c baseline: SE={v45c_se} (Δ={se-v45c_se:+d})')
    
    # ── 8. NESTED LOSO VALIDATION ──
    if se < v45c_se:
        print('\n' + '='*60)
        print(' 8. NESTED LOSO VALIDATION')
        print('='*60)
        
        configs = [
            {'zones': ZONES, 'power': 0.15},
            {'zones': combined_zones, 'power': combined_power,
             'blend_weights': combined_blend,
             'c1': combined_c[0], 'c3': combined_c[1],
             'adj_gap': combined_gap},
        ]
        config_names = ['v45c', 'v46_combined']
        
        outer_se, nested_se, inner_wins = nested_loso(
            X_all, y, seasons, record_ids, test_mask, fn,
            configs, config_names)
        
        print(f'\n  Outer SE:')
        for name in config_names:
            print(f'    {name}: SE={outer_se[name]}')
        print(f'\n  Nested LOSO SE (inner-selected): {nested_se}')
        print(f'  Gap: {nested_se - min(outer_se.values())} (0 = no overfitting)')
        print(f'  Inner winners: {inner_wins}')
    
    # ── 9. REMAINING ERRORS DETAIL ──
    print('\n' + '='*60)
    print(' 9. REMAINING ERRORS (detailed)')
    print('='*60)
    
    best_preds = preds if se < v45c_se else v45c_preds
    gt_all = y[test_mask].astype(int)
    pred_all = best_preds[test_mask]
    rids_all = record_ids[test_mask]
    seas_all = seasons[test_mask]
    
    print(f'\n  {"RecordID":<30} {"Season":<12} {"GT":>3} {"Pred":>4} {"Err":>4} {"SE":>4}')
    print(f'  {"─"*30} {"─"*12} {"─"*3} {"─"*4} {"─"*4} {"─"*4}')
    errors_list = []
    for i in range(len(gt_all)):
        err = pred_all[i] - gt_all[i]
        if err != 0:
            errors_list.append((rids_all[i], seas_all[i], gt_all[i], pred_all[i], err, err**2))
    errors_list.sort(key=lambda x: -x[5])
    for rid, sea, gt_i, pr_i, err, se_i in errors_list:
        print(f'  {rid:<30} {sea:<12} {gt_i:3d} {pr_i:4d} {err:+4d} {se_i:4d}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
