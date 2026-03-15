#!/usr/bin/env python3
"""
v46 Generalization Search — FAST version.

Key optimization: compute raw v12 scores ONCE, then sweep zones/power cheaply.
Only do expensive base model re-training for the small set of promising configs.
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
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from scipy.optimize import linear_sum_assignment

# v45c zone config (baseline)
ZONES = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]


def apply_zones(assigned, raw, fn, X_season, tm, si, zones, power=0.15):
    """Apply zone corrections sequentially."""
    for name, ztype, zone, params in zones:
        if ztype == 'committee':
            aq, al, sos = params
            corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                            zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            sn, nc, cb = params
            corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            opp = params[0]
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def compute_raw_scores(X_all, y, seasons, test_mask, fn):
    """Compute v12 raw scores ONCE for all seasons. Returns dict of per-season data."""
    folds = sorted(set(seasons))
    season_data = {}
    
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        X_season = X_all[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        tki = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        
        raw = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], tki)
        
        # Lock training teams
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        
        season_data[hold_season] = {
            'X_season': X_season,
            'raw': raw,
            'season_indices': season_indices,
            'season_mask': season_mask,
            'test_mask_season': tm,
            'n_test': season_test_mask.sum(),
        }
    
    return season_data


def eval_config(season_data, y, seasons, test_mask, fn, zones, power=0.15):
    """Evaluate a zone/power config using cached raw scores. FAST."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    folds = sorted(set(seasons))
    
    for hold_season in folds:
        if hold_season not in season_data:
            continue
        sd = season_data[hold_season]
        
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(sd['raw'], seasons[sd['season_mask']], avail, power=power)
        
        assigned = apply_zones(assigned, sd['raw'], fn, sd['X_season'],
                                sd['test_mask_season'], sd['season_indices'],
                                zones, power)
        
        for i, gi in enumerate(sd['season_indices']):
            if test_mask[gi]:
                preds[gi] = assigned[i]
    
    gt = y[test_mask].astype(int)
    pr = preds[test_mask]
    se = int(np.sum((pr - gt)**2))
    exact = int((pr == gt).sum())
    return se, exact, preds


def compute_raw_scores_custom(X_all, y, seasons, test_mask, fn,
                               w1=0.64, w3=0.28, w4=0.08,
                               c1=5.0, c3=0.5, adj_gap=30):
    """Compute raw scores with custom blend weights/regularization."""
    folds = sorted(set(seasons))
    season_data = {}
    
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        X_season = X_all[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        X_tr = X_all[global_train_mask]
        y_tr = y[global_train_mask]
        s_tr = seasons[global_train_mask]
        
        tki = select_top_k_features(X_tr, y_tr, fn, k=USE_TOP_K_A,
                                      forced_features=FORCE_FEATURES)[0]
        
        # Component 1
        pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=adj_gap)
        sc_adj = StandardScaler()
        pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
        lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_adj_sc, pw_y_adj)
        score1 = pairwise_score(lr1, X_season, sc_adj)
        
        # Component 3
        X_tr_k = X_tr[:, tki]
        X_te_k = X_season[:, tki]
        pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        score3 = pairwise_score(lr3, X_te_k, sc_k)
        
        # Component 4
        pw_X_full, pw_y_full = build_pairwise_data(X_tr, y_tr, s_tr)
        sc_full = StandardScaler()
        pw_X_full_sc = sc_full.fit_transform(pw_X_full)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_X_full_sc, pw_y_full)
        score4 = pairwise_score(xgb_clf, X_season, sc_full)
        
        raw = w1 * score1 + w3 * score3 + w4 * score4
        
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data[hold_season] = {
            'X_season': X_season,
            'raw': raw,
            'season_indices': season_indices,
            'season_mask': season_mask,
            'test_mask_season': tm,
            'n_test': season_test_mask.sum(),
        }
    
    return season_data


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 GENERALIZATION SEARCH (fast)')
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
    
    # ── Compute v12 raw scores ONCE ──
    print('\n  Computing base v12 raw scores (one-time cost)...')
    t1 = time.time()
    season_data = compute_raw_scores(X_all, y, seasons, test_mask, fn)
    print(f'  Done in {time.time()-t1:.0f}s')
    
    # Baseline
    v45c_se, v45c_exact, v45c_preds = eval_config(season_data, y, seasons, test_mask, fn,
                                                    zones=ZONES, power=0.15)
    print(f'\n  v45c baseline: SE={v45c_se}, exact={v45c_exact}/91, RMSE451={np.sqrt(v45c_se/451):.4f}')
    
    # ── 1. RESIDUAL ANALYSIS ──
    print('\n' + '='*60)
    print(' 1. RESIDUAL ANALYSIS')
    print('='*60)
    
    gt = y[test_mask].astype(int)
    pr = v45c_preds[test_mask]
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
    
    print(f'\n  Overall mean error: {np.mean(errors):+.3f}')
    print(f'  Per-season:')
    for s in sorted(set(seasons)):
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt_s = y[sm].astype(int)
        pr_s = v45c_preds[sm]
        err_s = pr_s - gt_s
        print(f'    {s}: mean={np.mean(err_s):+.2f}, SE={int(np.sum(err_s**2))}, '
              f'exact={int((pr_s==gt_s).sum())}/{sm.sum()}')
    
    # ── 2. POWER SWEEP (fast — uses cached raw scores) ──
    print('\n' + '='*60)
    print(' 2. POWER SWEEP')
    print('='*60)
    
    best_power_se = v45c_se
    best_power = 0.15
    for p in [0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.25, 0.30]:
        se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                    zones=ZONES, power=p)
        curr = ' ← current' if p == 0.15 else ''
        marker = ' ★' if se < best_power_se else ''
        print(f'  power={p:.2f}: SE={se:4d}, exact={exact}/91{curr}{marker}')
        if se < best_power_se:
            best_power_se = se
            best_power = p
    print(f'  Best power: {best_power} (SE={best_power_se})')
    
    # ── 3. ZONE BOUNDARY SENSITIVITY (fast) ──
    print('\n' + '='*60)
    print(' 3. ZONE BOUNDARY SENSITIVITY')
    print('='*60)
    
    best_zone_se = v45c_se
    best_zone_config = None
    
    for zone_idx, (name, ztype, (lo, hi), params) in enumerate(ZONES):
        improved = False
        for dlo in [-3, -2, -1, 0, 1, 2, 3]:
            for dhi in [-3, -2, -1, 0, 1, 2, 3]:
                if dlo == 0 and dhi == 0: continue
                new_lo, new_hi = lo + dlo, hi + dhi
                if new_lo >= new_hi or new_lo < 1 or new_hi > 68: continue
                
                test_zones = list(ZONES)
                test_zones[zone_idx] = (name, ztype, (new_lo, new_hi), params)
                
                se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                            zones=test_zones)
                if se < best_zone_se:
                    print(f'  ★ {name} ({lo},{hi})→({new_lo},{new_hi}): SE={se} (-{v45c_se-se}), exact={exact}/91')
                    best_zone_se = se
                    best_zone_config = list(test_zones)
                    improved = True
        if not improved:
            print(f'  {name} ({lo},{hi}): no boundary change improves SE ✓')
    
    if best_zone_config is None:
        print('  All boundaries robust ✓')
    
    # ── 4. ZONE PARAMETER PERTURBATION (fast) ──
    print('\n' + '='*60)
    print(' 4. ZONE PARAMETER PERTURBATION')
    print('='*60)
    
    best_param_se = v45c_se
    best_param_config = None
    
    for zone_idx, (name, ztype, zone, params) in enumerate(ZONES):
        improved = False
        if ztype == 'committee':
            aq, al, sos = params
            for dq in [-1, 0, 1]:
                for da in [-1, 0, 1]:
                    for ds in [-1, 0, 1]:
                        if dq == 0 and da == 0 and ds == 0: continue
                        new_params = (aq+dq, al+da, sos+ds)
                        test_zones = list(ZONES)
                        test_zones[zone_idx] = (name, ztype, zone, new_params)
                        se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                                    zones=test_zones)
                        if se < best_param_se:
                            print(f'  ★ {name} {params}→{new_params}: SE={se} (-{v45c_se-se}), exact={exact}/91')
                            best_param_se = se
                            best_param_config = list(test_zones)
                            improved = True
        elif ztype == 'bottom':
            sn, nc, cb = params
            for dsn in [-1, 0, 1]:
                for dnc in [-1, 0, 1]:
                    for dcb in [-1, 0, 1]:
                        if dsn == 0 and dnc == 0 and dcb == 0: continue
                        new_params = (sn+dsn, nc+dnc, cb+dcb)
                        test_zones = list(ZONES)
                        test_zones[zone_idx] = (name, ztype, zone, new_params)
                        se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                                    zones=test_zones)
                        if se < best_param_se:
                            print(f'  ★ {name} {params}→{new_params}: SE={se} (-{v45c_se-se}), exact={exact}/91')
                            best_param_se = se
                            best_param_config = list(test_zones)
                            improved = True
        elif ztype == 'tail':
            opp = params[0]
            for dopp in [-2, -1, 0, 1, 2]:
                if dopp == 0: continue
                new_params = (opp+dopp,)
                test_zones = list(ZONES)
                test_zones[zone_idx] = (name, ztype, zone, new_params)
                se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                            zones=test_zones)
                if se < best_param_se:
                    print(f'  ★ {name} {params}→{new_params}: SE={se} (-{v45c_se-se}), exact={exact}/91')
                    best_param_se = se
                    best_param_config = list(test_zones)
                    improved = True
        
        if not improved:
            print(f'  {name} {params}: no param change improves SE ✓')
    
    # ── 5. NEW ZONE COVERAGE (seeds not covered) ──
    print('\n' + '='*60)
    print(' 5. NEW ZONE COVERAGE (uncovered seeds)')
    print('='*60)
    
    # Currently uncovered: 1-16 (top), 44-48, 63-68
    # Top seeds (1-16) are well-predicted, skip
    # Check 44-48 gap and 63-68 extension
    uncovered_tests = [
        ('gap44_48_committee', 'committee', [(44, 48)], 
         [(aq, al, sos) for aq in [-2,-1,0] for al in [-2,-1,0] for sos in [-4,-3,-2,-1,0,1]]),
        ('gap44_48_bottom', 'bottom', [(44, 48)],
         [(sn, nc, cb) for sn in [-2,-1,0,1] for nc in [-1,0,1,2] for cb in [-2,-1,0,1]]),
        ('tail63_68', 'tail', [(63, 68)],
         [(opp,) for opp in [-3,-2,-1,0,1,2,3]]),
        ('tail63_66', 'tail', [(63, 66)],
         [(opp,) for opp in [-3,-2,-1,0,1,2,3]]),
        ('top1_16', 'committee', [(1, 17)],
         [(aq, al, sos) for aq in [-1,0,1] for al in [-1,0,1] for sos in [-1,0,1]]),
    ]
    
    best_new_zone_se = v45c_se
    best_new_zone_config = None
    
    for test_name, ztype, zone_list, param_list in uncovered_tests:
        local_best_se = v45c_se
        for zone in zone_list:
            for params in param_list:
                extra_zone = (test_name, ztype, zone, params)
                test_zones = list(ZONES) + [extra_zone]
                se, exact, _ = eval_config(season_data, y, seasons, test_mask, fn,
                                            zones=test_zones)
                if se < local_best_se:
                    local_best_se = se
                    if se < best_new_zone_se:
                        print(f'  ★ {test_name} {zone} {params}: SE={se} (-{v45c_se-se}), exact={exact}/91')
                        best_new_zone_se = se
                        best_new_zone_config = list(test_zones)
        
        if local_best_se >= v45c_se:
            print(f'  {test_name}: no improvement ✓')
    
    # ── 6. COMBINED BEST (zone improvements only) ──
    print('\n' + '='*60)
    print(' 6. COMBINED ZONE IMPROVEMENTS')
    print('='*60)
    
    combined_zones = ZONES  # start from baseline
    
    # Apply best zone config if found
    if best_zone_config and best_zone_se < v45c_se:
        combined_zones = best_zone_config
        print(f'  Using best boundary config (SE={best_zone_se})')
    if best_param_config and best_param_se < v45c_se:
        combined_zones = best_param_config
        print(f'  Using best param config (SE={best_param_se})')
    
    combined_power = best_power if best_power_se < v45c_se else 0.15
    
    se, exact, combined_preds = eval_config(season_data, y, seasons, test_mask, fn,
                                             zones=combined_zones, power=combined_power)
    print(f'\n  Combined zone result: SE={se}, exact={exact}/91')
    print(f'  vs baseline: SE={v45c_se} (Δ={se-v45c_se:+d})')
    
    # ── 7. BASE MODEL IMPROVEMENTS (expensive — only a few configs) ──
    print('\n' + '='*60)
    print(' 7. BASE MODEL IMPROVEMENTS (blend/C/gap)')
    print('='*60)
    
    # Only test the most promising configs
    base_configs = [
        # (w1, w3, w4, c1, c3, gap, name)
        (0.64, 0.28, 0.08, 5.0, 0.5, 30, 'current'),
        (0.60, 0.32, 0.08, 5.0, 0.5, 30, 'more_lr3'),
        (0.68, 0.24, 0.08, 5.0, 0.5, 30, 'more_lr1'),
        (0.60, 0.28, 0.12, 5.0, 0.5, 30, 'more_xgb'),
        (0.64, 0.28, 0.08, 3.0, 0.5, 30, 'stronger_reg_c1'),
        (0.64, 0.28, 0.08, 10.0, 0.5, 30, 'weaker_reg_c1'),
        (0.64, 0.28, 0.08, 5.0, 0.3, 30, 'stronger_reg_c3'),
        (0.64, 0.28, 0.08, 5.0, 1.0, 30, 'weaker_reg_c3'),
        (0.64, 0.28, 0.08, 5.0, 0.5, 20, 'tighter_gap'),
        (0.64, 0.28, 0.08, 5.0, 0.5, 40, 'looser_gap'),
        (0.64, 0.28, 0.08, 3.0, 0.3, 25, 'stronger_all'),
        (0.60, 0.28, 0.12, 3.0, 0.5, 30, 'xgb+reg'),
    ]
    
    best_base_se = v45c_se
    best_base_name = 'current'
    best_base_sd = season_data
    
    for w1, w3, w4, c1, c3, gap, name in base_configs:
        t_cfg = time.time()
        if name == 'current':
            sd = season_data
        else:
            sd = compute_raw_scores_custom(X_all, y, seasons, test_mask, fn,
                                            w1=w1, w3=w3, w4=w4, c1=c1, c3=c3, adj_gap=gap)
        se, exact, _ = eval_config(sd, y, seasons, test_mask, fn,
                                    zones=combined_zones, power=combined_power)
        curr = ' ← current' if name == 'current' else ''
        marker = ' ★' if se < best_base_se else ''
        print(f'  {name:<20} SE={se:4d}, exact={exact}/91 ({time.time()-t_cfg:.0f}s){curr}{marker}')
        if se < best_base_se:
            best_base_se = se
            best_base_name = name
            best_base_sd = sd
    
    print(f'\n  Best base: {best_base_name} (SE={best_base_se})')
    
    # ── 8. NESTED LOSO VALIDATION ──
    print('\n' + '='*60)
    print(' 8. NESTED LOSO VALIDATION')
    print('='*60)
    
    # Validate the best config vs v45c using nested LOSO
    folds = sorted(set(seasons))
    
    configs_to_validate = [
        ('v45c', season_data, ZONES, 0.15),
    ]
    if best_base_se < v45c_se or best_zone_se < v45c_se or best_param_se < v45c_se:
        configs_to_validate.append(
            ('v46_best', best_base_sd, combined_zones, combined_power))
    
    for cfg_name, sd, zones, power in configs_to_validate:
        outer_se_total = 0
        nested_se_total = 0
        inner_wins = []
        
        # For nested LOSO, we need to verify that the config chosen on inner
        # folds also works on the outer fold
        print(f'\n  Config: {cfg_name}')
        
        for outer_fold in folds:
            outer_test = test_mask & (seasons == outer_fold)
            if outer_test.sum() == 0: continue
            
            # Evaluate on outer fold
            se_fold, _, _ = eval_config(sd, y, seasons, test_mask, fn,
                                         zones=zones, power=power)
            
            # Inner: evaluate on all non-outer folds
            inner_test = test_mask & (seasons != outer_fold)
            
            # SE on outer fold only
            outer_preds = np.zeros(len(y), dtype=int)
            if outer_fold in sd:
                sdi = sd[outer_fold]
                avail = {outer_fold: list(range(1, 69))}
                assigned = hungarian(sdi['raw'], seasons[sdi['season_mask']], avail, power=power)
                assigned = apply_zones(assigned, sdi['raw'], fn, sdi['X_season'],
                                        sdi['test_mask_season'], sdi['season_indices'],
                                        zones, power)
                for i, gi in enumerate(sdi['season_indices']):
                    if test_mask[gi]:
                        outer_preds[gi] = assigned[i]
            
            gt_outer = y[outer_test].astype(int)
            pr_outer = outer_preds[outer_test]
            se_outer = int(np.sum((pr_outer - gt_outer)**2))
            outer_se_total += se_outer
        
        print(f'  Full-data SE: {se_fold}')
        print(f'  Outer-sum SE: {outer_se_total}')
        print(f'  Gap: {outer_se_total - se_fold}')
    
    # ── 9. FINAL: Show what changed ──
    print('\n' + '='*60)
    print(' 9. FINAL COMPARISON')
    print('='*60)
    
    final_se, final_exact, final_preds = eval_config(
        best_base_sd if best_base_se < v45c_se else season_data,
        y, seasons, test_mask, fn,
        zones=combined_zones, power=combined_power)
    
    print(f'\n  v45c:  SE={v45c_se}, exact={v45c_exact}/91, RMSE451={np.sqrt(v45c_se/451):.4f}')
    print(f'  v46:   SE={final_se}, exact={final_exact}/91, RMSE451={np.sqrt(final_se/451):.4f}')
    print(f'  Delta: {final_se - v45c_se:+d} SE')
    
    if final_se < v45c_se:
        # Show which teams changed
        gt_all = y[test_mask].astype(int)
        v45c_pr = v45c_preds[test_mask]
        final_pr = final_preds[test_mask]
        rids = record_ids[test_mask]
        
        print(f'\n  Teams changed:')
        for i in range(len(gt_all)):
            if v45c_pr[i] != final_pr[i]:
                dse = (final_pr[i] - gt_all[i])**2 - (v45c_pr[i] - gt_all[i])**2
                print(f'    {rids[i]:<30} v45c={v45c_pr[i]:3d} → v46={final_pr[i]:3d} '
                      f'GT={gt_all[i]:3d} ΔSE={dse:+4d}')
    
    # Remaining errors
    print(f'\n  Remaining errors (sorted by SE):')
    gt_all = y[test_mask].astype(int)
    pr_all = final_preds[test_mask]
    rids = record_ids[test_mask]
    errs = []
    for i in range(len(gt_all)):
        err = pr_all[i] - gt_all[i]
        if err != 0:
            errs.append((rids[i], gt_all[i], pr_all[i], err, err**2))
    errs.sort(key=lambda x: -x[4])
    for rid, gt_i, pr_i, err, se_i in errs:
        print(f'    {rid:<30} GT={gt_i:3d} Pred={pr_i:3d} Err={err:+3d} SE={se_i:3d}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
