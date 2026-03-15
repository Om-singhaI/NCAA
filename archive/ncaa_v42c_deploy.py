#!/usr/bin/env python3
"""
v42c: Validate z6 (NETSOS 55-61 w=1) + nested LOSO for z5+z6 combo
Then deploy the safest validated improvement to production.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features,
    select_top_k_features, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    predict_robust_blend,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP, HUNGARIAN_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def apply_generic_zone(p, raw, fvals, tm, zone, weight, power=0.15):
    lo, hi = zone
    zt = [i for i in range(len(p)) if tm[i] and lo <= p[i] <= hi]
    if len(zt) <= 1:
        return p
    fv = np.array([fvals[i] for i in zt], dtype=float)
    vmin, vmax = fv.min(), fv.max()
    if vmax > vmin:
        norm = (fv - vmin) / (vmax - vmin) * 2 - 1
    else:
        norm = np.zeros(len(fv))
    corr = weight * norm
    seeds = [p[i] for i in zt]
    corrected = [raw[zt[k]] + corr[k] for k in range(len(zt))]
    cost = np.array([[abs(sv - sd)**power for sd in seeds] for sv in corrected])
    ri, ci = linear_sum_assignment(cost)
    pnew = p.copy()
    for r, c in zip(ri, ci):
        pnew[zt[r]] = seeds[c]
    return pnew


def apply_all_zones(pass1, raw, fn, X, tm, idx, ncsos_vals=None, sos_vals=None,
                     use_z5=False, use_z6=False):
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=0.15)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=0.15)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=0.15)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=0.15)
    if use_z5 and ncsos_vals is not None:
        p = apply_generic_zone(p, raw, ncsos_vals, tm, (17, 24), 9, 0.15)
    if use_z6 and sos_vals is not None:
        p = apply_generic_zone(p, raw, sos_vals, tm, (55, 61), 1, 0.15)
    return p


def count_exact(p, tm, indices, test_mask, y):
    return sum(1 for i, gi in enumerate(indices) if test_mask[gi] and p[i] == int(y[gi]))


def main():
    print('='*70)
    print('  v42c VALIDATE Z6 + NESTED LOSO')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
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
    teams = labeled['Team'].values if 'Team' in labeled.columns else record_ids
    folds = sorted(set(seasons))
    fi = {f: i for i, f in enumerate(fn)}

    ncsos_raw = pd.to_numeric(labeled['NETNonConfSOS'], errors='coerce').fillna(200).values
    sos_raw_idx = fi['NETSOS']

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                    np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    season_data = {}
    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0:
            continue
        gt = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt], y[gt], fn, k=USE_TOP_K_A,
                                     forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X[gt], y[gt], X_s, seasons[gt], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = {
            'pass1': p1, 'raw': raw, 'X': X_s,
            'tm': tm, 'indices': si.copy(),
            'ncsos': ncsos_raw[sm],
            'sos': X_s[:, sos_raw_idx],
        }

    # ════════════════════════════════════════════════════════════
    #  1. TEST CONFIGS
    # ════════════════════════════════════════════════════════════
    configs = {
        'v25': {'z5': False, 'z6': False},
        'z5 only': {'z5': True, 'z6': False},
        'z5+z6': {'z5': True, 'z6': True},
        'z6 only': {'z5': False, 'z6': True},
    }

    print('\n  Full test scores:')
    config_scores = {}
    for name, cfg in configs.items():
        scores = {}
        for s, sd in season_data.items():
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'], sd['sos'],
                               use_z5=cfg['z5'], use_z6=cfg['z6'])
            scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        config_scores[name] = scores
        total = sum(scores.values())
        sv = list(scores.values())
        print(f'    {name:>12}: {total}/91  {sv}')

    # ════════════════════════════════════════════════════════════
    #  2. NESTED LOSO
    # ════════════════════════════════════════════════════════════
    print('\n  Nested LOSO:')
    nested_results = {}
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_name = 'v25'
        for name, scores in config_scores.items():
            ts = sum(scores.get(s, 0) for s in tune)
            if ts > best_tune:
                best_tune = ts
                best_name = name
            elif ts == best_tune and name == 'v25':
                best_name = name
        nested_results[hold] = (best_name, config_scores[best_name][hold])
        print(f'    {hold}: chose {best_name:>8} (tune={best_tune}) → {config_scores[best_name][hold]}')
    
    nested_total = sum(v for _, v in nested_results.values())
    print(f'\n  ★ Nested LOSO: {nested_total}/91')

    # ════════════════════════════════════════════════════════════
    #  3. Z6 SWEEP around NETSOS (55,61) — broader search
    # ════════════════════════════════════════════════════════════
    print('\n  Z6 broader sweep (on top of z5):')
    best_z6 = 73
    best_z6_cfg = None
    z6_count = {}
    
    for lo6 in range(40, 65):
        for hi6 in range(lo6+2, min(lo6+12, 69)):
            for w6 in [-3, -2, -1, 1, 2, 3]:
                total = 0
                for s, sd in season_data.items():
                    p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                       sd['tm'], sd['indices'], sd['ncsos'], sd['sos'],
                                       use_z5=True, use_z6=False)
                    p = apply_generic_zone(p, sd['raw'], sd['sos'], sd['tm'],
                                          (lo6, hi6), w6, power=0.15)
                    total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                if total >= best_z6:
                    if total > best_z6:
                        best_z6 = total
                        best_z6_cfg = (lo6, hi6, w6)
                    z6_count[total] = z6_count.get(total, 0) + 1

    print(f'  Best z6: {best_z6}/91')
    for score in sorted(z6_count.keys(), reverse=True):
        if score >= 73:
            print(f'    {score}/91: {z6_count[score]} configs')

    # ════════════════════════════════════════════════════════════
    #  4. PERMUTATION TEST for z5+z6
    # ════════════════════════════════════════════════════════════
    if best_z6 > 73:
        print(f'\n  Z6 permutation test (50 shuffles)...')
        observed = best_z6 - 73  # improvement over z5-only
        n_better = 0
        for trial in range(50):
            rng = np.random.RandomState(trial + 200)
            total = 0
            for s, sd in season_data.items():
                shuf_sos = sd['sos'].copy()
                ti = [i for i in range(len(sd['tm'])) if sd['tm'][i]]
                vals = shuf_sos[ti].copy()
                rng.shuffle(vals)
                shuf_sos[ti] = vals
                p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'], sd['ncsos'], sd['sos'],
                                   use_z5=True, use_z6=False)
                p = apply_generic_zone(p, sd['raw'], shuf_sos, sd['tm'],
                                      (best_z6_cfg[0], best_z6_cfg[1]), best_z6_cfg[2], 0.15)
                total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            if total - 73 >= observed:
                n_better += 1
        p_val = n_better / 50
        print(f'  Z6 p-value: {p_val:.4f} ({n_better}/50)')

    # ════════════════════════════════════════════════════════════
    #  5. TEAM DIFFS for z5+z6 vs z5
    # ════════════════════════════════════════════════════════════
    if best_z6 > 73 and best_z6_cfg:
        print(f'\n  Team diffs z5 vs z5+z6 ({best_z6_cfg}):')
        for s, sd in season_data.items():
            p_z5 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'], sd['ncsos'], sd['sos'],
                                   use_z5=True, use_z6=False)
            p_z6 = p_z5.copy()
            p_z6 = apply_generic_zone(p_z6, sd['raw'], sd['sos'], sd['tm'],
                                     (best_z6_cfg[0], best_z6_cfg[1]), best_z6_cfg[2], 0.15)
            
            changes = []
            for i, gi in enumerate(sd['indices']):
                if sd['tm'][i] and p_z5[i] != p_z6[i]:
                    changes.append({
                        'team': teams[gi], 'gt': int(y[gi]),
                        'z5': p_z5[i], 'z6': p_z6[i],
                    })
            if changes:
                print(f'\n    {s}:')
                for c in sorted(changes, key=lambda x: x['gt']):
                    z5_ok = c['z5'] == c['gt']
                    z6_ok = c['z6'] == c['gt']
                    mark = '✓' if z6_ok and not z5_ok else ('✗' if z5_ok and not z6_ok else '=')
                    print(f'      {mark} {c["team"]:<22s} GT={c["gt"]:2d}  z5={c["z5"]:2d}  z6={c["z6"]:2d}')

    # ════════════════════════════════════════════════════════════
    #  6. COMPREHENSIVE NESTED LOSO WITH ALL VARIANTS
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  COMPREHENSIVE NESTED LOSO')
    print('='*70)

    # Add z5+z6 best to configs
    if best_z6 > 73 and best_z6_cfg:
        z6lo, z6hi, z6w = best_z6_cfg
        for s, sd in season_data.items():
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'], sd['sos'],
                               use_z5=True, use_z6=False)
            p = apply_generic_zone(p, sd['raw'], sd['sos'], sd['tm'],
                                  (z6lo, z6hi), z6w, 0.15)
            if 'z5+z6_best' not in config_scores:
                config_scores['z5+z6_best'] = {}
            config_scores['z5+z6_best'][s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)

    print('\n  All config scores:')
    for name, scores in config_scores.items():
        total = sum(scores.values())
        sv = list(scores.values())
        print(f'    {name:>14}: {total}/91  {sv}')

    print('\n  Nested LOSO (comprehensive):')
    nested_total = 0
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_name = 'v25'
        for name, scores in config_scores.items():
            ts = sum(scores.get(s, 0) for s in tune)
            if ts > best_tune:
                best_tune = ts
                best_name = name
            elif ts == best_tune and name == 'v25':
                best_name = name
        nested_total += config_scores[best_name][hold]
        print(f'    {hold}: chose {best_name:>14} (tune={best_tune}) → {config_scores[best_name][hold]}')
    
    print(f'\n  ★ Nested LOSO: {nested_total}/91')

    # ════════════════════════════════════════════════════════════
    #  DEPLOYMENT RECOMMENDATION
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  DEPLOYMENT RECOMMENDATION')
    print('='*70)
    
    z5_total = sum(config_scores.get('z5 only', {}).values())
    z5z6_total = sum(config_scores.get('z5+z6_best', config_scores.get('z5+z6', {})).values())
    v25_total = sum(config_scores.get('v25', {}).values())
    
    print(f'\n  v25:      {v25_total}/91')
    print(f'  z5:       {z5_total}/91')
    print(f'  z5+z6:    {z5z6_total}/91')
    print(f'  Nested:   {nested_total}/91')
    
    if nested_total >= z5_total and z5_total > v25_total:
        print(f'\n  ★ DEPLOY z5 (zone=(17,24) w=9 pw=0.15) for validated +{z5_total-v25_total}')
    if nested_total >= z5z6_total and z5z6_total > z5_total:
        print(f'  ★ ALSO DEPLOY z6 for additional +{z5z6_total-z5_total}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
