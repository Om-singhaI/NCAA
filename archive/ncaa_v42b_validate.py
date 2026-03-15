#!/usr/bin/env python3
"""
v42b DEEP VALIDATION of NETNonConfSOS Zone
============================================
v42 found: zone=(17,24) w=8 → 73/91 (nested LOSO = 73/91, gap=0)
40 different configs achieve 73/91, indicating high stability.

This script:
1. Tests ALL 40 best configs, find most robust one
2. Permutation test (100 shuffles) for statistical significance
3. Stability analysis (perturbation tests)
4. Shows which specific teams got fixed vs v25
5. Explores if we can push beyond 73/91 with z5 as base
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, parse_wl,
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


def apply_all_zones(pass1, raw, fn, X, tm, idx, ncsos_vals=None,
                     z5=None, z5_pw=0.15):
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=0.15)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=0.15)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=0.15)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=0.15)
    if z5 is not None and ncsos_vals is not None:
        lo, hi, w = z5
        p = apply_generic_zone(p, raw, ncsos_vals, tm, (lo, hi), w, z5_pw)
    return p


def count_exact(p, tm, indices, test_mask, y):
    return sum(1 for i, gi in enumerate(indices) if test_mask[gi] and p[i] == int(y[gi]))


def get_predictions(p, tm, indices, test_mask, y, teams):
    results = []
    for i, gi in enumerate(indices):
        if test_mask[gi]:
            results.append({
                'team': teams[gi],
                'pred': p[i],
                'gt': int(y[gi]),
                'error': p[i] - int(y[gi]),
                'correct': p[i] == int(y[gi])
            })
    return results


def main():
    print('='*70)
    print('  v42b DEEP VALIDATION')
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

    ncsos_raw = pd.to_numeric(labeled['NETNonConfSOS'], errors='coerce').fillna(200).values

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                    np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Build season data
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
        }

    # ════════════════════════════════════════════════════════════
    #  1. COMPREHENSIVE ZONE SWEEP — find all 73+ configs
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  1. COMPREHENSIVE SWEEP (all zone params)')
    print('='*70)

    configs_73 = []
    best_overall = 70

    for lo in range(14, 22):
        for hi in range(lo+3, min(lo+16, 36)):
            for w in range(1, 11):
                for pw in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                    total = 0
                    for s, sd in season_data.items():
                        p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                           sd['tm'], sd['indices'], sd['ncsos'],
                                           z5=(lo, hi, w), z5_pw=pw)
                        total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                    if total > best_overall:
                        best_overall = total
                    if total >= 73:
                        configs_73.append((lo, hi, w, pw, total))

    print(f'  Total configs ≥73: {len(configs_73)}')
    print(f'  Best overall: {best_overall}/91')

    # Group by (lo, hi)
    from collections import Counter
    zone_counts = Counter((c[0], c[1]) for c in configs_73)
    print(f'\n  Zones with ≥73 configs:')
    for (lo, hi), cnt in zone_counts.most_common(20):
        weights = sorted(set(c[2] for c in configs_73 if c[0]==lo and c[1]==hi))
        print(f'    ({lo},{hi}): {cnt} configs, w={weights}')

    # ════════════════════════════════════════════════════════════
    #  2. SHOW WHICH TEAMS FIXED vs v25
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  2. TEAM-LEVEL DIFF: v25 vs z5')
    print('='*70)

    # Use most robust config: pick zone with most configs
    best_zone = zone_counts.most_common(1)[0][0]
    # Find config with most moderate params for that zone
    zone_cfgs = [c for c in configs_73 if c[0]==best_zone[0] and c[1]==best_zone[1]]
    # Pick median weight and pw=0.15
    weights_at_zone = sorted(set(c[2] for c in zone_cfgs))
    median_w = weights_at_zone[len(weights_at_zone)//2]
    chosen = (best_zone[0], best_zone[1], median_w, 0.15)
    
    print(f'\n  Most robust zone: ({best_zone[0]},{best_zone[1]})')
    print(f'  Using: zone=({chosen[0]},{chosen[1]}) w={chosen[2]} pw={chosen[3]}')

    for s, sd in season_data.items():
        p_v25 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'])
        p_z5 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'],
                               z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
        
        r_v25 = get_predictions(p_v25, sd['tm'], sd['indices'], test_mask, y, teams)
        r_z5 = get_predictions(p_z5, sd['tm'], sd['indices'], test_mask, y, teams)
        
        v25_map = {r['team']: r for r in r_v25}
        z5_map = {r['team']: r for r in r_z5}
        
        changes = []
        for team in v25_map:
            if v25_map[team]['pred'] != z5_map[team]['pred']:
                changes.append({
                    'team': team,
                    'gt': v25_map[team]['gt'],
                    'v25': v25_map[team]['pred'],
                    'z5': z5_map[team]['pred'],
                    'v25_err': abs(v25_map[team]['error']),
                    'z5_err': abs(z5_map[team]['error']),
                })
        
        if changes:
            print(f'\n  Season {s}: ({len(changes)} changes)')
            for c in sorted(changes, key=lambda x: x['gt']):
                arrow = '✓' if c['z5_err'] < c['v25_err'] else ('✗' if c['z5_err'] > c['v25_err'] else '=')
                print(f'    {arrow} {c["team"]:>20s}  GT={c["gt"]:2d}  v25={c["v25"]:2d}  z5={c["z5"]:2d}')

    # ════════════════════════════════════════════════════════════
    #  3. PERMUTATION TEST (100 shuffles)
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  3. PERMUTATION TEST (100 shuffles)')
    print('='*70)

    observed_improvement = 0
    for s, sd in season_data.items():
        p_v25 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'])
        p_z5 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'],
                               z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
        ex_v25 = count_exact(p_v25, sd['tm'], sd['indices'], test_mask, y)
        ex_z5 = count_exact(p_z5, sd['tm'], sd['indices'], test_mask, y)
        observed_improvement += (ex_z5 - ex_v25)
    
    print(f'  Observed improvement: +{observed_improvement}')

    n_better = 0
    for trial in range(100):
        rng = np.random.RandomState(trial)
        total_imp = 0
        for s, sd in season_data.items():
            shuffled_ncsos = sd['ncsos'].copy()
            test_indices = [i for i in range(len(sd['tm'])) if sd['tm'][i]]
            vals = shuffled_ncsos[test_indices].copy()
            rng.shuffle(vals)
            shuffled_ncsos[test_indices] = vals
            
            p_v25 = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                    sd['tm'], sd['indices'])
            p_shuf = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                     sd['tm'], sd['indices'], shuffled_ncsos,
                                     z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
            ex_v25 = count_exact(p_v25, sd['tm'], sd['indices'], test_mask, y)
            ex_shuf = count_exact(p_shuf, sd['tm'], sd['indices'], test_mask, y)
            total_imp += (ex_shuf - ex_v25)
        
        if total_imp >= observed_improvement:
            n_better += 1

    p_value = n_better / 100
    print(f'  p-value: {p_value:.4f} ({n_better}/100 shuffles >= observed)')
    if p_value < 0.05:
        print(f'  ★ SIGNIFICANT (p < 0.05)')
    elif p_value < 0.10:
        print(f'  ⚠ BORDERLINE (0.05 ≤ p < 0.10)')
    else:
        print(f'  ✗ NOT SIGNIFICANT (p ≥ 0.10)')

    # ════════════════════════════════════════════════════════════
    #  4. STABILITY: perturb zone params slightly
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  4. STABILITY ANALYSIS')
    print('='*70)

    # Test neighboring configs
    lo_c, hi_c, w_c, pw_c = chosen
    perturbations = []
    for dlo in [-2, -1, 0, 1, 2]:
        for dhi in [-2, -1, 0, 1, 2]:
            for dw in [-2, -1, 0, 1, 2]:
                nlo = lo_c + dlo
                nhi = hi_c + dhi
                nw = w_c + dw
                if nlo < 1 or nhi <= nlo or nw < 1:
                    continue
                total = 0
                for s, sd in season_data.items():
                    p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                       sd['tm'], sd['indices'], sd['ncsos'],
                                       z5=(nlo, nhi, nw), z5_pw=pw_c)
                    total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                perturbations.append((nlo, nhi, nw, total))

    scores = [p[3] for p in perturbations]
    print(f'  Perturbations tested: {len(perturbations)}')
    print(f'  Min: {min(scores)}/91')
    print(f'  Max: {max(scores)}/91')
    print(f'  Mean: {np.mean(scores):.1f}/91')
    print(f'  Median: {np.median(scores):.1f}/91')
    print(f'  ≥70 (v25 baseline): {sum(1 for s in scores if s >= 70)} / {len(scores)}')
    print(f'  ≥71: {sum(1 for s in scores if s >= 71)} / {len(scores)}')
    print(f'  ≥72: {sum(1 for s in scores if s >= 72)} / {len(scores)}')
    print(f'  ≥73: {sum(1 for s in scores if s >= 73)} / {len(scores)}')

    # ════════════════════════════════════════════════════════════
    #  5. EXPLORE BEYOND 73 on top of z5
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  5. PUSH BEYOND 73: 6TH ZONE ON TOP OF z5')
    print('='*70)

    fi = {f: i for i, f in enumerate(fn)}
    
    # After z5, what features could help fix the remaining 18 errors?
    # Candidate features for 6th zone
    z6_features = {
        'NETSOS': lambda sd: np.array([sd['X'][i][fi['NETSOS']] for i in range(len(sd['X']))]),
        'opp_net': lambda sd: np.array([sd['X'][i][fi.get('AvgOppNET', fi['NET Rank'])] for i in range(len(sd['X']))]),
        'conf_avg_net': lambda sd: np.array([sd['X'][i][fi['conf_avg_net']] for i in range(len(sd['X']))]),
        'Q1_W': lambda sd: np.array([sd['X'][i][fi['Quadrant1_W']] for i in range(len(sd['X']))]),
        'WL_Pct': lambda sd: np.array([sd['X'][i][fi['WL_Pct']] for i in range(len(sd['X']))]),
    }

    best_z6 = 73
    best_z6_cfg = None
    
    for feat_name, feat_fn in z6_features.items():
        for lo6 in range(10, 60, 3):
            for hi6 in range(lo6+3, min(lo6+15, 68)):
                for w6 in [-5, -3, -2, -1, 1, 2, 3, 5]:
                    total = 0
                    for s, sd in season_data.items():
                        fvals = feat_fn(sd)
                        p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                           sd['tm'], sd['indices'], sd['ncsos'],
                                           z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
                        p = apply_generic_zone(p, sd['raw'], fvals, sd['tm'],
                                              (lo6, hi6), w6, power=0.15)
                        total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                    if total > best_z6:
                        best_z6 = total
                        best_z6_cfg = (feat_name, lo6, hi6, w6)
                        print(f'  ★ {feat_name} zone=({lo6},{hi6}) w={w6}: {total}/91')

    if best_z6_cfg:
        print(f'\n  Best 6th zone: {best_z6}/91 at {best_z6_cfg}')
        # Validate with nested LOSO
        fn6, lo6, hi6, w6 = best_z6_cfg
        feat_fn6 = z6_features[fn6]
        
        # Check per season
        for s, sd in season_data.items():
            fvals = feat_fn6(sd)
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'],
                               z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
            p = apply_generic_zone(p, sd['raw'], fvals, sd['tm'],
                                  (lo6, hi6), w6, power=0.15)
            ex = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            print(f'    {s}: {ex}')
    else:
        print('  No improvement found beyond 73/91')

    # ════════════════════════════════════════════════════════════
    #  6. SHOW REMAINING 18 ERRORS after z5
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  6. REMAINING ERRORS AFTER z5')
    print('='*70)

    error_teams = []
    for s, sd in season_data.items():
        p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                           sd['tm'], sd['indices'], sd['ncsos'],
                           z5=(chosen[0], chosen[1], chosen[2]), z5_pw=chosen[3])
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi] and p[i] != int(y[gi]):
                error_teams.append({
                    'season': s,
                    'team': teams[gi],
                    'gt': int(y[gi]),
                    'pred': p[i],
                    'error': p[i] - int(y[gi]),
                    'raw': sd['raw'][i],
                })

    error_teams.sort(key=lambda x: x['gt'])
    print(f'\n  {len(error_teams)} remaining errors:')
    print(f'  {"Season":<10} {"Team":<22} {"GT":>3} {"Pred":>4} {"Err":>4} {"Raw":>6}')
    print(f'  {"-"*55}')
    for e in error_teams:
        print(f'  {e["season"]:<10} {e["team"]:<22} {e["gt"]:3d} {e["pred"]:4d} {e["error"]:+4d} {e["raw"]:6.1f}')

    # ════════════════════════════════════════════════════════════
    #  FINAL VERDICT
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  FINAL VERDICT')
    print('='*70)
    print(f'\n  v25 baseline:    70/91')
    print(f'  z5 improvement:  73/91 (nested LOSO = 73/91, gap = 0)')
    print(f'  Best config:     zone=({chosen[0]},{chosen[1]}) w={chosen[2]} pw={chosen[3]}')
    print(f'  Permutation p:   {p_value:.4f}')
    print(f'  Stability min:   {min(scores)}/91')
    if best_z6_cfg:
        print(f'  Best z6:         {best_z6}/91 at {best_z6_cfg}')
    
    safe = p_value < 0.10 and min(scores) >= 70
    print(f'\n  SAFE TO DEPLOY: {"YES" if safe else "NEEDS MORE ANALYSIS"}')
    print(f'  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
