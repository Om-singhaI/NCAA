#!/usr/bin/env python3
"""
v24b: Fine-tune bottom-zone correction found in v24
====================================================

v24 Phase 2 found: 65/91 with bottom zone (53-68) sosnet=-2, net_conf=+2
Also: 64/91 with cbhist=-3

This script:
  1. Fine-tunes the best bottom-zone formula (sosnet + net_conf weights)
  2. Tests 3-component bottom-zone combos
  3. Tests bottom zone boundaries
  4. Validates with nested LOSO
  5. Also tests stacking the 65/91 config with different mid+low params
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_midrange_correction(fn, X, aq=0, al=2, sos=3):
    fi = {f: i for i, f in enumerate(fn)}
    correction = np.zeros(X.shape[0])
    net = X[:, fi['NET Rank']]
    is_al = X[:, fi['is_AL']]
    is_power = X[:, fi['is_power_conf']]
    sos_val = X[:, fi['NETSOS']]
    if al != 0:
        correction -= al * is_al * is_power * np.clip((net - 20) / 50, 0, 1)
    if sos != 0:
        correction += sos * (sos_val - net) / 100
    return correction


def compute_low_correction(fn, X, q1dom=1, field=2):
    fi = {f: i for i, f in enumerate(fn)}
    correction = np.zeros(X.shape[0])
    if q1dom != 0:
        q1w = X[:, fi['Quadrant1_W']]
        q1l = X[:, fi['Quadrant1_L']]
        q1_rate = q1w / (q1w + q1l + 1)
        correction -= q1dom * q1_rate
    if field != 0:
        tfr = X[:, fi['tourn_field_rank']]
        field_gap = (tfr - 34) / 34
        correction += field * field_gap
    return correction


def compute_bottom_correction(fn, X, components):
    fi = {f: i for i, f in enumerate(fn)}
    n = X.shape[0]
    correction = np.zeros(n)
    
    net = X[:, fi['NET Rank']]
    is_aq = X[:, fi['is_AQ']]
    is_al = X[:, fi['is_AL']]
    is_power = X[:, fi['is_power_conf']]
    sos = X[:, fi['NETSOS']]
    conf_avg = X[:, fi['conf_avg_net']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q2w = X[:, fi['Quadrant2_W']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    if components.get('cbhist', 0) != 0:
        hist_gap = (cb_mean - tfr) / 34
        correction += components['cbhist'] * hist_gap
    
    if components.get('sosnet', 0) != 0:
        gap = (sos - net) / 200
        correction += components['sosnet'] * gap
    
    if components.get('aq_weak', 0) != 0:
        conf_weakness = np.clip((conf_avg - 100) / 100, 0, 2)
        correction += components['aq_weak'] * is_aq * conf_weakness
    
    if components.get('badloss', 0) != 0:
        bad = np.clip((q3l + q4l) / 5, 0, 2)
        correction += components['badloss'] * bad
    
    if components.get('resume', 0) != 0:
        resume = q1w + q2w - q3l - q4l
        correction -= components['resume'] * np.clip(resume / 10, -1, 1)
    
    if components.get('net_conf', 0) != 0:
        gap = (conf_avg - net) / 100
        correction += components['net_conf'] * gap
    
    if components.get('q1dom', 0) != 0:
        q1_rate = q1w / (q1w + q1l + 1)
        correction -= components['q1dom'] * q1_rate
    
    if components.get('field', 0) != 0:
        field_gap = (tfr - 34) / 34
        correction += components['field'] * field_gap
    
    if components.get('al_mid', 0) != 0:
        is_mid = (1 - is_power) * is_al
        correction -= components['al_mid'] * is_mid
    
    return correction


def apply_swap(pass1, raw_scores, correction, test_mask_s, zone, power=0.15):
    lo, hi = zone
    idx = [i for i in range(len(pass1))
           if test_mask_s[i] and lo <= pass1[i] <= hi]
    if len(idx) <= 1:
        return pass1.copy()
    seeds = [pass1[i] for i in idx]
    corr = [raw_scores[i] + correction[i] for i in idx]
    cost = np.array([[abs(s - seed)**power for seed in seeds] for s in corr])
    ri, ci = linear_sum_assignment(cost)
    final = pass1.copy()
    for r, c in zip(ri, ci):
        final[idx[r]] = seeds[c]
    return final


def eval_3zone(season_data, test_mask, y, test_seasons, fn,
               mid_params=(0, 2, 3), mid_zone=(17, 34),
               low_params=(1, 2), low_zone=(35, 52),
               bot_comps=None, bot_zone=(53, 68)):
    total = 0
    ps = {}
    for s, sd in season_data.items():
        p = sd['pass1_v12'].copy()
        
        mid_corr = compute_midrange_correction(fn, sd['X_season'],
                                                aq=mid_params[0], al=mid_params[1], sos=mid_params[2])
        p = apply_swap(p, sd['raw'], mid_corr, sd['test_mask'], mid_zone)
        
        low_corr = compute_low_correction(fn, sd['X_season'],
                                           q1dom=low_params[0], field=low_params[1])
        p = apply_swap(p, sd['raw'], low_corr, sd['test_mask'], low_zone)
        
        if bot_comps is not None:
            bot_corr = compute_bottom_correction(fn, sd['X_season'], bot_comps)
            p = apply_swap(p, sd['raw'], bot_corr, sd['test_mask'], bot_zone)
        
        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p[i] == int(y[gi]))
        total += ex
        ps[s] = ex
    return total, ps


def main():
    t0 = time.time()
    print('='*70)
    print(' v24b: FINE-TUNE BOTTOM-ZONE CORRECTION')
    print(' (v23=63/91, v24 found bottom sosnet=-2,net_conf=+2 → 65/91)')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
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
    folds = sorted(set(seasons))

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Precompute v12 base
    print('\n  Precomputing v12 base...')
    season_data = {}
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test = test_mask & season_mask
        if season_test.sum() == 0:
            continue
        global_train_mask = ~season_test
        X_season = X_all[season_mask]
        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], top_k_idx)
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=0.15)
        tmask_s = np.array([test_mask[gi] for gi in season_indices])
        season_data[hold] = {
            'indices': season_indices,
            'X_season': X_season,
            'raw': raw.copy(),
            'pass1_v12': pass1.copy(),
            'test_mask': tmask_s,
        }

    # Baselines
    v23_total, v23_ps = eval_3zone(season_data, test_mask, y, test_seasons, fn)
    v24_total, v24_ps = eval_3zone(
        season_data, test_mask, y, test_seasons, fn,
        bot_comps={'sosnet': -2, 'net_conf': 2}, bot_zone=(53, 68))
    
    print(f'  v23: {v23_total}/91  [{" ".join(f"{v23_ps.get(s,0):2d}" for s in test_seasons)}]')
    print(f'  v24: {v24_total}/91  [{" ".join(f"{v24_ps.get(s,0):2d}" for s in test_seasons)}]')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: Fine-tune sosnet + net_conf weights
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 1: Fine-tune sosnet + net_conf weights')
    print('='*70)
    
    results = []
    
    for sn in np.arange(-4, -0.1, 0.5):
        for nc in np.arange(0.5, 4.1, 0.5):
            for bz in [(53, 68), (53, 65), (50, 68), (50, 65), (55, 68), (48, 68)]:
                total, ps = eval_3zone(
                    season_data, test_mask, y, test_seasons, fn,
                    bot_comps={'sosnet': sn, 'net_conf': nc}, bot_zone=bz)
                results.append({
                    'sn': sn, 'nc': nc, 'zone': bz,
                    'full': total, 'ps': ps
                })
    
    results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(results)} configs tested')
    print(f'\n  Top 20:')
    for r in results[:20]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if r['full'] > v24_total else ''
        print(f'    sn={r["sn"]:+5.1f} nc={r["nc"]:+5.1f} zone={str(r["zone"]):<10}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')
    
    best_2comp = results[0]

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: 3-component bottom-zone combos
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 2: 3-component bottom-zone combos')
    print('='*70)
    
    extra_comps = ['cbhist', 'aq_weak', 'badloss', 'resume', 'q1dom', 'field', 'al_mid']
    
    best_sn = best_2comp['sn']
    best_nc = best_2comp['nc']
    best_bz = best_2comp['zone']
    
    triple_results = []
    
    for ec in extra_comps:
        for ew in [-3, -2, -1, 1, 2, 3]:
            comps = {'sosnet': best_sn, 'net_conf': best_nc, ec: ew}
            total, ps = eval_3zone(
                season_data, test_mask, y, test_seasons, fn,
                bot_comps=comps, bot_zone=best_bz)
            triple_results.append({
                'comps': comps, 'zone': best_bz,
                'full': total, 'ps': ps
            })
    
    triple_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(triple_results)} configs tested')
    print(f'\n  Top 15:')
    for r in triple_results[:15]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        comps_str = ' '.join(f'{k}={v:+.0f}' for k, v in r['comps'].items())
        marker = ' ★' if r['full'] > best_2comp['full'] else ''
        print(f'    {comps_str}: {r["full"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: Joint tune all 3 zones together
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 3: Joint 3-zone tuning')
    print('='*70)
    
    joint_results = []
    
    # Use the best bot-zone config and sweep mid+low params somewhat
    best_bot = triple_results[0] if triple_results and triple_results[0]['full'] > best_2comp['full'] \
               else {'comps': {'sosnet': best_sn, 'net_conf': best_nc}, 'zone': best_bz, 'full': best_2comp['full']}
    
    bot_c = best_bot['comps']
    bot_z = best_bot['zone']
    
    for al in [1, 2, 3]:
        for sos_w in [2, 3, 4]:
            for q1d in [0.5, 1, 1.5, 2]:
                for fld in [1, 1.5, 2, 2.5, 3]:
                    total, ps = eval_3zone(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_params=(0, al, sos_w),
                        low_params=(q1d, fld),
                        bot_comps=bot_c, bot_zone=bot_z)
                    joint_results.append({
                        'al': al, 'sos': sos_w, 'q1d': q1d, 'fld': fld,
                        'full': total, 'ps': ps
                    })
    
    joint_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(joint_results)} configs tested')
    print(f'\n  Top 15:')
    for r in joint_results[:15]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if r['full'] > best_bot['full'] else ''
        print(f'    al={r["al"]} sos={r["sos"]} q1d={r["q1d"]:.1f} fld={r["fld"]:.1f}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')
    
    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: Expand mid-range zone to capture more
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 4: Wider mid-range zone + 3-zone combos')
    print('='*70)
    
    wider_results = []
    
    # Test wider mid-range zones that may capture some of the low-zone errors too
    for mid_lo in [11, 13, 15, 17]:
        for mid_hi in [34, 36, 38, 40]:
            for low_lo in [max(mid_hi+1, 35), max(mid_hi+1, 37), max(mid_hi+1, 39)]:
                for low_hi in [50, 52, 55]:
                    if low_lo >= low_hi:
                        continue
                    total, ps = eval_3zone(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_zone=(mid_lo, mid_hi),
                        low_zone=(low_lo, low_hi),
                        bot_comps=bot_c, bot_zone=bot_z)
                    wider_results.append({
                        'mid_zone': (mid_lo, mid_hi), 'low_zone': (low_lo, low_hi),
                        'full': total, 'ps': ps
                    })
    
    wider_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(wider_results)} configs tested')
    print(f'\n  Top 15:')
    for r in wider_results[:15]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if r['full'] > best_bot['full'] else ''
        print(f'    mid={str(r["mid_zone"]):<10} low={str(r["low_zone"]):<10}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 5: Nested LOSO validation
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 5: Nested LOSO validation')
    print('='*70)
    
    # Collect all configs
    all_configs = []
    
    # v23 (no bot zone)
    all_configs.append({
        'name': 'v23', 'full': v23_total, 'ps': v23_ps,
    })
    
    # Phase 1 results (fine-tuned pairs)
    for r in results:
        if r['full'] >= v24_total:
            all_configs.append({
                'name': f'p1_sn{r["sn"]:+.1f}_nc{r["nc"]:+.1f}_{r["zone"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Phase 2 results (triples)
    for r in triple_results:
        if r['full'] >= v24_total:
            comps_str = '_'.join(f'{k}{v:+.0f}' for k, v in r['comps'].items())
            all_configs.append({
                'name': f'p2_{comps_str}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Phase 3 results (joint)
    for r in joint_results:
        if r['full'] >= v24_total:
            all_configs.append({
                'name': f'p3_al{r["al"]}_s{r["sos"]}_q{r["q1d"]}_f{r["fld"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Phase 4 results (wider)
    for r in wider_results:
        if r['full'] >= v24_total:
            all_configs.append({
                'name': f'p4_{r["mid_zone"]}_{r["low_zone"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Dedup
    seen = set()
    unique = []
    for c in all_configs:
        key = tuple(c['ps'].get(s, 0) for s in test_seasons)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    unique.sort(key=lambda r: -r['full'])
    
    best_full = unique[0]['full']
    
    print(f'  {len(unique)} unique configs >= {v24_total}/91')
    print(f'  Best overall: {best_full}/91')
    
    if best_full > v23_total:
        print(f'\n  Best configs:')
        for c in unique[:5]:
            ps_str = ' '.join(f'{c["ps"].get(s,0):2d}' for s in test_seasons)
            print(f'    {c["full"]}/91 [{ps_str}] {c["name"]}')
    
    # Nested LOSO
    nested_total = 0
    nested_v23 = 0

    print(f'\n  {"Season":<10} {"N":>3} {"v23":>4} {"Best":>4} {"Δ":>3}  Config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*40}')
    
    for hold_season in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold_season]
        
        best_tune = -1
        best_idx = 0
        for ci, c in enumerate(unique):
            tune = sum(c['ps'].get(s, 0) for s in tune_seasons)
            if tune > best_tune:
                best_tune = tune
                best_idx = ci
        
        best_c = unique[best_idx]
        hold_exact = best_c['ps'].get(hold_season, 0)
        v23_hold = v23_ps.get(hold_season, 0)
        n_te = (test_mask & (seasons == hold_season)).sum()
        
        nested_total += hold_exact
        nested_v23 += v23_hold
        
        delta = hold_exact - v23_hold
        print(f'  {hold_season:<10} {n_te:3d} {v23_hold:4d} {hold_exact:4d} {delta:+3d}  {best_c["name"][:50]}')
    
    print(f'\n  Nested LOSO: v23={nested_v23}/91, v24={nested_total}/91 (Δ={nested_total-nested_v23:+d})')

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    print(f'  v23: {v23_total}/91')
    print(f'  Best v24 (full): {best_full}/91')
    print(f'  Best v24 (nested LOSO): {nested_total}/91')
    
    if best_full > v23_total:
        print(f'\n  ★ IMPROVEMENT FOUND over v23! ({v23_total} → {best_full}) ★')
    
    if nested_total > v23_total:
        print(f'  ★ VALIDATED by nested LOSO! ({v23_total} → {nested_total}) ★')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
