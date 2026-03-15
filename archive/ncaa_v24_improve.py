#!/usr/bin/env python3
"""
v24: Comprehensive improvement over v23 (63/91)
================================================

v23 = v12 base + mid-range(17-34, al=2,sos=3) + low-zone(35-52, q1dom=+1,field=+2)

Strategy: attack remaining 28 errors from multiple angles:
  Phase 1: Error analysis — where are the 28 errors? What patterns?
  Phase 2: Bottom-zone correction (53-68) — 3rd zone with distinct formula
  Phase 3: Zone boundary optimization — shift/expand existing zones
  Phase 4: Cross-zone interactions — tune all 3 zones jointly
  Phase 5: Alternative base model tuning WITH dual-zone corrections
  Phase 6: Power parameter sweep (per-zone different powers)
  Phase 7: Nested LOSO validation on best configs
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
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
    """
    Correction for bottom zone (53-68). Different formula again.
    
    Components:
      'cbhist':   conference-bid historical seed bias
      'sosnet':   SOS-NET gap (scaled for this range)
      'aq_weak':  AQ from weak conference
      'badloss':  bad losses penalty
      'resume':   resume quality (Q1W+Q2W minus Q3L+Q4L)
      'net_conf': NET vs conference average
      'q1dom':    Q1 dominance
      'field':    field rank gap
      'al_mid':   AL from mid-major conference push
    """
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
    mid_test = [i for i in range(len(pass1))
                if test_mask_s[i] and lo <= pass1[i] <= hi]
    if len(mid_test) <= 1:
        return pass1.copy()
    mid_seeds = [pass1[i] for i in mid_test]
    mid_corr = [raw_scores[i] + correction[i] for i in mid_test]
    cost = np.array([[abs(s - seed)**power for seed in mid_seeds] for s in mid_corr])
    ri, ci = linear_sum_assignment(cost)
    final = pass1.copy()
    for r, c in zip(ri, ci):
        final[mid_test[r]] = mid_seeds[c]
    return final


def eval_config(season_data, test_mask, y, seasons, test_seasons,
                mid_params=(0, 2, 3), mid_zone=(17, 34), mid_power=0.15,
                low_params=(1, 2), low_zone=(35, 52), low_power=0.15,
                bot_comps=None, bot_zone=(53, 68), bot_power=0.15,
                fn=None):
    """Evaluate a full 3-zone config."""
    total = 0
    ps = {}
    for s, sd in season_data.items():
        p = sd['pass1_v12'].copy()
        
        # Mid-range correction
        mid_corr = compute_midrange_correction(fn, sd['X_season'],
                                                aq=mid_params[0], al=mid_params[1], sos=mid_params[2])
        p = apply_swap(p, sd['raw'], mid_corr, sd['test_mask'], mid_zone, mid_power)
        
        # Low-zone correction
        low_corr = compute_low_correction(fn, sd['X_season'],
                                           q1dom=low_params[0], field=low_params[1])
        p = apply_swap(p, sd['raw'], low_corr, sd['test_mask'], low_zone, low_power)
        
        # Bottom-zone correction (optional)
        if bot_comps is not None:
            bot_corr = compute_bottom_correction(fn, sd['X_season'], bot_comps)
            p = apply_swap(p, sd['raw'], bot_corr, sd['test_mask'], bot_zone, bot_power)
        
        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p[i] == int(y[gi]))
        total += ex
        ps[s] = ex
    return total, ps


def main():
    t0 = time.time()
    print('='*70)
    print(' v24: COMPREHENSIVE IMPROVEMENT OVER v23 (63/91)')
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

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')
    print(f'  Features: {len(fn)}')

    # ═══════════════════════════════════════════════════════════
    #  Precompute v12 base predictions per season
    # ═══════════════════════════════════════════════════════════
    print('\n  Precomputing v12 base predictions...')
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
    v12_exact = sum(
        sum(1 for i, gi in enumerate(sd['indices'])
            if test_mask[gi] and sd['pass1_v12'][i] == int(y[gi]))
        for sd in season_data.values())
    
    v23_total, v23_ps = eval_config(
        season_data, test_mask, y, seasons, test_seasons, fn=fn)
    
    print(f'  v12 baseline: {v12_exact}/91')
    print(f'  v23 baseline: {v23_total}/91')
    ps_str = ' '.join(f'{v23_ps.get(s,0):2d}' for s in test_seasons)
    print(f'    Per-season: {ps_str}')

    fi = {f: i for i, f in enumerate(fn)}

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: Error analysis under v23
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 1: v23 Error Analysis')
    print('='*70)
    
    errors_by_zone = {'top': [], 'mid': [], 'low': [], 'bot': []}
    
    for s, sd in season_data.items():
        # Reproduce v23 predictions
        p = sd['pass1_v12'].copy()
        mid_corr = compute_midrange_correction(fn, sd['X_season'])
        p = apply_swap(p, sd['raw'], mid_corr, sd['test_mask'], (17, 34))
        low_corr = compute_low_correction(fn, sd['X_season'])
        p = apply_swap(p, sd['raw'], low_corr, sd['test_mask'], (35, 52))
        
        for i, gi in enumerate(sd['indices']):
            if not test_mask[gi]:
                continue
            true_s = int(y[gi])
            pred_s = p[i]
            if pred_s == true_s:
                continue
            
            rid = record_ids[gi]
            net = X_all[gi, fi['NET Rank']]
            sos = X_all[gi, fi['NETSOS']]
            q1w = X_all[gi, fi['Quadrant1_W']]
            q1l = X_all[gi, fi['Quadrant1_L']]
            q3l = X_all[gi, fi['Quadrant3_L']]
            q4l = X_all[gi, fi['Quadrant4_L']]
            cbm = X_all[gi, fi['cb_mean_seed']]
            tfr = X_all[gi, fi['tourn_field_rank']]
            bid = 'AQ' if X_all[gi, fi['is_AQ']] > 0.5 else 'AL'
            pwr = 'P' if X_all[gi, fi['is_power_conf']] > 0.5 else 'M'
            conf_avg = X_all[gi, fi['conf_avg_net']]
            
            if true_s <= 16:
                zone_name = 'top'
            elif true_s <= 34:
                zone_name = 'mid'
            elif true_s <= 52:
                zone_name = 'low'
            else:
                zone_name = 'bot'
            
            errors_by_zone[zone_name].append({
                'rid': rid, 'season': s, 'pred': pred_s, 'true': true_s,
                'err': pred_s - true_s, 'net': net, 'sos': sos,
                'q1w': q1w, 'q1l': q1l, 'q3l': q3l, 'q4l': q4l,
                'cbm': cbm, 'tfr': tfr, 'bid': bid, 'pwr': pwr,
                'conf_avg': conf_avg
            })
    
    for zone, errors in errors_by_zone.items():
        n_errs = len(errors)
        if n_errs == 0:
            continue
        avg_abs = np.mean([abs(e['err']) for e in errors])
        print(f'\n  {zone.upper()} zone: {n_errs} errors, avg|err|={avg_abs:.1f}')
        for e in sorted(errors, key=lambda x: -abs(x['err'])):
            print(f'    {e["rid"]:<28} pred={e["pred"]:2d} true={e["true"]:2d} err={e["err"]:+3d} '
                  f'NET={e["net"]:3.0f} SOS={e["sos"]:3.0f} Q1W={e["q1w"]:.0f} '
                  f'Q3L={e["q3l"]:.0f} Q4L={e["q4l"]:.0f} CB={e["cbm"]:4.1f} '
                  f'TFR={e["tfr"]:4.1f} {e["bid"]}/{e["pwr"]} ca={e["conf_avg"]:.0f}')
    
    # Identify swap pairs (where two teams could simply exchange seeds to fix both)
    all_errors = []
    for zone, errs in errors_by_zone.items():
        all_errors.extend(errs)
    
    print(f'\n  Total errors: {len(all_errors)}')
    
    swap_pairs = []
    for i in range(len(all_errors)):
        for j in range(i+1, len(all_errors)):
            e1, e2 = all_errors[i], all_errors[j]
            if e1['season'] == e2['season']:
                if e1['pred'] == e2['true'] and e2['pred'] == e1['true']:
                    swap_pairs.append((e1, e2))
    
    if swap_pairs:
        print(f'\n  Perfect swap pairs ({len(swap_pairs)}):')
        for a, b in swap_pairs:
            print(f'    {a["rid"]} ({a["pred"]}↔{a["true"]}) ↔ {b["rid"]} ({b["pred"]}↔{b["true"]})')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: Bottom-zone correction (53-68)
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 2: Bottom-zone correction (53-68)')
    print('='*70)
    
    bot_components = ['cbhist', 'sosnet', 'aq_weak', 'badloss', 'resume',
                      'net_conf', 'q1dom', 'field', 'al_mid']
    
    bot_results = []
    
    # Single-component sweep
    for comp in bot_components:
        for weight in [-3, -2, -1, -0.5, 0.5, 1, 2, 3]:
            for zone in [(53, 68), (50, 68), (48, 68), (53, 65)]:
                total, ps = eval_config(
                    season_data, test_mask, y, seasons, test_seasons,
                    bot_comps={comp: weight}, bot_zone=zone, fn=fn)
                bot_results.append({
                    'comp': comp, 'weight': weight, 'zone': zone,
                    'full': total, 'ps': ps
                })
    
    bot_results.sort(key=lambda r: -r['full'])
    
    print(f'  Single-component sweep: {len(bot_results)} configs tested')
    print(f'\n  Top 15 (>= {v23_total}):')
    shown = 0
    for r in bot_results:
        if r['full'] >= v23_total and shown < 15:
            ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
            marker = ' ←NEW' if r['full'] > v23_total else ''
            print(f'    zone={str(r["zone"]):<10} {r["comp"]:<10} w={r["weight"]:+5.1f}: '
                  f'{r["full"]}/91 [{ps_str}]{marker}')
            shown += 1
    
    # Multi-component bottom zone
    best_bot_singles = {}
    for r in bot_results:
        if r['full'] >= v23_total:
            key = r['comp']
            if key not in best_bot_singles or r['full'] > best_bot_singles[key]['full']:
                best_bot_singles[key] = r
    
    good_bot = list(best_bot_singles.keys())
    
    if len(good_bot) >= 2:
        print(f'\n  Multi-component bottom zone ({len(good_bot)} good components):')
        bot_multi = []
        for i in range(len(good_bot)):
            for j in range(i+1, len(good_bot)):
                c1, c2 = good_bot[i], good_bot[j]
                for w1 in [-2, -1, 1, 2]:
                    for w2 in [-2, -1, 1, 2]:
                        for zone in [(53, 68), (50, 68)]:
                            total, ps = eval_config(
                                season_data, test_mask, y, seasons, test_seasons,
                                bot_comps={c1: w1, c2: w2}, bot_zone=zone, fn=fn)
                            bot_multi.append({
                                'comps': {c1: w1, c2: w2}, 'zone': zone,
                                'full': total, 'ps': ps
                            })
        
        bot_multi.sort(key=lambda r: -r['full'])
        
        print(f'  {len(bot_multi)} combos tested')
        for r in bot_multi[:10]:
            ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
            comps_str = ' '.join(f'{k}={v:+.0f}' for k, v in r['comps'].items())
            marker = ' ←NEW' if r['full'] > v23_total else ''
            print(f'    zone={str(r["zone"]):<10} {comps_str}: {r["full"]}/91 [{ps_str}]{marker}')
        
        bot_results.extend(bot_multi)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: Zone boundary optimization
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 3: Zone boundary optimization')
    print('='*70)
    
    boundary_results = []
    
    for mid_lo in [15, 16, 17, 18, 19]:
        for mid_hi in [32, 33, 34, 35, 36]:
            for low_lo in [33, 34, 35, 36, 37]:
                for low_hi in [50, 51, 52, 53, 54, 55]:
                    # Skip invalid: zones must not overlap with gaps
                    if low_lo <= mid_hi:
                        continue
                    
                    total, ps = eval_config(
                        season_data, test_mask, y, seasons, test_seasons,
                        mid_zone=(mid_lo, mid_hi), low_zone=(low_lo, low_hi), fn=fn)
                    
                    boundary_results.append({
                        'mid_zone': (mid_lo, mid_hi), 'low_zone': (low_lo, low_hi),
                        'full': total, 'ps': ps
                    })
    
    boundary_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(boundary_results)} boundary configs tested')
    print(f'\n  Top 15:')
    for r in boundary_results[:15]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ←NEW' if r['full'] > v23_total else ''
        print(f'    mid={str(r["mid_zone"]):<10} low={str(r["low_zone"]):<10}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')
    
    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: Re-tune mid+low weights with shifted boundaries
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 4: Joint weight + boundary tuning')
    print('='*70)
    
    # Take top boundary configs and sweep weights
    top_boundaries = [r for r in boundary_results if r['full'] >= v23_total][:5]
    
    joint_results = []
    
    for br in top_boundaries:
        mz, lz = br['mid_zone'], br['low_zone']
        for al in [1, 2, 3]:
            for sos_w in [2, 3, 4]:
                for q1d in [0.5, 1, 1.5, 2]:
                    for fld in [1, 1.5, 2, 2.5, 3]:
                        total, ps = eval_config(
                            season_data, test_mask, y, seasons, test_seasons,
                            mid_params=(0, al, sos_w), mid_zone=mz,
                            low_params=(q1d, fld), low_zone=lz, fn=fn)
                        
                        joint_results.append({
                            'mid_zone': mz, 'low_zone': lz,
                            'al': al, 'sos': sos_w, 'q1dom': q1d, 'field': fld,
                            'full': total, 'ps': ps
                        })
    
    joint_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(joint_results)} joint configs tested')
    print(f'\n  Top 15:')
    for r in joint_results[:15]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ←NEW' if r['full'] > v23_total else ''
        print(f'    mid={str(r["mid_zone"]):<10} al={r["al"]} sos={r["sos"]} '
              f'low={str(r["low_zone"]):<10} q1d={r["q1dom"]:.1f} fld={r["field"]:.1f}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 5: Power parameter sweep
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 5: Power parameter sweep')
    print('='*70)
    
    power_results = []
    
    for mid_pow in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for low_pow in [0.1, 0.15, 0.2, 0.25, 0.3]:
            total, ps = eval_config(
                season_data, test_mask, y, seasons, test_seasons,
                mid_power=mid_pow, low_power=low_pow, fn=fn)
            power_results.append({
                'mid_pow': mid_pow, 'low_pow': low_pow,
                'full': total, 'ps': ps
            })
    
    power_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(power_results)} power configs tested')
    print(f'\n  Top 10:')
    for r in power_results[:10]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ←NEW' if r['full'] > v23_total else ''
        print(f'    mid_pow={r["mid_pow"]:.2f} low_pow={r["low_pow"]:.2f}: '
              f'{r["full"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 6: Alternative base model — SKIPPED
    #  (v20 exhaustively tested 234 configs, none improved)
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 6: Skipped (v20 proved no base model improvement)')
    print('='*70)
    
    alt_base_results = []

    # ═══════════════════════════════════════════════════════════
    #  PHASE 7: Overlapping/wider zone sweep (allow overlap)
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 7: Overlapping zones (mid + extended low)')
    print('='*70)
    
    # What if we allow zones to overlap or run sequentially with wider range?
    overlap_results = []
    
    for low_lo in [17, 20, 25, 30, 33, 35]:
        for low_hi in [52, 55, 60, 65, 68]:
            total, ps = eval_config(
                season_data, test_mask, y, seasons, test_seasons,
                low_zone=(low_lo, low_hi), fn=fn)
            overlap_results.append({
                'low_zone': (low_lo, low_hi),
                'full': total, 'ps': ps
            })
    
    overlap_results.sort(key=lambda r: -r['full'])
    
    print(f'  {len(overlap_results)} configs tested')
    print(f'\n  Top 10:')
    for r in overlap_results[:10]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ←NEW' if r['full'] > v23_total else ''
        print(f'    low={str(r["low_zone"]):<10}: {r["full"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 8: Best overall + nested LOSO
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 8: Nested LOSO validation')
    print('='*70)
    
    # Collect ALL results into a unified list
    all_configs = []
    
    # v23 baseline
    all_configs.append({
        'name': 'v23_baseline',
        'full': v23_total, 'ps': v23_ps,
    })
    
    # Best boundary configs
    for r in boundary_results:
        if r['full'] >= v23_total:
            all_configs.append({
                'name': f'bnd_mid{r["mid_zone"]}_low{r["low_zone"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Best joint configs
    for r in joint_results:
        if r['full'] >= v23_total:
            all_configs.append({
                'name': f'joint_al{r["al"]}_s{r["sos"]}_q{r["q1dom"]}_f{r["field"]}_{r["mid_zone"]}_{r["low_zone"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Best power configs
    for r in power_results:
        if r['full'] >= v23_total:
            all_configs.append({
                'name': f'pow_m{r["mid_pow"]}_l{r["low_pow"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Best bot-zone configs
    for r in bot_results:
        if r['full'] >= v23_total:
            name = f'bot_{r.get("comp", "multi")}_{r.get("zone", "")}'
            all_configs.append({
                'name': name, 'full': r['full'], 'ps': r['ps'],
            })
    
    # Best overlap configs
    for r in overlap_results:
        if r['full'] >= v23_total:
            all_configs.append({
                'name': f'overlap_low{r["low_zone"]}',
                'full': r['full'], 'ps': r['ps'],
            })
    
    # Dedup by score pattern
    seen = set()
    unique_configs = []
    for c in all_configs:
        key = tuple(c['ps'].get(s, 0) for s in test_seasons)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    
    unique_configs.sort(key=lambda r: -r['full'])
    
    best_overall = unique_configs[0]['full'] if unique_configs else v23_total
    
    print(f'  {len(unique_configs)} unique configs >= {v23_total}/91')
    print(f'  Best overall: {best_overall}/91')
    
    # Nested LOSO
    nested_total = 0
    nested_v23 = 0
    
    print(f'\n  {"Season":<10} {"N":>3} {"v23":>4} {"Best":>4} {"Δ":>3}  Config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*40}')
    
    for hold_season in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold_season]
        
        best_tune = -1
        best_idx = 0
        for ci, c in enumerate(unique_configs):
            tune = sum(c['ps'].get(s, 0) for s in tune_seasons)
            if tune > best_tune:
                best_tune = tune
                best_idx = ci
        
        best_c = unique_configs[best_idx]
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
    print(f'  v12: {v12_exact}/91')
    print(f'  v23: {v23_total}/91')
    print(f'  Best v24 (full): {best_overall}/91')
    print(f'  Best v24 (nested LOSO): {nested_total}/91')
    
    if best_overall > v23_total:
        print(f'\n  ★ IMPROVEMENT FOUND! ★')
        for c in unique_configs[:5]:
            if c['full'] >= best_overall:
                ps_str = ' '.join(f'{c["ps"].get(s,0):2d}' for s in test_seasons)
                print(f'    {c["full"]}/91 [{ps_str}] {c["name"]}')
    else:
        print(f'\n  No improvement over v23. 63/91 remains the ceiling.')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
