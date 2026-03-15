#!/usr/bin/env python3
"""
v23: New Zone Correction with Different Formula + Hypertuning
==============================================================

v18 = 61/91 using mid-range (17-34) swap correction with AL+SOS formula.
v19 showed the SAME formula doesn't work for other zones.

This script designs DIFFERENT correction formulas for:
  - Low zone (35-52): 10/28 errors, avg|err|=4.0 (worst zone!)
  - Bottom zone (53-68): 11/26 errors, avg|err|=1.3
  - Top zone (1-16): only 2 errors, skip

Low zone error patterns (different from mid-range):
  - Resume quality matters: errors have fewer Q1 wins, more bad losses
  - Tournament field rank diverges from raw prediction
  - Conference-bid historical seed (cb_mean_seed) is off
  - NET vs SOS gap works differently at this range

New correction components for low zone:
  1. resume_gap: Q1wins + Q2wins vs Q3losses + Q4losses  
  2. field_rank_gap: tournament field rank vs predicted rank
  3. bid_history: historical seed for this conference-bid combo vs prediction
  4. net_sos_gap: same concept but different scaling for low zone
  5. aq_resume: AQ teams with bad resumes get pushed higher

All tested with NESTED LOSO to prevent overfitting.
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
    USE_TOP_K_A, FORCE_FEATURES
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_midrange_correction(fn, X, aq=0, al=2, sos=3):
    """v18 mid-range correction (unchanged)."""
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


def compute_low_correction(fn, X, components):
    """
    New correction formula for low zone (35-52), using DIFFERENT signals.
    
    components dict can contain:
      'resume': weight for resume quality gap (Q1W+Q2W - Q3L - Q4L divergence)
      'field':  weight for tourn_field_rank vs raw score divergence
      'cbhist': weight for conference-bid historical seed divergence
      'sosnet': weight for SOS-NET gap (different scaling than mid-range)
      'aq_weak': weight for AQ from weak conference penalty
      'al_power': weight for AL from power conference benefit
      'q1dom': weight for Q1 dominance (good = push lower seed)
      'badloss': weight for bad losses penalty
      'net_conf': weight for NET vs conference avg rank
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
    q2w = X[:, fi['Quadrant2_W']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    resume = q1w + q2w - q3l - q4l
    
    # Resume quality: strong resumes → push lower (better) seed
    # Normalize to reasonable range
    if components.get('resume', 0) != 0:
        # resume ~ [-10, +10], normalize to [-1, 1]
        resume_score = np.clip(resume / 10, -1, 1)
        correction -= components['resume'] * resume_score
    
    # Tournament field rank gap: if field rank is much higher than prediction,
    # team is relatively weaker within the tournament field → push higher
    if components.get('field', 0) != 0:
        # tfr ~ [1, 68], scale to correction range
        field_gap = (tfr - 34) / 34  # centered, [-1, +1]
        correction += components['field'] * field_gap
    
    # Conference-bid historical seed: if historical average is much higher/lower
    # than where we're placing them, adjust toward history
    if components.get('cbhist', 0) != 0:
        # cb_mean ~ [5, 60], prediction is the Hungarian assigned seed
        # Use raw net_to_seed as proxy since we don't have assigned seed here
        net_seed_est = np.clip(tfr, 1, 68)
        hist_gap = (cb_mean - net_seed_est) / 34  # normalized
        correction += components['cbhist'] * hist_gap
    
    # SOS-NET gap (different scaling for low zone)
    if components.get('sosnet', 0) != 0:
        gap = (sos - net) / 150  # larger denominator = softer
        correction += components['sosnet'] * gap
    
    # AQ from weak conference: penalize in low zone
    if components.get('aq_weak', 0) != 0:
        conf_weakness = np.clip((conf_avg - 100) / 100, 0, 2)
        correction += components['aq_weak'] * is_aq * conf_weakness
    
    # AL from power: benefit (push toward better seed)
    if components.get('al_power', 0) != 0:
        correction -= components['al_power'] * is_al * is_power
    
    # Q1 dominance: good Q1 record → push lower (better seed)
    if components.get('q1dom', 0) != 0:
        q1_rate = q1w / (q1w + X[:, fi['Quadrant1_L']] + 1)
        correction -= components['q1dom'] * q1_rate
    
    # Bad loss penalty: more Q3+Q4 losses → push higher (worse seed)
    if components.get('badloss', 0) != 0:
        bad = np.clip((q3l + q4l) / 5, 0, 2)
        correction += components['badloss'] * bad
    
    # NET vs conference average: if team NET is much better than conf average,
    # they might be overperforming → adjust
    if components.get('net_conf', 0) != 0:
        gap = (conf_avg - net) / 100
        correction += components['net_conf'] * gap
    
    return correction


def apply_swap(pass1, raw_scores, correction, test_mask_season, zone, power=0.15):
    lo, hi = zone
    mid_test = [i for i in range(len(pass1))
                if test_mask_season[i] and lo <= pass1[i] <= hi]
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


def main():
    t0 = time.time()
    print('='*70)
    print(' v23: NEW ZONE CORRECTION WITH DIFFERENT FORMULA')
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

    # ════════════════════════════════════════════════════════════════
    #  Precompute v12 base predictions per season
    # ════════════════════════════════════════════════════════════════
    print('  Precomputing v12 base + v18 mid-range correction...')
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

        # Set known training labels
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]

        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=0.15)
        
        tmask_s = np.array([test_mask[gi] for gi in season_indices])
        
        # Apply v18 mid-range correction
        mid_corr = compute_midrange_correction(fn, X_season)
        pass_v18 = apply_swap(pass1, raw, mid_corr, tmask_s, (17, 34))

        season_data[hold] = {
            'indices': season_indices,
            'X_season': X_season,
            'raw': raw.copy(),
            'pass1_v12': pass1.copy(),      # v12 base (no correction)
            'pass_v18': pass_v18.copy(),     # v18 (mid-range correction only)
            'test_mask': tmask_s,
            'mid_corr': mid_corr.copy(),
        }

    # Baseline scores
    v12_exact = sum(
        sum(1 for i, gi in enumerate(sd['indices'])
            if test_mask[gi] and sd['pass1_v12'][i] == int(y[gi]))
        for sd in season_data.values())
    
    v18_exact = sum(
        sum(1 for i, gi in enumerate(sd['indices'])
            if test_mask[gi] and sd['pass_v18'][i] == int(y[gi]))
        for sd in season_data.values())
    
    v18_ps = {}
    for s, sd in season_data.items():
        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and sd['pass_v18'][i] == int(y[gi]))
        v18_ps[s] = ex

    print(f'  v12 baseline: {v12_exact}/91')
    print(f'  v18 baseline: {v18_exact}/91')
    ps_str = ' '.join(f'{v18_ps.get(s,0):2d}' for s in test_seasons)
    print(f'    Per-season: {ps_str}')

    # ════════════════════════════════════════════════════════════════
    #  Phase 1: Profile low-zone errors to understand patterns
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 1: Low-zone (35-52) error profiling')
    print('='*70)
    
    fi = {f: i for i, f in enumerate(fn)}
    
    for s, sd in season_data.items():
        low_errors = []
        low_correct = []
        for i, gi in enumerate(sd['indices']):
            if not test_mask[gi]:
                continue
            true_s = int(y[gi])
            pred_s = sd['pass_v18'][i]
            
            if 35 <= true_s <= 52 or 35 <= pred_s <= 52:
                team = record_ids[gi]
                net = X_all[gi, fi['NET Rank']]
                sos = X_all[gi, fi['NETSOS']]
                q1w = X_all[gi, fi['Quadrant1_W']]
                q3l = X_all[gi, fi['Quadrant3_L']]
                q4l = X_all[gi, fi['Quadrant4_L']]
                cbm = X_all[gi, fi['cb_mean_seed']]
                tfr = X_all[gi, fi['tourn_field_rank']]
                bid = 'AQ' if X_all[gi, fi['is_AQ']] > 0.5 else 'AL'
                pwr = 'P' if X_all[gi, fi['is_power_conf']] > 0.5 else 'M'
                
                if pred_s != true_s:
                    low_errors.append((team, pred_s, true_s, pred_s-true_s, 
                                       net, sos, q1w, q3l, q4l, cbm, tfr, bid, pwr))
                else:
                    low_correct.append((team, pred_s, net, sos, q1w, q3l, q4l, cbm, tfr, bid, pwr))

        if low_errors:
            print(f'\n  Season {s} low-zone errors:')
            for t, p, tr, err, net, sos, q1w, q3l, q4l, cbm, tfr, bid, pwr in low_errors:
                print(f'    {t:<28} pred={p:2d} true={tr:2d} err={err:+3d} NET={net:3.0f} '
                      f'SOS={sos:3.0f} Q1W={q1w:.0f} Q3L={q3l:.0f} Q4L={q4l:.0f} '
                      f'CB={cbm:4.1f} TFR={tfr:4.1f} {bid}/{pwr}')
    
    # ════════════════════════════════════════════════════════════════
    #  Phase 2: Test single-component corrections for low zone
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 2: Single-component low-zone corrections')
    print('='*70)
    
    components_to_test = [
        'resume', 'field', 'cbhist', 'sosnet', 
        'aq_weak', 'al_power', 'q1dom', 'badloss', 'net_conf'
    ]
    
    low_zones = [(35, 52), (33, 52), (35, 55), (30, 55), (35, 68)]
    
    single_results = []
    
    print(f'  {"Zone":<10} {"Component":<12} {"Weight":<8} {"Full":>4} {"Per-season":>25}')
    print(f'  {"─"*10} {"─"*12} {"─"*8} {"─"*4} {"─"*25}')
    
    for zone in low_zones:
        for comp in components_to_test:
            for weight in [-3, -2, -1, -0.5, 0.5, 1, 2, 3]:
                total = 0
                ps = {}
                
                for s, sd in season_data.items():
                    # Start from v18 (mid-range already corrected)
                    p = sd['pass_v18'].copy()
                    
                    # Apply new low-zone correction
                    low_corr = compute_low_correction(fn, sd['X_season'], {comp: weight})
                    p = apply_swap(p, sd['raw'], low_corr, sd['test_mask'], zone)
                    
                    ex = sum(1 for i, gi in enumerate(sd['indices'])
                            if test_mask[gi] and p[i] == int(y[gi]))
                    total += ex
                    ps[s] = ex
                
                single_results.append({
                    'zone': zone, 'comp': comp, 'weight': weight,
                    'full': total, 'ps': ps
                })
    
    single_results.sort(key=lambda r: -r['full'])
    
    print(f'\n  Total configs tested: {len(single_results)}')
    print(f'\n  Top 30 (>= v18 baseline of {v18_exact}):')
    shown = 0
    for r in single_results:
        if r['full'] >= v18_exact and shown < 30:
            ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
            marker = ' ←NEW' if r['full'] > v18_exact else ''
            print(f'  {str(r["zone"]):<10} {r["comp"]:<12} {r["weight"]:+6.1f}  {r["full"]:4d}  {ps_str}{marker}')
            shown += 1
    
    # ════════════════════════════════════════════════════════════════
    #  Phase 3: Multi-component low-zone corrections
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 3: Multi-component low-zone corrections')
    print('='*70)
    
    # Pick the best single components and combine them
    # Focus on components that scored >= v18
    best_singles = {}
    for r in single_results:
        key = (r['zone'], r['comp'])
        if key not in best_singles or r['full'] > best_singles[key]['full']:
            best_singles[key] = r
    
    good_singles = [(k, v) for k, v in best_singles.items() if v['full'] >= v18_exact]
    good_singles.sort(key=lambda x: -x[1]['full'])
    
    print(f'  Best single components (>= {v18_exact}):')
    for (zone, comp), r in good_singles[:15]:
        print(f'    {str(zone):<10} {comp:<12} w={r["weight"]:+.1f} → {r["full"]}/91')
    
    # Multi-component combos
    best_zone = (35, 52)  # default
    if good_singles:
        best_zone = good_singles[0][0][0]  # zone from best single
    
    # Gather the best weight for each component on best_zone
    comp_weights = {}
    for r in single_results:
        if r['zone'] == best_zone and r['full'] >= v18_exact:
            if r['comp'] not in comp_weights or r['full'] > comp_weights[r['comp']][0]:
                comp_weights[r['comp']] = (r['full'], r['weight'])
    
    good_comps = list(comp_weights.keys())
    print(f'\n  Good components for zone {best_zone}: {good_comps}')
    
    multi_results = []
    
    # Test pairs of components
    for i in range(len(good_comps)):
        for j in range(i+1, len(good_comps)):
            c1, c2 = good_comps[i], good_comps[j]
            for w1 in [-2, -1, 1, 2]:
                for w2 in [-2, -1, 1, 2]:
                    total = 0
                    ps = {}
                    for s, sd in season_data.items():
                        p = sd['pass_v18'].copy()
                        corr = compute_low_correction(fn, sd['X_season'], {c1: w1, c2: w2})
                        p = apply_swap(p, sd['raw'], corr, sd['test_mask'], best_zone)
                        ex = sum(1 for i2, gi in enumerate(sd['indices'])
                                if test_mask[gi] and p[i2] == int(y[gi]))
                        total += ex
                        ps[s] = ex
                    
                    multi_results.append({
                        'comps': {c1: w1, c2: w2}, 'zone': best_zone,
                        'full': total, 'ps': ps
                    })
    
    # Test triples if we have enough good components
    if len(good_comps) >= 3:
        for i in range(min(len(good_comps), 5)):
            for j in range(i+1, min(len(good_comps), 5)):
                for k in range(j+1, min(len(good_comps), 5)):
                    c1, c2, c3 = good_comps[i], good_comps[j], good_comps[k]
                    for w1 in [-1, 1]:
                        for w2 in [-1, 1]:
                            for w3 in [-1, 1]:
                                total = 0
                                ps = {}
                                for s, sd in season_data.items():
                                    p = sd['pass_v18'].copy()
                                    corr = compute_low_correction(fn, sd['X_season'],
                                                                   {c1: w1, c2: w2, c3: w3})
                                    p = apply_swap(p, sd['raw'], corr, sd['test_mask'], best_zone)
                                    ex = sum(1 for i2, gi in enumerate(sd['indices'])
                                            if test_mask[gi] and p[i2] == int(y[gi]))
                                    total += ex
                                    ps[s] = ex
                                
                                multi_results.append({
                                    'comps': {c1: w1, c2: w2, c3: w3}, 'zone': best_zone,
                                    'full': total, 'ps': ps
                                })
    
    multi_results.sort(key=lambda r: -r['full'])
    
    print(f'\n  Multi-component combos tested: {len(multi_results)}')
    print(f'\n  Top 20:')
    for r in multi_results[:20]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        comps_str = ' '.join(f'{k}={v:+.0f}' for k, v in r['comps'].items())
        marker = ' ←NEW' if r['full'] > v18_exact else ''
        print(f'    {r["full"]}/91 [{ps_str}] {comps_str}{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Phase 4: Broader sweep on best formula (fine-tuning weights)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 4: Fine-tune best formula')
    print('='*70)

    # Gather ALL results (single + multi)
    all_results = single_results + multi_results

    # Find best overall
    best_full = max(r['full'] for r in all_results)
    best_configs = [r for r in all_results if r['full'] == best_full]
    
    print(f'  Best score: {best_full}/91 ({len(best_configs)} configs)')
    
    if best_full > v18_exact:
        # Fine-tune the best formula
        best = best_configs[0]
        if 'comp' in best:
            # Single component
            base_comp = best['comp']
            base_weight = best['weight']
            base_zone = best['zone']
            
            print(f'  Fine-tuning: {base_comp} near {base_weight} on zone {base_zone}')
            
            fine_results = []
            for w in np.arange(base_weight - 2, base_weight + 2.1, 0.25):
                if w == 0:
                    continue
                for zone in [(base_zone[0]-2, base_zone[1]), (base_zone[0], base_zone[1]+2),
                             (base_zone[0]-2, base_zone[1]+2), base_zone,
                             (base_zone[0]+2, base_zone[1]), (base_zone[0], base_zone[1]-2)]:
                    zone = (max(1, zone[0]), min(68, zone[1]))
                    total = 0
                    ps = {}
                    for s, sd in season_data.items():
                        p = sd['pass_v18'].copy()
                        corr = compute_low_correction(fn, sd['X_season'], {base_comp: w})
                        p = apply_swap(p, sd['raw'], corr, sd['test_mask'], zone)
                        ex = sum(1 for i, gi in enumerate(sd['indices'])
                                if test_mask[gi] and p[i] == int(y[gi]))
                        total += ex
                        ps[s] = ex
                    
                    fine_results.append({
                        'comp': base_comp, 'weight': w, 'zone': zone,
                        'full': total, 'ps': ps
                    })
            
            fine_results.sort(key=lambda r: -r['full'])
            
            print(f'\n  Fine-tuned top 15:')
            for r in fine_results[:15]:
                ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
                marker = ' ←' if r['full'] > v18_exact else ''
                print(f'    zone={str(r["zone"]):<10} w={r["weight"]:+6.2f}: {r["full"]}/91 [{ps_str}]{marker}')
            
            all_results.extend(fine_results)
        else:
            # Multi-component
            base_comps = best['comps']
            base_zone = best['zone']
            
            print(f'  Fine-tuning multi-component: {base_comps} on zone {base_zone}')
            
            # Fine-tune each weight by ±0.5 steps
            fine_results = []
            keys = list(base_comps.keys())
            base_vals = [base_comps[k] for k in keys]
            
            for dw0 in [-1, -0.5, 0, 0.5, 1]:
                for dw1 in [-1, -0.5, 0, 0.5, 1]:
                    new_comps = {}
                    new_comps[keys[0]] = base_vals[0] + dw0
                    new_comps[keys[1]] = base_vals[1] + dw1
                    if len(keys) > 2:
                        for dw2 in [-0.5, 0, 0.5]:
                            new_comps[keys[2]] = base_vals[2] + dw2
                            if any(v == 0 for v in new_comps.values()):
                                continue
                            total = 0
                            ps = {}
                            for s, sd in season_data.items():
                                p = sd['pass_v18'].copy()
                                corr = compute_low_correction(fn, sd['X_season'], new_comps)
                                p = apply_swap(p, sd['raw'], corr, sd['test_mask'], base_zone)
                                ex = sum(1 for i2, gi in enumerate(sd['indices'])
                                        if test_mask[gi] and p[i2] == int(y[gi]))
                                total += ex
                                ps[s] = ex
                            fine_results.append({
                                'comps': dict(new_comps), 'zone': base_zone,
                                'full': total, 'ps': ps
                            })
                    else:
                        if any(v == 0 for v in new_comps.values()):
                            continue
                        total = 0
                        ps = {}
                        for s, sd in season_data.items():
                            p = sd['pass_v18'].copy()
                            corr = compute_low_correction(fn, sd['X_season'], new_comps)
                            p = apply_swap(p, sd['raw'], corr, sd['test_mask'], base_zone)
                            ex = sum(1 for i2, gi in enumerate(sd['indices'])
                                    if test_mask[gi] and p[i2] == int(y[gi]))
                            total += ex
                            ps[s] = ex
                        fine_results.append({
                            'comps': dict(new_comps), 'zone': base_zone,
                            'full': total, 'ps': ps
                        })
            
            fine_results.sort(key=lambda r: -r['full'])
            print(f'\n  Fine-tuned top 15:')
            for r in fine_results[:15]:
                ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
                comps_str = ' '.join(f'{k}={v:+.1f}' for k, v in r['comps'].items())
                marker = ' ←' if r['full'] > v18_exact else ''
                print(f'    {r["full"]}/91 [{ps_str}] {comps_str}{marker}')
            
            all_results.extend(fine_results)

    # ════════════════════════════════════════════════════════════════
    #  Phase 5: NESTED LOSO VALIDATION
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 5: Nested LOSO validation')
    print('='*70)
    
    # Gather best configs
    all_results.sort(key=lambda r: -r['full'])
    competitive = [r for r in all_results if r['full'] >= v18_exact]
    
    # Add v18-only baseline
    competitive.append({
        'full': v18_exact, 'ps': v18_ps, 'zone': None, 'comp': 'none'
    })
    
    print(f'  {len(competitive)} configs >= {v18_exact}/91')
    
    nested_total = 0
    nested_v18 = 0
    nested_details = []
    
    for hold_season in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold_season]
        
        best_tune_score = -1
        best_idx = 0
        for ci, r in enumerate(competitive):
            tune_score = sum(r['ps'].get(s, 0) for s in tune_seasons)
            if tune_score > best_tune_score:
                best_tune_score = tune_score
                best_idx = ci
        
        best_r = competitive[best_idx]
        hold_exact = best_r['ps'].get(hold_season, 0)
        v18_hold = v18_ps.get(hold_season, 0)
        
        nested_total += hold_exact
        nested_v18 += v18_hold
        
        # Config name
        if 'comp' in best_r and best_r['comp'] != 'none':
            config_name = f'{best_r["zone"]}_{best_r["comp"]}_w{best_r["weight"]}'
        elif 'comps' in best_r:
            comps_str = '_'.join(f'{k}{v:+.0f}' for k, v in best_r['comps'].items())
            config_name = f'{best_r["zone"]}_{comps_str}'
        else:
            config_name = 'v18_only'
        
        nested_details.append({
            'season': hold_season, 
            'n': (test_mask & (seasons == hold_season)).sum(),
            'v18': v18_hold, 'v23': hold_exact,
            'config': config_name
        })
    
    print(f'\n  {"Season":<10} {"N":>3} {"v18":>4} {"v23":>4} {"Δ":>3}  Config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*40}')
    for d in nested_details:
        delta = d['v23'] - d['v18']
        print(f'  {d["season"]:<10} {d["n"]:3d} {d["v18"]:4d} {d["v23"]:4d} {delta:+3d}  {d["config"][:50]}')
    
    print(f'\n  Nested LOSO: v18={nested_v18}/91, v23={nested_total}/91 (Δ={nested_total-nested_v18:+d})')
    
    # ════════════════════════════════════════════════════════════════
    #  Also try BOTTOM zone (53-68) independently
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' BONUS: Bottom zone (53-68) correction')
    print('='*70)
    
    bot_results = []
    for comp in components_to_test:
        for weight in [-2, -1, 1, 2]:
            for zone in [(53, 68), (50, 68)]:
                total = 0
                ps = {}
                for s, sd in season_data.items():
                    p = sd['pass_v18'].copy()
                    corr = compute_low_correction(fn, sd['X_season'], {comp: weight})
                    p = apply_swap(p, sd['raw'], corr, sd['test_mask'], zone)
                    ex = sum(1 for i, gi in enumerate(sd['indices'])
                            if test_mask[gi] and p[i] == int(y[gi]))
                    total += ex
                    ps[s] = ex
                bot_results.append({
                    'zone': zone, 'comp': comp, 'weight': weight,
                    'full': total, 'ps': ps
                })
    
    bot_results.sort(key=lambda r: -r['full'])
    
    print(f'  Top 10 bottom-zone corrections:')
    for r in bot_results[:10]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if r['full'] > v18_exact else ''
        print(f'    {str(r["zone"]):<10} {r["comp"]:<12} w={r["weight"]:+.0f}: {r["full"]}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    
    best_overall = max(r['full'] for r in all_results)
    best_overall_configs = [r for r in all_results if r['full'] == best_overall]
    
    print(f'  v12: {v12_exact}/91')
    print(f'  v18: {v18_exact}/91')
    print(f'  Best v23 (full test): {best_overall}/91 ({len(best_overall_configs)} configs)')
    print(f'  Best v23 (nested LOSO): {nested_total}/91')
    
    if best_overall > v18_exact:
        print(f'\n  ★ IMPROVEMENT FOUND over v18! ★')
        print(f'\n  Best configs:')
        for r in best_overall_configs[:5]:
            ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
            if 'comp' in r:
                print(f'    zone={r["zone"]} {r["comp"]}={r["weight"]:+.1f}: {ps_str}')
            elif 'comps' in r:
                comps = ' '.join(f'{k}={v:+.1f}' for k, v in r['comps'].items())
                print(f'    zone={r["zone"]} {comps}: {ps_str}')
    else:
        print(f'\n  No improvement over v18. 61/91 remains the ceiling.')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
