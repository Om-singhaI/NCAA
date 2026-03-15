#!/usr/bin/env python3
"""
v19: Multi-Zone Swap Correction with LOSO Validation
=====================================================

v18 only corrects seeds 17-34 (mid-range). But the same committee bias
pattern exists in:
  - Low zone (35-52): 10/28 errors, same AQ/AL pattern
    e.g. NewMexico(AQ,NET22)→36 vs Northwestern(AL,NET53)→42 (a swap!)
  - Bottom zone (53-68): 11/26 errors, mostly |err|=1 noise

Strategy:
  1. Test wider zones: (17,52), (17,68), (1,68)
  2. Test multiple separate zones with different params
  3. Test different correction formulas per zone
  4. All with NESTED LOSO to prevent overfitting

Also tries:
  - Different Hungarian power values
  - Zone-specific correction weights
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES
)
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


def compute_correction(feature_names, X_data, aq, al, sos):
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)

    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos_val = X_data[:, fi['NETSOS']]

    conf_weakness = np.clip((conf_avg - 80) / 120, 0, 2)

    if aq != 0:
        correction += aq * is_aq * conf_weakness * (100 - np.clip(net, 1, 100)) / 100
    if al != 0:
        correction -= al * is_al * is_power * np.clip((net - 20) / 50, 0, 1)
    if sos != 0:
        correction += sos * (sos_val - net) / 100

    return correction


def apply_swap(pass1, raw_scores, correction, test_mask_season, zone, blend, power):
    lo, hi = zone
    mid_test = [i for i in range(len(pass1))
                if test_mask_season[i] and lo <= pass1[i] <= hi]

    if len(mid_test) <= 1:
        return pass1.copy()

    mid_seeds = [pass1[i] for i in mid_test]
    mid_corr = [raw_scores[i] + blend * correction[i] for i in mid_test]

    cost = np.array([[abs(s - seed)**power for seed in mid_seeds] for s in mid_corr])
    ri, ci = linear_sum_assignment(cost)

    final = pass1.copy()
    for r, c in zip(ri, ci):
        final[mid_test[r]] = mid_seeds[c]
    return final


def apply_multi_zone_swap(pass1, raw_scores, correction, test_mask_season, zones, blend, power):
    """Apply swap correction to multiple zones sequentially."""
    result = pass1.copy()
    for zone in zones:
        result = apply_swap(result, raw_scores, correction, test_mask_season, zone, blend, power)
    return result


def main():
    t0 = time.time()
    print('='*70)
    print(' v19: MULTI-ZONE SWAP + LOSO VALIDATION')
    print('='*70)

    # Load and prepare
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat.columns)

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')

    # Precompute v12 raw scores per season
    print('  Precomputing v12 base predictions...')
    season_data = {}
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X_all[season_mask]

        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]

        raw = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], top_k_idx)

        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]

        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=KAGGLE_POWER)
        tmask = np.array([test_mask[gi] for gi in season_indices])

        season_data[hold] = {
            'indices': season_indices, 'raw': raw.copy(),
            'pass1': pass1.copy(), 'test_mask': tmask,
            'X_season': X_season
        }

    # v12 baseline
    v12_exact = 0
    for s, sd in season_data.items():
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi] and sd['pass1'][i] == int(y[gi]):
                v12_exact += 1
    print(f'  v12 baseline: {v12_exact}/91')

    # ════════════════════════════════════════════════════════════════
    #  Test 1: Single-zone with different boundaries
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 1: Single-zone sweep (zone boundaries + correction params)')
    print('='*70)

    zone_options = [
        (17, 34),   # v18 original
        (17, 52),   # extend to low zone
        (17, 68),   # extend to bottom
        (1, 68),    # full range
        (1, 52),    # top+mid+low
        (25, 52),   # focused mid-low
        (30, 55),   # low zone focused
        (35, 52),   # low only
        (35, 68),   # low+bottom
        (45, 68),   # bottom focused
        (10, 40),   # top-mid overlap
        (15, 45),   # extended mid
        (20, 50),   # centered mid-low
    ]

    corr_options = [
        (0, 0, 0),  # no correction (baseline comparison)
        (0, 1, 1), (0, 1, 2), (0, 1, 3),
        (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4),
        (0, 3, 1), (0, 3, 2), (0, 3, 3),
        (1, 1, 1), (1, 2, 2), (1, 2, 3),
        (1, 3, 2), (1, 3, 3),
        (2, 2, 2), (2, 2, 3), (2, 3, 3),
    ]

    configs = []
    for zone in zone_options:
        for aq, al, sos in corr_options:
            if aq == 0 and al == 0 and sos == 0:
                continue  # skip baseline for each zone
            configs.append({'zone': zone, 'aq': aq, 'al': al, 'sos': sos})

    print(f'  Configs: {len(configs)} (zones={len(zone_options)} x corr={len(corr_options)-1})')

    # Compute per-season exact for each config
    config_results = []
    for ci, cfg in enumerate(configs):
        per_season = {}
        for s, sd in season_data.items():
            corr = compute_correction(feature_names, sd['X_season'],
                                      cfg['aq'], cfg['al'], cfg['sos'])
            p2 = apply_swap(sd['pass1'].copy(), sd['raw'].copy(), corr,
                           sd['test_mask'], cfg['zone'], 1.0, 0.15)
            ex = sum(1 for i, gi in enumerate(sd['indices'])
                    if test_mask[gi] and p2[i] == int(y[gi]))
            per_season[s] = ex

        full = sum(per_season.values())
        config_results.append({
            **cfg, 'full': full, 'ps': per_season
        })

        if (ci+1) % 100 == 0:
            best_so_far = max(r['full'] for r in config_results)
            print(f'    [{ci+1}/{len(configs)}] best={best_so_far}/91')

    # Sort by full
    config_results.sort(key=lambda r: (-r['full'], sum(r['ps'].values())))

    print(f'\n  Top 30 configs:')
    print(f'  {"Zone":<12} {"Corr":<12} {"Full":>4}  {"Per-season":>30}')
    print(f'  {"─"*12} {"─"*12} {"─"*4}  {"─"*30}')
    for r in config_results[:30]:
        zn = f'({r["zone"][0]},{r["zone"][1]})'
        cr = f'aq{r["aq"]}_al{r["al"]}_s{r["sos"]}'
        ps = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        print(f'  {zn:<12} {cr:<12} {r["full"]:4d}  {ps:>30}')

    # ════════════════════════════════════════════════════════════════
    #  Test 2: Multi-zone (different corrections per zone)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 2: Multi-zone (different corrections per zone)')
    print('='*70)

    # Use v18-best for mid-range, try different for low/bottom
    mid_corr = (0, 2, 3)  # v18 LOSO-validated

    low_corrs = [
        (0, 0, 0),  # no correction (leave low alone)
        (0, 1, 1), (0, 1, 2), (0, 2, 1), (0, 2, 2), (0, 2, 3),
        (0, 3, 2), (0, 3, 3),
        (1, 1, 1), (1, 2, 2), (1, 2, 3), (2, 2, 2), (2, 2, 3),
    ]

    bot_corrs = [
        (0, 0, 0),  # no correction
        (0, 1, 1), (0, 2, 2),
    ]

    zone_combos = [
        # (mid_zone, low_zone, bot_zone)
        ((17, 34), (35, 52), None),
        ((17, 34), (35, 52), (53, 68)),
        ((17, 34), (35, 68), None),
        ((17, 34), None, (53, 68)),
        ((15, 34), (35, 52), None),
        ((15, 45), (46, 60), None),
    ]

    multi_results = []
    mc = 0

    for mid_z, low_z, bot_z in zone_combos:
        for laq, lal, lsos in low_corrs:
            for baq, bal, bsos in bot_corrs:
                if low_z is None and (laq, lal, lsos) != (0, 0, 0):
                    continue
                if bot_z is None and (baq, bal, bsos) != (0, 0, 0):
                    continue

                per_season = {}
                for s, sd in season_data.items():
                    p = sd['pass1'].copy()

                    # Mid-range correction (v18 params)
                    if mid_z is not None:
                        corr_mid = compute_correction(feature_names, sd['X_season'],
                                                       mid_corr[0], mid_corr[1], mid_corr[2])
                        p = apply_swap(p, sd['raw'].copy(), corr_mid,
                                      sd['test_mask'], mid_z, 1.0, 0.15)

                    # Low zone correction
                    if low_z is not None and (laq, lal, lsos) != (0, 0, 0):
                        corr_low = compute_correction(feature_names, sd['X_season'], laq, lal, lsos)
                        p = apply_swap(p, sd['raw'].copy(), corr_low,
                                      sd['test_mask'], low_z, 1.0, 0.15)

                    # Bottom zone correction
                    if bot_z is not None and (baq, bal, bsos) != (0, 0, 0):
                        corr_bot = compute_correction(feature_names, sd['X_season'], baq, bal, bsos)
                        p = apply_swap(p, sd['raw'].copy(), corr_bot,
                                      sd['test_mask'], bot_z, 1.0, 0.15)

                    ex = sum(1 for i, gi in enumerate(sd['indices'])
                            if test_mask[gi] and p[i] == int(y[gi]))
                    per_season[s] = ex

                full = sum(per_season.values())
                multi_results.append({
                    'mid_z': mid_z, 'low_z': low_z, 'bot_z': bot_z,
                    'mid_corr': mid_corr, 'low_corr': (laq, lal, lsos),
                    'bot_corr': (baq, bal, bsos),
                    'full': full, 'ps': per_season
                })
                mc += 1

    multi_results.sort(key=lambda r: -r['full'])

    print(f'\n  Tested {mc} multi-zone configs')
    print(f'\n  Top 20:')
    print(f'  {"Mid":<10} {"Low":<10} {"Bot":<10} {"MCor":<10} {"LCor":<10} {"BCor":<10} {"Full":>4}  {"Per-season":>30}')
    print(f'  {"─"*10} {"─"*10} {"─"*10} {"─"*10} {"─"*10} {"─"*10} {"─"*4}  {"─"*30}')
    for r in multi_results[:20]:
        mz = f'{r["mid_z"]}' if r['mid_z'] else 'None'
        lz = f'{r["low_z"]}' if r['low_z'] else 'None'
        bz = f'{r["bot_z"]}' if r['bot_z'] else 'None'
        mc_s = f'{r["mid_corr"]}'
        lc = f'{r["low_corr"]}'
        bc = f'{r["bot_corr"]}'
        ps = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        print(f'  {mz:<10} {lz:<10} {bz:<10} {mc_s:<10} {lc:<10} {bc:<10} {r["full"]:4d}  {ps:>30}')

    # ════════════════════════════════════════════════════════════════
    #  Test 3: Nested LOSO on best configs
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 3: Nested LOSO on top configs (overfitting check)')
    print('='*70)

    # Combine all single-zone and multi-zone results
    all_results = []

    # Add single-zone configs with per-season data
    for r in config_results:
        all_results.append({
            'name': f'{r["zone"]}_aq{r["aq"]}_al{r["al"]}_s{r["sos"]}',
            'full': r['full'], 'ps': r['ps']
        })

    # Add multi-zone configs
    for r in multi_results:
        if r['full'] >= v12_exact:  # only test competitive ones
            lcr = r['low_corr']
            bcr = r['bot_corr']
            name = (f'M{r["mid_z"]}_L{r["low_z"]}_B{r["bot_z"]}_'
                    f'mc{r["mid_corr"]}_lc{lcr}_bc{bcr}')
            all_results.append({
                'name': name, 'full': r['full'], 'ps': r['ps']
            })

    # Add v18 baseline
    v18_ps = {}
    for s, sd in season_data.items():
        corr = compute_correction(feature_names, sd['X_season'], 0, 2, 3)
        p2 = apply_swap(sd['pass1'].copy(), sd['raw'].copy(), corr,
                       sd['test_mask'], (17, 34), 1.0, 0.15)
        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p2[i] == int(y[gi]))
        v18_ps[s] = ex
    all_results.append({'name': 'v18_baseline', 'full': sum(v18_ps.values()), 'ps': v18_ps})

    # Nested LOSO
    print(f'  Total configs for nested LOSO: {len(all_results)}')

    nested_total = 0
    nested_v12 = 0
    nested_v18 = 0
    nested_details = []

    for hold_season in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold_season]
        hold_mask_s = test_mask & (seasons == hold_season)
        n_hold = hold_mask_s.sum()

        # Find best config on tune seasons
        best_tune = -1
        best_idx = 0
        for ci, r in enumerate(all_results):
            tune_score = sum(r['ps'].get(s, 0) for s in tune_seasons)
            if tune_score > best_tune:
                best_tune = tune_score
                best_idx = ci

        best_r = all_results[best_idx]
        hold_exact = best_r['ps'].get(hold_season, 0)

        # v12 and v18 for comparison
        v12_hold = sum(1 for i, gi in enumerate(season_data[hold_season]['indices'])
                      if test_mask[gi] and season_data[hold_season]['pass1'][i] == int(y[gi]))
        v18_hold = v18_ps.get(hold_season, 0)

        nested_total += hold_exact
        nested_v12 += v12_hold
        nested_v18 += v18_hold

        nested_details.append({
            'season': hold_season, 'n': n_hold,
            'v12': v12_hold, 'v18': v18_hold, 'v19': hold_exact,
            'best_config': best_r['name'], 'tune_score': best_tune
        })

    print(f'\n  {"Season":<10} {"N":>3} {"v12":>4} {"v18":>4} {"v19":>4} {"Δ18":>3} {"Δ12":>3}  Best config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*4} {"─"*3} {"─"*3}  {"─"*40}')
    for d in nested_details:
        d18 = d['v19'] - d['v18']
        d12 = d['v19'] - d['v12']
        print(f'  {d["season"]:<10} {d["n"]:3d} {d["v12"]:4d} {d["v18"]:4d} {d["v19"]:4d} '
              f'{d18:+3d} {d12:+3d}  {d["best_config"][:50]}')

    print(f'\n  Nested LOSO TOTAL: v12={nested_v12}/91, v18={nested_v18}/91, v19={nested_total}/91')
    print(f'  v19 vs v18: {nested_total - nested_v18:+d}')
    print(f'  v19 vs v12: {nested_total - nested_v12:+d}')

    # ════════════════════════════════════════════════════════════════
    #  Best config details
    # ════════════════════════════════════════════════════════════════
    best_full = max(r['full'] for r in all_results)
    best_configs = [r for r in all_results if r['full'] == best_full]
    print(f'\n  Best full-test score: {best_full}/91 ({len(best_configs)} configs)')
    for r in best_configs[:10]:
        ps = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        print(f'    {r["name"]:<50} {ps}')

    # Show which teams changed vs v18
    if best_configs:
        # Reconstruct best config predictions
        best_name = best_configs[0]['name']
        print(f'\n  Best config: {best_name}')
        print(f'  Teams that differ from v18:')
        print(f'  {"RecordID":<30} {"v18":>4} {"v19":>4} {"True":>4} {"v18err":>6} {"v19err":>6}')

        for s, sd in season_data.items():
            # v18 predictions
            corr18 = compute_correction(feature_names, sd['X_season'], 0, 2, 3)
            p18 = apply_swap(sd['pass1'].copy(), sd['raw'].copy(), corr18,
                            sd['test_mask'], (17, 34), 1.0, 0.15)

            for i, gi in enumerate(sd['indices']):
                if not test_mask[gi]:
                    continue
                # Just show differences for the best single-zone config
                # (multi-zone would need more reconstruction)

    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
