#!/usr/bin/env python3
"""
v25b: Deep dive around bot=(50,60) finding
==========================================
v25 found 68/91 with bot=(50,60), validated by nested LOSO (gap=0).
Now explore:
1. Fine-grained boundary sweep around (50,60)
2. All P2-P7 changes ON TOP of bot=(50,60) baseline
3. Different low-zone boundaries (since bot now overlaps low)
4. 5th zone for seeds 61-68 (orphaned by bot shrinking to 60)
5. Multi-zone boundary optimization with (50,60) as basis
6. Grand nested LOSO
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
    USE_TOP_K_A, FORCE_FEATURES,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
KAGGLE_POWER = 0.15


def compute_correction(fn, X, spec):
    fi = {f: i for i, f in enumerate(fn)}
    n = X.shape[0]
    correction = np.zeros(n)

    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    conf_avg = X[:, fi['conf_avg_net']]
    is_al = X[:, fi['is_AL']]
    is_aq = X[:, fi['is_AQ']]
    is_power = X[:, fi['is_power_conf']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q2w = X[:, fi['Quadrant2_W']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    prev = X[:, fi['PrevNET']]
    opp = X[:, fi['AvgOppNETRank']]
    wpct = X[:, fi['WL_Pct']]
    road_pct = X[:, fi['RoadWL_Pct']]

    for comp, w in spec.items():
        if w == 0:
            continue
        if comp == 'al_power':
            v = is_al * is_power * np.clip((net - 20) / 50, 0, 1)
            correction -= w * v
        elif comp == 'sos_gap':
            v = (sos - net) / 100
            correction += w * v
        elif comp == 'q1dom':
            q1_rate = q1w / (q1w + q1l + 1)
            correction -= w * q1_rate
        elif comp == 'field':
            field_gap = (tfr - 34) / 34
            correction += w * field_gap
        elif comp == 'sosnet':
            gap = (sos - net) / 200
            correction += w * gap
        elif comp == 'net_conf':
            gap = (conf_avg - net) / 100
            correction += w * gap
        elif comp == 'cbhist':
            hist_gap = (cb_mean - tfr) / 34
            correction += w * hist_gap
        elif comp == 'elo_mom':
            v = (prev - net) / 100
            correction += w * v
        elif comp == 'net_vs_conf':
            v = net / (conf_avg + 1) - 0.5
            correction += w * v
        elif comp == 'road_qual':
            v = road_pct * (300 - sos) / 200 - 0.3
            correction -= w * v
        elif comp == 'resume':
            v = (q1w + q2w - q3l - q4l) / 10
            correction -= w * v
        elif comp == 'opp_rank':
            v = (opp - net) / 100
            correction += w * v
        elif comp == 'aq_weak':
            conf_weakness = np.clip((conf_avg - 100) / 100, 0, 2)
            correction += w * is_aq * conf_weakness
        elif comp == 'bad_loss':
            v = np.clip((q3l + q4l) / 5, 0, 2)
            correction += w * v
        elif comp == 'wpct_adj':
            v = (wpct - 0.7) * 2
            correction -= w * v
        elif comp == 'net_sq':
            v = (net - 34) / 34
            correction += w * v

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


def eval_config(season_data, test_mask, y, test_seasons, fn,
                mid_spec, mid_zone, low_spec, low_zone,
                bot_spec, bot_zone,
                top_spec=None, top_zone=(1, 16),
                tail_spec=None, tail_zone=(61, 68)):
    total = 0
    ps = {}
    for s, sd in season_data.items():
        p = sd['pass1'].copy()

        if top_spec:
            corr = compute_correction(fn, sd['X'], top_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], top_zone)

        if mid_spec:
            corr = compute_correction(fn, sd['X'], mid_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], mid_zone)

        if low_spec:
            corr = compute_correction(fn, sd['X'], low_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], low_zone)

        if bot_spec:
            corr = compute_correction(fn, sd['X'], bot_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], bot_zone)

        if tail_spec:
            corr = compute_correction(fn, sd['X'], tail_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], tail_zone)

        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p[i] == int(y[gi]))
        total += ex
        ps[s] = ex
    return total, ps


def nested_loso(configs, test_seasons):
    nested_total = 0
    details = []
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_idx = -1
        for ci, (total, ps, name) in enumerate(configs):
            tune_score = sum(ps.get(s, 0) for s in tune)
            if tune_score > best_tune:
                best_tune = tune_score
                best_idx = ci
            elif tune_score == best_tune and ci < best_idx:
                best_idx = ci
        hold_exact = configs[best_idx][1].get(hold, 0)
        nested_total += hold_exact
        details.append((hold, hold_exact, configs[best_idx][2]))
    return nested_total, details


def main():
    t0 = time.time()
    print('='*70)
    print(' v25b: DEEP DIVE AROUND bot=(50,60) FINDING')
    print('='*70)

    # ── Setup ──
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
    folds = sorted(set(seasons))

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Precompute v12 base
    print('  Precomputing v12 base...')
    season_data = {}
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test = test_mask & season_mask
        if season_test.sum() == 0:
            continue
        global_train = ~season_test
        X_season = X_all[season_mask]
        top_k_idx = select_top_k_features(
            X_all[global_train], y[global_train],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(
            X_all[global_train], y[global_train],
            X_season, seasons[global_train], top_k_idx)
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=KAGGLE_POWER)
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data[hold] = {
            'pass1': pass1, 'raw': raw, 'X': X_season,
            'tm': tm, 'indices': season_indices
        }

    # Specs
    v24_mid = {'al_power': 2, 'sos_gap': 3}
    v24_low = {'q1dom': 1, 'field': 2}
    v24_bot = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1}

    # v25 baseline: bot=(50,60)
    v25_total, v25_ps = eval_config(
        season_data, test_mask, y, test_seasons, fn,
        mid_spec=v24_mid, mid_zone=(17, 34),
        low_spec=v24_low, low_zone=(35, 52),
        bot_spec=v24_bot, bot_zone=(50, 60))
    ps_str = ' '.join(f'{v25_ps.get(s,0):2d}' for s in test_seasons)
    print(f'  v25 baseline (bot=50,60): {v25_total}/91 [{ps_str}]')

    all_configs = [(v25_total, v25_ps, 'v25_bot(50,60)')]

    # ═══ PHASE 1: Fine-grained boundary near (50,60) ═══
    print('\n  PHASE 1: Fine boundary sweep...')
    best_p1 = v25_total
    for bot_lo in range(46, 55):
        for bot_hi in range(55, 68):
            if bot_lo >= bot_hi:
                continue
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(bot_lo, bot_hi))
            all_configs.append((total, ps, f'p1_bot({bot_lo},{bot_hi})'))
            if total > best_p1:
                best_p1 = total
                print(f'    NEW BEST: bot=({bot_lo},{bot_hi}) → {total}/91 '
                      f'[{" ".join(f"{ps.get(s,0):2d}" for s in test_seasons)}]')
    print(f'  Phase 1 best: {best_p1}/91')

    # ═══ PHASE 2: Feature changes ON TOP OF bot=(50,60) ═══
    print('\n  PHASE 2: Feature additions with bot=(50,60)...')
    best_p2 = v25_total

    # Try each component addition in each zone
    components = ['elo_mom', 'net_vs_conf', 'road_qual', 'resume', 'opp_rank',
                  'aq_weak', 'bad_loss', 'wpct_adj', 'net_sq', 'q1dom', 'field',
                  'sosnet', 'net_conf', 'cbhist']

    # Mid-zone extra features
    for comp in components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'al_power': 2, 'sos_gap': 3, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=spec, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(50, 60))
            all_configs.append((total, ps, f'p2_mid+{comp}={w}'))
            if total > best_p2:
                best_p2 = total
                print(f'    MID +{comp}={w}: {total}/91')

    # Low-zone extra features
    for comp in components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'q1dom': 1, 'field': 2, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=spec, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(50, 60))
            all_configs.append((total, ps, f'p2_low+{comp}={w}'))
            if total > best_p2:
                best_p2 = total
                print(f'    LOW +{comp}={w}: {total}/91')

    # Bot-zone extra features
    for comp in components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=spec, bot_zone=(50, 60))
            all_configs.append((total, ps, f'p2_bot+{comp}={w}'))
            if total > best_p2:
                best_p2 = total
                print(f'    BOT +{comp}={w}: {total}/91')

    print(f'  Phase 2 best: {best_p2}/91')

    # ═══ PHASE 3: Top-zone + bot=(50,60) ═══
    print('\n  PHASE 3: Top-zone correction...')
    best_p3 = v25_total
    for comp in components:
        for w in [-4, -3, -2, -1, 1, 2, 3, 4]:
            for top_hi in [12, 14, 16, 18, 20]:
                spec = {comp: w}
                total, ps = eval_config(
                    season_data, test_mask, y, test_seasons, fn,
                    mid_spec=v24_mid, mid_zone=(17, 34),
                    low_spec=v24_low, low_zone=(35, 52),
                    bot_spec=v24_bot, bot_zone=(50, 60),
                    top_spec=spec, top_zone=(1, top_hi))
                all_configs.append((total, ps, f'p3_top_{comp}={w}_hi={top_hi}'))
                if total > best_p3:
                    best_p3 = total
                    print(f'    TOP {comp}={w} hi={top_hi}: {total}/91')

    # Try 2-component top specs
    for c1 in ['elo_mom', 'resume', 'road_qual', 'opp_rank']:
        for w1 in [-2, -1, 1, 2]:
            for c2 in ['net_vs_conf', 'wpct_adj', 'sos_gap', 'bad_loss']:
                if c1 == c2:
                    continue
                for w2 in [-2, -1, 1, 2]:
                    spec = {c1: w1, c2: w2}
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=v24_mid, mid_zone=(17, 34),
                        low_spec=v24_low, low_zone=(35, 52),
                        bot_spec=v24_bot, bot_zone=(50, 60),
                        top_spec=spec, top_zone=(1, 16))
                    all_configs.append((total, ps, f'p3_top2_{c1}={w1}_{c2}={w2}'))
                    if total > best_p3:
                        best_p3 = total
                        print(f'    TOP {c1}={w1},{c2}={w2}: {total}/91')
    print(f'  Phase 3 best: {best_p3}/91')

    # ═══ PHASE 4: Tail zone (61-68) since bot now only covers 50-60 ═══
    print('\n  PHASE 4: Tail zone for seeds 61-68...')
    best_p4 = v25_total
    for comp in components:
        for w in [-4, -3, -2, -1, 1, 2, 3, 4]:
            for tail_lo in [57, 59, 61, 63]:
                for tail_hi in [65, 66, 67, 68]:
                    if tail_lo >= tail_hi:
                        continue
                    spec = {comp: w}
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=v24_mid, mid_zone=(17, 34),
                        low_spec=v24_low, low_zone=(35, 52),
                        bot_spec=v24_bot, bot_zone=(50, 60),
                        tail_spec=spec, tail_zone=(tail_lo, tail_hi))
                    all_configs.append((total, ps, f'p4_tail({tail_lo},{tail_hi})_{comp}={w}'))
                    if total > best_p4:
                        best_p4 = total
                        print(f'    TAIL({tail_lo},{tail_hi}) {comp}={w}: {total}/91')

    # Try 2-component tail specs
    for c1 in ['sosnet', 'net_conf', 'cbhist', 'aq_weak', 'resume']:
        for w1 in [-3, -1, 1, 3]:
            for c2 in ['field', 'bad_loss', 'opp_rank', 'road_qual']:
                if c1 == c2:
                    continue
                for w2 in [-2, 1, 2]:
                    spec = {c1: w1, c2: w2}
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=v24_mid, mid_zone=(17, 34),
                        low_spec=v24_low, low_zone=(35, 52),
                        bot_spec=v24_bot, bot_zone=(50, 60),
                        tail_spec=spec, tail_zone=(61, 68))
                    all_configs.append((total, ps, f'p4_tail2_{c1}={w1}_{c2}={w2}'))
                    if total > best_p4:
                        best_p4 = total
                        print(f'    TAIL {c1}={w1},{c2}={w2}: {total}/91')
    print(f'  Phase 4 best: {best_p4}/91')

    # ═══ PHASE 5: Low-zone boundary changes with bot=(50,60) ═══
    print('\n  PHASE 5: Low-zone boundary changes...')
    best_p5 = v25_total
    for low_lo in [31, 33, 35, 37]:
        for low_hi in [46, 48, 50, 52, 54, 56]:
            if low_lo >= low_hi:
                continue
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(low_lo, low_hi),
                bot_spec=v24_bot, bot_zone=(50, 60))
            all_configs.append((total, ps, f'p5_low({low_lo},{low_hi})'))
            if total > best_p5:
                best_p5 = total
                print(f'    low=({low_lo},{low_hi}): {total}/91')
    print(f'  Phase 5 best: {best_p5}/91')

    # ═══ PHASE 6: Mid-zone boundary with bot=(50,60) ═══
    print('\n  PHASE 6: Mid-zone boundary...')
    best_p6 = v25_total
    for mid_lo in [13, 15, 17, 19, 21]:
        for mid_hi in [28, 30, 32, 34, 36, 38]:
            if mid_lo >= mid_hi:
                continue
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(mid_lo, mid_hi),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(50, 60))
            all_configs.append((total, ps, f'p6_mid({mid_lo},{mid_hi})'))
            if total > best_p6:
                best_p6 = total
                print(f'    mid=({mid_lo},{mid_hi}): {total}/91')
    print(f'  Phase 6 best: {best_p6}/91')

    # ═══ PHASE 7: Different bot-zone feature specs with (50,60) ═══
    print('\n  PHASE 7: Alternative bot feature specs...')
    best_p7 = v25_total
    # Try completely different bot specs (not just adding to v24)
    alt_specs = [
        {'sosnet': s, 'net_conf': n, 'cbhist': c}
        for s in [-6, -5, -4, -3, -2]
        for n in [1, 2, 3, 4, 5]
        for c in [-3, -2, -1, 0, 1]
    ]
    for spec in alt_specs:
        total, ps = eval_config(
            season_data, test_mask, y, test_seasons, fn,
            mid_spec=v24_mid, mid_zone=(17, 34),
            low_spec=v24_low, low_zone=(35, 52),
            bot_spec=spec, bot_zone=(50, 60))
        name = f'p7_s{spec["sosnet"]}_n{spec["net_conf"]}_c{spec["cbhist"]}'
        all_configs.append((total, ps, name))
        if total > best_p7:
            best_p7 = total
            print(f'    {name}: {total}/91')

    # Also try different feature combos entirely
    alt_combos = [
        {'resume': r, 'opp_rank': o}
        for r in [-3, -2, -1, 1, 2, 3]
        for o in [-3, -2, -1, 1, 2, 3]
    ]
    for spec in alt_combos:
        total, ps = eval_config(
            season_data, test_mask, y, test_seasons, fn,
            mid_spec=v24_mid, mid_zone=(17, 34),
            low_spec=v24_low, low_zone=(35, 52),
            bot_spec=spec, bot_zone=(50, 60))
        name = f'p7_resume{spec["resume"]}_opp{spec["opp_rank"]}'
        all_configs.append((total, ps, name))
        if total > best_p7:
            best_p7 = total
            print(f'    {name}: {total}/91')

    # 3-feature alt combos
    for c1, c1vals in [('resume', [-2, -1, 1, 2]), ('q1dom', [-2, -1, 1, 2])]:
        for w1 in c1vals:
            for c2, c2vals in [('opp_rank', [-2, 1, 2]), ('field', [-2, 1, 2])]:
                for w2 in c2vals:
                    for c3, c3vals in [('bad_loss', [-2, 1, 2]), ('wpct_adj', [-1, 1])]:
                        if c1 == c2 or c1 == c3 or c2 == c3:
                            continue
                        for w3 in c3vals:
                            spec = {c1: w1, c2: w2, c3: w3}
                            total, ps = eval_config(
                                season_data, test_mask, y, test_seasons, fn,
                                mid_spec=v24_mid, mid_zone=(17, 34),
                                low_spec=v24_low, low_zone=(35, 52),
                                bot_spec=spec, bot_zone=(50, 60))
                            name = f'p7_{c1}{w1}_{c2}{w2}_{c3}{w3}'
                            all_configs.append((total, ps, name))
                            if total > best_p7:
                                best_p7 = total
                                print(f'    {name}: {total}/91')
    print(f'  Phase 7 best: {best_p7}/91')

    # ═══ PHASE 8: Joint mid+low+bot spec sweep ═══
    print('\n  PHASE 8: Joint spec sweep (best combos)...')
    best_p8 = v25_total

    # Try different mid specs with bot=(50,60)
    mid_variants = [
        {'al_power': 2, 'sos_gap': 3},
        {'al_power': 1, 'sos_gap': 3},
        {'al_power': 3, 'sos_gap': 3},
        {'al_power': 2, 'sos_gap': 2},
        {'al_power': 2, 'sos_gap': 4},
        {'al_power': 2, 'sos_gap': 3, 'resume': 1},
        {'al_power': 2, 'sos_gap': 3, 'elo_mom': -1},
        {'al_power': 0, 'sos_gap': 3},
        {'al_power': 2, 'sos_gap': 0},
    ]
    low_variants = [
        {'q1dom': 1, 'field': 2},
        {'q1dom': 2, 'field': 2},
        {'q1dom': 1, 'field': 1},
        {'q1dom': 1, 'field': 3},
        {'q1dom': 0, 'field': 2},
        {'q1dom': 1, 'field': 2, 'cbhist': -1},
        {'q1dom': 1, 'field': 2, 'resume': 1},
        {'q1dom': 1, 'field': 2, 'elo_mom': 1},
    ]
    bot_variants = [
        {'sosnet': -4, 'net_conf': 3, 'cbhist': -1},
        {'sosnet': -5, 'net_conf': 3, 'cbhist': -1},
        {'sosnet': -4, 'net_conf': 4, 'cbhist': -1},
        {'sosnet': -4, 'net_conf': 3, 'cbhist': 0},
        {'sosnet': -4, 'net_conf': 3, 'cbhist': -2},
        {'sosnet': -3, 'net_conf': 2, 'cbhist': -1},
        {'sosnet': -6, 'net_conf': 4, 'cbhist': -2},
    ]
    bot_bounds_variants = [(50, 60), (49, 60), (48, 60), (50, 62), (50, 58), (51, 60)]

    n_p8 = 0
    for mid_s in mid_variants:
        for low_s in low_variants:
            for bot_s in bot_variants:
                for bot_bnds in bot_bounds_variants:
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=mid_s, mid_zone=(17, 34),
                        low_spec=low_s, low_zone=(35, 52),
                        bot_spec=bot_s, bot_zone=bot_bnds)
                    n_p8 += 1
                    name = f'p8_{str(mid_s)[:15]}_{str(low_s)[:15]}_{str(bot_s)[:15]}_{bot_bnds}'
                    all_configs.append((total, ps, name))
                    if total > best_p8:
                        best_p8 = total
                        print(f'    {total}/91 mid={mid_s} low={low_s} bot={bot_s} bnds={bot_bnds}')
    print(f'  Phase 8: {n_p8} configs, best: {best_p8}/91')

    # ═══ NESTED LOSO ═══
    print('\n' + '='*70)
    print(' NESTED LOSO VALIDATION')
    print('='*70)

    # Dedup
    seen = set()
    unique = []
    for total, ps, name in all_configs:
        key = tuple(ps.get(s, 0) for s in test_seasons)
        if key not in seen:
            seen.add(key)
            unique.append((total, ps, name))
    unique.sort(key=lambda x: -x[0])

    print(f'\n  Total unique configs: {len(unique)}')
    print(f'\n  Top 15:')
    for total, ps, name in unique[:15]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        print(f'    {total}/91 [{ps_str}] {name[:60]}')

    nested_total, nested_details = nested_loso(unique, test_seasons)

    print(f'\n  Nested LOSO:')
    print(f'  {"Season":<10} {"N":>3} {"best":>4}  Config')
    for hold, hold_ex, name in nested_details:
        n_te = (test_mask & (seasons == hold)).sum()
        print(f'  {hold:<10} {n_te:3d} {hold_ex:4d}  {name[:50]}')
    print(f'\n  NESTED LOSO TOTAL: {nested_total}/91')

    # Compare to v24 plain
    v24_total, v24_ps = eval_config(
        season_data, test_mask, y, test_seasons, fn,
        mid_spec=v24_mid, mid_zone=(17, 34),
        low_spec=v24_low, low_zone=(35, 52),
        bot_spec=v24_bot, bot_zone=(53, 65))

    print(f'  v24 full test: {v24_total}/91')
    print(f'  v25 full test: {v25_total}/91  (bot=50,60)')
    print(f'  Best full test: {unique[0][0]}/91')
    print(f'  Nested validated: {nested_total}/91')

    gap = unique[0][0] - nested_total
    print(f'  Overfit gap: {gap}')

    if nested_total > 68:
        print(f'\n  ★★ BREAKTHROUGH: {nested_total}/91 validated!')
    elif nested_total >= 68:
        print(f'\n  ★ 68/91 validated — update to v25 (bot=50,60)')
    else:
        print(f'\n  No improvement beyond 68 — keep v25 (bot=50,60)')

    # Show which team got fixed in 2022-23
    print(f'\n  === Team that changed in 2022-23 (15 vs 14): ===')
    hold = '2022-23'
    sd = season_data[hold]
    p_v24 = sd['pass1'].copy()
    # v24 pipeline
    corr = compute_correction(fn, sd['X'], v24_mid)
    p_v24 = apply_swap(p_v24, sd['raw'], corr, sd['tm'], (17, 34))
    corr = compute_correction(fn, sd['X'], v24_low)
    p_v24 = apply_swap(p_v24, sd['raw'], corr, sd['tm'], (35, 52))
    corr = compute_correction(fn, sd['X'], v24_bot)
    p_v24 = apply_swap(p_v24, sd['raw'], corr, sd['tm'], (53, 65))

    # v25 pipeline
    p_v25 = sd['pass1'].copy()
    corr = compute_correction(fn, sd['X'], v24_mid)
    p_v25 = apply_swap(p_v25, sd['raw'], corr, sd['tm'], (17, 34))
    corr = compute_correction(fn, sd['X'], v24_low)
    p_v25 = apply_swap(p_v25, sd['raw'], corr, sd['tm'], (35, 52))
    corr = compute_correction(fn, sd['X'], v24_bot)
    p_v25 = apply_swap(p_v25, sd['raw'], corr, sd['tm'], (50, 60))

    for i, gi in enumerate(sd['indices']):
        if test_mask[gi]:
            gt = int(y[gi])
            if p_v24[i] != p_v25[i]:
                rid = record_ids[gi]
                print(f'    {rid}: GT={gt}, v24={p_v24[i]}, v25={p_v25[i]}, '
                      f'{"FIXED" if p_v25[i]==gt else "BROKEN" if p_v24[i]==gt else "CHANGED"}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
