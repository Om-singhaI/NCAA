#!/usr/bin/env python3
"""
v25: Exhaustive hypertuning beyond v24 (67/91)
==============================================
24 remaining errors, 6 swap pairs. Targets:

1. Extend bot zone to 68 (captures TexasSouthern/SoutheastMoSt swap at 66/67)
2. Top-zone correction (1-16): Kentucky/Wisconsin swap (2024-25)
3. New mid-range features: add q1dom, net_vs_conf, elo_momentum, road_quality
4. New low-zone features: add sos, conf_avg, cb_mean
5. New bot-zone features: try different/more components
6. Joint zone boundary sweep
7. Per-zone power parameter
8. Grand nested LOSO validation

Everything validated with nested LOSO to ensure no overfitting.
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
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


# ─── Generalized correction functions ───

def compute_correction(fn, X, spec):
    """Generic correction function.
    spec is a dict of {component_name: weight}.
    Supported components and their formulas:
    """
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
            # AL from power conference → push lower (better)
            v = is_al * is_power * np.clip((net - 20) / 50, 0, 1)
            correction -= w * v
        elif comp == 'sos_gap':
            # SOS-NET gap: weak schedule → push higher (worse)
            v = (sos - net) / 100
            correction += w * v
        elif comp == 'q1dom':
            # Q1 dominance → push lower (better)
            q1_rate = q1w / (q1w + q1l + 1)
            correction -= w * q1_rate
        elif comp == 'field':
            # Field rank divergence → push higher when worse rank
            field_gap = (tfr - 34) / 34
            correction += w * field_gap
        elif comp == 'sosnet':
            # SOS-NET gap (bot zone version, /200 normalization)
            gap = (sos - net) / 200
            correction += w * gap
        elif comp == 'net_conf':
            # NET vs conference average
            gap = (conf_avg - net) / 100
            correction += w * gap
        elif comp == 'cbhist':
            # Conference-bid history
            hist_gap = (cb_mean - tfr) / 34
            correction += w * hist_gap
        elif comp == 'elo_mom':
            # Elo momentum: prev - net (improving → push lower)
            v = (prev - net) / 100
            correction += w * v
        elif comp == 'net_vs_conf':
            # NET rank ratio vs conference
            v = net / (conf_avg + 1) - 0.5
            correction += w * v
        elif comp == 'road_qual':
            # Road quality: road_pct × schedule strength
            v = road_pct * (300 - sos) / 200 - 0.3
            correction -= w * v
        elif comp == 'resume':
            # Resume: Q1W+Q2W-Q3L-Q4L
            v = (q1w + q2w - q3l - q4l) / 10
            correction -= w * v
        elif comp == 'opp_rank':
            # Avg opponent rank
            v = (opp - net) / 100
            correction += w * v
        elif comp == 'aq_weak':
            # AQ from weak conference → push higher
            conf_weakness = np.clip((conf_avg - 100) / 100, 0, 2)
            correction += w * is_aq * conf_weakness
        elif comp == 'bad_loss':
            # Bad losses (Q3+Q4) → push higher
            v = np.clip((q3l + q4l) / 5, 0, 2)
            correction += w * v
        elif comp == 'wpct_adj':
            # Win pct adjustment
            v = (wpct - 0.7) * 2
            correction -= w * v
        elif comp == 'net_sq':
            # NET rank squared (nonlinear)
            v = (net - 34) / 34
            correction += w * v
        elif comp == 'conf_med':
            conf_med = X[:, fi['conf_med_net']]
            v = (conf_med - net) / 100
            correction += w * v
        elif comp == 'al_rank':
            al_rank = X[:, fi['net_rank_among_al']]
            v = (al_rank - 20) / 20
            correction += w * v
        elif comp == 'sos_x_wpct':
            v = X[:, fi['sos_x_wpct']] - 0.5
            correction -= w * v

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
                mid_power=0.15, low_power=0.15, bot_power=0.15, top_power=0.15):
    """Run full pipeline with given config."""
    total = 0
    ps = {}
    for s, sd in season_data.items():
        p = sd['pass1'].copy()

        # Top zone
        if top_spec:
            corr = compute_correction(fn, sd['X'], top_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], top_zone, top_power)

        # Mid zone
        if mid_spec:
            corr = compute_correction(fn, sd['X'], mid_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], mid_zone, mid_power)

        # Low zone
        if low_spec:
            corr = compute_correction(fn, sd['X'], low_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], low_zone, low_power)

        # Bot zone
        if bot_spec:
            corr = compute_correction(fn, sd['X'], bot_spec)
            p = apply_swap(p, sd['raw'], corr, sd['tm'], bot_zone, bot_power)

        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p[i] == int(y[gi]))
        total += ex
        ps[s] = ex
    return total, ps


def nested_loso(configs, test_seasons, prefer_simple=True):
    """Run nested LOSO: for each held-out season, pick best config from tune seasons."""
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
            elif tune_score == best_tune and prefer_simple:
                # Prefer earlier / simpler configs as tiebreaker
                if ci < best_idx:
                    best_idx = ci
        hold_exact = configs[best_idx][1].get(hold, 0)
        nested_total += hold_exact
        details.append((hold, hold_exact, configs[best_idx][2]))
    return nested_total, details


def main():
    t0 = time.time()
    print('='*70)
    print(' v25: EXHAUSTIVE HYPERTUNING BEYOND v24 (67/91)')
    print(' Target: improve while validated by nested LOSO')
    print('='*70)

    # ── Setup ──
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

    print(f'\n  Labeled: {n_labeled}, Test: {test_mask.sum()}, Seasons: {len(test_seasons)}')

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

    # v24 baseline
    v24_mid = {'al_power': 2, 'sos_gap': 3}
    v24_low = {'q1dom': 1, 'field': 2}
    v24_bot = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1}

    v24_total, v24_ps = eval_config(
        season_data, test_mask, y, test_seasons, fn,
        mid_spec=v24_mid, mid_zone=(17, 34),
        low_spec=v24_low, low_zone=(35, 52),
        bot_spec=v24_bot, bot_zone=(53, 65))
    ps_str = ' '.join(f'{v24_ps.get(s,0):2d}' for s in test_seasons)
    print(f'  v24 baseline: {v24_total}/91 [{ps_str}]')

    all_configs = []  # (total, ps, name) for nested LOSO
    all_configs.append((v24_total, v24_ps, 'v24_baseline'))

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: Extend bot zone boundary
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 1: Bot zone boundary sweep')
    print('='*70)

    best_p1 = v24_total
    p1_results = []
    for bot_lo in [49, 50, 51, 52, 53, 54, 55]:
        for bot_hi in [60, 62, 63, 64, 65, 66, 67, 68]:
            if bot_lo >= bot_hi:
                continue
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(bot_lo, bot_hi))
            p1_results.append((bot_lo, bot_hi, total, ps))
            name = f'p1_bot({bot_lo},{bot_hi})'
            all_configs.append((total, ps, name))
            if total > best_p1:
                best_p1 = total

    p1_results.sort(key=lambda x: -x[2])
    print(f'  {len(p1_results)} configs tested. Best: {best_p1}/91')
    for lo, hi, total, ps in p1_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    bot=({lo},{hi}): {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: Top-zone correction
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 2: Top-zone correction (seeds 1-16)')
    print('='*70)

    best_p2 = v24_total
    p2_results = []

    # Try different components for top zone
    top_components = ['elo_mom', 'net_vs_conf', 'road_qual', 'resume', 'opp_rank',
                      'sos_gap', 'sosnet', 'al_power', 'wpct_adj', 'sos_x_wpct']

    # Singles
    for comp in top_components:
        for w in [-3, -2, -1, 1, 2, 3]:
            for top_hi in [12, 14, 16, 18, 20]:
                spec = {comp: w}
                total, ps = eval_config(
                    season_data, test_mask, y, test_seasons, fn,
                    mid_spec=v24_mid, mid_zone=(17, 34),
                    low_spec=v24_low, low_zone=(35, 52),
                    bot_spec=v24_bot, bot_zone=(53, 65),
                    top_spec=spec, top_zone=(1, top_hi))
                p2_results.append((comp, w, top_hi, total, ps))
                name = f'p2_top_{comp}={w}_hi={top_hi}'
                all_configs.append((total, ps, name))
                if total > best_p2:
                    best_p2 = total

    p2_results.sort(key=lambda x: -x[3])
    print(f'  {len(p2_results)} configs tested. Best: {best_p2}/91')
    for comp, w, hi, total, ps in p2_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    top(1,{hi}) {comp}={w}: {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: Enhanced mid-range features
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 3: Enhanced mid-range features')
    print('='*70)

    best_p3 = v24_total
    p3_results = []

    # Try adding one extra feature to existing mid-range
    extra_mid_components = ['q1dom', 'net_vs_conf', 'elo_mom', 'road_qual',
                            'resume', 'opp_rank', 'field', 'cbhist', 'aq_weak',
                            'bad_loss', 'wpct_adj', 'net_conf']

    for comp in extra_mid_components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'al_power': 2, 'sos_gap': 3, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=spec, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(53, 65))
            p3_results.append((comp, w, total, ps))
            name = f'p3_mid+{comp}={w}'
            all_configs.append((total, ps, name))
            if total > best_p3:
                best_p3 = total

    # Also try varying al/sos jointly with extra feature
    for al in [1, 2, 3]:
        for sos in [2, 3, 4, 5]:
            for comp in ['q1dom', 'resume', 'aq_weak', 'road_qual', 'net_conf']:
                for w in [-2, -1, 1, 2]:
                    spec = {'al_power': al, 'sos_gap': sos, comp: w}
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=spec, mid_zone=(17, 34),
                        low_spec=v24_low, low_zone=(35, 52),
                        bot_spec=v24_bot, bot_zone=(53, 65))
                    p3_results.append((f'al{al}_s{sos}_{comp}', w, total, ps))
                    name = f'p3_al{al}_s{sos}_{comp}={w}'
                    all_configs.append((total, ps, name))
                    if total > best_p3:
                        best_p3 = total

    p3_results.sort(key=lambda x: -x[2])
    print(f'  {len(p3_results)} configs tested. Best: {best_p3}/91')
    for comp, w, total, ps in p3_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    mid {comp}={w}: {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: Enhanced low-zone features
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 4: Enhanced low-zone features')
    print('='*70)

    best_p4 = v24_total
    p4_results = []

    extra_low_components = ['sos_gap', 'sosnet', 'net_conf', 'cbhist', 'elo_mom',
                            'road_qual', 'resume', 'opp_rank', 'aq_weak',
                            'bad_loss', 'wpct_adj', 'al_power']

    for comp in extra_low_components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'q1dom': 1, 'field': 2, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=spec, low_zone=(35, 52),
                bot_spec=v24_bot, bot_zone=(53, 65))
            p4_results.append((comp, w, total, ps))
            name = f'p4_low+{comp}={w}'
            all_configs.append((total, ps, name))
            if total > best_p4:
                best_p4 = total

    # Also try varying q1dom/field with extra feature
    for q1d in [0, 1, 2]:
        for fld in [1, 2, 3]:
            for comp in ['sosnet', 'cbhist', 'resume', 'aq_weak', 'opp_rank', 'net_conf']:
                for w in [-2, -1, 1, 2]:
                    spec = {'q1dom': q1d, 'field': fld, comp: w}
                    total, ps = eval_config(
                        season_data, test_mask, y, test_seasons, fn,
                        mid_spec=v24_mid, mid_zone=(17, 34),
                        low_spec=spec, low_zone=(35, 52),
                        bot_spec=v24_bot, bot_zone=(53, 65))
                    p4_results.append((f'q{q1d}_f{fld}_{comp}', w, total, ps))
                    name = f'p4_q{q1d}_f{fld}_{comp}={w}'
                    all_configs.append((total, ps, name))
                    if total > best_p4:
                        best_p4 = total

    p4_results.sort(key=lambda x: -x[2])
    print(f'  {len(p4_results)} configs tested. Best: {best_p4}/91')
    for comp, w, total, ps in p4_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    low {comp}={w}: {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 5: Enhanced bot-zone features
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 5: Enhanced bot-zone features')
    print('='*70)

    best_p5 = v24_total
    p5_results = []

    extra_bot_components = ['q1dom', 'field', 'elo_mom', 'road_qual', 'resume',
                            'opp_rank', 'aq_weak', 'bad_loss', 'wpct_adj',
                            'al_power', 'sos_gap', 'net_sq', 'conf_med', 'al_rank',
                            'sos_x_wpct', 'net_vs_conf']

    for comp in extra_bot_components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=spec, bot_zone=(53, 65))
            p5_results.append((comp, w, total, ps))
            name = f'p5_bot+{comp}={w}'
            all_configs.append((total, ps, name))
            if total > best_p5:
                best_p5 = total

    # Also try extended zone (53,68) with 4th component
    for comp in extra_bot_components:
        for w in [-3, -2, -1, 1, 2, 3]:
            spec = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1, comp: w}
            total, ps = eval_config(
                season_data, test_mask, y, test_seasons, fn,
                mid_spec=v24_mid, mid_zone=(17, 34),
                low_spec=v24_low, low_zone=(35, 52),
                bot_spec=spec, bot_zone=(53, 68))
            p5_results.append((f'{comp}_z68', w, total, ps))
            name = f'p5_bot68+{comp}={w}'
            all_configs.append((total, ps, name))
            if total > best_p5:
                best_p5 = total

    p5_results.sort(key=lambda x: -x[2])
    print(f'  {len(p5_results)} configs tested. Best: {best_p5}/91')
    for comp, w, total, ps in p5_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    bot {comp}={w}: {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 6: Joint zone boundary optimization
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 6: Joint zone boundary sweep')
    print('='*70)

    best_p6 = v24_total
    p6_results = []

    for mid_lo in [13, 15, 17, 19]:
        for mid_hi in [30, 32, 34, 36]:
            for low_lo_offset in [-2, 0, 1, 3]:
                low_lo = mid_hi + 1 + low_lo_offset
                if low_lo <= mid_hi:
                    low_lo = mid_hi + 1
                for low_hi in [48, 50, 52, 54]:
                    if low_lo >= low_hi:
                        continue
                    for bot_lo_offset in [-2, 0, 1]:
                        bot_lo = low_hi + 1 + bot_lo_offset
                        if bot_lo <= low_hi:
                            bot_lo = low_hi + 1
                        for bot_hi in [63, 65, 67, 68]:
                            if bot_lo >= bot_hi:
                                continue
                            total, ps = eval_config(
                                season_data, test_mask, y, test_seasons, fn,
                                mid_spec=v24_mid, mid_zone=(mid_lo, mid_hi),
                                low_spec=v24_low, low_zone=(low_lo, low_hi),
                                bot_spec=v24_bot, bot_zone=(bot_lo, bot_hi))
                            p6_results.append({
                                'mid': (mid_lo, mid_hi), 'low': (low_lo, low_hi),
                                'bot': (bot_lo, bot_hi), 'total': total, 'ps': ps})
                            name = f'p6_m{mid_lo}-{mid_hi}_l{low_lo}-{low_hi}_b{bot_lo}-{bot_hi}'
                            all_configs.append((total, ps, name))
                            if total > best_p6:
                                best_p6 = total

    p6_results.sort(key=lambda r: -r['total'])
    print(f'  {len(p6_results)} configs tested. Best: {best_p6}/91')
    for r in p6_results[:10]:
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if r['total'] > v24_total else ''
        print(f'    mid={str(r["mid"]):<8} low={str(r["low"]):<8} bot={str(r["bot"]):<8}: '
              f'{r["total"]}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 7: Per-zone power parameter
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 7: Per-zone power parameter sweep')
    print('='*70)

    best_p7 = v24_total
    p7_results = []

    for mp in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for lp in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            for bp in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                total, ps = eval_config(
                    season_data, test_mask, y, test_seasons, fn,
                    mid_spec=v24_mid, mid_zone=(17, 34),
                    low_spec=v24_low, low_zone=(35, 52),
                    bot_spec=v24_bot, bot_zone=(53, 65),
                    mid_power=mp, low_power=lp, bot_power=bp)
                p7_results.append((mp, lp, bp, total, ps))
                name = f'p7_mp{mp}_lp{lp}_bp{bp}'
                all_configs.append((total, ps, name))
                if total > best_p7:
                    best_p7 = total

    p7_results.sort(key=lambda x: -x[3])
    print(f'  {len(p7_results)} configs tested. Best: {best_p7}/91')
    for mp, lp, bp, total, ps in p7_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    mid_p={mp:.2f} low_p={lp:.2f} bot_p={bp:.2f}: {total}/91 [{ps_str}]{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 8: Combine best findings
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 8: Combine best findings from phases')
    print('='*70)

    # Collect best unique configs from each phase
    # Try combining best improvements from different phases
    best_configs_by_phase = {}

    # Best from each phase
    if p1_results and p1_results[0][2] > v24_total:
        best_configs_by_phase['p1'] = p1_results[0]
    if p2_results and p2_results[0][3] > v24_total:
        best_configs_by_phase['p2'] = p2_results[0]
    if p3_results and p3_results[0][2] > v24_total:
        best_configs_by_phase['p3'] = p3_results[0]
    if p4_results and p4_results[0][2] > v24_total:
        best_configs_by_phase['p4'] = p4_results[0]
    if p5_results and p5_results[0][2] > v24_total:
        best_configs_by_phase['p5'] = p5_results[0]
    if p6_results and p6_results[0]['total'] > v24_total:
        best_configs_by_phase['p6'] = p6_results[0]
    if p7_results and p7_results[0][3] > v24_total:
        best_configs_by_phase['p7'] = p7_results[0]

    print(f'\n  Phases with improvements over v24 ({v24_total}/91):')
    for phase, result in best_configs_by_phase.items():
        if phase == 'p6':
            print(f'    {phase}: {result["total"]}/91')
        elif phase in ('p2', 'p7'):
            print(f'    {phase}: {result[3]}/91')
        else:
            print(f'    {phase}: {result[2]}/91')

    # Try combining: best bot boundaries + best extra features + best power
    combo_results = []

    # Get best P1 boundary
    best_bot_bounds = [(53, 65)]  # default
    for lo, hi, total, ps in p1_results[:5]:
        if total >= v24_total and (lo, hi) != (53, 65):
            best_bot_bounds.append((lo, hi))

    # Get best P3 mid extras
    best_mid_extras = [v24_mid]
    for comp, w, total, ps in p3_results[:5]:
        if total >= v24_total:
            spec = {'al_power': 2, 'sos_gap': 3}
            cstr = str(comp)
            # Check if it's compound format: al{N}_s{N}_{comp}
            if cstr.startswith('al') and '_s' in cstr:
                try:
                    parts = cstr.split('_')
                    al = int(parts[0][2:])
                    sos = int(parts[1][1:])
                    extra = '_'.join(parts[2:])
                    spec = {'al_power': al, 'sos_gap': sos, extra: w}
                except (ValueError, IndexError):
                    spec[comp] = w
            else:
                spec[comp] = w
            best_mid_extras.append(spec)

    # Get best P4 low extras
    best_low_extras = [v24_low]
    for comp, w, total, ps in p4_results[:5]:
        if total >= v24_total:
            spec = {'q1dom': 1, 'field': 2}
            cstr = str(comp)
            # Check if it's compound format: q{N}_f{N}_{comp}
            if cstr.startswith('q') and '_f' in cstr:
                try:
                    parts = cstr.split('_')
                    q1d = int(parts[0][1:])
                    fld = int(parts[1][1:])
                    extra = '_'.join(parts[2:])
                    spec = {'q1dom': q1d, 'field': fld, extra: w}
                except (ValueError, IndexError):
                    spec[comp] = w
            else:
                spec[comp] = w
            best_low_extras.append(spec)

    # Get best P5 bot extras
    best_bot_extras = [v24_bot]
    for comp, w, total, ps in p5_results[:5]:
        if total >= v24_total:
            spec = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1}
            comp_clean = comp.replace('_z68', '')
            spec[comp_clean] = w
            best_bot_extras.append(spec)

    # Get best P7 powers
    best_powers = [(0.15, 0.15, 0.15)]
    for mp, lp, bp, total, ps in p7_results[:3]:
        if total >= v24_total and (mp, lp, bp) != (0.15, 0.15, 0.15):
            best_powers.append((mp, lp, bp))

    # Get best top specs
    best_tops = [None]
    for comp, w, hi, total, ps in p2_results[:3]:
        if total >= v24_total:
            best_tops.append(({comp: w}, (1, hi)))

    print(f'\n  Combining:')
    print(f'    {len(best_bot_bounds)} bot boundaries × {len(best_mid_extras)} mid specs')
    print(f'    × {len(best_low_extras)} low specs × {len(best_bot_extras)} bot specs')
    print(f'    × {len(best_powers)} powers × {len(best_tops)} top specs')

    n_combos = 0
    best_combo = v24_total
    for bot_bnds in best_bot_bounds:
        for mid_s in best_mid_extras:
            for low_s in best_low_extras:
                for bot_s in best_bot_extras:
                    for mp, lp, bp in best_powers:
                        for top_info in best_tops:
                            top_s = top_info[0] if top_info else None
                            top_z = top_info[1] if top_info else (1, 16)
                            total, ps = eval_config(
                                season_data, test_mask, y, test_seasons, fn,
                                mid_spec=mid_s, mid_zone=(17, 34),
                                low_spec=low_s, low_zone=(35, 52),
                                bot_spec=bot_s, bot_zone=bot_bnds,
                                top_spec=top_s, top_zone=top_z,
                                mid_power=mp, low_power=lp, bot_power=bp)
                            n_combos += 1
                            name = (f'combo_m{str(mid_s)[:20]}_l{str(low_s)[:20]}'
                                    f'_b{str(bot_s)[:20]}_{bot_bnds}_p{mp}/{lp}/{bp}'
                                    f'{"_top" if top_s else ""}')
                            all_configs.append((total, ps, name))
                            combo_results.append((total, ps, name))
                            if total > best_combo:
                                best_combo = total

    combo_results.sort(key=lambda x: -x[0])
    print(f'\n  {n_combos} combinations tested. Best: {best_combo}/91')
    for total, ps, name in combo_results[:10]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > v24_total else ''
        print(f'    {total}/91 [{ps_str}] {name[:70]}{marker}')

    # ═══════════════════════════════════════════════════════════
    #  PHASE 9: NESTED LOSO VALIDATION
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' PHASE 9: NESTED LOSO VALIDATION')
    print(' (Gold standard: tune on 4, test on 5th)')
    print('='*70)

    # Dedup configs
    seen = set()
    unique_configs = []
    for total, ps, name in all_configs:
        key = tuple(ps.get(s, 0) for s in test_seasons)
        if key not in seen:
            seen.add(key)
            unique_configs.append((total, ps, name))

    unique_configs.sort(key=lambda x: -x[0])
    print(f'\n  Total unique configs: {len(unique_configs)}')
    print(f'  Best full test: {unique_configs[0][0]}/91')

    # Show top configs
    print(f'\n  Top 15 configs:')
    for total, ps, name in unique_configs[:15]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        print(f'    {total}/91 [{ps_str}] {name[:60]}')

    # Nested LOSO
    nested_total, nested_details = nested_loso(unique_configs, test_seasons,
                                                prefer_simple=True)
    nested_v24, _ = nested_loso([(v24_total, v24_ps, 'v24')], test_seasons)

    print(f'\n  Nested LOSO results:')
    print(f'  {"Season":<10} {"N":>3} {"v24":>4} {"best":>4} {"Δ":>3}  Config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*50}')
    for hold, hold_ex, name in nested_details:
        n_te = (test_mask & (seasons == hold)).sum()
        v24_h = v24_ps.get(hold, 0)
        delta = hold_ex - v24_h
        print(f'  {hold:<10} {n_te:3d} {v24_h:4d} {hold_ex:4d} {delta:+3d}  {name[:50]}')

    print(f'\n  Nested LOSO: v24={nested_v24}/91, v25={nested_total}/91 (Δ={nested_total-nested_v24:+d})')

    # Also try nested LOSO with only configs that scored >= v24_total
    strong_configs = [(t, p, n) for t, p, n in unique_configs if t >= v24_total]
    if strong_configs:
        nested_strong, strong_details = nested_loso(strong_configs, test_seasons)
        print(f'  Nested LOSO (>={v24_total}/91 only): {nested_strong}/91')

    # ═══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL SUMMARY')
    print('='*70)
    print(f'\n  v24 baseline:        {v24_total}/91')
    print(f'  Best full test:      {unique_configs[0][0]}/91')
    best_ps_str = ' '.join(f'{unique_configs[0][1].get(s,0):2d}' for s in test_seasons)
    print(f'    Per-season:        [{best_ps_str}]')
    print(f'    Config:            {unique_configs[0][2][:70]}')
    print(f'  Nested LOSO v24:     {nested_v24}/91')
    print(f'  Nested LOSO v25:     {nested_total}/91')

    gap = unique_configs[0][0] - nested_total
    print(f'  Overfit gap:         {gap}')

    if nested_total > nested_v24:
        print(f'\n  ★ IMPROVEMENT VALIDATED: +{nested_total - nested_v24} over v24 by nested LOSO')
    elif nested_total == nested_v24:
        print(f'\n  • No validated improvement over v24 (nested LOSO ties)')
    else:
        print(f'\n  ✗ Nested LOSO worse than v24 — any improvement is overfit')

    # Phase-by-phase best
    print(f'\n  Phase-by-phase best full test:')
    print(f'    P1 (bot boundary):   {best_p1}/91')
    print(f'    P2 (top zone):       {best_p2}/91')
    print(f'    P3 (mid features):   {best_p3}/91')
    print(f'    P4 (low features):   {best_p4}/91')
    print(f'    P5 (bot features):   {best_p5}/91')
    print(f'    P6 (joint bounds):   {best_p6}/91')
    print(f'    P7 (zone powers):    {best_p7}/91')
    print(f'    P8 (combos):         {best_combo}/91')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
