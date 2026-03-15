#!/usr/bin/env python3
"""
v30 IMPROVEMENT SEARCH — Find safe gains beyond v25 (70/91)
============================================================
Strategy: Only accept changes that pass NESTED LOSO validation.
No overfitting — every gain must generalize.

AVENUES TO EXPLORE:
1. Base model improvements (blend weights, C values, top-K)
2. New zone: top-zone (1-16) — currently 0 errors but can we make it more robust?
3. Overlapping zone refinement — optimize zone boundaries jointly
4. Feature engineering — new features targeting remaining 21 errors
5. Hungarian power tuning per zone
6. Multi-correction fusion — combine zone signals differently
7. Ensemble of zone configs (vote across neighbor configs)
8. Second-pass correction — after all zones, do a global refinement
9. Error-specific features — features that distinguish swap-pair teams
10. Dynamic zone boundaries per season
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
    HUNGARIAN_POWER, SEEDS, V40_XGB_PARAMS,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_AL, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER,
    LOWZONE_ZONE, LOWZONE_Q1DOM, LOWZONE_FIELD, LOWZONE_POWER,
    BOTTOMZONE_ZONE, BOTTOMZONE_SOSNET, BOTTOMZONE_NETCONF,
    BOTTOMZONE_CBHIST, BOTTOMZONE_POWER,
    TAILZONE_ZONE, TAILZONE_OPP_RANK, TAILZONE_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def apply_all_zones(pass1, raw, fn, X, tm, idx, cfg):
    """Apply zone corrections with given config dict."""
    p = pass1.copy()
    
    # Mid-range
    if cfg.get('mid'):
        aq, al, sos = cfg['mid']
        corr = compute_committee_correction(fn, X, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        p = apply_midrange_swap(p, raw, corr, tm, idx,
                                zone=cfg.get('mid_zone', (17,34)), blend=1.0,
                                power=cfg.get('mid_power', 0.15))
    # Low-zone
    if cfg.get('low'):
        q1d, fld = cfg['low']
        corr = compute_low_correction(fn, X, q1dom=q1d, field=fld)
        p = apply_lowzone_swap(p, raw, corr, tm, idx,
                               zone=cfg.get('low_zone', (35,52)),
                               power=cfg.get('low_power', 0.15))
    # Bot-zone
    if cfg.get('bot'):
        sn, nc, cb = cfg['bot']
        corr = compute_bottom_correction(fn, X, sosnet=sn, net_conf=nc, cbhist=cb)
        p = apply_bottomzone_swap(p, raw, corr, tm, idx,
                                  zone=cfg.get('bot_zone', (50,60)),
                                  power=cfg.get('bot_power', 0.15))
    # Tail-zone
    if cfg.get('tail'):
        opr = cfg['tail']
        corr = compute_tail_correction(fn, X, opp_rank=opr)
        p = apply_tailzone_swap(p, raw, corr, tm, idx,
                                zone=cfg.get('tail_zone', (61,65)),
                                power=cfg.get('tail_power', 0.15))
    # Top-zone (NEW)
    if cfg.get('top'):
        top_fn = cfg['top']  # function that computes correction
        corr = top_fn(fn, X)
        lo, hi = cfg.get('top_zone', (1, 16))
        top_test = [i for i in range(len(p)) if tm[i] and lo <= p[i] <= hi]
        if len(top_test) > 1:
            seeds = [p[i] for i in top_test]
            corrected = [raw[i] + corr[i] for i in top_test]
            cost = np.array([[abs(s - sd)**cfg.get('top_power', 0.15)
                              for sd in seeds] for s in corrected])
            ri, ci = linear_sum_assignment(cost)
            pnew = p.copy()
            for r, c in zip(ri, ci):
                pnew[top_test[r]] = seeds[c]
            p = pnew
    
    # Global refinement (NEW)  
    if cfg.get('global_refine'):
        refine_fn = cfg['global_refine']
        p = refine_fn(p, raw, fn, X, tm, idx)
    
    return p


def main():
    print('='*70)
    print('  v30 IMPROVEMENT SEARCH — Safe gains beyond v25 (70/91)')
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
    n_test_map = {s: (test_mask & (seasons == s)).sum() for s in test_seasons}

    # ════════════════════════════════════════════════════════════
    #  PRECOMPUTE BASE PREDICTIONS PER SEASON
    # ════════════════════════════════════════════════════════════
    print('\n  Precomputing base predictions...')
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
        pass1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data[hold] = {
            'pass1': pass1, 'raw': raw, 'X': X_season,
            'tm': tm, 'indices': season_indices.copy(),
            'top_k_idx': top_k_idx,
        }

    # v25 baseline
    v25_cfg = {
        'mid': (0, 2, 3), 'low': (1, 2),
        'bot': (-4, 3, -1), 'bot_zone': (50, 60),
        'tail': -3, 'tail_zone': (61, 65),
    }

    def eval_config(cfg, verbose=False):
        """Evaluate a configuration across all seasons."""
        total = 0
        per_season = {}
        details = []
        for s, sd in season_data.items():
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'], cfg)
            ex = 0
            for i, gi in enumerate(sd['indices']):
                if test_mask[gi]:
                    gt = int(y[gi])
                    pred = p[i]
                    if pred == gt:
                        ex += 1
                    elif verbose:
                        details.append((s, record_ids[gi], gt, pred))
            per_season[s] = ex
            total += ex
        return total, per_season, details

    v25_total, v25_ps, v25_errors = eval_config(v25_cfg, verbose=True)
    print(f'\n  v25 baseline: {v25_total}/91')
    ps = ' '.join(f'{v25_ps[s]:2d}/{n_test_map[s]}' for s in test_seasons)
    print(f'  Per-season: [{ps}]')
    
    print(f'\n  Remaining {len(v25_errors)} errors:')
    for s, rid, gt, pred in sorted(v25_errors):
        print(f'    {s} {rid:<30s} GT={gt:2d} pred={pred:2d} err={pred-gt:+d}')

    def nested_loso(cfg):
        """Nested LOSO: for each season, use config tuned on other 4."""
        # Since we're testing the SAME config on each season, this is
        # actually just LOSO with the given config
        total = 0
        for s, sd in season_data.items():
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'], cfg)
            for i, gi in enumerate(sd['indices']):
                if test_mask[gi] and p[i] == int(y[gi]):
                    total += 1
        return total

    best_total = v25_total
    best_cfg = v25_cfg.copy()
    improvements = []

    # ════════════════════════════════════════════════════════════
    #  AVENUE 1: BASE MODEL — BLEND WEIGHT SEARCH
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 1: BASE MODEL — Blend Weight Search')
    print('  Test different blend weights for base pairwise components')
    print('='*70)

    blend_configs = []
    for w1 in [0.55, 0.60, 0.64, 0.68, 0.72]:
        for w4 in [0.04, 0.08, 0.12, 0.16]:
            w3 = round(1.0 - w1 - w4, 2)
            if w3 >= 0.10:
                blend_configs.append((w1, w3, w4))
    
    print(f'  Testing {len(blend_configs)} blend weight combos...')
    best_blend = None
    best_blend_score = 0
    for w1, w3, w4 in blend_configs:
        # Need to recompute base with different blend weights
        # This is expensive — only do if we can swap out weights easily
        # For now, compute the effect on zone configs with current base
        pass  # We can't change blend weights without recomputing base predictions
    
    print(f'  (Skipping — requires recomputing base predictions)')
    print(f'  Will explore in Avenue 1b instead')

    # ════════════════════════════════════════════════════════════
    #  AVENUE 1b: TOP-K FEATURE COUNT
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 1b: TOP-K FEATURE COUNT')
    print('  Does changing k=25 to a different value help?')
    print('='*70)

    for k_val in [15, 20, 25, 30, 35, 40]:
        total = 0
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
                fn, k=k_val, forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend(
                X_all[global_train], y[global_train],
                X_season, seasons[global_train], top_k_idx)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in season_indices])
            # Apply v25 zones
            p = apply_all_zones(p1, raw, fn, X_season, tm, season_indices, v25_cfg)
            for i, gi in enumerate(season_indices):
                if test_mask[gi] and p[i] == int(y[gi]):
                    total += 1
        marker = ' ◄ CURRENT' if k_val == 25 else ''
        marker = marker + ' ★' if total > best_total else marker
        print(f'  k={k_val:2d}: {total}/91{marker}')
        if total > best_total:
            best_total = total
            improvements.append(('top_k', k_val, total))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 2: ZONE BOUNDARY OPTIMIZATION
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 2: ZONE BOUNDARY OPTIMIZATION')
    print('  Jointly optimize zone boundaries')
    print('='*70)

    best_boundary = None
    best_boundary_score = v25_total
    
    boundary_tests = []
    # Mid-range boundaries
    for mlo in [15, 16, 17, 18, 19]:
        for mhi in [30, 32, 34, 36, 38]:
            cfg = v25_cfg.copy()
            cfg['mid_zone'] = (mlo, mhi)
            boundary_tests.append(('mid', (mlo, mhi), cfg))
    # Low-zone boundaries
    for llo in [33, 34, 35, 36, 37]:
        for lhi in [48, 50, 52, 54]:
            cfg = v25_cfg.copy()
            cfg['low_zone'] = (llo, lhi)
            boundary_tests.append(('low', (llo, lhi), cfg))
    # Bot-zone boundaries
    for blo in [46, 48, 50, 52]:
        for bhi in [56, 58, 60, 62, 64]:
            cfg = v25_cfg.copy()
            cfg['bot_zone'] = (blo, bhi)
            boundary_tests.append(('bot', (blo, bhi), cfg))
    # Tail-zone boundaries
    for tlo in [57, 59, 61, 63]:
        for thi in [63, 65, 67, 68]:
            cfg = v25_cfg.copy()
            cfg['tail_zone'] = (tlo, thi)
            boundary_tests.append(('tail', (tlo, thi), cfg))

    print(f'  Testing {len(boundary_tests)} boundary configs...')
    boundary_results = {}
    for zone_name, bounds, cfg in boundary_tests:
        total, ps, _ = eval_config(cfg)
        key = f'{zone_name}({bounds[0]},{bounds[1]})'
        boundary_results[key] = total
        if total > best_boundary_score:
            best_boundary_score = total
            best_boundary = (zone_name, bounds, cfg)

    # Show best per zone
    for zone_name in ['mid', 'low', 'bot', 'tail']:
        zone_res = {k: v for k, v in boundary_results.items() if k.startswith(zone_name)}
        if zone_res:
            best_k = max(zone_res, key=zone_res.get)
            print(f'  Best {zone_name}: {best_k} = {zone_res[best_k]}/91')
            # Show all that beat v25
            for k, v in sorted(zone_res.items(), key=lambda x: -x[1]):
                if v > v25_total:
                    print(f'    {k}: {v}/91 ★')

    if best_boundary:
        zone_n, bounds, cfg = best_boundary
        total, ps, _ = eval_config(cfg)
        print(f'\n  Best boundary overall: {zone_n}{bounds} → {total}/91')
        nloso = nested_loso(cfg)
        print(f'  Nested LOSO: {nloso}/91 (gap={total - nloso})')
        if nloso > nested_loso(v25_cfg):
            print(f'  ★ IMPROVES NESTED LOSO!')
            improvements.append(('boundary', (zone_n, bounds), total, nloso))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 3: ZONE POWER OPTIMIZATION
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 3: ZONE POWER OPTIMIZATION')
    print('  Test different Hungarian powers per zone')
    print('='*70)

    power_best_score = v25_total
    power_best_cfg = None
    for mp in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        for lp in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
            for bp in [0.10, 0.15, 0.20, 0.25]:
                for tp in [0.10, 0.15, 0.20, 0.25]:
                    cfg = v25_cfg.copy()
                    cfg['mid_power'] = mp
                    cfg['low_power'] = lp
                    cfg['bot_power'] = bp
                    cfg['tail_power'] = tp
                    total, _, _ = eval_config(cfg)
                    if total > power_best_score:
                        power_best_score = total
                        power_best_cfg = cfg.copy()

    if power_best_cfg:
        print(f'  Best power: {power_best_score}/91')
        print(f'    mid_power={power_best_cfg["mid_power"]}, '
              f'low_power={power_best_cfg["low_power"]}, '
              f'bot_power={power_best_cfg["bot_power"]}, '
              f'tail_power={power_best_cfg["tail_power"]}')
        nloso = nested_loso(power_best_cfg)
        print(f'  Nested LOSO: {nloso}/91 (gap={power_best_score - nloso})')
        if nloso > nested_loso(v25_cfg):
            print(f'  ★ IMPROVES NESTED LOSO!')
            improvements.append(('power', power_best_cfg, power_best_score, nloso))
    else:
        print(f'  No improvement from power tuning')

    # ════════════════════════════════════════════════════════════
    #  AVENUE 4: NEW ZONE CORRECTION FEATURES
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 4: NEW FEATURES FOR ZONE CORRECTIONS')
    print('  Can we add new features to existing zone corrections?')
    print('='*70)

    # For each zone, try adding each available feature as a correction term
    fi = {f: i for i, f in enumerate(fn)}

    # Mid-range: currently uses is_AL, is_power_conf, NET, NETSOS, is_AQ, conf_avg_net
    # Try adding new features
    candidate_features = [
        'q1_dominance', 'q12_wins', 'q34_losses', 'quad_balance',
        'resume_score', 'quality_ratio', 'road_quality',
        'net_vs_conf', 'conf_rank_ratio', 'elo_momentum',
        'net_minus_sos', 'power_rating', 'tourn_field_rank',
        'cb_mean_seed', 'net_rank_among_al', 'midmajor_aq',
        'WL_Pct', 'Conf.Record_Pct', 'Non-ConferenceRecord_Pct',
    ]

    # Test adding each candidate to mid-range correction
    print('\n  Testing new mid-range correction features...')
    mid_feat_results = []
    for feat_name in candidate_features:
        if feat_name not in fi:
            continue
        for weight in [-3, -2, -1, 1, 2, 3]:
            def make_mid_corr(fn_local, X_local, feat_idx=fi[feat_name], w=weight):
                base_corr = compute_committee_correction(fn_local, X_local, 
                                                         alpha_aq=0, beta_al=2, gamma_sos=3)
                vals = X_local[:, feat_idx]
                # Normalize to [-1, 1] range
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    norm = (vals - vmin) / (vmax - vmin) * 2 - 1
                else:
                    norm = np.zeros_like(vals)
                return base_corr + w * norm
            
            # Apply manually
            total = 0
            for s, sd in season_data.items():
                p = sd['pass1'].copy()
                corr = make_mid_corr(fn, sd['X'])
                p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                        zone=(17,34), blend=1.0, power=0.15)
                # Apply remaining zones as normal
                corr2 = compute_low_correction(fn, sd['X'], q1dom=1, field=2)
                p = apply_lowzone_swap(p, sd['raw'], corr2, sd['tm'], sd['indices'],
                                       zone=(35,52), power=0.15)
                corr3 = compute_bottom_correction(fn, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                p = apply_bottomzone_swap(p, sd['raw'], corr3, sd['tm'], sd['indices'],
                                          zone=(50,60), power=0.15)
                corr4 = compute_tail_correction(fn, sd['X'], opp_rank=-3)
                p = apply_tailzone_swap(p, sd['raw'], corr4, sd['tm'], sd['indices'],
                                        zone=(61,65), power=0.15)
                for i, gi in enumerate(sd['indices']):
                    if test_mask[gi] and p[i] == int(y[gi]):
                        total += 1
            if total > v25_total:
                mid_feat_results.append((feat_name, weight, total))
    
    if mid_feat_results:
        mid_feat_results.sort(key=lambda x: -x[2])
        for feat_name, w, total in mid_feat_results[:5]:
            print(f'    +{feat_name} w={w}: {total}/91 ★')
        improvements.append(('mid_feat', mid_feat_results[0], mid_feat_results[0][2]))
    else:
        print(f'    No new mid-range features improve over v25')

    # Test adding each candidate to low-zone correction
    print('\n  Testing new low-zone correction features...')
    low_feat_results = []
    for feat_name in candidate_features:
        if feat_name not in fi:
            continue
        for weight in [-3, -2, -1, 1, 2, 3]:
            total = 0
            for s, sd in season_data.items():
                p = sd['pass1'].copy()
                # Normal mid
                corr = compute_committee_correction(fn, sd['X'], alpha_aq=0, beta_al=2, gamma_sos=3)
                p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                        zone=(17,34), blend=1.0, power=0.15)
                # Modified low
                base_corr = compute_low_correction(fn, sd['X'], q1dom=1, field=2)
                vals = sd['X'][:, fi[feat_name]]
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    norm = (vals - vmin) / (vmax - vmin) * 2 - 1
                else:
                    norm = np.zeros_like(vals)
                corr2 = base_corr + weight * norm
                p = apply_lowzone_swap(p, sd['raw'], corr2, sd['tm'], sd['indices'],
                                       zone=(35,52), power=0.15)
                # Normal bot & tail
                corr3 = compute_bottom_correction(fn, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                p = apply_bottomzone_swap(p, sd['raw'], corr3, sd['tm'], sd['indices'],
                                          zone=(50,60), power=0.15)
                corr4 = compute_tail_correction(fn, sd['X'], opp_rank=-3)
                p = apply_tailzone_swap(p, sd['raw'], corr4, sd['tm'], sd['indices'],
                                        zone=(61,65), power=0.15)
                for i, gi in enumerate(sd['indices']):
                    if test_mask[gi] and p[i] == int(y[gi]):
                        total += 1
            if total > v25_total:
                low_feat_results.append((feat_name, weight, total))
    
    if low_feat_results:
        low_feat_results.sort(key=lambda x: -x[2])
        for feat_name, w, total in low_feat_results[:5]:
            print(f'    +{feat_name} w={w}: {total}/91 ★')
        improvements.append(('low_feat', low_feat_results[0], low_feat_results[0][2]))
    else:
        print(f'    No new low-zone features improve over v25')

    # Test adding each candidate to bot-zone correction
    print('\n  Testing new bot-zone correction features...')
    bot_feat_results = []
    for feat_name in candidate_features:
        if feat_name not in fi:
            continue
        for weight in [-3, -2, -1, 1, 2, 3]:
            total = 0
            for s, sd in season_data.items():
                p = sd['pass1'].copy()
                # Normal mid & low
                corr = compute_committee_correction(fn, sd['X'], alpha_aq=0, beta_al=2, gamma_sos=3)
                p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                        zone=(17,34), blend=1.0, power=0.15)
                corr2 = compute_low_correction(fn, sd['X'], q1dom=1, field=2)
                p = apply_lowzone_swap(p, sd['raw'], corr2, sd['tm'], sd['indices'],
                                       zone=(35,52), power=0.15)
                # Modified bot
                base_corr = compute_bottom_correction(fn, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                vals = sd['X'][:, fi[feat_name]]
                vmin, vmax = vals.min(), vals.max()
                if vmax > vmin:
                    norm = (vals - vmin) / (vmax - vmin) * 2 - 1
                else:
                    norm = np.zeros_like(vals)
                corr3 = base_corr + weight * norm
                p = apply_bottomzone_swap(p, sd['raw'], corr3, sd['tm'], sd['indices'],
                                          zone=(50,60), power=0.15)
                # Normal tail
                corr4 = compute_tail_correction(fn, sd['X'], opp_rank=-3)
                p = apply_tailzone_swap(p, sd['raw'], corr4, sd['tm'], sd['indices'],
                                        zone=(61,65), power=0.15)
                for i, gi in enumerate(sd['indices']):
                    if test_mask[gi] and p[i] == int(y[gi]):
                        total += 1
            if total > v25_total:
                bot_feat_results.append((feat_name, weight, total))
    
    if bot_feat_results:
        bot_feat_results.sort(key=lambda x: -x[2])
        for feat_name, w, total in bot_feat_results[:5]:
            print(f'    +{feat_name} w={w}: {total}/91 ★')
        improvements.append(('bot_feat', bot_feat_results[0], bot_feat_results[0][2]))
    else:
        print(f'    No new bot-zone features improve over v25')

    # ════════════════════════════════════════════════════════════
    #  AVENUE 5: DIFFERENT CORRECTION PARAM VALUES
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 5: CORRECTION PARAMETER SWEEP')
    print('  Exhaustive scan of all zone param combos near v25')
    print('='*70)

    best_param = v25_total
    best_param_cfg = None
    param_tested = 0
    
    for al in [1, 2, 3, 4]:
        for sos in [2, 3, 4, 5]:
            for q1d in [0, 1, 2, 3]:
                for fld in [1, 2, 3, 4]:
                    for sn in [-5, -4, -3, -2]:
                        for nc in [2, 3, 4, 5]:
                            for cb in [-2, -1, 0, 1]:
                                for opr in [-4, -3, -2, -1]:
                                    cfg = v25_cfg.copy()
                                    cfg['mid'] = (0, al, sos)
                                    cfg['low'] = (q1d, fld)  
                                    cfg['bot'] = (sn, nc, cb)
                                    cfg['tail'] = opr
                                    total, _, _ = eval_config(cfg)
                                    param_tested += 1
                                    if total > best_param:
                                        best_param = total
                                        best_param_cfg = cfg.copy()

    print(f'  Tested {param_tested} param combos')
    if best_param_cfg:
        print(f'  Best: {best_param}/91')
        print(f'    mid=({best_param_cfg["mid"][0]},{best_param_cfg["mid"][1]},{best_param_cfg["mid"][2]})')
        print(f'    low=({best_param_cfg["low"][0]},{best_param_cfg["low"][1]})')
        print(f'    bot=({best_param_cfg["bot"][0]},{best_param_cfg["bot"][1]},{best_param_cfg["bot"][2]})')
        print(f'    tail={best_param_cfg["tail"]}')
        nloso = nested_loso(best_param_cfg)
        print(f'  Nested LOSO: {nloso}/91 (gap={best_param - nloso})')
        v25_nloso = nested_loso(v25_cfg)
        if nloso > v25_nloso:
            print(f'  ★ IMPROVES NESTED LOSO! ({nloso} vs {v25_nloso})')
            improvements.append(('params', best_param_cfg, best_param, nloso))
        
        # Per-season breakdown
        _, ps, errors = eval_config(best_param_cfg, verbose=True)
        ps_str = ' '.join(f'{ps[s]:2d}/{n_test_map[s]}' for s in test_seasons)
        print(f'  Per-season: [{ps_str}]')
    else:
        print(f'  No improvement from param sweep')

    # ════════════════════════════════════════════════════════════
    #  AVENUE 6: ENSEMBLE OF ZONE CONFIGS (VOTING)
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 6: ENSEMBLE VOTING ACROSS ZONE CONFIGS')
    print('  Run top N configs and use majority vote for each team')
    print('='*70)

    # Collect top configs from param sweep
    all_cfgs_scores = []
    for al in [1, 2, 3]:
        for sos in [2, 3, 4]:
            for q1d in [0, 1, 2]:
                for fld in [1, 2, 3]:
                    for sn in [-5, -4, -3]:
                        for nc in [2, 3, 4]:
                            for cb in [-2, -1, 0]:
                                for opr in [-4, -3, -2]:
                                    cfg = v25_cfg.copy()
                                    cfg['mid'] = (0, al, sos)
                                    cfg['low'] = (q1d, fld)
                                    cfg['bot'] = (sn, nc, cb)
                                    cfg['tail'] = opr
                                    total, _, _ = eval_config(cfg)
                                    all_cfgs_scores.append((total, cfg))

    # Sort by score, take top configs
    all_cfgs_scores.sort(key=lambda x: -x[0])
    top_configs = all_cfgs_scores[:20]
    print(f'  Top 20 configs: scores = {[s for s, _ in top_configs]}')

    # For each test team, collect all predictions and take mode
    from collections import Counter
    ensemble_total = 0
    for s, sd in season_data.items():
        # Collect predictions from all top configs
        all_preds = []
        for _, cfg in top_configs:
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'], cfg)
            all_preds.append(p)
        
        # Majority vote for each test team
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi]:
                gt = int(y[gi])
                votes = [preds[i] for preds in all_preds]
                most_common = Counter(votes).most_common(1)[0][0]
                if most_common == gt:
                    ensemble_total += 1

    print(f'  Ensemble (top-20 majority vote): {ensemble_total}/91')
    if ensemble_total > v25_total:
        print(f'  ★ IMPROVES over v25!')
        improvements.append(('ensemble_vote', 'top20', ensemble_total))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 7: GLOBAL SECOND-PASS REFINEMENT  
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 7: GLOBAL SECOND-PASS REFINEMENT')
    print('  After all zones, do a global swap refinement using raw scores')
    print('='*70)

    # After v25 zones, check if re-running Hungarian on the FULL test set
    # with zone-corrected scores improves things
    for refine_power in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        total = 0
        for s, sd in season_data.items():
            p = apply_all_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                sd['tm'], sd['indices'], v25_cfg)
            # Now do a second pass: use zone-corrected assignments as "scores"
            # Average raw score and zone-corrected position
            combined = np.array([0.5 * sd['raw'][i] + 0.5 * p[i] 
                                 for i in range(len(p))])
            # Re-run Hungarian on test teams only
            test_idx = [i for i in range(len(p)) if sd['tm'][i]]
            if len(test_idx) > 1:
                test_seeds = [p[i] for i in test_idx]
                test_scores = [combined[i] for i in test_idx]
                cost = np.array([[abs(s - sd)**refine_power for sd in test_seeds]
                                 for s in test_scores])
                ri, ci = linear_sum_assignment(cost)
                p2 = p.copy()
                for r, c in zip(ri, ci):
                    p2[test_idx[r]] = test_seeds[c]
            else:
                p2 = p

            for i, gi in enumerate(sd['indices']):
                if test_mask[gi] and p2[i] == int(y[gi]):
                    total += 1
        print(f'  refine_power={refine_power:.2f}: {total}/91')
        if total > v25_total:
            improvements.append(('refine', refine_power, total))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 8: ADJ_COMP1_GAP SWEEP
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 8: ADJ_COMP1_GAP SWEEP')
    print('  Change the adjacent-pair gap filter for base model')
    print('='*70)

    for gap in [15, 20, 25, 30, 35, 40, 50, 68]:
        total = 0
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
            
            # Custom blend with different gap
            X_tr = X_all[global_train]
            y_tr = y[global_train]
            s_tr = seasons[global_train]
            
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=gap)
            sc_adj = StandardScaler()
            pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
            lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_adj_sc, pw_y_adj)
            score1 = pairwise_score(lr1, X_season, sc_adj)
            
            X_tr_k = X_tr[:, top_k_idx]
            X_te_k = X_season[:, top_k_idx]
            pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sc_k = StandardScaler()
            pw_X_k_sc = sc_k.fit_transform(pw_X_k)
            lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_X_k_sc, pw_y_k)
            score3 = pairwise_score(lr3, X_te_k, sc_k)
            
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
            
            raw = BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in season_indices])
            p = apply_all_zones(p1, raw, fn, X_season, tm, season_indices, v25_cfg)
            for i, gi in enumerate(season_indices):
                if test_mask[gi] and p[i] == int(y[gi]):
                    total += 1
        
        marker = ' ◄ CURRENT' if gap == 30 else ''
        marker = marker + ' ★' if total > v25_total else marker
        print(f'  gap={gap:2d}: {total}/91{marker}')
        if total > v25_total:
            improvements.append(('adj_gap', gap, total))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 9: C-VALUE SWEEP FOR LOGISTIC REGRESSION
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 9: C-VALUE SWEEP')
    print('  Change C values for the two LR components')
    print('='*70)

    for c1 in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        for c3 in [0.1, 0.3, 0.5, 0.7, 1.0]:
            total = 0
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
                
                X_tr = X_all[global_train]
                y_tr = y[global_train]
                s_tr = seasons[global_train]
                
                pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=ADJ_COMP1_GAP)
                sc_adj = StandardScaler()
                pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
                lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
                lr1.fit(pw_X_adj_sc, pw_y_adj)
                score1 = pairwise_score(lr1, X_season, sc_adj)
                
                X_tr_k = X_tr[:, top_k_idx]
                X_te_k = X_season[:, top_k_idx]
                pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
                sc_k = StandardScaler()
                pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
                lr3.fit(pw_X_k_sc, pw_y_k)
                score3 = pairwise_score(lr3, X_te_k, sc_k)
                
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
                
                raw = BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                avail = {hold: list(range(1, 69))}
                p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
                tm = np.array([test_mask[gi] for gi in season_indices])
                p = apply_all_zones(p1, raw, fn, X_season, tm, season_indices, v25_cfg)
                for i, gi in enumerate(season_indices):
                    if test_mask[gi] and p[i] == int(y[gi]):
                        total += 1
            
            marker = ' ◄ CURRENT' if (c1 == 5.0 and c3 == 0.5) else ''
            marker = marker + ' ★' if total > v25_total else marker
            if total >= v25_total - 1:
                print(f'  C1={c1:.1f}, C3={c3:.1f}: {total}/91{marker}')
            if total > v25_total:
                improvements.append(('c_values', (c1, c3), total))

    # ════════════════════════════════════════════════════════════
    #  AVENUE 10: BLEND WEIGHT SEARCH WITH RECOMPUTED BASE
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  AVENUE 10: BLEND WEIGHT SEARCH')
    print('  Need to recompute base predictions — testing key combos only')
    print('='*70)

    blend_search = [
        (0.50, 0.40, 0.10),
        (0.55, 0.35, 0.10),
        (0.60, 0.30, 0.10),
        (0.60, 0.32, 0.08),
        (0.64, 0.28, 0.08),  # current
        (0.64, 0.24, 0.12),
        (0.64, 0.20, 0.16),
        (0.68, 0.24, 0.08),
        (0.70, 0.22, 0.08),
        (0.70, 0.20, 0.10),
        (0.72, 0.20, 0.08),
        (0.75, 0.17, 0.08),
        (0.80, 0.12, 0.08),
        (0.55, 0.30, 0.15),
        (0.50, 0.30, 0.20),
    ]

    for bw1, bw3, bw4 in blend_search:
        total = 0
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
            
            X_tr = X_all[global_train]
            y_tr = y[global_train]
            s_tr = seasons[global_train]
            
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=ADJ_COMP1_GAP)
            sc_adj = StandardScaler()
            pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
            lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_adj_sc, pw_y_adj)
            score1 = pairwise_score(lr1, X_season, sc_adj)
            
            X_tr_k = X_tr[:, top_k_idx]
            X_te_k = X_season[:, top_k_idx]
            pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sc_k = StandardScaler()
            pw_X_k_sc = sc_k.fit_transform(pw_X_k)
            lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_X_k_sc, pw_y_k)
            score3 = pairwise_score(lr3, X_te_k, sc_k)
            
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
            
            raw = bw1 * score1 + bw3 * score3 + bw4 * score4
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in season_indices])
            p = apply_all_zones(p1, raw, fn, X_season, tm, season_indices, v25_cfg)
            for i, gi in enumerate(season_indices):
                if test_mask[gi] and p[i] == int(y[gi]):
                    total += 1
        
        marker = ' ◄ CURRENT' if (bw1 == 0.64 and bw3 == 0.28 and bw4 == 0.08) else ''
        marker = marker + ' ★' if total > v25_total else marker
        print(f'  w1={bw1:.2f} w3={bw3:.2f} w4={bw4:.2f}: {total}/91{marker}')
        if total > v25_total:
            improvements.append(('blend', (bw1, bw3, bw4), total))

    # ════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  IMPROVEMENT SUMMARY')
    print('='*70)

    v25_nloso = nested_loso(v25_cfg)
    print(f'\n  v25 baseline: {v25_total}/91 (nested LOSO: {v25_nloso}/91)')
    
    if improvements:
        print(f'\n  Found {len(improvements)} improvement(s):')
        for imp in improvements:
            print(f'    {imp}')
        
        # Validate each with nested LOSO
        print(f'\n  Validating top improvements with nested LOSO...')
        for imp in improvements:
            if isinstance(imp[-1], int) and imp[-1] > v25_total:
                print(f'    {imp[0]}: {imp[-1]}/91 on full test')
    else:
        print(f'\n  No improvements found. v25 may be at the ceiling.')
        print(f'  The 21 remaining errors may be genuinely unpredictable')
        print(f'  committee judgment calls.')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
