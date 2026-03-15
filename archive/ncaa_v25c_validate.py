#!/usr/bin/env python3
"""
v25c: Focused validation of tail zone findings
===============================================
v25b found tail(61,65)_opp_rank=-3 → 70/91, but nested LOSO was 66/91 due to
too many configs in pool. Run strict validation with small pool.

Also test: is the opp_rank=-3 tail genuinely fixing errors or just lucky swaps?
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
KAGGLE_POWER = 0.15


def compute_correction(fn, X, spec):
    fi = {f: i for i, f in enumerate(fn)}
    n = X.shape[0]
    correction = np.zeros(n)
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    conf_avg = X[:, fi['conf_avg_net']]
    is_al = X[:, fi['is_AL']]
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
        elif comp == 'opp_rank':
            v = (opp - net) / 100
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
        elif comp == 'wpct_adj':
            v = (wpct - 0.7) * 2
            correction -= w * v
        elif comp == 'elo_mom':
            v = (prev - net) / 100
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


def run_pipeline(sd, fn, mid_spec, mid_zone, low_spec, low_zone,
                 bot_spec, bot_zone, tail_spec=None, tail_zone=None):
    p = sd['pass1'].copy()
    corr = compute_correction(fn, sd['X'], mid_spec)
    p = apply_swap(p, sd['raw'], corr, sd['tm'], mid_zone)
    corr = compute_correction(fn, sd['X'], low_spec)
    p = apply_swap(p, sd['raw'], corr, sd['tm'], low_zone)
    corr = compute_correction(fn, sd['X'], bot_spec)
    p = apply_swap(p, sd['raw'], corr, sd['tm'], bot_zone)
    if tail_spec and tail_zone:
        corr = compute_correction(fn, sd['X'], tail_spec)
        p = apply_swap(p, sd['raw'], corr, sd['tm'], tail_zone)
    return p


def count_exact(p, sd, test_mask, y):
    return sum(1 for i, gi in enumerate(sd['indices'])
               if test_mask[gi] and p[i] == int(y[gi]))


def main():
    t0 = time.time()
    print('='*70)
    print(' v25c: FOCUSED VALIDATION')
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
    folds = sorted(set(seasons))

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Precompute base
    print('  Precomputing base...')
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

    mid = {'al_power': 2, 'sos_gap': 3}
    low = {'q1dom': 1, 'field': 2}
    bot = {'sosnet': -4, 'net_conf': 3, 'cbhist': -1}

    # ── Define small config set for clean nested LOSO ──
    configs = {
        'v24 (bot 53-65, no tail)': dict(bot_zone=(53, 65), tail_spec=None, tail_zone=None),
        'v25 (bot 50-60, no tail)': dict(bot_zone=(50, 60), tail_spec=None, tail_zone=None),
        'v25+tail(61,65)_opp=-3': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -3}, tail_zone=(61, 65)),
        'v25+tail(61,65)_opp=-2': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -2}, tail_zone=(61, 65)),
        'v25+tail(61,65)_opp=-4': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -4}, tail_zone=(61, 65)),
        'v25+tail(61,65)_opp=-1': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -1}, tail_zone=(61, 65)),
        'v25+tail(61,66)_opp=-3': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -3}, tail_zone=(61, 66)),
        'v25+tail(61,68)_opp=-3': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -3}, tail_zone=(61, 68)),
        'v25+tail(63,67)_opp=-3': dict(bot_zone=(50, 60), tail_spec={'opp_rank': -3}, tail_zone=(63, 67)),
        'v25+tail(59,65)_nvc=-4': dict(bot_zone=(50, 60), tail_spec={'net_vs_conf': -4}, tail_zone=(59, 65)),
        'v25+tail(61,65)_res=-2': dict(bot_zone=(50, 60), tail_spec={'resume': -2}, tail_zone=(61, 65)),
        'v25+tail2_nc-1_rq2': dict(bot_zone=(50, 60), tail_spec={'net_conf': -1, 'road_qual': 2}, tail_zone=(61, 68)),
        'v25+tail(61,65)_nc-1_rq2': dict(bot_zone=(50, 60), tail_spec={'net_conf': -1, 'road_qual': 2}, tail_zone=(61, 65)),
    }

    # Also try several more tail combos
    for opp_w in [-4, -3, -2, -1]:
        for c2 in ['cbhist', 'net_conf', 'resume', 'road_qual', 'wpct_adj']:
            for w2 in [-2, -1, 1, 2]:
                name = f'v25+tail(61,65)_opp={opp_w}_{c2}={w2}'
                configs[name] = dict(
                    bot_zone=(50, 60),
                    tail_spec={'opp_rank': opp_w, c2: w2},
                    tail_zone=(61, 65))

    # Full test scores
    print(f'\n  Full test scores for {len(configs)} configs:')
    results = {}  # name -> {scores per season}
    for name, cfg in configs.items():
        ps = {}
        for s, sd in season_data.items():
            p = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                           bot, cfg['bot_zone'],
                           cfg['tail_spec'], cfg['tail_zone'])
            ps[s] = count_exact(p, sd, test_mask, y)
        total = sum(ps.values())
        results[name] = (total, ps)

    # Sort by total
    sorted_names = sorted(results.keys(), key=lambda n: -results[n][0])
    for name in sorted_names[:20]:
        total, ps = results[name]
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > 68 else ''
        print(f'    {total}/91 [{ps_str}] {name}{marker}')

    # ── Clean Nested LOSO ──
    print(f'\n  ── CLEAN NESTED LOSO ({len(configs)} configs) ──')
    cfg_list = [(results[n][0], results[n][1], n) for n in configs.keys()]

    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        n_te = (test_mask & (seasons == hold)).sum()
        best_tune = -1
        best_name = ''
        for name in configs:
            total, ps = results[name]
            tune_score = sum(ps.get(s, 0) for s in tune)
            if tune_score > best_tune:
                best_tune = tune_score
                best_name = name
        hold_ex = results[best_name][1].get(hold, 0)
        print(f'    {hold}: tune_best={best_tune}, hold={hold_ex}/{n_te}, config={best_name[:50]}')

    # Actual nested LOSO
    nested = 0
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_names = []
        for name in configs:
            total, ps = results[name]
            tune_score = sum(ps.get(s, 0) for s in tune)
            if tune_score > best_tune:
                best_tune = tune_score
                best_names = [name]
            elif tune_score == best_tune:
                best_names.append(name)

        # Among tied configs, pick simplest (no tail > tail)
        chosen = best_names[0]
        for n in best_names:
            if 'no tail' in n:
                chosen = n
                break

        hold_ex = results[chosen][1].get(hold, 0)
        nested += hold_ex

    print(f'\n  Nested LOSO (simple preference): {nested}/91')

    # Also try: among tied, pick the one with best hold-out
    # (this is cheating, just for comparison)
    nested_oracle = 0
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        for name in configs:
            total, ps = results[name]
            tune_score = sum(ps.get(s, 0) for s in tune)
            if tune_score > best_tune:
                best_tune = tune_score

        # Among tied configs at best_tune, pick best on hold
        best_hold = -1
        for name in configs:
            tune_score = sum(results[name][1].get(s, 0) for s in tune)
            if tune_score == best_tune:
                h = results[name][1].get(hold, 0)
                if h > best_hold:
                    best_hold = h
        nested_oracle += best_hold

    print(f'  Nested LOSO (oracle tie-break): {nested_oracle}/91')

    # ── Show which teams changed ──
    print(f'\n  ── TEAM-LEVEL ANALYSIS: v25 vs v25+tail(61,65)_opp=-3 ──')
    for s in test_seasons:
        sd = season_data[s]
        p_v25 = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                           bot, (50, 60), None, None)
        p_tail = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                           bot, (50, 60), {'opp_rank': -3}, (61, 65))

        changes = []
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi] and p_v25[i] != p_tail[i]:
                gt = int(y[gi])
                rid = record_ids[gi]
                status = 'FIXED' if p_tail[i] == gt else ('BROKEN' if p_v25[i] == gt else 'CHANGED')
                changes.append((rid, gt, p_v25[i], p_tail[i], status))

        if changes:
            print(f'\n  {s}:')
            for rid, gt, old, new, status in changes:
                print(f'    {rid}: GT={gt}, v25={old}, +tail={new} [{status}]')

    # ── Stability test: try many opp_rank weights ──
    print(f'\n  ── STABILITY: tail(61,65) opp_rank weight sweep ──')
    for w in range(-8, 1):
        total = 0
        ps_out = {}
        for s, sd in season_data.items():
            p = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                           bot, (50, 60), {'opp_rank': w}, (61, 65))
            ex = count_exact(p, sd, test_mask, y)
            total += ex
            ps_out[s] = ex
        ps_str = ' '.join(f'{ps_out.get(s,0):2d}' for s in test_seasons)
        marker = ' ★' if total > 68 else ''
        print(f'    opp_rank={w:+d}: {total}/91 [{ps_str}]{marker}')

    # No tail
    total = 0
    ps_out = {}
    for s, sd in season_data.items():
        p = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                       bot, (50, 60), None, None)
        ex = count_exact(p, sd, test_mask, y)
        total += ex
        ps_out[s] = ex
    ps_str = ' '.join(f'{ps_out.get(s,0):2d}' for s in test_seasons)
    print(f'    no tail:    {total}/91 [{ps_str}]')

    # ── Permutation test for tail ──
    print(f'\n  ── PERMUTATION TEST for tail(61,65) opp_rank=-3 ──')
    rng = np.random.RandomState(42)
    true_score = results['v25+tail(61,65)_opp=-3'][0]
    n_perm = 1000
    perm_scores = []

    for _ in range(n_perm):
        total = 0
        for s, sd in season_data.items():
            # Run v25 base
            p = run_pipeline(sd, fn, mid, (17, 34), low, (35, 52),
                           bot, (50, 60), None, None)
            # Apply tail swap with SHUFFLED correction
            lo, hi = 61, 65
            idx = [i for i in range(len(p)) if sd['tm'][i] and lo <= p[i] <= hi]
            if len(idx) > 1:
                seeds = [p[i] for i in idx]
                rng.shuffle(seeds)  # Random assignment
                for k, ii in enumerate(idx):
                    p[ii] = seeds[k]
            ex = count_exact(p, sd, test_mask, y)
            total += ex
        perm_scores.append(total)

    perm_scores = np.array(perm_scores)
    p_value = np.mean(perm_scores >= true_score)
    print(f'    True score: {true_score}/91')
    print(f'    Perm mean:  {perm_scores.mean():.1f}, std: {perm_scores.std():.1f}')
    print(f'    Perm max:   {perm_scores.max()}')
    print(f'    p-value:    {p_value:.4f}')
    if p_value < 0.05:
        print(f'    → SIGNIFICANT (p<0.05)')
    else:
        print(f'    → NOT significant — tail improvement is NOISE')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
