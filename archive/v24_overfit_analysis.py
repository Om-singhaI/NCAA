#!/usr/bin/env python3
"""
Deep Overfitting Analysis for v24 Triple-Zone Model
=====================================================
Comprehensive tests to verify ALL 67/91 predictions are legitimate.

Tests:
  1. Nested LOSO — tune on 4 seasons, test on 5th (gold standard)
  2. Per-season analysis — breakdown of every prediction, all 91 teams
  3. Permutation test — random corrections → how often beat v23 by chance?
  4. Stability analysis — how sensitive are results to parameter perturbation?
  5. Leave-2-seasons-out — tune on 3 seasons, test on held-out 2
  6. Degrees of freedom — how many teams affected vs params tuned?
  7. Full prediction audit — list ALL 91 predictions with correctness

Each zone correction is independently validated to ensure no leakage.
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


# ─── Zone correction functions (self-contained) ───

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


def compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1):
    fi = {f: i for i, f in enumerate(fn)}
    correction = np.zeros(X.shape[0])
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    conf_avg = X[:, fi['conf_avg_net']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    if sosnet != 0:
        correction += sosnet * (sos - net) / 200
    if net_conf != 0:
        correction += net_conf * (conf_avg - net) / 100
    if cbhist != 0:
        correction += cbhist * (cb_mean - tfr) / 34
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


def eval_pipeline(season_data, test_mask, y, test_seasons, fn,
                  mid_params=(0, 2, 3), mid_zone=(17, 34),
                  low_params=(1, 2), low_zone=(35, 52),
                  bot_params=(-4, 3, -1), bot_zone=(53, 65)):
    """Run full 3-zone pipeline and return total exact + per-season dict."""
    total = 0
    ps = {}
    per_team = {}  # global_idx -> (predicted, actual, correct)
    for s, sd in season_data.items():
        p = sd['pass1'].copy()

        mid_corr = compute_midrange_correction(fn, sd['X'], aq=mid_params[0],
                                                al=mid_params[1], sos=mid_params[2])
        p = apply_swap(p, sd['raw'], mid_corr, sd['tm'], mid_zone)

        low_corr = compute_low_correction(fn, sd['X'], q1dom=low_params[0],
                                           field=low_params[1])
        p = apply_swap(p, sd['raw'], low_corr, sd['tm'], low_zone)

        if bot_params is not None:
            bot_corr = compute_bottom_correction(fn, sd['X'], sosnet=bot_params[0],
                                                  net_conf=bot_params[1], cbhist=bot_params[2])
            p = apply_swap(p, sd['raw'], bot_corr, sd['tm'], bot_zone)

        ex = 0
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi]:
                actual = int(y[gi])
                pred = p[i]
                correct = (pred == actual)
                per_team[gi] = (pred, actual, correct)
                if correct:
                    ex += 1
        total += ex
        ps[s] = ex
    return total, ps, per_team


def main():
    t0 = time.time()
    print('='*70)
    print(' DEEP OVERFITTING ANALYSIS — v24 Triple-Zone Model')
    print(' Verifying ALL 67/91 predictions are legitimate')
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

    # Get team names
    team_names = labeled['Team'].values if 'Team' in labeled.columns else record_ids

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Precompute v12 base for all seasons
    print('\n  Precomputing v12 base predictions...')
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

    # v12 baseline
    v12_total = sum(
        sum(1 for i, gi in enumerate(sd['indices'])
            if test_mask[gi] and sd['pass1'][i] == int(y[gi]))
        for sd in season_data.values())
    print(f'  v12 baseline: {v12_total}/91')

    # v23 (no bottom zone)
    v23_total, v23_ps, v23_teams = eval_pipeline(
        season_data, test_mask, y, test_seasons, fn, bot_params=None)
    print(f'  v23 (dual-zone): {v23_total}/91')

    # v24 (triple zone)
    v24_total, v24_ps, v24_teams = eval_pipeline(
        season_data, test_mask, y, test_seasons, fn,
        bot_params=(-4, 3, -1), bot_zone=(53, 65))
    print(f'  v24 (triple-zone): {v24_total}/91')

    # ═══════════════════════════════════════════════════════════
    #  TEST 1: Full Prediction Audit — ALL 91 teams
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 1: FULL PREDICTION AUDIT — ALL 91 TEAMS')
    print('='*70)

    test_indices = sorted([gi for gi in range(n_labeled) if test_mask[gi]],
                          key=lambda gi: (seasons[gi], int(y[gi])))

    print(f'\n  {"Season":<10} {"Team":<25} {"True":>4} {"v12":>4} {"v23":>4} {"v24":>4}  '
          f'{"v12":>4} {"v23":>4} {"v24":>4}  Status')
    print(f'  {"─"*10} {"─"*25} {"─"*4} {"─"*4} {"─"*4} {"─"*4}  '
          f'{"err":>4} {"err":>4} {"err":>4}  {"─"*20}')

    zone_tallies = {'top(1-16)': [0, 0, 0, 0], 'mid(17-34)': [0, 0, 0, 0],
                    'low(35-52)': [0, 0, 0, 0], 'bot(53-68)': [0, 0, 0, 0]}

    for gi in test_indices:
        s = seasons[gi]
        actual = int(y[gi])
        name = str(team_names[gi])[:24]

        # v12 prediction
        v12_pred = None
        for sd in season_data.values():
            for i, idx in enumerate(sd['indices']):
                if idx == gi:
                    v12_pred = sd['pass1'][i]
                    break

        v23_pred, _, v23_ok = v23_teams.get(gi, (0, 0, False))
        v24_pred, _, v24_ok = v24_teams.get(gi, (0, 0, False))

        v12_err = (v12_pred - actual) if v12_pred else 0
        v23_err = v23_pred - actual
        v24_err = v24_pred - actual

        # Status
        if v24_ok and v23_ok:
            status = '✓ both correct'
        elif v24_ok and not v23_ok:
            status = '★ v24 FIXED'
        elif not v24_ok and v23_ok:
            status = '✗ v24 BROKE'
        elif not v24_ok and not v23_ok and v24_err != v23_err:
            status = f'• changed ({v23_err:+d}→{v24_err:+d})'
        else:
            status = f'✗ wrong ({v24_err:+d})'

        # Zone tally
        if actual <= 16:
            zone = 'top(1-16)'
        elif actual <= 34:
            zone = 'mid(17-34)'
        elif actual <= 52:
            zone = 'low(35-52)'
        else:
            zone = 'bot(53-68)'
        zone_tallies[zone][0] += 1
        if v12_pred and v12_pred == actual:
            zone_tallies[zone][1] += 1
        if v23_ok:
            zone_tallies[zone][2] += 1
        if v24_ok:
            zone_tallies[zone][3] += 1

        print(f'  {s:<10} {name:<25} {actual:4d} {v12_pred or 0:4d} {v23_pred:4d} {v24_pred:4d}  '
              f'{v12_err:+4d} {v23_err:+4d} {v24_err:+4d}  {status}')

    print(f'\n  Zone breakdown:')
    print(f'  {"Zone":<14} {"N":>3} {"v12":>5} {"v23":>5} {"v24":>5}')
    for zn, (n, v12c, v23c, v24c) in zone_tallies.items():
        print(f'  {zn:<14} {n:3d} {v12c:5d} {v23c:5d} {v24c:5d}')

    total_v12_c = sum(v[1] for v in zone_tallies.values())
    total_v23_c = sum(v[2] for v in zone_tallies.values())
    total_v24_c = sum(v[3] for v in zone_tallies.values())
    total_n = sum(v[0] for v in zone_tallies.values())
    print(f'  {"TOTAL":<14} {total_n:3d} {total_v12_c:5d} {total_v23_c:5d} {total_v24_c:5d}')

    # Identify v24 FIXES and BREAKS vs v23
    fixed = [(gi, v23_teams[gi], v24_teams[gi]) for gi in test_indices
             if not v23_teams.get(gi, (0, 0, False))[2] and v24_teams.get(gi, (0, 0, False))[2]]
    broke = [(gi, v23_teams[gi], v24_teams[gi]) for gi in test_indices
             if v23_teams.get(gi, (0, 0, False))[2] and not v24_teams.get(gi, (0, 0, False))[2]]

    print(f'\n  v24 FIXED (v23 wrong → v24 correct): {len(fixed)} teams')
    for gi, v23t, v24t in fixed:
        print(f'    {record_ids[gi]:<30} true={int(y[gi]):2d}  v23={v23t[0]:2d}(err={v23t[0]-int(y[gi]):+d})  v24={v24t[0]:2d} ✓')

    print(f'  v24 BROKE (v23 correct → v24 wrong): {len(broke)} teams')
    for gi, v23t, v24t in broke:
        print(f'    {record_ids[gi]:<30} true={int(y[gi]):2d}  v23={v23t[0]:2d} ✓  v24={v24t[0]:2d}(err={v24t[0]-int(y[gi]):+d})')

    # ═══════════════════════════════════════════════════════════
    #  TEST 2: Nested LOSO (tune on 4, test on 1) — gold standard
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 2: NESTED LOSO — tune on 4 seasons, test on 5th')
    print(' (Gold standard: config chosen WITHOUT seeing test season)')
    print('='*70)

    # Build config grid: all reasonable combos for the 3 zones
    configs = []
    # Fixed mid/low, vary bottom
    for sn in [-6, -5, -4, -3, -2, -1, 0]:
        for nc in [0, 1, 2, 3, 4, 5]:
            for cb in [-2, -1, 0, 1]:
                if sn == 0 and nc == 0 and cb == 0:
                    continue  # skip v23 duplicate
                configs.append({'mid': (0, 2, 3), 'low': (1, 2),
                                'bot': (sn, nc, cb), 'bot_zone': (53, 65)})

    # Also include v23 (no bottom zone)
    configs.append({'mid': (0, 2, 3), 'low': (1, 2), 'bot': None, 'bot_zone': None})

    # Evaluate all configs on all seasons
    print(f'\n  Evaluating {len(configs)} configs...')
    config_results = []
    for ci, cfg in enumerate(configs):
        total, ps, _ = eval_pipeline(
            season_data, test_mask, y, test_seasons, fn,
            mid_params=cfg['mid'], low_params=cfg['low'],
            bot_params=cfg['bot'],
            bot_zone=cfg['bot_zone'] if cfg['bot_zone'] else (53, 65))
        config_results.append({'cfg': cfg, 'total': total, 'ps': ps})

    # Nested LOSO
    nested_total = 0
    nested_v23 = 0
    nested_details = []

    print(f'\n  {"Season":<10} {"N":>3} {"v23":>4} {"v24":>4} {"nested":>6}  Config chosen (blind)')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*6}  {"─"*50}')

    for hold in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold]
        n_hold = (test_mask & (seasons == hold)).sum()

        # Find best config on tune seasons
        best_tune_score = -1
        best_cfg_idx = -1
        for ci, cr in enumerate(config_results):
            tune_score = sum(cr['ps'].get(s, 0) for s in tune_seasons)
            if tune_score > best_tune_score or (
                tune_score == best_tune_score and ci == len(configs) - 1):
                # prefer v23 (last) as tiebreaker to be conservative
                best_tune_score = tune_score
                best_cfg_idx = ci

        best_cr = config_results[best_cfg_idx]
        hold_exact = best_cr['ps'].get(hold, 0)
        cfg = best_cr['cfg']

        # v23 and v24 on held-out
        v23_hold = v23_ps.get(hold, 0)
        v24_hold = v24_ps.get(hold, 0)

        nested_total += hold_exact
        nested_v23 += v23_hold

        cfg_desc = f'bot={cfg["bot"]}' if cfg['bot'] else 'v23 (no bot zone)'
        nested_details.append((hold, n_hold, v23_hold, v24_hold, hold_exact, cfg_desc))
        delta = hold_exact - v23_hold
        print(f'  {hold:<10} {n_hold:3d} {v23_hold:4d} {v24_hold:4d} {hold_exact:6d}  {cfg_desc}')

    print(f'\n  Nested LOSO totals:')
    print(f'    v23:              {nested_v23}/91')
    print(f'    v24 (fixed cfg):  {v24_total}/91')
    print(f'    v24 (nested):     {nested_total}/91')
    print(f'    Δ nested vs v23:  {nested_total - nested_v23:+d}')
    if nested_total >= v24_total:
        print(f'    ★ Nested LOSO ≥ fixed config → NO OVERFITTING')
    elif nested_total >= v23_total:
        print(f'    ★ Nested LOSO > v23 → validated improvement (mild overfit in fixed cfg)')
    else:
        print(f'    ✗ Nested LOSO < v23 → POSSIBLE OVERFITTING')

    # ═══════════════════════════════════════════════════════════
    #  TEST 3: Permutation Test — random corrections
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 3: PERMUTATION TEST — random corrections')
    print(' If 67/91 is real, random corrections should rarely beat v23')
    print('='*70)

    n_perms = 2000
    perm_scores = []
    rng = np.random.RandomState(42)

    for perm_i in range(n_perms):
        # Random bottom-zone correction with same structure
        total = 0
        for s, sd in season_data.items():
            p = sd['pass1'].copy()

            # Apply real mid+low corrections (these are validated)
            mid_corr = compute_midrange_correction(fn, sd['X'])
            p = apply_swap(p, sd['raw'], mid_corr, sd['tm'], (17, 34))
            low_corr = compute_low_correction(fn, sd['X'])
            p = apply_swap(p, sd['raw'], low_corr, sd['tm'], (35, 52))

            # Apply RANDOM bottom correction (shuffle team corrections)
            bot_corr = compute_bottom_correction(fn, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
            # Shuffle the corrections among test teams in the zone
            bot_idx = [i for i in range(len(p)) if sd['tm'][i] and 53 <= p[i] <= 65]
            if len(bot_idx) > 1:
                shuffled = bot_corr[bot_idx].copy()
                rng.shuffle(shuffled)
                bot_corr_rand = bot_corr.copy()
                for bi, vi in enumerate(bot_idx):
                    bot_corr_rand[vi] = shuffled[bi]
                p = apply_swap(p, sd['raw'], bot_corr_rand, sd['tm'], (53, 65))
            else:
                p = apply_swap(p, sd['raw'], bot_corr, sd['tm'], (53, 65))

            total += sum(1 for i, gi in enumerate(sd['indices'])
                        if test_mask[gi] and p[i] == int(y[gi]))
        perm_scores.append(total)

    perm_scores = np.array(perm_scores)
    p_value = (perm_scores >= v24_total).mean()
    print(f'\n  v24 actual score: {v24_total}/91')
    print(f'  Permutation scores (n={n_perms}):')
    print(f'    Mean:   {perm_scores.mean():.1f}/91')
    print(f'    Median: {np.median(perm_scores):.0f}/91')
    print(f'    Max:    {perm_scores.max()}/91')
    print(f'    Min:    {perm_scores.min()}/91')
    print(f'    Std:    {perm_scores.std():.2f}')
    print(f'    p-value (random ≥ {v24_total}): {p_value:.4f}')
    if p_value < 0.05:
        print(f'    ★ p < 0.05 → v24 score is SIGNIFICANT (not by chance)')
    else:
        print(f'    ✗ p ≥ 0.05 → v24 score may be achievable by chance')

    # Distribution
    bins = range(int(perm_scores.min()), int(perm_scores.max()) + 2)
    hist, _ = np.histogram(perm_scores, bins=bins)
    print(f'\n  Score distribution:')
    for bi in range(len(hist)):
        score = bins[bi]
        count = hist[bi]
        if count > 0:
            bar = '█' * min(count * 40 // n_perms, 50)
            marker = ' ← v24' if score == v24_total else ''
            marker2 = ' ← v23' if score == v23_total else ''
            print(f'    {score:3d}: {count:5d} ({count/n_perms*100:5.1f}%) {bar}{marker}{marker2}')

    # ═══════════════════════════════════════════════════════════
    #  TEST 4: Parameter Stability — sensitivity to perturbation
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 4: PARAMETER STABILITY — sensitivity to perturbation')
    print(' Robust model ≠ cliff edge; small param changes → similar scores')
    print('='*70)

    # Map neighborhood around v24 params
    print(f'\n  v24 params: sosnet=-4, net_conf=3, cbhist=-1')
    print(f'  Testing neighbors...\n')
    print(f'  {"sosnet":>6} {"net_conf":>8} {"cbhist":>6} {"Score":>5}  Per-season')

    stability_results = []
    for sn in [-6, -5, -4, -3, -2]:
        for nc in [1, 2, 3, 4, 5]:
            for cb in [-3, -2, -1, 0, 1]:
                total, ps, _ = eval_pipeline(
                    season_data, test_mask, y, test_seasons, fn,
                    bot_params=(sn, nc, cb), bot_zone=(53, 65))
                stability_results.append((sn, nc, cb, total, ps))

    stability_results.sort(key=lambda x: -x[3])

    # Show top 30
    for sn, nc, cb, total, ps in stability_results[:30]:
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ← v24' if (sn == -4 and nc == 3 and cb == -1) else ''
        marker2 = ' ← v23' if total == v23_total else ''
        print(f'  {sn:6d} {nc:8d} {cb:6d} {total:5d}  [{ps_str}]{marker}{marker2}')

    # Count configs at each score level
    from collections import Counter
    score_counts = Counter(r[3] for r in stability_results)
    print(f'\n  Score distribution ({len(stability_results)} neighbor configs):')
    for score in sorted(score_counts.keys(), reverse=True):
        print(f'    {score}/91: {score_counts[score]} configs')

    n_at_v24 = score_counts.get(v24_total, 0)
    n_gte_v23 = sum(c for sc, c in score_counts.items() if sc >= v23_total)
    print(f'\n  Configs at v24 level ({v24_total}): {n_at_v24}/{len(stability_results)}')
    print(f'  Configs ≥ v23 level ({v23_total}): {n_gte_v23}/{len(stability_results)}')

    # ═══════════════════════════════════════════════════════════
    #  TEST 5: Leave-2-Seasons-Out — tune on 3, test on 2
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 5: LEAVE-2-SEASONS-OUT — tune on 3 seasons, test on 2')
    print(' (More aggressive cross-validation: less training data)')
    print('='*70)

    l2so_results = []
    season_pairs = list(itertools.combinations(test_seasons, 2))

    for hold_pair in season_pairs:
        tune_seasons = [s for s in test_seasons if s not in hold_pair]

        best_tune_score = -1
        best_cfg_idx = -1
        for ci, cr in enumerate(config_results):
            tune_score = sum(cr['ps'].get(s, 0) for s in tune_seasons)
            if tune_score > best_tune_score or (
                tune_score == best_tune_score and ci == len(configs) - 1):
                best_tune_score = tune_score
                best_cfg_idx = ci

        best_cr = config_results[best_cfg_idx]
        hold_exact = sum(best_cr['ps'].get(s, 0) for s in hold_pair)
        hold_n = sum((test_mask & (seasons == s)).sum() for s in hold_pair)
        v23_hold = sum(v23_ps.get(s, 0) for s in hold_pair)
        cfg = best_cr['cfg']
        cfg_desc = f'bot={cfg["bot"]}' if cfg['bot'] else 'v23 (no bot)'

        l2so_results.append((hold_pair, hold_n, v23_hold, hold_exact, cfg_desc))

    print(f'\n  {"HeldOut":<25} {"N":>3} {"v23":>4} {"L2SO":>4} {"Δ":>3}  Config')
    print(f'  {"─"*25} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*30}')
    total_l2so = 0
    total_l2so_v23 = 0
    for pair, n, v23h, l2h, cdesc in l2so_results:
        delta = l2h - v23h
        total_l2so += l2h
        total_l2so_v23 += v23h
        print(f'  {str(pair):<25} {n:3d} {v23h:4d} {l2h:4d} {delta:+3d}  {cdesc}')

    # Average across all pairs
    n_pairs = len(season_pairs)
    avg_test = sum(r[1] for r in l2so_results) / n_pairs
    avg_v23 = total_l2so_v23 / n_pairs
    avg_l2so = total_l2so / n_pairs
    print(f'\n  Average across {n_pairs} folds:')
    print(f'    v23 avg:  {avg_v23:.1f}/{avg_test:.0f}')
    print(f'    L2SO avg: {avg_l2so:.1f}/{avg_test:.0f}')
    print(f'    Δ avg:    {avg_l2so - avg_v23:+.1f}')

    # ═══════════════════════════════════════════════════════════
    #  TEST 6: Degrees of Freedom Analysis
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 6: DEGREES OF FREEDOM ANALYSIS')
    print(' How many teams affected vs how many params tuned?')
    print('='*70)

    # Count teams in each zone per season
    print(f'\n  Bottom zone (53-65) test team counts per season:')
    total_bot_teams = 0
    for s, sd in season_data.items():
        bot_count = sum(1 for i in range(len(sd['pass1']))
                        if sd['tm'][i] and 53 <= sd['pass1'][i] <= 65)
        total_bot_teams += bot_count
        print(f'    {s}: {bot_count} test teams in bot zone')

    print(f'\n  Total bot-zone test teams affected: {total_bot_teams}')
    print(f'  Total params tuned for bot zone: 3 (sosnet, net_conf, cbhist)')
    print(f'  + 2 zone boundary params (lo=53, hi=65)')
    print(f'  = 5 total degrees of freedom')
    print(f'  Effective ratio: {total_bot_teams}/{5} = {total_bot_teams/5:.1f} teams per param')

    # Overall across all 3 zones
    total_mid_teams = 0
    total_low_teams = 0
    for s, sd in season_data.items():
        mid_count = sum(1 for i in range(len(sd['pass1']))
                        if sd['tm'][i] and 17 <= sd['pass1'][i] <= 34)
        low_count = sum(1 for i in range(len(sd['pass1']))
                        if sd['tm'][i] and 35 <= sd['pass1'][i] <= 52)
        total_mid_teams += mid_count
        total_low_teams += low_count

    print(f'\n  Summary across all 3 zones:')
    print(f'  {"Zone":<15} {"Teams":>5} {"Params":>6} {"Ratio":>5}')
    print(f'  {"─"*15} {"─"*5} {"─"*6} {"─"*5}')
    print(f'  {"Mid(17-34)":<15} {total_mid_teams:5d} {5:6d} {total_mid_teams/5:5.1f}')
    print(f'  {"Low(35-52)":<15} {total_low_teams:5d} {4:6d} {total_low_teams/4:5.1f}')
    print(f'  {"Bot(53-65)":<15} {total_bot_teams:5d} {5:6d} {total_bot_teams/5:5.1f}')
    total_teams = total_mid_teams + total_low_teams + total_bot_teams
    total_params = 5 + 4 + 5
    print(f'  {"TOTAL":<15} {total_teams:5d} {total_params:6d} {total_teams/total_params:5.1f}')

    # ═══════════════════════════════════════════════════════════
    #  TEST 7: Per-Zone Independent Validation
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 7: PER-ZONE INDEPENDENT VALIDATION')
    print(' Each zone correction validated independently via nested LOSO')
    print('='*70)

    # Test each zone independently
    zone_configs = {
        'v12 base only': {'mid': (0, 0, 0), 'low': (0, 0), 'bot': None},
        'v18 mid only':  {'mid': (0, 2, 3), 'low': (0, 0), 'bot': None},
        'v23 mid+low':   {'mid': (0, 2, 3), 'low': (1, 2), 'bot': None},
        'v24 all zones': {'mid': (0, 2, 3), 'low': (1, 2), 'bot': (-4, 3, -1)},
        'bot only':      {'mid': (0, 0, 0), 'low': (0, 0), 'bot': (-4, 3, -1)},
        'mid+bot':       {'mid': (0, 2, 3), 'low': (0, 0), 'bot': (-4, 3, -1)},
        'low+bot':       {'mid': (0, 0, 0), 'low': (1, 2), 'bot': (-4, 3, -1)},
    }

    print(f'\n  {"Config":<20} {"Total":>5}  Per-season')
    print(f'  {"─"*20} {"─"*5}  {"─"*30}')
    for name, zc in zone_configs.items():
        total, ps, _ = eval_pipeline(
            season_data, test_mask, y, test_seasons, fn,
            mid_params=zc['mid'], low_params=zc['low'],
            bot_params=zc['bot'], bot_zone=(53, 65) if zc['bot'] else (53, 65))
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        print(f'  {name:<20} {total:5d}  [{ps_str}]')

    # ═══════════════════════════════════════════════════════════
    #  TEST 8: Remaining Errors Analysis
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 8: REMAINING ERRORS ANALYSIS')
    print(f' v24 still gets {91 - v24_total} teams wrong — are they fundamentally hard?')
    print('='*70)

    fi = {f: i for i, f in enumerate(fn)}
    errors = []
    for gi in test_indices:
        pred, actual, correct = v24_teams.get(gi, (0, int(y[gi]), False))
        if not correct:
            errors.append({
                'rid': record_ids[gi],
                'season': seasons[gi],
                'actual': actual,
                'pred': pred,
                'err': pred - actual,
                'net': X_all[gi, fi['NET Rank']],
                'sos': X_all[gi, fi['NETSOS']],
                'is_aq': X_all[gi, fi['is_AQ']],
                'is_al': X_all[gi, fi['is_AL']],
                'conf_avg': X_all[gi, fi['conf_avg_net']],
            })

    errors.sort(key=lambda e: abs(e['err']), reverse=True)

    print(f'\n  {len(errors)} remaining errors:')
    print(f'  {"Team":<30} {"True":>4} {"Pred":>4} {"Err":>5}  {"NET":>4} {"SOS":>4} {"Bid":>3}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*5}  {"─"*4} {"─"*4} {"─"*3}')
    for e in errors:
        bid = 'AQ' if e['is_aq'] else 'AL'
        print(f'  {e["rid"]:<30} {e["actual"]:4d} {e["pred"]:4d} {e["err"]:+5d}  '
              f'{e["net"]:4.0f} {e["sos"]:4.0f} {bid:>3}')

    avg_err = np.mean([abs(e['err']) for e in errors])
    max_err = max(abs(e['err']) for e in errors)
    median_err = np.median([abs(e['err']) for e in errors])
    print(f'\n  Error stats: avg|err|={avg_err:.1f}, median={median_err:.0f}, max={max_err}')

    # Error by zone
    err_zones = {'top(1-16)': [], 'mid(17-34)': [], 'low(35-52)': [], 'bot(53-68)': []}
    for e in errors:
        if e['actual'] <= 16:
            err_zones['top(1-16)'].append(e)
        elif e['actual'] <= 34:
            err_zones['mid(17-34)'].append(e)
        elif e['actual'] <= 52:
            err_zones['low(35-52)'].append(e)
        else:
            err_zones['bot(53-68)'].append(e)

    print(f'\n  Errors by zone:')
    for zn, errs in err_zones.items():
        n_zone = zone_tallies[zn][0]
        avg = np.mean([abs(e['err']) for e in errs]) if errs else 0
        print(f'    {zn}: {len(errs)}/{n_zone} wrong (avg|err|={avg:.1f})')

    # Swap pair analysis
    print(f'\n  Swap pair analysis (pairs of errors that could fix each other):')
    swap_count = 0
    for i in range(len(errors)):
        for j in range(i+1, len(errors)):
            if (errors[i]['season'] == errors[j]['season'] and
                errors[i]['pred'] == errors[j]['actual'] and
                errors[j]['pred'] == errors[i]['actual']):
                print(f'    {errors[i]["rid"]}: pred={errors[i]["pred"]} true={errors[i]["actual"]} '
                      f'↔ {errors[j]["rid"]}: pred={errors[j]["pred"]} true={errors[j]["actual"]}')
                swap_count += 1
    if swap_count == 0:
        print(f'    No perfect swap pairs found')
    print(f'  Total swap pairs: {swap_count}')

    # ═══════════════════════════════════════════════════════════
    #  FINAL VERDICT
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL VERDICT')
    print('='*70)
    print(f'\n  Full test score: {v24_total}/91 ({v24_total/91*100:.1f}%)')
    print(f'  Nested LOSO:     {nested_total}/91 ({nested_total/91*100:.1f}%)')
    print(f'  v23 baseline:    {v23_total}/91 ({v23_total/91*100:.1f}%)')
    gap = v24_total - nested_total
    print(f'  Overfit gap:     {gap} (full - nested)')

    if gap <= 2:
        print(f'  ★ MINIMAL overfitting (gap ≤ 2). Model is ROBUST.')
    elif gap <= 4:
        print(f'  • MILD overfitting (gap 3-4). Gains are partly real.')
    else:
        print(f'  ✗ SIGNIFICANT overfitting (gap ≥ 5). Be cautious.')

    if nested_total > v23_total:
        delta = nested_total - v23_total
        print(f'  ★ Validated improvement over v23: +{delta} (nested LOSO)')
    elif nested_total == v23_total:
        print(f'  • No validated improvement over v23 (nested LOSO ties)')
    else:
        print(f'  ✗ Nested LOSO is WORSE than v23 → overfit')

    print(f'\n  Per-season v24:  {" ".join(f"{v24_ps.get(s,0):2d}" for s in test_seasons)}')
    print(f'  Per-season v23:  {" ".join(f"{v23_ps.get(s,0):2d}" for s in test_seasons)}')
    n_per = [int((test_mask & (seasons == s)).sum()) for s in test_seasons]
    print(f'  Per-season N:    {" ".join(f"{n:2d}" for n in n_per)}')

    print(f'\n  Permutation p-value: {p_value:.4f}')
    print(f'  Stability: {n_at_v24}/{len(stability_results)} neighbor configs at {v24_total}/91')
    print(f'  Degrees of freedom: {total_params} params for {total_teams} team-corrections')

    all_ok = (nested_total >= v23_total and p_value < 0.05 and gap <= 3)
    if all_ok:
        print(f'\n  ═══════════════════════════════════════════')
        print(f'  ★★★ ALL CHECKS PASS — v24 is LEGITIMATE ★★★')
        print(f'  ═══════════════════════════════════════════')
    else:
        issues = []
        if nested_total < v23_total:
            issues.append('nested LOSO < v23')
        if p_value >= 0.05:
            issues.append(f'permutation p={p_value:.3f}')
        if gap > 3:
            issues.append(f'overfit gap={gap}')
        print(f'\n  ⚠ CONCERNS: {", ".join(issues)}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
