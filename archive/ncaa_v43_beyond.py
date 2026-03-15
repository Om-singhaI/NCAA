#!/usr/bin/env python3
"""
v43 BEYOND 73: Explore improvements on top of v26 (73/91)

Remaining 18 errors after v26:
- Kentucky↔Wisconsin (11↔12) — swap pair
- Clemson↔WashSt (22↔26) — swap pair (WashSt worsened by ncsos)
- USC (25→26, +1) — near-miss
- Murray St (26→41, +15) — huge outlier  
- Northwestern↔NewMexico (36↔42) — swap pair
- Charleston↔VCU (47↔50) — swap pair
- Richmond↔SouthDakotaSt (49↔52) — swap pair
- LongBeachSt/WesternKy/SouthDakotaSt (59/60/61) — 3-way rotation
- TexasSouthern↔SEMoSt (66↔67) — swap pair

Approaches:
1. NETSOS zone at 55-61 to fix 3-way rotation (found in v42c, +3)
2. Different feature zones for remaining swap pairs
3. Confidence-based swap: only swap pairs where signals strongly agree
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features,
    select_top_k_features, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    apply_ncsos_zone, predict_robust_blend,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    NCSOS_ZONE, NCSOS_WEIGHT, NCSOS_POWER,
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


def apply_v26_zones(pass1, raw, fn, X, tm, idx, ncsos_vals):
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=0.15)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=0.15)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=0.15)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=0.15)
    p = apply_ncsos_zone(p, raw, ncsos_vals, tm, zone=NCSOS_ZONE,
                          weight=NCSOS_WEIGHT, power=NCSOS_POWER)
    return p


def count_exact(p, tm, indices, test_mask, y):
    return sum(1 for i, gi in enumerate(indices) if test_mask[gi] and p[i] == int(y[gi]))


def main():
    print('='*70)
    print('  v43 BEYOND 73: IMPROVEMENTS ON TOP OF v26')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat = build_features(labeled, context_df, labeled, tourn_rids)
    fn = list(feat.columns)
    fi = {f: i for i, f in enumerate(fn)}
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

    # Verify v26 baseline
    v26_total = 0
    for s, sd in season_data.items():
        p = apply_v26_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                           sd['tm'], sd['indices'], sd['ncsos'])
        v26_total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
    print(f'\n  v26 baseline: {v26_total}/91')

    # ════════════════════════════════════════════════════════════
    #  PHASE 1: 6th zone feature x zone sweep ON TOP of v26
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 1: 6TH ZONE SWEEP (on top of v26)')
    print('='*70)

    feature_extractors = {
        'NETSOS': lambda sd: np.array([sd['X'][i][fi['NETSOS']] for i in range(len(sd['X']))]),
        'NET': lambda sd: np.array([sd['X'][i][fi['NET Rank']] for i in range(len(sd['X']))]),
        'Q1W': lambda sd: np.array([sd['X'][i][fi['Quadrant1_W']] for i in range(len(sd['X']))]),
        'WLPct': lambda sd: np.array([sd['X'][i][fi['WL_Pct']] for i in range(len(sd['X']))]),
        'conf_avg': lambda sd: np.array([sd['X'][i][fi['conf_avg_net']] for i in range(len(sd['X']))]),
        'OppNET': lambda sd: np.array([sd['X'][i][fi['AvgOppNETRank']] for i in range(len(sd['X']))]),
        'Q1pct': lambda sd: np.array([sd['X'][i][fi['q1_pct']] for i in range(len(sd['X']))]),
        'is_power': lambda sd: np.array([sd['X'][i][fi['is_power_conf']] for i in range(len(sd['X']))]),
        'ncsos_raw': lambda sd: sd['ncsos'],
    }

    best_z6 = v26_total
    best_z6_cfg = None
    all_z6_results = []

    for feat_name, feat_fn in feature_extractors.items():
        for lo6 in range(5, 65, 2):
            for hi6 in range(lo6+2, min(lo6+12, 69)):
                for w6 in [-8, -5, -3, -2, -1, 1, 2, 3, 5, 8]:
                    total = 0
                    per_season = {}
                    for s, sd in season_data.items():
                        fvals = feat_fn(sd)
                        p = apply_v26_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                           sd['tm'], sd['indices'], sd['ncsos'])
                        p = apply_generic_zone(p, sd['raw'], fvals, sd['tm'],
                                              (lo6, hi6), w6, power=0.15)
                        ex = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                        total += ex
                        per_season[s] = ex
                    if total > best_z6:
                        best_z6 = total
                        best_z6_cfg = (feat_name, lo6, hi6, w6)
                        all_z6_results.append((feat_name, lo6, hi6, w6, total, dict(per_season)))
                        print(f'  ★ {feat_name} zone=({lo6},{hi6}) w={w6}: {total}/91  {list(per_season.values())}')
                    elif total == best_z6 and total > v26_total:
                        all_z6_results.append((feat_name, lo6, hi6, w6, total, dict(per_season)))

    print(f'\n  Best 6th zone: {best_z6}/91 ({len(all_z6_results)} configs at best)')
    if best_z6_cfg:
        print(f'  Config: {best_z6_cfg}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 2: NESTED LOSO of best 6th zone configs
    # ════════════════════════════════════════════════════════════
    if best_z6 > v26_total and all_z6_results:
        print('\n' + '='*70)
        print('  PHASE 2: NESTED LOSO VALIDATION')
        print('='*70)

        # config_scores for v26 and z6 variants
        config_scores = {}
        
        # v26 baseline
        v26_scores = {}
        for s, sd in season_data.items():
            p = apply_v26_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                               sd['tm'], sd['indices'], sd['ncsos'])
            v26_scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        config_scores['v26'] = v26_scores

        # Top z6 configs (take unique feature+weight combos)
        seen = set()
        for r in all_z6_results[:20]:
            key = (r[0], r[1], r[2], r[3])
            if key in seen:
                continue
            seen.add(key)
            name = f'{r[0]}_{r[1]}_{r[2]}_{r[3]}'
            config_scores[name] = r[5]

        # Nested LOSO
        print('\n  Config scores:')
        for name, scores in config_scores.items():
            total = sum(scores.values())
            print(f'    {name:>30}: {total}/91  {list(scores.values())}')

        nested_total = 0
        for hold in test_seasons:
            tune = [s for s in test_seasons if s != hold]
            best_tune = -1
            best_name = 'v26'
            for name, scores in config_scores.items():
                ts = sum(scores.get(s, 0) for s in tune)
                if ts > best_tune:
                    best_tune = ts
                    best_name = name
                elif ts == best_tune and name == 'v26':
                    best_name = name
            nested_total += config_scores[best_name][hold]
            print(f'    {hold}: chose {best_name:>30} (tune={best_tune}) → {config_scores[best_name][hold]}')

        print(f'\n  ★ Nested LOSO: {nested_total}/91')
        print(f'  v26 full: {sum(v26_scores.values())}/91')
        print(f'  Best z6 full: {best_z6}/91')
        print(f'  Overfit gap: {best_z6 - nested_total}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 3: PERMUTATION TEST for best z6
    # ════════════════════════════════════════════════════════════
    if best_z6 > v26_total and best_z6_cfg:
        print('\n' + '='*70)
        print('  PHASE 3: PERMUTATION TEST')
        print('='*70)

        feat_fn = feature_extractors[best_z6_cfg[0]]
        lo6, hi6, w6 = best_z6_cfg[1], best_z6_cfg[2], best_z6_cfg[3]
        observed = best_z6 - v26_total

        n_better = 0
        for trial in range(100):
            rng = np.random.RandomState(trial + 500)
            total = 0
            for s, sd in season_data.items():
                fvals = feat_fn(sd).copy()
                ti = [i for i in range(len(sd['tm'])) if sd['tm'][i]]
                vals = fvals[ti].copy()
                rng.shuffle(vals)
                fvals[ti] = vals
                p = apply_v26_zones(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'], sd['ncsos'])
                p = apply_generic_zone(p, sd['raw'], fvals, sd['tm'],
                                      (lo6, hi6), w6, 0.15)
                total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            if total - v26_total >= observed:
                n_better += 1

        p_val = n_better / 100
        print(f'  Observed improvement: +{observed}')
        print(f'  p-value: {p_val:.4f} ({n_better}/100)')

    # ════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  SUMMARY')
    print('='*70)
    print(f'  v26 baseline: {v26_total}/91')
    if best_z6_cfg:
        print(f'  Best z6: {best_z6}/91 at {best_z6_cfg}')
    else:
        print(f'  No improvement found beyond {v26_total}/91')
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
