#!/usr/bin/env python3
"""
v51 SWAP ROBUSTNESS ANALYSIS
=============================
The swap can't be replaced by features/zones. Can we make it MORE robust?

Key questions:
1. Does the AQ↔AL bias pattern exist in TRAINING data too? (validates pattern)
2. How sensitive is the swap to its threshold parameters?
3. Can we verify it doesn't hurt in non-firing seasons?
4. Can we make it fire in more seasons (broader evidence base)?
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
np.random.seed(42)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    apply_aq_al_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)


def main():
    t0 = time.time()
    print('='*72)
    print(' v51 SWAP ROBUSTNESS ANALYSIS')
    print(' Can we make the AQ↔AL swap principled/robust?')
    print('='*72)

    # Load data
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    feat = build_features(labeled, context_df, labeled, set(labeled['RecordID'].values))
    fn = list(feat.columns)
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    rids = labeled['RecordID'].values.astype(str)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in rids])
    fi = {f: i for i, f in enumerate(fn)}

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    X_all = KNNImputer(n_neighbors=10, weights='distance').fit_transform(X_raw)

    folds = sorted(set(seasons))

    # ════════════════════════════════════════════════════════════════════
    # Q1: Does the AQ↔AL bias pattern exist in ALL teams (not just test)?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q1: AQ↔AL BIAS PATTERN — Does it exist across ALL data?')
    print('='*72)

    print(f'\n  Checking EVERY team (test + training) for AQ↔AL bias...')
    print(f'  Pattern: AQ teams where actual_seed - NET > 10')
    print(f'           (committee seeds AQ team WORSE than NET suggests)')
    print(f'  {"Season":<10} {"Team":<25} {"AQ/AL":<6} {"Seed":>5} {"NET":>5} {"Gap":>5} {"Test?"}')
    print(f'  {"─"*10} {"─"*25} {"─"*6} {"─"*5} {"─"*5} {"─"*5} {"─"*5}')

    aq_bias_count = 0
    al_bias_count = 0
    aq_bias_by_season = {}
    al_bias_by_season = {}

    for i in range(len(y)):
        seed = int(y[i])
        net = X_all[i, fi['NET Rank']]
        is_aq = int(X_all[i, fi['is_AQ']])
        is_al = int(X_all[i, fi['is_AL']])
        ssn = seasons[i]
        rid = rids[i]
        is_test = test_mask[i]

        # AQ teams under-seeded: actual seed WORSE than NET
        if is_aq and not is_al and seed - net > 8 and 20 <= seed <= 55:
            team = rid.split('-')[-1] if '-' in rid else rid
            print(f'  {ssn:<10} {team:<25} {"AQ":<6} {seed:5d} {net:5.0f} {seed-net:+5.0f} {"TEST" if is_test else "train"}')
            aq_bias_count += 1
            aq_bias_by_season[ssn] = aq_bias_by_season.get(ssn, 0) + 1

    print(f'\n  AQ under-seeded (gap>8, seeds 20-55): {aq_bias_count} teams across {len(aq_bias_by_season)} seasons')
    for s in sorted(aq_bias_by_season):
        print(f'    {s}: {aq_bias_by_season[s]} teams')

    print(f'\n  Now checking AL teams OVER-seeded: actual seed BETTER than NET')
    print(f'  Pattern: AL teams where NET - actual_seed > 5')
    print(f'  {"Season":<10} {"Team":<25} {"AQ/AL":<6} {"Seed":>5} {"NET":>5} {"Gap":>5} {"Test?"}')
    print(f'  {"─"*10} {"─"*25} {"─"*6} {"─"*5} {"─"*5} {"─"*5} {"─"*5}')

    for i in range(len(y)):
        seed = int(y[i])
        net = X_all[i, fi['NET Rank']]
        is_aq = int(X_all[i, fi['is_AQ']])
        is_al = int(X_all[i, fi['is_AL']])
        ssn = seasons[i]
        rid = rids[i]
        is_test = test_mask[i]

        # AL teams over-seeded: actual seed BETTER than NET
        if is_al and not is_aq and net - seed > 5 and 20 <= seed <= 55:
            team = rid.split('-')[-1] if '-' in rid else rid
            print(f'  {ssn:<10} {team:<25} {"AL":<6} {seed:5d} {net:5.0f} {net-seed:+5.0f} {"TEST" if is_test else "train"}')
            al_bias_count += 1
            al_bias_by_season[ssn] = al_bias_by_season.get(ssn, 0) + 1

    print(f'\n  AL over-seeded (gap>5, seeds 20-55): {al_bias_count} teams across {len(al_bias_by_season)} seasons')
    for s in sorted(al_bias_by_season):
        print(f'    {s}: {al_bias_by_season[s]} teams')

    # ════════════════════════════════════════════════════════════════════
    # Q2: What does the swap actually do — PRECISELY?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q2: WHAT DOES THE SWAP ACTUALLY FIX?')
    print(' Analyzing the specific teams affected')
    print('='*72)

    # First build predictions WITHOUT swap (from the v50 pipeline minus swap)
    from ncaa_v51_principled_fast import cache_v12_raw, run_with_comm_features, se_exact, per_season

    V12_ZONES_V50 = [
        ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
        ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
        ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
        ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
        ('tail',    'tail', (60,63), dict(opp=1)),
        ('xtail',   'bot',  (63,68), dict(sn=1, nc=-1, cb=-1)),
    ]
    COMM_ZONES_V50 = [
        ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
        ('uppermid','comm', (34,44), dict(aq=-6, al=1, sos=-6)),
        ('midbot2', 'bot',  (42,50), dict(sn=-4, nc=2, cb=-3)),
        ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
        ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
        ('tail',    'tail', (60,63), dict(opp=1)),
    ]

    X_min8 = build_min8_features(X_all, fn)
    cache = cache_v12_raw(X_all, y, fn, seasons, test_mask)

    p_no_swap = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                        V12_ZONES_V50, COMM_ZONES_V50, blend=0.25, do_swap=False)
    p_with_swap = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                          V12_ZONES_V50, COMM_ZONES_V50, blend=0.25, do_swap=True)

    print(f'\n  Teams where swap changes predictions:')
    print(f'  {"RID":<30} {"Ssn":<10} {"GT":>3} {"NoSwap":>7} {"Swap":>5} {"AQ":>3} {"AL":>3} {"NET":>5}')
    print(f'  {"─"*30} {"─"*10} {"─"*3} {"─"*7} {"─"*5} {"─"*3} {"─"*3} {"─"*5}')

    for i in range(len(y)):
        if test_mask[i] and p_no_swap[i] != p_with_swap[i]:
            rid = rids[i]
            ssn = seasons[i]
            gt = int(y[i])
            ns = p_no_swap[i]
            ws = p_with_swap[i]
            is_aq = int(X_all[i, fi['is_AQ']])
            is_al = int(X_all[i, fi['is_AL']])
            net = X_all[i, fi['NET Rank']]
            team = rid.split('-')[-1] if '-' in rid else rid
            correct = '✓' if ws == gt else ('~' if abs(ws-gt) <= 1 else '✗')
            print(f'  {team:<30} {ssn:<10} {gt:3d} {ns:7d} {ws:5d} {is_aq:3d} {is_al:3d} {net:5.0f} {correct}')

    # ════════════════════════════════════════════════════════════════════
    # Q3: Swap parameter sensitivity (on v50 full pipeline)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q3: SWAP PARAMETER SENSITIVITY')
    print(' How robust is it to threshold changes?')
    print('='*72)

    no_swap_se, no_swap_ex = se_exact(p_no_swap, y, test_mask)
    print(f'\n  Baseline (no swap): SE={no_swap_se}, {no_swap_ex}/91')
    print(f'\n  {"net_gap":>8} {"pred_gap":>9} {"zone":>10} {"SE":>5} {"Exact":>6} {"Fires":>6}')
    print(f'  {"─"*8} {"─"*9} {"─"*10} {"─"*5} {"─"*6} {"─"*6}')

    for net_gap in [5, 8, 10, 12, 15, 20]:
        for pred_gap in [3, 4, 6, 8, 10]:
            for zone_lo, zone_hi in [(25,50), (30,45), (30,50), (35,45)]:
                p = apply_aq_al_swap(p_no_swap.copy(), X_all, fn, seasons, test_mask,
                                      net_gap=net_gap, pred_gap=pred_gap,
                                      swap_zone=(zone_lo, zone_hi))
                se, ex = se_exact(p, y, test_mask)
                # Count fires
                fires = sum(1 for i in range(len(y)) if test_mask[i] and p[i] != p_no_swap[i])
                if se != no_swap_se:  # only show configs that fire
                    marker = ' ★' if se <= 14 else ''
                    print(f'  {net_gap:8d} {pred_gap:9d} ({zone_lo},{zone_hi}){"":<4} {se:5d} {ex:>3}/91 {fires:>5d}{marker}')

    # ════════════════════════════════════════════════════════════════════
    # Q4: Does the pattern exist in TRAINING (non-test) teams?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q4: PATTERN VALIDATION IN TRAINING DATA')
    print(' Do TRAINING teams show the same AQ↔AL misalignment?')
    print('='*72)

    # For training teams we know the TRUE seed. Check: do AQ training teams
    # have pred-NET gaps similar to the test teams where the swap fires?

    # We need to check model predictions vs NET for training teams too
    print(f'\n  Training teams in seeds 30-45 with AQ/AL pattern:')
    print(f'  {"Ssn":<10} {"Team":<25} {"Status":<6} {"Seed":>5} {"NET":>5} {"AQ":>3} {"AL":>3} {"Seed-NET":>9}')
    print(f'  {"─"*10} {"─"*25} {"─"*6} {"─"*5} {"─"*5} {"─"*3} {"─"*3} {"─"*9}')

    for ssn in folds:
        sm = (seasons == ssn)
        for i in np.where(sm)[0]:
            seed = int(y[i])
            if 30 <= seed <= 45:
                net = X_all[i, fi['NET Rank']]
                is_aq = int(X_all[i, fi['is_AQ']])
                is_al = int(X_all[i, fi['is_AL']])
                rid = rids[i]
                team = rid.split('-')[-1] if '-' in rid else rid
                status = "TEST" if test_mask[i] else "train"
                gap = seed - net
                if abs(gap) >= 5:
                    print(f'  {ssn:<10} {team:<25} {status:<6} {seed:5d} {net:5.0f} {is_aq:3d} {is_al:3d} {gap:+9.0f}')

    # ════════════════════════════════════════════════════════════════════
    # Q5: Can we make the swap fire from LEARNED signals?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q5: LEARNING A SWAP CLASSIFIER')
    print(' Can we predict when a swap should fire?')
    print('='*72)

    # Create training data: for ALL teams in seeds 30-45, compute features
    # that indicate "this team is being mis-seeded"
    # Label: 1 if swap would improve prediction, 0 otherwise
    # Then train a classifier on training data, apply to test data

    # Actually a simpler approach: compute a "should_swap_score" for each team
    # based on how well its features predict its actual seed residual

    # For each team in 30-45 range, compute:
    # residual = actual_seed - predicted_seed (from model without swap)
    # If AQ and residual > 0 (model under-seeds AQ), flag for potential swap up
    # If AL and residual < 0 (model over-seeds AL), flag for potential swap down

    # This is a well-defined signal across ALL seasons

    print(f'\n  Seed residuals for teams in seeds 30-45:')
    print(f'  (residual > 0 = model predicts too low = under-predicted)')
    print(f'  {"Ssn":<10} {"Team":<25} {"GT":>3} {"Pred":>5} {"Resid":>6} {"AQ":>3} {"AL":>3} {"NET":>5}')
    print(f'  {"─"*10} {"─"*25} {"─"*3} {"─"*5} {"─"*6} {"─"*3} {"─"*3} {"─"*5}')

    for ssn in folds:
        sm = (seasons == ssn)
        for i in np.where(sm)[0]:
            if not test_mask[i]:
                continue
            seed = int(y[i])
            pred = int(p_no_swap[i])
            if 30 <= seed <= 50 or 30 <= pred <= 50:
                net = X_all[i, fi['NET Rank']]
                is_aq = int(X_all[i, fi['is_AQ']])
                is_al = int(X_all[i, fi['is_AL']])
                rid = rids[i]
                team = rid.split('-')[-1] if '-' in rid else rid
                resid = seed - pred
                if abs(resid) >= 2 or (is_aq and 30 <= seed <= 45) or (is_al and 30 <= seed <= 45):
                    marker = ''
                    if is_aq and seed - net > 8:
                        marker = ' ← AQ mis-seeded'
                    elif is_al and net - seed > 5:
                        marker = ' ← AL over-seeded'
                    print(f'  {ssn:<10} {team:<25} {seed:3d} {pred:5d} {resid:+6d} {is_aq:3d} {is_al:3d} {net:5.0f}{marker}')

    # ════════════════════════════════════════════════════════════════════
    # Q6: Soft swap — continuous adjustment instead of binary
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' Q6: SOFT SWAP — Continuous adjustment')
    print(' Instead of swapping, adjust raw scores before Hungarian')
    print('='*72)

    # The idea: instead of post-hoc swapping predictions, adjust the
    # v12 raw scores for AQ teams (push up = higher seed = worse)
    # and AL teams (push down = lower seed if NET is worse)
    # BEFORE Hungarian assignment, so it's part of the optimization

    print(f'\n  Testing: add AQ/AL bias to v12 raw scores before zones')
    print(f'  For AQ teams with pred-NET>thresh: raw_v12[i] += adjustment')
    print(f'  For AL teams with NET>pred: raw_v12[i] -= adjustment')

    # Modified cache approach: adjust v12 raw scores
    from ncaa_v51_principled_fast import apply_zone_list

    best_soft_se = no_swap_se
    best_soft_cfg = None

    for aq_boost in [1, 2, 3, 4, 5, 6, 8]:
        for al_boost in [0, 1, 2, 3, 4]:
            for threshold in [5, 8, 10, 12]:
                n = len(y)
                preds = np.zeros(n, dtype=int)

                for hold_season, c in cache.items():
                    sm = c['sm']; si = c['si']; tm = c['tm']; avail = c['avail']
                    X_s = c['X_s']

                    # Start with raw v12 scores + apply bias
                    raw_adj = c['raw_v12'].copy()
                    for j, gi in enumerate(si):
                        if not test_mask[gi]:
                            continue
                        net = X_all[gi, fi['NET Rank']]
                        is_aq = int(X_all[gi, fi['is_AQ']])
                        is_al = int(X_all[gi, fi['is_AL']])

                        # If AQ team is predicted too low (pred << actual position
                        # based on NET), push the raw score UP (higher = worse seed)
                        if is_aq and not is_al and raw_adj[j] - net > threshold:
                            raw_adj[j] += aq_boost

                        # If AL team has NET worse than prediction, push down
                        if is_al and not is_aq and net > raw_adj[j]:
                            raw_adj[j] -= al_boost

                    # Re-run Hungarian with adjusted raw
                    a_v12 = hungarian(raw_adj, seasons[sm], avail, power=0.15)
                    a_v12 = apply_zone_list(a_v12, raw_adj, fn, X_s, tm, si,
                                            V12_ZONES_V50, power=0.15)

                    # Committee branch (unchanged)
                    sc = StandardScaler()
                    r = Ridge(alpha=10.0)
                    r.fit(sc.fit_transform(X_min8[c['gt_mask']]), y[c['gt_mask']])
                    raw_comm = r.predict(sc.transform(X_min8[sm]))
                    for j, gi in enumerate(si):
                        if not test_mask[gi]:
                            raw_comm[j] = y[gi]
                    a_comm = hungarian(raw_comm, seasons[sm], avail, power=0.15)
                    a_comm = apply_zone_list(a_comm, raw_comm, fn, X_s, tm, si,
                                            COMM_ZONES_V50, power=0.15)

                    # Blend
                    avg = 0.75 * a_v12.astype(float) + 0.25 * a_comm.astype(float)
                    for j, gi in enumerate(si):
                        if not test_mask[gi]:
                            avg[j] = y[gi]
                    final = hungarian(avg, seasons[sm], avail, power=0.15)
                    for j, gi in enumerate(si):
                        if test_mask[gi]:
                            preds[gi] = final[j]

                se, ex = se_exact(preds, y, test_mask)
                if se < best_soft_se:
                    best_soft_se = se
                    best_soft_cfg = (aq_boost, al_boost, threshold, se, ex)

    if best_soft_cfg:
        aq_b, al_b, thr, se, ex = best_soft_cfg
        print(f'\n  Best soft swap: aq_boost={aq_b}, al_boost={al_b}, thresh={thr}')
        print(f'  SE={se}, {ex}/91 (vs no-swap SE={no_swap_se}, vs hard-swap SE=14)')

        # Per season
        n = len(y)
        preds = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['sm']; si = c['si']; tm = c['tm']; avail = c['avail']
            X_s = c['X_s']
            raw_adj = c['raw_v12'].copy()
            for j, gi in enumerate(si):
                if not test_mask[gi]: continue
                net = X_all[gi, fi['NET Rank']]
                is_aq = int(X_all[gi, fi['is_AQ']])
                is_al = int(X_all[gi, fi['is_AL']])
                if is_aq and not is_al and raw_adj[j] - net > thr:
                    raw_adj[j] += aq_b
                if is_al and not is_aq and net > raw_adj[j]:
                    raw_adj[j] -= al_b
            a_v12 = hungarian(raw_adj, seasons[sm], avail, power=0.15)
            a_v12 = apply_zone_list(a_v12, raw_adj, fn, X_s, tm, si, V12_ZONES_V50, power=0.15)
            sc = StandardScaler()
            r = Ridge(alpha=10.0)
            r.fit(sc.fit_transform(X_min8[c['gt_mask']]), y[c['gt_mask']])
            raw_comm = r.predict(sc.transform(X_min8[sm]))
            for j, gi in enumerate(si):
                if not test_mask[gi]: raw_comm[j] = y[gi]
            a_comm = hungarian(raw_comm, seasons[sm], avail, power=0.15)
            a_comm = apply_zone_list(a_comm, raw_comm, fn, X_s, tm, si, COMM_ZONES_V50, power=0.15)
            avg = 0.75 * a_v12.astype(float) + 0.25 * a_comm.astype(float)
            for j, gi in enumerate(si):
                if not test_mask[gi]: avg[j] = y[gi]
            final = hungarian(avg, seasons[sm], avail, power=0.15)
            for j, gi in enumerate(si):
                if test_mask[gi]: preds[gi] = final[j]

        ps = per_season(preds, y, seasons, test_mask, folds)
        for s, (se_s, ex_s, n_s) in ps.items():
            print(f'    {s}: SE={se_s}, {ex_s}/{n_s}')

        # Show changes
        print(f'\n  Teams changed by soft swap:')
        print(f'  {"Team":<25} {"Ssn":<10} {"GT":>3} {"No-adj":>7} {"Adj":>5}')
        for i in range(len(y)):
            if test_mask[i] and preds[i] != p_no_swap[i]:
                team = rids[i].split('-')[-1] if '-' in rids[i] else rids[i]
                print(f'  {team:<25} {seasons[i]:<10} {int(y[i]):3d} {p_no_swap[i]:7d} {preds[i]:5d}')
    else:
        print(f'\n  No soft swap config improved over no-swap baseline (SE={no_swap_se})')

    # ════════════════════════════════════════════════════════════════════
    # FINAL ASSESSMENT
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' FINAL ASSESSMENT')
    print('='*72)

    print(f'\n  v50 with hard swap:                SE=14')
    print(f'  v50 without swap (7 zones):        SE={no_swap_se}')
    if best_soft_cfg:
        print(f'  v50 with soft swap (pre-Hungarian): SE={best_soft_cfg[3]}')
    print(f'\n  Key insight: the AQ↔AL pattern is a REAL committee bias')
    print(f'  (AQ teams from weak conferences get under-seeded despite strong NET).')
    print(f'  The question is whether 5 seasons of evidence is enough to trust it.')

    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*72)


if __name__ == '__main__':
    main()
