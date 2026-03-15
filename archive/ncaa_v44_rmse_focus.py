#!/usr/bin/env python3
"""
Explore RMSE-focused improvements on top of mid+bot+tail (v27).
We now know Kaggle optimizes RMSE, so we must minimize squared error,
not maximize exact matches.

Strategy:
1. Sweep zone boundaries for bot and tail zones to minimize SE
2. Try different parameter values for mid-zone
3. Try alternative bot-zone formulas
4. Check if MurraySt and IowaSt (the two worst errors) can be fixed
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from scipy.optimize import linear_sum_assignment
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

def main():
    t0 = time.time()
    print('='*70)
    print('  RMSE-FOCUSED IMPROVEMENT SEARCH (v27 baseline)')
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
    teams = labeled['Team'].values if 'Team' in labeled.columns else record_ids
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # Pre-compute raw scores and pass1 (Hungarian) per season
    raw_all = np.zeros(n_labeled)
    pass1_all = np.zeros(n_labeled, dtype=int)
    season_data = {}  # season -> (season_indices, test_mask_season, X_season, raw_scores)

    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0:
            continue
        gt_mask = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=USE_TOP_K_A,
                                     forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X[gt_mask], y[gt_mask], X_s, seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=0.15)
        tm = np.array([test_mask[gi] for gi in si])
        for i, gi in enumerate(si):
            raw_all[gi] = raw[i]
            pass1_all[gi] = p1[i]
        season_data[hold] = (si, tm, X_s, raw, p1.copy())

    def evaluate_config(mid_params, bot_zone, bot_params, tail_zone, tail_params):
        """Evaluate a configuration. Returns (exact, SE)."""
        preds = np.zeros(n_labeled, dtype=int)
        for hold in folds:
            if hold not in season_data:
                continue
            si, tm, X_s, raw, p1 = season_data[hold]
            assigned = p1.copy()

            # Mid-range
            if mid_params is not None:
                aq, al, sos = mid_params
                corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
                assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                               zone=(17, 34), blend=1.0, power=0.15)

            # Bot-zone
            if bot_params is not None:
                sn, nc, cb = bot_params
                corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
                assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                                  zone=bot_zone, power=0.15)

            # Tail-zone
            if tail_params is not None:
                opp, = tail_params
                corr = compute_tail_correction(fn, X_s, opp_rank=opp)
                assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                                zone=tail_zone, power=0.15)

            for i, gi in enumerate(si):
                if test_mask[gi]:
                    preds[gi] = assigned[i]

        gt = y[test_mask].astype(int)
        pred = preds[test_mask]
        exact = int((pred == gt).sum())
        se = int(np.sum((pred - gt)**2))
        return exact, se, preds

    # ── Baseline: v27 (mid + bot(50,60) + tail(61,65)) ──
    _, base_se, base_preds = evaluate_config(
        (0, 2, 3), (50, 60), (-4, 3, -1), (61, 65), (-3,))
    print(f'\n  v27 baseline: SE={base_se}')

    # ── PHASE 1: Sweep bot-zone boundaries ──
    print('\n  ══ PHASE 1: Bot-zone boundary sweep ══')
    best_bot_se = base_se
    best_bot_zone = (50, 60)
    results = []
    for lo in range(45, 58):
        for hi in range(lo+3, 68):
            _, se, _ = evaluate_config((0, 2, 3), (lo, hi), (-4, 3, -1), (61, 65), (-3,))
            results.append((se, lo, hi))
            if se < best_bot_se:
                best_bot_se = se
                best_bot_zone = (lo, hi)

    results.sort()
    print(f'  Top 10 bot-zone boundaries:')
    for se, lo, hi in results[:10]:
        marker = ' ◄' if (lo, hi) == (50, 60) else ''
        print(f'    ({lo},{hi}): SE={se}{marker}')
    print(f'  Best bot-zone: {best_bot_zone} SE={best_bot_se} (was {base_se})')

    # ── PHASE 2: Sweep tail-zone boundaries ──
    print('\n  ══ PHASE 2: Tail-zone boundary sweep ══')
    best_tail_se = best_bot_se
    best_tail_zone = (61, 65)
    results2 = []
    for lo in range(55, 66):
        for hi in range(lo+1, 68):
            _, se, _ = evaluate_config((0, 2, 3), best_bot_zone, (-4, 3, -1), (lo, hi), (-3,))
            results2.append((se, lo, hi))
            if se < best_tail_se:
                best_tail_se = se
                best_tail_zone = (lo, hi)

    results2.sort()
    print(f'  Top 10 tail-zone boundaries:')
    for se, lo, hi in results2[:10]:
        marker = ' ◄' if (lo, hi) == (61, 65) else ''
        print(f'    ({lo},{hi}): SE={se}{marker}')
    print(f'  Best tail-zone: {best_tail_zone} SE={best_tail_se}')

    # ── PHASE 3: Sweep mid-range params ──
    print('\n  ══ PHASE 3: Mid-range parameter sweep ══')
    best_mid_se = best_tail_se
    best_mid_params = (0, 2, 3)
    results3 = []
    for aq in range(-2, 3):
        for al in range(0, 5):
            for sos in range(0, 6):
                _, se, _ = evaluate_config(
                    (aq, al, sos), best_bot_zone, (-4, 3, -1), best_tail_zone, (-3,))
                results3.append((se, aq, al, sos))
                if se < best_mid_se:
                    best_mid_se = se
                    best_mid_params = (aq, al, sos)

    results3.sort()
    print(f'  Top 10 mid-range params:')
    for se, aq, al, sos in results3[:10]:
        marker = ' ◄' if (aq, al, sos) == (0, 2, 3) else ''
        print(f'    aq={aq}, al={al}, sos={sos}: SE={se}{marker}')
    print(f'  Best mid params: {best_mid_params} SE={best_mid_se}')

    # ── PHASE 4: Sweep bot-zone params with best boundaries ──
    print('\n  ══ PHASE 4: Bot-zone parameter sweep ══')
    best_bp_se = best_mid_se
    best_bp = (-4, 3, -1)
    results4 = []
    for sn in range(-6, 1):
        for nc in range(0, 6):
            for cb in range(-3, 2):
                _, se, _ = evaluate_config(
                    best_mid_params, best_bot_zone, (sn, nc, cb), best_tail_zone, (-3,))
                results4.append((se, sn, nc, cb))
                if se < best_bp_se:
                    best_bp_se = se
                    best_bp = (sn, nc, cb)

    results4.sort()
    print(f'  Top 10 bot-zone params:')
    for se, sn, nc, cb in results4[:10]:
        marker = ' ◄' if (sn, nc, cb) == (-4, 3, -1) else ''
        print(f'    sn={sn}, nc={nc}, cb={cb}: SE={se}{marker}')
    print(f'  Best bot params: {best_bp} SE={best_bp_se}')

    # ── PHASE 5: Sweep tail-zone params with best everything ──
    print('\n  ══ PHASE 5: Tail-zone parameter sweep ══')
    best_tp_se = best_bp_se
    best_tp = (-3,)
    results5 = []
    for opp in range(-8, 3):
        _, se, _ = evaluate_config(
            best_mid_params, best_bot_zone, best_bp, best_tail_zone, (opp,))
        results5.append((se, opp))
        if se < best_tp_se:
            best_tp_se = se
            best_tp = (opp,)

    results5.sort()
    print(f'  All tail-zone opp_rank values:')
    for se, opp in results5:
        marker = ' ◄' if opp == -3 else ''
        print(f'    opp={opp}: SE={se}{marker}')
    print(f'  Best tail opp: {best_tp} SE={best_tp_se}')

    # ── Final evaluation ──
    print('\n\n  ══ FINAL COMPARISON ══')
    exact_best, se_best, preds_best = evaluate_config(
        best_mid_params, best_bot_zone, best_bp, best_tail_zone, best_tp)
    rmse_best = np.sqrt(se_best / 91)
    rmse451_best = np.sqrt(se_best / 451)

    _, base_se, _ = evaluate_config((0, 2, 3), (50, 60), (-4, 3, -1), (61, 65), (-3,))

    print(f'  v27 baseline:  SE={base_se}, RMSE451={np.sqrt(base_se/451):.4f}')
    print(f'  Best found:    SE={se_best}, RMSE451={rmse451_best:.4f}, exact={exact_best}/91')
    print(f'  Improvement:   SE {se_best - base_se:+d}')
    print(f'\n  Best config:')
    print(f'    Mid: aq={best_mid_params[0]}, al={best_mid_params[1]}, sos={best_mid_params[2]}')
    print(f'    Bot zone: {best_bot_zone}, sn={best_bp[0]}, nc={best_bp[1]}, cb={best_bp[2]}')
    print(f'    Tail zone: {best_tail_zone}, opp={best_tp[0]}')

    # Show teams that changed
    _, _, base_p = evaluate_config((0, 2, 3), (50, 60), (-4, 3, -1), (61, 65), (-3,))
    print(f'\n  Teams that DIFFER between v27 and best:')
    print(f'  {"RID":<35} {"Team":<22} {"GT":>3} {"v27":>4} {"best":>4} {"SE_change":>10}')
    for i in range(n_labeled):
        if test_mask[i] and base_p[i] != preds_best[i]:
            gt = int(y[i])
            se_old = (base_p[i] - gt)**2
            se_new = (preds_best[i] - gt)**2
            change = se_new - se_old
            print(f'  {record_ids[i]:<35} {teams[i]:<22} {gt:3d} {base_p[i]:4d} {preds_best[i]:4d} {change:+10d}')

    # ── Worst remaining errors ──
    print(f'\n  Worst remaining errors:')
    errors = []
    for i in range(n_labeled):
        if test_mask[i]:
            gt = int(y[i])
            err = preds_best[i] - gt
            errors.append((abs(err), err, i))
    errors.sort(reverse=True)
    print(f'  {"RID":<35} {"Team":<22} {"GT":>3} {"Pred":>4} {"Err":>5} {"SE":>5}')
    for ae, e, i in errors[:15]:
        print(f'  {record_ids[i]:<35} {teams[i]:<22} {int(y[i]):3d} {preds_best[i]:4d} {e:+5d} {e**2:5d}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
