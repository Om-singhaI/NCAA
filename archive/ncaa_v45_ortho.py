#!/usr/bin/env python3
"""
Explore additional improvements on top of v45b (SE=263).
Try:
1. Extended bot-zone to capture swapped pairs at 47-52
2. Fine-tuned mid-zone params with v45b uncorr
3. Additional new zones (top seeds, far-tail)
4. Different tail-zone boundaries
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

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


def full_pipeline(sd, test_mask, y, folds, fn,
                  mid_params=(0, 0, 3), mid_zone=(17, 34),
                  uncorr_params=(-2, -3, -4), uncorr_zone=(34, 44),
                  bot_zone=(50, 60), bot_params=(-4, 3, -1),
                  tail_zone=(60, 61), tail_params=(-3,),
                  extra_zone=None, extra_params=None, extra_type='mid'):
    """Full v45 pipeline with optional extra zone."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for hold in folds:
        if hold not in sd:
            continue
        si, tm, X_s, raw, p1 = sd[hold]
        assigned = p1.copy()

        # Mid-range
        aq, al, sos = mid_params
        corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                       zone=mid_zone, blend=1.0, power=0.15)
        # Uncorr zone
        if uncorr_params:
            u_aq, u_al, u_sos = uncorr_params
            corr = compute_committee_correction(fn, X_s, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                           zone=uncorr_zone, blend=1.0, power=0.15)
        # Bot-zone
        sn, nc, cb = bot_params
        corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                          zone=bot_zone, power=0.15)
        # Tail-zone
        opp = tail_params[0]
        corr = compute_tail_correction(fn, X_s, opp_rank=opp)
        assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                        zone=tail_zone, power=0.15)
        # Extra zone
        if extra_zone and extra_params:
            if extra_type == 'mid':
                e_aq, e_al, e_sos = extra_params
                corr = compute_committee_correction(fn, X_s, alpha_aq=e_aq, beta_al=e_al, gamma_sos=e_sos)
                assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                               zone=extra_zone, blend=1.0, power=0.15)
            elif extra_type == 'bot':
                e_sn, e_nc, e_cb = extra_params
                corr = compute_bottom_correction(fn, X_s, sosnet=e_sn, net_conf=e_nc, cbhist=e_cb)
                assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                                  zone=extra_zone, power=0.15)
            elif extra_type == 'tail':
                corr = compute_tail_correction(fn, X_s, opp_rank=extra_params[0])
                assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                                zone=extra_zone, power=0.15)

        for i, gi in enumerate(si):
            preds[gi] = assigned[i]

    gt = y[test_mask].astype(int)
    pred = preds[test_mask]
    return int((pred == gt).sum()), int(np.sum((pred - gt)**2)), preds


def main():
    t0 = time.time()
    print('='*70)
    print('  ORTHOGONAL IMPROVEMENTS ON TOP OF v45b (SE=263)')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
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

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # Build season data
    sd = {}
    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0: continue
        gt_mask = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=USE_TOP_K_A,
                                     forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X[gt_mask], y[gt_mask], X_s, seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=0.15)
        tm = np.array([test_mask[gi] for gi in si])
        sd[hold] = (si, tm, X_s, raw, p1.copy())

    # Baseline
    base_ex, base_se, _ = full_pipeline(sd, test_mask, y, folds, fn)
    print(f'\n  v45b BASELINE: exact={base_ex}/91, SE={base_se}')

    # ── 1. Extended bot-zone ──
    print(f'\n  ══ 1. BOT-ZONE BOUNDARY SWEEP ══')
    best_bot = base_se
    for blo in range(45, 53):
        for bhi in range(55, 65):
            if bhi <= blo: continue
            for sn in [-6, -5, -4, -3, -2]:
                for nc in [1, 2, 3, 4, 5]:
                    for cb in [-3, -2, -1, 0, 1]:
                        ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                                   bot_zone=(blo, bhi),
                                                   bot_params=(sn, nc, cb))
                        if se < best_bot:
                            best_bot = se
                            best_bot_cfg = ((blo, bhi), (sn, nc, cb))
                            print(f'    ★ bot=({blo},{bhi}) sn={sn} nc={nc} cb={cb}: '
                                  f'SE={se} ({se-base_se:+d})')
    if best_bot < base_se:
        print(f'  Best bot: {best_bot_cfg}, SE={best_bot}')
    else:
        print(f'  No improvement')

    # ── 2. Mid-range params re-tune ──
    print(f'\n  ══ 2. MID-RANGE PARAMS RE-TUNE ══')
    best_mid = base_se
    for aq in [-3, -2, -1, 0, 1, 2]:
        for al in [-2, -1, 0, 1, 2, 3]:
            for sos in [1, 2, 3, 4, 5]:
                ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                           mid_params=(aq, al, sos))
                if se < best_mid:
                    best_mid = se
                    best_mid_cfg = (aq, al, sos)
                    print(f'    ★ aq={aq} al={al} sos={sos}: SE={se} ({se-base_se:+d})')
    if best_mid < base_se:
        print(f'  Best mid: {best_mid_cfg}, SE={best_mid}')
    else:
        print(f'  No improvement')

    # ── 3. Uncorr zone params fine-tune ──
    print(f'\n  ══ 3. UNCORR ZONE PARAMS FINE-TUNE ══')
    best_uncorr = base_se
    for u_aq in [-4, -3, -2, -1, 0]:
        for u_al in [-5, -4, -3, -2, -1]:
            for u_sos in [-6, -5, -4, -3, -2]:
                ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                           uncorr_params=(u_aq, u_al, u_sos))
                if se < best_uncorr:
                    best_uncorr = se
                    best_uncorr_cfg = (u_aq, u_al, u_sos)
                    print(f'    ★ aq={u_aq} al={u_al} sos={u_sos}: SE={se} ({se-base_se:+d})')
    if best_uncorr < base_se:
        print(f'  Best uncorr: {best_uncorr_cfg}, SE={best_uncorr}')
    else:
        print(f'  No improvement')

    # ── 4. Tail-zone sweep ──
    print(f'\n  ══ 4. TAIL-ZONE SWEEP ══')
    best_tail = base_se
    for tlo in range(58, 64):
        for thi in range(tlo+1, 68):
            for opp in [-5, -4, -3, -2, -1, 1, 2, 3]:
                ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                           tail_zone=(tlo, thi),
                                           tail_params=(opp,))
                if se < best_tail:
                    best_tail = se
                    best_tail_cfg = ((tlo, thi), opp)
                    print(f'    ★ tail=({tlo},{thi}) opp={opp}: SE={se} ({se-base_se:+d})')
    if best_tail < base_se:
        print(f'  Best tail: {best_tail_cfg}, SE={best_tail}')
    else:
        print(f'  No improvement')

    # ── 5. Additional zone using committee_correction ──
    print(f'\n  ══ 5. ADDITIONAL ZONES ══')
    best_extra = base_se
    zones_to_try = [(44, 50), (44, 52), (45, 52), (45, 55),
                     (1, 16), (5, 16), (62, 68), (63, 67)]
    for ez in zones_to_try:
        for e_aq in [-3, -1, 0, 1, 3]:
            for e_al in [-3, -1, 0, 1, 3]:
                for e_sos in [-3, -1, 0, 1, 3]:
                    ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                               extra_zone=ez,
                                               extra_params=(e_aq, e_al, e_sos),
                                               extra_type='mid')
                    if se < best_extra:
                        best_extra = se
                        best_extra_cfg = (ez, (e_aq, e_al, e_sos), 'mid')
                        print(f'    ★ zone={ez} aq={e_aq} al={e_al} sos={e_sos}: '
                              f'SE={se} ({se-base_se:+d})')
    # Also try bot-type additional zone
    for ez in [(44, 52), (45, 52), (47, 55)]:
        for e_sn in [-5, -3, -1, 1, 3]:
            for e_nc in [-3, -1, 1, 3, 5]:
                for e_cb in [-3, -1, 0, 1]:
                    ex, se, _ = full_pipeline(sd, test_mask, y, folds, fn,
                                               extra_zone=ez,
                                               extra_params=(e_sn, e_nc, e_cb),
                                               extra_type='bot')
                    if se < best_extra:
                        best_extra = se
                        best_extra_cfg = (ez, (e_sn, e_nc, e_cb), 'bot')
                        print(f'    ★ zone={ez} sn={e_sn} nc={e_nc} cb={e_cb} (bot): '
                              f'SE={se} ({se-base_se:+d})')
    if best_extra < base_se:
        print(f'  Best extra: {best_extra_cfg}, SE={best_extra}')
    else:
        print(f'  No improvement')

    # ── SUMMARY ──
    print(f'\n{"="*70}')
    print(f'  IMPROVEMENT SUMMARY (over v45b SE={base_se})')
    print(f'{"="*70}')
    results = []
    if best_bot < base_se:
        results.append(('Bot-zone', best_bot, str(best_bot_cfg)))
    if best_mid < base_se:
        results.append(('Mid-range', best_mid, str(best_mid_cfg)))
    if best_uncorr < base_se:
        results.append(('Uncorr zone', best_uncorr, str(best_uncorr_cfg)))
    if best_tail < base_se:
        results.append(('Tail-zone', best_tail, str(best_tail_cfg)))
    if best_extra < base_se:
        results.append(('Extra zone', best_extra, str(best_extra_cfg)))
    
    if results:
        for name, se, desc in sorted(results, key=lambda x: x[1]):
            print(f'  {name:<15} SE={se:4d} ({se-base_se:+4d}) — {desc}')
    else:
        print(f'  No improvements found!')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
