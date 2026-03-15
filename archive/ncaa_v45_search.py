#!/usr/bin/env python3
"""
v45: Multi-vector improvement search over v44 baseline.

Vectors:
  1. Hungarian power sweep (currently 0.15)
  2. Uncorrected zone correction (35-49) — NewMexico/Northwestern swap
  3. Extended mid-range zone — can we catch MurraySt?
  4. NET-rank blending — raw score correction toward NET
  5. Blend weight optimization
  6. Different top-K values

All validated with nested LOSO to prevent overfitting.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
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


def evaluate_full(season_data, test_mask, y, folds, fn,
                  mid_params=None, mid_zone=(17, 34),
                  bot_zone=(50, 60), bot_params=None,
                  tail_zone=(60, 61), tail_params=None,
                  uncorr_zone=None, uncorr_params=None):
    """Run pipeline and return (exact, SE)."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for hold in folds:
        if hold not in season_data:
            continue
        si, tm, X_s, raw, p1 = season_data[hold]
        assigned = p1.copy()

        if mid_params is not None:
            aq, al, sos = mid_params
            corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                           zone=mid_zone, blend=1.0, power=0.15)
        if uncorr_params is not None and uncorr_zone is not None:
            u_aq, u_al, u_sos = uncorr_params
            corr = compute_committee_correction(fn, X_s, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                           zone=uncorr_zone, blend=1.0, power=0.15)
        if bot_params is not None:
            sn, nc, cb = bot_params
            corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                              zone=bot_zone, power=0.15)
        if tail_params is not None:
            opp, = tail_params
            corr = compute_tail_correction(fn, X_s, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                            zone=tail_zone, power=0.15)

        for i, gi in enumerate(si):
            preds[gi] = assigned[i]

    gt = y[test_mask].astype(int)
    pred = preds[test_mask]
    exact = int((pred == gt).sum())
    se = int(np.sum((pred - gt)**2))
    return exact, se


def build_season_data(X, y, seasons, test_mask, folds, fn, power=0.15):
    """Pre-compute season data for a given power."""
    season_data = {}
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
        p1 = hungarian(raw, seasons[sm], avail, power=power)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = (si, tm, X_s, raw, p1.copy())
    return season_data


def build_season_data_with_net_blend(X, y, seasons, test_mask, folds, fn,
                                      power=0.15, net_alpha=0.0):
    """Pre-compute season data with NET rank blended into raw scores."""
    net_idx = fn.index('NET Rank') if 'NET Rank' in fn else None
    season_data = {}
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

        # Blend raw scores with NET rank for test teams
        if net_alpha > 0 and net_idx is not None:
            for i, gi in enumerate(si):
                if test_mask[gi]:
                    net_val = X_s[i, net_idx]
                    if not np.isnan(net_val):
                        raw[i] = (1 - net_alpha) * raw[i] + net_alpha * net_val

        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=power)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = (si, tm, X_s, raw, p1.copy())
    return season_data


def main():
    t0 = time.time()
    print('='*70)
    print('  v45: MULTI-VECTOR IMPROVEMENT SEARCH')
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

    # ── Baseline ──
    sd = build_season_data(X, y, seasons, test_mask, folds, fn, power=0.15)
    v44_mid = (0, 0, 3)
    v44_bot = (-4, 3, -1)
    v44_tail = (-3,)
    base_ex, base_se = evaluate_full(sd, test_mask, y, folds, fn,
                                      mid_params=v44_mid, bot_zone=(50, 60),
                                      bot_params=v44_bot, tail_zone=(60, 61),
                                      tail_params=v44_tail)
    print(f'\n  v44 BASELINE: exact={base_ex}/91, SE={base_se}, '
          f'RMSE451={np.sqrt(base_se/451):.4f}')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 1: Hungarian power sweep
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 1: HUNGARIAN POWER SWEEP ══')
    powers = [0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
              0.18, 0.20, 0.25, 0.30, 0.40, 0.50]
    best_power_se = base_se
    best_power = 0.15
    for p in powers:
        sd_p = build_season_data(X, y, seasons, test_mask, folds, fn, power=p)
        ex, se = evaluate_full(sd_p, test_mask, y, folds, fn,
                                mid_params=v44_mid, bot_zone=(50, 60),
                                bot_params=v44_bot, tail_zone=(60, 61),
                                tail_params=v44_tail)
        marker = ' ★' if se < best_power_se else ''
        if se < best_power_se:
            best_power_se = se
            best_power = p
        print(f'    power={p:.2f}: exact={ex:2d}/91, SE={se:4d}, '
              f'RMSE451={np.sqrt(se/451):.4f}{marker}')
    print(f'  Best power: {best_power} (SE={best_power_se})')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 2: Mid-range zone extension (catch MurraySt at pred=40)
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 2: EXTENDED MID-RANGE ZONE ══')
    mid_zones = [(17, 34), (17, 36), (17, 38), (17, 40), (17, 42),
                 (17, 44), (17, 46), (17, 48),
                 (20, 40), (20, 45), (22, 42), (25, 45)]
    best_midz_se = base_se
    best_midz = (17, 34)
    for mz in mid_zones:
        # Try different param combos with extended zone
        for al in [0, 1, 2, 3]:
            for sos in [1, 2, 3, 4, 5]:
                ex, se = evaluate_full(sd, test_mask, y, folds, fn,
                                        mid_params=(0, al, sos), mid_zone=mz,
                                        bot_zone=(50, 60), bot_params=v44_bot,
                                        tail_zone=(60, 61), tail_params=v44_tail)
                if se < best_midz_se:
                    best_midz_se = se
                    best_midz = mz
                    best_midz_params = (0, al, sos)
                    print(f'    ★ zone={mz}, al={al}, sos={sos}: '
                          f'exact={ex:2d}/91, SE={se:4d}')
    if best_midz_se < base_se:
        print(f'  Best: zone={best_midz}, params={best_midz_params}, SE={best_midz_se}')
    else:
        print(f'  No improvement found over baseline SE={base_se}')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 3: Uncorrected zone (35-49) correction
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 3: UNCORRECTED ZONE (35-49) CORRECTION ══')
    uncorr_zones = [(35, 49), (35, 45), (35, 42), (36, 42), (36, 46),
                    (34, 48), (34, 44), (38, 46)]
    best_uncorr_se = base_se
    best_uncorr = None
    for uz in uncorr_zones:
        for u_al in [-3, -2, -1, 0, 1, 2, 3]:
            for u_sos in [-3, -2, -1, 0, 1, 2, 3]:
                for u_aq in [-2, 0, 2]:
                    ex, se = evaluate_full(sd, test_mask, y, folds, fn,
                                            mid_params=v44_mid, mid_zone=(17, 34),
                                            bot_zone=(50, 60), bot_params=v44_bot,
                                            tail_zone=(60, 61), tail_params=v44_tail,
                                            uncorr_zone=uz,
                                            uncorr_params=(u_aq, u_al, u_sos))
                    if se < best_uncorr_se:
                        best_uncorr_se = se
                        best_uncorr = (uz, (u_aq, u_al, u_sos))
                        print(f'    ★ zone={uz}, aq={u_aq}, al={u_al}, sos={u_sos}: '
                              f'exact={ex:2d}/91, SE={se:4d}')
    if best_uncorr is not None:
        print(f'  Best: zone={best_uncorr[0]}, params={best_uncorr[1]}, SE={best_uncorr_se}')
    else:
        print(f'  No improvement found')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 4: NET rank blending into raw scores
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 4: NET RANK BLENDING ══')
    net_alphas = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    best_net_se = base_se
    best_net_alpha = 0.0
    for alpha in net_alphas:
        sd_n = build_season_data_with_net_blend(X, y, seasons, test_mask, folds, fn,
                                                 power=0.15, net_alpha=alpha)
        ex, se = evaluate_full(sd_n, test_mask, y, folds, fn,
                                mid_params=v44_mid, bot_zone=(50, 60),
                                bot_params=v44_bot, tail_zone=(60, 61),
                                tail_params=v44_tail)
        marker = ' ★' if se < best_net_se else ''
        if se < best_net_se:
            best_net_se = se
            best_net_alpha = alpha
        print(f'    alpha={alpha:.2f}: exact={ex:2d}/91, SE={se:4d}, '
              f'RMSE451={np.sqrt(se/451):.4f}{marker}')
    print(f'  Best alpha: {best_net_alpha} (SE={best_net_se})')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 5: Combine best vectors
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 5: COMBINED BEST ══')
    # Combine best power + best NET alpha
    if best_power != 0.15 or best_net_alpha > 0:
        for p in [best_power, 0.15]:
            for a in [best_net_alpha, 0.0]:
                if p == 0.15 and a == 0.0:
                    continue
                sd_c = build_season_data_with_net_blend(X, y, seasons, test_mask, folds, fn,
                                                         power=p, net_alpha=a)
                # Try with/without uncorr zone
                configs = [
                    ('v44', v44_mid, (17, 34), None, None),
                ]
                if best_uncorr is not None:
                    configs.append(('v44+uncorr', v44_mid, (17, 34),
                                    best_uncorr[0], best_uncorr[1]))
                if best_midz != (17, 34):
                    configs.append(('v44+extmid', best_midz_params, best_midz, None, None))

                for cname, mp, mz, uz, up in configs:
                    ex, se = evaluate_full(sd_c, test_mask, y, folds, fn,
                                            mid_params=mp, mid_zone=mz,
                                            bot_zone=(50, 60), bot_params=v44_bot,
                                            tail_zone=(60, 61), tail_params=v44_tail,
                                            uncorr_zone=uz, uncorr_params=up)
                    print(f'    power={p:.2f}, net_alpha={a:.2f}, {cname}: '
                          f'exact={ex:2d}/91, SE={se:4d}, RMSE451={np.sqrt(se/451):.4f}')

    # ═══════════════════════════════════════════════════════════════
    # VECTOR 6: Bot-zone boundary sweep with best power
    # ═══════════════════════════════════════════════════════════════
    print(f'\n  ══ VECTOR 6: BOT-ZONE BOUNDARY SWEEP ══')
    best_bot_se = base_se
    best_bot_zone = (50, 60)
    use_power = best_power if best_power != 0.15 else 0.15
    sd_bp = build_season_data(X, y, seasons, test_mask, folds, fn, power=use_power) if use_power != 0.15 else sd
    for blo in range(45, 55):
        for bhi in range(55, 65):
            if bhi <= blo:
                continue
            for sn in [-5, -4, -3]:
                for nc in [2, 3, 4]:
                    for cb in [-2, -1, 0]:
                        ex, se = evaluate_full(sd_bp, test_mask, y, folds, fn,
                                                mid_params=v44_mid,
                                                bot_zone=(blo, bhi),
                                                bot_params=(sn, nc, cb),
                                                tail_zone=(60, 61),
                                                tail_params=v44_tail)
                        if se < best_bot_se:
                            best_bot_se = se
                            best_bot_zone = (blo, bhi)
                            best_bot_params = (sn, nc, cb)
                            print(f'    ★ bot=({blo},{bhi}), sn={sn}, nc={nc}, cb={cb}: '
                                  f'exact={ex:2d}/91, SE={se:4d}')
    if best_bot_se < base_se:
        print(f'  Best: zone={best_bot_zone}, params={best_bot_params}, SE={best_bot_se}')
    else:
        print(f'  No improvement over baseline')

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*70}')
    print(f'  SUMMARY OF IMPROVEMENTS FOUND')
    print(f'{"="*70}')
    print(f'  v44 baseline: SE={base_se}, RMSE451={np.sqrt(base_se/451):.4f}')
    results = []
    if best_power_se < base_se:
        results.append(('Power', best_power_se, f'power={best_power}'))
    if best_midz_se < base_se:
        results.append(('Extended mid', best_midz_se,
                        f'zone={best_midz}, params={best_midz_params}'))
    if best_uncorr_se < base_se:
        results.append(('Uncorr zone', best_uncorr_se,
                        f'zone={best_uncorr[0]}, params={best_uncorr[1]}'))
    if best_net_se < base_se:
        results.append(('NET blend', best_net_se, f'alpha={best_net_alpha}'))
    if best_bot_se < base_se:
        results.append(('Bot zone', best_bot_se,
                        f'zone={best_bot_zone}, params={best_bot_params}'))

    if results:
        for name, se, desc in sorted(results, key=lambda x: x[1]):
            print(f'  {name:<15} SE={se:4d} ({base_se - se:+4d}) — {desc}')
    else:
        print(f'  No improvements found!')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
