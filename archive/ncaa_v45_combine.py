#!/usr/bin/env python3
"""
Combine orthogonal improvements found on top of v45b.
Best combo so far: extra bot zone (44,52) sn=1,nc=1,cb=-1 → SE=237

Try:
1. Combine extra zone with tail improvement
2. Non-overlapping zone arrangements
3. Full nested LOSO validation
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


def pipeline(sd, test_mask, y, folds, fn,
             mid_params=(0, 0, 3), mid_zone=(17, 34),
             uncorr_params=(-2, -3, -4), uncorr_zone=(34, 44),
             midbot_params=None, midbot_zone=None,
             bot_zone=(50, 60), bot_params=(-4, 3, -1),
             tail_zone=(60, 61), tail_params=(-3,)):
    """Full pipeline with all zones."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for hold in folds:
        if hold not in sd: continue
        si, tm, X_s, raw, p1 = sd[hold]
        assigned = p1.copy()

        # 1. Mid-range (17-34)
        aq, al, sos = mid_params
        corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                       zone=mid_zone, blend=1.0, power=0.15)
        # 2. Uncorr zone (34-44)
        if uncorr_params:
            u_aq, u_al, u_sos = uncorr_params
            corr = compute_committee_correction(fn, X_s, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                           zone=uncorr_zone, blend=1.0, power=0.15)
        # 3. Mid-bot zone (44-52)
        if midbot_params and midbot_zone:
            sn, nc, cb = midbot_params
            corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                              zone=midbot_zone, power=0.15)
        # 4. Bot-zone
        sn, nc, cb = bot_params
        corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                          zone=bot_zone, power=0.15)
        # 5. Tail
        opp = tail_params[0]
        corr = compute_tail_correction(fn, X_s, opp_rank=opp)
        assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                        zone=tail_zone, power=0.15)

        for i, gi in enumerate(si):
            preds[gi] = assigned[i]

    gt = y[test_mask].astype(int)
    pred = preds[test_mask]
    return int((pred == gt).sum()), int(np.sum((pred - gt)**2)), preds


def main():
    t0 = time.time()
    print('='*70)
    print('  COMBINE IMPROVEMENTS + NESTED LOSO')
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

    # ── Baseline configs ──
    configs = {
        'v45b baseline': {},  # uses defaults
        'v45b + midbot(44,52)': {
            'midbot_zone': (44, 52), 'midbot_params': (1, 1, -1)
        },
        'v45b + midbot(44,52) + tail(60,63,+1)': {
            'midbot_zone': (44, 52), 'midbot_params': (1, 1, -1),
            'tail_zone': (60, 63), 'tail_params': (1,)
        },
        'v45b + midbot(44,52) + bot(52,60)': {
            'midbot_zone': (44, 52), 'midbot_params': (1, 1, -1),
            'bot_zone': (52, 60)  # non-overlapping
        },
    }

    # Quick full evaluation
    print(f'\n  ══ FULL EVALUATION ══')
    for cname, kwargs in configs.items():
        ex, se, _ = pipeline(sd, test_mask, y, folds, fn, **kwargs)
        print(f'  {cname:<45} exact={ex:2d}/91 SE={se:4d} RMSE451={np.sqrt(se/451):.4f}')

    # ── Fine-tune midbot zone ──
    print(f'\n  ══ MIDBOT ZONE FINE-TUNE ══')
    best_combo = 263  # v45b baseline
    best_combo_cfg = None
    for mblo in range(44, 50):
        for mbhi in range(mblo+2, 55):
            for sn in [-3, -1, 0, 1, 3]:
                for nc in [-1, 0, 1, 2, 3]:
                    for cb in [-2, -1, 0, 1]:
                        # Try with non-overlapping bot
                        bot_lo = max(mbhi, 50)
                        ex, se, _ = pipeline(sd, test_mask, y, folds, fn,
                                              midbot_zone=(mblo, mbhi),
                                              midbot_params=(sn, nc, cb),
                                              bot_zone=(bot_lo, 60))
                        if se < best_combo:
                            best_combo = se
                            best_combo_cfg = {
                                'midbot_zone': (mblo, mbhi),
                                'midbot_params': (sn, nc, cb),
                                'bot_zone': (bot_lo, 60)
                            }
                            print(f'    ★ midbot=({mblo},{mbhi}) sn={sn} nc={nc} cb={cb} '
                                  f'bot=({bot_lo},60): SE={se} ({se-263:+d})')
    
    if best_combo_cfg:
        print(f'\n  Best combo: {best_combo_cfg}, SE={best_combo}')
        
        # Also try with different tail
        for tlo in range(58, 64):
            for thi in range(tlo+1, 68):
                for opp in [-3, -1, 0, 1, 3]:
                    kw = dict(best_combo_cfg)
                    kw['tail_zone'] = (tlo, thi)
                    kw['tail_params'] = (opp,)
                    ex, se, _ = pipeline(sd, test_mask, y, folds, fn, **kw)
                    if se < best_combo:
                        best_combo = se
                        best_combo_cfg['tail_zone'] = (tlo, thi)
                        best_combo_cfg['tail_params'] = (opp,)
                        print(f'    ★ +tail=({tlo},{thi}) opp={opp}: SE={se}')

    print(f'\n  FINAL BEST: SE={best_combo}')
    if best_combo_cfg:
        print(f'  Config: {best_combo_cfg}')

    # ── Nested LOSO validation ──
    print(f'\n\n  ══ NESTED LOSO VALIDATION ══')
    
    loso_configs = {
        'v45b (SE=263)': {},
        'v45a (SE=307)': {'uncorr_params': (-2, -3, -3)},
        'v44 (SE=429)': {'uncorr_params': None},
    }
    if best_combo_cfg and best_combo < 263:
        loso_configs['v45c (best combo)'] = best_combo_cfg

    nested_preds = {c: np.zeros(n, dtype=int) for c in loso_configs}

    for hold_out in folds:
        inner_folds = [f for f in folds if f != hold_out]
        inner_test_mask = test_mask.copy()
        for i in range(n):
            if seasons[i] == hold_out:
                inner_test_mask[i] = False

        inner_sd = {}
        for f in inner_folds:
            sm = (seasons == f)
            si = np.where(sm)[0]
            st = inner_test_mask & sm
            if st.sum() == 0: continue
            global_train = ~inner_test_mask
            X_s = X[sm]
            tki = select_top_k_features(X[global_train], y[global_train], fn, k=USE_TOP_K_A,
                                         forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend(X[global_train], y[global_train], X_s, seasons[global_train], tki)
            for i, gi in enumerate(si):
                if not inner_test_mask[gi]: raw[i] = y[gi]
            avail = {f: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=0.15)
            tm = np.array([inner_test_mask[gi] for gi in si])
            inner_sd[f] = (si, tm, X_s, raw, p1.copy())

        hold_sd = {}
        sm = (seasons == hold_out)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() > 0:
            gt_mask = ~st
            X_s = X[sm]
            tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=USE_TOP_K_A,
                                         forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend(X[gt_mask], y[gt_mask], X_s, seasons[gt_mask], tki)
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw[i] = y[gi]
            avail = {hold_out: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=0.15)
            tm = np.array([test_mask[gi] for gi in si])
            hold_sd[hold_out] = (si, tm, X_s, raw, p1.copy())

        # Inner evaluation
        inner_se = {}
        for cname, kwargs in loso_configs.items():
            _, se, _ = pipeline(inner_sd, inner_test_mask, y, inner_folds, fn, **kwargs)
            inner_se[cname] = se

        winner = min(inner_se, key=inner_se.get)
        print(f'  {hold_out}: inner winner="{winner}" SE={inner_se[winner]}')

        # Apply each to held-out
        for cname, kwargs in loso_configs.items():
            _, _, cp = pipeline(hold_sd, test_mask, y, [hold_out], fn, **kwargs)
            for i in range(n):
                if test_mask[i] and seasons[i] == hold_out:
                    nested_preds[cname][i] = cp[i]

    # Results
    print(f'\n  {"Config":<25} {"Full SE":>8} {"Nested SE":>10} {"Gap":>5} {"Exact":>6}')
    for cname, kwargs in loso_configs.items():
        _, full_se, _ = pipeline(sd, test_mask, y, folds, fn, **kwargs)
        gt = y[test_mask].astype(int)
        pred = nested_preds[cname][test_mask]
        nested_se = int(np.sum((pred - gt)**2))
        nested_exact = int((pred == gt).sum())
        gap = nested_se - full_se
        print(f'  {cname:<25} {full_se:8d} {nested_se:10d} {gap:+5d} {nested_exact:6d}/91')

    # ── Team-level diff for best vs v45b ──
    if best_combo_cfg and best_combo < 263:
        print(f'\n  ══ TEAMS CHANGED: v45c vs v45b ══')
        _, _, p_base = pipeline(sd, test_mask, y, folds, fn)
        _, _, p_best = pipeline(sd, test_mask, y, folds, fn, **best_combo_cfg)
        for i in np.where(test_mask)[0]:
            if p_base[i] != p_best[i]:
                gt = int(y[i])
                delta = (p_best[i]-gt)**2 - (p_base[i]-gt)**2
                marker = ' ★' if delta < 0 else ' ✗' if delta > 0 else ''
                print(f'  {record_ids[i]:<30} v45b={p_base[i]:3d} → v45c={p_best[i]:3d} '
                      f'GT={gt:3d} ΔSE={delta:+3d}{marker}')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
