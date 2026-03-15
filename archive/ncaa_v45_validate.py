#!/usr/bin/env python3
"""
Validate uncorrected zone (34,44) correction with nested LOSO.
Also analyze exactly which teams change and whether this generalizes.
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

# v44 params
MID_PARAMS = (0, 0, 3)
BOT_ZONE = (50, 60)
BOT_PARAMS = (-4, 3, -1)
TAIL_ZONE = (60, 61)
TAIL_PARAMS = (-3,)

# New uncorr zone
UNCORR_ZONE = (34, 44)
UNCORR_PARAMS = (-2, -3, -3)


def evaluate(season_data, test_mask, y, folds, fn,
             use_uncorr=False, uncorr_zone=UNCORR_ZONE, uncorr_params=UNCORR_PARAMS):
    """Run pipeline, return (exact, SE, preds)."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for hold in folds:
        if hold not in season_data:
            continue
        si, tm, X_s, raw, p1 = season_data[hold]
        assigned = p1.copy()
        
        # Mid-range
        aq, al, sos = MID_PARAMS
        corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                       zone=(17, 34), blend=1.0, power=0.15)
        # Uncorr zone
        if use_uncorr:
            u_aq, u_al, u_sos = uncorr_params
            corr = compute_committee_correction(fn, X_s, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                           zone=uncorr_zone, blend=1.0, power=0.15)
        # Bot-zone
        sn, nc, cb = BOT_PARAMS
        corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                          zone=BOT_ZONE, power=0.15)
        # Tail
        opp, = TAIL_PARAMS
        corr = compute_tail_correction(fn, X_s, opp_rank=opp)
        assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                        zone=TAIL_ZONE, power=0.15)
        
        for i, gi in enumerate(si):
            preds[gi] = assigned[i]
    
    gt = y[test_mask].astype(int)
    pred = preds[test_mask]
    exact = int((pred == gt).sum())
    se = int(np.sum((pred - gt)**2))
    return exact, se, preds


def main():
    t0 = time.time()
    print('='*70)
    print('  VALIDATE UNCORRECTED ZONE (34-44) CORRECTION')
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
    season_data = {}
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
        season_data[hold] = (si, tm, X_s, raw, p1.copy())

    # ── Full evaluation comparison ──
    print('\n  ══ FULL EVALUATION ══')
    _, v44_se, v44_preds = evaluate(season_data, test_mask, y, folds, fn, use_uncorr=False)
    _, v45_se, v45_preds = evaluate(season_data, test_mask, y, folds, fn, use_uncorr=True)
    print(f'  v44 (baseline):    SE={v44_se}')
    print(f'  v45 (+uncorr):     SE={v45_se} ({v45_se - v44_se:+d})')
    
    # ── What teams change? ──
    print(f'\n  ══ TEAMS CHANGED ══')
    print(f'  {"RecordID":<30} {"v44":>4} {"v45":>4} {"GT":>4} {"ΔSE":>5}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*4} {"─"*5}')
    
    total_delta = 0
    for i in np.where(test_mask)[0]:
        v44_p = int(v44_preds[i])
        v45_p = int(v45_preds[i])
        if v44_p != v45_p:
            gt = int(y[i])
            delta_se = (v45_p - gt)**2 - (v44_p - gt)**2
            total_delta += delta_se
            better = ' ★' if delta_se < 0 else ' ✗' if delta_se > 0 else ''
            print(f'  {record_ids[i]:<30} {v44_p:4d} {v45_p:4d} {gt:4d} {delta_se:+5d}{better}')
    print(f'  Total ΔSE: {total_delta:+d}')

    # ── Per-season SE ──
    print(f'\n  ══ PER-SEASON SE ══')
    print(f'  {"Season":<12} {"v44":>6} {"v45":>6} {"Δ":>6} {"Teams":>5}')
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        se44 = int(np.sum((v44_preds[sm] - gt)**2))
        se45 = int(np.sum((v45_preds[sm] - gt)**2))
        print(f'  {s:<12} {se44:6d} {se45:6d} {se45-se44:+6d} {sm.sum():5d}')

    # ── Parameter stability check ──
    print(f'\n  ══ PARAMETER STABILITY ══')
    print(f'  Sweep nearby params to check stability:')
    for u_aq in [-3, -2, -1]:
        for u_al in [-4, -3, -2]:
            for u_sos in [-4, -3, -2]:
                _, se, _ = evaluate(season_data, test_mask, y, folds, fn,
                                     use_uncorr=True,
                                     uncorr_params=(u_aq, u_al, u_sos))
                if se <= v44_se:
                    print(f'    aq={u_aq:+d}, al={u_al:+d}, sos={u_sos:+d}: SE={se:4d} '
                          f'({se-v44_se:+4d})')

    # Also check zone boundary stability
    print(f'\n  Zone boundary stability:')
    for lo in [33, 34, 35, 36]:
        for hi in [42, 43, 44, 45, 46]:
            _, se, _ = evaluate(season_data, test_mask, y, folds, fn,
                                 use_uncorr=True,
                                 uncorr_zone=(lo, hi),
                                 uncorr_params=(-2, -3, -3))
            if se <= v44_se:
                print(f'    zone=({lo},{hi}): SE={se:4d} ({se-v44_se:+4d})')

    # ═══════════════════════════════════════════════════════════════
    # NESTED LOSO
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n  ══ NESTED LOSO VALIDATION ══')
    
    # Configs to compare
    configs = {
        'v44 (baseline)': {'use_uncorr': False, 'uncorr_zone': None, 'uncorr_params': None},
        'v45 (34,44) -2/-3/-3': {'use_uncorr': True, 'uncorr_zone': (34, 44), 'uncorr_params': (-2, -3, -3)},
        'v45 (35,42) -2/-3/-3': {'use_uncorr': True, 'uncorr_zone': (35, 42), 'uncorr_params': (-2, -3, -3)},
        'v45 (34,44) -2/-3/-2': {'use_uncorr': True, 'uncorr_zone': (34, 44), 'uncorr_params': (-2, -3, -2)},
        'v45 (34,44) -1/-3/-3': {'use_uncorr': True, 'uncorr_zone': (34, 44), 'uncorr_params': (-1, -3, -3)},
    }
    
    nested_preds = {c: np.zeros(n, dtype=int) for c in configs}
    
    for hold_out in folds:
        inner_folds = [f for f in folds if f != hold_out]
        inner_test_mask = test_mask.copy()
        for i in range(n):
            if seasons[i] == hold_out:
                inner_test_mask[i] = False
        
        # Build inner season data
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
        
        # Evaluate on inner folds (for selection)
        inner_scores = {}
        for cname, cfg in configs.items():
            _, se, _ = evaluate(inner_sd, inner_test_mask, y, inner_folds, fn,
                                 use_uncorr=cfg['use_uncorr'],
                                 uncorr_zone=cfg['uncorr_zone'] or UNCORR_ZONE,
                                 uncorr_params=cfg['uncorr_params'] or UNCORR_PARAMS)
            inner_scores[cname] = se
        
        winner = min(inner_scores, key=inner_scores.get)
        
        # Evaluate held-out season 
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
        
        # Apply each config to held-out season
        for cname, cfg in configs.items():
            _, _, cp = evaluate(hold_sd, test_mask, y, [hold_out], fn,
                                 use_uncorr=cfg['use_uncorr'],
                                 uncorr_zone=cfg['uncorr_zone'] or UNCORR_ZONE,
                                 uncorr_params=cfg['uncorr_params'] or UNCORR_PARAMS)
            for i in range(n):
                if test_mask[i] and seasons[i] == hold_out:
                    nested_preds[cname][i] = cp[i]
        
        print(f'  {hold_out}: inner winner="{winner}" (SE={inner_scores[winner]})')
        for cname in configs:
            print(f'    {cname}: inner SE={inner_scores[cname]}')

    # Compute nested results
    print(f'\n  {"Config":<30} {"Nested SE":>10} {"Full SE":>8} {"Gap":>5} {"Exact":>6}')
    for cname in configs:
        gt = y[test_mask].astype(int)
        pred = nested_preds[cname][test_mask]
        nested_se = int(np.sum((pred - gt)**2))
        nested_exact = int((pred == gt).sum())
        cfg = configs[cname]
        _, full_se, _ = evaluate(season_data, test_mask, y, folds, fn,
                                  use_uncorr=cfg['use_uncorr'],
                                  uncorr_zone=cfg['uncorr_zone'] or UNCORR_ZONE,
                                  uncorr_params=cfg['uncorr_params'] or UNCORR_PARAMS)
        gap = nested_se - full_se
        print(f'  {cname:<30} {nested_se:10d} {full_se:8d} {gap:+5d} {nested_exact:6d}/91')

    # ── Permutation test ──
    print(f'\n  ══ PERMUTATION TEST (v45 vs v44) ══')
    gt = y[test_mask].astype(int)
    v44_e = v44_preds[test_mask] - gt
    v45_e = v45_preds[test_mask] - gt
    observed = np.sum(v44_e**2) - np.sum(v45_e**2)
    
    n_perm = 2000
    count = 0
    for _ in range(n_perm):
        swap = np.random.random(len(gt)) < 0.5
        pv44 = np.where(swap, v45_e, v44_e)
        pv45 = np.where(swap, v44_e, v45_e)
        d = np.sum(pv44**2) - np.sum(pv45**2)
        if d >= observed:
            count += 1
    p_val = count / n_perm
    print(f'  Observed diff: {observed} (v44 SE - v45 SE)')
    print(f'  p-value: {p_val:.4f}')
    print(f'  {"SIGNIFICANT" if p_val < 0.10 else "NOT SIGNIFICANT"} at p<0.10')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
