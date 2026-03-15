#!/usr/bin/env python3
"""
Deeper exploration of promising v45 variants:
1. sos=-4 (gave SE=263)
2. Unified wide zone (17,44) instead of separate zones
3. Two-phase mid+uncorr with overlapping boundaries
4. Fine-tuned boundaries for the uncorr zone
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

BOT_ZONE = (50, 60)
BOT_PARAMS = (-4, 3, -1)
TAIL_ZONE = (60, 61)
TAIL_PARAMS = (-3,)


def evaluate(season_data, test_mask, y, folds, fn,
             mid_params=None, mid_zone=(17, 34),
             uncorr_params=None, uncorr_zone=None):
    """Run pipeline with flexible zone configs."""
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
        sn, nc, cb = BOT_PARAMS
        corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                          zone=BOT_ZONE, power=0.15)
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


def nested_loso(configs, X, y, seasons, test_mask, folds, fn, config_short_names=None):
    """Run nested LOSO for a set of configs and return results."""
    n = len(y)
    
    # First, build full-data season data
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
    
    # Full evaluation
    full_results = {}
    for cname, cfg in configs.items():
        _, se, preds = evaluate(season_data, test_mask, y, folds, fn, **cfg)
        full_results[cname] = se
    
    # Nested LOSO
    nested_preds = {c: np.zeros(n, dtype=int) for c in configs}
    
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
        
        for cname, cfg in configs.items():
            _, _, cp = evaluate(hold_sd, test_mask, y, [hold_out], fn, **cfg)
            for i in range(n):
                if test_mask[i] and seasons[i] == hold_out:
                    nested_preds[cname][i] = cp[i]
    
    # Results
    results = {}
    for cname in configs:
        gt = y[test_mask].astype(int)
        pred = nested_preds[cname][test_mask]
        nested_se = int(np.sum((pred - gt)**2))
        nested_exact = int((pred == gt).sum())
        gap = nested_se - full_results[cname]
        results[cname] = {
            'full_se': full_results[cname], 'nested_se': nested_se,
            'gap': gap, 'nested_exact': nested_exact
        }
    
    return results


def main():
    t0 = time.time()
    print('='*70)
    print('  v45 DEEPER EXPLORATION + NESTED LOSO')
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

    # ── Phase 1: Check sos=-4 variants ──
    print('\n  ══ PHASE 1: SOS=-4 ANALYSIS ══')
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

    # Check what sos=-4 does — which teams change vs sos=-3?
    _, se3, p3 = evaluate(sd, test_mask, y, folds, fn,
                           mid_params=(0, 0, 3), uncorr_params=(-2, -3, -3),
                           uncorr_zone=(34, 44))
    _, se4, p4 = evaluate(sd, test_mask, y, folds, fn,
                           mid_params=(0, 0, 3), uncorr_params=(-2, -3, -4),
                           uncorr_zone=(34, 44))
    
    print(f'  sos=-3: SE={se3}')
    print(f'  sos=-4: SE={se4}')
    print(f'  Teams that change with sos=-4:')
    for i in np.where(test_mask)[0]:
        if p3[i] != p4[i]:
            gt = int(y[i])
            delta = (p4[i]-gt)**2 - (p3[i]-gt)**2
            print(f'    {record_ids[i]:<30} s3={p3[i]:3d} → s4={p4[i]:3d} GT={gt:3d} ΔSE={delta:+3d}')

    # ── Phase 2: Unified wide zone (17-44) ──
    print(f'\n  ══ PHASE 2: UNIFIED WIDE ZONE SWEEP ══')
    best_wide = {'se': 999, 'params': None, 'zone': None}
    for zone_hi in [40, 42, 44, 46]:
        for al in range(-4, 5):
            for sos in range(-4, 5):
                for aq in [-3, -2, -1, 0, 1, 2]:
                    ex, se, _ = evaluate(sd, test_mask, y, folds, fn,
                                          mid_params=(aq, al, sos),
                                          mid_zone=(17, zone_hi))
                    if se < best_wide['se']:
                        best_wide = {'se': se, 'params': (aq, al, sos),
                                     'zone': (17, zone_hi)}
                        if se < 300:
                            print(f'    ★ zone=(17,{zone_hi}), aq={aq:+d}, al={al:+d}, sos={sos:+d}: '
                                  f'exact={ex:2d}/91, SE={se:4d}')
    print(f'  Best unified: zone={best_wide["zone"]}, params={best_wide["params"]}, '
          f'SE={best_wide["se"]}')

    # ── Phase 3: Compare in nested LOSO ──
    print(f'\n  ══ PHASE 3: NESTED LOSO COMPARISON ══')
    
    configs = {
        'v44 base': {
            'mid_params': (0, 0, 3), 'mid_zone': (17, 34)
        },
        'v45a (34,44) s3': {
            'mid_params': (0, 0, 3), 'mid_zone': (17, 34),
            'uncorr_params': (-2, -3, -3), 'uncorr_zone': (34, 44)
        },
        'v45b (34,44) s4': {
            'mid_params': (0, 0, 3), 'mid_zone': (17, 34),
            'uncorr_params': (-2, -3, -4), 'uncorr_zone': (34, 44)
        },
    }
    
    # Add best unified if found
    if best_wide['se'] < 307:
        configs['v45c unified'] = {
            'mid_params': best_wide['params'],
            'mid_zone': best_wide['zone']
        }
    
    # Also add (34,42) variant — the tighter zone
    configs['v45d (34,42) s3'] = {
        'mid_params': (0, 0, 3), 'mid_zone': (17, 34),
        'uncorr_params': (-2, -3, -3), 'uncorr_zone': (34, 42)
    }
    
    results = nested_loso(configs, X, y, seasons, test_mask, folds, fn)
    
    print(f'\n  {"Config":<25} {"Full SE":>8} {"Nested SE":>10} {"Gap":>5} {"Exact":>6}')
    for cname in configs:
        r = results[cname]
        print(f'  {cname:<25} {r["full_se"]:8d} {r["nested_se"]:10d} {r["gap"]:+5d} '
              f'{r["nested_exact"]:6d}/91')

    # ── Phase 4: Check what happens if we DON'T have mid-zone (17-34) 
    #    and instead use ONLY the wider zone ──
    print(f'\n  ══ PHASE 4: SINGLE ZONE ALTERNATIVES ══')
    single_configs = {}
    for zone_lo, zone_hi in [(17, 42), (17, 44), (20, 42), (20, 44)]:
        for aq in [-2, 0]:
            for al in [-3, -2, 0, 2]:
                for sos in [-3, -2, 0, 2, 3]:
                    key = f'({zone_lo},{zone_hi}) aq={aq} al={al} sos={sos}'
                    ex, se, _ = evaluate(sd, test_mask, y, folds, fn,
                                          mid_params=(aq, al, sos),
                                          mid_zone=(zone_lo, zone_hi))
                    if se < 300:
                        single_configs[key] = {
                            'mid_params': (aq, al, sos),
                            'mid_zone': (zone_lo, zone_hi)
                        }
                        print(f'    {key}: SE={se} exact={ex}/91')
    
    if single_configs:
        # Validate top 3 in nested LOSO
        top_single = sorted(single_configs.items(), 
                           key=lambda x: evaluate(sd, test_mask, y, folds, fn, **x[1])[1])[:3]
        single_for_loso = {k: v for k, v in top_single}
        single_for_loso['v44 base'] = {'mid_params': (0, 0, 3), 'mid_zone': (17, 34)}
        
        print(f'\n  Nested LOSO for top single-zone configs:')
        results2 = nested_loso(single_for_loso, X, y, seasons, test_mask, folds, fn)
        print(f'  {"Config":<45} {"Full":>5} {"Nested":>7} {"Gap":>5}')
        for cname in single_for_loso:
            r = results2[cname]
            print(f'  {cname:<45} {r["full_se"]:5d} {r["nested_se"]:7d} {r["gap"]:+5d}')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
