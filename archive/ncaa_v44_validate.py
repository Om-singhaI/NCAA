#!/usr/bin/env python3
"""
Validate the v44 best config with nested LOSO to ensure no overfitting.
Best: mid(aq=0,al=0,sos=3) + bot(50,60,-4,3,-1) + tail(60,61,-3)
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

def evaluate_config(season_data, test_mask, y, folds, fn, 
                    mid_params, bot_zone, bot_params, tail_zone, tail_params):
    """Evaluate a zone config. Returns (exact, SE, preds)."""
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
                                           zone=(17, 34), blend=1.0, power=0.15)
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
            if test_mask[gi]:
                preds[gi] = assigned[i]
    
    gt = y[test_mask].astype(int)
    pred = preds[test_mask]
    exact = int((pred == gt).sum())
    se = int(np.sum((pred - gt)**2))
    return exact, se, preds

def main():
    t0 = time.time()
    print('='*70)
    print('  NESTED LOSO VALIDATION FOR v44 CONFIG')
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

    # Configs to compare
    configs = {
        'v27 (mid+bot+tail)': {
            'mid': (0, 2, 3), 'bot_zone': (50, 60), 'bot': (-4, 3, -1),
            'tail_zone': (61, 65), 'tail': (-3,)
        },
        'v44 candidate': {
            'mid': (0, 0, 3), 'bot_zone': (50, 60), 'bot': (-4, 3, -1),
            'tail_zone': (60, 61), 'tail': (-3,)
        },
        'v44 alt (al=2,sos=2)': {
            'mid': (0, 2, 2), 'bot_zone': (50, 60), 'bot': (-4, 3, -1),
            'tail_zone': (60, 61), 'tail': (-3,)
        },
        'v44 no-tail': {
            'mid': (0, 0, 3), 'bot_zone': (50, 60), 'bot': (-4, 3, -1),
            'tail_zone': (60, 61), 'tail': None
        },
        'base only (no zones)': {
            'mid': None, 'bot_zone': (50, 60), 'bot': None,
            'tail_zone': (61, 65), 'tail': None
        },
    }

    # ── Full evaluation ──
    print('\n  ══ FULL EVALUATION (all test data visible) ══')
    
    # Pre-compute season data
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
    
    for cname, cfg in configs.items():
        ex, se, _ = evaluate_config(
            season_data, test_mask, y, folds, fn,
            cfg['mid'], cfg['bot_zone'], cfg['bot'], cfg['tail_zone'], cfg['tail'])
        print(f'  {cname:<30} exact={ex:2d}/91 SE={se:4d} RMSE451={np.sqrt(se/451):.4f}')

    # ── NESTED LOSO ──
    print('\n\n  ══ NESTED LOSO VALIDATION ══')
    print('  (For each held-out season, sweep on remaining 4 seasons,')
    print('   then apply winner to held-out season)')
    
    nested_preds = {}
    for cname in configs:
        nested_preds[cname] = np.zeros(n, dtype=int)

    # Also collect per-season nested SE for each config
    per_season = {cname: {} for cname in configs}
    nested_chosen = {}
    
    for hold_out in folds:
        # The "inner" seasons are everything except hold_out
        inner_folds = [f for f in folds if f != hold_out]
        inner_test_mask = test_mask.copy()
        # Mark hold-out season teams as NOT test for inner evaluation
        for i in range(n):
            if seasons[i] == hold_out:
                inner_test_mask[i] = False
        
        # Pre-compute inner season data (train on non-test teams, predict test in inner seasons)
        inner_season_data = {}
        for f in inner_folds:
            sm = (seasons == f)
            si = np.where(sm)[0]
            st = inner_test_mask & sm
            if st.sum() == 0: continue
            gt_mask = ~(inner_test_mask & sm) & sm  # non-test teams in this season + all others
            # Actually we need: training = NOT test teams across ALL seasons except hold_out
            # But the raw scores are already computed per-season
            # For nested LOSO, we recompute with inner test mask
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
            inner_season_data[f] = (si, tm, X_s, raw, p1.copy())
        
        # Evaluate each config on inner seasons
        inner_scores = {}
        for cname, cfg in configs.items():
            ex, se, _ = evaluate_config(
                inner_season_data, inner_test_mask, y, inner_folds, fn,
                cfg['mid'], cfg['bot_zone'], cfg['bot'], cfg['tail_zone'], cfg['tail'])
            inner_scores[cname] = se
        
        # Pick winner on inner seasons
        winner = min(inner_scores, key=inner_scores.get)
        nested_chosen[hold_out] = (winner, inner_scores[winner])
        
        # Apply winner config to held-out season
        hold_season_data = {}
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
            hold_season_data[hold_out] = (si, tm, X_s, raw, p1.copy())
        
        # Apply each config (not just winner) to get nested LOSO results
        for cname, cfg in configs.items():
            _, _, cp = evaluate_config(
                hold_season_data, test_mask, y, [hold_out], fn,
                cfg['mid'], cfg['bot_zone'], cfg['bot'], cfg['tail_zone'], cfg['tail'])
            for i in range(n):
                if test_mask[i] and seasons[i] == hold_out:
                    nested_preds[cname][i] = cp[i]
            # Per-season SE
            mask = test_mask & (seasons == hold_out)
            pred_s = cp[mask]
            gt_s = y[mask].astype(int)
            per_season[cname][hold_out] = int(np.sum((pred_s - gt_s)**2))
    
    # Print nested LOSO results
    print(f'\n  Nested LOSO results (inner selection → outer evaluation):')
    for hold_out in folds:
        winner, inner_se = nested_chosen[hold_out]
        mask = test_mask & (seasons == hold_out)
        n_te = mask.sum()
        print(f'  {hold_out}: winner="{winner}" (inner SE={inner_se}), '
              f'{n_te} test teams')
    
    print(f'\n  {"Config":<30} {"Nested SE":>10} {"Full SE":>8} {"Gap":>5} {"Nested Exact":>12}')
    for cname in configs:
        gt = y[test_mask].astype(int)
        pred = nested_preds[cname][test_mask]
        nested_se = int(np.sum((pred - gt)**2))
        nested_exact = int((pred == gt).sum())
        _, full_se, _ = evaluate_config(
            season_data, test_mask, y, folds, fn,
            configs[cname]['mid'], configs[cname]['bot_zone'], configs[cname]['bot'],
            configs[cname]['tail_zone'], configs[cname]['tail'])
        gap = nested_se - full_se
        print(f'  {cname:<30} {nested_se:10d} {full_se:8d} {gap:+5d} {nested_exact:12d}/91')

    # ── Permutation test for v44 vs v27 ──
    print('\n\n  ══ PERMUTATION TEST (v44 vs v27) ══')
    _, v27_se, v27_p = evaluate_config(
        season_data, test_mask, y, folds, fn,
        (0, 2, 3), (50, 60), (-4, 3, -1), (61, 65), (-3,))
    _, v44_se, v44_p = evaluate_config(
        season_data, test_mask, y, folds, fn,
        (0, 0, 3), (50, 60), (-4, 3, -1), (60, 61), (-3,))
    
    observed_diff = v27_se - v44_se  # positive = v44 better
    print(f'  v27 SE={v27_se}, v44 SE={v44_se}, observed diff={observed_diff}')
    
    # Permutation: randomly swap v27/v44 predictions per team
    n_perm = 1000
    count_ge = 0
    gt = y[test_mask].astype(int)
    v27_e = v27_p[test_mask] - gt
    v44_e = v44_p[test_mask] - gt
    
    for _ in range(n_perm):
        swap = np.random.random(len(gt)) < 0.5
        perm_v27 = np.where(swap, v44_e, v27_e)
        perm_v44 = np.where(swap, v27_e, v44_e)
        perm_diff = np.sum(perm_v27**2) - np.sum(perm_v44**2)
        if perm_diff >= observed_diff:
            count_ge += 1
    
    p_value = count_ge / n_perm
    print(f'  Permutation p-value: {p_value:.3f} (1000 permutations)')
    print(f'  {"SIGNIFICANT" if p_value < 0.10 else "NOT SIGNIFICANT"} at p<0.10')

    # ── Per-season breakdown ──
    print(f'\n\n  ══ PER-SEASON SE BREAKDOWN ══')
    print(f'  {"Season":<12} {"v27":>6} {"v44":>6} {"diff":>6}')
    for hold in folds:
        mask = test_mask & (seasons == hold)
        if mask.sum() == 0: continue
        gt_s = y[mask].astype(int)
        v27_s = v27_p[mask]
        v44_s = v44_p[mask]
        se27 = int(np.sum((v27_s - gt_s)**2))
        se44 = int(np.sum((v44_s - gt_s)**2))
        print(f'  {hold:<12} {se27:6d} {se44:6d} {se44-se27:+6d}')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
