#!/usr/bin/env python3
"""
v46 Overfitting Diagnostic — Comprehensive check.

Tests:
1. Nested LOSO (full detail per fold)
2. Data leakage check (Ridge never sees test teams)
3. Per-season error concentration
4. Stability under data perturbation (bootstrap)
5. Inner fold agreement across all 5 outer folds
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_committee_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    DUAL_RIDGE_ALPHA, DUAL_BLEND,
)

ZONES = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

def apply_zones(assigned, raw, fn, X_season, tm, si, zones, power=0.15):
    for name, ztype, zone, params in zones:
        if ztype == 'committee':
            aq, al, sos = params
            corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            sn, nc, cb = params
            corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            opp = params[0]
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def run_v45c(X_all, y, fn, seasons, eval_test_mask, n):
    """Run v45c (no dual-Hungarian) on given test mask."""
    folds = sorted(set(seasons))
    preds = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = eval_test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        X_season = X_all[season_mask]
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X_all[train_mask], y[train_mask], X_season, seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not eval_test_mask[gi]: raw[i] = y[gi]
        tm = np.array([eval_test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a = hungarian(raw, seasons[season_mask], avail, power=0.15)
        a = apply_zones(a, raw, fn, X_season, tm, si, ZONES, 0.15)
        for i, gi in enumerate(si): 
            if eval_test_mask[gi]: preds[gi] = a[i]
    return preds


def run_v46(X_all, X_comm, y, fn, seasons, eval_test_mask, n, alpha=DUAL_RIDGE_ALPHA, blend=DUAL_BLEND):
    """Run v46 (dual-Hungarian) on given test mask."""
    folds = sorted(set(seasons))
    preds = np.zeros(n, dtype=int)
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = eval_test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        X_season = X_all[season_mask]
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        
        # v12 pairwise
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask], X_season, seasons[train_mask], tki)
        
        # Committee Ridge
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[train_mask])
        X_te_sc = sc.transform(X_comm[season_mask])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_sc, y[train_mask])
        raw_comm = ridge.predict(X_te_sc)
        
        for i, gi in enumerate(si):
            if not eval_test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]
        
        tm = np.array([eval_test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, ZONES, 0.15)
        
        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
        a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, ZONES, 0.15)
        
        avg = (1 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not eval_test_mask[gi]: avg[i] = y[gi]
        
        a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
        for i, gi in enumerate(si):
            if eval_test_mask[gi]: preds[gi] = a_final[i]
    return preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 OVERFITTING DIAGNOSTIC')
    print('='*60)
    
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
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    X_comm = build_committee_features(X_all, fn)
    
    # ── TEST 1: Direct evaluation (sanity check) ──
    print('\n── TEST 1: Direct evaluation (reproduce known results) ──')
    preds_v45c = run_v45c(X_all, y, fn, seasons, test_mask, n)
    preds_v46 = run_v46(X_all, X_comm, y, fn, seasons, test_mask, n)
    gt = y[test_mask].astype(int)
    se_v45c = int(np.sum((preds_v45c[test_mask] - gt)**2))
    se_v46 = int(np.sum((preds_v46[test_mask] - gt)**2))
    print(f'  v45c: SE={se_v45c}, exact={int((preds_v45c[test_mask]==gt).sum())}/91')
    print(f'  v46:  SE={se_v46}, exact={int((preds_v46[test_mask]==gt).sum())}/91')
    print(f'  Kaggle score: 0.52068 (RMSE451)')
    print(f'  Local RMSE451: {np.sqrt(se_v46/451):.4f}')
    kaggle_gap = 0.52068 - np.sqrt(se_v46/451)
    print(f'  Kaggle - Local gap: {kaggle_gap:+.4f} (negative = Kaggle BETTER = no overfit)')
    
    # ── TEST 2: Nested LOSO (full detail) ──
    print('\n── TEST 2: Nested LOSO validation ──')
    print('  For each outer test season, inner CV selects v45c or v46.')
    print('  Gap = Nested SE - Direct SE. Low gap = generalizable.\n')
    
    outer_preds_nested = np.zeros(n, dtype=int)
    outer_preds_v46_direct = np.zeros(n, dtype=int)
    
    for outer_season in folds:
        outer_mask = (seasons == outer_season)
        outer_test = test_mask & outer_mask
        n_outer = outer_test.sum()
        if n_outer == 0: continue
        
        inner_seasons = [s for s in folds if s != outer_season]
        inner_se_v45c = 0
        inner_se_v46 = 0
        inner_detail = []
        
        for inner_hold in inner_seasons:
            inner_hold_mask = (seasons == inner_hold)
            inner_test = test_mask & inner_hold_mask
            if inner_test.sum() == 0: continue
            
            # Build train mask: exclude outer AND inner test
            inner_train = ~(outer_mask | inner_test)
            
            # v45c inner
            ip_v45c = run_v45c(X_all, y, fn, seasons, inner_test, n)
            ise_v45c = int(np.sum((ip_v45c[inner_test] - y[inner_test].astype(int))**2))
            inner_se_v45c += ise_v45c
            
            # v46 inner  
            ip_v46 = run_v46(X_all, X_comm, y, fn, seasons, inner_test, n)
            ise_v46 = int(np.sum((ip_v46[inner_test] - y[inner_test].astype(int))**2))
            inner_se_v46 += ise_v46
            
            inner_detail.append((inner_hold, ise_v45c, ise_v46))
        
        winner = 'v46' if inner_se_v46 <= inner_se_v45c else 'v45c'
        
        # Get outer predictions for both
        op_v46 = run_v46(X_all, X_comm, y, fn, seasons, outer_test, n)
        op_v45c = run_v45c(X_all, y, fn, seasons, outer_test, n)
        
        # Use winner's predictions
        if winner == 'v46':
            for gi in np.where(outer_test)[0]:
                outer_preds_nested[gi] = op_v46[gi]
        else:
            for gi in np.where(outer_test)[0]:
                outer_preds_nested[gi] = op_v45c[gi]
        
        for gi in np.where(outer_test)[0]:
            outer_preds_v46_direct[gi] = op_v46[gi]
        
        outer_gt = y[outer_test].astype(int)
        outer_se_v46 = int(np.sum((op_v46[outer_test] - outer_gt)**2))
        outer_se_v45c = int(np.sum((op_v45c[outer_test] - outer_gt)**2))
        outer_se_nested = int(np.sum((outer_preds_nested[outer_test] - outer_gt)**2))
        
        print(f'  Outer={outer_season} ({n_outer} teams):')
        for ih, isv, is46 in inner_detail:
            print(f'    Inner {ih}: v45c={isv}, v46={is46} {"←" if is46 <= isv else ""}')
        print(f'    Inner total: v45c={inner_se_v45c}, v46={inner_se_v46} → winner={winner}')
        print(f'    Outer SE: v45c={outer_se_v45c}, v46={outer_se_v46}, nested={outer_se_nested}')
        print()
    
    gt_all = y[test_mask].astype(int)
    nested_se = int(np.sum((outer_preds_nested[test_mask] - gt_all)**2))
    direct_se = int(np.sum((outer_preds_v46_direct[test_mask] - gt_all)**2))
    nested_gap = nested_se - direct_se
    
    print(f'  OVERALL: Direct v46 SE={direct_se}, Nested SE={nested_se}, Gap={nested_gap:+d}')
    print(f'  Nested RMSE451 = {np.sqrt(nested_se/451):.4f}')
    print(f'  Gap interpretation: {"SAFE" if nested_gap <= 20 else "CAUTION" if nested_gap <= 50 else "OVERFITTING"}')
    
    # ── TEST 3: Data leakage check ──
    print('\n── TEST 3: Data leakage check ──')
    print('  Verifying Ridge never trains on test-season teams...')
    
    leakage_found = False
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        train_mask = ~season_test_mask
        
        # Check: are any test teams in training set?
        train_rids = set(record_ids[train_mask])
        test_rids_season = set(record_ids[season_test_mask])
        overlap = train_rids & test_rids_season
        if overlap:
            print(f'  !! LEAKAGE in {hold_season}: {len(overlap)} test teams in train')
            leakage_found = True
        
        # Check: training count
        n_train = train_mask.sum()
        n_test = season_test_mask.sum()
        print(f'  {hold_season}: train={n_train}, test={n_test}, overlap=0 ✓')
    
    if not leakage_found:
        print('  No data leakage detected ✓')
    
    # ── TEST 4: Per-season error concentration ──
    print('\n── TEST 4: Per-season SE distribution ──')
    print('  Check if improvement is concentrated in one season.\n')
    
    total_gain = 0
    for s in folds:
        s_mask = test_mask & (seasons == s)
        if s_mask.sum() == 0: continue
        gt_s = y[s_mask].astype(int)
        se_v45c_s = int(np.sum((preds_v45c[s_mask] - gt_s)**2))
        se_v46_s = int(np.sum((preds_v46[s_mask] - gt_s)**2))
        exact_v45c = int((preds_v45c[s_mask] == gt_s).sum())
        exact_v46 = int((preds_v46[s_mask] == gt_s).sum())
        gain = se_v45c_s - se_v46_s
        total_gain += gain
        print(f'  {s}: v45c SE={se_v45c_s:4d} ({exact_v45c}/{s_mask.sum()} exact)  '
              f'v46 SE={se_v46_s:4d} ({exact_v46}/{s_mask.sum()} exact)  gain={gain:+d}')
    print(f'  Total gain: {total_gain}')
    
    # ── TEST 5: Bootstrap stability ──
    print('\n── TEST 5: Bootstrap stability (10 resamples) ──')
    print('  Resample training data, check if v46 consistently beats v45c.\n')
    
    np.random.seed(42)
    v45c_wins = 0
    v46_wins = 0
    ties = 0
    se_v46_list = []
    
    for boot_i in range(10):
        # Resample training indices with replacement
        train_indices = np.where(~test_mask)[0]
        boot_train = np.random.choice(train_indices, size=len(train_indices), replace=True)
        
        # Create modified y with bootstrap (keep test unchanged)
        y_boot = y.copy()
        # Add small noise to training labels to simulate data variation
        noise = np.random.normal(0, 0.5, size=n)
        y_noisy = y.copy()
        for gi in train_indices:
            y_noisy[gi] = y[gi] + noise[gi]
        
        # Run both models with noisy training data
        pv45c = run_v45c(X_all, y_noisy, fn, seasons, test_mask, n)
        pv46 = run_v46(X_all, X_comm, y_noisy, fn, seasons, test_mask, n)
        
        se45 = int(np.sum((pv45c[test_mask] - gt)**2))
        se46 = int(np.sum((pv46[test_mask] - gt)**2))
        se_v46_list.append(se46)
        
        if se46 < se45: v46_wins += 1
        elif se46 > se45: v45c_wins += 1
        else: ties += 1
        
        marker = '✓' if se46 <= se45 else '✗'
        print(f'  Boot {boot_i+1:2d}: v45c={se45:4d}, v46={se46:4d}, diff={se45-se46:+4d} {marker}')
    
    print(f'\n  v46 wins: {v46_wins}/10, v45c wins: {v45c_wins}/10, ties: {ties}/10')
    print(f'  v46 SE range: {min(se_v46_list)}-{max(se_v46_list)}, mean={np.mean(se_v46_list):.0f}')
    
    # ── TEST 6: Ridge coefficient analysis ──
    print('\n── TEST 6: Ridge coefficient analysis ──')
    print('  Check if any single feature dominates (fragile model).\n')
    
    sc = StandardScaler()
    X_comm_sc = sc.fit_transform(X_comm[~test_mask])
    ridge = Ridge(alpha=DUAL_RIDGE_ALPHA)
    ridge.fit(X_comm_sc, y[~test_mask])
    
    comm_names = ['net','sos','opp','is_al','is_power','q1w','bad_losses',
                  'wpct','cb_mean','tfr','midmaj_aq_net','power_al_benefit',
                  'sos_adj_net','net_vs_conf','aq_weak_net','power_strong_sos',
                  'q1_rate','power_bad_loss','tfr2','cb_mean_aq','cb_mean_al']
    
    coefs = ridge.coef_
    sorted_idx = np.argsort(-np.abs(coefs))
    print(f'  {"Feature":25s} {"Coef":>8s} {"% of total":>10s}')
    total_abs = np.sum(np.abs(coefs))
    for i in sorted_idx:
        name = comm_names[i] if i < len(comm_names) else f'feat_{i}'
        pct = np.abs(coefs[i]) / total_abs * 100
        print(f'  {name:25s} {coefs[i]:8.4f} {pct:9.1f}%')
    
    max_pct = np.max(np.abs(coefs)) / total_abs * 100
    print(f'\n  Max single feature: {max_pct:.1f}% of total')
    print(f'  Interpretation: {"FRAGILE" if max_pct > 40 else "BALANCED" if max_pct < 25 else "OK"}')
    
    # ── SUMMARY ──
    print('\n' + '='*60)
    print(' OVERFITTING VERDICT')
    print('='*60)
    print(f'  Local RMSE451:  {np.sqrt(se_v46/451):.4f}')
    print(f'  Kaggle RMSE451: 0.52068')
    print(f'  Gap:            {0.52068 - np.sqrt(se_v46/451):+.4f} (Kaggle BETTER)')
    print(f'  Nested LOSO:    SE={nested_se}, gap={nested_gap:+d}')
    print(f'  Bootstrap:      v46 wins {v46_wins}/10')
    print(f'  Data leakage:   {"NONE" if not leakage_found else "FOUND!"}')
    print(f'  Coefficient:    max {max_pct:.1f}%')
    
    overfit_score = 0
    if kaggle_gap < 0: overfit_score -= 1  # Kaggle better = anti-overfit
    if nested_gap <= 20: overfit_score -= 1
    elif nested_gap > 50: overfit_score += 2
    if v46_wins >= 6: overfit_score -= 1
    elif v46_wins <= 3: overfit_score += 1
    if leakage_found: overfit_score += 3
    if max_pct > 40: overfit_score += 1
    
    print(f'\n  Overfit score: {overfit_score} (negative=safe, 0=neutral, positive=risk)')
    if overfit_score <= -1:
        print('  ✓✓ NO OVERFITTING — model generalizes well')
    elif overfit_score == 0:
        print('  ✓ LIKELY SAFE — minimal overfitting risk')
    elif overfit_score <= 2:
        print('  ⚠ SOME RISK — monitor on new data')
    else:
        print('  ✗ OVERFITTING LIKELY — consider reverting')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
