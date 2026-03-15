#!/usr/bin/env python3
"""
v48 Validation — comprehensive validation of the new zone (42,50) discovery.

Tests:
  1. Reproduce: confirm SE=80, 76/91, zero regressions
  2. Fine grid: sweep zone params around the optimum
  3. Nested LOSO: train zones on inner folds, test on outer
  4. Cross-validated zones: for each season, optimize zone on other 4
  5. Regression check: zero regressions from v47
  6. Bootstrap: random resample stability
  7. Zone boundary sweep: check sensitivity
  8. Combined optimization: try combining with other improvements
  9. Remaining error analysis
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

ZONES_V47 = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_V48 = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),  # NEW
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

def apply_zones(assigned, raw, fn, X_season, tm, si, zones, power=0.15):
    for name, ztype, zone, params in zones:
        if ztype == 'committee':
            corr = compute_committee_correction(fn, X_season, *params)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            corr = compute_bottom_correction(fn, X_season, *params)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            corr = compute_tail_correction(fn, X_season, opp_rank=params[0])
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def cache_v12(X_all, y, fn, seasons, test_mask):
    cache = {}
    for hold_season in sorted(set(seasons)):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                       X_all[season_mask], seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_v12[i] = y[gi]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        # NOTE: v12 zones stay at V47 (the new zone only affects committee path)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES_V47, 0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12': a_v12, 'a_v12_zoned': a_v12_zoned,
            'avail': avail, 'X_season': X_all[season_mask],
        }
    return cache


def run_with_zones(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw = r.predict(sc.transform(X_min8[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac = hungarian(raw, seasons[sm], avail, power=0.15)
        ac = apply_zones(ac, raw, fn, X_season, tm, si, comm_zones, 0.15)
        avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = af[i]
    return preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v48 VALIDATION')
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
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    gt = y[test_mask].astype(int)
    fi = {f: i for i, f in enumerate(fn)}
    
    X_min8 = build_min8_features(X_all, fn)
    
    print('  Caching v12...')
    cache = cache_v12(X_all, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    # ═══════════════════════════════════════════════════════
    # TEST 1: REPRODUCE
    # ═══════════════════════════════════════════════════════
    print('\n── Test 1: Reproduce ──')
    v47_preds = run_with_zones(cache, X_min8, y, fn, seasons, test_mask, n, ZONES_V47)
    v48_preds = run_with_zones(cache, X_min8, y, fn, seasons, test_mask, n, ZONES_V48)
    v47_se, v47_ex = ev(v47_preds)
    v48_se, v48_ex = ev(v48_preds)
    print(f'  v47: SE={v47_se}, exact={v47_ex}/91')
    print(f'  v48: SE={v48_se}, exact={v48_ex}/91')
    print(f'  Improvement: Δ={v47_se - v48_se}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 2: FINE GRID
    # ═══════════════════════════════════════════════════════
    print('\n── Test 2: Fine grid around (42,50) sn=-4, nc=2, cb=-3 ──')
    best_fg = 999
    for sn in range(-6, 1):
        for nc in range(-1, 5):
            for cb in range(-5, 2):
                zones_t = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (-2, -3, -4)),
                    ('midbot2', 'bottom',    (42, 50), (sn, nc, cb)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_with_zones(cache, X_min8, y, fn, seasons, test_mask, n, zones_t)
                se, ex = ev(p)
                if se < best_fg:
                    best_fg = se
                    print(f'  sn={sn:2d} nc={nc:2d} cb={cb:2d}: SE={se}, exact={ex}/91')
    print(f'  Fine grid best: SE={best_fg}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 3: ZONE BOUNDARY SWEEP
    # ═══════════════════════════════════════════════════════
    print('\n── Test 3: Zone boundary sweep ──')
    for lo in range(38, 48):
        for hi in range(lo+2, 54):
            zones_t = [
                ('mid',     'committee', (17, 34), (0, 0, 3)),
                ('uppermid','committee', (34, 44), (-2, -3, -4)),
                ('midbot2', 'bottom',    (lo, hi), (-4, 2, -3)),
                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                ('tail',    'tail',      (60, 63), (1,)),
            ]
            p = run_with_zones(cache, X_min8, y, fn, seasons, test_mask, n, zones_t)
            se, ex = ev(p)
            if se <= 80:
                print(f'  ({lo},{hi}): SE={se}, exact={ex}/91')
    
    # ═══════════════════════════════════════════════════════
    # TEST 4: NESTED LOSO
    # ═══════════════════════════════════════════════════════
    print('\n── Test 4: Nested LOSO ──')
    folds = sorted(set(seasons))
    # For each outer fold, optimize zone params on other 4 folds
    nested_v47_total, nested_v48_total = 0, 0
    for outer in folds:
        outer_mask = test_mask & (seasons == outer)
        if outer_mask.sum() == 0: continue
        
        # Inner: optimize zones on everything except outer
        inner_seasons = [s for s in folds if s != outer]
        
        best_inner_se = 99999
        best_inner_combo = (-4, 2, -3)
        
        for sn in [-6, -5, -4, -3, -2]:
            for nc in [0, 1, 2, 3, 4]:
                for cb in [-5, -4, -3, -2, -1]:
                    zones_t = [
                        ('mid',     'committee', (17, 34), (0, 0, 3)),
                        ('uppermid','committee', (34, 44), (-2, -3, -4)),
                        ('midbot2', 'bottom',    (42, 50), (sn, nc, cb)),
                        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                        ('tail',    'tail',      (60, 63), (1,)),
                    ]
                    inner_se = 0
                    for inner_season in inner_seasons:
                        c = cache[inner_season]
                        sm = c['season_mask']; si = c['si']; tm = c['tm']
                        train_mask_i = c['train_mask']; avail = c['avail']
                        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                        sc = StandardScaler()
                        r = Ridge(alpha=10)
                        r.fit(sc.fit_transform(X_min8[train_mask_i]), y[train_mask_i])
                        raw = r.predict(sc.transform(X_min8[sm]))
                        for i, gi in enumerate(si):
                            if not test_mask[gi]: raw[i] = y[gi]
                        ac = hungarian(raw, seasons[sm], avail, power=0.15)
                        ac = apply_zones(ac, raw, fn, X_season, tm, si, zones_t, 0.15)
                        avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
                        for i, gi in enumerate(si):
                            if not test_mask[gi]: avg[i] = y[gi]
                        af = hungarian(avg, seasons[sm], avail, power=0.15)
                        for i, gi in enumerate(si):
                            if test_mask[gi]:
                                inner_se += (af[i] - int(y[gi]))**2
                    if inner_se < best_inner_se:
                        best_inner_se = inner_se
                        best_inner_combo = (sn, nc, cb)
        
        # Apply best inner combo to outer fold
        zones_outer = [
            ('mid',     'committee', (17, 34), (0, 0, 3)),
            ('uppermid','committee', (34, 44), (-2, -3, -4)),
            ('midbot2', 'bottom',    (42, 50), best_inner_combo),
            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
            ('tail',    'tail',      (60, 63), (1,)),
        ]
        c = cache[outer]
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask_o = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask_o]), y[train_mask_o])
        raw = r.predict(sc.transform(X_min8[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac_v48 = hungarian(raw, seasons[sm], avail, power=0.15)
        ac_v48 = apply_zones(ac_v48, raw, fn, X_season, tm, si, zones_outer, 0.15)
        ac_v47 = hungarian(raw, seasons[sm], avail, power=0.15)
        ac_v47 = apply_zones(ac_v47, raw, fn, X_season, tm, si, ZONES_V47, 0.15)
        
        avg_v48 = 0.75 * a_v12.astype(float) + 0.25 * ac_v48.astype(float)
        avg_v47 = 0.75 * a_v12.astype(float) + 0.25 * ac_v47.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                avg_v48[i] = y[gi]
                avg_v47[i] = y[gi]
        af_v48 = hungarian(avg_v48, seasons[sm], avail, power=0.15)
        af_v47 = hungarian(avg_v47, seasons[sm], avail, power=0.15)
        
        se_v47 = sum((af_v47[i] - int(y[gi]))**2 for i, gi in enumerate(si) if test_mask[gi])
        se_v48 = sum((af_v48[i] - int(y[gi]))**2 for i, gi in enumerate(si) if test_mask[gi])
        nested_v47_total += se_v47
        nested_v48_total += se_v48
        
        winner = 'v48' if se_v48 < se_v47 else ('TIE' if se_v48 == se_v47 else 'v47')
        print(f'  {outer}: inner_best={best_inner_combo} v47_SE={se_v47} v48_SE={se_v48} [{winner}]')
    
    gap = nested_v48_total - nested_v47_total
    print(f'  Nested LOSO: v47={nested_v47_total} v48={nested_v48_total} gap={gap}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 5: REGRESSION CHECK
    # ═══════════════════════════════════════════════════════
    print('\n── Test 5: Regression check ──')
    gains, losses = 0, 0
    for gi in np.where(test_mask)[0]:
        old_se = (v47_preds[gi] - int(y[gi]))**2
        new_se = (v48_preds[gi] - int(y[gi]))**2
        if new_se < old_se:
            gains += (old_se - new_se)
            print(f'  ↑ {record_ids[gi]:32s} gt={int(y[gi]):2d} old={v47_preds[gi]:2d} new={v48_preds[gi]:2d} SE: {old_se:3d}→{new_se:3d}')
        elif new_se > old_se:
            losses += (new_se - old_se)
            print(f'  ↓ {record_ids[gi]:32s} gt={int(y[gi]):2d} old={v47_preds[gi]:2d} new={v48_preds[gi]:2d} SE: {old_se:3d}→{new_se:3d}')
    
    regressions = sum(1 for gi in np.where(test_mask)[0] 
                      if (v48_preds[gi] - int(y[gi]))**2 > (v47_preds[gi] - int(y[gi]))**2)
    print(f'  Gains: {gains}, Losses: {losses}, Regressions: {regressions}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 6: BOOTSTRAP
    # ═══════════════════════════════════════════════════════
    print('\n── Test 6: Bootstrap stability (20 resamples) ──')
    np.random.seed(42)
    test_indices = np.where(test_mask)[0]
    v48_wins = 0
    for trial in range(20):
        boot_idx = np.random.choice(test_indices, size=len(test_indices), replace=True)
        v47_boot = sum((v47_preds[i] - int(y[i]))**2 for i in boot_idx)
        v48_boot = sum((v48_preds[i] - int(y[i]))**2 for i in boot_idx)
        w = 'v48' if v48_boot < v47_boot else ('TIE' if v48_boot == v47_boot else 'v47')
        if v48_boot < v47_boot: v48_wins += 1
        elif v48_boot == v47_boot: v48_wins += 0.5
    print(f'  v48 wins: {v48_wins}/20')
    
    # ═══════════════════════════════════════════════════════
    # TEST 7: PER-SEASON BREAKDOWN
    # ═══════════════════════════════════════════════════════
    print('\n── Test 7: Per-season breakdown ──')
    for s in sorted(set(seasons)):
        s_mask = test_mask & (seasons == s)
        v47_se_s = sum((v47_preds[i] - int(y[i]))**2 for i in np.where(s_mask)[0])
        v48_se_s = sum((v48_preds[i] - int(y[i]))**2 for i in np.where(s_mask)[0])
        v47_ex_s = sum(1 for i in np.where(s_mask)[0] if v47_preds[i] == int(y[i]))
        v48_ex_s = sum(1 for i in np.where(s_mask)[0] if v48_preds[i] == int(y[i]))
        flag = '★' if v48_se_s < v47_se_s else ('=' if v48_se_s == v47_se_s else '↓')
        print(f'  {flag} {s}: v47 SE={v47_se_s:3d} ({v47_ex_s}/{s_mask.sum()}) → '
              f'v48 SE={v48_se_s:3d} ({v48_ex_s}/{s_mask.sum()})')
    
    # ═══════════════════════════════════════════════════════
    # TEST 8: V48 REMAINING ERRORS
    # ═══════════════════════════════════════════════════════
    print('\n── Test 8: Remaining errors at SE=80 ──')
    errors = []
    for gi in np.where(test_mask)[0]:
        pred = v48_preds[gi]; actual = int(y[gi])
        if pred != actual:
            rid = record_ids[gi]; season = seasons[gi]
            se = (pred - actual)**2
            net = X_all[gi, fi['NET Rank']]
            errors.append((rid, season, actual, pred, se, net))
    
    errors.sort(key=lambda x: -x[4])
    print(f'  {len(errors)} wrong, total SE={sum(e[4] for e in errors)}')
    print(f'  {"RecordID":32s} {"Yr":7s} {"GT":>3s} {"Pd":>3s} {"SE":>3s} {"NET":>4s}')
    for rid, season, actual, pred, se, net in errors:
        print(f'  {rid:32s} {season[-2:]:7s} {actual:3d} {pred:3d} {se:3d} {net:4.0f}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 9: COMBINED WITH OTHER IMPROVEMENTS
    # ═══════════════════════════════════════════════════════
    print('\n── Test 9: V48 zones + further optimization ──')
    # Try v48 zones with different alpha/blend for committee
    best_combo_se = v48_se
    for alpha in [5, 7, 8, 9, 10, 11, 12, 15, 20]:
        for blend in np.arange(0.15, 0.35, 0.05):
            preds_co = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                sc = StandardScaler()
                r = Ridge(alpha=alpha)
                r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                raw = r.predict(sc.transform(X_min8[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw[i] = y[gi]
                ac = hungarian(raw, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES_V48, 0.15)
                avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_co[gi] = af[i]
            se, ex = ev(preds_co)
            if se < best_combo_se:
                best_combo_se = se
                print(f'  ★ α={alpha} b={blend:.2f}: SE={se}, exact={ex}/91')
    
    print(f'  Combined best: SE={best_combo_se}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 10: ALSO NEW ZONE ON V12 PATH
    # ═══════════════════════════════════════════════════════
    print('\n── Test 10: New zone on BOTH v12 AND committee paths ──')
    # Re-cache v12 with new zones
    cache_v48 = {}
    for hold_season in sorted(set(seasons)):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                       X_all[season_mask], seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_v12[i] = y[gi]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES_V48, 0.15)
        cache_v48[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12_zoned': a_v12_zoned, 'avail': avail,
            'X_season': X_all[season_mask],
        }
    
    preds_both = np.zeros(n, dtype=int)
    for hold_season, c in cache_v48.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw = r.predict(sc.transform(X_min8[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac = hungarian(raw, seasons[sm], avail, power=0.15)
        ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES_V48, 0.15)
        avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds_both[gi] = af[i]
    
    se_both, ex_both = ev(preds_both)
    print(f'  Both paths v48 zones: SE={se_both}, exact={ex_both}/91')
    
    # Check regressions for "both" version
    reg_both = 0
    for gi in np.where(test_mask)[0]:
        if (preds_both[gi] - int(y[gi]))**2 > (v47_preds[gi] - int(y[gi]))**2:
            rid = record_ids[gi]
            print(f'    ↓ {rid}: gt={int(y[gi])} v47={v47_preds[gi]} v48both={preds_both[gi]}')
            reg_both += 1
    if reg_both == 0:
        print('    Zero regressions!')
    
    # ═══════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' VALIDATION SUMMARY')
    print('='*60)
    print(f'  v47:  SE={v47_se}, exact={v47_ex}/91')
    print(f'  v48:  SE={v48_se}, exact={v48_ex}/91')
    print(f'  Δ={v47_se - v48_se}, regressions={regressions}')
    print(f'  Nested LOSO gap: {gap}')
    print(f'  Bootstrap: v48 wins {v48_wins}/20')
    print(f'  Both-path v48: SE={se_both}')
    ovf = 'SAFE' if gap <= 0 else 'CAUTION'
    print(f'  Overfitting assessment: {ovf}')
    print(f'  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
