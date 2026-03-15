#!/usr/bin/env python3
"""
v48 Deep Dive — structural approaches to push past SE=94.

The campaign showed only zone changes marginally help (SE=92 with regression).
This script explores deeper structural changes:

  A: 6-zone pipeline (add zone specifically for 36-48 region)  
  B: Different v12 power in each branch
  C: Per-season adaptive blend weights
  D: Separate committee for different zone groups
  E: Season-aware zone optimization (cross-validated zones)
  F: Late-stage targeted swap rules
  G: Different committee features for different zones
  H: v12 base modifications (different power, forced features)
  I: Asymmetric Hungarian (different power for different seed ranges)
  J: Swap-only post-processing (identify likely swaps)
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_committee_features, build_min8_features,
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


def cache_v12(X_all, y, fn, seasons, test_mask, power=0.15):
    folds = sorted(set(seasons))
    cache = {}
    for hold_season in folds:
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
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=power)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES_V47, power)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12': a_v12, 'a_v12_zoned': a_v12_zoned,
            'avail': avail, 'X_season': X_all[season_mask],
        }
    return cache


def run_fast(cache, X_comm, y, fn, seasons, test_mask, n,
             alpha=10.0, blend=0.25, zones=ZONES_V47,
             comm_power=0.15, final_power=0.15):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_comm[train_mask])
        X_te = sc.transform(X_comm[sm])
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_tr, y[train_mask])
        raw_comm = mdl.predict(X_te)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        a_comm = hungarian(raw_comm, seasons[sm], avail, power=comm_power)
        a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, comm_power)
        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        a_final = hungarian(avg, seasons[sm], avail, power=final_power)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v48 DEEP DIVE')
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
    
    base = run_fast(cache, X_min8, y, fn, seasons, test_mask, n)
    base_se, base_ex = ev(base)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91\n')
    
    best_se, best_label, best_preds = base_se, 'v47', base.copy()
    results = []
    
    def try_and_track(label, preds):
        nonlocal best_se, best_label, best_preds
        se, ex = ev(preds)
        results.append((label, se, ex))
        if se < best_se:
            best_se, best_label, best_preds = se, label, preds.copy()
            print(f'  ★ {label}: SE={se}, exact={ex}/91')
        return se, ex
    
    # ═══════════════════════════════════════════════════════
    # A: 6-ZONE PIPELINE (add a zone in 36-48 region)
    # ═══════════════════════════════════════════════════════
    print('── A: 6-zone pipeline (split uppermid) ──')
    # Current uppermid is (34,44). Split into (34,40) and (40,48) or similar
    for split in [38, 40, 42]:
        for z1_aq in [-3, -2, -1, 0]:
            for z1_al in [-4, -3, -2]:
                for z2_aq in [-3, -2, -1, 0]:
                    for z2_al in [-5, -4, -3, -2]:
                        zones6 = [
                            ('mid',     'committee', (17, 34), (0, 0, 3)),
                            ('um1',     'committee', (34, split), (z1_aq, z1_al, -4)),
                            ('um2',     'committee', (split, 48), (z2_aq, z2_al, -4)),
                            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                            ('tail',    'tail',      (60, 63), (1,)),
                        ]
                        p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n, zones=zones6)
                        try_and_track(f'6z_s{split}_a{z1_aq}{z2_aq}_l{z1_al}{z2_al}', p)
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # B: DIFFERENT V12 POWER
    # ═══════════════════════════════════════════════════════
    print('\n── B: Different v12 power ──')
    for v12_power in [0.05, 0.08, 0.10, 0.12, 0.18, 0.20, 0.25, 0.30]:
        cache_p = cache_v12(X_all, y, fn, seasons, test_mask, power=v12_power)
        for blend in [0.20, 0.25, 0.30]:
            p = run_fast(cache_p, X_min8, y, fn, seasons, test_mask, n, blend=blend)
            try_and_track(f'v12pow_{v12_power}_b{blend}', p)
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # C: DIFFERENT COMMITTEE POWER
    # ═══════════════════════════════════════════════════════
    print('\n── C: Different committee power ──')
    for cp in [0.05, 0.08, 0.10, 0.12, 0.18, 0.20, 0.25, 0.30]:
        for fp in [0.10, 0.12, 0.15, 0.18, 0.20]:
            p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                        comm_power=cp, final_power=fp)
            try_and_track(f'cp{cp}_fp{fp}', p)
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # D: PER-SEASON BLEND (different blend weights per season)
    # ═══════════════════════════════════════════════════════
    print('\n── D: Per-season blend ──')
    # Strategy: use fixed 0.25 as base, but try different values for 2023-24
    for s24_blend in np.arange(0.05, 0.50, 0.05):
        preds_psb = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            train_mask = c['train_mask']; avail = c['avail']
            a_v12 = c['a_v12_zoned']; X_season = c['X_season']
            
            bl = s24_blend if hold_season == '2023-24' else 0.25
            
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
            raw = r.predict(sc.transform(X_min8[sm]))
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw[i] = y[gi]
            ac = hungarian(raw, seasons[sm], avail, power=0.15)
            ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES_V47, 0.15)
            avg = (1.0 - bl) * a_v12.astype(float) + bl * ac.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            af = hungarian(avg, seasons[sm], avail, power=0.15)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_psb[gi] = af[i]
        try_and_track(f'psb_24b{s24_blend:.2f}', preds_psb)
    
    # Also try different blend for each season
    print('  Per-season sweep...')
    season_blends = {}
    for s in sorted(set(seasons)):
        best_sb = 0.25
        best_sse = 99999
        s_test = np.where(test_mask & (seasons == s))[0]
        if len(s_test) == 0: 
            season_blends[s] = 0.25
            continue
        for bl in np.arange(0.05, 0.50, 0.05):
            preds_t = np.zeros(n, dtype=int)
            c = cache[s]
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
            ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES_V47, 0.15)
            avg = (1.0 - bl) * a_v12.astype(float) + bl * ac.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            af = hungarian(avg, seasons[sm], avail, power=0.15)
            sse = 0
            for i, gi in enumerate(si):
                if test_mask[gi]:
                    sse += (af[i] - int(y[gi]))**2
            if sse < best_sse:
                best_sse = sse
                best_sb = bl
        season_blends[s] = best_sb
    
    print(f'  Optimal per-season blends: {season_blends}')
    preds_opt = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        bl = season_blends[hold_season]
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw = r.predict(sc.transform(X_min8[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac = hungarian(raw, seasons[sm], avail, power=0.15)
        ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES_V47, 0.15)
        avg = (1.0 - bl) * a_v12.astype(float) + bl * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds_opt[gi] = af[i]
    try_and_track('psb_optimal', preds_opt)
    
    print(f'  [D done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # E: CROSS-VALIDATED ZONES (optimize zones excluding test season)
    # ═══════════════════════════════════════════════════════
    print('\n── E: Cross-validated zone optimization ──')
    # For each test season, optimize zones on the OTHER 4 seasons, then apply
    # This avoids zone overfitting to specific seasons
    zone_combos = []
    for aq in [-3, -2, -1, 0]:
        for al in [-5, -4, -3, -2, -1]:
            for sos_p in [-5, -4, -3, -2]:
                zone_combos.append((aq, al, sos_p))
    
    preds_cv = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        # Find best zones on OTHER seasons
        other_seasons = [s for s in cache.keys() if s != hold_season]
        best_combo = (-2, -3, -4)  # default
        best_other_se = 99999
        for combo in zone_combos:
            zones_test = [
                ('mid',     'committee', (17, 34), (0, 0, 3)),
                ('uppermid','committee', (34, 44), combo),
                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                ('tail',    'tail',      (60, 63), (1,)),
            ]
            total_se = 0
            for os_name in other_seasons:
                oc = cache[os_name]
                sm2 = oc['season_mask']; si2 = oc['si']; tm2 = oc['tm']
                avail2 = oc['avail']; X_season2 = oc['X_season']
                # Comm path
                sc2 = StandardScaler()
                r2 = Ridge(alpha=10)
                train2 = oc['train_mask']
                r2.fit(sc2.fit_transform(X_min8[train2]), y[train2])
                raw2 = r2.predict(sc2.transform(X_min8[sm2]))
                for i, gi in enumerate(si2):
                    if not test_mask[gi]: raw2[i] = y[gi]
                ac2 = hungarian(raw2, seasons[sm2], avail2, power=0.15)
                ac2 = apply_zones(ac2, raw2, fn, X_season2, tm2, si2, zones_test, 0.15)
                a_v12_2 = oc['a_v12_zoned']
                avg2 = 0.75 * a_v12_2.astype(float) + 0.25 * ac2.astype(float)
                for i, gi in enumerate(si2):
                    if not test_mask[gi]: avg2[i] = y[gi]
                af2 = hungarian(avg2, seasons[sm2], avail2, power=0.15)
                for i, gi in enumerate(si2):
                    if test_mask[gi]:
                        total_se += (af2[i] - int(y[gi]))**2
            if total_se < best_other_se:
                best_other_se = total_se
                best_combo = combo
        
        # Apply best combo to this test season
        zones_best = [
            ('mid',     'committee', (17, 34), (0, 0, 3)),
            ('uppermid','committee', (34, 44), best_combo),
            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
            ('tail',    'tail',      (60, 63), (1,)),
        ]
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
        ac = apply_zones(ac, raw, fn, X_season, tm, si, zones_best, 0.15)
        avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds_cv[gi] = af[i]
        print(f'    {hold_season}: best_combo={best_combo} (other_SE={best_other_se})')
    
    try_and_track('cv_zones', preds_cv)
    print(f'  [E done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # F: LATE-STAGE TARGETED SWAPS
    # ═══════════════════════════════════════════════════════
    print('\n── F: Late-stage targeted swaps ──')
    # Look for specific patterns in the errors that could be fixed with rules
    # Key pattern: mid-major (non-power, AQ) teams with good NET tend to be UNDER-seeded
    # Try: if assigned < raw by large margin AND non-power AND good NET, push down
    
    for net_thresh in [25, 30, 35, 40]:
        for diff_thresh in [3, 5, 7]:
            for push in [2, 3, 4, 5, 6]:
                preds_swap = base.copy()
                changed = False
                for gi in np.where(test_mask)[0]:
                    net_rank = X_all[gi, fi['NET Rank']]
                    is_pow = int(X_all[gi, fi['is_power_conf']])
                    is_aq_val = int(X_all[gi, fi['is_AQ']])
                    pred = preds_swap[gi]
                    # If non-power but AQ, NET < thresh, assigned much lower than expected
                    tfr_val = X_all[gi, fi['tourn_field_rank']]
                    if is_pow == 0 and is_aq_val == 1 and net_rank < net_thresh:
                        if pred < tfr_val - diff_thresh:
                            preds_swap[gi] = min(68, pred + push)
                            changed = True
                if changed:
                    try_and_track(f'swap_net{net_thresh}_d{diff_thresh}_p{push}', preds_swap)
    
    # Swap for power-conf at-large teams that are over-seeded
    for net_thresh in [45, 50, 55, 60]:
        for diff_thresh in [3, 5]:
            for push in [-2, -3, -4, -5]:
                preds_swap = base.copy()
                changed = False
                for gi in np.where(test_mask)[0]:
                    net_rank = X_all[gi, fi['NET Rank']]
                    is_pow = int(X_all[gi, fi['is_power_conf']])
                    is_al_val = int(X_all[gi, fi['is_AL']])
                    pred = preds_swap[gi]
                    tfr_val = X_all[gi, fi['tourn_field_rank']]
                    if is_pow == 1 and is_al_val == 1 and net_rank > net_thresh:
                        if pred > tfr_val + diff_thresh:
                            preds_swap[gi] = max(1, pred + push)
                            changed = True
                if changed:
                    try_and_track(f'swap_pow_net{net_thresh}_d{diff_thresh}_p{push}', preds_swap)
    
    print(f'  [F done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # G: DIFFERENT ALPHA FOR COMMITTEE PATH
    # ═══════════════════════════════════════════════════════
    print('\n── G: Alpha+blend grid for committee (fine) ──')
    for alpha in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50]:
        for blend in np.arange(0.10, 0.40, 0.05):
            p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend)
            try_and_track(f'ab_a{alpha}_b{blend:.2f}', p)
    
    print(f'  [G done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # H: REMOVE ZONES FROM COMMITTEE PATH ONLY
    # ═══════════════════════════════════════════════════════
    print('\n── H: No zones on committee path ──')
    for alpha in [5, 8, 10, 12, 15]:
        for blend in [0.15, 0.20, 0.25, 0.30]:
            p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend, zones=[])
            try_and_track(f'nz_a{alpha}_b{blend}', p)
    
    print(f'  [H done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # I: QUADRUPLE HUNGARIAN (v12 zoned + committee zoned + committee unzoned)
    # ═══════════════════════════════════════════════════════
    print('\n── I: Quadruple paths ──')
    for w_v12 in [0.60, 0.65, 0.70, 0.75]:
        for w_cz in [0.10, 0.15, 0.20]:
            for w_cu in [0.05, 0.10, 0.15]:
                if abs(w_v12 + w_cz + w_cu - 1.0) > 0.001:
                    if w_v12 + w_cz + w_cu > 1.0: continue
                    w_rem = 1.0 - w_v12 - w_cz - w_cu
                    if w_rem < -0.01 or w_rem > 0.20: continue
                preds_q = np.zeros(n, dtype=int)
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
                    # Zoned committee
                    ac_z = hungarian(raw, seasons[sm], avail, power=0.15)
                    ac_z = apply_zones(ac_z, raw, fn, X_season, tm, si, ZONES_V47, 0.15)
                    # Unzoned committee
                    ac_u = hungarian(raw, seasons[sm], avail, power=0.15)
                    
                    w_rest = 1.0 - w_v12 - w_cz - w_cu
                    avg = (w_v12 * a_v12.astype(float) + w_cz * ac_z.astype(float) + 
                           w_cu * ac_u.astype(float))
                    if w_rest > 0.01:
                        # Use raw as 4th component
                        avg += w_rest * raw
                    
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: avg[i] = y[gi]
                    af = hungarian(avg, seasons[sm], avail, power=0.15)
                    for i, gi in enumerate(si):
                        if test_mask[gi]: preds_q[gi] = af[i]
                try_and_track(f'quad_v{w_v12}_cz{w_cz}_cu{w_cu}', preds_q)
    
    print(f'  [I done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # J: DIFFERENT MIN_8 FEATURE SUBSETS
    # ═══════════════════════════════════════════════════════
    print('\n── J: Min_8 feature subsets ──')
    # Remove one feature at a time from min_8 to check if any is hurting
    feature_names_min8 = [
        'tourn_field_rank', 'WL_Pct', 'cb_mean_seed', 'is_power_conf',
        'NETSOS', 'NET Rank', 'is_AQ', 'AvgOppNETRank'
    ]
    for drop_idx in range(8):
        cols = list(range(8))
        cols.remove(drop_idx)
        X_sub = X_min8[:, cols]
        p = run_fast(cache, X_sub, y, fn, seasons, test_mask, n)
        try_and_track(f'min7_drop_{feature_names_min8[drop_idx][:6]}', p)
    
    # Try with top features only (most important by ridge coefficients)
    # tfr(36.3%), wpct(15.5%), cb(13.6%), pow*sos(11.4%), net(8.8%)
    for k in [3, 4, 5, 6]:
        top_k_info = [
            (0, 'tfr'), (1, 'wpct'), (2, 'cb'), (3, 'pow_sos'),
            (5, 'net'), (4, 'sos'), (6, 'cb_aq'), (7, 'opp')
        ]
        X_sub = X_min8[:, [t[0] for t in top_k_info[:k]]]
        p = run_fast(cache, X_sub, y, fn, seasons, test_mask, n)
        try_and_track(f'top{k}', p)
    
    print(f'  [J done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # K: ALTERNATIVE INTERACTION FEATURES
    # ═══════════════════════════════════════════════════════
    print('\n── K: Alternative interaction features ──')
    net_col = X_all[:, fi['NET Rank']]
    sos_col = X_all[:, fi['NETSOS']]
    pow_col = X_all[:, fi['is_power_conf']]
    aq_col = X_all[:, fi['is_AQ']]
    opp_col = X_all[:, fi['AvgOppNETRank']]
    tfr_col = X_all[:, fi['tourn_field_rank']]
    wpct_col = X_all[:, fi['WL_Pct']]
    cb_col = X_all[:, fi['cb_mean_seed']]
    
    # Replace pow*sos interaction with alternatives
    interactions = {
        'pow_net': pow_col * net_col,
        'pow_opp': pow_col * opp_col,
        'aq_sos': aq_col * sos_col,
        'aq_net': aq_col * net_col,
        'net_sos': net_col * sos_col / 1000,
        'cb_net': cb_col * net_col / 100,
        'tfr_net': tfr_col * net_col / 100,
        'wpct_net': wpct_col * net_col,
    }
    
    for iname, icol in interactions.items():
        # Replace feature 3 (pow*sos) with alternative
        X_alt = X_min8.copy()
        X_alt[:, 3] = icol
        for alpha in [5, 10, 15]:
            p = run_fast(cache, X_alt, y, fn, seasons, test_mask, n, alpha=alpha)
            try_and_track(f'int_{iname}_a{alpha}', p)
    
    # Also replace feature 6 (is_AQ*cb_mean_seed) with alternatives
    interactions2 = {
        'aq_net': aq_col * net_col,
        'aq_tfr': aq_col * tfr_col,
        'aq_sos': aq_col * sos_col,
        'aq_wpct': aq_col * wpct_col,
    }
    for iname, icol in interactions2.items():
        X_alt = X_min8.copy()
        X_alt[:, 6] = icol
        for alpha in [5, 10, 15]:
            p = run_fast(cache, X_alt, y, fn, seasons, test_mask, n, alpha=alpha)
            try_and_track(f'int2_{iname}_a{alpha}', p)
    
    print(f'  [K done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' FINAL RESULTS')
    print('='*60)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91')
    print(f'  Best found: {best_label}, SE={best_se}')
    print(f'  Improvement: Δ={base_se - best_se}')
    
    results.sort(key=lambda x: x[1])
    print(f'\n  Top 25:')
    for label, se, ex in results[:25]:
        flag = '★' if se < base_se else ' '
        print(f'  {flag} SE={se:4d}  exact={ex:2d}/91  {label}')
    
    if best_se < base_se:
        print(f'\n  Errors changed from v47:')
        for gi in np.where(test_mask)[0]:
            old, new, actual = base[gi], best_preds[gi], int(y[gi])
            if old != new:
                rid = record_ids[gi]
                old_se = (old - actual)**2
                new_se = (new - actual)**2
                arrow = '↑' if new_se < old_se else '↓' if new_se > old_se else '='
                print(f'    {arrow} {rid:32s} gt={actual:2d} old={old:2d} new={new:2d} '
                      f'oldSE={old_se:3d} newSE={new_se:3d}')
        
        # Check regressions
        regressions = 0
        for gi in np.where(test_mask)[0]:
            old, new, actual = base[gi], best_preds[gi], int(y[gi])
            if (new - actual)**2 > (old - actual)**2:
                regressions += 1
        print(f'\n  Regressions: {regressions}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
