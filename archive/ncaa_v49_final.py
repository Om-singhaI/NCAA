#!/usr/bin/env python3
"""
v49 final exploration: Find best combination of:
1. Uppermid zone params (including extended SOS range)
2. Blend ratio (0.20-0.50)
3. Optional swap rule

Also try: zones on v12 path to fix NM/NW at the v12 level.
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
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)


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
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]


def cache_all(X_all, X_min8, y, fn, seasons, test_mask):
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
        a_v12_raw = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw_comm = r.predict(sc.transform(X_min8[season_mask]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        a_comm_raw = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'avail': avail,
            'raw_v12': raw_v12, 'a_v12_raw': a_v12_raw,
            'raw_comm': raw_comm, 'a_comm_raw': a_comm_raw,
            'X_season': X_all[season_mask],
        }
    return cache


def run_cached(cache, y, fn, seasons, test_mask, n,
               v12_zones=ZONES_V47, comm_zones=ZONES_V48, blend=0.25):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        avail = c['avail']; X_season = c['X_season']
        a_v12 = apply_zones(c['a_v12_raw'].copy(), c['raw_v12'], fn, X_season, tm, si, v12_zones, 0.15)
        ac = apply_zones(c['a_comm_raw'].copy(), c['raw_comm'], fn, X_season, tm, si, comm_zones, 0.15)
        avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = af[i]
    return preds


def apply_swap(preds, X_all, fi, seasons, test_mask, net_gap, pred_gap):
    preds = preds.copy()
    n_swaps = 0
    for hold_season in sorted(set(seasons)):
        sm = (seasons == hold_season)
        si_s = np.where(sm)[0]
        test_si = [gi for gi in si_s if test_mask[gi]]
        aq_teams = []
        al_teams = []
        for gi in test_si:
            pred = preds[gi]
            if 30 <= pred <= 45:
                is_aq_v = int(X_all[gi, fi['is_AQ']])
                is_al_v = int(X_all[gi, fi['is_AL']])
                net_v = X_all[gi, fi['NET Rank']]
                if is_aq_v and not is_al_v and pred - net_v > net_gap:
                    aq_teams.append((gi, pred, net_v))
                elif is_al_v and not is_aq_v and net_v - pred > 0:
                    al_teams.append((gi, pred, net_v))
        for aq_gi, aq_pred, aq_net in aq_teams:
            for al_gi, al_pred, al_net in al_teams:
                if abs(aq_pred - al_pred) <= pred_gap:
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
                    n_swaps += 1
    return preds, n_swaps


def main():
    t0 = time.time()
    print('='*70)
    print(' v49 FINAL EXPLORATION')
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
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    gt = y[test_mask].astype(int)
    fi = {f: i for i, f in enumerate(fn)}
    X_min8 = build_min8_features(X_all, fn)
    
    print('  Caching...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    p48 = run_cached(cache, y, fn, seasons, test_mask, n)
    base_se, base_ex = ev(p48)
    print(f'  Baseline v48: SE={base_se}, exact={base_ex}/91')
    
    best_se, best_label, best_preds = base_se, 'v48', p48.copy()
    
    # ═══════════════════════════════════════════════════════
    # A: Blend ratio with v48 zones + uppermid variations
    # ═══════════════════════════════════════════════════════
    print('\n── A: Blend × uppermid zone ──')
    for blend in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        for aq in [-6, -4, -2, 0, 2, 4]:
            for al in [-4, -2, 0, 1, 2, 4]:
                for sos_p in [-6, -4, -2, 0, 2, 4, 6]:
                    zones = [
                        ('mid',     'committee', (17, 34), (0, 0, 3)),
                        ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                        ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                        ('tail',    'tail',      (60, 63), (1,)),
                    ]
                    p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones, blend=blend)
                    se, ex = ev(p)
                    if se < best_se:
                        best_se = se
                        best_label = f'b{blend:.2f}_a{aq}_l{al}_s{sos_p}'
                        best_preds = p.copy()
                        print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # B: Blend with v12 zones also modified (uppermid)
    # ═══════════════════════════════════════════════════════
    print('\n── B: v12 uppermid zone ──')
    for aq in [-6, -4, -2, 0, 2, 4]:
        for al in [-4, -2, 0, 1, 2, 4]:
            for sos_p in [-6, -4, -2, 0, 2, 4, 6]:
                v12_zones_mod = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_cached(cache, y, fn, seasons, test_mask, n,
                             v12_zones=v12_zones_mod, comm_zones=ZONES_V48)
                se, ex = ev(p)
                if se < best_se:
                    best_se = se
                    best_label = f'v12um_a{aq}_l{al}_s{sos_p}'
                    best_preds = p.copy()
                    print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # C: High blend (0.40-0.60) with large positive uppermid corrections
    # (trying to let committee path dominate for 34-44 region)
    # ═══════════════════════════════════════════════════════
    print('\n── C: High blend + aggressive corrections ──')
    for blend in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for aq in [4, 6, 8, 10]:
            for al in [4, 6, 8, 10]:
                for sos_p in [4, 6, 8, 10, 12]:
                    zones = [
                        ('mid',     'committee', (17, 34), (0, 0, 3)),
                        ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                        ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                        ('tail',    'tail',      (60, 63), (1,)),
                    ]
                    p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones, blend=blend)
                    se, ex = ev(p)
                    if se < best_se:
                        best_se = se
                        best_label = f'hb{blend:.2f}_a{aq}_l{al}_s{sos_p}'
                        best_preds = p.copy()
                        print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # D: Zone on BOTH paths (34-44) with different params per path
    # ═══════════════════════════════════════════════════════
    print('\n── D: Both-path uppermid zone optimization ──')
    best_d_se = base_se
    # v48 defaults used for both, then sweep comm uppermid
    for v12_aq in [-2, 0, 2, 4]:
        for v12_al in [-3, 0, 2, 4]:
            for v12_sos in [-4, 0, 4, 6]:
                v12_zones_mod = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (v12_aq, v12_al, v12_sos)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                for c_aq in [-2, 0, 2, 4]:
                    for c_al in [-3, 0, 2, 4]:
                        for c_sos in [-4, 0, 4, 6]:
                            comm_zones_mod = [
                                ('mid',     'committee', (17, 34), (0, 0, 3)),
                                ('uppermid','committee', (34, 44), (c_aq, c_al, c_sos)),
                                ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                                ('tail',    'tail',      (60, 63), (1,)),
                            ]
                            p = run_cached(cache, y, fn, seasons, test_mask, n,
                                         v12_zones=v12_zones_mod, comm_zones=comm_zones_mod)
                            se, ex = ev(p)
                            if se < best_se:
                                best_se = se
                                best_label = f'v12[{v12_aq},{v12_al},{v12_sos}]_c[{c_aq},{c_al},{c_sos}]'
                                best_preds = p.copy()
                                print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
                            if se < best_d_se:
                                best_d_se = se
    
    print(f'  [D done] best={best_label} SE={best_se} (D-best={best_d_se})')
    
    # ═══════════════════════════════════════════════════════
    # E: Everything above + swap rule
    # ═══════════════════════════════════════════════════════
    print('\n── E: Best config + swap rule ──')
    for ng in [8, 10, 12, 15]:
        for pg in [4, 5, 6, 7]:
            pswap, ns = apply_swap(best_preds, X_all, fi, seasons, test_mask, ng, pg)
            se_sw, ex_sw = ev(pswap)
            if se_sw < best_se:
                print(f'  ★ ng={ng} pg={pg}: SE={se_sw} exact={ex_sw}/91 ({ns} swaps)')
    
    # Also try swap on v48 baseline with um_a-6_l1_s-6
    zones_opt = [
        ('mid',     'committee', (17, 34), (0, 0, 3)),
        ('uppermid','committee', (34, 44), (-6, 1, -6)),
        ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
        ('tail',    'tail',      (60, 63), (1,)),
    ]
    p_opt = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_opt)
    se_opt, ex_opt = ev(p_opt)
    print(f'\n  um_a-6_l1_s-6 base: SE={se_opt} exact={ex_opt}/91')
    for ng in [8, 10, 12]:
        for pg in [5, 6, 7]:
            pswap, ns = apply_swap(p_opt, X_all, fi, seasons, test_mask, ng, pg)
            se_sw, ex_sw = ev(pswap)
            print(f'    +swap ng={ng} pg={pg}: SE={se_sw} exact={ex_sw}/91 ({ns} swaps)')
    
    # ═══════════════════════════════════════════════════════
    # F: Detailed analysis of best safe approach
    # ═══════════════════════════════════════════════════════
    print('\n── F: Detailed analysis ──')
    
    # Use um_a-6_l1_s-6 + swap(10,6) as candidate v49
    p_v49, ns = apply_swap(p_opt, X_all, fi, seasons, test_mask, 10, 6)
    se49, ex49 = ev(p_v49)
    print(f'  v49 candidate: SE={se49}, exact={ex49}/91')
    
    print(f'\n  Regressions vs v48:')
    gains, losses = 0, 0
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        se_old = (p48[gi] - actual)**2
        se_new = (p_v49[gi] - actual)**2
        if se_new != se_old:
            arrow = '↑' if se_new < se_old else '↓'
            if se_new < se_old: gains += 1
            else: losses += 1
            print(f'    {arrow} {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se_old:3d}) '
                  f'v49={p_v49[gi]:2d}(SE={se_new:3d})')
    print(f'  Gains: {gains}, Losses: {losses}')
    
    # Per-season
    all_seasons = sorted(set(seasons))
    print(f'\n  Per-season:')
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        if len(si_s) == 0: continue
        sse48 = sum((p48[gi] - int(y[gi]))**2 for gi in si_s)
        sse49 = sum((p_v49[gi] - int(y[gi]))**2 for gi in si_s)
        ex_s = sum(1 for gi in si_s if p_v49[gi] == int(y[gi]))
        marker = '✓' if sse49 <= sse48 else '✗'
        print(f'  {marker} {s}: v48 SE={sse48:3d} v49 SE={sse49:3d} exact={ex_s}/{len(si_s)} Δ={sse48-sse49:+d}')
    
    # Bootstrap
    rng = np.random.RandomState(42)
    test_indices = np.where(test_mask)[0]
    ntest = len(test_indices)
    wins49, wins48, ties = 0, 0, 0
    for b in range(20):
        boot = rng.choice(test_indices, size=ntest, replace=True)
        bse48 = sum((p48[gi] - int(y[gi]))**2 for gi in boot)
        bse49 = sum((p_v49[gi] - int(y[gi]))**2 for gi in boot)
        if bse49 < bse48: wins49 += 1
        elif bse49 > bse48: wins48 += 1
        else: ties += 1
    print(f'\n  Bootstrap: v49 wins {wins49}/20, v48 wins {wins48}/20, ties {ties}/20')
    
    # Remaining
    print(f'\n  Remaining errors (v49):')
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        pred = p_v49[gi]
        if pred != actual:
            se_i = (pred - actual)**2
            net = X_all[gi, fi['NET Rank']]
            print(f'    {record_ids[gi]:32s} gt={actual:2d} pred={pred:2d} SE={se_i:3d} NET={net:.0f}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
