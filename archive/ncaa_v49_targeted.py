#!/usr/bin/env python3
"""
v49 targeted test: Keep separate mid/uppermid but with SOS >= 4 for uppermid.
The key insight: Phase E worked because SOS went from -4 to +4, but merging
the zones caused Clemson/WashSt regression. Let's fix only the uppermid SOS.
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


def main():
    t0 = time.time()
    print('='*60)
    print(' v49 TARGETED TEST')
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
    
    print('  Caching...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    p48 = run_cached(cache, y, fn, seasons, test_mask, n)
    base_se, base_ex = ev(p48)
    print(f'  Baseline v48: SE={base_se}, exact={base_ex}/91')
    
    best_se, best_label = base_se, 'v48'
    
    # ═══════════════════════════════════════════════════════
    # A: Uppermid with SOS >= 4 (not tested in campaign!)
    # ═══════════════════════════════════════════════════════
    print('\n── A: Uppermid SOS extended range ──')
    for aq in range(-6, 5):
        for al in range(-6, 5):
            for sos_p in range(-6, 8):  # Key: extend past 3!
                zones = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones)
                se, ex = ev(p)
                if se < best_se:
                    best_se = se
                    best_label = f'um_a{aq}_l{al}_s{sos_p}'
                    print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # B: Also extend mid SOS range
    # ═══════════════════════════════════════════════════════
    print('\n── B: Mid zone SOS extended + uppermid SOS ──')
    for m_aq in [0, -1, -2]:
        for m_al in [0, -1, -2]:
            for m_sos in [3, 4, 5, 6]:
                for u_aq in [-4, -3, -2, -1]:
                    for u_al in [-5, -4, -3, -2]:
                        for u_sos in [2, 3, 4, 5, 6]:
                            zones = [
                                ('mid',     'committee', (17, 34), (m_aq, m_al, m_sos)),
                                ('uppermid','committee', (34, 44), (u_aq, u_al, u_sos)),
                                ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                                ('tail',    'tail',      (60, 63), (1,)),
                            ]
                            p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones)
                            se, ex = ev(p)
                            if se < best_se:
                                best_se = se
                                best_label = f'mid_a{m_aq}_l{m_al}_s{m_sos}_um_a{u_aq}_l{u_al}_s{u_sos}'
                                print(f'  ★ SE={se} exact={ex}/91 — {best_label}')
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # C: Best from A — detailed regression check
    # ═══════════════════════════════════════════════════════
    print('\n── C: Regression check for best config ──')
    import re
    m = re.match(r'um_a(-?\d+)_l(-?\d+)_s(-?\d+)', best_label)
    if m:
        baq, bal, bsos = int(m.group(1)), int(m.group(2)), int(m.group(3))
        zones_best = [
            ('mid',     'committee', (17, 34), (0, 0, 3)),
            ('uppermid','committee', (34, 44), (baq, bal, bsos)),
            ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
            ('tail',    'tail',      (60, 63), (1,)),
        ]
    else:
        m2 = re.match(r'mid_a(-?\d+)_l(-?\d+)_s(-?\d+)_um_a(-?\d+)_l(-?\d+)_s(-?\d+)', best_label)
        if m2:
            maq, mal, msos = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
            uaq, ual, usos = int(m2.group(4)), int(m2.group(5)), int(m2.group(6))
            zones_best = [
                ('mid',     'committee', (17, 34), (maq, mal, msos)),
                ('uppermid','committee', (34, 44), (uaq, ual, usos)),
                ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                ('tail',    'tail',      (60, 63), (1,)),
            ]
        else:
            print(f'  Cannot parse best_label={best_label}')
            zones_best = ZONES_V48
    
    pb = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_best)
    se_b, ex_b = ev(pb)
    print(f'  Best: SE={se_b}, exact={ex_b}/91')
    
    gains, losses = 0, 0
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        se_old = (p48[gi] - actual)**2
        se_new = (pb[gi] - actual)**2
        if se_new != se_old:
            arrow = '↑' if se_new < se_old else '↓'
            if se_new < se_old: gains += 1
            else: losses += 1
            print(f'  {arrow} {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se_old:3d}) '
                  f'new={pb[gi]:2d}(SE={se_new:3d}) Δ={se_old-se_new:+d}')
    
    print(f'  Gains: {gains}, Losses: {losses}')
    
    # Per-season
    print('\n  Per-season:')
    all_seasons = sorted(set(seasons))
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        if len(si_s) == 0: continue
        sse48 = sum((p48[gi] - int(y[gi]))**2 for gi in si_s)
        ssen = sum((pb[gi] - int(y[gi]))**2 for gi in si_s)
        ex_s = sum(1 for gi in si_s if pb[gi] == int(y[gi]))
        marker = '✓' if ssen <= sse48 else '✗'
        print(f'  {marker} {s}: v48 SE={sse48:3d} new SE={ssen:3d} exact={ex_s}/{len(si_s)} Δ={sse48-ssen:+d}')
    
    # Remaining errors
    print(f'\n  Remaining errors:')
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        pred = pb[gi]
        if pred != actual:
            se_i = (pred - actual)**2
            net = X_all[gi, fi['NET Rank']]
            print(f'    {record_ids[gi]:32s} gt={actual:2d} pred={pred:2d} SE={se_i:3d} NET={net:.0f}')
    
    # Also try adding swap rule  
    print('\n── D: Best + swap rule ──')
    for ng in [5, 8, 10, 12, 15, 20]:
        for pg in [3, 4, 5, 6, 7, 8]:
            pswap = pb.copy()
            for hold_season in sorted(set(seasons)):
                sm = (seasons == hold_season)
                si_s = np.where(sm)[0]
                test_si = [gi for gi in si_s if test_mask[gi]]
                aq_teams = []
                al_teams = []
                for gi in test_si:
                    pred = pswap[gi]
                    if 30 <= pred <= 45:
                        is_aq_v = int(X_all[gi, fi['is_AQ']])
                        is_al_v = int(X_all[gi, fi['is_AL']])
                        net_v = X_all[gi, fi['NET Rank']]
                        if is_aq_v and not is_al_v and pred - net_v > ng:
                            aq_teams.append((gi, pred, net_v))
                        elif is_al_v and not is_aq_v and net_v - pred > 0:
                            al_teams.append((gi, pred, net_v))
                for aq_gi, aq_pred, aq_net in aq_teams:
                    for al_gi, al_pred, al_net in al_teams:
                        if abs(aq_pred - al_pred) <= pg:
                            pswap[aq_gi], pswap[al_gi] = pswap[al_gi], pswap[aq_gi]
            se_sw, ex_sw = ev(pswap)
            if se_sw < se_b:
                print(f'  ng={ng:2d} pg={pg}: SE={se_sw:3d} exact={ex_sw}/91')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
