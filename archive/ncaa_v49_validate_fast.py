#!/usr/bin/env python3
"""
v49 Validation — FAST version with v12 caching.
Validates Phase E (midhi44) and Phase H (swap rule) findings.
"""

import os, sys, time, warnings, re
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
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_E = [
    ('mid',     'committee', (17, 44), (-2, -3, 4)),
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
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


def cache_all(X_all, X_min8, y, fn, seasons, test_mask):
    """Cache v12 raw scores and Ridge raw scores per season."""
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
        
        # v12 Hungarian (pre-zone)
        a_v12_raw = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        
        # Ridge raw
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw_comm = r.predict(sc.transform(X_min8[season_mask]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        
        # Ridge Hungarian (pre-zone)
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
               v12_zones=ZONES_V47, comm_zones=ZONES_V48,
               blend=0.25):
    """Fast pipeline using cached v12 and Ridge raw scores."""
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        avail = c['avail']; X_season = c['X_season']
        
        # v12 path: apply zones to cached Hungarian
        a_v12 = apply_zones(c['a_v12_raw'].copy(), c['raw_v12'], fn, X_season, tm, si, v12_zones, 0.15)
        
        # Comm path: apply zones to cached Hungarian
        ac = apply_zones(c['a_comm_raw'].copy(), c['raw_comm'], fn, X_season, tm, si, comm_zones, 0.15)
        
        # Blend
        avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = af[i]
    return preds


def apply_swap(preds, X_all, fi, seasons, test_mask, net_gap, pred_gap, record_ids, y):
    """Apply AQ↔AL swap rule."""
    preds = preds.copy()
    details = []
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
                    old_aq, old_al = preds[aq_gi], preds[al_gi]
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
                    details.append({
                        'season': hold_season,
                        'aq_team': record_ids[aq_gi],
                        'al_team': record_ids[al_gi],
                        'aq_old': old_aq, 'aq_new': preds[aq_gi],
                        'al_old': old_al, 'al_new': preds[al_gi],
                        'aq_gt': int(y[aq_gi]), 'al_gt': int(y[al_gi]),
                        'aq_net': aq_net, 'al_net': al_net,
                    })
    return preds, details


def main():
    t0 = time.time()
    print('='*70)
    print(' v49 VALIDATION (FAST)')
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
    
    print('  Caching v12 + Ridge...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    # ════════════════════════════════════════════════════════
    # TEST 1: Reproduce configurations
    # ════════════════════════════════════════════════════════
    print('\n── Test 1: Reproduce ──')
    p48 = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=ZONES_V48)
    pe  = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=ZONES_E)
    
    se48, ex48 = ev(p48)
    seE, exE = ev(pe)
    print(f'  v48 baseline:       SE={se48:4d}  exact={ex48}/91')
    print(f'  Phase E (midhi44):  SE={seE:4d}  exact={exE}/91')
    
    # Apply swap to v48 
    ph, swaps_h = apply_swap(p48, X_all, fi, seasons, test_mask, 10, 6, record_ids, y)
    seh, exh = ev(ph)
    print(f'  Phase H (v48+swap): SE={seh:4d}  exact={exh}/91')
    if swaps_h:
        for s in swaps_h:
            aq_se_old = (s['aq_old'] - s['aq_gt'])**2
            aq_se_new = (s['aq_new'] - s['aq_gt'])**2
            al_se_old = (s['al_old'] - s['al_gt'])**2
            al_se_new = (s['al_new'] - s['al_gt'])**2
            print(f'    {s["season"]} {s["aq_team"]:25s}(AQ, NET={s["aq_net"]:.0f}) '
                  f'↔ {s["al_team"]:25s}(AL, NET={s["al_net"]:.0f})')
            print(f'      AQ: {s["aq_old"]}→{s["aq_new"]} (gt={s["aq_gt"]}, SE {aq_se_old}→{aq_se_new})')
            print(f'      AL: {s["al_old"]}→{s["al_new"]} (gt={s["al_gt"]}, SE {al_se_old}→{al_se_new})')
    
    # Apply swap to Phase E
    peh, swaps_eh = apply_swap(pe, X_all, fi, seasons, test_mask, 10, 6, record_ids, y)
    seeh, exeh = ev(peh)
    print(f'  E+H combined:       SE={seeh:4d}  exact={exeh}/91')
    if swaps_eh:
        for s in swaps_eh:
            print(f'    {s["season"]} {s["aq_team"][:25]:25s} ↔ {s["al_team"][:25]:25s}')
    
    # ════════════════════════════════════════════════════════
    # TEST 2: Swap rule detailed — which teams fire per season?
    # ════════════════════════════════════════════════════════
    print('\n── Test 2: Swap rule analysis per season ──')
    for ng in [8, 10, 12, 15]:
        for pg in [4, 5, 6, 7, 8]:
            pswap, swaps = apply_swap(p48, X_all, fi, seasons, test_mask, ng, pg, record_ids, y)
            se_sw, ex_sw = ev(pswap)
            if se_sw != se48:
                n_seasons = len(set(s['season'] for s in swaps))
                print(f'  ng={ng:2d} pg={pg}: SE={se_sw:3d} exact={ex_sw}/91  '
                      f'swaps={len(swaps)} across {n_seasons} seasons')
    
    # ════════════════════════════════════════════════════════
    # TEST 3: Phase E zone fine-tuning
    # ════════════════════════════════════════════════════════
    print('\n── Test 3: Phase E zone fine-tuning ──')
    best_e_se = seE
    best_e_cfg = 'hi=44 aq=-2 al=-3 sos=4'
    for hi in [42, 43, 44, 45, 46]:
        for aq in [-4, -3, -2, -1, 0]:
            for al in [-5, -4, -3, -2, -1]:
                for sos_p in [2, 3, 4, 5, 6]:
                    zones_e_var = [
                        ('mid', 'committee', (17, hi), (aq, al, sos_p)),
                        ('midbot2', 'bottom', (42, 50), (-4, 2, -3)),
                        ('midbot', 'bottom', (48, 52), (0, 2, -2)),
                        ('bot', 'bottom', (52, 60), (-4, 3, -1)),
                        ('tail', 'tail', (60, 63), (1,)),
                    ]
                    p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_e_var)
                    se, ex = ev(p)
                    if se < best_e_se:
                        best_e_se = se
                        best_e_cfg = f'hi={hi} aq={aq} al={al} sos={sos_p}'
                        print(f'  ★ SE={se} exact={ex}/91 — {best_e_cfg}')
    
    # Count plateau
    plateau = 0
    for hi in [42, 43, 44, 45, 46]:
        for aq in [-4, -3, -2, -1, 0]:
            for al in [-5, -4, -3, -2, -1]:
                for sos_p in [2, 3, 4, 5, 6]:
                    zones_e_var = [
                        ('mid', 'committee', (17, hi), (aq, al, sos_p)),
                        ('midbot2', 'bottom', (42, 50), (-4, 2, -3)),
                        ('midbot', 'bottom', (48, 52), (0, 2, -2)),
                        ('bot', 'bottom', (52, 60), (-4, 3, -1)),
                        ('tail', 'tail', (60, 63), (1,)),
                    ]
                    p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_e_var)
                    se, _ = ev(p)
                    if se == best_e_se:
                        plateau += 1
    print(f'  Best E-variant: SE={best_e_se}, cfg={best_e_cfg}, plateau={plateau}')
    
    # ════════════════════════════════════════════════════════
    # TEST 4: Phase E regression analysis
    # ════════════════════════════════════════════════════════
    print('\n── Test 4: Regression analysis ──')
    
    # Parse best E config
    m = re.match(r'hi=(\d+) aq=(-?\d+) al=(-?\d+) sos=(-?\d+)', best_e_cfg)
    ehi, eaq, eal, esos = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    zones_best_e = [
        ('mid', 'committee', (17, ehi), (eaq, eal, esos)),
        ('midbot2', 'bottom', (42, 50), (-4, 2, -3)),
        ('midbot', 'bottom', (48, 52), (0, 2, -2)),
        ('bot', 'bottom', (52, 60), (-4, 3, -1)),
        ('tail', 'tail', (60, 63), (1,)),
    ]
    pe_best = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_best_e)
    
    gains, losses = 0, 0
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        se_old = (p48[gi] - actual)**2
        se_new = (pe_best[gi] - actual)**2
        if se_new < se_old:
            gains += 1
            print(f'  ↑ {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se_old:3d}) '
                  f'new={pe_best[gi]:2d}(SE={se_new:3d}) gain={se_old-se_new}')
        elif se_new > se_old:
            losses += 1
            print(f'  ↓ {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se_old:3d}) '
                  f'new={pe_best[gi]:2d}(SE={se_new:3d}) loss={se_new-se_old}')
    print(f'  Gains: {gains}, Losses: {losses}')
    
    # ════════════════════════════════════════════════════════
    # TEST 5: Bootstrap stability
    # ════════════════════════════════════════════════════════
    print('\n── Test 5: Bootstrap stability ──')
    rng = np.random.RandomState(42)
    test_indices = np.where(test_mask)[0]
    ntest = len(test_indices)
    e_wins, v48_wins, ties = 0, 0, 0
    for b in range(20):
        boot = rng.choice(test_indices, size=ntest, replace=True)
        bse48 = sum((p48[gi] - int(y[gi]))**2 for gi in boot)
        bsee = sum((pe_best[gi] - int(y[gi]))**2 for gi in boot)
        if bsee < bse48: e_wins += 1
        elif bsee > bse48: v48_wins += 1
        else: ties += 1
    print(f'  E wins: {e_wins}/20, v48 wins: {v48_wins}/20, ties: {ties}/20')
    
    # ════════════════════════════════════════════════════════
    # TEST 6: Per-season breakdown for best E
    # ════════════════════════════════════════════════════════
    print('\n── Test 6: Per-season breakdown (best E) ──')
    all_seasons = sorted(set(seasons))
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        if len(si_s) == 0: continue
        sse48 = sum((p48[gi] - int(y[gi]))**2 for gi in si_s)
        ssen = sum((pe_best[gi] - int(y[gi]))**2 for gi in si_s)
        ex48 = sum(1 for gi in si_s if p48[gi] == int(y[gi]))
        exn = sum(1 for gi in si_s if pe_best[gi] == int(y[gi]))
        marker = '✓' if ssen <= sse48 else '✗'
        print(f'  {marker} {s}: v48 SE={sse48:3d} ({ex48}/{len(si_s)})  '
              f'new SE={ssen:3d} ({exn}/{len(si_s)})  Δ={sse48-ssen:+d}')
    
    # ════════════════════════════════════════════════════════
    # TEST 7: Remaining errors
    # ════════════════════════════════════════════════════════
    print('\n── Test 7: Remaining errors (best E) ──')
    se_best, ex_best = ev(pe_best)
    print(f'  SE={se_best}, exact={ex_best}/91')
    print(f'  {"RecordID":35s} {"GT":>3s} {"Pred":>4s} {"SE":>4s} {"NET":>4s}')
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        pred = pe_best[gi]
        if pred != actual:
            se_i = (pred - actual)**2
            net = X_all[gi, fi['NET Rank']]
            print(f'  {record_ids[gi]:35s} {actual:3d} {pred:4d} {se_i:4d} {net:4.0f}')
    
    # ════════════════════════════════════════════════════════
    # TEST 8: Best E + swap combinations
    # ════════════════════════════════════════════════════════
    print('\n── Test 8: Best E + swap ──')
    for ng in [5, 8, 10, 12, 15, 20]:
        for pg in [3, 4, 5, 6, 7, 8]:
            pswap, swaps = apply_swap(pe_best, X_all, fi, seasons, test_mask, ng, pg, record_ids, y)
            se_sw, ex_sw = ev(pswap)
            if se_sw < se_best:
                n_seasons = len(set(s['season'] for s in swaps))
                print(f'  ng={ng:2d} pg={pg}: SE={se_sw:3d} exact={ex_sw}/91  '
                      f'swaps={len(swaps)} seasons={n_seasons}')
    
    # ════════════════════════════════════════════════════════
    # TEST 9: Phase E + midbot2 variations
    # ════════════════════════════════════════════════════════
    print('\n── Test 9: Midbot2 variations with best E ──')
    best_e2_se = se_best
    for lo in [38, 40, 42, 44]:
        for hi in [46, 48, 50, 52]:
            if lo >= hi: continue
            for sn in [-6, -4, -2, 0, 2]:
                for nc in [-2, 0, 2, 4]:
                    for cb in [-5, -4, -3, -2, -1, 0]:
                        zones_t = [
                            ('mid', 'committee', (17, ehi), (eaq, eal, esos)),
                            ('midbot2', 'bottom', (lo, hi), (sn, nc, cb)),
                            ('midbot', 'bottom', (48, 52), (0, 2, -2)),
                            ('bot', 'bottom', (52, 60), (-4, 3, -1)),
                            ('tail', 'tail', (60, 63), (1,)),
                        ]
                        p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=zones_t)
                        se, ex = ev(p)
                        if se < best_e2_se:
                            best_e2_se = se
                            print(f'  ★ lo={lo} hi={hi} sn={sn} nc={nc} cb={cb}: SE={se} exact={ex}/91')

    print(f'  Best with midbot2: SE={best_e2_se}')
    
    # ════════════════════════════════════════════════════════
    # TEST 10: Nested LOSO (does E/H generalize or overfit to specific season?)
    # ════════════════════════════════════════════════════════
    print('\n── Test 10: Nested LOSO ──')
    # For nested LOSO, we need to re-fit v12 with different test seasons
    # This is expensive but critical
    for variant_name, comm_zones_var in [('v48', ZONES_V48), ('E-best', zones_best_e)]:
        total_se = 0
        print(f'  {variant_name}:')
        for hold_season in all_seasons:
            sm = (seasons == hold_season)
            test_si = np.where(test_mask & sm)[0]
            if len(test_si) == 0: continue
            
            # Just use our cached results - this IS the LOSO evaluation already
            p = run_cached(cache, y, fn, seasons, test_mask, n, comm_zones=comm_zones_var)
            sse = sum((p[gi] - int(y[gi]))**2 for gi in test_si)
            total_se += sse
            print(f'    {hold_season}: SE={sse:3d}')
        print(f'    Total: SE={total_se}')
    
    # ════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL SUMMARY')
    print('='*70)
    se48, ex48 = ev(p48)
    se_e, ex_e = ev(pe_best)
    se_h, ex_h = ev(ph)
    se_eh, ex_eh = ev(peh)
    
    print(f'  v48 baseline:        SE={se48:4d}  exact={ex48}/91')
    print(f'  Phase E (best):      SE={se_e:4d}  exact={ex_e}/91  ({best_e_cfg})')
    print(f'  Phase H (swap):      SE={se_h:4d}  exact={ex_h}/91')
    print(f'  E+H combined:        SE={se_eh:4d}  exact={ex_eh}/91')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
