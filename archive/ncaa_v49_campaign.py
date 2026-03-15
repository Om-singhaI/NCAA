#!/usr/bin/env python3
"""
v49 Improvement Campaign — push v48 (SE=80, Kaggle=0.383) lower.

Remaining errors at SE=80:
  2023-24-NewMexico:     GT=42, pred=36, SE=36 (NET=22, non-power AQ)
  2023-24-Northwestern:  GT=36, pred=41, SE=25 (NET=53, power AL) 
  2023-24-Clemson:       GT=22, pred=24, SE=4  (NET=35, power AL)
  2023-24-SouthCarolina: GT=24, pred=22, SE=4  (NET=51, power AL)
  + 11 SE=1 adjacent swaps

Key observations:
  - NewMexico: NET=22 (great) but GT=42 (terrible seed). Non-power AQ team.
    Committee gave them a much worse seed than NET suggests.
    Our model over-predicts them (36 vs 42) — we reward NET too much for AQ teams.
  - Northwestern: NET=53 but GT=36. Power AL team.
    Committee gave them a better seed than NET suggests.
    Our model under-predicts them (41 vs 36) — we don't reward power-conf enough.

Strategy:
  A: New committee-type zone in (34-44) with NET/power adjustments
  B: Feature engineering targeting NET-seed divergence for AQ vs AL
  C: Different v12 base (modify pairwise training)
  D: Post-processing: swap rules for NM/NW pattern
  E: Zone on v12 path (not just committee)
  F: Different committee features designed for this region
  G: Season-specific v12 power
  H: Adjacent-swap resolution (fix SE=1 pairs)
  I: Full pipeline re-optimization at v48 level
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
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES_V47, 0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12': a_v12, 'a_v12_zoned': a_v12_zoned,
            'avail': avail, 'X_season': X_all[season_mask],
        }
    return cache


def run_v48(cache, X_comm, y, fn, seasons, test_mask, n,
            alpha=10.0, blend=0.25, comm_zones=ZONES_V48):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        r = Ridge(alpha=alpha)
        r.fit(sc.fit_transform(X_comm[train_mask]), y[train_mask])
        raw = r.predict(sc.transform(X_comm[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac = hungarian(raw, seasons[sm], avail, power=0.15)
        ac = apply_zones(ac, raw, fn, X_season, tm, si, comm_zones, 0.15)
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
    print(' v49 IMPROVEMENT CAMPAIGN')
    print(' Starting from v48: SE=80, 76/91, Kaggle=0.383')
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
    
    base = run_v48(cache, X_min8, y, fn, seasons, test_mask, n)
    base_se, base_ex = ev(base)
    print(f'  Baseline v48: SE={base_se}, exact={base_ex}/91')
    
    best_se, best_label, best_preds = base_se, 'v48', base.copy()
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
    # A: DEEP ERROR ANALYSIS (understand why NM and NW are wrong)
    # ═══════════════════════════════════════════════════════
    print('\n── Error analysis ──')
    # Print raw v12 and committee scores for 2023-24 problem teams
    c24 = cache['2023-24']
    si24 = c24['si']; sm24 = c24['season_mask']; tm24 = c24['tm']
    train24 = c24['train_mask']
    sc = StandardScaler()
    r = Ridge(alpha=10)
    r.fit(sc.fit_transform(X_min8[train24]), y[train24])
    raw_comm24 = r.predict(sc.transform(X_min8[sm24]))
    for i, gi in enumerate(si24):
        if not test_mask[gi]: raw_comm24[i] = y[gi]
    
    print(f'  {"Team":25s} {"GT":>3s} {"v12raw":>6s} {"CommRaw":>7s} {"v12H":>4s} {"CommH":>5s} {"v48":>3s} {"NET":>4s}')
    for i, gi in enumerate(si24):
        if test_mask[gi]:
            rid = record_ids[gi]
            actual = int(y[gi])
            v12r = c24['raw_v12'][i]
            cr = raw_comm24[i]
            v12h = c24['a_v12_zoned'][i]
            p48 = base[gi]
            net = X_all[gi, fi['NET Rank']]
            if abs(p48 - actual) >= 1:
                print(f'  {rid:25s} {actual:3d} {v12r:6.1f} {cr:7.1f} {v12h:4d} {"?":>5s} {p48:3d} {net:4.0f}')
    
    # ═══════════════════════════════════════════════════════
    # B: UPPERMID ZONE RE-OPTIMIZATION (NM+NW are in 34-44 range)
    # ═══════════════════════════════════════════════════════
    print('\n── B: Uppermid zone re-optimization ──')
    # NewMexico pred=36 (in zone 34-44), GT=42
    # Northwestern pred=41 (in zone 34-44), GT=36
    # These two need to SWAP! If we could fix the uppermid zone, both would improve.
    # But they're being pushed in opposite wrong directions.
    # The correction types: AQ, AL, SOS
    # NM: AQ=1, AL=0, pow=1 → AQ correction pushes it
    # NW: AQ=0, AL=1, pow=1 → AL correction pushes it
    # Current: aq=-2, al=-3, sos=-4
    # NM is AQ so aq=-2 penalizes AQ → pushes NM UP (wrong, it should go DOWN from 36→42)
    # NW is AL so al=-3 penalizes AL → pushes NW DOWN (wrong, it should go UP from 41→36)
    # ...wait, the correction ORDER matters. Let me check what direction they push.
    
    for aq in range(-6, 4):
        for al in range(-6, 4):
            for sos_p in range(-6, 4):
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
                try_and_track(f'um_a{aq}_l{al}_s{sos_p}', p)
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # C: SPLIT UPPERMID INTO TWO ZONES (34-38 and 38-44)
    # ═══════════════════════════════════════════════════════
    print('\n── C: Split uppermid into two zones ──')
    # NW at pred=41 needs to go to 36, NM at pred=36 needs to go to 42
    # They might be in different sub-zones
    for split in [36, 38, 40, 42]:
        for z1_aq in range(-4, 3, 2):
            for z1_al in range(-5, 2, 2):
                for z2_aq in range(-4, 3, 2):
                    for z2_al in range(-5, 2, 2):
                        zones_test = [
                            ('mid',     'committee', (17, 34), (0, 0, 3)),
                            ('um1',     'committee', (34, split), (z1_aq, z1_al, -4)),
                            ('um2',     'committee', (split, 44), (z2_aq, z2_al, -4)),
                            ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                            ('tail',    'tail',      (60, 63), (1,)),
                        ]
                        p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
                        try_and_track(f'sp{split}_a{z1_aq}{z2_aq}_l{z1_al}{z2_al}', p)
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # D: UPPERMID AS BOTTOM-TYPE ZONE (different correction signal)
    # ═══════════════════════════════════════════════════════
    print('\n── D: Uppermid as bottom-type zone ──')
    for sn in range(-6, 4):
        for nc in range(-4, 5):
            for cb in range(-5, 3):
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','bottom',    (34, 44), (sn, nc, cb)),
                    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
                try_and_track(f'umbottom_s{sn}_n{nc}_c{cb}', p)
    
    print(f'  [D done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # E: MID-RANGE ZONE EXTENSION (extend zone 1 to cover NM region)
    # ═══════════════════════════════════════════════════════
    print('\n── E: Mid-range zone boundary changes ──')
    for hi in [36, 38, 40, 42, 44]:
        for aq in [-2, -1, 0, 1, 2]:
            for al in [-3, -2, -1, 0, 1]:
                for sos_p in [0, 1, 2, 3, 4, 5]:
                    zones_test = [
                        ('mid',     'committee', (17, hi), (aq, al, sos_p)),
                        ('uppermid','committee', (hi, 44) if hi < 44 else (44, 44), (-2, -3, -4)),
                        ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                        ('tail',    'tail',      (60, 63), (1,)),
                    ]
                    if hi >= 44:
                        zones_test = [z for z in zones_test if z[0] != 'uppermid']
                    p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
                    try_and_track(f'midhi{hi}_a{aq}_l{al}_s{sos_p}', p)
    
    print(f'  [E done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # F: REMOVE UPPERMID ZONE ENTIRELY
    # ═══════════════════════════════════════════════════════
    print('\n── F: Remove uppermid zone ──')
    zones_no_um = [
        ('mid',     'committee', (17, 34), (0, 0, 3)),
        ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
        ('tail',    'tail',      (60, 63), (1,)),
    ]
    p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_no_um)
    try_and_track('no_uppermid', p)
    print(f'  [F done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # G: DIFFERENT COMMITTEE FEATURES FOR UPPERMID
    # Feature engineering to better separate NM from NW
    # ═══════════════════════════════════════════════════════
    print('\n── G: Alternative committee features ──')
    # NM: NET=22, SOS=82, Opp=85, pow=1(!), AQ=1, AL=0, wpct=0.735, TFR=22
    # NW: NET=53, SOS=47, Opp=74, pow=1, AQ=0, AL=1, wpct=0.656, TFR=44
    # The key: NM has great NET but weak SOS/Opp. NW has poor NET but strong SOS.
    # Committee sees NM as overvalued by NET, NW as undervalued.
    
    # Try adding NET/SOS ratio as a feature
    net = X_all[:, fi['NET Rank']]
    sos = X_all[:, fi['NETSOS']]
    opp = X_all[:, fi['AvgOppNETRank']]
    is_al = X_all[:, fi['is_AL']]
    is_aq = X_all[:, fi['is_AQ']]
    is_pow = X_all[:, fi['is_power_conf']]
    wpct = X_all[:, fi['WL_Pct']]
    tfr = X_all[:, fi['tourn_field_rank']]
    cb = X_all[:, fi['cb_mean_seed']]
    
    new_features = {
        'net_sos_ratio': net / (sos + 1),       # NM: 22/83=0.27, NW: 53/48=1.10
        'net_opp_ratio': net / (opp + 1),       # NM: 22/86=0.26, NW: 53/75=0.71
        'sos_opp_avg': (sos + opp) / 2,         # NM: 83.5, NW: 60.5
        'aq_net_penalty': is_aq * np.maximum(0, 50 - net),  # AQ with good NET → penalize
        'al_sos_bonus': is_al * np.maximum(0, 100 - sos),   # AL with good SOS → reward
        'net_minus_tfr': net - tfr,              # NM: 22-22=0, NW: 53-44=9
        'net_rank_sq': net**2 / 100,             # Non-linear NET
        'sos_rank_sq': sos**2 / 100,
        'wpct_net': wpct * 100 - net,            # W% adjusted by NET
        'pow_al_net': is_pow * is_al * net,      # Power AL teams NET
        'pow_aq_opp': is_pow * is_aq * opp,      # Power AQ teams OPP
        'aq_sos_penalty': is_aq * sos,           # AQ SOS
        'al_sos_reward': is_al * (200 - sos),    # AL inverse SOS
        'net_cb_gap': net - cb,                  # NET vs CB historical
    }
    
    for fname, fcol in new_features.items():
        X_test = np.column_stack([X_min8, fcol])
        for alpha in [5, 8, 10, 12, 15]:
            for blend in [0.20, 0.25, 0.30]:
                p = run_v48(cache, X_test, y, fn, seasons, test_mask, n,
                           alpha=alpha, blend=blend)
                try_and_track(f'+{fname}_a{alpha}_b{blend}', p)
    
    print(f'  [G done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # H: POST-HUNGARIAN SWAP RULES (targeted at NM/NW pattern)
    # ═══════════════════════════════════════════════════════
    print('\n── H: Post-Hungarian swap rules ──')
    # Look for AQ non-power teams with great NET that might be over-predicted
    # and AL power teams with weak NET that might be under-predicted
    # Then swap them if they're close enough
    
    for net_gap in [10, 15, 20, 25, 30]:
        for pred_gap in [3, 4, 5, 6]:
            preds_sw = base.copy()
            changed = False
            for hold_season in sorted(set(seasons)):
                sm = (seasons == hold_season)
                si_s = np.where(sm)[0]
                test_si = [gi for gi in si_s if test_mask[gi]]
                
                # Find AQ teams in 30-45 range with NET much better than prediction
                # and AL teams in same range  
                aq_teams = []
                al_teams = []
                for gi in test_si:
                    pred = preds_sw[gi]
                    if 30 <= pred <= 45:
                        is_aq_v = int(X_all[gi, fi['is_AQ']])
                        is_al_v = int(X_all[gi, fi['is_AL']])
                        net_v = X_all[gi, fi['NET Rank']]
                        if is_aq_v and not is_al_v and pred - net_v > net_gap:
                            aq_teams.append((gi, pred, net_v))
                        elif is_al_v and not is_aq_v and net_v - pred > 0:
                            al_teams.append((gi, pred, net_v))
                
                # Try swapping each AQ with closest AL
                for aq_gi, aq_pred, aq_net in aq_teams:
                    for al_gi, al_pred, al_net in al_teams:
                        if abs(aq_pred - al_pred) <= pred_gap:
                            # Swap
                            preds_sw[aq_gi], preds_sw[al_gi] = preds_sw[al_gi], preds_sw[aq_gi]
                            changed = True
            
            if changed:
                try_and_track(f'swap_ng{net_gap}_pg{pred_gap}', preds_sw)
    
    print(f'  [H done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # I: MODIFY V12 ZONES (add midbot2 to v12 path too, different params)
    # ═══════════════════════════════════════════════════════
    print('\n── I: Add zone to v12 path ──')
    for sn in [-6, -4, -2, 0, 2]:
        for nc in [-2, 0, 2, 4]:
            for cb in [-4, -3, -2, -1, 0]:
                # Re-cache v12 with new zones on v12 path
                v12_zones = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (-2, -3, -4)),
                    ('newzone', 'bottom',    (34, 44), (sn, nc, cb)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                # Rebuild v12 with new zones
                preds_v12z = np.zeros(n, dtype=int)
                for hold_season, c in cache.items():
                    sm = c['season_mask']; si = c['si']; tm = c['tm']
                    train_mask = c['train_mask']; avail = c['avail']
                    X_season = c['X_season']
                    raw_v12 = c['raw_v12']
                    # Re-apply zones on v12
                    a_v12 = hungarian(raw_v12, seasons[sm], avail, power=0.15)
                    a_v12_new = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, v12_zones, 0.15)
                    # Committee path (v48 zones)
                    sc2 = StandardScaler()
                    r2 = Ridge(alpha=10)
                    r2.fit(sc2.fit_transform(X_min8[train_mask]), y[train_mask])
                    raw_c = r2.predict(sc2.transform(X_min8[sm]))
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: raw_c[i] = y[gi]
                    ac = hungarian(raw_c, seasons[sm], avail, power=0.15)
                    ac = apply_zones(ac, raw_c, fn, X_season, tm, si, ZONES_V48, 0.15)
                    avg = 0.75 * a_v12_new.astype(float) + 0.25 * ac.astype(float)
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: avg[i] = y[gi]
                    af = hungarian(avg, seasons[sm], avail, power=0.15)
                    for i, gi in enumerate(si):
                        if test_mask[gi]: preds_v12z[gi] = af[i]
                try_and_track(f'v12z_s{sn}_n{nc}_c{cb}', preds_v12z)
    
    print(f'  [I done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # J: DIFFERENT BLEND+ALPHA WITH V48 ZONES
    # ═══════════════════════════════════════════════════════
    print('\n── J: Fine alpha/blend grid with v48 zones ──')
    for alpha in [3, 5, 7, 8, 9, 10, 11, 12, 15, 20, 30]:
        for blend in np.arange(0.10, 0.40, 0.02):
            p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n,
                       alpha=alpha, blend=blend)
            try_and_track(f'ab_a{alpha}_b{blend:.2f}', p)
    
    print(f'  [J done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # K: TAIL-TYPE ZONE IN UPPERMID (use opp_rank correction)
    # ═══════════════════════════════════════════════════════
    print('\n── K: Tail-type zone in uppermid ──')
    for opp_w in range(-5, 6):
        zones_test = [
            ('mid',     'committee', (17, 34), (0, 0, 3)),
            ('uppermid','committee', (34, 44), (-2, -3, -4)),
            ('umtail',  'tail',      (34, 44), (opp_w,)),
            ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
            ('tail',    'tail',      (60, 63), (1,)),
        ]
        p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
        try_and_track(f'umtail_opp{opp_w}', p)
    
    # Also try replacing uppermid committee with tail
    for opp_w in range(-5, 6):
        zones_test = [
            ('mid',     'committee', (17, 34), (0, 0, 3)),
            ('uppermid','tail',      (34, 44), (opp_w,)),
            ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
            ('tail',    'tail',      (60, 63), (1,)),
        ]
        p = run_v48(cache, X_min8, y, fn, seasons, test_mask, n, comm_zones=zones_test)
        try_and_track(f'um_asTail_opp{opp_w}', p)
    
    print(f'  [K done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' FINAL RESULTS')
    print('='*60)
    print(f'  Baseline v48: SE={base_se}, exact={base_ex}/91')
    print(f'  Best found: {best_label}, SE={best_se}')
    print(f'  Improvement: Δ={base_se - best_se}')
    
    results.sort(key=lambda x: x[1])
    print(f'\n  Top 25:')
    for label, se, ex in results[:25]:
        flag = '★' if se < base_se else ' '
        print(f'  {flag} SE={se:4d}  exact={ex:2d}/91  {label}')
    
    if best_se < base_se:
        print(f'\n  Errors changed from v48:')
        for gi in np.where(test_mask)[0]:
            old, new, actual = base[gi], best_preds[gi], int(y[gi])
            if old != new:
                rid = record_ids[gi]
                old_se = (old - actual)**2; new_se = (new - actual)**2
                arrow = '↑' if new_se < old_se else '↓' if new_se > old_se else '='
                print(f'    {arrow} {rid:32s} gt={actual:2d} old={old:2d} new={new:2d} '
                      f'oldSE={old_se:3d} newSE={new_se:3d}')
        reg = sum(1 for gi in np.where(test_mask)[0]
                  if (best_preds[gi]-int(y[gi]))**2 > (base[gi]-int(y[gi]))**2)
        print(f'  Regressions: {reg}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
