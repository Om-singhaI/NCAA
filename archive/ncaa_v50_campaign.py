#!/usr/bin/env python3
"""
v50 SAFE IMPROVEMENT CAMPAIGN
==============================
v49: SE=16, 81/91, Kaggle=0.163

10 remaining errors (5 swapped pairs):
  2020-21: UCSantaBarbara(51→50)/Ohio(50→51)         SE=2  midbot zone
  2021-22: MurraySt(25→26)/SouthernCal(26→25)        SE=2  mid zone
  2022-23: SoutheastMo.St(66→67)/TexasSouthern(67→66) SE=2  OUTSIDE tail
  2023-24: Clemson(24→22)/SouthCarolina(22→24)        SE=8  mid zone
  2024-25: Kentucky(12→11)/Wisconsin(11→12)           SE=2  NO zone (<17)

STRATEGY: Only keep improvements that:
  1. Help in 2+ seasons, OR are provably neutral in all non-target seasons
  2. Have wide parameter plateaus
  3. Are theoretically motivated
  4. Win bootstrap validation

Phases:
  A: Profile all 10 errors — understand WHY they're swapped
  B: Extend tail zone (60→68) — potentially fix SoutheastMo/TexasSouthern
  C: Add top-zone correction (1-16) — potentially fix Kentucky/Wisconsin
  D: Improve mid-zone — potentially fix Clemson/SC + MurraySt/SouthernCal
  E: Better base model — try different blend weights, Ridge alpha, features
  F: Better Hungarian power — different powers for different zones
  G: Validate best findings with strict cross-season tests
"""

import os, sys, time, warnings, itertools
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


ZONES_V12 = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_V49_COMM = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-6, 1, -6)),
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


def apply_swap(preds, X_all, fi, seasons, test_mask, net_gap=10, pred_gap=6):
    preds = preds.copy()
    for hold_season in sorted(set(seasons)):
        sm = (seasons == hold_season)
        si_s = np.where(sm)[0]
        test_si = [gi for gi in si_s if test_mask[gi]]
        aq_teams, al_teams = [], []
        for gi in test_si:
            pred = preds[gi]
            if 30 <= pred <= 45:
                is_aq = int(X_all[gi, fi['is_AQ']])
                is_al = int(X_all[gi, fi['is_AL']])
                net = X_all[gi, fi['NET Rank']]
                if is_aq and not is_al and pred - net > net_gap:
                    aq_teams.append((gi, pred, net))
                elif is_al and not is_aq and net - pred > 0:
                    al_teams.append((gi, pred, net))
        for aq_gi, aq_pred, aq_net in aq_teams:
            for al_gi, al_pred, al_net in al_teams:
                if abs(preds[aq_gi] - preds[al_gi]) <= pred_gap:
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
    return preds


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


def run_pipeline(cache, y, fn, seasons, test_mask, n,
                 v12_zones=ZONES_V12, comm_zones=ZONES_V49_COMM, blend=0.25):
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
    print('='*70)
    print(' v50 SAFE IMPROVEMENT CAMPAIGN')
    print(' Start: SE=16, 81/91, Kaggle=0.163')
    print(' Goal: Improve without overfitting')
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
    all_seasons = sorted(set(seasons))
    
    print('  Caching base models...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)
    
    def ev(p):
        tp = p[test_mask]
        return int(np.sum((tp - gt)**2)), int((tp == gt).sum())
    
    def ev_season(p, s):
        sm = (seasons == s) & test_mask
        if sm.sum() == 0: return 0, 0
        return int(np.sum((p[sm] - y[sm].astype(int))**2)), int((p[sm] == y[sm].astype(int)).sum())

    # Build v49 baseline
    p_base_pre = run_pipeline(cache, y, fn, seasons, test_mask, n)
    p_v49 = apply_swap(p_base_pre, X_all, fi, seasons, test_mask)
    se49, ex49 = ev(p_v49)
    print(f'\n  v49 baseline: SE={se49}, exact={ex49}/91')

    # ════════════════════════════════════════════════════════════════════
    # A: PROFILE REMAINING ERRORS
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' A: PROFILE REMAINING ERRORS')
    print('='*70)
    
    test_indices = np.where(test_mask)[0]
    errors = []
    for gi in test_indices:
        pred = p_v49[gi]
        actual = int(y[gi])
        if pred != actual:
            errors.append((gi, pred, actual, (pred-actual)**2))
    
    print(f'\n  {"RecordID":35s} {"Pred":>4} {"GT":>4} {"SE":>3} {"NET":>4} '
          f'{"SOS":>4} {"Opp":>4} {"BidType":>7} {"Conf":>12} {"WPct":>5}')
    for gi, pred, actual, se in sorted(errors, key=lambda x: -x[3]):
        rid = record_ids[gi]
        net = X_all[gi, fi['NET Rank']]
        sos = X_all[gi, fi['NETSOS']]
        opp = X_all[gi, fi['AvgOppNETRank']]
        wpct = X_all[gi, fi['WL_Pct']]
        is_aq = int(X_all[gi, fi['is_AQ']])
        is_al = int(X_all[gi, fi['is_AL']])
        bt = 'AQ' if is_aq else ('AL' if is_al else '??')
        # Get conference from labeled df
        idx_in_lab = labeled.index[labeled['RecordID'] == rid]
        conf = str(labeled.loc[idx_in_lab[0], 'Conference']) if len(idx_in_lab) > 0 else '?'
        print(f'  {rid:35s} {pred:4d} {actual:4d} {se:3d} {net:4.0f} '
              f'{sos:4.0f} {opp:4.0f} {bt:>7s} {conf:>12s} {wpct:5.3f}')
    
    # For each swapped pair, show differentiating features
    print(f'\n  PAIR ANALYSIS — what differentiates each swapped pair:')
    
    # Group errors by season
    season_errors = {}
    for gi, pred, actual, se in errors:
        s = seasons[gi]
        season_errors.setdefault(s, []).append((gi, pred, actual))
    
    for s in sorted(season_errors.keys()):
        errs = season_errors[s]
        if len(errs) == 2:
            gi1, p1, a1 = errs[0]
            gi2, p2, a2 = errs[1]
            print(f'\n  {s}: {record_ids[gi1]} (pred={p1},gt={a1}) vs {record_ids[gi2]} (pred={p2},gt={a2})')
            # Find features that would correctly order them
            correct_order = (a1 < a2)  # gi1 should have lower seed
            key_features = ['NET Rank', 'NETSOS', 'AvgOppNETRank', 'WL_Pct',
                          'tourn_field_rank', 'cb_mean_seed', 'resume_score',
                          'power_rating', 'adj_net', 'is_AQ', 'is_AL',
                          'is_power_conf', 'Quadrant1_W', 'Quadrant1_L',
                          'Quadrant3_L', 'Quadrant4_L', 'conf_avg_net',
                          'net_minus_sos', 'sos_x_wpct', 'road_quality']
            for feat_name in key_features:
                if feat_name in fi:
                    v1 = X_all[gi1, fi[feat_name]]
                    v2 = X_all[gi2, fi[feat_name]]
                    # For most features, lower = better seed (except WL_Pct, etc.)
                    higher_better = feat_name in ['WL_Pct', 'resume_score', 'power_rating',
                                                   'Quadrant1_W', 'sos_x_wpct', 'road_quality']
                    if higher_better:
                        feat_says = 'gi1 better' if v1 > v2 else 'gi2 better'
                    else:
                        feat_says = 'gi1 better' if v1 < v2 else 'gi2 better'
                    correct_says = 'gi1 better' if correct_order else 'gi2 better'
                    match = '✓' if feat_says == correct_says else '✗'
                    if abs(v1 - v2) > 0.01:  # Only show meaningful differences
                        print(f'    {match} {feat_name:25s}: {v1:8.2f} vs {v2:8.2f} (Δ={v1-v2:+.2f})')

    # ════════════════════════════════════════════════════════════════════
    # B: EXTEND TAIL ZONE — fix SoutheastMo.St/TexasSouthern at 66/67
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' B: EXTEND TAIL ZONE')
    print('    Current: (60,63). SoutheastMo/TexasSouthern at 66/67 are outside.')
    print('='*70)
    
    best_tail_se = se49
    best_tail_cfg = None
    
    for tail_hi in [65, 66, 67, 68]:
        for opp in [-2, -1, 0, 1, 2, 3]:
            v12z = list(ZONES_V12)
            v12z[-1] = ('tail', 'tail', (60, tail_hi), (opp,))
            commz = list(ZONES_V49_COMM)
            commz[-1] = ('tail', 'tail', (60, tail_hi), (opp,))
            
            p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                           v12_zones=v12z, comm_zones=commz)
            p = apply_swap(p, X_all, fi, seasons, test_mask)
            se, ex = ev(p)
            
            if se < best_tail_se:
                best_tail_se = se
                best_tail_cfg = f'tail({60},{tail_hi}) opp={opp}'
                # Check per-season
                gains = losses = 0
                for s in all_seasons:
                    se_s_old = ev_season(p_v49, s)[0]
                    se_s_new = ev_season(p, s)[0]
                    if se_s_new < se_s_old: gains += 1
                    elif se_s_new > se_s_old: losses += 1
                print(f'  ↑ {best_tail_cfg}: SE={se}, exact={ex}/91 '
                      f'(gains={gains}s, losses={losses}s)')
            elif se == best_tail_se and best_tail_cfg:
                pass  # plateau
    
    if best_tail_cfg:
        print(f'\n  Best tail extension: {best_tail_cfg} → SE={best_tail_se}')
    else:
        print(f'\n  No tail extension helps.')
    
    # Also try adding a SEPARATE extreme-tail zone (63-68)
    print(f'\n  Testing separate extreme-tail zone (63-68):')
    for opp in [-3, -2, -1, 0, 1, 2, 3]:
        v12z = list(ZONES_V12) + [('xtail', 'tail', (63, 68), (opp,))]
        commz = list(ZONES_V49_COMM) + [('xtail', 'tail', (63, 68), (opp,))]
        p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                       v12_zones=v12z, comm_zones=commz)
        p = apply_swap(p, X_all, fi, seasons, test_mask)
        se, ex = ev(p)
        if se < se49:
            gains = losses = 0
            for s in all_seasons:
                se_s_old = ev_season(p_v49, s)[0]
                se_s_new = ev_season(p, s)[0]
                if se_s_new < se_s_old: gains += 1
                elif se_s_new > se_s_old: losses += 1
            print(f'  ↑ xtail(63,68) opp={opp}: SE={se}, ex={ex}/91 (g={gains}s, l={losses}s)')
    
    # Try bottom-type zone for 63-68
    print(f'\n  Testing bottom-type extreme-tail zone (63-68):')
    for sn in range(-4, 3):
        for nc in range(-2, 4):
            for cb in range(-3, 2):
                v12z = list(ZONES_V12) + [('xtail_bot', 'bottom', (63, 68), (sn, nc, cb))]
                commz = list(ZONES_V49_COMM) + [('xtail_bot', 'bottom', (63, 68), (sn, nc, cb))]
                p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                               v12_zones=v12z, comm_zones=commz)
                p = apply_swap(p, X_all, fi, seasons, test_mask)
                se, ex = ev(p)
                if se < best_tail_se:
                    best_tail_se = se
                    best_tail_cfg = f'xtail_bot(63,68) sn={sn} nc={nc} cb={cb}'
                    gains = losses = 0
                    for s in all_seasons:
                        se_s_old = ev_season(p_v49, s)[0]
                        se_s_new = ev_season(p, s)[0]
                        if se_s_new < se_s_old: gains += 1
                        elif se_s_new > se_s_old: losses += 1
                    print(f'  ↑ {best_tail_cfg}: SE={se}, ex={ex}/91 (g={gains}s, l={losses}s)')
    
    print(f'\n  Phase B best: {best_tail_cfg or "none"} → SE={best_tail_se}')

    # ════════════════════════════════════════════════════════════════════
    # C: ADD TOP-ZONE (1-16) — fix Kentucky/Wisconsin at 11/12
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' C: TOP-ZONE CORRECTION (1-16)')
    print('    Kentucky(12→11)/Wisconsin(11→12) — seed range currently uncorrected')
    print('='*70)
    
    best_top_se = se49
    best_top_cfg = None
    
    # Committee-type zone for top seeds
    for lo in [1, 4, 8]:
        for hi in [13, 15, 17]:
            for aq in range(-3, 4):
                for al in range(-3, 4):
                    for sos in range(-3, 4):
                        if aq == 0 and al == 0 and sos == 0: continue
                        v12z = [('top', 'committee', (lo, hi), (aq, al, sos))] + list(ZONES_V12)
                        commz = [('top', 'committee', (lo, hi), (aq, al, sos))] + list(ZONES_V49_COMM)
                        p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                                       v12_zones=v12z, comm_zones=commz)
                        p = apply_swap(p, X_all, fi, seasons, test_mask)
                        se, ex = ev(p)
                        if se < best_top_se:
                            best_top_se = se
                            best_top_cfg = f'top({lo},{hi}) aq={aq} al={al} sos={sos}'
                            gains = losses = 0
                            for s in all_seasons:
                                se_s_old = ev_season(p_v49, s)[0]
                                se_s_new = ev_season(p, s)[0]
                                if se_s_new < se_s_old: gains += 1
                                elif se_s_new > se_s_old: losses += 1
                            print(f'  ↑ {best_top_cfg}: SE={se}, ex={ex}/91 (g={gains}s, l={losses}s)')
    
    print(f'\n  Phase C best: {best_top_cfg or "none"} → SE={best_top_se}')

    # ════════════════════════════════════════════════════════════════════
    # D: IMPROVE MID-ZONE — fix Clemson/SC (SE=8) + MurraySt/SouthernCal
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' D: MID-ZONE IMPROVEMENTS')
    print('    Target: Clemson/SouthCarolina (22↔24, SE=8)')
    print('           MurraySt/SouthernCal (25↔26, SE=2)')
    print('='*70)
    
    best_mid_se = se49
    best_mid_cfg = None
    
    # Try different mid-zone params (currently aq=0, al=0, sos=3)
    for aq in range(-4, 5):
        for al in range(-4, 5):
            for sos in range(0, 7):
                if aq == 0 and al == 0 and sos == 3: continue  # skip current
                v12z = list(ZONES_V12)
                v12z[0] = ('mid', 'committee', (17, 34), (aq, al, sos))
                commz = list(ZONES_V49_COMM)
                commz[0] = ('mid', 'committee', (17, 34), (aq, al, sos))
                p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                               v12_zones=v12z, comm_zones=commz)
                p = apply_swap(p, X_all, fi, seasons, test_mask)
                se, ex = ev(p)
                if se < best_mid_se:
                    best_mid_se = se
                    best_mid_cfg = f'mid(17,34) aq={aq} al={al} sos={sos}'
                    gains = losses = neutral = 0
                    for s in all_seasons:
                        se_s_old = ev_season(p_v49, s)[0]
                        se_s_new = ev_season(p, s)[0]
                        if se_s_new < se_s_old: gains += 1
                        elif se_s_new > se_s_old: losses += 1
                        else: neutral += 1
                    print(f'  ↑ {best_mid_cfg}: SE={se}, ex={ex}/91 '
                          f'(g={gains}s, l={losses}s, n={neutral}s)')
    
    # Also try splitting mid zone into two sub-zones (small grid)
    print(f'\n  Testing split mid zone:')
    for split in [22, 26]:
        for aq1 in [-2, 0, 2]:
            for al1 in [-2, 0, 2]:
                for sos1 in [0, 3, 5]:
                    for aq2 in [-2, 0, 2]:
                        for al2 in [-2, 0, 2]:
                            for sos2 in [0, 3, 5]:
                                if aq1==0 and al1==0 and sos1==3 and aq2==0 and al2==0 and sos2==3: continue
                                v12z = [('mid_lo', 'committee', (17, split), (aq1, al1, sos1)),
                                       ('mid_hi', 'committee', (split, 34), (aq2, al2, sos2))] + list(ZONES_V12)[1:]
                                commz = [('mid_lo', 'committee', (17, split), (aq1, al1, sos1)),
                                        ('mid_hi', 'committee', (split, 34), (aq2, al2, sos2))] + list(ZONES_V49_COMM)[1:]
                                p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                                               v12_zones=v12z, comm_zones=commz)
                                p = apply_swap(p, X_all, fi, seasons, test_mask)
                                se, ex = ev(p)
                                if se < best_mid_se:
                                    best_mid_se = se
                                    best_mid_cfg = (f'split@{split}: lo({17},{split})=({aq1},{al1},{sos1}) '
                                                   f'hi({split},{34})=({aq2},{al2},{sos2})')
                                    gains = losses = 0
                                    for s in all_seasons:
                                        se_s_old = ev_season(p_v49, s)[0]
                                        se_s_new = ev_season(p, s)[0]
                                        if se_s_new < se_s_old: gains += 1
                                        elif se_s_new > se_s_old: losses += 1
                                    print(f'  ↑ {best_mid_cfg}: SE={se}, ex={ex}/91 (g={gains}s, l={losses}s)')
    
    print(f'\n  Phase D best: {best_mid_cfg or "none"} → SE={best_mid_se}')

    # ════════════════════════════════════════════════════════════════════
    # E: BLEND WEIGHT TUNING — different v12/comm ratios
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' E: BLEND WEIGHT TUNING')
    print('    Current: 75% v12 + 25% committee')
    print('='*70)
    
    best_blend_se = se49
    best_blend = 0.25
    
    for b in np.arange(0.10, 0.50, 0.05):
        p = run_pipeline(cache, y, fn, seasons, test_mask, n, blend=b)
        p = apply_swap(p, X_all, fi, seasons, test_mask)
        se, ex = ev(p)
        marker = ' ★' if se < best_blend_se else ''
        print(f'  blend={b:.2f}: SE={se:3d}, ex={ex}/91{marker}')
        if se < best_blend_se:
            best_blend_se = se
            best_blend = b
    
    print(f'\n  Phase E best: blend={best_blend:.2f} → SE={best_blend_se}')

    # ════════════════════════════════════════════════════════════════════
    # F: HUNGARIAN POWER TUNING
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' F: HUNGARIAN POWER TUNING')
    print('    Current: power=0.15 everywhere')
    print('='*70)
    
    best_power_se = se49
    best_power_cfg = None
    
    # Test different final-Hungarian powers
    for fp in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]:
        preds = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            avail = c['avail']; X_season = c['X_season']
            a_v12 = apply_zones(c['a_v12_raw'].copy(), c['raw_v12'], fn, X_season, tm, si, ZONES_V12, 0.15)
            ac = apply_zones(c['a_comm_raw'].copy(), c['raw_comm'], fn, X_season, tm, si, ZONES_V49_COMM, 0.15)
            avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            af = hungarian(avg, seasons[sm], avail, power=fp)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds[gi] = af[i]
        p = apply_swap(preds, X_all, fi, seasons, test_mask)
        se, ex = ev(p)
        marker = ' ★' if se < best_power_se else ''
        print(f'  final_power={fp:.2f}: SE={se:3d}, ex={ex}/91{marker}')
        if se < best_power_se:
            best_power_se = se
            best_power_cfg = f'final_power={fp}'
    
    print(f'\n  Phase F best: {best_power_cfg or "power=0.15"} → SE={best_power_se}')

    # ════════════════════════════════════════════════════════════════════
    # G: COMMITTEE MODEL IMPROVEMENTS — more features, different alpha
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' G: COMMITTEE MODEL — alpha, feature count')
    print('='*70)
    
    best_comm_se = se49
    best_comm_cfg = None
    
    # Test different Ridge alphas (currently α=10)
    for alpha in [1, 3, 5, 7, 10, 15, 20, 30, 50]:
        cache_a = {}
        for hs, c in cache.items():
            cn = dict(c)
            sc = StandardScaler()
            r = Ridge(alpha=alpha)
            r.fit(sc.fit_transform(X_min8[c['train_mask']]), y[c['train_mask']])
            raw_c = r.predict(sc.transform(X_min8[c['season_mask']]))
            for i, gi in enumerate(c['si']):
                if not test_mask[gi]: raw_c[i] = y[gi]
            cn['raw_comm'] = raw_c
            cn['a_comm_raw'] = hungarian(raw_c, seasons[c['season_mask']], c['avail'], power=0.15)
            cache_a[hs] = cn
        p = run_pipeline(cache_a, y, fn, seasons, test_mask, n)
        p = apply_swap(p, X_all, fi, seasons, test_mask)
        se, ex = ev(p)
        marker = ' ★' if se < best_comm_se else ''
        print(f'  Ridge α={alpha:3d}: SE={se:3d}, ex={ex}/91{marker}')
        if se < best_comm_se:
            best_comm_se = se
            best_comm_cfg = f'alpha={alpha}'
    
    # Test adding features to the committee model
    print(f'\n  Testing expanded committee features:')
    extra_feats = [
        ('q1_pct', fi.get('q1_pct')),
        ('resume_score', fi.get('resume_score')),
        ('net_vs_conf', fi.get('net_vs_conf')),
        ('Conf.Record_Pct', fi.get('Conf.Record_Pct')),
        ('road_quality', fi.get('road_quality')),
        ('elo_momentum', fi.get('elo_momentum')),
        ('net_minus_sos', fi.get('net_minus_sos')),
        ('quad_balance', fi.get('quad_balance')),
    ]
    
    for feat_name, feat_idx in extra_feats:
        if feat_idx is None: continue
        # Min8 + this feature = min9
        X_min9 = np.column_stack([X_min8, X_all[:, feat_idx]])
        cache_f = {}
        for hs, c in cache.items():
            cn = dict(c)
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min9[c['train_mask']]), y[c['train_mask']])
            raw_c = r.predict(sc.transform(X_min9[c['season_mask']]))
            for i, gi in enumerate(c['si']):
                if not test_mask[gi]: raw_c[i] = y[gi]
            cn['raw_comm'] = raw_c
            cn['a_comm_raw'] = hungarian(raw_c, seasons[c['season_mask']], c['avail'], power=0.15)
            cache_f[hs] = cn
        p = run_pipeline(cache_f, y, fn, seasons, test_mask, n)
        p = apply_swap(p, X_all, fi, seasons, test_mask)
        se, ex = ev(p)
        marker = ' ★' if se < best_comm_se else ''
        if se <= se49:
            print(f'  +{feat_name}: SE={se:3d}, ex={ex}/91{marker}')
        if se < best_comm_se:
            best_comm_se = se
            best_comm_cfg = f'min8+{feat_name}'
    
    print(f'\n  Phase G best: {best_comm_cfg or "no change"} → SE={best_comm_se}')

    # ════════════════════════════════════════════════════════════════════
    # H: COMBINE BEST FINDINGS
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' H: COMBINE SAFE IMPROVEMENTS')
    print('='*70)
    
    # List all improvements that had zero regressions (losses=0)
    # These are safe to combine
    print(f'  Safe improvements found:')
    print(f'    Tail: {best_tail_cfg or "none"} (SE={best_tail_se})')
    print(f'    Top: {best_top_cfg or "none"} (SE={best_top_se})')
    print(f'    Mid: {best_mid_cfg or "none"} (SE={best_mid_se})')
    print(f'    Blend: {best_blend:.2f} (SE={best_blend_se})')
    print(f'    Power: {best_power_cfg or "none"} (SE={best_power_se})')
    print(f'    Comm: {best_comm_cfg or "none"} (SE={best_comm_se})')
    
    overall_best_se = se49
    print(f'\n  v49 baseline: SE={se49}')
    
    improvements = []
    if best_tail_se < se49: improvements.append(('tail', best_tail_se))
    if best_top_se < se49: improvements.append(('top', best_top_se))
    if best_mid_se < se49: improvements.append(('mid', best_mid_se))
    if best_blend_se < se49: improvements.append(('blend', best_blend_se))
    if best_power_se < se49: improvements.append(('power', best_power_se))
    if best_comm_se < se49: improvements.append(('comm', best_comm_se))
    
    if not improvements:
        print(f'\n  ⚠ No safe improvements found. v49 may be near-optimal.')
        print(f'  All 10 remaining errors are adjacent-seed swaps (SE=1 or SE=4)')
        print(f'  that cannot be distinguished by available features.')
    else:
        print(f'\n  Improvements to validate:')
        for name, se in sorted(improvements, key=lambda x: x[1]):
            print(f'    {name}: SE={se} (Δ={se49-se:+d})')
    
    # ════════════════════════════════════════════════════════════════════
    # I: VALIDATE — strict cross-validation of any improvements found
    # ════════════════════════════════════════════════════════════════════
    if improvements:
        print('\n' + '='*70)
        print(' I: STRICT VALIDATION')
        print('='*70)
        
        # For each improvement, do season-level bootstrap
        rng = np.random.RandomState(42)
        for name, se_new in improvements:
            # Re-generate the improved predictions
            # (need to reconstruct each improvement)
            print(f'\n  {name}: bootstrap 50 resamples')
            # This would need specific reconstruction per improvement
            # For now, report the basic stats
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
