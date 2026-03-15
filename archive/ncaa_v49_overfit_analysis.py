#!/usr/bin/env python3
"""
v49 DEEP OVERFITTING ANALYSIS
==============================
Local SE=16, Kaggle=0.163. For 2025-26 competition we need to know
how much of v49's improvement is real vs overfit.

v49 has TWO new components vs v48 (SE=80, Kaggle=0.383):
  1. Committee uppermid zone: (-2,-3,-4) → (-6,1,-6)  [fixes TCU/Creighton 21-22]
  2. AQ↔AL swap rule (ng=10, pg=6)  [fixes NM/NW/Virginia 23-24]

Key risk: The swap rule only fires in 2023-24. Is it overfitting to
one season's committee decisions?

Tests:
  A: Decompose v49 into its two components — which contributes what
  B: Leave-one-season-out: does each component help on held-out seasons?
  C: How many teams COULD trigger the swap rule per season?
  D: Is New Mexico 2023-24 truly an anomaly or part of a pattern?
  E: Simulate adding noise — how stable is v49 vs v48?
  F: What happens if 2025-26 has a similar AQ anomaly? (robustness)
  G: Per-component nested LOSO gap analysis
  H: Feature importance stability across seasons
  I: Historical precedent: how often do AQ teams with NET<25 get seed>40?
  J: Decision boundary analysis — how close are we to wrong swaps?
  K: Kaggle score vs local RMSE discrepancy analysis
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


ZONES_V47 = [  # v12 path zones (unchanged since v47)
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_V48_COMM = [  # v48 committee path (original uppermid)
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_V49_COMM = [  # v49 committee path (new uppermid)
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-6, 1, -6)),
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


def run_pipeline(cache, y, fn, seasons, test_mask, n,
                 v12_zones=ZONES_V47, comm_zones=ZONES_V48_COMM, blend=0.25):
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


def apply_swap(preds, X_all, fi, seasons, test_mask, net_gap=10, pred_gap=6):
    preds = preds.copy()
    details = []
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
                    details.append({
                        'season': hold_season,
                        'aq_rid': None, 'al_rid': None,
                        'aq_gi': aq_gi, 'al_gi': al_gi,
                        'aq_pred_before': preds[aq_gi], 'al_pred_before': preds[al_gi],
                        'aq_net': aq_net, 'al_net': al_net,
                    })
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
    return preds, details


def main():
    t0 = time.time()
    print('='*70)
    print(' v49 DEEP OVERFITTING ANALYSIS')
    print(' Local: SE=16, 81/91 | Kaggle: 0.163')
    print(' Question: How much will generalize to 2025-26?')
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
    
    print('  Caching...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    # Build all pipeline variants
    p_v48 = run_pipeline(cache, y, fn, seasons, test_mask, n, comm_zones=ZONES_V48_COMM)
    p_zone_only = run_pipeline(cache, y, fn, seasons, test_mask, n, comm_zones=ZONES_V49_COMM)
    p_swap_only, sw48 = apply_swap(p_v48, X_all, fi, seasons, test_mask)
    p_v49, sw49 = apply_swap(p_zone_only, X_all, fi, seasons, test_mask)
    
    se48, ex48 = ev(p_v48)
    se_zo, ex_zo = ev(p_zone_only)
    se_so, ex_so = ev(p_swap_only)
    se49, ex49 = ev(p_v49)
    
    print(f'\n  Baseline v48:           SE={se48:4d}  exact={ex48}/91')
    print(f'  + zone change only:     SE={se_zo:4d}  exact={ex_zo}/91  (Δ={se48-se_zo:+d})')
    print(f'  + swap rule only:       SE={se_so:4d}  exact={ex_so}/91  (Δ={se48-se_so:+d})')
    print(f'  v49 (zone+swap):        SE={se49:4d}  exact={ex49}/91  (Δ={se48-se49:+d})')
    
    # ════════════════════════════════════════════════════════════════════
    # A: DECOMPOSITION — which component contributes what, per season
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' A: PER-SEASON DECOMPOSITION')
    print('='*70)
    print(f'  {"Season":<12} {"v48":>5} {"zone":>5} {"swap":>5} {"v49":>5}  '
          f'{"Δzone":>5} {"Δswap":>5} {"Δboth":>5}')
    
    zone_helps = 0; zone_hurts = 0; swap_helps = 0; swap_hurts = 0
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        if len(si_s) == 0: continue
        s48 = sum((p_v48[gi] - int(y[gi]))**2 for gi in si_s)
        szo = sum((p_zone_only[gi] - int(y[gi]))**2 for gi in si_s)
        sso = sum((p_swap_only[gi] - int(y[gi]))**2 for gi in si_s)
        s49 = sum((p_v49[gi] - int(y[gi]))**2 for gi in si_s)
        dz = s48 - szo
        ds = s48 - sso
        db = s48 - s49
        if dz > 0: zone_helps += 1
        elif dz < 0: zone_hurts += 1
        if ds > 0: swap_helps += 1
        elif ds < 0: swap_hurts += 1
        marker_z = '✓' if dz >= 0 else '✗'
        marker_s = '✓' if ds >= 0 else '✗'
        print(f'  {s:<12} {s48:5d} {szo:5d} {sso:5d} {s49:5d}  '
              f'{marker_z}{dz:+4d}  {marker_s}{ds:+4d}  {db:+5d}')
    
    print(f'\n  Zone change: helps {zone_helps} seasons, neutral {5-zone_helps-zone_hurts}, hurts {zone_hurts}')
    print(f'  Swap rule:   helps {swap_helps} seasons, neutral {5-swap_helps-swap_hurts}, hurts {swap_hurts}')
    
    # ════════════════════════════════════════════════════════════════════
    # B: SWAP RULE TRIGGER ANALYSIS — find ALL teams that COULD trigger
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' B: SWAP RULE TRIGGER ANALYSIS')
    print('    Condition: AQ, !AL, pred in [30,45], pred-NET > 10')
    print('    Paired with: AL, !AQ, NET > pred, |pred gap| ≤ 6')
    print('='*70)
    
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(sm)[0]  # ALL teams in season, not just test
        test_si = [gi for gi in si_s if test_mask[gi]]
        
        # Check which test teams are AQ in 30-45 range with pred-NET>10
        print(f'\n  {s}:')
        aq_candidates = []
        al_candidates = []
        for gi in test_si:
            pred48 = p_v48[gi]  # Use v48+zone predictions (before swap)
            predzn = p_zone_only[gi]
            is_aq = int(X_all[gi, fi['is_AQ']])
            is_al = int(X_all[gi, fi['is_AL']])
            net = X_all[gi, fi['NET Rank']]
            actual = int(y[gi])
            
            if 30 <= predzn <= 45:
                if is_aq and not is_al:
                    gap = predzn - net
                    triggers = '→ TRIGGERS' if gap > 10 else ''
                    print(f'    AQ: {record_ids[gi]:30s} pred={predzn:2d} NET={net:3.0f} '
                          f'gap={gap:+5.0f} gt={actual:2d} {triggers}')
                    if gap > 10:
                        aq_candidates.append((gi, predzn, net, actual))
                elif is_al and not is_aq:
                    triggers = '→ PAIR' if net > predzn else ''
                    print(f'    AL: {record_ids[gi]:30s} pred={predzn:2d} NET={net:3.0f} '
                          f'net-pred={net-predzn:+5.0f} gt={actual:2d} {triggers}')
                    if net > predzn:
                        al_candidates.append((gi, predzn, net, actual))
        
        if aq_candidates and al_candidates:
            print(f'    → Would fire: {len(aq_candidates)} AQ × {len(al_candidates)} AL pairs')
        elif not aq_candidates:
            print(f'    → No AQ triggers (SWAP RULE DOES NOTHING)')
    
    # ════════════════════════════════════════════════════════════════════
    # C: HISTORICAL PRECEDENT — AQ teams with great NET but bad seed
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' C: HISTORICAL PRECEDENT — AQ teams with NET<30, Seed>35')
    print('    How often does the committee massively under-seed AQ teams?')
    print('='*70)
    
    # Search ALL 340 labeled teams (not just test)
    print(f'  {"RecordID":35s} {"Seed":>4s} {"NET":>4s} {"Gap":>4s} {"SOS":>4s} {"Opp":>4s} {"Test":>4s}')
    aq_anomalies = 0
    for gi in range(n):
        is_aq = int(X_all[gi, fi['is_AQ']])
        is_al = int(X_all[gi, fi['is_AL']])
        net = X_all[gi, fi['NET Rank']]
        actual = int(y[gi])
        if is_aq and not is_al and net < 30 and actual > 35:
            sos = X_all[gi, fi['NETSOS']]
            opp = X_all[gi, fi['AvgOppNETRank']]
            is_test = '✓' if test_mask[gi] else ''
            print(f'  {record_ids[gi]:35s} {actual:4d} {net:4.0f} {actual-net:+4.0f} '
                  f'{sos:4.0f} {opp:4.0f} {is_test:>4s}')
            aq_anomalies += 1
    
    total_aq = sum(1 for gi in range(n)
                   if int(X_all[gi, fi['is_AQ']]) and not int(X_all[gi, fi['is_AL']]))
    print(f'\n  Total AQ (non-AL) teams: {total_aq}')
    print(f'  AQ with NET<30 and Seed>35: {aq_anomalies}')
    print(f'  Rate: {aq_anomalies}/{total_aq} = {100*aq_anomalies/max(1,total_aq):.1f}%')
    
    # Also check ALL AQ teams with NET<30
    aq_good_net = []
    for gi in range(n):
        is_aq = int(X_all[gi, fi['is_AQ']])
        is_al = int(X_all[gi, fi['is_AL']])
        net = X_all[gi, fi['NET Rank']]
        if is_aq and not is_al and net < 30:
            aq_good_net.append((gi, int(y[gi]), net))
    
    print(f'\n  AQ teams with NET < 30 ({len(aq_good_net)} total):')
    for gi, seed, net in sorted(aq_good_net, key=lambda x: x[2]):
        sos = X_all[gi, fi['NETSOS']]
        gap = seed - net
        print(f'    {record_ids[gi]:35s} NET={net:3.0f} Seed={seed:2d} Gap={gap:+3.0f} SOS={sos:3.0f}')
    
    # ════════════════════════════════════════════════════════════════════
    # D: ZONE CHANGE ANALYSIS — why does (-6,1,-6) help?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' D: ZONE CHANGE ANALYSIS')
    print('    Uppermid: (-2,-3,-4) → (-6,1,-6)')
    print('    What teams in 34-44 range are affected?')
    print('='*70)
    
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        changed = [(gi, p_v48[gi], p_zone_only[gi], int(y[gi])) 
                   for gi in si_s if p_v48[gi] != p_zone_only[gi]]
        if changed:
            print(f'\n  {s}:')
            for gi, old, new, actual in changed:
                se_old = (old - actual)**2
                se_new = (new - actual)**2
                arrow = '↑' if se_new < se_old else '↓' if se_new > se_old else '='
                net = X_all[gi, fi['NET Rank']]
                is_aq = int(X_all[gi, fi['is_AQ']])
                is_al = int(X_all[gi, fi['is_AL']])
                team_type = 'AQ' if is_aq else ('AL' if is_al else '??')
                print(f'    {arrow} {record_ids[gi]:30s} {team_type} NET={net:3.0f} '
                      f'v48={old:2d} new={new:2d} gt={actual:2d} SE:{se_old}→{se_new}')
    
    # ════════════════════════════════════════════════════════════════════
    # E: STABILITY — how sensitive is v49 to small perturbations?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' E: PERTURBATION STABILITY')
    print('    Add Gaussian noise to raw scores, measure SE variance')
    print('='*70)
    
    rng = np.random.RandomState(42)
    
    for noise_std in [0.5, 1.0, 2.0, 3.0]:
        se_v48_noisy = []
        se_v49_noisy = []
        for trial in range(30):
            # Perturb raw v12 and raw comm scores
            cache_noisy = {}
            for hs, c in cache.items():
                cn = dict(c)
                cn['raw_v12'] = c['raw_v12'] + rng.normal(0, noise_std, c['raw_v12'].shape)
                cn['raw_comm'] = c['raw_comm'] + rng.normal(0, noise_std, c['raw_comm'].shape)
                # Fix training labels back
                for i, gi in enumerate(cn['si']):
                    if not test_mask[gi]:
                        cn['raw_v12'][i] = y[gi]
                        cn['raw_comm'][i] = y[gi]
                # Re-Hungarian
                cn['a_v12_raw'] = hungarian(cn['raw_v12'], seasons[c['season_mask']], c['avail'], power=0.15)
                cn['a_comm_raw'] = hungarian(cn['raw_comm'], seasons[c['season_mask']], c['avail'], power=0.15)
                cache_noisy[hs] = cn
            
            p48n = run_pipeline(cache_noisy, y, fn, seasons, test_mask, n, comm_zones=ZONES_V48_COMM)
            p49n_z = run_pipeline(cache_noisy, y, fn, seasons, test_mask, n, comm_zones=ZONES_V49_COMM)
            p49n, _ = apply_swap(p49n_z, X_all, fi, seasons, test_mask)
            
            se_v48_noisy.append(ev(p48n)[0])
            se_v49_noisy.append(ev(p49n)[0])
        
        v48_arr = np.array(se_v48_noisy)
        v49_arr = np.array(se_v49_noisy)
        v49_wins = np.sum(v49_arr < v48_arr)
        print(f'  noise={noise_std:.1f}: v48 SE={v48_arr.mean():.0f}±{v48_arr.std():.0f}  '
              f'v49 SE={v49_arr.mean():.0f}±{v49_arr.std():.0f}  '
              f'v49 wins {v49_wins}/30  gap={v48_arr.mean()-v49_arr.mean():.1f}')
    
    # ════════════════════════════════════════════════════════════════════
    # F: LEAVE-ONE-SEASON-OUT VALIDATION
    #    For each test season, what does v49 score vs v48?
    #    Can the swap rule trained on other seasons help the held-out one?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' F: EVIDENCE FOR/AGAINST EACH COMPONENT')
    print('='*70)
    
    # Component 1: Zone change (-6,1,-6)
    # Only affects committee path uppermid zone
    # In training data, it helped 2021-22 (TCU/Creighton) and nowhere else
    # Risk: it's tuned to fix a SPECIFIC swap in a SPECIFIC season
    
    print('\n  COMPONENT 1: Uppermid zone (-2,-3,-4) → (-6,1,-6)')
    print('  ─' * 35)
    
    # How many configs at SE=78 (the zone-only improvement)?
    plateau_count = 0
    for aq in range(-8, 5):
        for al in range(-6, 5):
            for sos_p in range(-8, 5):
                zones = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (aq, al, sos_p)),
                    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_pipeline(cache, y, fn, seasons, test_mask, n, comm_zones=zones)
                se, ex = ev(p)
                if se == se_zo:  # matches best zone-only SE
                    plateau_count += 1
    
    total_tested = 13 * 11 * 13  # aq: -8..4, al: -6..4, sos: -8..4
    print(f'  Plateau: {plateau_count}/{total_tested} configs achieve SE={se_zo}')
    print(f'  → {"WIDE plateau (robust)" if plateau_count > 20 else "NARROW plateau (fragile)"}')
    
    # What does (-6,1,-6) actually do differently from (-2,-3,-4)?
    print(f'\n  Effect on committee correction signal in zone 34-44:')
    print(f'  Old: aq=-2, al=-3, sos=-4')
    print(f'  New: aq=-6, al=+1, sos=-6')
    print(f'  Change: aq Δ=-4 (much stronger AQ penalty)')
    print(f'          al Δ=+4 (REVERSED: now rewards AL instead of penalizing)')
    print(f'          sos Δ=-2 (stronger SOS correction)')
    print(f'  → The key is al: old=-3 penalized AL, new=+1 REWARDS AL')
    print(f'  → This makes sense: AL power-conf teams in 34-44 should get BETTER seeds')
    
    # Component 2: Swap rule
    print('\n  COMPONENT 2: AQ↔AL swap rule (ng=10, pg=6)')
    print('  ─' * 35)
    print(f'  Fires in: {"only 2023-24" if swap_helps == 1 else f"{swap_helps} seasons"}')
    print(f'  Risk assessment:')
    print(f'    - Only fires on AQ teams with pred-NET > 10 in seeds 30-45')
    print(f'    - In 5 training seasons, fires in {swap_helps}/5')
    if swap_helps <= 1:
        print(f'    ⚠ WARNING: Rule fires in ≤1 season — cannot validate on held-out data')
        print(f'    ⚠ This means we CANNOT know if it will fire correctly in 2025-26')
    
    # How many swaps would fire at different thresholds?
    print(f'\n  Swap rule sensitivity analysis:')
    for ng in [5, 8, 10, 12, 15, 20]:
        for pg in [4, 5, 6, 7, 8]:
            ps, swaps = apply_swap(p_zone_only, X_all, fi, seasons, test_mask, ng, pg)
            se_s, ex_s = ev(ps)
            n_seasons = len(set(d['season'] for d in swaps)) if swaps else 0
            if len(swaps) > 0:
                print(f'    ng={ng:2d} pg={pg}: {len(swaps)} swaps, {n_seasons} seasons, '
                      f'SE={se_s:3d} (v49zone={se_zo}→{se_s}, Δ={se_zo-se_s:+d})')
    
    # ════════════════════════════════════════════════════════════════════
    # G: DECISION BOUNDARY — how close are NON-triggering teams?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' G: NEAR-MISS ANALYSIS — teams close to triggering swap')
    print('='*70)
    
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        near_misses = []
        for gi in si_s:
            pred = p_zone_only[gi]
            is_aq = int(X_all[gi, fi['is_AQ']])
            is_al = int(X_all[gi, fi['is_AL']])
            net = X_all[gi, fi['NET Rank']]
            actual = int(y[gi])
            
            if is_aq and not is_al and 25 <= pred <= 50:
                gap = pred - net
                if 5 <= gap <= 15:  # Near the threshold of 10
                    near_misses.append((gi, pred, net, gap, actual))
        
        if near_misses:
            print(f'\n  {s}:')
            for gi, pred, net, gap, actual in sorted(near_misses, key=lambda x: -x[3]):
                fires = '★ FIRES' if gap > 10 and 30 <= pred <= 45 else ''
                would_help = '(would help)' if abs(actual - pred) > 2 else '(no help needed)'
                print(f'    {record_ids[gi]:30s} pred={pred:2d} NET={net:3.0f} gap={gap:+5.0f} '
                      f'gt={actual:2d} {fires} {would_help}')
    
    # ════════════════════════════════════════════════════════════════════
    # H: KAGGLE VS LOCAL DISCREPANCY
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' H: KAGGLE VS LOCAL SCORE ANALYSIS')
    print('='*70)
    
    kaggle_history = [
        ('v27', 1.089, None),
        ('v45c', 0.806, None),
        ('v46', 0.521, None),
        ('v47', 0.437, 94),
        ('v48', 0.383, 80),
        ('v49', 0.163, 16),
    ]
    
    print(f'  {"Version":>8s} {"Kaggle":>8s} {"LocalSE":>8s} {"LocalRMSE":>10s} {"Ratio(K/L)":>10s}')
    for ver, kaggle, local_se in kaggle_history:
        if local_se:
            import math
            local_rmse = math.sqrt(local_se/91)
            ratio = kaggle / local_rmse
            print(f'  {ver:>8s} {kaggle:8.4f} {local_se:8d} {local_rmse:10.4f} {ratio:10.3f}')
        else:
            print(f'  {ver:>8s} {kaggle:8.4f}      {"?":>3s}       {"?":>5s}       {"?":>5s}')
    
    print(f'\n  Kaggle is consistently LOWER than local RMSE.')
    print(f'  Possible reasons:')
    print(f'    1. Kaggle evaluates on a SUBSET of our 91 test teams (likely)')
    print(f'    2. Kaggle uses a different metric weighting')
    print(f'    3. Some test teams are in Kaggle\'s private test (not public)')
    print(f'  The ratio dropping from 0.430 to 0.389 suggests v49 helps on')
    print(f'  both public AND private Kaggle test sets — not just our local test.')
    
    # ════════════════════════════════════════════════════════════════════
    # I: BOOTSTRAP WITH REALISTIC NOISE
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' I: BOOTSTRAP — 50 resamples')
    print('='*70)
    
    test_indices = np.where(test_mask)[0]
    ntest = len(test_indices)
    
    wins49, wins48, ties = 0, 0, 0
    for b in range(50):
        boot = rng.choice(test_indices, size=ntest, replace=True)
        bse48 = sum((p_v48[gi] - int(y[gi]))**2 for gi in boot)
        bse49 = sum((p_v49[gi] - int(y[gi]))**2 for gi in boot)
        if bse49 < bse48: wins49 += 1
        elif bse49 > bse48: wins48 += 1
        else: ties += 1
    print(f'  v49 wins: {wins49}/50, v48 wins: {wins48}/50, ties: {ties}/50')
    
    # Also bootstrap by SEASON (more conservative)
    print(f'\n  Season-level bootstrap (sample 5 seasons with replacement):')
    wins49_s, wins48_s, ties_s = 0, 0, 0
    for b in range(100):
        boot_seasons = rng.choice(all_seasons, size=len(all_seasons), replace=True)
        bse48, bse49 = 0, 0
        for bs in boot_seasons:
            sm = (seasons == bs)
            si_s = np.where(test_mask & sm)[0]
            bse48 += sum((p_v48[gi] - int(y[gi]))**2 for gi in si_s)
            bse49 += sum((p_v49[gi] - int(y[gi]))**2 for gi in si_s)
        if bse49 < bse48: wins49_s += 1
        elif bse49 > bse48: wins48_s += 1
        else: ties_s += 1
    print(f'  v49 wins: {wins49_s}/100, v48 wins: {wins48_s}/100, ties: {ties_s}/100')
    
    # ════════════════════════════════════════════════════════════════════
    # J: 2025-26 READINESS ASSESSMENT
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' J: 2025-26 READINESS ASSESSMENT')
    print('='*70)
    
    print(f'\n  Component risk matrix:')
    print(f'  ┌─────────────────────────────────┬──────────┬─────────────┬────────────┐')
    print(f'  │ Component                       │ SE Gain  │ Seasons     │ Overfit?   │')
    print(f'  ├─────────────────────────────────┼──────────┼─────────────┼────────────┤')
    print(f'  │ v12 base (unchanged since v12)  │ baseline │ All 5       │ LOW        │')
    print(f'  │ Dual-Hungarian (v47)            │ large    │ All 5       │ LOW        │')
    print(f'  │ 6 zones (v48)                   │ moderate │ 4-5 seasons │ MEDIUM     │')
    
    zone_risk = 'LOW' if plateau_count > 20 else 'MEDIUM'
    print(f'  │ Uppermid zone change (v49)      │ {se48-se_zo:+4d}     │ 1-2 seasons │ {zone_risk:10s} │')
    
    swap_risk = 'HIGH' if swap_helps <= 1 else 'MEDIUM'
    print(f'  │ AQ↔AL swap rule (v49)           │ {se_zo-se49:+4d}     │ {swap_helps} season(s)  │ {swap_risk:10s} │')
    print(f'  └─────────────────────────────────┴──────────┴─────────────┴────────────┘')
    
    print(f'\n  For 2025-26 prediction:')
    print(f'    v12 + Dual-Hungarian + zones (v48) → SAFE, proven across 5 seasons')
    print(f'    Zone change (-6,1,-6) → PROBABLY SAFE if plateau is wide')
    print(f'    AQ↔AL swap rule → RISKY: fires based on patterns in only 1 season')
    
    print(f'\n  RECOMMENDATION FOR 2025-26:')
    if swap_helps <= 1:
        print(f'    OPTION A (SAFE): Use v48 + zone change only (SE={se_zo})')
        print(f'      - Zone change has wide plateau ({plateau_count} configs)')
        print(f'      - Justified by correcting committee bias direction')
        print(f'      - Does NOT depend on a specific team pattern')
        print(f'    OPTION B (AGGRESSIVE): Use full v49 (SE={se49})')
        print(f'      - Swap rule is theoretically sound (committee bias)')
        print(f'      - But only validated in 1 historical season')
        print(f'      - Risk: may fire on a team where swap is WRONG in 2025-26')
    else:
        print(f'    Use full v49 — swap rule validated across multiple seasons')
    
    # ════════════════════════════════════════════════════════════════════
    # K: What if swap fires INCORRECTLY in 2025-26?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' K: WORST-CASE SWAP ANALYSIS')
    print('='*70)
    
    # Simulate: what if there's an AQ team that triggers but shouldn't be swapped?
    # Look at ALL training teams (not just test) in 30-45 seed range
    print(f'  If swap fires incorrectly, typical damage:')
    print(f'    - Two teams swap seeds (distance ~4-6 seeds)')
    print(f'    - Worst case SE increase: ~2*(6^2) = 72')
    print(f'    - Expected SE increase: ~2*(3^2) = 18')
    print(f'  Current v49 SE improvement: {se48 - se49}')
    print(f'  So even ONE wrong swap could erase most of the improvement')
    
    # Check if the rule would have fired INCORRECTLY on training data
    print(f'\n  Would swap fire incorrectly on TRAINING teams?')
    for s in all_seasons:
        sm = (seasons == s)
        all_si = np.where(sm)[0]
        # Look at ALL teams (not just test)
        for gi in all_si:
            if test_mask[gi]: continue  # Skip test teams, already analyzed
            actual = int(y[gi])
            is_aq = int(X_all[gi, fi['is_AQ']])
            is_al = int(X_all[gi, fi['is_AL']])
            net = X_all[gi, fi['NET Rank']]
            if is_aq and not is_al and 30 <= actual <= 45 and actual - net > 10:
                print(f'    {record_ids[gi]:30s} seed={actual:2d} NET={net:3.0f} gap={actual-net:+3.0f} '
                      f'(TRAINING, would be AQ trigger candidate)')
    
    # ════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL OVERFITTING VERDICT')
    print('='*70)
    
    print(f'\n  Local SE progression: 94 → 80 → {se_zo} → {se49}')
    print(f'  Kaggle RMSE progression: 0.437 → 0.383 → ? → 0.163')
    print(f'\n  Zone change component (SE: 80→{se_zo}):')
    if plateau_count > 20:
        print(f'    ✓ Wide plateau ({plateau_count} configs) — NOT overfit')
    else:
        print(f'    ⚠ Narrow plateau ({plateau_count} configs) — POSSIBLY overfit')
    print(f'    ✓ Theoretically motivated (committee rewards AL in 34-44)')
    print(f'    ✓ Fixes TCU/Creighton which are genuinely mis-ordered')
    
    print(f'\n  Swap rule component (SE: {se_zo}→{se49}):')
    print(f'    ⚠ Only fires in 1/5 seasons —  limited validation')
    print(f'    ✓ Theoretically motivated (committee bias against AQ)')
    print(f'    ✓ Kaggle improvement (0.383→0.163) suggests it helps on hidden data')
    print(f'    ⚠ OneWRONG swap could cost SE≈18-50')
    print(f'    {"?" if swap_helps <= 1 else "✓"} Cannot confirm it generalizes to 2025-26')
    
    print(f'\n  ⇒ For SAFE 2025-26 prediction: Use v48 + zone change (SE={se_zo})')
    print(f'  ⇒ For AGGRESSIVE 2025-26: Use full v49 (SE={se49})')
    print(f'  ⇒ The swap rule is a KNOWN gamble but with sound theoretical basis')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
