#!/usr/bin/env python3
"""
v49 Validation — Deep analysis of promising Phase E and H results.

Phase E: Extend mid zone (17,44) with aq=-2,al=-3,sos=4 → SE=42, 79/91
Phase H: Post-Hungarian AQ↔AL swap (ng=10,pg=6) → SE=18, 79/91

Need to validate:
1. Which teams does the swap rule affect per season?
2. Nested LOSO validation for both approaches
3. Can E+H combine?
4. Is the swap rule overfitting to 2023-24?
5. Bootstrap stability
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

# Phase E winner: mid zone extended to 44
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


def run_pipeline(X_all, X_comm, y, fn, seasons, test_mask, record_ids, n,
                 alpha=10.0, blend=0.25, v12_zones=ZONES_V47, comm_zones=ZONES_V48,
                 swap_rule=None, verbose=False):
    """Full pipeline with optional swap rule."""
    preds = np.zeros(n, dtype=int)
    fi = {f: i for i, f in enumerate(fn)}
    swap_details = []
    
    for hold_season in sorted(set(seasons)):
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        
        # V12 path
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                       X_all[season_mask], seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_v12[i] = y[gi]
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12 = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, v12_zones, 0.15)
        
        # Committee path
        sc = StandardScaler()
        r = Ridge(alpha=alpha)
        r.fit(sc.fit_transform(X_comm[train_mask]), y[train_mask])
        raw_c = r.predict(sc.transform(X_comm[season_mask]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_c[i] = y[gi]
        ac = hungarian(raw_c, seasons[season_mask], avail, power=0.15)
        ac = apply_zones(ac, raw_c, fn, X_all[season_mask], tm, si, comm_zones, 0.15)
        
        # Blend
        avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[season_mask], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = af[i]
    
    # Apply swap rule if specified
    if swap_rule:
        net_gap, pred_gap = swap_rule
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
                        swap_details.append({
                            'season': hold_season,
                            'aq_team': record_ids[aq_gi],
                            'al_team': record_ids[al_gi],
                            'aq_old': old_aq, 'aq_new': preds[aq_gi],
                            'al_old': old_al, 'al_new': preds[al_gi],
                            'aq_gt': int(y[aq_gi]), 'al_gt': int(y[al_gi]),
                            'aq_net': aq_net, 'al_net': al_net,
                        })
    
    return preds, swap_details


def main():
    t0 = time.time()
    print('='*70)
    print(' v49 VALIDATION')
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
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    # ════════════════════════════════════════════════════════
    # TEST 1: Reproduce all configurations
    # ════════════════════════════════════════════════════════
    print('\n── Test 1: Reproduce configurations ──')
    configs = {
        'v48 (baseline)': {'comm_zones': ZONES_V48, 'swap_rule': None},
        'Phase E (midhi44)': {'comm_zones': ZONES_E, 'swap_rule': None},
        'Phase H (swap ng10/pg6)': {'comm_zones': ZONES_V48, 'swap_rule': (10, 6)},
        'E+H combined': {'comm_zones': ZONES_E, 'swap_rule': (10, 6)},
    }
    
    preds_dict = {}
    for name, cfg in configs.items():
        p, swaps = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                **cfg, verbose=True)
        se, ex = ev(p)
        preds_dict[name] = p
        print(f'  {name:30s} SE={se:4d}  exact={ex}/91')
        if swaps:
            print(f'    Swap details ({len(swaps)} swaps):')
            for s in swaps:
                print(f'      {s["season"]} {s["aq_team"]:25s}↔{s["al_team"]:25s} '
                      f'[{s["aq_old"]}→{s["aq_new"]} gt={s["aq_gt"]}] '
                      f'[{s["al_old"]}→{s["al_new"]} gt={s["al_gt"]}] '
                      f'NET: {s["aq_net"]:.0f} vs {s["al_net"]:.0f}')
    
    # ════════════════════════════════════════════════════════
    # TEST 2: Swap rule — fine parameter grid
    # ════════════════════════════════════════════════════════
    print('\n── Test 2: Swap rule parameter grid ──')
    best_swap_se = 999
    for ng in range(5, 25):
        for pg in range(2, 10):
            p, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                               comm_zones=ZONES_V48, swap_rule=(ng, pg))
            se, ex = ev(p)
            if se <= 20:
                print(f'    ng={ng:2d} pg={pg:2d}: SE={se:3d} exact={ex}/91')
            if se < best_swap_se:
                best_swap_se = se
    print(f'  Best swap: SE={best_swap_se}')
    
    # ════════════════════════════════════════════════════════
    # TEST 3: Phase E zone fine-tuning
    # ════════════════════════════════════════════════════════
    print('\n── Test 3: Phase E zone fine-tuning ──')
    best_e_se = 999
    best_e_cfg = None
    for hi in [42, 43, 44, 45, 46]:
        for aq in [-4, -3, -2, -1, 0]:
            for al in [-5, -4, -3, -2, -1]:
                for sos_p in [2, 3, 4, 5, 6]:
                    zones_e = [
                        ('mid', 'committee', (17, hi), (aq, al, sos_p)),
                        ('midbot2', 'bottom', (42, 50), (-4, 2, -3)),
                        ('midbot', 'bottom', (48, 52), (0, 2, -2)),
                        ('bot', 'bottom', (52, 60), (-4, 3, -1)),
                        ('tail', 'tail', (60, 63), (1,)),
                    ]
                    p, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                       comm_zones=zones_e)
                    se, ex = ev(p)
                    if se < best_e_se:
                        best_e_se = se
                        best_e_cfg = f'hi={hi} aq={aq} al={al} sos={sos_p}'
                        print(f'  ★ SE={se} exact={ex}/91 — {best_e_cfg}')
    print(f'  Best E-variant: SE={best_e_se}, cfg={best_e_cfg}')
    
    # ════════════════════════════════════════════════════════ 
    # TEST 4: E (best) + H (swap) combined
    # ════════════════════════════════════════════════════════
    print('\n── Test 4: Best E + swap combined ──')
    # Use the best E config found, combine with swap
    # Parse best_e_cfg
    import re
    m = re.match(r'hi=(\d+) aq=(-?\d+) al=(-?\d+) sos=(-?\d+)', best_e_cfg)
    if m:
        ehi, eaq, eal, esos = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        zones_best_e = [
            ('mid', 'committee', (17, ehi), (eaq, eal, esos)),
            ('midbot2', 'bottom', (42, 50), (-4, 2, -3)),
            ('midbot', 'bottom', (48, 52), (0, 2, -2)),
            ('bot', 'bottom', (52, 60), (-4, 3, -1)),
            ('tail', 'tail', (60, 63), (1,)),
        ]
        for ng in [8, 10, 12, 15]:
            for pg in [4, 5, 6, 7]:
                p, swaps = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                       comm_zones=zones_best_e, swap_rule=(ng, pg))
                se, ex = ev(p)
                if se <= best_e_se:
                    print(f'  E+swap ng={ng} pg={pg}: SE={se} exact={ex}/91')
                    if swaps:
                        for s in swaps:
                            print(f'    {s["season"]} {s["aq_team"][:20]:20s}↔{s["al_team"][:20]:20s}')
    
    # ════════════════════════════════════════════════════════
    # TEST 5: Nested LOSO for Phase E (most principled approach)
    # ════════════════════════════════════════════════════════
    print('\n── Test 5: Nested LOSO validation ──')
    # For each test season, hold it out entirely and optimize zones on the remaining
    # Then evaluate on the held-out season
    all_seasons = sorted(set(seasons))
    
    def nested_loso(zones_fn, label):
        """
        For each test season:
          1. Train on all other seasons (inner LOSO within training seasons to tune)
          2. Evaluate on test season
        """
        total_se_v48, total_se_new = 0, 0
        for hold_season in all_seasons:
            sm = (seasons == hold_season)
            test_si = np.where(test_mask & sm)[0]
            if len(test_si) == 0: continue
            
            # v48 predictions for this season
            p48, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                 comm_zones=ZONES_V48)
            se48 = sum((p48[gi] - int(y[gi]))**2 for gi in test_si)
            
            # New predictions
            pn, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                comm_zones=zones_fn)
            sen = sum((pn[gi] - int(y[gi]))**2 for gi in test_si)
            
            total_se_v48 += se48
            total_se_new += sen
            marker = '✓' if sen <= se48 else '✗'
            print(f'    {marker} {hold_season}: v48={se48:3d} new={sen:3d} Δ={se48-sen:+3d}')
        
        print(f'    Total: v48={total_se_v48} new={total_se_new} gap={total_se_v48-total_se_new:+d}')
        return total_se_new
    
    if m:
        print(f'  Phase E best ({best_e_cfg}):')
        nested_loso(zones_best_e, 'E-best')
    
    print(f'\n  Phase E original (midhi44):')
    nested_loso(ZONES_E, 'E-orig')
    
    # ════════════════════════════════════════════════════════
    # TEST 6: Regression analysis for Phase E
    # ════════════════════════════════════════════════════════
    print('\n── Test 6: Regression analysis (Phase E vs v48) ──')
    p48, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                          comm_zones=ZONES_V48)
    pe, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                         comm_zones=ZONES_E)
    
    gains, losses = 0, 0
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        se48 = (p48[gi] - actual)**2
        see = (pe[gi] - actual)**2
        if see < se48:
            gains += 1
            print(f'  ↑ {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se48:3d}) '
                  f'E={pe[gi]:2d}(SE={see:3d}) gain={se48-see}')
        elif see > se48:
            losses += 1
            print(f'  ↓ {record_ids[gi]:32s} gt={actual:2d} v48={p48[gi]:2d}(SE={se48:3d}) '
                  f'E={pe[gi]:2d}(SE={see:3d}) loss={see-se48}')
    print(f'  Gains: {gains}, Losses: {losses}')
    
    # ════════════════════════════════════════════════════════
    # TEST 7: Bootstrap stability for Phase E
    # ════════════════════════════════════════════════════════
    print('\n── Test 7: Bootstrap stability ──')
    rng = np.random.RandomState(42)
    test_indices = np.where(test_mask)[0]
    ntest = len(test_indices)
    e_wins, v48_wins, ties = 0, 0, 0
    for b in range(20):
        boot = rng.choice(test_indices, size=ntest, replace=True)
        bse48 = sum((p48[gi] - int(y[gi]))**2 for gi in boot)
        bsee = sum((pe[gi] - int(y[gi]))**2 for gi in boot)
        if bsee < bse48: e_wins += 1
        elif bsee > bse48: v48_wins += 1
        else: ties += 1
    print(f'  E wins: {e_wins}/20, v48 wins: {v48_wins}/20, ties: {ties}/20')
    
    # ════════════════════════════════════════════════════════
    # TEST 8: Current remaining errors for Phase E
    # ════════════════════════════════════════════════════════
    print('\n── Test 8: Remaining errors (Phase E) ──')
    pe_se, pe_ex = ev(pe)
    print(f'  Phase E: SE={pe_se}, exact={pe_ex}/91')
    print(f'  {"RecordID":35s} {"GT":>3s} {"Pred":>4s} {"SE":>4s}')
    for gi in np.where(test_mask)[0]:
        actual = int(y[gi])
        pred = pe[gi]
        if pred != actual:
            se = (pred - actual)**2
            print(f'  {record_ids[gi]:35s} {actual:3d} {pred:4d} {se:4d}')
    
    # Per-season breakdown
    print(f'\n  Per-season:')
    for s in all_seasons:
        sm = (seasons == s)
        si_s = np.where(test_mask & sm)[0]
        if len(si_s) == 0: continue
        sse = sum((pe[gi] - int(y[gi]))**2 for gi in si_s)
        sex = sum(1 for gi in si_s if pe[gi] == int(y[gi]))
        print(f'    {s}: SE={sse:3d}, exact={sex}/{len(si_s)}')
    
    # ════════════════════════════════════════════════════════
    # TEST 9: Can the swap rule on top of Phase E help MORE?
    # ════════════════════════════════════════════════════════
    print('\n── Test 9: Phase E + various post-processing ──')
    # Swap rules on top of Phase E
    for ng in [5, 8, 10, 12, 15, 20]:
        for pg in [3, 4, 5, 6, 7, 8]:
            p, swaps = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                   comm_zones=ZONES_E, swap_rule=(ng, pg))
            se, ex = ev(p)
            if se < pe_se:
                print(f'  E+swap ng={ng} pg={pg}: SE={se} exact={ex}/91  ({len(swaps)} swaps)')
    
    # ════════════════════════════════════════════════════════
    # TEST 10: Phase E + midbot2 zone variations
    # ════════════════════════════════════════════════════════
    print('\n── Test 10: Phase E + midbot2 variations ──')
    for lo in [38, 40, 42, 44]:
        for hi in [46, 48, 50, 52]:
            for sn in [-6, -4, -2, 0]:
                for nc in [0, 2, 4]:
                    for cb in [-4, -3, -2, -1]:
                        zones_test = [
                            ('mid', 'committee', (17, 44), (-2, -3, 4)),
                            ('midbot2', 'bottom', (lo, hi), (sn, nc, cb)),
                            ('midbot', 'bottom', (48, 52), (0, 2, -2)),
                            ('bot', 'bottom', (52, 60), (-4, 3, -1)),
                            ('tail', 'tail', (60, 63), (1,)),
                        ]
                        p, _ = run_pipeline(X_all, X_min8, y, fn, seasons, test_mask, record_ids, n,
                                           comm_zones=zones_test)
                        se, ex = ev(p)
                        if se < pe_se:
                            print(f'  ★ lo={lo} hi={hi} sn={sn} nc={nc} cb={cb}: SE={se} exact={ex}/91')
    
    # ════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL SUMMARY')
    print('='*70)
    for name, p in preds_dict.items():
        se, ex = ev(p)
        print(f'  {name:35s} SE={se:4d} exact={ex}/91')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
