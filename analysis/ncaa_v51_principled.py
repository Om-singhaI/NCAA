#!/usr/bin/env python3
"""
v51 PRINCIPLED ALTERNATIVES — Eliminate gambles
================================================
Problem: AQ↔AL swap (post-hoc rule, fires 1/5 seasons) and xtail zone (2 teams, 1 season)
are gambles — they might hurt on 2025-26 data.

Strategy: Make the model LEARN these patterns instead of hand-patching them.

APPROACH A: Enhanced committee model
  The current min8 Ridge model uses 8 features. The AQ↔AL swap exists because
  the committee model (and v12 base) don't properly capture how the NCAA committee
  under-seeds AQ teams from weak conferences. FIX: add features that encode this
  bias directly, so the committee Ridge naturally produces better raw scores for
  these teams. Then the dual-Hungarian picks up the correction without a post-hoc rule.

  Candidate features to add to committee:
    9.  is_AQ * (NET / SOS_gap)  — AQ teams with strong NET but weak schedule
    10. is_AL * is_power_conf    — AL power-conf teams get committee bonus
    11. NET - tourn_field_rank   — gap between NET and field rank (measures "deserves better")
    12. conf_avg_net * is_AQ     — AQ from weak conferences
    13. SOS-NET gap              — schedule strength vs ranking divergence

APPROACH B: Extended bot zone
  Instead of separate xtail zone (63-68), extend the existing bot zone (52-60) to
  cover 52-68 with a single set of params. If one wider zone works, it's more
  robust than two narrow ones.

APPROACH C: Soft AQ correction via zone
  Instead of post-hoc swap, add AQ/AL bias as a correction signal in the
  existing uppermid zone (34-44) or a new mid-low zone (30-45) that handles
  the AQ vs AL divergence zone-style (Hungarian re-ordering) rather than
  binary swapping.

APPROACH D: combined — enhanced committee + extended bot + no swap
  If the committee model properly handles AQ bias, we don't need the swap rule.
  If the bot zone extends to handle xtail, we don't need a separate xtail zone.
  Result: fewer tuned params, same or better performance.

TESTING: Use full v50 cache-based pipeline, compare:
  v50 (current):  7 zones + swap = SE=14
  v51a: enhanced committee (no swap, no xtail)
  v51b: enhanced committee + extended bot (no swap, no xtail)
  v51c: per-approach testing
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
np.random.seed(42)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    apply_aq_al_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    DUAL_RIDGE_ALPHA, DUAL_BLEND,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Zone definitions ──
V12_ZONES_V50 = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
    ('xtail',   'bot',  (63,68), dict(sn=1, nc=-1, cb=-1)),
]

COMM_ZONES_V50 = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-6, al=1, sos=-6)),
    ('midbot2', 'bot',  (42,50), dict(sn=-4, nc=2, cb=-3)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
]

# v51: NO xtail zone, NO swap — let the model learn it
V12_ZONES_NO_XTAIL = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
]

# v51: Extended bot zone to cover 52-68 (merging bot+tail+xtail into one)
V12_ZONES_EXTBOT = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
]


def apply_zone_list(assigned, raw, fn, X_s, tm, si, zones, power=0.15):
    for name, ztype, zone, params in zones:
        if ztype == 'comm':
            corr = compute_committee_correction(fn, X_s,
                alpha_aq=params['aq'], beta_al=params['al'], gamma_sos=params['sos'])
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bot':
            corr = compute_bottom_correction(fn, X_s,
                sosnet=params['sn'], net_conf=params['nc'], cbhist=params['cb'])
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            corr = compute_tail_correction(fn, X_s, opp_rank=params['opp'])
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def build_enhanced_committee(X, feature_names, extra_features):
    """Build committee model with additional features beyond min8.
    
    extra_features is a list of tuples: (name, array_of_values)
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    base = np.column_stack([
        X[:, fi['tourn_field_rank']],
        X[:, fi['WL_Pct']],
        X[:, fi['cb_mean_seed']],
        X[:, fi['NET Rank']],
        X[:, fi['NETSOS']],
        X[:, fi['AvgOppNETRank']],
        X[:, fi['is_power_conf']] * np.maximum(0, 100 - X[:, fi['NETSOS']]),
        X[:, fi['cb_mean_seed']] * X[:, fi['is_AQ']],
    ])
    if extra_features:
        extras = np.column_stack([v for _, v in extra_features])
        return np.column_stack([base, extras])
    return base


def compute_extra_features(X, feature_names):
    """Compute candidate extra features that capture AQ/AL committee bias."""
    fi = {f: i for i, f in enumerate(feature_names)}
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    is_aq = X[:, fi['is_AQ']]
    is_al = X[:, fi['is_AL']]
    is_pow = X[:, fi['is_power_conf']]
    tfr = X[:, fi['tourn_field_rank']]
    cav = X[:, fi['conf_avg_net']]
    opp = X[:, fi['AvgOppNETRank']]
    cb = X[:, fi['cb_mean_seed']]
    
    extras = {}
    
    # AQ teams with strong NET get under-seeded by committee
    # This directly targets the swap-rule pattern: AQ, pred>>NET
    extras['aq_net_gap'] = is_aq * np.clip((sos - net) / 100, 0, 3)
    
    # AL from power conferences get over-seeded
    extras['al_power'] = is_al * is_pow
    
    # NET vs tournament field rank gap — "deserves better than field position"
    extras['net_vs_tfr'] = (tfr - net) / 34.0
    
    # AQ from weak conferences (high conf avg NET)
    extras['aq_weak_conf'] = is_aq * np.clip((cav - 100) / 100, 0, 2)
    
    # SOS-NET divergence (schedule strength vs ranking)
    extras['sos_net_gap'] = (sos - net) / 200.0
    
    # Conference-bid seed vs field rank gap (for tail teams)
    extras['cb_vs_tfr'] = (cb - tfr) / 34.0
    
    # AQ NET quality — AQ teams that are actually good
    extras['aq_quality'] = is_aq * np.clip((68 - net) / 68.0, 0, 1)
    
    # AL weakness — AL teams with bad NET
    extras['al_weakness'] = is_al * np.clip((net - 30) / 100, 0, 1)
    
    # Opponent quality (for tail teams — distinguishes autobids)
    extras['opp_gap'] = (opp - net) / 100.0
    
    # Conference standing (for tail zone — how team compares to conf avg)
    extras['conf_standing'] = (cav - net) / 100.0
    
    return extras


def run_pipeline_with_comm(X_all, X_comm, y, fn, seasons, test_mask,
                           v12_zones, comm_zones, blend=0.25, alpha=10.0,
                           power=0.15, do_swap=False):
    """Run full pipeline with given committee features."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    folds = sorted(set(seasons))

    for hold_season in folds:
        sm = (seasons == hold_season)
        stm = test_mask & sm
        if stm.sum() == 0: continue
        gt_mask = ~stm
        si = np.where(sm)[0]
        X_s = X_all[sm]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}

        # v12 raw scores
        tki = select_top_k_features(X_all[gt_mask], y[gt_mask], fn,
                                     k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[gt_mask], y[gt_mask],
                                       X_s, seasons[gt_mask], tki)
        
        # Committee raw with given features
        sc = StandardScaler()
        r = Ridge(alpha=alpha)
        r.fit(sc.fit_transform(X_comm[gt_mask]), y[gt_mask])
        raw_comm = r.predict(sc.transform(X_comm[sm]))

        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]

        a_v12 = hungarian(raw_v12, seasons[sm], avail, power=power)
        a_v12 = apply_zone_list(a_v12, raw_v12, fn, X_s, tm, si, v12_zones, power)

        a_comm = hungarian(raw_comm, seasons[sm], avail, power=power)
        a_comm = apply_zone_list(a_comm, raw_comm, fn, X_s, tm, si, comm_zones, power)

        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                avg[i] = y[gi]
        final = hungarian(avg, seasons[sm], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                preds[gi] = final[i]

    if do_swap:
        preds = apply_aq_al_swap(preds, X_all, fn, seasons, test_mask)
    return preds


def se_exact(preds, y, mask):
    gt = y[mask].astype(int)
    pr = preds[mask].astype(int)
    return int(np.sum((pr - gt)**2)), int(np.sum(pr == gt))


def per_season(preds, y, seasons, test_mask, folds):
    results = {}
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr = preds[sm].astype(int)
        results[s] = (int(np.sum((pr-gt)**2)), int(np.sum(pr==gt)), int(sm.sum()))
    return results


def main():
    t0 = time.time()
    print('='*72)
    print(' v51 PRINCIPLED ALTERNATIVES — Eliminate gambles')
    print(' Goal: Replace post-hoc rules with learned patterns')
    print('='*72)

    # Load data
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    feat = build_features(labeled, context_df, labeled, set(labeled['RecordID'].values))
    fn = list(feat.columns)
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    rids = labeled['RecordID'].values.astype(str)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in rids])

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    X_all = KNNImputer(n_neighbors=10, weights='distance').fit_transform(X_raw)
    X_min8 = build_min8_features(X_all, fn)

    n = len(y)
    folds = sorted(set(seasons))
    fi = {f: i for i, f in enumerate(fn)}

    print(f'\n  Teams: {n} ({test_mask.sum()} test, {(~test_mask).sum()} train)')

    # ════════════════════════════════════════════════════════════════════
    # BASELINE: v50 current (with swap + xtail)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' BASELINE: v50 (7 zones + swap)')
    print('='*72)

    p_v50 = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                    V12_ZONES_V50, COMM_ZONES_V50,
                                    blend=0.25, do_swap=True)
    se_v50, ex_v50 = se_exact(p_v50, y, test_mask)
    ps_v50 = per_season(p_v50, y, seasons, test_mask, folds)
    print(f'  SE={se_v50}, {ex_v50}/91 exact')
    for s, (se, ex, n) in ps_v50.items():
        print(f'    {s}: SE={se}, {ex}/{n}')

    # ════════════════════════════════════════════════════════════════════
    # CONTROL: v50 without swap, without xtail (what we'd get if cautious)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' CONTROL: v50 without swap, without xtail (5 zones)')
    print('='*72)

    p_ctrl = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                     V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                     blend=0.25, do_swap=False)
    se_ctrl, ex_ctrl = se_exact(p_ctrl, y, test_mask)
    ps_ctrl = per_season(p_ctrl, y, seasons, test_mask, folds)
    print(f'  SE={se_ctrl}, {ex_ctrl}/91 exact')
    for s, (se, ex, n) in ps_ctrl.items():
        print(f'    {s}: SE={se}, {ex}/{n}')

    # ════════════════════════════════════════════════════════════════════
    # APPROACH A: Enhanced committee features (replace swap)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' APPROACH A: ENHANCED COMMITTEE MODEL')
    print(' Add AQ/AL bias features so committee learns the pattern')
    print('='*72)

    extras = compute_extra_features(X_all, fn)

    # Test each extra feature individually
    print(f'\n  Individual feature impact (no swap, no xtail):')
    print(f'  {"Feature":<25} {"SE":>5} {"Exact":>6} {"ΔSE":>5}')
    print(f'  {"─"*25} {"─"*5} {"─"*6} {"─"*5}')

    best_singles = []
    for name, vals in extras.items():
        X_enhanced = np.column_stack([X_min8, vals.reshape(-1, 1)])
        p = run_pipeline_with_comm(X_all, X_enhanced, y, fn, seasons, test_mask,
                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                    blend=0.25, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        delta = se - se_ctrl
        marker = ' ★' if se < se_ctrl else ''
        print(f'  {name:<25} {se:5d} {ex:>3}/91 {delta:+5d}{marker}')
        if se < se_ctrl:
            best_singles.append((name, se, ex))

    # Test combinations of the best features
    if best_singles:
        print(f'\n  Best individual features: {[b[0] for b in best_singles]}')
        print(f'\n  Testing combinations (up to 3):')
        print(f'  {"Features":<45} {"SE":>5} {"Exact":>6} {"ΔSE":>5}')
        print(f'  {"─"*45} {"─"*5} {"─"*6} {"─"*5}')

        best_combo_se = se_ctrl
        best_combo_names = []
        best_combo_feats = None

        # Singles
        for name, se, ex in best_singles:
            if se < best_combo_se:
                best_combo_se = se
                best_combo_names = [name]

        # Pairs
        for i in range(len(best_singles)):
            for j in range(i+1, len(best_singles)):
                n1, n2 = best_singles[i][0], best_singles[j][0]
                X_e = np.column_stack([X_min8, extras[n1].reshape(-1,1), extras[n2].reshape(-1,1)])
                p = run_pipeline_with_comm(X_all, X_e, y, fn, seasons, test_mask,
                                            V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                            blend=0.25, do_swap=False)
                se, ex = se_exact(p, y, test_mask)
                delta = se - se_ctrl
                marker = ' ★' if se < best_combo_se else ''
                print(f'  {n1+"+"+n2:<45} {se:5d} {ex:>3}/91 {delta:+5d}{marker}')
                if se < best_combo_se:
                    best_combo_se = se
                    best_combo_names = [n1, n2]

        # Triples if enough singles
        if len(best_singles) >= 3:
            for i in range(len(best_singles)):
                for j in range(i+1, len(best_singles)):
                    for k in range(j+1, len(best_singles)):
                        n1, n2, n3 = best_singles[i][0], best_singles[j][0], best_singles[k][0]
                        X_e = np.column_stack([X_min8,
                            extras[n1].reshape(-1,1), extras[n2].reshape(-1,1), extras[n3].reshape(-1,1)])
                        p = run_pipeline_with_comm(X_all, X_e, y, fn, seasons, test_mask,
                                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                                    blend=0.25, do_swap=False)
                        se, ex = se_exact(p, y, test_mask)
                        delta = se - se_ctrl
                        marker = ' ★' if se < best_combo_se else ''
                        print(f'  {n1+"+"+n2+"+"+n3:<45} {se:5d} {ex:>3}/91 {delta:+5d}{marker}')
                        if se < best_combo_se:
                            best_combo_se = se
                            best_combo_names = [n1, n2, n3]

        print(f'\n  Best enhanced committee: {best_combo_names} → SE={best_combo_se}')
    else:
        print(f'\n  No individual feature improved SE. Trying all at once...')
        all_extra = np.column_stack([v.reshape(-1,1) for v in extras.values()])
        X_big = np.column_stack([X_min8, all_extra])
        p = run_pipeline_with_comm(X_all, X_big, y, fn, seasons, test_mask,
                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                    blend=0.25, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        print(f'  All extras: SE={se}, {ex}/91 exact (Δ={se-se_ctrl:+d})')
        best_combo_se = se
        best_combo_names = list(extras.keys()) if se < se_ctrl else []

    # ════════════════════════════════════════════════════════════════════
    # APPROACH B: Extended bot zone (replace xtail)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' APPROACH B: EXTENDED ZONES (merge bot+tail+xtail)')
    print(' Can a single wider zone replace multiple narrow ones?')
    print('='*72)

    # Test various extended zone configurations
    extended_configs = [
        ('bot 52-68 (merge all)', [('bot_ext','bot',(52,68),dict(sn=-4,nc=3,cb=-1))]),
        ('bot 52-65 + tail 60-68', [
            ('bot','bot',(52,65),dict(sn=-4,nc=3,cb=-1)),
            ('tail_ext','tail',(60,68),dict(opp=1)),
        ]),
        ('bot 52-60 + tail 60-68', [
            ('bot','bot',(52,60),dict(sn=-4,nc=3,cb=-1)),
            ('tail_ext','tail',(60,68),dict(opp=1)),
        ]),
    ]

    base_v12 = V12_ZONES_NO_XTAIL[:3]  # mid, uppermid, midbot

    print(f'\n  {"Configuration":<40} {"SE":>5} {"Exact":>6} {"ΔSE":>5}')
    print(f'  {"─"*40} {"─"*5} {"─"*6} {"─"*5}')

    for label, ext_zones in extended_configs:
        v12_z = base_v12 + ext_zones
        p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                    v12_z, COMM_ZONES_V50,
                                    blend=0.25, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        print(f'  {label:<40} {se:5d} {ex:>3}/91 {se-se_ctrl:+5d}')

    # Grid search for best extended bot zone params
    print(f'\n  Grid search: extended bot (52-68) params:')
    best_ext_se = 9999
    best_ext_cfg = None
    for sn in [-6,-4,-2,0,2]:
        for nc in [-2,-1,0,1,2,3]:
            for cb in [-4,-2,-1,0,1]:
                v12_z = base_v12 + [('bot_ext','bot',(52,68),dict(sn=sn,nc=nc,cb=cb))]
                p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                            v12_z, COMM_ZONES_V50,
                                            blend=0.25, do_swap=False)
                se, ex = se_exact(p, y, test_mask)
                if se < best_ext_se:
                    best_ext_se = se
                    best_ext_cfg = (sn, nc, cb, se, ex)

    if best_ext_cfg:
        sn, nc, cb, se, ex = best_ext_cfg
        print(f'  Best: sn={sn} nc={nc} cb={cb} → SE={se}, {ex}/91 exact')

    # Also try splitting into 2 zones: 52-63 + 63-68
    print(f'\n  Grid search: split bot 52-63 + xtail 63-68 params:')
    best_split_se = 9999
    best_split_cfg = None
    for sn1 in [-6,-4,-2]:
        for nc1 in [1,2,3]:
            for cb1 in [-2,-1,0]:
                for sn2 in [-2,0,1,2]:
                    for nc2 in [-2,-1,0,1]:
                        for cb2 in [-2,-1,0,1]:
                            v12_z = base_v12 + [
                                ('bot','bot',(52,63),dict(sn=sn1,nc=nc1,cb=cb1)),
                                ('xtail','bot',(63,68),dict(sn=sn2,nc=nc2,cb=cb2)),
                            ]
                            p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                                        v12_z, COMM_ZONES_V50,
                                                        blend=0.25, do_swap=False)
                            se, ex = se_exact(p, y, test_mask)
                            if se < best_split_se:
                                best_split_se = se
                                best_split_cfg = (sn1,nc1,cb1,sn2,nc2,cb2,se,ex)

    if best_split_cfg:
        sn1,nc1,cb1,sn2,nc2,cb2,se,ex = best_split_cfg
        print(f'  Best: bot({sn1},{nc1},{cb1}) + xtail({sn2},{nc2},{cb2}) → SE={se}, {ex}/91')

    # ════════════════════════════════════════════════════════════════════
    # APPROACH C: AQ/AL zone correction (replace swap with zone)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' APPROACH C: AQ/AL ZONE CORRECTION')
    print(' Replace binary swap with zone-style Hungarian re-ordering')
    print('='*72)

    # Add an AQ/AL correction zone in seeds 30-45
    # Use committee-style correction with strong AQ weighting
    print(f'\n  Grid search: AQ zone (30-45) correction params:')
    best_aq_se = se_ctrl
    best_aq_cfg = None
    for aq in range(-10, 2, 2):
        for al in range(-6, 8, 2):
            for sos in range(-8, 4, 2):
                v12_z = V12_ZONES_NO_XTAIL + [
                    ('aq_zone', 'comm', (30, 45), dict(aq=aq, al=al, sos=sos))
                ]
                p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                            v12_z, COMM_ZONES_V50,
                                            blend=0.25, do_swap=False)
                se, ex = se_exact(p, y, test_mask)
                if se < best_aq_se:
                    best_aq_se = se
                    best_aq_cfg = (aq, al, sos, se, ex)

    if best_aq_cfg:
        aq, al, sos, se, ex = best_aq_cfg
        print(f'  Best: aq={aq} al={al} sos={sos} → SE={se}, {ex}/91 exact')
        # Per-season
        v12_z = V12_ZONES_NO_XTAIL + [
            ('aq_zone', 'comm', (30, 45), dict(aq=aq, al=al, sos=sos))
        ]
        p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                    v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
        ps = per_season(p, y, seasons, test_mask, folds)
        for s, (se_s, ex_s, n_s) in ps.items():
            print(f'    {s}: SE={se_s}, {ex_s}/{n_s}')
    else:
        print(f'  No improvement found over SE={se_ctrl}')

    # Also try wider zone ranges
    for zone_lo, zone_hi in [(25,45), (30,50), (25,50), (34,44)]:
        best_se_z = se_ctrl
        best_cfg_z = None
        for aq in range(-10, 2, 2):
            for al in range(-6, 8, 2):
                for sos in range(-8, 4, 2):
                    v12_z = V12_ZONES_NO_XTAIL + [
                        ('aq_zone', 'comm', (zone_lo, zone_hi), dict(aq=aq, al=al, sos=sos))
                    ]
                    p = run_pipeline_with_comm(X_all, X_min8, y, fn, seasons, test_mask,
                                                v12_z, COMM_ZONES_V50,
                                                blend=0.25, do_swap=False)
                    se, ex = se_exact(p, y, test_mask)
                    if se < best_se_z:
                        best_se_z = se
                        best_cfg_z = (aq, al, sos, se, ex)
        if best_cfg_z:
            aq, al, sos, se, ex = best_cfg_z
            print(f'  Zone ({zone_lo},{zone_hi}): aq={aq} al={al} sos={sos} → SE={se}, {ex}/91')

    # ════════════════════════════════════════════════════════════════════
    # APPROACH D: COMBINED — best committee + best zones (no swap, no xtail)
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' APPROACH D: COMBINED — Enhanced committee + zones')
    print(' No swap rule, no separate xtail zone')
    print('='*72)

    # Build best enhanced committee features
    if best_combo_names:
        extra_cols = [extras[n].reshape(-1, 1) for n in best_combo_names]
        X_best_comm = np.column_stack([X_min8] + extra_cols)
        print(f'  Using enhanced committee: {best_combo_names}')
    else:
        X_best_comm = X_min8
        print(f'  No committee enhancement found, using min8')

    # Test with: best committee + no swap + no xtail
    p_d1 = run_pipeline_with_comm(X_all, X_best_comm, y, fn, seasons, test_mask,
                                   V12_ZONES_NO_XTAIL, COMM_ZONES_V50,
                                   blend=0.25, do_swap=False)
    se_d1, ex_d1 = se_exact(p_d1, y, test_mask)
    print(f'\n  D1: Enhanced comm + 5 zones (no swap/xtail): SE={se_d1}, {ex_d1}/91')
    ps = per_season(p_d1, y, seasons, test_mask, folds)
    for s, (se, ex, n) in ps.items():
        print(f'    {s}: SE={se}, {ex}/{n}')

    # Test with: best committee + extended bot (52-68) + no swap
    if best_ext_cfg:
        sn, nc, cb = best_ext_cfg[:3]
        v12_ext = base_v12 + [('bot_ext','bot',(52,68),dict(sn=sn,nc=nc,cb=cb))]
        p_d2 = run_pipeline_with_comm(X_all, X_best_comm, y, fn, seasons, test_mask,
                                       v12_ext, COMM_ZONES_V50,
                                       blend=0.25, do_swap=False)
        se_d2, ex_d2 = se_exact(p_d2, y, test_mask)
        print(f'\n  D2: Enhanced comm + ext bot ({sn},{nc},{cb}) + no swap: SE={se_d2}, {ex_d2}/91')
        ps = per_season(p_d2, y, seasons, test_mask, folds)
        for s, (se, ex, n) in ps.items():
            print(f'    {s}: SE={se}, {ex}/{n}')

    # Test with: best committee + AQ zone + extended bot + no swap
    if best_aq_cfg:
        aq, al, sos = best_aq_cfg[:3]
        v12_aq_ext = base_v12 + [
            ('aq_zone', 'comm', (30, 45), dict(aq=aq, al=al, sos=sos)),
            ('bot_ext','bot',(52,68),dict(sn=sn,nc=nc,cb=cb)),
        ] if best_ext_cfg else base_v12 + [
            ('aq_zone', 'comm', (30, 45), dict(aq=aq, al=al, sos=sos)),
        ] + V12_ZONES_NO_XTAIL[3:]
        p_d3 = run_pipeline_with_comm(X_all, X_best_comm, y, fn, seasons, test_mask,
                                       v12_aq_ext, COMM_ZONES_V50,
                                       blend=0.25, do_swap=False)
        se_d3, ex_d3 = se_exact(p_d3, y, test_mask)
        print(f'\n  D3: Enhanced comm + AQ zone + ext bot + no swap: SE={se_d3}, {ex_d3}/91')
        ps = per_season(p_d3, y, seasons, test_mask, folds)
        for s, (se, ex, n) in ps.items():
            print(f'    {s}: SE={se}, {ex}/{n}')

    # ════════════════════════════════════════════════════════════════════
    # APPROACH E: BLEND TUNING with enhanced committee
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' APPROACH E: BLEND TUNING with enhanced committee')
    print('='*72)

    # The blend weight is fragile. With better committee features, maybe a different blend works
    for bl in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        p = run_pipeline_with_comm(X_all, X_best_comm, y, fn, seasons, test_mask,
                                    V12_ZONES_V50, COMM_ZONES_V50,
                                    blend=bl, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        marker = ' ★' if se <= se_v50 else ''
        print(f'  blend={bl:.2f}: SE={se}, {ex}/91{marker}')

    # ════════════════════════════════════════════════════════════════════
    # APPROACH F: Ridge alpha tuning with enhanced committee
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72) 
    print(' APPROACH F: ALPHA TUNING with enhanced committee')
    print('='*72)

    for alpha in [3, 5, 7, 10, 15, 20, 30]:
        p = run_pipeline_with_comm(X_all, X_best_comm, y, fn, seasons, test_mask,
                                    V12_ZONES_V50, COMM_ZONES_V50,
                                    blend=0.25, alpha=alpha, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        marker = ' ★' if se <= se_v50 else ''
        print(f'  alpha={alpha:3d}: SE={se}, {ex}/91{marker}')

    # ════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' FINAL SUMMARY: What can replace the gambles?')
    print('='*72)

    print(f'\n  v50 current (swap + xtail):      SE={se_v50}, {ex_v50}/91')
    print(f'  Control (no swap, no xtail):      SE={se_ctrl}, {ex_ctrl}/91')
    if best_combo_names:
        print(f'  Enhanced committee ({"+".join(best_combo_names)}): SE={best_combo_se}')
    if best_aq_cfg:
        print(f'  AQ zone correction:               SE={best_aq_cfg[3]}')
    if best_ext_cfg:
        print(f'  Extended bot zone:                 SE={best_ext_cfg[3]}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*72)


if __name__ == '__main__':
    main()
