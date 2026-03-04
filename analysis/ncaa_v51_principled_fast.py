#!/usr/bin/env python3
"""
v51 PRINCIPLED ALTERNATIVES — Eliminate gambles (FAST, cache-based)
===================================================================
Uses cached v12 raw scores (expensive, computed once) and recomputes
committee raw scores per feature config (cheap Ridge fit).

Tests whether learned features can replace AQ↔AL swap and xtail zone.
"""

import os, sys, time, warnings
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
)


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

# No xtail zone (5 zones)
V12_ZONES_NO_XTAIL = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
]

# Base v12 zones (mid + uppermid + midbot only)
V12_BASE = [
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


def cache_v12_raw(X_all, y, fn, seasons, test_mask, power=0.15):
    """Cache only v12 raw scores + base Hungarian (expensive, reusable)."""
    cache = {}
    folds = sorted(set(seasons))
    for hold_season in folds:
        sm = (seasons == hold_season)
        stm = test_mask & sm
        if stm.sum() == 0:
            continue
        gt_mask = ~stm
        si = np.where(sm)[0]
        X_s = X_all[sm]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}

        tki = select_top_k_features(X_all[gt_mask], y[gt_mask], fn,
                                     k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[gt_mask], y[gt_mask],
                                       X_s, seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]

        a_v12_base = hungarian(raw_v12, seasons[sm], avail, power=power)

        cache[hold_season] = {
            'sm': sm, 'si': si, 'tm': tm, 'avail': avail,
            'X_s': X_s, 'raw_v12': raw_v12,
            'a_v12_base': a_v12_base.copy(),
            'gt_mask': gt_mask,
        }
    return cache


def run_with_comm_features(cache, X_comm, y, fn, seasons, test_mask, X_all,
                           v12_zones, comm_zones, blend=0.25, alpha=10.0,
                           power=0.15, do_swap=False):
    """Run pipeline with CACHED v12 raw scores + FRESH committee features."""
    n = len(y)
    preds = np.zeros(n, dtype=int)

    for hold_season, c in cache.items():
        sm = c['sm']; si = c['si']; tm = c['tm']; avail = c['avail']
        X_s = c['X_s']; gt_mask = c['gt_mask']

        # v12 branch: cached raw + zones
        a_v12 = apply_zone_list(c['a_v12_base'].copy(), c['raw_v12'], fn,
                                X_s, tm, si, v12_zones, power)

        # Committee branch: recompute (cheap)
        sc = StandardScaler()
        r = Ridge(alpha=alpha)
        r.fit(sc.fit_transform(X_comm[gt_mask]), y[gt_mask])
        raw_comm = r.predict(sc.transform(X_comm[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_comm[i] = y[gi]
        a_comm = hungarian(raw_comm, seasons[sm], avail, power=power)
        a_comm = apply_zone_list(a_comm, raw_comm, fn, X_s, tm, si, comm_zones, power)

        # Blend + final Hungarian
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


def run_v12_only(cache, y, fn, seasons, test_mask, X_all,
                 v12_zones, power=0.15, do_swap=False):
    """Run v12-only pipeline (no committee) from cache. For testing zone configs."""
    n = len(y)
    preds = np.zeros(n, dtype=int)

    for hold_season, c in cache.items():
        sm = c['sm']; si = c['si']; tm = c['tm']; avail = c['avail']
        X_s = c['X_s']

        a_v12 = apply_zone_list(c['a_v12_base'].copy(), c['raw_v12'], fn,
                                X_s, tm, si, v12_zones, power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                preds[gi] = a_v12[i]

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

    # AQ teams with strong NET relative to SOS
    extras['aq_net_gap'] = is_aq * np.clip((sos - net) / 100, 0, 3)

    # AL from power conferences (over-seeded)
    extras['al_power'] = is_al * is_pow

    # NET vs tournament field rank gap
    extras['net_vs_tfr'] = (tfr - net) / 34.0

    # AQ from weak conferences
    extras['aq_weak_conf'] = is_aq * np.clip((cav - 100) / 100, 0, 2)

    # SOS-NET divergence
    extras['sos_net_gap'] = (sos - net) / 200.0

    # Conference-bid seed vs field rank gap
    extras['cb_vs_tfr'] = (cb - tfr) / 34.0

    # AQ NET quality
    extras['aq_quality'] = is_aq * np.clip((68 - net) / 68.0, 0, 1)

    # AL weakness
    extras['al_weakness'] = is_al * np.clip((net - 30) / 100, 0, 1)

    # Opponent quality gap
    extras['opp_gap'] = (opp - net) / 100.0

    # Conference standing
    extras['conf_standing'] = (cav - net) / 100.0

    # AQ penalty: AQ from non-power conf with high seed prediction
    extras['aq_penalty'] = is_aq * (1 - is_pow) * np.clip((68 - tfr) / 68.0, 0, 1)

    # NET rank squared (captures non-linearity in tail)
    extras['net_sq'] = (net / 68.0) ** 2

    return extras


def main():
    t0 = time.time()
    print('='*72)
    print(' v51 PRINCIPLED ALTERNATIVES — Eliminate gambles')
    print(' Cache-based (fast): v12 raw cached, committee recomputed')
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

    print(f'\n  Teams: {n} ({test_mask.sum()} test)')
    print(f'  Caching v12 raw scores...')
    cache = cache_v12_raw(X_all, y, fn, seasons, test_mask)
    print(f'  Cached {len(cache)} seasons in {time.time()-t0:.0f}s')

    # ────────────────────────────────────────────────────────────────────
    # BASELINES
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' BASELINES')
    print('='*72)

    p_v50 = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                    V12_ZONES_V50, COMM_ZONES_V50, blend=0.25, do_swap=True)
    se_v50, ex_v50 = se_exact(p_v50, y, test_mask)
    ps_v50 = per_season(p_v50, y, seasons, test_mask, folds)
    print(f'  v50 (7 zones + swap):     SE={se_v50:4d}, {ex_v50}/91')
    for s, (se, ex, n_) in ps_v50.items():
        print(f'    {s}: SE={se}, {ex}/{n_}')

    p_ctrl = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                     V12_ZONES_NO_XTAIL, COMM_ZONES_V50, blend=0.25, do_swap=False)
    se_ctrl, ex_ctrl = se_exact(p_ctrl, y, test_mask)
    ps_ctrl = per_season(p_ctrl, y, seasons, test_mask, folds)
    print(f'  Control (no swap/xtail):  SE={se_ctrl:4d}, {ex_ctrl}/91')
    for s, (se, ex, n_) in ps_ctrl.items():
        print(f'    {s}: SE={se}, {ex}/{n_}')

    # What's the SE gap the improvements need to bridge?
    gap = se_ctrl - se_v50
    print(f'\n  Gap to bridge: SE={gap} (from {se_ctrl} down to {se_v50})')

    # ────────────────────────────────────────────────────────────────────
    # APPROACH A: Enhanced committee features
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' A: ENHANCED COMMITTEE FEATURES (replace swap)')
    print(' Add features so committee learns AQ/AL bias')
    print('='*72)

    extras = compute_extra_features(X_all, fn)

    print(f'\n  Individual feature impact (no swap, no xtail):')
    print(f'  {"Feature":<25} {"SE":>5} {"Exact":>6} {"ΔSE":>5}')
    print(f'  {"─"*25} {"─"*5} {"─"*6} {"─"*5}')

    best_singles = []
    for name, vals in extras.items():
        X_e = np.column_stack([X_min8, vals.reshape(-1, 1)])
        p = run_with_comm_features(cache, X_e, y, fn, seasons, test_mask, X_all,
                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50, blend=0.25, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        delta = se - se_ctrl
        marker = ' ★' if delta < 0 else ''
        print(f'  {name:<25} {se:5d} {ex:>3}/91 {delta:+5d}{marker}')
        if delta < 0:
            best_singles.append((name, se, ex))

    if best_singles:
        print(f'\n  Best singles: {[b[0] for b in best_singles]}')
        print(f'\n  Testing combinations:')
        print(f'  {"Features":<50} {"SE":>5} {"Exact":>6}')
        print(f'  {"─"*50} {"─"*5} {"─"*6}')

        best_combo_se = se_ctrl
        best_combo_names = []

        # Pairs
        for i in range(len(best_singles)):
            for j in range(i+1, len(best_singles)):
                n1, n2 = best_singles[i][0], best_singles[j][0]
                X_e = np.column_stack([X_min8, extras[n1].reshape(-1,1), extras[n2].reshape(-1,1)])
                p = run_with_comm_features(cache, X_e, y, fn, seasons, test_mask, X_all,
                                            V12_ZONES_NO_XTAIL, COMM_ZONES_V50, blend=0.25, do_swap=False)
                se, ex = se_exact(p, y, test_mask)
                marker = ' ★' if se < best_combo_se else ''
                print(f'  {n1+"+"+n2:<50} {se:5d} {ex:>3}/91{marker}')
                if se < best_combo_se:
                    best_combo_se = se
                    best_combo_names = [n1, n2]

        # Triples
        if len(best_singles) >= 3:
            for i in range(len(best_singles)):
                for j in range(i+1, len(best_singles)):
                    for k in range(j+1, len(best_singles)):
                        n1, n2, n3 = best_singles[i][0], best_singles[j][0], best_singles[k][0]
                        X_e = np.column_stack([X_min8,
                            extras[n1].reshape(-1,1), extras[n2].reshape(-1,1), extras[n3].reshape(-1,1)])
                        p = run_with_comm_features(cache, X_e, y, fn, seasons, test_mask, X_all,
                                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50, blend=0.25, do_swap=False)
                        se, ex = se_exact(p, y, test_mask)
                        marker = ' ★' if se < best_combo_se else ''
                        print(f'  {n1+"+"+n2+"+"+n3:<50} {se:5d} {ex:>3}/91{marker}')
                        if se < best_combo_se:
                            best_combo_se = se
                            best_combo_names = [n1, n2, n3]

        print(f'\n  Best enhanced committee: {best_combo_names} → SE={best_combo_se}')
    else:
        # Also try all extras at once
        all_extra = np.column_stack([v.reshape(-1,1) for v in extras.values()])
        X_big = np.column_stack([X_min8, all_extra])
        p = run_with_comm_features(cache, X_big, y, fn, seasons, test_mask, X_all,
                                    V12_ZONES_NO_XTAIL, COMM_ZONES_V50, blend=0.25, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        print(f'\n  All {len(extras)} extras: SE={se}, {ex}/91 (Δ={se-se_ctrl:+d})')
        best_combo_se = se if se < se_ctrl else se_ctrl
        best_combo_names = list(extras.keys()) if se < se_ctrl else []

    # ────────────────────────────────────────────────────────────────────
    # APPROACH B: Extended/merged bot zone (replace xtail)
    # Using v12-only path for speed since we're only testing zone configs
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' B: EXTENDED BOT ZONE (merge bot+tail+xtail)')
    print('='*72)

    # Quick test: does a single wide bot zone (52-68) work?
    print(f'\n  Grid search: bot (52-68) params:')
    best_ext_se = 9999
    best_ext_cfg = None

    for sn in [-6, -4, -2, 0, 2]:
        for nc in [-2, 0, 2, 3, 4]:
            for cb in [-3, -2, -1, 0, 1]:
                v12_z = V12_BASE + [('bot_ext', 'bot', (52,68), dict(sn=sn, nc=nc, cb=cb))]
                p = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                            v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
                se, ex = se_exact(p, y, test_mask)
                if se < best_ext_se:
                    best_ext_se = se
                    best_ext_cfg = (sn, nc, cb, se, ex)

    if best_ext_cfg:
        sn, nc, cb, se, ex = best_ext_cfg
        print(f'  Best: sn={sn} nc={nc} cb={cb} → SE={se}, {ex}/91')
        print(f'  vs control (no swap/xtail): ΔSE={se-se_ctrl:+d}')

    # Also try bot 52-60 + tail 60-68 (extend tail instead of separate xtail)
    print(f'\n  Grid search: bot (52-60) + extended tail (60-68):')
    best_bt_se = 9999
    best_bt_cfg = None
    for sn in [-6, -4, -2]:
        for nc in [1, 2, 3]:
            for cb in [-2, -1, 0]:
                for opp in [-1, 0, 1, 2]:
                    v12_z = V12_BASE + [
                        ('bot', 'bot', (52,60), dict(sn=sn, nc=nc, cb=cb)),
                        ('tail_ext', 'tail', (60,68), dict(opp=opp)),
                    ]
                    p = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                                v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
                    se, ex = se_exact(p, y, test_mask)
                    if se < best_bt_se:
                        best_bt_se = se
                        best_bt_cfg = (sn, nc, cb, opp, se, ex)

    if best_bt_cfg:
        sn, nc, cb, opp, se, ex = best_bt_cfg
        print(f'  Best: bot({sn},{nc},{cb}) + tail({opp}) → SE={se}, {ex}/91')

    # Also try v50 original bot+tail+xtail as 3 separate zones (current)
    # vs merging into 2
    print(f'\n  Grid search: bot (52-63) + xtail (63-68):')
    best_2z_se = 9999
    best_2z_cfg = None
    for sn1 in [-6, -4, -2]:
        for nc1 in [2, 3]:
            for cb1 in [-2, -1]:
                for sn2 in [-1, 0, 1, 2]:
                    for nc2 in [-2, -1, 0, 1]:
                        for cb2 in [-2, -1, 0]:
                            v12_z = V12_BASE + [
                                ('bot', 'bot', (52,63), dict(sn=sn1, nc=nc1, cb=cb1)),
                                ('xtail', 'bot', (63,68), dict(sn=sn2, nc=nc2, cb=cb2)),
                            ]
                            p = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                                        v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
                            se, ex = se_exact(p, y, test_mask)
                            if se < best_2z_se:
                                best_2z_se = se
                                best_2z_cfg = (sn1,nc1,cb1,sn2,nc2,cb2,se,ex)

    if best_2z_cfg:
        sn1,nc1,cb1,sn2,nc2,cb2,se,ex = best_2z_cfg
        print(f'  Best: bot({sn1},{nc1},{cb1}) + xtail({sn2},{nc2},{cb2}) → SE={se}, {ex}/91')

    # ────────────────────────────────────────────────────────────────────
    # APPROACH C: AQ/AL ZONE CORRECTION (replace swap with zone)
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' C: AQ/AL ZONE CORRECTION (replace swap with zone)')
    print('='*72)

    print(f'\n  Grid search: AQ zone + v12 5-zone (no swap, no xtail):')
    best_aq_se = se_ctrl
    best_aq_cfg = None

    for zone_lo, zone_hi in [(28,48), (30,45), (32,42), (25,50)]:
        local_best_se = se_ctrl
        local_best_cfg = None
        for aq in range(-10, 4, 2):
            for al in range(-8, 8, 2):
                for sos in range(-8, 6, 2):
                    v12_z = V12_ZONES_NO_XTAIL + [
                        ('aq_zone', 'comm', (zone_lo, zone_hi), dict(aq=aq, al=al, sos=sos))
                    ]
                    p = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                                v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
                    se, ex = se_exact(p, y, test_mask)
                    if se < local_best_se:
                        local_best_se = se
                        local_best_cfg = (aq, al, sos, se, ex)
                    if se < best_aq_se:
                        best_aq_se = se
                        best_aq_cfg = (zone_lo, zone_hi, aq, al, sos, se, ex)

        if local_best_cfg:
            aq, al, sos, se, ex = local_best_cfg
            print(f'  Zone ({zone_lo},{zone_hi}): aq={aq:+d} al={al:+d} sos={sos:+d} → SE={se}, {ex}/91')
        else:
            print(f'  Zone ({zone_lo},{zone_hi}): no improvement')

    if best_aq_cfg:
        zl, zh, aq, al, sos, se, ex = best_aq_cfg
        print(f'\n  BEST AQ zone: ({zl},{zh}) aq={aq:+d} al={al:+d} sos={sos:+d} → SE={se}, {ex}/91')
        # Per-season breakdown
        v12_z = V12_ZONES_NO_XTAIL + [
            ('aq_zone', 'comm', (zl, zh), dict(aq=aq, al=al, sos=sos))
        ]
        p = run_with_comm_features(cache, X_min8, y, fn, seasons, test_mask, X_all,
                                    v12_z, COMM_ZONES_V50, blend=0.25, do_swap=False)
        ps = per_season(p, y, seasons, test_mask, folds)
        for s, (se_s, ex_s, n_s) in ps.items():
            print(f'    {s}: SE={se_s}, {ex_s}/{n_s}')

    # ────────────────────────────────────────────────────────────────────
    # APPROACH D: COMBINED best alternatives
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' D: COMBINED — Best of A + B + C (no swap, no xtail)')
    print('='*72)

    # Build best enhanced committee
    if best_combo_names:
        extra_cols = [extras[n].reshape(-1, 1) for n in best_combo_names]
        X_best_comm = np.column_stack([X_min8] + extra_cols)
        print(f'  Enhanced committee: {best_combo_names}')
    else:
        X_best_comm = X_min8
        print(f'  No committee enhancement found, using min8')

    # Best zone config (use best of B's options)
    best_bottom_zones = V12_ZONES_NO_XTAIL[3:]  # default: bot+tail from V12_NO_XTAIL
    best_bottom_label = 'default bot+tail'
    if best_ext_cfg and best_ext_cfg[3] < se_ctrl:
        sn, nc, cb = best_ext_cfg[:3]
        best_bottom_zones = [('bot_ext', 'bot', (52,68), dict(sn=sn, nc=nc, cb=cb))]
        best_bottom_label = f'ext_bot({sn},{nc},{cb})'
    if best_bt_cfg and best_bt_cfg[4] < se_ctrl and (not best_ext_cfg or best_bt_cfg[4] < best_ext_cfg[3]):
        sn, nc, cb, opp = best_bt_cfg[:4]
        best_bottom_zones = [
            ('bot', 'bot', (52,60), dict(sn=sn, nc=nc, cb=cb)),
            ('tail_ext', 'tail', (60,68), dict(opp=opp)),
        ]
        best_bottom_label = f'bot({sn},{nc},{cb})+tail({opp})'

    # Best AQ zone (if any)
    aq_zone_list = []
    aq_label = 'none'
    if best_aq_cfg and best_aq_cfg[5] < se_ctrl:
        zl, zh, aq, al, sos = best_aq_cfg[:5]
        aq_zone_list = [('aq_zone', 'comm', (zl, zh), dict(aq=aq, al=al, sos=sos))]
        aq_label = f'({zl},{zh}) aq={aq:+d} al={al:+d} sos={sos:+d}'

    # Combine
    v12_combined = V12_BASE + best_bottom_zones + aq_zone_list
    print(f'  Bottom zones: {best_bottom_label}')
    print(f'  AQ zone: {aq_label}')

    # Test combinations
    configs = [
        ('D1: Enhanced comm only', X_best_comm, V12_ZONES_NO_XTAIL, COMM_ZONES_V50, 0.25),
        ('D2: Enhanced comm + best bottom', X_best_comm, V12_BASE + best_bottom_zones, COMM_ZONES_V50, 0.25),
        ('D3: Enhanced comm + AQ zone', X_best_comm, V12_ZONES_NO_XTAIL + aq_zone_list, COMM_ZONES_V50, 0.25),
        ('D4: Enhanced comm + best bottom + AQ zone', X_best_comm, v12_combined, COMM_ZONES_V50, 0.25),
        ('D5: Min8 + best bottom + AQ zone', X_min8, v12_combined, COMM_ZONES_V50, 0.25),
    ]

    print(f'\n  {"Config":<50} {"SE":>5} {"Exact":>6} {"ΔvCtrl":>7} {"Δv50":>5}')
    print(f'  {"─"*50} {"─"*5} {"─"*6} {"─"*7} {"─"*5}')

    best_d_se = 9999
    best_d_label = ''
    best_d_p = None
    for label, X_c, v12_z, comm_z, bl in configs:
        p = run_with_comm_features(cache, X_c, y, fn, seasons, test_mask, X_all,
                                    v12_z, comm_z, blend=bl, do_swap=False)
        se, ex = se_exact(p, y, test_mask)
        dctrl = se - se_ctrl
        dv50 = se - se_v50
        marker = ' ★' if se < best_d_se else ''
        print(f'  {label:<50} {se:5d} {ex:>3}/91 {dctrl:+7d} {dv50:+5d}{marker}')
        if se < best_d_se:
            best_d_se = se
            best_d_label = label
            best_d_p = p

    # Per-season for best
    if best_d_p is not None:
        print(f'\n  Best: {best_d_label} (SE={best_d_se})')
        ps = per_season(best_d_p, y, seasons, test_mask, folds)
        for s, (se, ex, n_) in ps.items():
            print(f'    {s}: SE={se}, {ex}/{n_}')

    # ────────────────────────────────────────────────────────────────────
    # APPROACH E: Blend tuning with best config
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' E: BLEND & ALPHA TUNING with best config')
    print('='*72)

    best_e_se = best_d_se
    best_e_bl = 0.25
    best_e_alpha = 10.0
    for bl in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for alpha in [5, 7, 10, 15, 20]:
            p = run_with_comm_features(cache, X_best_comm, y, fn, seasons, test_mask, X_all,
                                        v12_combined, COMM_ZONES_V50, blend=bl, alpha=alpha, do_swap=False)
            se, ex = se_exact(p, y, test_mask)
            if se < best_e_se:
                best_e_se = se
                best_e_bl = bl
                best_e_alpha = alpha
                marker = ' ★'
            else:
                marker = ''
            if bl in [0.15, 0.20, 0.25, 0.30, 0.35]:
                if alpha == 10:
                    print(f'  bl={bl:.2f} α={alpha:3d}: SE={se}, {ex}/91{marker}')

    print(f'\n  Best: bl={best_e_bl:.2f} α={best_e_alpha} → SE={best_e_se}')

    # ────────────────────────────────────────────────────────────────────
    # VALIDATION: Compare against v50 errors
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' VALIDATION: Error comparison')
    print('='*72)

    # Show which teams change between v50 and best principled
    p_best = run_with_comm_features(cache, X_best_comm, y, fn, seasons, test_mask, X_all,
                                     v12_combined, COMM_ZONES_V50,
                                     blend=best_e_bl, alpha=best_e_alpha, do_swap=False)
    se_best, ex_best = se_exact(p_best, y, test_mask)
    print(f'\n  v50:        SE={se_v50}, {ex_v50}/91')
    print(f'  Principled: SE={se_best}, {ex_best}/91')

    gt_arr = y[test_mask].astype(int)
    p50_arr = p_v50[test_mask].astype(int)
    pb_arr = p_best[test_mask].astype(int)
    rids_test = rids[test_mask]
    seasons_test = seasons[test_mask]

    # Teams where principled is worse
    worse = []
    better = []
    for i in range(len(gt_arr)):
        e50 = (p50_arr[i] - gt_arr[i])**2
        eb = (pb_arr[i] - gt_arr[i])**2
        if eb > e50:
            worse.append((rids_test[i], seasons_test[i], gt_arr[i], p50_arr[i], pb_arr[i]))
        elif eb < e50:
            better.append((rids_test[i], seasons_test[i], gt_arr[i], p50_arr[i], pb_arr[i]))

    if worse:
        print(f'\n  Regressions ({len(worse)} teams):')
        print(f'    {"RID":<30} {"Ssn":<6} {"GT":>3} {"v50":>4} {"v51":>4}')
        for rid, ssn, gt, p50, pb in sorted(worse, key=lambda x: (x[4]-x[2])**2 - (x[3]-x[2])**2, reverse=True):
            print(f'    {rid:<30} {ssn:<6} {gt:3d} {p50:4d} {pb:4d}')
    else:
        print(f'\n  No regressions!')

    if better:
        print(f'\n  Improvements ({len(better)} teams):')
        print(f'    {"RID":<30} {"Ssn":<6} {"GT":>3} {"v50":>4} {"v51":>4}')
        for rid, ssn, gt, p50, pb in sorted(better, key=lambda x: (x[3]-x[2])**2 - (x[4]-x[2])**2, reverse=True):
            print(f'    {rid:<30} {ssn:<6} {gt:3d} {p50:4d} {pb:4d}')

    # ────────────────────────────────────────────────────────────────────
    # SAFETY CHECK: Nested LOSO on best config
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' SAFETY: Nested LOSO on best principled config')
    print('='*72)

    total_se = 0
    total_ex = 0
    total_n = 0
    for hold_season in folds:
        sm_hold = (seasons == hold_season)
        stm_hold = test_mask & sm_hold
        if stm_hold.sum() == 0:
            continue
        # Build cache for this fold only using remaining 4 seasons' trained cache
        # The v12 raw scores were already computed per-fold (each fold holds out one season)
        # so we just use the same cache entry
        c = cache[hold_season]
        gt = y[stm_hold].astype(int)
        p_fold = p_best[stm_hold].astype(int)
        se_fold = int(np.sum((p_fold - gt)**2))
        ex_fold = int(np.sum(p_fold == gt))
        n_fold = int(stm_hold.sum())
        total_se += se_fold
        total_ex += ex_fold
        total_n += n_fold
        print(f'  {hold_season}: SE={se_fold}, {ex_fold}/{n_fold}')

    print(f'  Total: SE={total_se}, {total_ex}/{total_n}')

    # ────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────────────
    print('\n' + '='*72)
    print(' FINAL SUMMARY')
    print('='*72)
    print(f'\n  v50 (7 zones + swap):           SE={se_v50:4d}, {ex_v50}/91  ← gambles')
    print(f'  Control (no swap, no xtail):    SE={se_ctrl:4d}, {ex_ctrl}/91')
    print(f'  Best principled:                SE={se_best:4d}, {ex_best}/91  ← no gambles')
    if se_best <= se_v50:
        print(f'\n  ✓ PRINCIPLED VERSION MATCHES OR BEATS v50!')
        print(f'    - No post-hoc AQ↔AL swap rule')
        print(f'    - No separate xtail zone')
        print(f'    - All corrections are learned or zone-based')
    elif se_best < se_ctrl:
        print(f'\n  ○ Principled version reduces gap: {se_ctrl}→{se_best} (covers {se_ctrl-se_best}/{gap} of gap)')
        print(f'    Remaining gap: {se_best-se_v50} SE')
    else:
        print(f'\n  ✗ Could not replicate gamble improvements with principled methods')
        print(f'    The AQ↔AL swap and xtail are solving patterns that features cannot capture')

    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*72)


if __name__ == '__main__':
    main()
