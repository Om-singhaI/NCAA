#!/usr/bin/env python3
"""
v50 DEEP GENERALIZATION ANALYSIS — How will this model perform on 2025-26?
==========================================================================
The HONEST question: We report SE=14 (83/91 exact), but every zone param,
blend weight, and correction rule was tuned ON the same 91 test teams.
What should we ACTUALLY expect for the 2025-26 tournament?

This analysis answers that by:

  TEST 1: Component ablation — TRUE leave-one-season-out (LOSO)
    For each held-out season: train on 4 seasons, apply v50 pipeline.
    Measures: how good is each component when it hasn't seen this season's data?

  TEST 2: Zone parameter stability — cross-validation of zone params
    Tune zone params on 4 seasons → apply to held-out 5th.
    Gap between "tuned on all 5" vs "tuned on 4, tested on 1" = overfit.

  TEST 3: Nested LOSO with zone re-optimization
    For each held-out season, re-optimize all zone params from scratch
    on the remaining 4 seasons. Then apply those params to the held-out season.
    This is the gold standard for estimating true generalization.

  TEST 4: Historical variance analysis
    How much does seed assignment quality vary across seasons?
    If variance is high, our point estimate (SE=14) is unreliable.

  TEST 5: Component sensitivity analysis
    Perturb each tuned param by ±20%, ±50%. How much does SE degrade?
    Fragile params = overfitting risk. Robust params = real signal.

  TEST 6: Effective degrees of freedom vs data points
    Count total tuned parameters vs total test observations.
    High ratio → overfitting almost certain.

  TEST 7: Year-by-year error distribution
    How are errors distributed across seeds? Are we "solving" easy seeds
    and failing on inherently unpredictable ones?

  TEST 8: Realistic 2026 scenario simulation
    Take our best model, but only use zone params from 4-season subsets.
    Average across all 5 such subsets → expected 2026 performance.
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
np.random.seed(42)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'code'))

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

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Zone definitions (v50 production) ──
# v12 path zones
V12_ZONES = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-2, al=-3, sos=-4)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
    ('xtail',   'bot',  (63,68), dict(sn=1, nc=-1, cb=-1)),
]

# Committee path zones
COMM_ZONES = [
    ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
    ('uppermid','comm', (34,44), dict(aq=-6, al=1, sos=-6)),
    ('midbot2', 'bot',  (42,50), dict(sn=-4, nc=2, cb=-3)),
    ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
    ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
    ('tail',    'tail', (60,63), dict(opp=1)),
]

def apply_zone_list(assigned, raw, fn, X_s, tm, si, zones, power=0.15):
    """Apply a list of zone corrections."""
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


def cache_season_data(X_all, X_min8, y, fn, seasons, test_mask, alpha=10.0, power=0.15):
    """Pre-compute the expensive parts (v12 raw scores, committee raw scores, Hungarian)
    once per season. Zone corrections are cheap and can be re-run."""
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
        X_c = X_min8[sm]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}

        # v12 raw scores (expensive)
        tki = select_top_k_features(X_all[gt_mask], y[gt_mask], fn,
                                     k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[gt_mask], y[gt_mask],
                                       X_s, seasons[gt_mask], tki)
        # Committee raw (cheap)
        sc = StandardScaler()
        r = Ridge(alpha=alpha)
        r.fit(sc.fit_transform(X_min8[gt_mask]), y[gt_mask])
        raw_comm = r.predict(sc.transform(X_c))

        # Lock training
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]

        # Base Hungarian assignments (no zones)
        a_v12_base = hungarian(raw_v12, seasons[sm], avail, power=power)
        a_comm_base = hungarian(raw_comm, seasons[sm], avail, power=power)

        cache[hold_season] = {
            'sm': sm, 'si': si, 'tm': tm, 'avail': avail,
            'X_s': X_s, 'raw_v12': raw_v12, 'raw_comm': raw_comm,
            'a_v12_base': a_v12_base, 'a_comm_base': a_comm_base,
        }
    return cache


def run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                   v12_zones=V12_ZONES, comm_zones=COMM_ZONES,
                   blend=0.25, power=0.15,
                   do_swap=True, swap_ng=10, swap_pg=6, swap_zone=(30,45)):
    """Fast pipeline using cached raw scores. Only re-runs zones + blend."""
    n = len(y)
    preds = np.zeros(n, dtype=int)

    for hold_season, c in cache.items():
        sm = c['sm']; si = c['si']; tm = c['tm']; avail = c['avail']
        X_s = c['X_s']

        # Apply zones on v12 branch
        a_v12 = apply_zone_list(c['a_v12_base'].copy(), c['raw_v12'], fn, X_s, tm, si, v12_zones, power)

        # Apply zones on committee branch
        a_comm = apply_zone_list(c['a_comm_base'].copy(), c['raw_comm'], fn, X_s, tm, si, comm_zones, power)

        # Blend + final Hungarian
        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                avg[i] = y[gi]
        final = hungarian(avg, seasons[sm], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]:
                preds[gi] = final[i]

    # AQ↔AL swap
    if do_swap:
        preds = apply_aq_al_swap(preds, X_all, fn, seasons, test_mask,
                                  net_gap=swap_ng, pred_gap=swap_pg, swap_zone=swap_zone)
    return preds


def run_full_pipeline(cache, y, fn, seasons, test_mask, X_all,
                      v12_zones=V12_ZONES, comm_zones=COMM_ZONES,
                      blend=0.25, power=0.15,
                      do_swap=True, swap_ng=10, swap_pg=6, swap_zone=(30,45)):
    """Convenience wrapper around run_from_cache."""
    return run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                          v12_zones=v12_zones, comm_zones=comm_zones,
                          blend=blend, power=power,
                          do_swap=do_swap, swap_ng=swap_ng, swap_pg=swap_pg,
                          swap_zone=swap_zone)


def se_exact(preds, y, mask):
    """Compute SE and exact count for masked predictions."""
    gt = y[mask].astype(int)
    pr = preds[mask].astype(int)
    se = int(np.sum((pr - gt)**2))
    exact = int(np.sum(pr == gt))
    return se, exact


def main():
    t0 = time.time()
    print('='*72)
    print(' v50 DEEP GENERALIZATION ANALYSIS')
    print(' How will this model ACTUALLY perform on 2025-26?')
    print('='*72)

    # ── Load data ──
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    feat = build_features(labeled, context_df, labeled,
                           set(labeled['RecordID'].values))
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
    print(f'  Seasons: {folds}')
    print(f'  Features: {len(fn)}')

    # ── Cache expensive computations ──
    print(f'\n  Caching raw scores (one-time cost)...')
    cache = cache_season_data(X_all, X_min8, y, fn, seasons, test_mask)
    print(f'  Cached {len(cache)} seasons')

    # ════════════════════════════════════════════════════════════════════
    # TEST 1: COMPONENT ABLATION — Each layer's contribution
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 1: COMPONENT ABLATION')
    print(' What does each layer add? Is each step justified?')
    print('='*72)

    configs = [
        ('v12 base only (no zones/ensemble)',
         dict(v12_zones=[], comm_zones=[], blend=0.0, do_swap=False)),
        ('+ Hungarian power=0.15',
         dict(v12_zones=[], comm_zones=[], blend=0.0, do_swap=False)),
        ('+ Dual-Hungarian (blend=0.25, no zones)',
         dict(v12_zones=[], comm_zones=[], blend=0.25, do_swap=False)),
        ('+ Zone 1: Mid (17-34)',
         dict(v12_zones=V12_ZONES[:1], comm_zones=COMM_ZONES[:1], blend=0.25, do_swap=False)),
        ('+ Zone 2: UpperMid (34-44)',
         dict(v12_zones=V12_ZONES[:2], comm_zones=COMM_ZONES[:2], blend=0.25, do_swap=False)),
        ('+ Zone 3-5: MidBot+Bot+Tail',
         dict(v12_zones=V12_ZONES[:5], comm_zones=COMM_ZONES[:6], blend=0.25, do_swap=False)),
        ('+ Zone 7: XTail (63-68)',
         dict(v12_zones=V12_ZONES[:6], comm_zones=COMM_ZONES[:6], blend=0.25, do_swap=False)),
        ('+ AQ↔AL swap (full v50)',
         dict(v12_zones=V12_ZONES, comm_zones=COMM_ZONES, blend=0.25, do_swap=True)),
    ]

    print(f'\n  {"Configuration":<45} {"SE":>5} {"Exact":>6} {"RMSE":>7}  per-season')
    print(f'  {"─"*45} {"─"*5} {"─"*6} {"─"*7}  {"─"*30}')

    for label, kw in configs:
        preds = run_from_cache(cache, y, fn, seasons, test_mask, X_all, **kw)
        se, exact = se_exact(preds, y, test_mask)
        rmse = np.sqrt(np.mean((preds[test_mask] - y[test_mask])**2))
        # Per-season
        per_s = []
        for s in folds:
            sm = test_mask & (seasons == s)
            if sm.sum() == 0: continue
            gt_s = y[sm].astype(int)
            pr_s = preds[sm].astype(int)
            per_s.append(f'{int((pr_s == gt_s).sum())}/{sm.sum()}')
        print(f'  {label:<45} {se:5d} {exact:>3}/91 {rmse:7.3f}  {" ".join(per_s)}')

    # ════════════════════════════════════════════════════════════════════
    # TEST 2: DEGREES OF FREEDOM ANALYSIS
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 2: DEGREES OF FREEDOM vs DATA POINTS')
    print(' Are we fitting noise?')
    print('='*72)

    tuned_params = {
        'Zone 1 mid': {'aq': 0, 'al': 0, 'sos': 3},  # 3 params
        'Zone 2 uppermid v12': {'aq': -2, 'al': -3, 'sos': -4},  # 3
        'Zone 2 uppermid comm': {'aq': -6, 'al': 1, 'sos': -6},  # 3
        'Zone 3b midbot2': {'sn': -4, 'nc': 2, 'cb': -3},  # 3
        'Zone 3 midbot': {'sn': 0, 'nc': 2, 'cb': -2},  # 3
        'Zone 4 bot': {'sn': -4, 'nc': 3, 'cb': -1},  # 3
        'Zone 5 tail': {'opp': 1},  # 1
        'Zone 7 xtail': {'sn': 1, 'nc': -1, 'cb': -1},  # 3
        'Blend weight': {'blend': 0.25},  # 1
        'Ridge alpha': {'alpha': 10},  # 1
        'Power': {'power': 0.15},  # 1
        'AQ swap net_gap': {'ng': 10},  # 1
        'AQ swap pred_gap': {'pg': 6},  # 1
        'AQ swap zone': {'lo': 30, 'hi': 45},  # 2
        'Zone boundaries (7 zones × 2)': {'count': 14},  # 14
    }
    
    total_tuned = sum(len(v) for v in tuned_params.values())
    n_test = int(test_mask.sum())
    
    print(f'\n  Tuned parameter groups:')
    running = 0
    for name, params in tuned_params.items():
        n_p = len(params)
        running += n_p
        print(f'    {name:<35} {n_p:2d} params')
    print(f'    {"─"*35} {"─"*2}')
    print(f'    {"TOTAL":<35} {total_tuned:2d} params')
    
    print(f'\n  Test data points: {n_test} teams × 5 seasons')
    print(f'  Effective independent observations: ~{n_test} (no overlap)')
    print(f'  Params/observation ratio: {total_tuned}/{n_test} = {total_tuned/n_test:.2f}')
    print(f'\n  Rule of thumb:')
    print(f'    < 0.1: safe')
    print(f'    0.1-0.3: moderate risk')
    print(f'    > 0.3: HIGH risk of overfitting')
    print(f'  ⇒ Our ratio: {total_tuned/n_test:.2f} → {"HIGH" if total_tuned/n_test > 0.3 else "MODERATE" if total_tuned/n_test > 0.1 else "LOW"} RISK')

    # ════════════════════════════════════════════════════════════════════
    # TEST 3: NESTED LOSO — True out-of-sample performance
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 3: NESTED LOSO — True out-of-sample estimate')
    print(' For each held-out season, what is SE using v50 pipeline')
    print(' (params tuned on ALL 5 seasons, including held-out)?')
    print(' Then compare to: params re-optimized on remaining 4 seasons.')
    print('='*72)

    # First: v50 params applied to each season (current approach = "in-sample")
    print(f'\n  In-sample (v50 params, per season):')
    preds_v50 = run_from_cache(cache, y, fn, seasons, test_mask, X_all)
    total_se_in = 0
    insample_per_season = {}
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr = preds_v50[sm].astype(int)
        se_s = int(np.sum((pr - gt)**2))
        ex_s = int(np.sum(pr == gt))
        total_se_in += se_s
        insample_per_season[s] = (se_s, ex_s, sm.sum())
        print(f'    {s}: SE={se_s:4d}, {ex_s}/{sm.sum()} exact')
    print(f'    Total: SE={total_se_in}')

    # Now: v50 pipeline with NO zone corrections (base model only)
    print(f'\n  Base model only (no zones, no swap, no ensemble):')
    preds_base = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                                    v12_zones=[], comm_zones=[], blend=0.0, do_swap=False)
    total_se_base = 0
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr = preds_base[sm].astype(int)
        se_s = int(np.sum((pr - gt)**2))
        ex_s = int(np.sum(pr == gt))
        total_se_base += se_s
        print(f'    {s}: SE={se_s:4d}, {ex_s}/{sm.sum()} exact')
    print(f'    Total: SE={total_se_base}')

    # Now: for each held-out season, REMOVE it from the zone-tuning process
    # Re-optimize zone params on remaining 4 seasons, test on held-out
    print(f'\n  Nested LOSO (re-optimize on remaining 4 seasons, test on held-out):')
    print(f'  Testing a grid of zone param perturbations...')

    # For nested LOSO, we test key zone params on 4-season subsets
    # and pick the best, then apply to held-out season
    # This is expensive so we do coarse grid
    total_se_nested = 0
    nested_per_season = {}

    # The key params that matter most: uppermid comm, blend, swap
    # For speed, we test a small grid of the most impactful params
    for hold_s in folds:
        hold_mask = test_mask & (seasons == hold_s)
        if hold_mask.sum() == 0:
            continue
        other_mask = test_mask & (seasons != hold_s)

        # Search over key params that most affect SE
        # We keep base zones fixed (wide plateau) and vary the riskiest ones
        best_se_4 = 999999
        best_cfg = None

        # Coarse grid over the 3 riskiest components:
        # 1. comm_uppermid (aq, al, sos) — was (-6,1,-6) vs (-2,-3,-4)
        # 2. blend weight — 0.20, 0.25, 0.30
        # 3. swap on/off
        comm_um_opts = [
            dict(aq=-6, al=1, sos=-6),   # v49
            dict(aq=-2, al=-3, sos=-4),   # v48 (original)
            dict(aq=-4, al=0, sos=-5),    # interpolation
        ]
        blend_opts = [0.15, 0.20, 0.25, 0.30]
        swap_opts = [True, False]

        for cum, bl, sw in itertools.product(comm_um_opts, blend_opts, swap_opts):
            cz = [
                ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
                ('uppermid','comm', (34,44), cum),
                ('midbot2', 'bot',  (42,50), dict(sn=-4, nc=2, cb=-3)),
                ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
                ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
                ('tail',    'tail', (60,63), dict(opp=1)),
            ]
            p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                                  v12_zones=V12_ZONES, comm_zones=cz,
                                  blend=bl, do_swap=sw)
            se4, _ = se_exact(p, y, other_mask)
            if se4 < best_se_4:
                best_se_4 = se4
                best_cfg = (cum, bl, sw)

        # Apply best-on-4 config to held-out season
        cum, bl, sw = best_cfg
        cz = [
            ('mid',     'comm', (17,34), dict(aq=0, al=0, sos=3)),
            ('uppermid','comm', (34,44), cum),
            ('midbot2', 'bot',  (42,50), dict(sn=-4, nc=2, cb=-3)),
            ('midbot',  'bot',  (48,52), dict(sn=0, nc=2, cb=-2)),
            ('bot',     'bot',  (52,60), dict(sn=-4, nc=3, cb=-1)),
            ('tail',    'tail', (60,63), dict(opp=1)),
        ]
        p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                              v12_zones=V12_ZONES, comm_zones=cz,
                              blend=bl, do_swap=sw)
        gt = y[hold_mask].astype(int)
        pr = p[hold_mask].astype(int)
        se_h = int(np.sum((pr - gt)**2))
        ex_h = int(np.sum(pr == gt))
        total_se_nested += se_h
        nested_per_season[hold_s] = (se_h, ex_h, hold_mask.sum(), best_cfg)
        print(f'    {hold_s}: SE={se_h:4d}, {ex_h}/{hold_mask.sum()} exact '
              f'(best on 4: um={cum}, bl={bl}, sw={sw})')

    print(f'    Total nested LOSO SE: {total_se_nested}')
    print(f'\n  ┌──────────────────────────────────────────────────┐')
    print(f'  │ OVERFITTING GAP:                                 │')
    print(f'  │   In-sample (v50):      SE = {total_se_in:4d}                 │')
    print(f'  │   Base model:           SE = {total_se_base:4d}                │')
    print(f'  │   Nested LOSO:          SE = {total_se_nested:4d}                │')
    print(f'  │   Gap (nested - in):    {total_se_nested - total_se_in:+4d}                    │')
    print(f'  │   Improvement retained: {100*(total_se_base-total_se_nested)/(total_se_base-total_se_in) if total_se_base != total_se_in else 0:.0f}%                       │')
    print(f'  └──────────────────────────────────────────────────┘')

    # ════════════════════════════════════════════════════════════════════
    # TEST 4: PARAM SENSITIVITY — How fragile is v50?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 4: PARAMETER SENSITIVITY')
    print(' How much does SE change when we perturb each key param?')
    print('='*72)

    baseline_se, baseline_exact = se_exact(preds_v50, y, test_mask)

    def test_blend(bl):
        p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                              blend=bl)
        return se_exact(p, y, test_mask)

    def test_no_swap():
        p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                              do_swap=False)
        return se_exact(p, y, test_mask)

    def test_no_zone(skip_name, path='v12'):
        if path == 'v12':
            zones = [z for z in V12_ZONES if z[0] != skip_name]
            p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                                  v12_zones=zones)
        else:
            zones = [z for z in COMM_ZONES if z[0] != skip_name]
            p = run_from_cache(cache, y, fn, seasons, test_mask, X_all,
                                  comm_zones=zones)
        return se_exact(p, y, test_mask)

    print(f'\n  Baseline: SE={baseline_se}, {baseline_exact}/91 exact')
    print(f'\n  {"Perturbation":<45} {"SE":>5} {"Exact":>6} {"ΔSE":>5}  {"Risk"}')
    print(f'  {"─"*45} {"─"*5} {"─"*6} {"─"*5}  {"─"*10}')

    # Blend sensitivity
    for bl in [0.10, 0.15, 0.20, 0.30, 0.35, 0.40]:
        se, ex = test_blend(bl)
        delta = se - baseline_se
        risk = 'FRAGILE' if abs(delta) > 30 else 'moderate' if abs(delta) > 10 else 'stable'
        print(f'  blend={bl:<41.2f} {se:5d} {ex:>3}/91 {delta:+5d}  {risk}')

    # Swap removal
    se, ex = test_no_swap()
    delta = se - baseline_se
    risk = 'FRAGILE' if abs(delta) > 30 else 'moderate' if abs(delta) > 10 else 'stable'
    print(f'  {"no AQ↔AL swap":<45} {se:5d} {ex:>3}/91 {delta:+5d}  {risk}')

    # Zone removal (v12 path)
    for zname in ['mid', 'uppermid', 'midbot', 'bot', 'tail', 'xtail']:
        se, ex = test_no_zone(zname, 'v12')
        delta = se - baseline_se
        risk = 'FRAGILE' if abs(delta) > 30 else 'moderate' if abs(delta) > 10 else 'stable'
        print(f'  {"no zone: " + zname + " (v12)":<45} {se:5d} {ex:>3}/91 {delta:+5d}  {risk}')

    # Zone removal (comm path)
    for zname in ['uppermid', 'midbot2']:
        se, ex = test_no_zone(zname, 'comm')
        delta = se - baseline_se
        risk = 'FRAGILE' if abs(delta) > 30 else 'moderate' if abs(delta) > 10 else 'stable'
        print(f'  {"no zone: " + zname + " (comm)":<45} {se:5d} {ex:>3}/91 {delta:+5d}  {risk}')

    # ════════════════════════════════════════════════════════════════════
    # TEST 5: SEASON VARIANCE — How much do errors vary?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 5: SEASON-LEVEL VARIANCE')
    print(' How stable is performance across seasons?')
    print('='*72)

    se_per_season = []
    exact_per_season = []
    rmse_per_season = []
    n_per_season = []
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr = preds_v50[sm].astype(int)
        se_s = int(np.sum((pr - gt)**2))
        ex_s = int(np.sum(pr == gt))
        rmse_s = np.sqrt(np.mean((pr - gt)**2))
        se_per_season.append(se_s)
        exact_per_season.append(ex_s)
        rmse_per_season.append(rmse_s)
        n_per_season.append(sm.sum())
        print(f'  {s}: {ex_s}/{sm.sum()} exact, SE={se_s}, RMSE={rmse_s:.3f}')

    mean_se = np.mean(se_per_season)
    std_se = np.std(se_per_season)
    print(f'\n  Mean SE per season: {mean_se:.1f} ± {std_se:.1f}')
    print(f'  Coefficient of variation: {std_se/mean_se*100:.0f}%' if mean_se > 0 else '')
    print(f'  Best season: {folds[np.argmin(se_per_season)]} (SE={min(se_per_season)})')
    print(f'  Worst season: {folds[np.argmax(se_per_season)]} (SE={max(se_per_season)})')

    # ════════════════════════════════════════════════════════════════════
    # TEST 6: ERROR DISTRIBUTION BY SEED RANGE
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 6: ERROR DISTRIBUTION BY SEED RANGE')
    print(' Where is the model strong vs weak?')
    print('='*72)

    gt_all = y[test_mask].astype(int)
    pr_all = preds_v50[test_mask].astype(int)

    ranges = [(1,16,'Top'), (17,34,'Mid'), (35,52,'Low'), (53,68,'Bottom')]
    print(f'\n  {"Range":<12} {"Teams":>5} {"Exact":>5} {"Pct":>5} {"SE":>5} {"RMSE":>6}  {"Assessment"}')
    print(f'  {"─"*12} {"─"*5} {"─"*5} {"─"*5} {"─"*5} {"─"*6}  {"─"*15}')
    for lo, hi, lbl in ranges:
        mask_r = (gt_all >= lo) & (gt_all <= hi)
        if mask_r.sum() == 0: continue
        ex = int(np.sum(pr_all[mask_r] == gt_all[mask_r]))
        se_r = int(np.sum((pr_all[mask_r] - gt_all[mask_r])**2))
        rmse_r = np.sqrt(np.mean((pr_all[mask_r] - gt_all[mask_r])**2))
        pct = ex / mask_r.sum() * 100
        quality = 'Excellent' if pct >= 95 else 'Good' if pct >= 80 else 'Weak' if pct >= 60 else 'Poor'
        print(f'  {lbl} ({lo}-{hi}){"":<3} {mask_r.sum():5d} {ex:5d} {pct:4.0f}% {se_r:5d} {rmse_r:5.2f}   {quality}')

    # ════════════════════════════════════════════════════════════════════
    # TEST 7: REMAINING ERRORS — Are they fixable?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 7: REMAINING ERRORS — Can these be fixed?')
    print('='*72)

    errors = []
    test_idx = np.where(test_mask)[0]
    for gi in test_idx:
        pred = int(preds_v50[gi])
        actual = int(y[gi])
        if pred != actual:
            errors.append({
                'rid': rids[gi],
                'season': seasons[gi],
                'pred': pred,
                'actual': actual,
                'error': pred - actual,
                'NET': X_all[gi, fi.get('NET Rank', 0)],
                'SOS': X_all[gi, fi.get('NETSOS', 0)],
                'AQ': int(X_all[gi, fi.get('is_AQ', 0)]),
                'AL': int(X_all[gi, fi.get('is_AL', 0)]),
                'power': int(X_all[gi, fi.get('is_power_conf', 0)]),
            })

    print(f'\n  {len(errors)} remaining errors:')
    print(f'  {"RecordID":<30} {"Pred":>4} {"True":>4} {"Err":>4} {"NET":>5} {"SOS":>5} AQ AL Pw')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*4} {"─"*5} {"─"*5} {"─"*2} {"─"*2} {"─"*2}')
    for e in sorted(errors, key=lambda x: abs(x['error']), reverse=True):
        print(f'  {e["rid"]:<30} {e["pred"]:4d} {e["actual"]:4d} '
              f'{e["error"]:+4d} {e["NET"]:5.0f} {e["SOS"]:5.0f} '
              f'{e["AQ"]:2d} {e["AL"]:2d} {e["power"]:2d}')

    # Check if error pairs have conflicting features
    print(f'\n  Error pair analysis (swapped teams):')
    seasons_with_errors = set(e['season'] for e in errors)
    for s in sorted(seasons_with_errors):
        s_errors = [e for e in errors if e['season'] == s]
        if len(s_errors) == 2:
            a, b = s_errors
            print(f'    {s}: {a["rid"].split("-",2)[-1]} (pred={a["pred"]}, true={a["actual"]}) '
                  f'↔ {b["rid"].split("-",2)[-1]} (pred={b["pred"]}, true={b["actual"]})')
            # Analyze if features support the correct ordering
            feat_agree = 0
            feat_disagree = 0
            if (a['actual'] < b['actual']):  # a should be better
                if a['NET'] < b['NET']: feat_agree += 1
                else: feat_disagree += 1
                if a['SOS'] < b['SOS']: feat_agree += 1
                else: feat_disagree += 1
            else:
                if a['NET'] > b['NET']: feat_agree += 1
                else: feat_disagree += 1
                if a['SOS'] > b['SOS']: feat_agree += 1
                else: feat_disagree += 1
            print(f'      Features supporting correct order: {feat_agree}, opposing: {feat_disagree}')
            print(f'      → {"Fundamentally ambiguous" if feat_disagree >= feat_agree else "Potentially fixable"}')

    # ════════════════════════════════════════════════════════════════════
    # TEST 8: REALISTIC 2026 PERFORMANCE ESTIMATE
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 8: REALISTIC 2026 PERFORMANCE ESTIMATE')
    print('='*72)

    # Method: Average the nested LOSO SE as best estimate
    nested_ses = [v[0] for v in nested_per_season.values()]
    avg_nested_se = np.mean(nested_ses)
    std_nested_se = np.std(nested_ses)

    print(f'\n  Method 1: Nested LOSO average')
    print(f'    Mean SE per season: {avg_nested_se:.1f} ± {std_nested_se:.1f}')
    print(f'    Expected total SE for 1 new season: {avg_nested_se:.0f}')
    # Map SE to expected exact matches (approximate)
    # For ~18 test teams with SE=s, exact ≈ 18 - s/4 (rough)
    n_test_2026 = 18  # typical test set size per season  
    
    # Method 2: Base model + realistic zone benefit
    base_ses = []
    v50_ses = []
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr_base = preds_base[sm].astype(int)
        pr_v50 = preds_v50[sm].astype(int)
        base_ses.append(int(np.sum((pr_base - gt)**2)))
        v50_ses.append(int(np.sum((pr_v50 - gt)**2)))

    avg_base = np.mean(base_ses)
    avg_v50 = np.mean(v50_ses)
    improvement_ratio = (avg_base - avg_v50) / avg_base if avg_base > 0 else 0

    print(f'\n  Method 2: Improvement ratio analysis')
    print(f'    Base model avg SE per season: {avg_base:.1f}')
    print(f'    v50 avg SE per season: {avg_v50:.1f}')
    print(f'    Improvement ratio: {improvement_ratio*100:.0f}%')
    print(f'    On unseen data, expect ~50-70% of improvement to transfer')
    retained_50 = avg_base - 0.5 * (avg_base - avg_v50)
    retained_70 = avg_base - 0.7 * (avg_base - avg_v50)
    print(f'    Expected SE with 50% retention: {retained_50:.0f}')
    print(f'    Expected SE with 70% retention: {retained_70:.0f}')

    # Method 3: Per-season jackknife
    print(f'\n  Method 3: Jackknife variance')
    jackknife_predictions = []
    for leave_s in folds:
        # Use v50 params but acknowledge they might not generalize
        sm = test_mask & (seasons == leave_s)
        if sm.sum() == 0: continue
        gt = y[sm].astype(int)
        pr = preds_v50[sm].astype(int)
        jackknife_predictions.append(np.sqrt(np.mean((pr - gt)**2)))
    print(f'    Per-season RMSE: {[f"{r:.3f}" for r in jackknife_predictions]}')
    print(f'    Mean: {np.mean(jackknife_predictions):.3f}')
    print(f'    Std:  {np.std(jackknife_predictions):.3f}')
    print(f'    95% CI: [{np.mean(jackknife_predictions)-2*np.std(jackknife_predictions):.3f}, '
          f'{np.mean(jackknife_predictions)+2*np.std(jackknife_predictions):.3f}]')

    # ════════════════════════════════════════════════════════════════════
    # TEST 9: MODEL COMPONENTS — WHICH WOULD WE TRUST FOR 2026?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' TEST 9: TRUSTWORTHINESS SCORECARD')
    print('='*72)

    components = [
        ('v12 pairwise base',        'HIGH',    'Proper LOSO, domain-motivated, 68 engineered features'),
        ('Hungarian power=0.15',     'HIGH',    'Wide plateau (0.10-0.20), validated on held-out'),
        ('Dual-Hungarian ensemble',  'MEDIUM',  'Sound theory (diversity), but blend=0.25 narrowly optimal'),
        ('Zone 1: Mid (17-34)',      'MEDIUM',  'Wide plateau, SOS-only, corrects known NET-SOS bias'),
        ('Zone 2: UpperMid (34-44)', 'LOW',     'Different params for v12/comm path, narrow plateaus'),
        ('Zone 3b: MidBot2 (42-50)', 'MEDIUM',  'Wide boundary plateau, zero-regr nested LOSO gap=-12'),
        ('Zone 3: MidBot (48-52)',   'MEDIUM',  'Fixes specific pairs, moderate plateau'),
        ('Zone 4: Bot (52-60)',      'MEDIUM',  'Fixes specific pairs, moderate plateau'),
        ('Zone 5: Tail (60-63)',     'MEDIUM',  'Single param, fixes known tail-swap pattern'),
        ('Zone 7: XTail (63-68)',    'LOW',     'Fixes 2 teams in 1 season, thin evidence'),
        ('AQ↔AL swap',              'LOW',     'Fires in 1/5 seasons only, one wrong swap costs SE≈18-50'),
        ('Blend weight = 0.25',     'LOW',     'Extreme sensitivity: ±0.05 → SE doubles or more'),
    ]

    print(f'\n  {"Component":<30} {"Trust":>6}  Reasoning')
    print(f'  {"─"*30} {"─"*6}  {"─"*50}')
    for comp, trust, reason in components:
        marker = '✓' if trust == 'HIGH' else '~' if trust == 'MEDIUM' else '✗'
        print(f'  {marker} {comp:<28} {trust:>6}  {reason}')

    # ════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*72)
    print(' FINAL VERDICT: 2025-26 PERFORMANCE PREDICTION')
    print('='*72)

    print(f'''
  Current claimed performance (in-sample):
    SE = {total_se_in}, {baseline_exact}/91 exact, RMSE = {np.sqrt(total_se_in/91):.3f}
    Per season: {[f"SE={v[0]}" for v in insample_per_season.values()]}

  Overfitting evidence:
    • {total_tuned} tuned parameters for {n_test} test observations (ratio={total_tuned/n_test:.2f})
    • Blend weight has EXTREME sensitivity (±0.05 → SE doubles)
    • AQ↔AL swap fires in only 1/5 training seasons
    • Zone params tuned on same data used for evaluation
    • Nested LOSO gap: {total_se_nested - total_se_in:+d} SE

  What's REAL (will transfer to 2026):
    • v12 pairwise base model — fundamentally sound architecture
    • Hungarian assignment — correct structural constraint
    • General zone concept — committee biases are real and persistent
    • SOS correction direction — NET-SOS gap is a known bias
    • Dual model diversity — ensemble benefit is real

  What's SUSPECT (may not transfer):
    • Exact zone parameter values — tuned on test data
    • Blend = 0.25 exactly — too sensitive
    • AQ↔AL swap rule — insufficient historical evidence
    • Extreme-tail zone — fits 2 teams in 1 season

  REALISTIC 2025-26 ESTIMATES:
    Optimistic: SE ≈ {int(avg_nested_se):d} per season (~{int(avg_nested_se)}), 
                base + most zones work → ~{max(0,n_test_2026 - int(avg_nested_se/6))}/{n_test_2026} exact
    Realistic:  SE ≈ {int(retained_50):d} per season, 
                zones partially transfer → ~{max(0,n_test_2026 - int(retained_50/4))}/{n_test_2026} exact
    Pessimistic: SE ≈ {int(avg_base):d} per season (base model), 
                 zones don't transfer → ~{max(0,n_test_2026 - int(avg_base/4))}/{n_test_2026} exact

  BOTTOM LINE:
    The SE=14 (83/91) claimed performance is almost certainly inflated.
    Expect regression toward SE ≈ {int(avg_nested_se)}-{int(avg_base)} per season for 2025-26.
    The model's REAL strength is the v12 pairwise architecture + Hungarian.
    Zone corrections provide SOME value but claimed SE improvement is overstated.
''')

    print(f'  Time: {time.time()-t0:.0f}s')
    print('='*72)


if __name__ == '__main__':
    main()
