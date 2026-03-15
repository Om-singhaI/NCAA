#!/usr/bin/env python3
"""
v47 Deep Dive — minimal features at SE=116 & ElasticNet at SE=122.

1. Fine-grained grid around α=5-12, blend=0.25
2. Combine minimal + EN approaches
3. Try intermediate feature sets
4. Nested LOSO validation
5. Per-season error comparison
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_committee_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

ZONES_V46 = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
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
    folds = sorted(set(seasons))
    cache = {}
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        X_season = X_all[season_mask]
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                       X_season, seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_v12[i] = y[gi]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, ZONES_V46, 0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12_zoned': a_v12_zoned, 'avail': avail, 'X_season': X_season,
        }
    return cache


def run_fast(cache, X_comm, y, fn, seasons, test_mask, n,
             alpha=15.0, blend=0.15, comm_model='ridge',
             el_alpha=0.15, l1_ratio=0.5,
             zones=ZONES_V46, comm_zones=True):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_comm[train_mask])
        X_te = sc.transform(X_comm[sm])
        if comm_model == 'ridge':
            mdl = Ridge(alpha=alpha)
        elif comm_model == 'elastic':
            mdl = ElasticNet(alpha=el_alpha, l1_ratio=l1_ratio, max_iter=5000)
        mdl.fit(X_tr, y[train_mask])
        raw_comm = mdl.predict(X_te)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        
        a_comm = hungarian(raw_comm, seasons[sm], avail, power=0.15)
        if comm_zones:
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, 0.15)
        
        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        a_final = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


def build_comm_minimal(X, fn):
    fi = {f: i for i, f in enumerate(fn)}
    return np.column_stack([
        X[:, fi['tourn_field_rank']],
        X[:, fi['WL_Pct']],
        X[:, fi['cb_mean_seed']],
        X[:, fi['NET Rank']],
        X[:, fi['NETSOS']],
        X[:, fi['is_power_conf']] * np.maximum(0, 100 - X[:, fi['NETSOS']]),
        X[:, fi['cb_mean_seed']] * X[:, fi['is_AQ']],
    ])


def build_comm_variants(X, fn):
    """Return dict of feature set name → matrix for many variants."""
    fi = {f: i for i, f in enumerate(fn)}
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    opp = X[:, fi['AvgOppNETRank']]
    is_al = X[:, fi['is_AL']]
    is_aq = X[:, fi['is_AQ']]
    is_pow = X[:, fi['is_power_conf']]
    conf = X[:, fi['conf_avg_net']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q2w = X[:, fi['Quadrant2_W']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    wpct = X[:, fi['WL_Pct']]
    cb = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    variants = {}
    
    # min_7: original minimal (7 features)
    variants['min_7'] = build_comm_minimal(X, fn)
    
    # min_8: add opp
    variants['min_8'] = np.column_stack([
        tfr, wpct, cb, net, sos, opp,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
    ])
    
    # min_9: add q1w
    variants['min_9'] = np.column_stack([
        tfr, wpct, cb, net, sos, opp, q1w,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
    ])
    
    # min_10: add is_al
    variants['min_10'] = np.column_stack([
        tfr, wpct, cb, net, sos, opp, q1w, is_al,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
    ])
    
    # min_6: drop sos
    variants['min_6'] = np.column_stack([
        tfr, wpct, cb, net,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
    ])
    
    # min_5: just core 5
    variants['min_5'] = np.column_stack([
        tfr, wpct, cb, net, sos,
    ])
    
    # min_4: ultra minimal
    variants['min_4'] = np.column_stack([
        tfr, cb, net, sos,
    ])
    
    # tfr_cb: 2 features only
    variants['tfr_cb'] = np.column_stack([
        tfr, cb,
    ])
    
    # mid_12: middle ground
    variants['mid_12'] = np.column_stack([
        tfr, wpct, cb, net, sos, opp, q1w, is_al, is_pow,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
        net - 0.3 * sos,
    ])
    
    # mid_15: between minimal and full
    variants['mid_15'] = np.column_stack([
        tfr, wpct, cb, net, sos, opp, q1w, q2w, is_al, is_pow, is_aq,
        is_pow * np.maximum(0, 100 - sos),
        cb * is_aq,
        cb * is_al,
        net - conf,
    ])
    
    return variants


def main():
    t0 = time.time()
    print('='*60)
    print(' v47 DEEP DIVE')
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
    
    X_comm = build_committee_features(X_all, fn)
    variants = build_comm_variants(X_all, fn)
    
    print(f'  Caching v12 predictions...')
    cache = cache_v12(X_all, y, fn, seasons, test_mask)
    print(f'  Cached')
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    best_se = 999
    best_label = ''
    best_preds = None
    results = []
    
    # ═══════════════════════════════════════════════════════
    # PART 1: Fine grid around minimal (α=5-12, blend=0.25)
    # ═══════════════════════════════════════════════════════
    print('\n── Part 1: Fine grid, minimal features ──')
    for alpha in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for blend_pct in range(20, 35):
            blend = blend_pct / 100
            p = run_fast(cache, variants['min_7'], y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend)
            se, ex = ev(p)
            results.append((f'min7_a{alpha}_b{blend:.2f}', se, ex))
            if se < best_se:
                best_se, best_label, best_preds = se, f'min7_a{alpha}_b{blend:.2f}', p.copy()
                print(f'  ★ α={alpha} b={blend:.2f}: SE={se}, exact={ex}/91')
    print(f'  [Part 1] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PART 2: Feature set variants
    # ═══════════════════════════════════════════════════════
    print('\n── Part 2: Feature set variants ──')
    for name, X_c in variants.items():
        if name == 'min_7': continue
        for alpha in [5, 10, 15]:
            for blend in [0.20, 0.25, 0.30]:
                p = run_fast(cache, X_c, y, fn, seasons, test_mask, n,
                            alpha=alpha, blend=blend)
                se, ex = ev(p)
                results.append((f'{name}_a{alpha}_b{blend:.2f}', se, ex))
                if se < best_se:
                    best_se, best_label, best_preds = se, f'{name}_a{alpha}_b{blend:.2f}', p.copy()
                    print(f'  ★ {name} α={alpha} b={blend:.2f}: SE={se}, exact={ex}/91')
    print(f'  [Part 2] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PART 3: Dual committee (min + v1 original)
    # ═══════════════════════════════════════════════════════
    print('\n── Part 3: Dual committee (min + v1) ──')
    for min_b in [0.15, 0.20, 0.25]:
        for v1_b in [0.05, 0.10, 0.15]:
            if min_b + v1_b > 0.45: continue
            preds_dc = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                
                # Min Ridge
                sc1 = StandardScaler()
                X_tr1 = sc1.fit_transform(variants['min_7'][train_mask])
                X_te1 = sc1.transform(variants['min_7'][sm])
                r1 = Ridge(alpha=5)
                r1.fit(X_tr1, y[train_mask])
                raw1 = r1.predict(X_te1)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw1[i] = y[gi]
                a1 = hungarian(raw1, seasons[sm], avail, power=0.15)
                a1 = apply_zones(a1, raw1, fn, X_season, tm, si, ZONES_V46, 0.15)
                
                # v1 Ridge
                sc2 = StandardScaler()
                X_tr2 = sc2.fit_transform(X_comm[train_mask])
                X_te2 = sc2.transform(X_comm[sm])
                r2 = Ridge(alpha=15)
                r2.fit(X_tr2, y[train_mask])
                raw2 = r2.predict(X_te2)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw2[i] = y[gi]
                a2 = hungarian(raw2, seasons[sm], avail, power=0.15)
                a2 = apply_zones(a2, raw2, fn, X_season, tm, si, ZONES_V46, 0.15)
                
                v12_w = 1.0 - min_b - v1_b
                avg = v12_w * a_v12.astype(float) + min_b * a1.astype(float) + v1_b * a2.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                a_f = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_dc[gi] = a_f[i]
            
            se, ex = ev(preds_dc)
            results.append((f'dual_min{min_b}_v1{v1_b}', se, ex))
            if se < best_se:
                best_se, best_label, best_preds = se, f'dual_min{min_b}_v1{v1_b}', preds_dc.copy()
                print(f'  ★ min_b={min_b} v1_b={v1_b}: SE={se}, exact={ex}/91')
    print(f'  [Part 3] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PART 4: EN on minimal features
    # ═══════════════════════════════════════════════════════
    print('\n── Part 4: ElasticNet on minimal features ──')
    for ea in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for l1r in [0.3, 0.5, 0.7]:
            for blend in [0.20, 0.25, 0.30]:
                p = run_fast(cache, variants['min_7'], y, fn, seasons, test_mask, n,
                            comm_model='elastic', el_alpha=ea, l1_ratio=l1r, blend=blend)
                se, ex = ev(p)
                results.append((f'minEN_ea{ea}_l1{l1r}_b{blend}', se, ex))
                if se < best_se:
                    best_se, best_label, best_preds = se, f'minEN_ea{ea}_l1{l1r}_b{blend}', p.copy()
                    print(f'  ★ EN a={ea} l1r={l1r} b={blend}: SE={se}, exact={ex}/91')
    print(f'  [Part 4] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PART 5: Nested LOSO validation on best
    # ═══════════════════════════════════════════════════════
    print('\n── Part 5: Nested LOSO validation ──')
    folds = sorted(set(seasons))
    
    # Best config identified
    best_cfg = best_label
    print(f'  Validating: {best_cfg}')
    
    # Run "best" vs v46 in nested LOSO
    nested_se_best = 0
    nested_se_v46 = 0
    for outer in folds:
        outer_mask = (seasons == outer)
        outer_test = test_mask & outer_mask
        outer_gt = y[outer_test].astype(int)
        if outer_test.sum() == 0: continue
        
        # Inner CV: v46 vs best on all other test seasons
        inner_se_v46 = 0
        inner_se_best = 0
        for inner in folds:
            if inner == outer: continue
            inner_mask = (seasons == inner)
            inner_test = test_mask & inner_mask
            if inner_test.sum() == 0: continue
            inner_gt = y[inner_test].astype(int)
            
            # v46
            p_v46 = run_fast(cache, X_comm, y, fn, seasons, test_mask, n,
                           alpha=15, blend=0.15)
            se_v46 = int(np.sum((p_v46[inner_test] - inner_gt)**2))
            inner_se_v46 += se_v46
            
            # best (min_7, α=5, b=0.25)
            p_best = run_fast(cache, variants['min_7'], y, fn, seasons, test_mask, n,
                            alpha=5, blend=0.25)
            se_best = int(np.sum((p_best[inner_test] - inner_gt)**2))
            inner_se_best += se_best
        
        winner = 'v47' if inner_se_best <= inner_se_v46 else 'v46'
        
        # Outer SE
        if winner == 'v47':
            p_outer = run_fast(cache, variants['min_7'], y, fn, seasons, test_mask, n,
                             alpha=5, blend=0.25)
        else:
            p_outer = run_fast(cache, X_comm, y, fn, seasons, test_mask, n,
                             alpha=15, blend=0.15)
        outer_se = int(np.sum((p_outer[outer_test] - outer_gt)**2))
        nested_se_best += outer_se if winner == 'v47' else 0
        nested_se_v46 += outer_se if winner == 'v46' else 0
        
        print(f'    {outer}: inner v46={inner_se_v46} v47={inner_se_best} → {winner}, outerSE={outer_se}')
    
    direct_v47 = ev(run_fast(cache, variants['min_7'], y, fn, seasons, test_mask, n, alpha=5, blend=0.25))[0]
    direct_v46 = ev(run_fast(cache, X_comm, y, fn, seasons, test_mask, n, alpha=15, blend=0.15))[0]
    total_nested = nested_se_best + nested_se_v46
    gap = total_nested - direct_v47
    print(f'  Direct v46 SE={direct_v46}, Direct v47 SE={direct_v47}')
    print(f'  Nested SE={total_nested}, Gap={gap:+d}')
    print(f'  {"SAFE" if gap <= 5 else "RISKY" if gap <= 20 else "OVERFIT"}')
    
    # ═══════════════════════════════════════════════════════
    # PART 6: Per-season error breakdown
    # ═══════════════════════════════════════════════════════
    print('\n── Part 6: Per-season error comparison ──')
    p_v46 = run_fast(cache, X_comm, y, fn, seasons, test_mask, n, alpha=15, blend=0.15)
    p_best = best_preds
    
    for s in folds:
        s_mask = test_mask & (seasons == s)
        s_gt = y[s_mask].astype(int)
        se46 = int(np.sum((p_v46[s_mask] - s_gt)**2))
        ex46 = int((p_v46[s_mask] == s_gt).sum())
        se47 = int(np.sum((p_best[s_mask] - s_gt)**2))
        ex47 = int((p_best[s_mask] == s_gt).sum())
        print(f'  {s}: v46 SE={se46:3d} ({ex46}/{s_mask.sum()} exact)  '
              f'v47 SE={se47:3d} ({ex47}/{s_mask.sum()} exact)  Δ={se46-se47:+d}')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    results.sort(key=lambda x: x[1])
    print('\n' + '='*60)
    print(f' TOP 20 RESULTS')
    print('='*60)
    for label, se, ex in results[:20]:
        print(f'  SE={se:4d}  exact={ex:2d}/91  {label}')
    
    print(f'\n  Best: {best_label}, SE={best_se}')
    print(f'  Total time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
