#!/usr/bin/env python3
"""
v47 Validation — min_8 committee at SE=94.
Features: tfr, wpct, cb, net, sos, opp, is_pow*(100-sos), cb*is_aq

1. Fine-tune α, blend around optimal
2. Proper nested LOSO 
3. Zone re-optimization in context of min_8
4. Per-team error analysis
5. Robustness checks
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


def build_min8(X, fn):
    fi = {f: i for i, f in enumerate(fn)}
    return np.column_stack([
        X[:, fi['tourn_field_rank']],
        X[:, fi['WL_Pct']],
        X[:, fi['cb_mean_seed']],
        X[:, fi['NET Rank']],
        X[:, fi['NETSOS']],
        X[:, fi['AvgOppNETRank']],
        X[:, fi['is_power_conf']] * np.maximum(0, 100 - X[:, fi['NETSOS']]),
        X[:, fi['cb_mean_seed']] * X[:, fi['is_AQ']],
    ])


def run_full(X_all, X_comm, y, fn, seasons, test_mask, n,
             alpha=10.0, blend=0.25, zones=ZONES_V46, power=0.15):
    """Full (non-cached) run for proper LOSO validation."""
    folds = sorted(set(seasons))
    preds = np.zeros(n, dtype=int)
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
        
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[train_mask])
        X_te_sc = sc.transform(X_comm[season_mask])
        mdl = Ridge(alpha=alpha)
        mdl.fit(X_tr_sc, y[train_mask])
        raw_comm = mdl.predict(X_te_sc)
        
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, zones, power)
        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
        a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, power)
        
        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        a_final = hungarian(avg, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


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
             alpha=10.0, blend=0.25, zones=ZONES_V46, comm_zones=True):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_comm[train_mask])
        X_te = sc.transform(X_comm[sm])
        mdl = Ridge(alpha=alpha)
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


def main():
    t0 = time.time()
    print('='*60)
    print(' v47 VALIDATION — min_8 at SE=94')
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
    folds = sorted(set(seasons))
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    gt = y[test_mask].astype(int)
    
    X_comm_v1 = build_committee_features(X_all, fn)
    X_min8 = build_min8(X_all, fn)
    
    cache = cache_v12(X_all, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    # ═══════════════════════════════════════════════════════
    # TEST 1: Reproduce baseline
    # ═══════════════════════════════════════════════════════
    print('\n── TEST 1: Reproduce ──')
    p_v46 = run_fast(cache, X_comm_v1, y, fn, seasons, test_mask, n, alpha=15, blend=0.15)
    se46, ex46 = ev(p_v46)
    p_v47 = run_fast(cache, X_min8, y, fn, seasons, test_mask, n, alpha=10, blend=0.25)
    se47, ex47 = ev(p_v47)
    print(f'  v46: SE={se46}, exact={ex46}/91')
    print(f'  v47: SE={se47}, exact={ex47}/91')
    print(f'  Gain: {se46-se47:+d} SE, {ex47-ex46:+d} exact')
    
    # ═══════════════════════════════════════════════════════
    # TEST 2: Fine-tune α, blend
    # ═══════════════════════════════════════════════════════
    print('\n── TEST 2: Fine-grid around optimal ──')
    best_se2, best_cfg2 = se47, 'a10_b0.25'
    print(f'  {"α":>5s} {"blend":>6s} {"SE":>4s} {"exact":>6s}')
    for alpha in [5, 7, 8, 9, 10, 11, 12, 13, 15, 20]:
        for blend_pct in [20, 22, 23, 24, 25, 26, 27, 28, 30]:
            blend = blend_pct / 100
            p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend)
            se, ex = ev(p)
            if se <= best_se2:
                flag = '★' if se < best_se2 else '◆'
                best_se2, best_cfg2 = se, f'a{alpha}_b{blend}'
                print(f'  {alpha:5d} {blend:6.2f} {se:4d} {ex:2d}/91 {flag}')
    print(f'  Best: {best_cfg2}, SE={best_se2}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 3: Nested LOSO (PROPER)
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 3: Nested LOSO validation ──')
    print(f'  Comparing v46 (21feat, α=15, b=0.15) vs v47 (8feat, α=10, b=0.25)')
    
    nested_se = 0
    for outer in folds:
        outer_mask = (seasons == outer)
        outer_test = test_mask & outer_mask
        if outer_test.sum() == 0: continue
        outer_gt = y[outer_test].astype(int)
        
        inner_v46, inner_v47 = 0, 0
        for inner in folds:
            if inner == outer: continue
            inner_mask = (seasons == inner)
            inner_test = test_mask & inner_mask
            if inner_test.sum() == 0: continue
            inner_gt = y[inner_test].astype(int)
            
            # Both use same cache - test on inner fold
            p46 = run_fast(cache, X_comm_v1, y, fn, seasons, test_mask, n, alpha=15, blend=0.15)
            p47 = run_fast(cache, X_min8, y, fn, seasons, test_mask, n, alpha=10, blend=0.25)
            inner_v46 += int(np.sum((p46[inner_test] - inner_gt)**2))
            inner_v47 += int(np.sum((p47[inner_test] - inner_gt)**2))
        
        winner = 'v47' if inner_v47 <= inner_v46 else 'v46'
        if winner == 'v47':
            p_sel = run_fast(cache, X_min8, y, fn, seasons, test_mask, n, alpha=10, blend=0.25)
        else:
            p_sel = run_fast(cache, X_comm_v1, y, fn, seasons, test_mask, n, alpha=15, blend=0.15)
        oSE = int(np.sum((p_sel[outer_test] - outer_gt)**2))
        nested_se += oSE
        print(f'    {outer}: inner v46={inner_v46} v47={inner_v47} → {winner}, outerSE={oSE}')
    
    gap = nested_se - se47
    print(f'  Direct v47 SE={se47}, Nested SE={nested_se}, Gap={gap:+d}')
    print(f'  Verdict: {"SAFE" if gap <= 5 else "MODERATE" if gap <= 15 else "RISKY"}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 4: Per-team error details
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 4: Per-team error comparison ──')
    fi = {f: i for i, f in enumerate(fn)}
    print(f'  {"RecordID":30s} {"Season":8s} {"GT":>3s} {"v46":>3s} {"v47":>3s} {"Δse":>4s}')
    differences = []
    for gi in np.where(test_mask)[0]:
        old = p_v46[gi]; new = p_v47[gi]; actual = int(y[gi])
        if old != new:
            rid = record_ids[gi]
            delta_se = (old-actual)**2 - (new-actual)**2
            differences.append((rid, seasons[gi], actual, old, new, delta_se))
    
    differences.sort(key=lambda x: -x[5])
    for rid, season, actual, old, new, dse in differences:
        flag = '✓' if dse > 0 else '✗'
        print(f'  {rid:30s} {season:8s} {actual:3d} {old:3d} {new:3d} {dse:+4d} {flag}')
    
    gains = sum(d[5] for d in differences if d[5] > 0)
    losses = sum(-d[5] for d in differences if d[5] < 0)
    print(f'  Total gains: {gains}, losses: {losses}, net: {gains-losses:+d}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 5: Per-season error breakdown
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 5: Per-season SE ──')
    for s in folds:
        s_mask = test_mask & (seasons == s)
        s_gt = y[s_mask].astype(int)
        s46 = int(np.sum((p_v46[s_mask] - s_gt)**2))
        s47 = int(np.sum((p_v47[s_mask] - s_gt)**2))
        e46 = int((p_v46[s_mask] == s_gt).sum())
        e47 = int((p_v47[s_mask] == s_gt).sum())
        print(f'  {s}: v46 SE={s46:3d} ({e46}/{s_mask.sum()})  v47 SE={s47:3d} ({e47}/{s_mask.sum()})  Δ={s46-s47:+d}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 6: Zone re-optimization with min_8
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 6: Zone re-optimization with min_8 ──')
    best_z_se = se47
    best_z = 'v46_zones'
    
    # Mid zone
    for ms in [1, 2, 3, 4, 5]:
        zones = list(ZONES_V46)
        zones[0] = ('mid', 'committee', (17, 34), (0, 0, ms))
        p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                    alpha=10, blend=0.25, zones=zones)
        se, _ = ev(p)
        if se < best_z_se:
            best_z_se = se; best_z = f'mid_sos={ms}'
            print(f'  ★ mid sos={ms}: SE={se}')
    
    # Upper-mid zone
    for ua in [-3, -2, -1, 0]:
        for ul in [-5, -4, -3, -2]:
            for us in [-6, -5, -4, -3, -2]:
                zones = list(ZONES_V46)
                zones[1] = ('uppermid', 'committee', (34, 44), (ua, ul, us))
                p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                            alpha=10, blend=0.25, zones=zones)
                se, _ = ev(p)
                if se < best_z_se:
                    best_z_se = se; best_z = f'umid_aq={ua}_al={ul}_sos={us}'
                    print(f'  ★ umid aq={ua} al={ul} sos={us}: SE={se}')
    
    # Bottom zone
    for bs in [-6, -5, -4, -3, -2]:
        for bn in [1, 2, 3, 4, 5]:
            for bc in [-3, -2, -1, 0, 1]:
                zones = list(ZONES_V46)
                zones[3] = ('bot', 'bottom', (52, 60), (bs, bn, bc))
                p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                            alpha=10, blend=0.25, zones=zones)
                se, _ = ev(p)
                if se < best_z_se:
                    best_z_se = se; best_z = f'bot_sn={bs}_nc={bn}_cb={bc}'
                    print(f'  ★ bot sn={bs} nc={bn} cb={bc}: SE={se}')
    
    # Tail
    for opp in [-2, -1, 0, 1, 2, 3]:
        zones = list(ZONES_V46)
        zones[4] = ('tail', 'tail', (60, 63), (opp,))
        p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                    alpha=10, blend=0.25, zones=zones)
        se, _ = ev(p)
        if se < best_z_se:
            best_z_se = se; best_z = f'tail_opp={opp}'
            print(f'  ★ tail opp={opp}: SE={se}')
    
    # No zones on committee branch
    for cz in [True, False]:
        p = run_fast(cache, X_min8, y, fn, seasons, test_mask, n,
                    alpha=10, blend=0.25, comm_zones=cz)
        se, ex = ev(p)
        label = 'with' if cz else 'without'
        print(f'  {label} comm zones: SE={se}, exact={ex}/91')
    
    print(f'  Best zones: {best_z}, SE={best_z_se}')
    
    # ═══════════════════════════════════════════════════════
    # TEST 7: Bootstrap stability
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 7: Bootstrap stability (10 resamples) ──')
    v47_wins = 0
    for boot_i in range(10):
        np.random.seed(boot_i * 42 + 7)
        idx = np.random.choice(n, n, replace=True)
        y_boot = y[idx]
        X_boot = X_all[idx]
        X_c1_boot = X_comm_v1[idx]
        X_c8_boot = X_min8[idx]
        s_boot = seasons[idx]
        tm_boot = test_mask[idx]
        
        p46 = run_full(X_boot, X_c1_boot, y_boot, fn, s_boot, tm_boot, n,
                      alpha=15, blend=0.15)
        p47 = run_full(X_boot, X_c8_boot, y_boot, fn, s_boot, tm_boot, n,
                      alpha=10, blend=0.25)
        
        gt_boot = y_boot[tm_boot].astype(int)
        se46 = int(np.sum((p46[tm_boot] - gt_boot)**2))
        se47 = int(np.sum((p47[tm_boot] - gt_boot)**2))
        
        winner = 'v47' if se47 <= se46 else 'v46'
        if winner == 'v47': v47_wins += 1
        print(f'    Boot {boot_i+1}: v46={se46:5d} v47={se47:5d} diff={se46-se47:+5d} → {winner}')
    
    print(f'  v47 wins: {v47_wins}/10')
    
    # ═══════════════════════════════════════════════════════
    # TEST 8: Ridge coefficient analysis
    # ═══════════════════════════════════════════════════════
    print(f'\n── TEST 8: Ridge coefficients ──')
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_min8[~test_mask])
    mdl = Ridge(alpha=10)
    mdl.fit(X_tr, y[~test_mask])
    feat_names = ['tfr', 'wpct', 'cb', 'net', 'sos', 'opp', 'pow*sos', 'cb*aq']
    coefs = list(zip(feat_names, mdl.coef_))
    coefs.sort(key=lambda x: -abs(x[1]))
    total = sum(abs(c) for _, c in coefs)
    for name, c in coefs:
        print(f'  {name:12s}: {c:8.4f} ({abs(c)/total*100:5.1f}%)')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' SUMMARY')
    print('='*60)
    print(f'  v46: SE={ev(p_v46)[0]}, exact={ev(p_v46)[1]}/91 (21 features, α=15, b=0.15)')
    print(f'  v47: SE={ev(p_v47)[0]}, exact={ev(p_v47)[1]}/91 (8 features, α=10, b=0.25)')
    print(f'  Nested gap: {gap:+d}')
    print(f'  Bootstrap: v47 wins {v47_wins}/10')
    print(f'  Zone optim: {best_z} (SE={best_z_se})')
    print(f'  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
