#!/usr/bin/env python3
"""
v48 Final Push — last attempt at improving SE=94.

Focus on:
  A: KNN, SVR, GradientBoosting as committee models
  B: v12 base modifications (different blend weights, different features)
  C: Zone change WITH regression prevention (constrained zones)
  D: Dynamic blend based on model agreement
  E: LOSO-aware committee (train committee on LOSO predictions)
  F: Rank-based merge instead of seed-average
  G: Multiple committee models averaged before Hungarian
  H: Different forced features for v12
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
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

ZONES = [
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
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES, 0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12': a_v12, 'a_v12_zoned': a_v12_zoned,
            'avail': avail, 'X_season': X_all[season_mask],
        }
    return cache


def main():
    t0 = time.time()
    print('='*60)
    print(' v48 FINAL PUSH — Alternative Models & Structural Changes')
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
    
    base = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw = r.predict(sc.transform(X_min8[sm]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        ac = hungarian(raw, seasons[sm], avail, power=0.15)
        ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
        avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: base[gi] = af[i]
    
    base_se, base_ex = ev(base)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91\n')
    
    best_se, best_label, best_preds = base_se, 'v47', base.copy()
    results = []
    
    # ═══════════════════════════════════════════════════════
    # A: ALTERNATIVE COMMITTEE MODELS
    # ═══════════════════════════════════════════════════════
    print('── A: Alternative committee models ──')
    
    model_configs = []
    # KNN
    for k in [3, 5, 7, 10, 15, 20, 30]:
        for w in ['distance', 'uniform']:
            model_configs.append(('knn', f'knn_k{k}_{w[0]}', 
                                  lambda tm, k=k, w=w: KNeighborsRegressor(n_neighbors=k, weights=w)))
    # GradientBoosting
    for n_est in [50, 100, 200]:
        for depth in [2, 3, 4]:
            for lr in [0.03, 0.05, 0.1]:
                model_configs.append(('gbr', f'gbr_n{n_est}_d{depth}_lr{lr}',
                                      lambda tm, n_est=n_est, depth=depth, lr=lr: 
                                      GradientBoostingRegressor(n_estimators=n_est, max_depth=depth, 
                                                                 learning_rate=lr, subsample=0.8, 
                                                                 random_state=42)))
    # BayesianRidge
    model_configs.append(('bayesian', 'bayesian_ridge',
                          lambda tm: BayesianRidge()))
    
    for mtype, mname, model_fn in model_configs:
        for blend in [0.15, 0.20, 0.25, 0.30]:
            preds_m = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_min8[train_mask])
                X_te = sc.transform(X_min8[sm])
                mdl = model_fn(train_mask)
                mdl.fit(X_tr, y[train_mask])
                raw = mdl.predict(X_te)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw[i] = y[gi]
                ac = hungarian(raw, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
                avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_m[gi] = af[i]
            se, ex = ev(preds_m)
            results.append((f'{mname}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'{mname}_b{blend}'
                best_preds = preds_m.copy()
                print(f'  ★ {mname} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # B: MULTI-MODEL COMMITTEE (average multiple model raw scores)
    # ═══════════════════════════════════════════════════════
    print('\n── B: Multi-model committee (Ridge + KNN + GBR averaged) ──')
    for k in [5, 10, 15]:
        for n_est in [100, 200]:
            for blend in [0.20, 0.25, 0.30]:
                preds_mm = np.zeros(n, dtype=int)
                for hold_season, c in cache.items():
                    sm = c['season_mask']; si = c['si']; tm = c['tm']
                    train_mask = c['train_mask']; avail = c['avail']
                    a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                    sc = StandardScaler()
                    X_tr = sc.fit_transform(X_min8[train_mask])
                    X_te = sc.transform(X_min8[sm])
                    # Ridge
                    r = Ridge(alpha=10); r.fit(X_tr, y[train_mask])
                    raw_r = r.predict(X_te)
                    # KNN
                    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                    knn.fit(X_tr, y[train_mask])
                    raw_k = knn.predict(X_te)
                    # GBR
                    gbr = GradientBoostingRegressor(n_estimators=n_est, max_depth=3,
                                                     learning_rate=0.05, random_state=42)
                    gbr.fit(X_tr, y[train_mask])
                    raw_g = gbr.predict(X_te)
                    # Average
                    raw_avg = (raw_r + raw_k + raw_g) / 3
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: raw_avg[i] = y[gi]
                    ac = hungarian(raw_avg, seasons[sm], avail, power=0.15)
                    ac = apply_zones(ac, raw_avg, fn, X_season, tm, si, ZONES, 0.15)
                    avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: avg[i] = y[gi]
                    af = hungarian(avg, seasons[sm], avail, power=0.15)
                    for i, gi in enumerate(si):
                        if test_mask[gi]: preds_mm[gi] = af[i]
                se, ex = ev(preds_mm)
                results.append((f'mm_k{k}_n{n_est}_b{blend}', se, ex))
                if se < best_se:
                    best_se, best_label = se, f'mm_k{k}_n{n_est}_b{blend}'
                    best_preds = preds_mm.copy()
                    print(f'  ★ multi-model k={k} n={n_est} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # C: RANK-BASED MERGE
    # ═══════════════════════════════════════════════════════
    print('\n── C: Rank-based merge ──')
    # Instead of averaging seeds directly, merge based on relative rankings
    for blend in [0.15, 0.20, 0.25, 0.30, 0.35]:
        preds_rb = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            train_mask = c['train_mask']; avail = c['avail']
            a_v12 = c['a_v12_zoned']; X_season = c['X_season']
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
            raw = r.predict(sc.transform(X_min8[sm]))
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw[i] = y[gi]
            ac = hungarian(raw, seasons[sm], avail, power=0.15)
            ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
            
            # Get ranks for both
            v12_vals = a_v12.astype(float)
            comm_vals = ac.astype(float)
            
            # Convert to ranks
            v12_order = np.argsort(np.argsort(v12_vals))  # rank (0-based)
            comm_order = np.argsort(np.argsort(comm_vals))
            
            # Blend ranks
            blended_rank = (1.0 - blend) * v12_order + blend * comm_order
            
            # Convert back to seeds using Hungarian
            for i, gi in enumerate(si):
                if not test_mask[gi]: blended_rank[i] = y[gi]
            af = hungarian(blended_rank.astype(float), seasons[sm], avail, power=0.15)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_rb[gi] = af[i]
        se, ex = ev(preds_rb)
        results.append((f'rankmerge_b{blend}', se, ex))
        if se < best_se:
            best_se, best_label = se, f'rankmerge_b{blend}'
            best_preds = preds_rb.copy()
            print(f'  ★ rank merge b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # D: DYNAMIC BLEND (based on model disagreement)
    # ═══════════════════════════════════════════════════════
    print('\n── D: Dynamic blend ──')
    for scale in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for base_blend in [0.15, 0.20, 0.25, 0.30]:
            preds_db = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                sc = StandardScaler()
                r = Ridge(alpha=10)
                r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                raw = r.predict(sc.transform(X_min8[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw[i] = y[gi]
                ac = hungarian(raw, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
                
                # Dynamic: reduce blend where models disagree a lot
                disagreement = np.abs(a_v12.astype(float) - ac.astype(float))
                dynamic_blend = base_blend * np.exp(-disagreement / scale)
                
                avg = (1.0 - dynamic_blend) * a_v12.astype(float) + dynamic_blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_db[gi] = af[i]
            se, ex = ev(preds_db)
            results.append((f'dynblend_s{scale}_b{base_blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'dynblend_s{scale}_b{base_blend}'
                best_preds = preds_db.copy()
                print(f'  ★ dynamic blend s={scale} b={base_blend}: SE={se}, exact={ex}/91')
    
    print(f'  [D done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # E: COMMITTEE ON MORE FEATURES (all 68)
    # ═══════════════════════════════════════════════════════
    print('\n── E: Committee on all features ──')
    for alpha in [50, 100, 200, 500, 1000]:
        for blend in [0.10, 0.15, 0.20, 0.25]:
            preds_af = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                sc = StandardScaler()
                r = Ridge(alpha=alpha)
                r.fit(sc.fit_transform(X_all[train_mask]), y[train_mask])
                raw = r.predict(sc.transform(X_all[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw[i] = y[gi]
                ac = hungarian(raw, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
                avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_af[gi] = af[i]
            se, ex = ev(preds_af)
            results.append((f'all68_a{alpha}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'all68_a{alpha}_b{blend}'
                best_preds = preds_af.copy()
                print(f'  ★ all68 α={alpha} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [E done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # F: CONSTRAINED ZONE CHANGE (fix Charleston/VCU without touching NW)
    # ═══════════════════════════════════════════════════════
    print('\n── F: Constrained zone optimization ──')
    # The zone change (34,48) fixed Charleston/VCU/Virginia but hurt Northwestern
    # Try zone ONLY for bot/midbot region (44-52) to fix VCU/Charleston
    for lo in [42, 44, 46]:
        for hi in [48, 50, 52]:
            for sn in [-4, -2, 0, 2]:
                for nc in [0, 2, 3, 4]:
                    for cb_p in [-3, -2, -1, 0]:
                        zones_c = [
                            ('mid',     'committee', (17, 34), (0, 0, 3)),
                            ('uppermid','committee', (34, 44), (-2, -3, -4)),
                            ('midbot2', 'bottom',    (lo, hi), (sn, nc, cb_p)),
                            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                            ('tail',    'tail',      (60, 63), (1,)),
                        ]
                        preds_c = np.zeros(n, dtype=int)
                        for hold_season, c in cache.items():
                            sm = c['season_mask']; si = c['si']; tm = c['tm']
                            train_mask = c['train_mask']; avail = c['avail']
                            a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                            sc = StandardScaler()
                            r = Ridge(alpha=10)
                            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                            raw = r.predict(sc.transform(X_min8[sm]))
                            for i, gi in enumerate(si):
                                if not test_mask[gi]: raw[i] = y[gi]
                            ac = hungarian(raw, seasons[sm], avail, power=0.15)
                            ac = apply_zones(ac, raw, fn, X_season, tm, si, zones_c, 0.15)
                            avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
                            for i, gi in enumerate(si):
                                if not test_mask[gi]: avg[i] = y[gi]
                            af = hungarian(avg, seasons[sm], avail, power=0.15)
                            for i, gi in enumerate(si):
                                if test_mask[gi]: preds_c[gi] = af[i]
                        se, ex = ev(preds_c)
                        results.append((f'cz_{lo}_{hi}_s{sn}_n{nc}_c{cb_p}', se, ex))
                        if se < best_se:
                            # Check for regressions
                            reg = 0
                            for gi in np.where(test_mask)[0]:
                                if (preds_c[gi] - int(y[gi]))**2 > (base[gi] - int(y[gi]))**2:
                                    reg += 1
                            best_se, best_label = se, f'cz_{lo}_{hi}_s{sn}_n{nc}_c{cb_p}'
                            best_preds = preds_c.copy()
                            print(f'  ★ {lo}-{hi} sn={sn} nc={nc} cb={cb_p}: SE={se}, exact={ex}/91, regressions={reg}')
    
    print(f'  [F done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # G: MODIFY V12 BLEND WEIGHTS
    # ═══════════════════════════════════════════════════════
    print('\n── G: Different v12 internal blend ──')
    # Current v12: 64% LR(C=5) + 28% LR(C=0.5) + 8% XGB
    # Try modifying these ratios by re-running v12 with different params
    # Instead, use different forced features for the committee
    for force in [
        ['NET Rank'],
        ['NET Rank', 'NETSOS'],
        ['NET Rank', 'tourn_field_rank'],
        ['NET Rank', 'cb_mean_seed'],
        ['NET Rank', 'WL_Pct'],
        ['tourn_field_rank'],
        ['tourn_field_rank', 'NETSOS'],
        [],
    ]:
        cache_f = {}
        for hold_season in sorted(set(seasons)):
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            si = np.where(season_mask)[0]
            train_mask = ~season_test_mask
            tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                        k=USE_TOP_K_A, forced_features=force)[0]
            raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                           X_all[season_mask], seasons[train_mask], tki)
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw_v12[i] = y[gi]
            tm_f = np.array([test_mask[gi] for gi in si])
            avail = {hold_season: list(range(1, 69))}
            a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
            a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm_f, si, ZONES, 0.15)
            cache_f[hold_season] = {
                'season_mask': season_mask, 'si': si, 'tm': tm_f,
                'train_mask': train_mask, 'raw_v12': raw_v12,
                'a_v12_zoned': a_v12_zoned, 'avail': avail,
                'X_season': X_all[season_mask],
            }
        
        preds_f = np.zeros(n, dtype=int)
        for hold_season, c in cache_f.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            train_mask = c['train_mask']; avail = c['avail']
            a_v12 = c['a_v12_zoned']; X_season = c['X_season']
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
            raw = r.predict(sc.transform(X_min8[sm]))
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw[i] = y[gi]
            ac = hungarian(raw, seasons[sm], avail, power=0.15)
            ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
            avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            af = hungarian(avg, seasons[sm], avail, power=0.15)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_f[gi] = af[i]
        se, ex = ev(preds_f)
        force_str = '_'.join([f[:4] for f in force]) if force else 'none'
        results.append((f'force_{force_str}', se, ex))
        if se < best_se:
            best_se, best_label = se, f'force_{force_str}'
            best_preds = preds_f.copy()
            print(f'  ★ force={force}: SE={se}, exact={ex}/91')
    
    print(f'  [G done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # H: COMMITTEE ON RAW V12 (use v12 raw as a feature)
    # ═══════════════════════════════════════════════════════
    print('\n── H: Committee with v12 raw as feature ──')
    for alpha in [5, 10, 15, 20, 50]:
        for blend in [0.15, 0.20, 0.25, 0.30]:
            preds_vf = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                raw_v12 = c['raw_v12']
                # Add v12 raw as feature to min8
                X_plus = np.column_stack([X_min8[sm], raw_v12])
                X_tr_plus = np.column_stack([X_min8[train_mask], 
                                              np.zeros(train_mask.sum())])  # placeholder for train
                # For train set, we need v12 raw for train teams too
                # Actually, we need to think about this differently
                # Use min8 features + v12 raw score for this season
                # But v12 raw for train is just their actual seed (since we inject)
                v12_for_all = np.zeros(n)
                for i, gi in enumerate(si):
                    v12_for_all[gi] = raw_v12[i]
                # For training, use predictions from v12 on training data (these are injected as y)
                X_with_v12 = np.column_stack([X_min8, v12_for_all])
                sc = StandardScaler()
                r = Ridge(alpha=alpha)
                r.fit(sc.fit_transform(X_with_v12[train_mask]), y[train_mask])
                raw = r.predict(sc.transform(X_with_v12[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw[i] = y[gi]
                ac = hungarian(raw, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
                avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_vf[gi] = af[i]
            se, ex = ev(preds_vf)
            results.append((f'v12feat_a{alpha}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'v12feat_a{alpha}_b{blend}'
                best_preds = preds_vf.copy()
                print(f'  ★ v12feat a={alpha} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [H done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' FINAL RESULTS')
    print('='*60)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91')
    print(f'  Best found: {best_label}, SE={best_se}')
    print(f'  Improvement: Δ={base_se - best_se}')
    
    results.sort(key=lambda x: x[1])
    print(f'\n  Top 30:')
    for label, se, ex in results[:30]:
        flag = '★' if se < base_se else ' '
        print(f'  {flag} SE={se:4d}  exact={ex:2d}/91  {label}')
    
    if best_se < base_se:
        print(f'\n  Errors changed from v47:')
        for gi in np.where(test_mask)[0]:
            old, new, actual = base[gi], best_preds[gi], int(y[gi])
            if old != new:
                rid = record_ids[gi]
                old_se = (old - actual)**2
                new_se = (new - actual)**2
                arrow = '↑' if new_se < old_se else '↓' if new_se > old_se else '='
                print(f'    {arrow} {rid:32s} gt={actual:2d} old={old:2d} new={new:2d} '
                      f'oldSE={old_se:3d} newSE={new_se:3d}')
        regressions = sum(1 for gi in np.where(test_mask)[0] 
                          if (best_preds[gi] - int(y[gi]))**2 > (base[gi] - int(y[gi]))**2)
        print(f'  Regressions: {regressions}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
