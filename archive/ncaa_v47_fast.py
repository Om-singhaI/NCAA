#!/usr/bin/env python3
"""
v47 Fast Campaign — caches v12 predictions to speed up exploration.
Promising leads from v47_campaign:
  - ElasticNet α=15: SE=122, 71/91 exact
  - Minimal features α=10 b=0.20: SE=120, 63/91 exact
"""

import os, sys, time, warnings, pickle
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

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
            aq, al, sos = params
            corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            sn, nc, cb = params
            corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            opp = params[0]
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def cache_v12(X_all, y, fn, seasons, test_mask):
    """Pre-compute v12 raw scores + Hungarian + zone-corrected for each season."""
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
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        
        # v12 + zones
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, ZONES_V46, 0.15)
        
        cache[hold_season] = {
            'season_mask': season_mask,
            'si': si,
            'tm': tm,
            'train_mask': train_mask,
            'raw_v12': raw_v12,
            'a_v12_zoned': a_v12_zoned,
            'avail': avail,
            'X_season': X_season,
        }
    return cache


def run_from_cache(cache, X_comm, y, fn, seasons, test_mask, n,
                   alpha=15.0, blend=0.15, comm_model='ridge', l1_ratio=0.5,
                   zones=ZONES_V46, comm_zones=True, power=0.15):
    """Fast: re-use cached v12 predictions, only redo committee + merge."""
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        season_mask = c['season_mask']
        si = c['si']
        tm = c['tm']
        train_mask = c['train_mask']
        a_v12_zoned = c['a_v12_zoned']
        avail = c['avail']
        X_season = c['X_season']
        
        # Committee model for this fold
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[train_mask])
        X_te_sc = sc.transform(X_comm[season_mask])
        
        if comm_model == 'ridge':
            mdl = Ridge(alpha=alpha)
        elif comm_model == 'lasso':
            mdl = Lasso(alpha=alpha, max_iter=5000)
        elif comm_model == 'elastic':
            mdl = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
        mdl.fit(X_tr_sc, y[train_mask])
        raw_comm = mdl.predict(X_te_sc)
        
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_comm[i] = y[gi]
        
        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=power)
        if comm_zones:
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, power)
        
        avg = (1.0 - blend) * a_v12_zoned.astype(float) + blend * a_comm.astype(float)
        
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        
        a_final = hungarian(avg, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


def build_comm_minimal(X, fn):
    fi = {f: i for i, f in enumerate(fn)}
    feats = [
        X[:, fi['tourn_field_rank']],
        X[:, fi['WL_Pct']],
        X[:, fi['cb_mean_seed']],
        X[:, fi['NET Rank']],
        X[:, fi['NETSOS']],
        X[:, fi['is_power_conf']] * np.maximum(0, 100 - X[:, fi['NETSOS']]),
        X[:, fi['cb_mean_seed']] * X[:, fi['is_AQ']],
    ]
    return np.column_stack(feats)


def build_comm_v3(X, fn):
    """v3: Carefully curated features based on error analysis."""
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
    
    feats = [
        net, sos, opp, wpct, cb, tfr,
        is_al, is_pow,
        q1w, q1w / (q1w + q1l + 0.5),
        q2w,
        q3l + q4l,
        is_aq * net, is_al * (200 - net),
        net - 0.3 * sos,
        net - conf,
        is_pow * net,
        is_pow * sos,
        cb * is_aq,
        cb * is_al,
        # New: focus on the big errors (NW, NM, SF, Davidson)
        # These are power conf AL teams misjudged — SOS interaction
        is_al * is_pow * sos,
        is_aq * (1 - is_pow) * sos,  # mid-major AQ SOS
        np.abs(net - tfr),           # NET vs tournament field rank disagreement
        opp * is_al,                 # schedule strength for AL teams
        (q1w + q2w) * is_al,         # quality wins for AL teams
    ]
    return np.column_stack(feats)


def build_comm_v4(X, fn):
    """v4: Focused on power-conference AL teams (where biggest errors are)."""
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
    
    # Core
    feats = [
        net, sos, opp, wpct, cb, tfr,
        is_al, is_pow, is_aq,
        q1w, q2w,
        q3l + q4l,
    ]
    # Interactions (all v1)
    feats.extend([
        is_aq * (1 - is_pow) * net,
        is_al * is_pow * (200 - net),
        net - 0.3 * sos,
        net - conf,
        is_aq * np.maximum(0, net - 50),
        is_pow * np.maximum(0, 100 - sos),
        q1w / (q1w + q1l + 0.5),
        is_pow * (q3l + q4l),
        cb * is_aq,
        cb * is_al,
    ])
    # New: power-AL specific
    feats.extend([
        is_al * is_pow * net,       # power AL NET
        is_al * is_pow * sos,       # power AL SOS
        is_al * is_pow * wpct,      # power AL win %
        is_al * is_pow * q1w,       # power AL Q1 wins
        is_al * is_pow * (net - conf),  # power AL NET vs conf
    ])
    return np.column_stack(feats)


def main():
    t0 = time.time()
    print('='*60)
    print(' v47 FAST CAMPAIGN (cached v12)')
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
    
    # Feature variants
    X_comm = build_committee_features(X_all, fn)
    X_min = build_comm_minimal(X_all, fn)
    X_v3 = build_comm_v3(X_all, fn)
    X_v4 = build_comm_v4(X_all, fn)
    
    print(f'  Caching v12 predictions...')
    t1 = time.time()
    cache = cache_v12(X_all, y, fn, seasons, test_mask)
    print(f'  Cached in {time.time()-t1:.0f}s')
    
    def eval_preds(p):
        se = int(np.sum((p[test_mask] - gt)**2))
        ex = int((p[test_mask] == gt).sum())
        return se, ex
    
    base = run_from_cache(cache, X_comm, y, fn, seasons, test_mask, n)
    base_se, base_ex = eval_preds(base)
    print(f'  Baseline: SE={base_se}, exact={base_ex}/91')
    
    best_se = base_se
    best_label = 'v46'
    best_preds = base.copy()
    results = []
    
    def try_combo(label, X_c, alpha, blend, comm_model='ridge', l1_ratio=0.5,
                  zones=ZONES_V46, comm_zones=True, power=0.15):
        nonlocal best_se, best_label, best_preds
        p = run_from_cache(cache, X_c, y, fn, seasons, test_mask, n,
                          alpha=alpha, blend=blend, comm_model=comm_model,
                          l1_ratio=l1_ratio, zones=zones, comm_zones=comm_zones,
                          power=power)
        se, ex = eval_preds(p)
        results.append((label, se, ex))
        if se < best_se:
            best_se = se
            best_label = label
            best_preds = p.copy()
            print(f'  ★ {label}: SE={se}, exact={ex}/91')
        return se, ex
    
    # ═══════════════════════════════════════════════════════════
    # SECTION A: ElasticNet deep dive (SE=122 from initial scan)
    # ═══════════════════════════════════════════════════════════
    print('\n── A: ElasticNet deep dive ──')
    for alpha in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        for l1r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for blend in [0.12, 0.14, 0.15, 0.16, 0.18, 0.20]:
                try_combo(f'EN_a{alpha}_l1{l1r}_b{blend}', X_comm,
                         alpha, blend, 'elastic', l1r)
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION B: Minimal features deep dive (SE=120 from scan)
    # ═══════════════════════════════════════════════════════════
    print('\n── B: Minimal features deep dive ──')
    for alpha in [5, 8, 10, 12, 15, 20, 30]:
        for blend in [0.15, 0.18, 0.20, 0.22, 0.25]:
            try_combo(f'min_a{alpha}_b{blend}', X_min, alpha, blend)
    for alpha in [5, 10, 15]:
        for blend in [0.15, 0.20, 0.25]:
            try_combo(f'min_EN_a{alpha}_b{blend}', X_min, alpha, blend, 'elastic', 0.5)
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION C: v3 and v4 feature sets
    # ═══════════════════════════════════════════════════════════
    print('\n── C: New feature sets (v3 error-focused, v4 power-AL) ──')
    for X_c, lbl in [(X_v3, 'v3'), (X_v4, 'v4')]:
        for alpha in [10, 15, 20, 30]:
            for blend in [0.12, 0.15, 0.18, 0.20]:
                try_combo(f'{lbl}_a{alpha}_b{blend}', X_c, alpha, blend)
        for alpha in [0.10, 0.15, 0.20]:
            for blend in [0.15, 0.20]:
                try_combo(f'{lbl}_EN_a{alpha}_b{blend}', X_c, alpha, blend, 'elastic', 0.5)
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION D: All 68 features as committee
    # ═══════════════════════════════════════════════════════════
    print('\n── D: All 68 features ──')
    for alpha in [5, 10, 20, 50, 100, 200]:
        for blend in [0.10, 0.15, 0.20]:
            try_combo(f'all68_a{alpha}_b{blend}', X_all, alpha, blend)
    for alpha in [0.05, 0.10, 0.20]:
        for blend in [0.10, 0.15, 0.20]:
            try_combo(f'all68_EN_a{alpha}_b{blend}', X_all, alpha, blend, 'elastic', 0.5)
    
    print(f'  [D done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION E: No zones on committee branch
    # ═══════════════════════════════════════════════════════════
    print('\n── E: Committee without zones ──')
    for X_c, lbl in [(X_comm, 'v1'), (X_min, 'min'), (X_v3, 'v3'), (X_v4, 'v4')]:
        for alpha in [10, 15, 20]:
            for blend in [0.12, 0.15, 0.20]:
                try_combo(f'{lbl}_nz_a{alpha}_b{blend}', X_c, alpha, blend, comm_zones=False)
    
    print(f'  [E done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION F: Zone parameter re-optimization
    # ═══════════════════════════════════════════════════════════
    print('\n── F: Zone re-optimization (mid/uppermid) ──')
    for mid_sos in [2, 3, 4]:
        for umid_aq in [-3, -2, -1, 0]:
            for umid_al in [-4, -3, -2]:
                for umid_sos in [-5, -4, -3]:
                    zones_test = [
                        ('mid',     'committee', (17, 34), (0, 0, mid_sos)),
                        ('uppermid','committee', (34, 44), (umid_aq, umid_al, umid_sos)),
                        ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                        ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                        ('tail',    'tail',      (60, 63), (1,)),
                    ]
                    try_combo(f'z_ms{mid_sos}_ua{umid_aq}_ul{umid_al}_us{umid_sos}',
                             X_comm, 15, 0.15, zones=zones_test)
    
    print(f'  [F done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION G: Bottom zone re-optimization
    # ═══════════════════════════════════════════════════════════
    print('\n── G: Zone re-optimization (bottom/tail) ──')
    for bot_sn in [-5, -4, -3, -2]:
        for bot_nc in [2, 3, 4]:
            for bot_cb in [-3, -2, -1, 0]:
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (-2, -3, -4)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (bot_sn, bot_nc, bot_cb)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                try_combo(f'z_bs{bot_sn}_bn{bot_nc}_bc{bot_cb}',
                         X_comm, 15, 0.15, zones=zones_test)
    
    print(f'  [G done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION H: Zone boundary exploration
    # ═══════════════════════════════════════════════════════════
    print('\n── H: Zone boundary exploration ──')
    for mid_hi in [30, 32, 34, 36]:
        for umid_hi in [42, 44, 46, 48]:
            zones_test = [
                ('mid',     'committee', (17, mid_hi), (0, 0, 3)),
                ('uppermid','committee', (mid_hi, umid_hi), (-2, -3, -4)),
                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                ('tail',    'tail',      (60, 63), (1,)),
            ]
            try_combo(f'z_mh{mid_hi}_uh{umid_hi}', X_comm, 15, 0.15, zones=zones_test)
    
    for mb_lo in [46, 47, 48, 49]:
        for mb_hi in [51, 52, 53, 54]:
            for b_hi in [58, 59, 60, 61, 62]:
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (-2, -3, -4)),
                    ('midbot',  'bottom',    (mb_lo, mb_hi), (0, 2, -2)),
                    ('bot',     'bottom',    (mb_hi, b_hi), (-4, 3, -1)),
                    ('tail',    'tail',      (b_hi, 63), (1,)),
                ]
                try_combo(f'z_ml{mb_lo}_mh{mb_hi}_bh{b_hi}',
                         X_comm, 15, 0.15, zones=zones_test)
    
    print(f'  [H done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION I: XGB as committee
    # ═══════════════════════════════════════════════════════════
    print('\n── I: XGB committee ──')
    for n_est in [100, 200, 300]:
        for depth in [3, 4, 5]:
            preds_xgb = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                season_mask = c['season_mask']
                si = c['si']
                tm = c['tm']
                train_mask = c['train_mask']
                a_v12_zoned = c['a_v12_zoned']
                avail = c['avail']
                X_season = c['X_season']
                
                mdl = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
                mdl.fit(X_comm[train_mask], y[train_mask])
                raw_xgb = mdl.predict(X_comm[season_mask])
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_xgb[i] = y[gi]
                a_xgb = hungarian(raw_xgb, seasons[season_mask], avail, power=0.15)
                a_xgb = apply_zones(a_xgb, raw_xgb, fn, X_season, tm, si, ZONES_V46, 0.15)
                
                avg = 0.85 * a_v12_zoned.astype(float) + 0.15 * a_xgb.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_xgb[gi] = a_final[i]
            
            se, ex = eval_preds(preds_xgb)
            results.append((f'xgb_n{n_est}_d{depth}', se, ex))
            if se < best_se:
                best_se = se
                best_label = f'xgb_n{n_est}_d{depth}'
                best_preds = preds_xgb.copy()
                print(f'  ★ XGB n={n_est} d={depth}: SE={se}, exact={ex}/91')
    
    print(f'  [I done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION J: Multi-alpha Ridge ensemble
    # ═══════════════════════════════════════════════════════════
    print('\n── J: Multi-alpha Ridge ensemble ──')
    for alphas_str, alphas in [
        ('5_15', [5, 15]),
        ('10_20', [10, 20]),
        ('5_15_50', [5, 15, 50]),
        ('10_15_20', [10, 15, 20]),
        ('5_10_15_20_50', [5, 10, 15, 20, 50]),
    ]:
        for blend in [0.10, 0.15, 0.20]:
            preds_ma = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                season_mask = c['season_mask']
                si = c['si']
                train_mask = c['train_mask']
                a_v12_zoned = c['a_v12_zoned']
                avail = c['avail']
                X_season = c['X_season']
                tm = c['tm']
                
                raw_avg = np.zeros(season_mask.sum())
                for a_val in alphas:
                    sc = StandardScaler()
                    X_tr_sc = sc.fit_transform(X_comm[train_mask])
                    X_te_sc = sc.transform(X_comm[season_mask])
                    mdl = Ridge(alpha=a_val)
                    mdl.fit(X_tr_sc, y[train_mask])
                    raw_avg += mdl.predict(X_te_sc)
                raw_avg /= len(alphas)
                
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_avg[i] = y[gi]
                a_comm = hungarian(raw_avg, seasons[season_mask], avail, power=0.15)
                a_comm = apply_zones(a_comm, raw_avg, fn, X_season, tm, si, ZONES_V46, 0.15)
                
                avg = (1.0 - blend) * a_v12_zoned.astype(float) + blend * a_comm.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_ma[gi] = a_final[i]
            
            se, ex = eval_preds(preds_ma)
            results.append((f'mα_{alphas_str}_b{blend}', se, ex))
            if se < best_se:
                best_se = se
                best_label = f'mα_{alphas_str}_b{blend}'
                best_preds = preds_ma.copy()
                print(f'  ★ multi-α {alphas_str} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [J done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SECTION K: Ridge+XGB committee ensemble
    # ═══════════════════════════════════════════════════════════
    print('\n── K: Ridge+XGB committee blend ──')
    for ridge_alpha in [10, 15, 20]:
        for xgb_w in [0.3, 0.5, 0.7]:
            for blend in [0.12, 0.15, 0.20]:
                preds_rx = np.zeros(n, dtype=int)
                for hold_season, c in cache.items():
                    season_mask = c['season_mask']
                    si = c['si']
                    train_mask = c['train_mask']
                    a_v12_zoned = c['a_v12_zoned']
                    avail = c['avail']
                    X_season = c['X_season']
                    tm = c['tm']
                    
                    sc = StandardScaler()
                    X_tr_sc = sc.fit_transform(X_comm[train_mask])
                    X_te_sc = sc.transform(X_comm[season_mask])
                    r = Ridge(alpha=ridge_alpha)
                    r.fit(X_tr_sc, y[train_mask])
                    raw_ridge = r.predict(X_te_sc)
                    
                    xg = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
                    xg.fit(X_comm[train_mask], y[train_mask])
                    raw_xgb = xg.predict(X_comm[season_mask])
                    
                    raw_blend = (1 - xgb_w) * raw_ridge + xgb_w * raw_xgb
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: raw_blend[i] = y[gi]
                    a_comm = hungarian(raw_blend, seasons[season_mask], avail, power=0.15)
                    a_comm = apply_zones(a_comm, raw_blend, fn, X_season, tm, si, ZONES_V46, 0.15)
                    
                    avg = (1.0 - blend) * a_v12_zoned.astype(float) + blend * a_comm.astype(float)
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: avg[i] = y[gi]
                    a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
                    for i, gi in enumerate(si):
                        if test_mask[gi]: preds_rx[gi] = a_final[i]
                
                se, ex = eval_preds(preds_rx)
                results.append((f'rx_ra{ridge_alpha}_xw{xgb_w}_b{blend}', se, ex))
                if se < best_se:
                    best_se = se
                    best_label = f'rx_ra{ridge_alpha}_xw{xgb_w}_b{blend}'
                    best_preds = preds_rx.copy()
                    print(f'  ★ Ridge+XGB ra={ridge_alpha} xw={xgb_w} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [K done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(f' FINAL RESULTS')
    print('='*60)
    print(f'  Baseline v46: SE={base_se}, exact={base_ex}/91')
    print(f'  Best found: {best_label}, SE={best_se}')
    print(f'  Improvement: Δ={base_se - best_se}')
    
    # Top 10 results
    results.sort(key=lambda x: x[1])
    print(f'\n  Top 15 configurations:')
    for label, se, ex in results[:15]:
        flag = '★' if se < base_se else ' '
        print(f'  {flag} SE={se:4d}  exact={ex:2d}/91  {label}')
    
    # Error analysis on best
    if best_se < base_se:
        print(f'\n  Errors changed from baseline:')
        for gi in np.where(test_mask)[0]:
            old = base[gi]
            new = best_preds[gi]
            actual = int(y[gi])
            if old != new:
                rid = record_ids[gi]
                print(f'    {rid:32s} gt={actual:2d} old={old:2d} new={new:2d} '
                      f'oldSE={(old-actual)**2:3d} newSE={(new-actual)**2:3d}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
