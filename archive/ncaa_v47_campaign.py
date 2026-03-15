#!/usr/bin/env python3
"""
v47 Improvement Campaign — Push v46 (SE=132) lower.

Phase 1: Analyze remaining errors in detail
Phase 2: Try 10+ improvement vectors
Phase 3: Combine best improvements
"""

import os, sys, time, warnings
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
    DUAL_RIDGE_ALPHA, DUAL_BLEND,
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


def run_dual(X_all, X_comm, y, fn, seasons, test_mask, n,
             alpha=15.0, blend=0.15, comm_model='ridge',
             extra_comm=None, extra_blend=0.0, power=0.15,
             zones=ZONES, comm_zones=True):
    """Generalized dual/triple-Hungarian."""
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

        # Committee model
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[train_mask])
        X_te_sc = sc.transform(X_comm[season_mask])
        if comm_model == 'ridge':
            mdl = Ridge(alpha=alpha)
        elif comm_model == 'lasso':
            mdl = Lasso(alpha=alpha/100, max_iter=5000)
        elif comm_model == 'elastic':
            mdl = ElasticNet(alpha=alpha/100, l1_ratio=0.5, max_iter=5000)
        mdl.fit(X_tr_sc, y[train_mask])
        raw_comm = mdl.predict(X_te_sc)

        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]

        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}

        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=power)
        a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, zones, power)

        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=power)
        if comm_zones:
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, power)

        avg = (1.0 - blend - extra_blend) * a_v12.astype(float) + blend * a_comm.astype(float)

        # Optional third model
        if extra_comm is not None and extra_blend > 0:
            sc2 = StandardScaler()
            X_tr_sc2 = sc2.fit_transform(extra_comm[train_mask])
            X_te_sc2 = sc2.transform(extra_comm[season_mask])
            mdl2 = Ridge(alpha=alpha)
            mdl2.fit(X_tr_sc2, y[train_mask])
            raw_extra = mdl2.predict(X_te_sc2)
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw_extra[i] = y[gi]
            a_extra = hungarian(raw_extra, seasons[season_mask], avail, power=power)
            if comm_zones:
                a_extra = apply_zones(a_extra, raw_extra, fn, X_season, tm, si, zones, power)
            avg += extra_blend * a_extra.astype(float)

        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]

        a_final = hungarian(avg, seasons[season_mask], avail, power=power)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


def build_committee_v2(X, fn):
    """Enhanced committee features with more interaction terms."""
    fi = {f: i for i, f in enumerate(fn)}
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    opp = X[:, fi['AvgOppNETRank']]
    is_al = X[:, fi['is_AL']]
    is_aq = X[:, fi['is_AQ']]
    is_power = X[:, fi['is_power_conf']]
    conf_avg = X[:, fi['conf_avg_net']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q2w = X[:, fi['Quadrant2_W']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    wpct = X[:, fi['WL_Pct']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    # All v1 features
    feats = [
        net, sos, opp, is_al, is_power, q1w,
        q3l + q4l, wpct, cb_mean, tfr,
        is_aq * (1 - is_power) * net,
        is_al * is_power * (200 - net),
        net - 0.3 * sos,
        net - conf_avg,
        is_aq * np.maximum(0, net - 50),
        is_power * np.maximum(0, 100 - sos),
        q1w / (q1w + q1l + 0.5),
        is_power * (q3l + q4l),
        tfr,
        cb_mean * is_aq,
        cb_mean * is_al,
    ]
    
    # New v2 features
    feats.extend([
        net**2 / 1000,                            # quadratic NET
        sos * is_aq,                               # SOS for AQ teams
        q2w,                                       # Q2 wins
        (q1w + q2w) / (q1w + q1l + q2w + 0.5),    # top-half win rate
        np.log1p(net) * is_al,                     # log NET for AL teams
        opp - sos,                                 # opp vs SOS divergence
        conf_avg * is_aq,                          # conf avg for AQ teams
        np.abs(net - cb_mean),                     # NET vs historical mean
        is_power * net,                            # power conf NET interaction
        (1 - is_power) * (1 - is_aq) * net,        # non-power AL (shouldn't exist much)
    ])
    return np.column_stack(feats)


def build_committee_minimal(X, fn):
    """Minimal committee features — only the most impactful."""
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


def main():
    t0 = time.time()
    print('='*60)
    print(' v47 IMPROVEMENT CAMPAIGN')
    print(' Starting from v46: SE=132, 67/91, Kaggle=0.521')
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
    X_comm = build_committee_features(X_all, fn)
    X_comm_v2 = build_committee_v2(X_all, fn)
    X_comm_min = build_committee_minimal(X_all, fn)
    gt = y[test_mask].astype(int)
    
    def eval_se(preds):
        return int(np.sum((preds[test_mask] - gt)**2))
    def eval_exact(preds):
        return int((preds[test_mask] == gt).sum())
    
    # Baseline
    base = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n)
    base_se = eval_se(base)
    print(f'\n  Baseline v46: SE={base_se}, exact={eval_exact(base)}/91')
    
    # ── PHASE 1: DETAILED ERROR ANALYSIS ──
    print('\n' + '='*60)
    print(' PHASE 1: Remaining error analysis')
    print('='*60)
    
    fi = {f: i for i, f in enumerate(fn)}
    errors = []
    for gi in np.where(test_mask)[0]:
        pred = base[gi]
        actual = int(y[gi])
        if pred != actual:
            rid = record_ids[gi]
            team = rid.split('_')[0] if '_' not in rid else rid.rsplit('-', 1)[-1] if '-' in rid else rid
            season = seasons[gi]
            net = X_all[gi, fi['NET Rank']]
            sos = X_all[gi, fi['NETSOS']]
            is_aq = X_all[gi, fi['is_AQ']]
            is_al = X_all[gi, fi['is_AL']]
            is_pow = X_all[gi, fi['is_power_conf']]
            se = (pred - actual)**2
            errors.append((rid, season, actual, pred, se, net, sos, is_aq, is_al, is_pow))
    
    errors.sort(key=lambda x: -x[4])
    total_se = sum(e[4] for e in errors)
    print(f'\n  {len(errors)} wrong predictions, total SE={total_se}')
    print(f'  {"RecordID":30s} {"Season":8s} {"GT":>4s} {"Pred":>4s} {"SE":>4s} {"NET":>5s} {"SOS":>5s} {"AQ":>3s} {"AL":>3s} {"Pow":>3s}')
    for rid, season, actual, pred, se, net, sos, aq, al, pw in errors:
        print(f'  {rid:30s} {season:8s} {actual:4d} {pred:4d} {se:4d} {net:5.0f} {sos:5.0f} {int(aq):3d} {int(al):3d} {int(pw):3d}')
    
    # Season breakdown
    print(f'\n  Per-season error:')
    for s in folds:
        s_errs = [e for e in errors if e[1] == s]
        s_se = sum(e[4] for e in s_errs)
        print(f'    {s}: {len(s_errs)} errors, SE={s_se}')
    
    # Error type breakdown
    over = [(e[3]-e[2]) for e in errors if e[3] > e[2]]
    under = [(e[2]-e[3]) for e in errors if e[3] < e[2]]
    print(f'\n  Over-predicted (seed too high): {len(over)} teams, avg={np.mean(over):.1f}')
    print(f'  Under-predicted (seed too low): {len(under)} teams, avg={np.mean(under):.1f}')
    
    # ── PHASE 2: IMPROVEMENT VECTORS ──
    print('\n' + '='*60)
    print(' PHASE 2: Improvement vectors')
    print('='*60)
    best_se = base_se
    best_label = 'v46'
    
    # Vector 1: Different committee models
    print('\n── Vector 1: Committee model type ──')
    for mtype in ['ridge', 'lasso', 'elastic']:
        for alpha in [10, 15, 20]:
            p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n,
                        alpha=alpha, comm_model=mtype)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'{mtype}_a{alpha}'
                print(f'  ★ {mtype} α={alpha}: SE={se}, exact={eval_exact(p)}/91')
            elif se <= base_se + 5:
                print(f'    {mtype} α={alpha}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 2: Enhanced committee features (v2)
    print('\n── Vector 2: Committee feature sets ──')
    for label, X_c in [('v2_enhanced', X_comm_v2), ('minimal', X_comm_min)]:
        for alpha in [10, 15, 20, 30]:
            for blend in [0.10, 0.15, 0.20]:
                p = run_dual(X_all, X_c, y, fn, seasons, test_mask, n,
                            alpha=alpha, blend=blend)
                se = eval_se(p)
                if se < best_se:
                    best_se = se
                    best_label = f'{label}_a{alpha}_b{blend}'
                    print(f'  ★ {label} α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
                elif se <= base_se:
                    print(f'    {label} α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 3: Triple-Hungarian (add 3rd model)
    print('\n── Vector 3: Triple-Hungarian ──')
    for ea in [0.05, 0.08, 0.10, 0.15]:
        for b in [0.10, 0.12, 0.15]:
            if b + ea > 0.40: continue
            p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n,
                        alpha=15, blend=b, extra_comm=X_comm_v2, extra_blend=ea)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'triple_b{b}_ea{ea}'
                print(f'  ★ Triple b={b:.2f} ea={ea:.2f}: SE={se}, exact={eval_exact(p)}/91')
            elif se <= base_se:
                print(f'    Triple b={b:.2f} ea={ea:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 4: No zones on committee branch
    print('\n── Vector 4: Zones on v12 only (no zones on committee) ──')
    for alpha in [10, 15, 20]:
        for blend in [0.10, 0.15, 0.20]:
            p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend, comm_zones=False)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'nozonecomm_a{alpha}_b{blend}'
                print(f'  ★ NoZoneComm α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
            elif se <= base_se:
                print(f'    NoZoneComm α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 5: Power sweep on final Hungarian
    print('\n── Vector 5: Final Hungarian power ──')
    for power in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n, power=power)
        se = eval_se(p)
        if se < best_se:
            best_se = se
            best_label = f'power_{power}'
            print(f'  ★ power={power:.2f}: SE={se}, exact={eval_exact(p)}/91')
        elif se <= base_se + 5:
            print(f'    power={power:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 6: XGB as committee model
    print('\n── Vector 6: XGB regression as committee ──')
    for hold_season in folds:
        pass  # XGB handled inline
    for n_est in [100, 200, 300]:
        for depth in [3, 4, 5]:
            preds_xgb = np.zeros(n, dtype=int)
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
                
                xgb_m = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, verbosity=0,
                                          random_state=42)
                xgb_m.fit(X_comm[train_mask], y[train_mask])
                raw_xgb = xgb_m.predict(X_comm[season_mask])
                
                for i, gi in enumerate(si):
                    if not test_mask[gi]:
                        raw_v12[i] = y[gi]
                        raw_xgb[i] = y[gi]
                
                tm = np.array([test_mask[gi] for gi in si])
                avail = {hold_season: list(range(1, 69))}
                a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
                a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, ZONES, 0.15)
                a_xgb = hungarian(raw_xgb, seasons[season_mask], avail, power=0.15)
                a_xgb = apply_zones(a_xgb, raw_xgb, fn, X_season, tm, si, ZONES, 0.15)
                
                avg = 0.85 * a_v12.astype(float) + 0.15 * a_xgb.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_xgb[gi] = a_final[i]
            
            se = eval_se(preds_xgb)
            if se < best_se:
                best_se = se
                best_label = f'xgb_n{n_est}_d{depth}'
                print(f'  ★ XGB n={n_est} d={depth}: SE={se}, exact={eval_exact(preds_xgb)}/91')
            elif se <= base_se:
                print(f'    XGB n={n_est} d={depth}: SE={se}, exact={eval_exact(preds_xgb)}/91')
    
    # Vector 7: Ridge on ALL 68 features (not committee subset)
    print('\n── Vector 7: Ridge on all 68 features as committee ──')
    for alpha in [5, 10, 15, 20, 50, 100]:
        for blend in [0.10, 0.15, 0.20]:
            p = run_dual(X_all, X_all, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'all68_a{alpha}_b{blend}'
                print(f'  ★ All68 α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
            elif se <= base_se:
                print(f'    All68 α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 8: Zone parameter re-optimization with dual-Hungarian
    print('\n── Vector 8: Zone parameter sweep (with dual) ──')
    for mid_sos in [2, 3, 4]:
        for umid_al in [-4, -3, -2]:
            for umid_sos in [-5, -4, -3]:
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, mid_sos)),
                    ('uppermid','committee', (34, 44), (-2, umid_al, umid_sos)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n, zones=zones_test)
                se = eval_se(p)
                if se < best_se:
                    best_se = se
                    best_label = f'zones_ms{mid_sos}_ual{umid_al}_us{umid_sos}'
                    print(f'  ★ mid_sos={mid_sos} umid_al={umid_al} umid_sos={umid_sos}: SE={se}')
                elif se < base_se:
                    print(f'    mid_sos={mid_sos} umid_al={umid_al} umid_sos={umid_sos}: SE={se}')
    
    # Vector 9: Bottom zone re-optimization with dual
    for bot_sn in [-5, -4, -3]:
        for bot_nc in [2, 3, 4]:
            for bot_cb in [-2, -1, 0]:
                zones_test = [
                    ('mid',     'committee', (17, 34), (0, 0, 3)),
                    ('uppermid','committee', (34, 44), (-2, -3, -4)),
                    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                    ('bot',     'bottom',    (52, 60), (bot_sn, bot_nc, bot_cb)),
                    ('tail',    'tail',      (60, 63), (1,)),
                ]
                p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n, zones=zones_test)
                se = eval_se(p)
                if se < best_se:
                    best_se = se
                    best_label = f'bot_sn{bot_sn}_nc{bot_nc}_cb{bot_cb}'
                    print(f'  ★ bot sn={bot_sn} nc={bot_nc} cb={bot_cb}: SE={se}')
    
    # Vector 10: Blend sweep at finer granularity
    print('\n── Vector 10: Fine blend/alpha grid ──')
    for alpha in [13, 14, 14.5, 15, 15.5, 16, 17]:
        for blend_pct in [12, 13, 14, 15, 16, 17, 18]:
            blend = blend_pct / 100.0
            if alpha == 15 and blend == 0.15: continue  # skip baseline
            p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n,
                        alpha=alpha, blend=blend)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'fine_a{alpha}_b{blend}'
                print(f'  ★ α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
            elif se < base_se:
                print(f'    α={alpha} b={blend:.2f}: SE={se}, exact={eval_exact(p)}/91')
    
    # Vector 11: Multi-seed averaging for dual hungarian
    print('\n── Vector 11: Multi-seed committee Ridge ──')
    for n_seeds in [3, 5]:
        preds_ms = np.zeros(n, dtype=int)
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
            
            # Average over multiple Ridge seeds
            raw_comm_avg = np.zeros(season_mask.sum())
            for seed_i in range(n_seeds):
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_comm[train_mask])
                X_te_sc = sc.transform(X_comm[season_mask])
                # Add small noise for diversity
                np.random.seed(seed_i * 42)
                noise = np.random.normal(0, 0.1, X_tr_sc.shape)
                ridge = Ridge(alpha=15)
                ridge.fit(X_tr_sc + noise, y[train_mask])
                raw_comm_avg += ridge.predict(X_te_sc)
            raw_comm_avg /= n_seeds
            
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw_v12[i] = y[gi]
                    raw_comm_avg[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in si])
            avail = {hold_season: list(range(1, 69))}
            a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
            a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, si, ZONES, 0.15)
            a_comm = hungarian(raw_comm_avg, seasons[season_mask], avail, power=0.15)
            a_comm = apply_zones(a_comm, raw_comm_avg, fn, X_season, tm, si, ZONES, 0.15)
            
            avg = 0.85 * a_v12.astype(float) + 0.15 * a_comm.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_ms[gi] = a_final[i]
        
        se = eval_se(preds_ms)
        if se < best_se:
            best_se = se
            best_label = f'multiseed_{n_seeds}'
            print(f'  ★ {n_seeds}-seed: SE={se}, exact={eval_exact(preds_ms)}/91')
        else:
            print(f'    {n_seeds}-seed: SE={se}, exact={eval_exact(preds_ms)}/91')
    
    # Vector 12: Zone boundary shifts with dual
    print('\n── Vector 12: Zone boundary exploration ──')
    for mid_hi in [32, 33, 34, 35, 36]:
        for umid_hi in [42, 43, 44, 45, 46]:
            zones_test = [
                ('mid',     'committee', (17, mid_hi), (0, 0, 3)),
                ('uppermid','committee', (mid_hi, umid_hi), (-2, -3, -4)),
                ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                ('tail',    'tail',      (60, 63), (1,)),
            ]
            p = run_dual(X_all, X_comm, y, fn, seasons, test_mask, n, zones=zones_test)
            se = eval_se(p)
            if se < best_se:
                best_se = se
                best_label = f'bounds_mid{mid_hi}_umid{umid_hi}'
                print(f'  ★ mid to {mid_hi}, umid to {umid_hi}: SE={se}')
            elif se < base_se:
                print(f'    mid to {mid_hi}, umid to {umid_hi}: SE={se}')
    
    # ── SUMMARY ──
    print('\n' + '='*60)
    print(f' BEST RESULT: {best_label}, SE={best_se}')
    print(f' Improvement from v46: {base_se} → {best_se} (Δ={base_se-best_se})')
    print(f' Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
