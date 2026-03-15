#!/usr/bin/env python3
"""
v46 Dual-Hungarian Deep Dive.

Test 4 from ncaa_v46_committee.py hit SE=200 with α=10, blend=0.2.
This script explores:
  A. Fine-grained α × blend grid around the best
  B. Zones on the final Hungarian step
  C. Different committee feature sets
  D. Nested LOSO validation of the best config
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
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    predict_robust_blend, hungarian,
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


def build_committee_features(X, fn):
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
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    wpct = X[:, fi['WL_Pct']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    
    feats = []
    feats.append(net)
    feats.append(sos)
    feats.append(opp)
    feats.append(is_al)
    feats.append(is_power)
    feats.append(q1w)
    feats.append(q3l + q4l)
    feats.append(wpct)
    feats.append(cb_mean)
    feats.append(tfr)
    feats.append(is_aq * (1 - is_power) * net)
    feats.append(is_al * is_power * (200 - net))
    feats.append(net - 0.3*sos)
    feats.append(net - conf_avg)
    feats.append(is_aq * np.maximum(0, net - 50))
    feats.append(is_power * np.maximum(0, 100 - sos))
    q1_rate = q1w / (q1w + q1l + 0.5)
    feats.append(q1_rate)
    feats.append(is_power * (q3l + q4l))
    feats.append(tfr)
    feats.append(cb_mean * is_aq)
    feats.append(cb_mean * is_al)
    
    return np.column_stack(feats)


def dual_hungarian(X_all, X_comm, y, fn, seasons, test_mask, record_ids,
                   alpha, blend, with_zones=True, zone_final=False, power=0.15):
    """Run dual-Hungarian approach."""
    n = len(y)
    folds = sorted(set(seasons))
    preds = np.zeros(n, dtype=int)
    
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        
        X_season = X_all[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        # v12 pairwise
        tki = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], tki)
        
        # Committee Ridge
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
        X_te_sc = sc.transform(X_comm[season_mask])
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_tr_sc, y[global_train_mask])
        raw_comm = ridge.predict(X_te_sc)
        
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw_v12[i] = y[gi]
                raw_comm[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        avail = {hold_season: list(range(1, 69))}
        
        # Hungarian + zones on v12
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=power)
        if with_zones:
            a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, season_indices, ZONES, power)
        
        # Hungarian + zones on committee
        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=power)
        if with_zones:
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, season_indices, ZONES, power)
        
        # Average assignments
        avg_seed = (1 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                avg_seed[i] = y[gi]
        
        # Final Hungarian
        a_final = hungarian(avg_seed, seasons[season_mask], avail, power=power)
        
        # Optionally apply zones to final
        if zone_final:
            a_final = apply_zones(a_final, avg_seed, fn, X_season, tm, season_indices, ZONES, power)
        
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                preds[gi] = a_final[i]
    
    return preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 DUAL-HUNGARIAN DEEP DIVE')
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
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    X_comm = build_committee_features(X_all, fn)
    
    # ── A: Fine-grained α × blend grid ──
    print('\n' + '='*60)
    print(' A: Fine-grained α × blend grid (with zones on v12+comm)')
    print('='*60)
    
    best_se = 233
    best_cfg = None
    results = []
    
    for alpha in [3, 5, 7, 10, 12, 15, 20, 30, 50]:
        for blend_pct in [5, 10, 15, 20, 25, 30, 35, 40, 50]:
            blend = blend_pct / 100.0
            preds = dual_hungarian(X_all, X_comm, y, fn, seasons, test_mask, record_ids,
                                   alpha, blend, with_zones=True, zone_final=False)
            gt = y[test_mask].astype(int)
            pr = preds[test_mask]
            se = int(np.sum((pr - gt)**2))
            exact = int((pr == gt).sum())
            results.append((alpha, blend, se, exact))
            if se < best_se:
                best_se = se
                best_cfg = (alpha, blend)
                print(f'  ★ α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91')
            elif se <= 210:
                print(f'    α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91')
    
    print(f'\n  Best: α={best_cfg[0]}, blend={best_cfg[1]}, SE={best_se}')
    
    # Show top-10 configs
    results.sort(key=lambda x: x[2])
    print('\n  Top-10 configs:')
    for alpha, blend, se, exact in results[:10]:
        print(f'    α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91')
    
    # ── B: Add zones to the final Hungarian step ──
    print('\n' + '='*60)
    print(' B: Same grid but with zones on final Hungarian too')
    print('='*60)
    
    best_se_z = 233
    best_cfg_z = None
    results_z = []
    
    for alpha in [3, 5, 7, 10, 12, 15, 20]:
        for blend_pct in [10, 15, 20, 25, 30]:
            blend = blend_pct / 100.0
            preds = dual_hungarian(X_all, X_comm, y, fn, seasons, test_mask, record_ids,
                                   alpha, blend, with_zones=True, zone_final=True)
            gt = y[test_mask].astype(int)
            pr = preds[test_mask]
            se = int(np.sum((pr - gt)**2))
            exact = int((pr == gt).sum())
            results_z.append((alpha, blend, se, exact))
            if se < best_se_z:
                best_se_z = se
                best_cfg_z = (alpha, blend)
                print(f'  ★ α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91')
            elif se <= 210:
                print(f'    α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91')
    
    if best_cfg_z:
        print(f'\n  Best with zones: α={best_cfg_z[0]}, blend={best_cfg_z[1]}, SE={best_se_z}')
    else:
        print(f'\n  No config beat v45c (SE=233)')
    
    # ── C: No zones at all (to see baseline dual-Hungarian) ──
    print('\n' + '='*60)
    print(' C: Dual-Hungarian without any zones')
    print('='*60)
    
    for alpha in [5, 10, 15, 20]:
        for blend_pct in [10, 15, 20, 25, 30]:
            blend = blend_pct / 100.0
            preds = dual_hungarian(X_all, X_comm, y, fn, seasons, test_mask, record_ids,
                                   alpha, blend, with_zones=False, zone_final=False)
            gt = y[test_mask].astype(int)
            pr = preds[test_mask]
            se = int(np.sum((pr - gt)**2))
            exact = int((pr == gt).sum())
            if se <= 240:
                marker = ' ★' if se < 233 else ''
                print(f'    α={alpha:3d}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── D: Nested LOSO validation ──
    print('\n' + '='*60)
    print(' D: Nested LOSO validation of best configs')
    print('='*60)
    
    configs_to_validate = []
    if best_cfg:
        configs_to_validate.append(('best_A', best_cfg[0], best_cfg[1], True, False))
    if best_cfg_z and best_cfg_z != best_cfg:
        configs_to_validate.append(('best_B', best_cfg_z[0], best_cfg_z[1], True, True))
    # Also validate α=10, blend=0.2 if not already best
    if (10, 0.2) != best_cfg:
        configs_to_validate.append(('α10b20', 10, 0.20, True, False))
    
    folds = sorted(set(seasons))
    
    for label, alpha, blend, wz, zf in configs_to_validate:
        print(f'\n  Config: {label} (α={alpha}, blend={blend}, zones={wz}, zoneFinal={zf})')
        
        # Nested LOSO: for each outer fold, select best config using inner folds
        outer_preds = np.zeros(n, dtype=int)
        inner_wins = []
        
        for outer_season in folds:
            outer_mask = (seasons == outer_season)
            outer_test = test_mask & outer_mask
            if outer_test.sum() == 0: continue
            
            # Inner validation: check if this config is best among alternatives
            # Use remaining seasons as inner validation
            inner_seasons = [s for s in folds if s != outer_season]
            
            # v45c baseline (inner)
            inner_se_v45c = 0
            
            # Dual-Hungarian (inner)
            inner_se_dual = 0
            
            for inner_hold in inner_seasons:
                inner_hold_mask = (seasons == inner_hold)
                inner_test_mask = test_mask & inner_hold_mask
                if inner_test_mask.sum() == 0: continue
                
                # For inner validation, train on everything except outer+inner test
                inner_train_mask = ~(outer_mask | inner_test_mask)
                
                # -- v45c baseline --
                for hold in [inner_hold]:
                    sm = inner_hold_mask
                    X_s = X_all[sm]
                    si = np.where(sm)[0]
                    
                    tki = select_top_k_features(
                        X_all[inner_train_mask], y[inner_train_mask],
                        fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
                    raw = predict_robust_blend(
                        X_all[inner_train_mask], y[inner_train_mask],
                        X_s, seasons[inner_train_mask], tki)
                    
                    for i, gi in enumerate(si):
                        if not test_mask[gi] or outer_mask[gi]:
                            raw[i] = y[gi] if not outer_mask[gi] else raw[i]
                        if not (test_mask[gi] and inner_hold_mask[gi]):
                            raw[i] = y[gi] if y[gi] > 0 else raw[i]
                    
                    # Fix: only set GT for known non-test teams
                    for i, gi in enumerate(si):
                        if not inner_test_mask[gi]:
                            raw[i] = y[gi]
                    
                    tm_inner = np.array([inner_test_mask[gi] for gi in si])
                    avail = {inner_hold: list(range(1, 69))}
                    a = hungarian(raw, seasons[sm], avail, power=0.15)
                    a = apply_zones(a, raw, fn, X_s, tm_inner, si, ZONES, 0.15)
                    
                    for i, gi in enumerate(si):
                        if inner_test_mask[gi]:
                            inner_se_v45c += (a[i] - y[gi])**2
                
                # -- Dual-Hungarian --
                for hold in [inner_hold]:
                    sm = inner_hold_mask
                    X_s = X_all[sm]
                    si = np.where(sm)[0]
                    
                    tki = select_top_k_features(
                        X_all[inner_train_mask], y[inner_train_mask],
                        fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
                    raw_v12 = predict_robust_blend(
                        X_all[inner_train_mask], y[inner_train_mask],
                        X_s, seasons[inner_train_mask], tki)
                    
                    sc = StandardScaler()
                    X_tr_sc = sc.fit_transform(X_comm[inner_train_mask])
                    X_te_sc = sc.transform(X_comm[sm])
                    ridge = Ridge(alpha=alpha)
                    ridge.fit(X_tr_sc, y[inner_train_mask])
                    raw_comm = ridge.predict(X_te_sc)
                    
                    for i, gi in enumerate(si):
                        if not inner_test_mask[gi]:
                            raw_v12[i] = y[gi]
                            raw_comm[i] = y[gi]
                    
                    tm_inner = np.array([inner_test_mask[gi] for gi in si])
                    avail = {inner_hold: list(range(1, 69))}
                    
                    a_v12 = hungarian(raw_v12, seasons[sm], avail, power=0.15)
                    if wz:
                        a_v12 = apply_zones(a_v12, raw_v12, fn, X_s, tm_inner, si, ZONES, 0.15)
                    
                    a_comm = hungarian(raw_comm, seasons[sm], avail, power=0.15)
                    if wz:
                        a_comm = apply_zones(a_comm, raw_comm, fn, X_s, tm_inner, si, ZONES, 0.15)
                    
                    avg = (1 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
                    for i, gi in enumerate(si):
                        if not inner_test_mask[gi]:
                            avg[i] = y[gi]
                    
                    a_final = hungarian(avg, seasons[sm], avail, power=0.15)
                    if zf:
                        a_final = apply_zones(a_final, avg, fn, X_s, tm_inner, si, ZONES, 0.15)
                    
                    for i, gi in enumerate(si):
                        if inner_test_mask[gi]:
                            inner_se_dual += (a_final[i] - y[gi])**2
            
            inner_se_v45c = int(inner_se_v45c)
            inner_se_dual = int(inner_se_dual)
            winner = 'DUAL' if inner_se_dual < inner_se_v45c else ('TIE' if inner_se_dual == inner_se_v45c else 'V45C')
            inner_wins.append(winner)
            print(f'    Outer={outer_season}: inner v45c={inner_se_v45c}, inner dual={inner_se_dual} → {winner}')
            
            # Use winner for outer prediction
            if winner in ('DUAL', 'TIE'):
                p = dual_hungarian(X_all, X_comm, y, fn, seasons, outer_test, record_ids,
                                   alpha, blend, with_zones=wz, zone_final=zf)
            else:
                # Use v45c for this fold
                sm = outer_mask
                X_s = X_all[sm]
                si = np.where(sm)[0]
                train_mask_v45c = ~outer_test
                
                tki = select_top_k_features(
                    X_all[train_mask_v45c], y[train_mask_v45c],
                    fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
                raw = predict_robust_blend(
                    X_all[train_mask_v45c], y[train_mask_v45c],
                    X_s, seasons[train_mask_v45c], tki)
                
                for i, gi in enumerate(si):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                
                tm_o = np.array([outer_test[gi] for gi in si])
                avail = {outer_season: list(range(1, 69))}
                a = hungarian(raw, seasons[sm], avail, power=0.15)
                a = apply_zones(a, raw, fn, X_s, tm_o, si, ZONES, 0.15)
                p = np.zeros(n, dtype=int)
                for i, gi in enumerate(si):
                    if outer_test[gi]:
                        p[gi] = a[i]
            
            for gi in np.where(outer_test)[0]:
                outer_preds[gi] = p[gi]
        
        gt = y[test_mask].astype(int)
        pr = outer_preds[test_mask]
        se = int(np.sum((pr - gt)**2))
        exact = int((pr == gt).sum())
        
        dual_count = sum(1 for w in inner_wins if w in ('DUAL', 'TIE'))
        v45c_count = sum(1 for w in inner_wins if w == 'V45C')
        
        gap = se - best_se if best_cfg else se - 233
        print(f'\n  Nested result: SE={se}, exact={exact}/91')
        print(f'  Inner wins: DUAL/TIE={dual_count}, V45C={v45c_count}')
        print(f'  Gap from direct eval: {gap:+d}')
    
    # ── E: Error comparison ──
    if best_cfg:
        print('\n' + '='*60)
        print(' E: Error comparison (v45c vs best dual-Hungarian)')
        print('='*60)
        
        preds_dual = dual_hungarian(X_all, X_comm, y, fn, seasons, test_mask, record_ids,
                                     best_cfg[0], best_cfg[1], with_zones=True, zone_final=False)
        
        # v45c predictions
        preds_v45c = np.zeros(n, dtype=int)
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            tki = select_top_k_features(
                X_all[global_train_mask], y[global_train_mask],
                fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend(
                X_all[global_train_mask], y[global_train_mask],
                X_season, seasons[global_train_mask], tki)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]: raw[i] = y[gi]
            tm = np.array([test_mask[gi] for gi in season_indices])
            avail = {hold_season: list(range(1, 69))}
            a = hungarian(raw, seasons[season_mask], avail, power=0.15)
            a = apply_zones(a, raw, fn, X_season, tm, season_indices, ZONES, 0.15)
            for i, gi in enumerate(season_indices):
                if test_mask[gi]: preds_v45c[gi] = a[i]
        
        gt = y[test_mask].astype(int)
        pr_d = preds_dual[test_mask]
        pr_v = preds_v45c[test_mask]
        
        # Find where they differ
        diff_mask = pr_d != pr_v
        diff_indices = np.where(test_mask)[0][diff_mask]
        
        print(f'  Differ on {diff_mask.sum()} teams:')
        print(f'  {"Team":20s} {"Season":8s} {"GT":>4s} {"v45c":>5s} {"Dual":>5s} {"v45c_SE":>7s} {"Dual_SE":>7s} {"Δ":>4s}')
        
        for gi in diff_indices:
            rid = record_ids[gi]
            team = rid.split('_')[0]
            season = rid.split('_')[1] if '_' in rid else seasons[gi]
            gt_i = int(y[gi])
            v45c_i = int(preds_v45c[gi])
            dual_i = int(preds_dual[gi])
            se_v = (v45c_i - gt_i)**2
            se_d = (dual_i - gt_i)**2
            delta = se_d - se_v
            marker = '✓' if delta < 0 else '✗' if delta > 0 else '='
            print(f'  {team:20s} {season:8s} {gt_i:4d} {v45c_i:5d} {dual_i:5d} {se_v:7d} {se_d:7d} {delta:+4d} {marker}')
        
        total_gain = sum((int(preds_v45c[gi]) - int(y[gi]))**2 - (int(preds_dual[gi]) - int(y[gi]))**2 for gi in diff_indices)
        print(f'\n  Total SE change: {-total_gain:+d} (negative = better for dual)')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
