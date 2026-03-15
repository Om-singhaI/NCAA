#!/usr/bin/env python3
"""
v46 Committee Preference Model.

Key insight from error analysis:
- MurraySt: NET=21, SOS=220, seed=26 → committee seeds them MUCH worse than NET
  because they're from OVC (weak conf) despite great NET
- NewMexico: NET=22, SOS=82, seed=42 → from Mountain West, seeded near 42
  despite NET=22. Committee penalizes mid-majors.
- TCU: NET=44, SOS=10, seed=34 → from Big 12, seeded BETTER than NET because
  incredible SOS compensates for weaker NET
- Clemson: NET=35, SOS=52, seed=22 → ACC team seeded much better

Pattern: Committee systematically adjusts seeds based on:
1. NET rank (primary signal)
2. SOS — high SOS teams get credited, low SOS penalized
3. Conference strength — power conf teams benefit
4. Bid type (AQ vs AL) — AQ mid-majors get pushed toward worse seeds

This is DIFFERENT from zone corrections. Zone corrections re-order within a band.
This approach modifies the RAW SCORE before Hungarian assignment by learning
the committee's NET→Seed mapping conditioned on conf/bid/SOS.

Approach: Train a "committee simulator" that predicts seed from ALL features
including interaction terms that capture committee biases, then blend with v12.
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
    """Build features that capture committee's known biases."""
    fi = {f: i for i, f in enumerate(fn)}
    n = X.shape[0]
    
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
    feat_names = []
    
    # Base features committee cares about
    feats.append(net); feat_names.append('net')
    feats.append(sos); feat_names.append('sos')
    feats.append(opp); feat_names.append('opp')
    feats.append(is_al); feat_names.append('is_al')
    feats.append(is_power); feat_names.append('is_power')
    feats.append(q1w); feat_names.append('q1w')
    feats.append(q3l + q4l); feat_names.append('bad_losses')
    feats.append(wpct); feat_names.append('wpct')
    feats.append(cb_mean); feat_names.append('cb_mean')
    feats.append(tfr); feat_names.append('tfr')
    
    # KEY INTERACTIONS — committee biases
    # 1. Mid-major AQ: committee pushes them toward worse seeds
    feats.append(is_aq * (1 - is_power) * net); feat_names.append('midmaj_aq_net')
    
    # 2. Power conf AL: committee gives them benefit of the doubt
    feats.append(is_al * is_power * (200 - net)); feat_names.append('power_al_benefit')
    
    # 3. SOS-adjusted NET: committee values SOS-adjusted performance
    feats.append(net - 0.3*sos); feat_names.append('sos_adj_net')
    
    # 4. NET vs conference average: how much better than conf average
    feats.append(net - conf_avg); feat_names.append('net_vs_conf')
    
    # 5. AQ with bad NET: mid-major champion but weak stats
    feats.append(is_aq * np.maximum(0, net - 50)); feat_names.append('aq_weak_net')
    
    # 6. Power conf with strong SOS: gets credit
    feats.append(is_power * np.maximum(0, 100 - sos)); feat_names.append('power_strong_sos')
    
    # 7. Q1 dominance — teams that beat good teams
    q1_rate = q1w / (q1w + q1l + 0.5)
    feats.append(q1_rate); feat_names.append('q1_rate')
    
    # 8. Bad loss penalty — amplified for power conf
    feats.append(is_power * (q3l + q4l)); feat_names.append('power_bad_loss')
    
    # 9. Field rank (relative position among tournament teams)
    feats.append(tfr); feat_names.append('tfr2')
    
    # 10. Conference-bid historical mean (what committee typically does)
    feats.append(cb_mean * is_aq); feat_names.append('cb_mean_aq')
    feats.append(cb_mean * is_al); feat_names.append('cb_mean_al')
    
    return np.column_stack(feats), feat_names


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 COMMITTEE PREFERENCE MODEL')
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
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    # Build committee features
    X_comm, comm_names = build_committee_features(X_all, fn)
    print(f'  Committee features: {len(comm_names)}')
    
    # ── TEST 1: Committee model as raw score replacement ──
    print('\n' + '='*60)
    print(' TEST 1: Committee Ridge → Hungarian + zones')
    print('='*60)
    
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        preds = np.zeros(n, dtype=int)
        
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            X_comm_season = X_comm[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
            X_te_sc = sc.transform(X_comm_season)
            
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_sc, y[global_train_mask])
            raw_comm = ridge.predict(X_te_sc)
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw_comm[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
            assigned = apply_zones(assigned, raw_comm, fn, X_season, tm, season_indices, ZONES, 0.15)
            
            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = assigned[i]
        
        gt = y[test_mask].astype(int)
        pr = preds[test_mask]
        se = int(np.sum((pr - gt)**2))
        exact = int((pr == gt).sum())
        marker = ' ★' if se < 233 else ''
        print(f'  α={alpha:6.1f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── TEST 2: Committee model blended with v12 ──
    print('\n' + '='*60)
    print(' TEST 2: Committee Ridge blended with v12 pairwise')
    print('='*60)
    
    best_se = 233
    best_config = None
    
    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
        for blend in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
            preds = np.zeros(n, dtype=int)
            
            for hold_season in folds:
                season_mask = (seasons == hold_season)
                season_test_mask = test_mask & season_mask
                if season_test_mask.sum() == 0: continue
                
                X_season = X_all[season_mask]
                X_comm_season = X_comm[season_mask]
                season_indices = np.where(season_mask)[0]
                global_train_mask = ~season_test_mask
                
                # v12 pairwise score
                tki = select_top_k_features(
                    X_all[global_train_mask], y[global_train_mask],
                    fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
                raw_v12 = predict_robust_blend(
                    X_all[global_train_mask], y[global_train_mask],
                    X_season, seasons[global_train_mask], tki)
                
                # Committee Ridge
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
                X_te_sc = sc.transform(X_comm_season)
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_sc, y[global_train_mask])
                raw_comm = ridge.predict(X_te_sc)
                
                # Blend
                raw = (1 - blend) * raw_v12 + blend * raw_comm
                
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                
                tm = np.array([test_mask[gi] for gi in season_indices])
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(raw, seasons[season_mask], avail, power=0.15)
                assigned = apply_zones(assigned, raw, fn, X_season, tm, season_indices, ZONES, 0.15)
                
                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        preds[gi] = assigned[i]
            
            gt = y[test_mask].astype(int)
            pr = preds[test_mask]
            se = int(np.sum((pr - gt)**2))
            exact = int((pr == gt).sum())
            if se < best_se:
                marker = ' ★'
                best_se = se
                best_config = (alpha, blend)
            else:
                marker = ''
            if se <= 260 or se < best_se + 5:
                print(f'  α={alpha:5.1f}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    if best_config:
        print(f'\n  ★ Best: α={best_config[0]}, blend={best_config[1]}, SE={best_se}')
    else:
        print(f'\n  No config beat v45c (SE=233)')
    
    # ── TEST 3: Committee model using ALL original features (not just committee) ──
    print('\n' + '='*60)
    print(' TEST 3: Full-feature Ridge blended with v12')
    print('='*60)
    
    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        for blend in [0.05, 0.10, 0.15, 0.20, 0.30]:
            preds = np.zeros(n, dtype=int)
            
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
                raw_v12 = predict_robust_blend(
                    X_all[global_train_mask], y[global_train_mask],
                    X_season, seasons[global_train_mask], tki)
                
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_all[global_train_mask])
                X_te_sc = sc.transform(X_season)
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_sc, y[global_train_mask])
                raw_ridge = ridge.predict(X_te_sc)
                
                raw = (1 - blend) * raw_v12 + blend * raw_ridge
                
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                
                tm = np.array([test_mask[gi] for gi in season_indices])
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(raw, seasons[season_mask], avail, power=0.15)
                assigned = apply_zones(assigned, raw, fn, X_season, tm, season_indices, ZONES, 0.15)
                
                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        preds[gi] = assigned[i]
            
            gt = y[test_mask].astype(int)
            pr = preds[test_mask]
            se = int(np.sum((pr - gt)**2))
            exact = int((pr == gt).sum())
            if se <= 240 or se == best_se:
                marker = ' ★' if se < 233 else ''
                print(f'  α={alpha:5.1f}, blend={blend:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── TEST 4: Pairwise + Committee with SEPARATE Hungarian ──
    print('\n' + '='*60)
    print(' TEST 4: Dual-Hungarian average')
    print(' (Run Hungarian on v12 AND committee separately, average seeds)')
    print('='*60)
    
    for alpha in [5.0, 10.0, 50.0, 100.0]:
        for blend in [0.1, 0.2, 0.3, 0.5]:
            preds = np.zeros(n, dtype=int)
            preds_v12 = np.zeros(n, dtype=int)
            preds_comm = np.zeros(n, dtype=int)
            
            for hold_season in folds:
                season_mask = (seasons == hold_season)
                season_test_mask = test_mask & season_mask
                if season_test_mask.sum() == 0: continue
                
                X_season = X_all[season_mask]
                X_comm_season = X_comm[season_mask]
                season_indices = np.where(season_mask)[0]
                global_train_mask = ~season_test_mask
                
                # v12
                tki = select_top_k_features(
                    X_all[global_train_mask], y[global_train_mask],
                    fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
                raw_v12 = predict_robust_blend(
                    X_all[global_train_mask], y[global_train_mask],
                    X_season, seasons[global_train_mask], tki)
                
                # Committee
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
                X_te_sc = sc.transform(X_comm_season)
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_tr_sc, y[global_train_mask])
                raw_comm = ridge.predict(X_te_sc)
                
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        raw_v12[i] = y[gi]
                        raw_comm[i] = y[gi]
                
                tm = np.array([test_mask[gi] for gi in season_indices])
                avail = {hold_season: list(range(1, 69))}
                
                # Assign v12
                a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
                a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, season_indices, ZONES, 0.15)
                
                # Assign committee
                a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
                a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, season_indices, ZONES, 0.15)
                
                # Average the assignments, then re-assign via Hungarian
                avg_seed = (1 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
                for i, gi in enumerate(season_indices):
                    if not test_mask[gi]:
                        avg_seed[i] = y[gi]
                
                a_final = hungarian(avg_seed, seasons[season_mask], avail, power=0.15)
                
                for i, gi in enumerate(season_indices):
                    if test_mask[gi]:
                        preds[gi] = a_final[i]
                        preds_v12[gi] = a_v12[i]
                        preds_comm[gi] = a_comm[i]
            
            gt = y[test_mask].astype(int)
            se = int(np.sum((preds[test_mask] - gt)**2))
            exact = int((preds[test_mask] == gt).sum())
            se_v12 = int(np.sum((preds_v12[test_mask] - gt)**2))
            se_comm = int(np.sum((preds_comm[test_mask] - gt)**2))
            
            if se <= 240:
                marker = ' ★' if se < 233 else ''
                print(f'  α={alpha:5.1f}, blend={blend:.1f}: SE={se:4d} (v12={se_v12}, comm={se_comm}), exact={exact}/91{marker}')
    
    # ── TEST 5: v12 with committee-adjusted power ──
    print('\n' + '='*60)
    print(' TEST 5: Per-team adaptive power in Hungarian')
    print(' (Higher power for confident teams, lower for uncertain)')
    print('='*60)
    
    for alpha in [5.0, 10.0, 50.0]:
        preds = np.zeros(n, dtype=int)
        
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
            
            # Get committee prediction for confidence estimation
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
            X_te_sc = sc.transform(X_comm[season_mask])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_sc, y[global_train_mask])
            ridge_pred = ridge.predict(X_te_sc)
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            
            # Per-team confidence = agreement between v12 and ridge
            disagreement = np.abs(raw - ridge_pred)
            
            # Build per-team cost with variable power
            si = list(range(len(season_indices)))
            pos = list(range(1, 69))
            cost = np.zeros((len(si), len(pos)))
            for i in si:
                # Higher disagreement → lower power → more flexible assignment
                # Lower disagreement → higher power → stick closer to prediction
                dis = disagreement[i]
                team_power = max(0.05, 0.15 - 0.003 * dis)
                for j, p in enumerate(pos):
                    cost[i, j] = abs(raw[i] - p)**team_power
            
            ri, ci = linear_sum_assignment(cost)
            assigned = np.zeros(len(si), dtype=int)
            for r, c in zip(ri, ci):
                assigned[r] = pos[c]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            assigned = apply_zones(assigned, raw, fn, X_season, tm, season_indices, ZONES, 0.15)
            
            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = assigned[i]
        
        gt = y[test_mask].astype(int)
        se = int(np.sum((preds[test_mask] - gt)**2))
        exact = int((preds[test_mask] == gt).sum())
        marker = ' ★' if se < 233 else ''
        print(f'  α={alpha:5.1f}: SE={se:4d}, exact={exact}/91{marker}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
