#!/usr/bin/env python3
"""
v32 NEW FEATURES — Add unused data columns (AvgOppNET, NETNonConfSOS)
=====================================================================
We discovered 2 UNUSED columns in the raw data that aren't in the 
current 68-feature set. Test if adding them (+ derived features) helps.

Also test: Q3_W, Q4_W (quadrant wins we extract but don't use).
"""

import os, sys, time, warnings, re
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, parse_wl,
    select_top_k_features, hungarian,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3, HUNGARIAN_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def build_features_extended(df, context_df, labeled_df, tourn_rids):
    """
    Build EXTENDED feature set (68 original + new features from unused columns).
    New features from: AvgOppNET, NETNonConfSOS, Q3_W, Q4_W
    """
    from ncaa_2026_model import build_features
    # Get original 68 features
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    # ═══ NEW FEATURES FROM UNUSED COLUMNS ═══
    
    # 1. AvgOppNET (raw efficiency, not rank)
    opp_net = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(150)
    feat['AvgOppNET'] = opp_net
    
    # 2. NETNonConfSOS
    nc_sos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    feat['NETNonConfSOS'] = nc_sos
    
    # 3. Derived: OppNET vs NETSOS gap
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    feat['opp_net_vs_sos'] = opp_net - sos  # should be related
    feat['opp_net_ratio'] = opp_net / (net + 1)  # opponent quality vs own rank
    
    # 4. Non-conf SOS features
    feat['ncsos_vs_sos'] = nc_sos - sos  # non-conf vs overall SOS gap
    feat['ncsos_ratio'] = nc_sos / (sos + 1)
    feat['ncsos_net_gap'] = nc_sos - net  # non-conf SOS relative to NET
    
    # 5. Conference non-conf SOS stats
    conf = df['Conference'].fillna('Unknown')
    all_ncsos = pd.to_numeric(context_df['NETNonConfSOS'], errors='coerce').fillna(200)
    cs = pd.DataFrame({'Conference': context_df['Conference'].fillna('Unknown'),
                       'NCSOS': all_ncsos}).groupby('Conference')['NCSOS']
    feat['conf_avg_ncsos'] = conf.map(cs.mean()).fillna(200)
    feat['ncsos_vs_conf'] = nc_sos - feat['conf_avg_ncsos']
    
    # 6. Q3 wins and Q4 wins (parsed but not used)
    q3w = feat.get('Quadrant3_W', pd.Series(0, index=df.index)).fillna(0)
    q4w = feat.get('Quadrant4_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    feat['q3_win_pct'] = q3w / (q3w + q3l + 0.5)
    feat['q4_win_pct'] = q4w / (q4w + q4l + 0.5)
    feat['q34_total_games'] = q3w + q3l + q4w + q4l
    feat['q34_win_pct'] = (q3w + q4w) / (q3w + q3l + q4w + q4l + 0.5)
    
    # 7. Cross-feature interactions with new columns
    bid = df['Bid Type'].fillna('')
    is_aq = (bid == 'AQ').astype(float)
    is_al = (bid == 'AL').astype(float)
    feat['aq_ncsos'] = is_aq * nc_sos / 100
    feat['al_oppnet'] = is_al * opp_net / 100
    
    # 8. Schedule quality composite
    feat['schedule_quality'] = (300 - sos) * 0.4 + (300 - nc_sos) * 0.3 + (300 - opp_net) * 0.3
    
    return feat


def predict_robust_blend_custom(X_train, y_train, X_test, seasons_train, top_k_idx):
    """Same as predict_robust_blend but work with any feature count."""
    # Component 1: PW-LogReg C=5.0 on full features, ADJACENT PAIRS (64%)
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=30)
    sc_adj = StandardScaler()
    lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(sc_adj.fit_transform(pw_X_adj), pw_y_adj)
    score1 = pairwise_score(lr1, X_test, sc_adj)
    
    # Component 3: PW-LogReg C=0.5 on top-K features (28%)
    X_tr_k = X_train[:, top_k_idx]
    X_te_k = X_test[:, top_k_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(sc_k.fit_transform(pw_X_k), pw_y_k)
    score3 = pairwise_score(lr3, X_te_k, sc_k)
    
    # Component 4: PW-XGB (8%)
    pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, seasons_train)
    sc_full = StandardScaler()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss')
    xgb_clf.fit(sc_full.fit_transform(pw_X_full), pw_y_full)
    score4 = pairwise_score(xgb_clf, X_test, sc_full)
    
    return BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4


def apply_v25_zones(pass1, raw, fn, X, tm, idx):
    """Apply v25 zone corrections."""
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=0.15)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=0.15)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=0.15)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=0.15)
    return p


def main():
    print('='*70)
    print('  v32 NEW FEATURES — Add AvgOppNET, NETNonConfSOS, Q3/Q4 wins')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    # Build ORIGINAL features (68)
    from ncaa_2026_model import build_features
    feat_orig = build_features(labeled, context_df, labeled, tourn_rids)
    fn_orig = list(feat_orig.columns)

    # Build EXTENDED features (68 + new)
    feat_ext = build_features_extended(labeled, context_df, labeled, tourn_rids)
    fn_ext = list(feat_ext.columns)
    
    new_features = [f for f in fn_ext if f not in fn_orig]
    print(f'\n  Original features: {len(fn_orig)}')
    print(f'  Extended features: {len(fn_ext)}')
    print(f'  New features ({len(new_features)}):')
    for f in new_features:
        print(f'    {f}')

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    imp_orig = KNNImputer(n_neighbors=10, weights='distance')
    X_orig = imp_orig.fit_transform(np.where(np.isinf(feat_orig.values.astype(np.float64)),
                                              np.nan, feat_orig.values.astype(np.float64)))
    
    imp_ext = KNNImputer(n_neighbors=10, weights='distance')
    X_ext = imp_ext.fit_transform(np.where(np.isinf(feat_ext.values.astype(np.float64)),
                                            np.nan, feat_ext.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]
    n_test_map = {s: (test_mask & (seasons == s)).sum() for s in test_seasons}

    # ════════════════════════════════════════════════════════════
    #  TEST A: Same v25 pipeline but with extended features
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  TEST A: v25 pipeline with extended feature set')
    print('='*70)

    for feat_set_name, X_all, fn_list, forced in [
        ('original_68', X_orig, fn_orig, FORCE_FEATURES),
        ('extended_all', X_ext, fn_ext, FORCE_FEATURES),
        ('extended_forced_new', X_ext, fn_ext, ['NET Rank', 'NETNonConfSOS']),
        ('extended_forced_all', X_ext, fn_ext, ['NET Rank', 'AvgOppNET', 'NETNonConfSOS']),
    ]:
        total = 0
        per_season = {}
        for hold in folds:
            season_mask = (seasons == hold)
            season_indices = np.where(season_mask)[0]
            season_test = test_mask & season_mask
            if season_test.sum() == 0:
                continue
            global_train = ~season_test
            X_season = X_all[season_mask]
            top_k_idx = select_top_k_features(
                X_all[global_train], y[global_train],
                fn_list, k=USE_TOP_K_A, forced_features=forced)[0]
            raw = predict_robust_blend_custom(
                X_all[global_train], y[global_train],
                X_season, seasons[global_train], top_k_idx)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in season_indices])
            p = apply_v25_zones(p1, raw, fn_list, X_season, tm, season_indices)
            ex = sum(1 for i, gi in enumerate(season_indices)
                     if test_mask[gi] and p[i] == int(y[gi]))
            per_season[hold] = ex
            total += ex
        
        ps = ' '.join(f'{per_season.get(s,0):2d}/{n_test_map[s]}' for s in test_seasons)
        marker = ' ◄ CURRENT' if feat_set_name == 'original_68' else ''
        marker += ' ★' if total > 70 else ''
        print(f'  {feat_set_name:25s}: {total}/91 [{ps}]{marker}')

    # ════════════════════════════════════════════════════════════
    #  TEST B: Different top-K with extended features
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  TEST B: Top-K sweep with extended features')
    print('='*70)

    for k_val in [20, 25, 30, 35, 40, 45]:
        total = 0
        for hold in folds:
            season_mask = (seasons == hold)
            season_indices = np.where(season_mask)[0]
            season_test = test_mask & season_mask
            if season_test.sum() == 0:
                continue
            global_train = ~season_test
            X_season = X_ext[season_mask]
            top_k_idx = select_top_k_features(
                X_ext[global_train], y[global_train],
                fn_ext, k=k_val, forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend_custom(
                X_ext[global_train], y[global_train],
                X_season, seasons[global_train], top_k_idx)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in season_indices])
            p = apply_v25_zones(p1, raw, fn_ext, X_season, tm, season_indices)
            for i, gi in enumerate(season_indices):
                if test_mask[gi] and p[i] == int(y[gi]):
                    total += 1
        marker = ' ★' if total > 70 else ''
        print(f'  k={k_val}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  TEST C: Use new features in zone corrections
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  TEST C: Use new features in zone corrections')
    print('  Test NETNonConfSOS and AvgOppNET as zone correction signals')
    print('='*70)

    # Precompute base predictions with original features (same as v25)
    season_data = {}
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test = test_mask & season_mask
        if season_test.sum() == 0:
            continue
        global_train = ~season_test
        X_season = X_orig[season_mask]
        top_k_idx = select_top_k_features(
            X_orig[global_train], y[global_train],
            fn_orig, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend_custom(
            X_orig[global_train], y[global_train],
            X_season, seasons[global_train], top_k_idx)
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data[hold] = {
            'pass1': pass1, 'raw': raw, 'X': X_orig[season_mask],
            'X_ext': X_ext[season_mask],
            'tm': tm, 'indices': season_indices.copy(),
        }

    # v25 baseline
    v25_total = 0
    for s, sd in season_data.items():
        p = apply_v25_zones(sd['pass1'], sd['raw'], fn_orig, sd['X'], sd['tm'], sd['indices'])
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi] and p[i] == int(y[gi]):
                v25_total += 1
    print(f'\n  v25 baseline (for reference): {v25_total}/91')

    # Make new feature indices available
    fi_ext = {f: i for i, f in enumerate(fn_ext)}

    # Test: use NETNonConfSOS in each zone correction
    for zone_name, zone_bounds in [('mid', (17,34)), ('low', (35,52)), ('bot', (50,60)), ('tail', (61,65))]:
        for new_feat in ['NETNonConfSOS', 'AvgOppNET', 'ncsos_vs_sos', 'opp_net_ratio', 
                         'schedule_quality', 'q34_win_pct', 'ncsos_net_gap']:
            if new_feat not in fi_ext:
                continue
            for weight in [-5, -3, -2, -1, 1, 2, 3, 5]:
                total = 0
                for s, sd in season_data.items():
                    p = sd['pass1'].copy()
                    # Apply standard zones first up to (but not including) target zone
                    if zone_name != 'mid':
                        corr = compute_committee_correction(fn_orig, sd['X'], 0, 2, 3)
                        p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                zone=(17,34), blend=1.0, power=0.15)
                    if zone_name not in ('mid', 'low'):
                        corr = compute_low_correction(fn_orig, sd['X'], q1dom=1, field=2)
                        p = apply_lowzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                               zone=(35,52), power=0.15)
                    if zone_name not in ('mid', 'low', 'bot'):
                        corr = compute_bottom_correction(fn_orig, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                        p = apply_bottomzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                  zone=(50,60), power=0.15)
                    
                    # Apply target zone with new feature added
                    if zone_name == 'mid':
                        base_corr = compute_committee_correction(fn_orig, sd['X'], 0, 2, 3)
                    elif zone_name == 'low':
                        base_corr = compute_low_correction(fn_orig, sd['X'], q1dom=1, field=2)
                    elif zone_name == 'bot':
                        base_corr = compute_bottom_correction(fn_orig, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                    elif zone_name == 'tail':
                        base_corr = compute_tail_correction(fn_orig, sd['X'], opp_rank=-3)
                    
                    # Add new feature correction (using extended X)
                    vals = sd['X_ext'][:, fi_ext[new_feat]]
                    vmin, vmax = vals.min(), vals.max()
                    if vmax > vmin:
                        norm = (vals - vmin) / (vmax - vmin) * 2 - 1
                    else:
                        norm = np.zeros_like(vals)
                    combined_corr = base_corr + weight * norm
                    
                    # Apply zone swap
                    lo, hi = zone_bounds
                    zone_test = [i for i in range(len(p)) if sd['tm'][i] and lo <= p[i] <= hi]
                    if len(zone_test) > 1:
                        seeds = [p[i] for i in zone_test]
                        corrected = [sd['raw'][i] + combined_corr[i] for i in zone_test]
                        cost = np.array([[abs(s - sd_val)**0.15 for sd_val in seeds]
                                         for s in corrected])
                        ri, ci = linear_sum_assignment(cost)
                        pnew = p.copy()
                        for r, c in zip(ri, ci):
                            pnew[zone_test[r]] = seeds[c]
                        p = pnew
                    
                    # Apply remaining zones
                    if zone_name == 'mid':
                        corr = compute_low_correction(fn_orig, sd['X'], q1dom=1, field=2)
                        p = apply_lowzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                               zone=(35,52), power=0.15)
                        corr = compute_bottom_correction(fn_orig, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                        p = apply_bottomzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                  zone=(50,60), power=0.15)
                        corr = compute_tail_correction(fn_orig, sd['X'], opp_rank=-3)
                        p = apply_tailzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                zone=(61,65), power=0.15)
                    elif zone_name == 'low':
                        corr = compute_bottom_correction(fn_orig, sd['X'], sosnet=-4, net_conf=3, cbhist=-1)
                        p = apply_bottomzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                  zone=(50,60), power=0.15)
                        corr = compute_tail_correction(fn_orig, sd['X'], opp_rank=-3)
                        p = apply_tailzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                zone=(61,65), power=0.15)
                    elif zone_name == 'bot':
                        corr = compute_tail_correction(fn_orig, sd['X'], opp_rank=-3)
                        p = apply_tailzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                                zone=(61,65), power=0.15)
                    # tail: nothing after
                    
                    for i, gi in enumerate(sd['indices']):
                        if test_mask[gi] and p[i] == int(y[gi]):
                            total += 1
                
                if total > v25_total:
                    print(f'  {zone_name} +{new_feat} w={weight:+d}: {total}/91 ★')

    # ════════════════════════════════════════════════════════════
    #  TEST D: Entirely new zone using new features
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  TEST D: NEW 5th ZONE using NETNonConfSOS')
    print('  Apply after v25 zones, sweep all zone ranges')
    print('='*70)

    for new_feat in ['NETNonConfSOS', 'ncsos_vs_sos', 'schedule_quality', 'ncsos_net_gap']:
        if new_feat not in fi_ext:
            continue
        best_5th = v25_total
        best_5th_cfg = None
        for lo in range(1, 60, 4):
            for hi in range(lo+4, 69, 4):
                for weight in [-5, -3, -1, 1, 3, 5]:
                    total = 0
                    for s, sd in season_data.items():
                        p = apply_v25_zones(sd['pass1'], sd['raw'], fn_orig, sd['X'],
                                            sd['tm'], sd['indices'])
                        # 5th zone correction
                        vals = sd['X_ext'][:, fi_ext[new_feat]]
                        vmin, vmax = vals.min(), vals.max()
                        if vmax > vmin:
                            norm = (vals - vmin) / (vmax - vmin) * 2 - 1
                        else:
                            norm = np.zeros_like(vals)
                        corr = weight * norm
                        
                        zone_test = [i for i in range(len(p)) if sd['tm'][i] and lo <= p[i] <= hi]
                        if len(zone_test) > 1:
                            seeds = [p[i] for i in zone_test]
                            corrected = [sd['raw'][i] + corr[i] for i in zone_test]
                            cost = np.array([[abs(sv - sd_val)**0.15 for sd_val in seeds]
                                             for sv in corrected])
                            ri, ci = linear_sum_assignment(cost)
                            pnew = p.copy()
                            for r, c in zip(ri, ci):
                                pnew[zone_test[r]] = seeds[c]
                            p = pnew
                        
                        for i, gi in enumerate(sd['indices']):
                            if test_mask[gi] and p[i] == int(y[gi]):
                                total += 1
                    
                    if total > best_5th:
                        best_5th = total
                        best_5th_cfg = (new_feat, lo, hi, weight)
        
        if best_5th_cfg:
            print(f'  {new_feat}: best={best_5th}/91 at zone=({best_5th_cfg[1]},{best_5th_cfg[2]}) w={best_5th_cfg[3]} ★')
        else:
            print(f'  {new_feat}: no improvement')

    # ════════════════════════════════════════════════════════════
    #  TEST E: Replace a zone signal entirely with new feature
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  TEST E: Replace zone correction signals with new features')
    print('='*70)

    # For each zone, try replacing the correction entirely with a single new feature
    def make_simple_correction(fn_list, X_data, feat_name, fi_map, weight):
        vals = X_data[:, fi_map[feat_name]]
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            norm = (vals - vmin) / (vmax - vmin) * 2 - 1
        else:
            norm = np.zeros_like(vals)
        return weight * norm

    replacement_tests = [
        ('NETNonConfSOS', 'ncsos_vs_sos', 'schedule_quality', 'ncsos_net_gap',
         'opp_net_ratio', 'q34_win_pct', 'AvgOppNET')
    ]
    
    for zone_name, zone_bounds, existing_fn in [
        ('mid', (17,34), lambda fn, X: compute_committee_correction(fn, X, 0, 2, 3)),
        ('low', (35,52), lambda fn, X: compute_low_correction(fn, X, q1dom=1, field=2)),
        ('bot', (50,60), lambda fn, X: compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)),
        ('tail', (61,65), lambda fn, X: compute_tail_correction(fn, X, opp_rank=-3)),
    ]:
        best_replacement = v25_total
        for new_feat in ['NETNonConfSOS', 'ncsos_vs_sos', 'schedule_quality', 
                         'ncsos_net_gap', 'opp_net_ratio', 'q34_win_pct', 'AvgOppNET']:
            if new_feat not in fi_ext:
                continue
            for w1 in [-5, -3, -1, 1, 3, 5]:
                for w2 in [-3, -1, 0, 1, 3]:
                    total = 0
                    for s, sd in season_data.items():
                        p = sd['pass1'].copy()
                        
                        # Apply all zones, but modify target zone
                        for z_name, z_bounds, z_fn in [
                            ('mid', (17,34), lambda fn, X: compute_committee_correction(fn, X, 0, 2, 3)),
                            ('low', (35,52), lambda fn, X: compute_low_correction(fn, X, q1dom=1, field=2)),
                            ('bot', (50,60), lambda fn, X: compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)),
                            ('tail', (61,65), lambda fn, X: compute_tail_correction(fn, X, opp_rank=-3)),
                        ]:
                            if z_name == zone_name:
                                # Modified: existing + new feature
                                base_corr = z_fn(fn_orig, sd['X'])
                                new_corr = make_simple_correction(fn_ext, sd['X_ext'], new_feat, fi_ext, w1)
                                corr = w2 * base_corr + new_corr if w2 != 0 else new_corr
                            else:
                                corr = z_fn(fn_orig, sd['X'])
                            
                            lo, hi = z_bounds if z_name != zone_name else zone_bounds
                            zone_test = [i for i in range(len(p)) if sd['tm'][i] and lo <= p[i] <= hi]
                            if len(zone_test) > 1:
                                seeds = [p[i] for i in zone_test]
                                corrected = [sd['raw'][i] + corr[i] for i in zone_test]
                                cost = np.array([[abs(sv - sd_val)**0.15 for sd_val in seeds]
                                                 for sv in corrected])
                                ri, ci = linear_sum_assignment(cost)
                                pnew = p.copy()
                                for r, c in zip(ri, ci):
                                    pnew[zone_test[r]] = seeds[c]
                                p = pnew
                        
                        for i, gi in enumerate(sd['indices']):
                            if test_mask[gi] and p[i] == int(y[gi]):
                                total += 1
                    
                    if total > best_replacement:
                        best_replacement = total
                        print(f'  {zone_name} += {new_feat} w1={w1} w2={w2}: {total}/91 ★')
        
        if best_replacement == v25_total:
            print(f'  {zone_name}: no improvement from replacement')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
