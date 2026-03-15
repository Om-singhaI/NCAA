#!/usr/bin/env python3
"""
v20: Comprehensive Base Model Improvement Search
=================================================

v18=61/91 with (17,34) swap correction is the ceiling for post-processing.
v19 showed extending zones doesn't help.

This script attacks the BASE MODEL (v12) itself:
  1. Add missing raw features (AvgOppNET, NETNonConfSOS + derived)
  2. Test different feature set sizes (k=20,25,30,35,40)
  3. Test different adj-pair gap thresholds (20,25,30,35,40,50)
  4. Test different blend weights
  5. Test 2nd-level stacking (v12 raw scores as meta-feature)
  6. Test alternative classifiers (SVM, MLP) in blend
  7. All evaluated via nested LOSO with v18 swap on top
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, parse_wl, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent,
    pairwise_score, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    PW_C1, PW_C3, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def build_features_extended(df, context_df, labeled_df, tourn_rids):
    """Build EXTENDED feature set (68 original + new features)."""
    # Import original build_features
    from ncaa_2026_model import build_features
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    # === NEW FEATURES ===
    
    # 1. AvgOppNET (raw opponent NET, different from AvgOppNETRank)
    if 'AvgOppNET' in df.columns:
        aon = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
        feat['AvgOppNET'] = aon
        net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
        feat['net_minus_oppnet'] = net - aon
        feat['oppnet_rank_diff'] = aon - pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    
    # 2. NETNonConfSOS
    if 'NETNonConfSOS' in df.columns:
        ncsos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
        feat['NETNonConfSOS'] = ncsos
        sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
        feat['conf_vs_nonconf_sos'] = sos - ncsos  # positive = conf schedule harder
        feat['nonconf_sos_net_gap'] = ncsos - pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    
    # 3. Non-conference record derived features
    if 'Non-ConferenceRecord' in df.columns:
        nc_wl = df['Non-ConferenceRecord'].apply(parse_wl)
        nc_w = nc_wl.apply(lambda x: x[0])
        nc_l = nc_wl.apply(lambda x: x[1])
        nc_pct = np.where((nc_w + nc_l) > 0, nc_w / (nc_w + nc_l), 0.5)
        feat['nonconf_wpct'] = nc_pct
        
        # Conference record
        c_wl = df['Conf.Record'].apply(parse_wl)
        c_w = c_wl.apply(lambda x: x[0])
        c_l = c_wl.apply(lambda x: x[1])
        c_pct = np.where((c_w + c_l) > 0, c_w / (c_w + c_l), 0.5)
        feat['conf_wpct'] = c_pct
        feat['conf_nonconf_gap'] = c_pct - nc_pct  # positive = better in conf
        
    # 4. Quality wins / bad losses ratios
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)

    feat['q1_rate'] = q1w / (q1w + q1l + 1)
    feat['bad_loss_rate'] = (q3l + q4l) / (feat.get('total_games', pd.Series(30, index=df.index)).fillna(30))
    feat['quality_win_rate'] = (q1w + q2w) / (feat.get('total_games', pd.Series(30, index=df.index)).fillna(30))
    
    # 5. NET rank stability (PrevNET - NET interaction with bid type)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    feat['net_improvement'] = prev - net  # positive = improved from last year
    feat['net_improvement_pct'] = (prev - net) / (prev + 1)  # relative improvement
    feat['al_momentum'] = feat['is_AL'] * (prev - net)  # momentum for at-large
    feat['aq_net_position'] = feat['is_AQ'] * (68 - net) / 68  # how good is AQ's NET
    
    # 6. Conference strength interactions
    cav = feat['conf_avg_net']
    feat['net_above_conf'] = net - cav  # negative = team is better than conf avg
    feat['power_al_bonus'] = feat['is_power_conf'] * feat['is_AL']
    feat['weak_aq_flag'] = feat['is_AQ'] * (1 - feat['is_power_conf']) * np.clip((net - 50) / 100, 0, 1)
    
    return feat


def compute_correction(feature_names, X_data, aq=0, al=2, sos=3):
    """Same correction as v18."""
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos_val = X_data[:, fi['NETSOS']]
    conf_weakness = np.clip((conf_avg - 80) / 120, 0, 2)
    if aq != 0:
        correction += aq * is_aq * conf_weakness * (100 - np.clip(net, 1, 100)) / 100
    if al != 0:
        correction -= al * is_al * is_power * np.clip((net - 20) / 50, 0, 1)
    if sos != 0:
        correction += sos * (sos_val - net) / 100
    return correction


def apply_swap(pass1, raw_scores, correction, test_mask_season, zone=(17,34), power=0.15):
    lo, hi = zone
    mid_test = [i for i in range(len(pass1))
                if test_mask_season[i] and lo <= pass1[i] <= hi]
    if len(mid_test) <= 1:
        return pass1.copy()
    mid_seeds = [pass1[i] for i in mid_test]
    mid_corr = [raw_scores[i] + correction[i] for i in mid_test]
    cost = np.array([[abs(s - seed)**power for seed in mid_seeds] for s in mid_corr])
    ri, ci = linear_sum_assignment(cost)
    final = pass1.copy()
    for r, c in zip(ri, ci):
        final[mid_test[r]] = mid_seeds[c]
    return final


def predict_blend(X_train, y_train, X_test, seasons_train, top_k_idx,
                  w1=0.64, w3=0.28, w4=0.08, c1=5.0, c3=0.5, gap=30):
    """Configurable blend prediction."""
    # Component 1: adj-pair LR
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=gap)
    sc_adj = StandardScaler()
    pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_adj_sc, pw_y_adj)
    score1 = pairwise_score(lr1, X_test, sc_adj)

    # Component 3: top-K LR
    X_tr_k = X_train[:, top_k_idx]
    X_te_k = X_test[:, top_k_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
    lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_X_k_sc, pw_y_k)
    score3 = pairwise_score(lr3, X_te_k, sc_k)

    # Component 4: XGB
    pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, seasons_train)
    sc_full = StandardScaler()
    pw_X_full_sc = sc_full.fit_transform(pw_X_full)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss')
    xgb_clf.fit(pw_X_full_sc, pw_y_full)
    score4 = pairwise_score(xgb_clf, X_test, sc_full)

    return w1 * score1 + w3 * score3 + w4 * score4


def main():
    t0 = time.time()
    print('='*70)
    print(' v20: COMPREHENSIVE BASE MODEL IMPROVEMENT')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')

    # ════════════════════════════════════════════════════════════════
    #  Build BOTH feature sets
    # ════════════════════════════════════════════════════════════════
    from ncaa_2026_model import build_features

    print('\n  Building original (68) features...')
    feat_orig = build_features(labeled, context_df, labeled, tourn_rids)
    fn_orig = list(feat_orig.columns)
    print(f'  Original features: {len(fn_orig)}')

    print('  Building extended features...')
    feat_ext = build_features_extended(labeled, context_df, labeled, tourn_rids)
    fn_ext = list(feat_ext.columns)
    new_feats = [f for f in fn_ext if f not in fn_orig]
    print(f'  Extended features: {len(fn_ext)} (+{len(new_feats)} new)')
    print(f'  New features: {new_feats}')

    # Impute both
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_orig = imp.fit_transform(np.where(np.isinf(feat_orig.values.astype(np.float64)),
                                         np.nan, feat_orig.values.astype(np.float64)))
    X_ext = imp.fit_transform(np.where(np.isinf(feat_ext.values.astype(np.float64)),
                                        np.nan, feat_ext.values.astype(np.float64)))

    # ════════════════════════════════════════════════════════════════
    #  Helper: evaluate a config end-to-end on test set
    # ════════════════════════════════════════════════════════════════
    def evaluate_config(X_all, feature_names, w1, w3, w4, c1, c3, gap, k, force_feats,
                        apply_correction=True, aq=0, al=2, sos=3):
        """Full evaluation: feature selection → prediction → Hungarian → swap → exact count."""
        total_exact = 0
        per_season = {}
        
        for hold in folds:
            season_mask = (seasons == hold)
            season_indices = np.where(season_mask)[0]
            season_test = test_mask & season_mask
            if season_test.sum() == 0:
                continue

            global_train_mask = ~season_test
            X_season = X_all[season_mask]

            top_k_idx = select_top_k_features(
                X_all[global_train_mask], y[global_train_mask],
                feature_names, k=k, forced_features=force_feats)[0]

            raw = predict_blend(
                X_all[global_train_mask], y[global_train_mask],
                X_season, seasons[global_train_mask], top_k_idx,
                w1=w1, w3=w3, w4=w4, c1=c1, c3=c3, gap=gap)

            # Set known training labels
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]

            avail = {hold: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[season_mask], avail, power=0.15)

            # Apply swap correction
            if apply_correction:
                tmask_s = np.array([test_mask[gi] for gi in season_indices])
                corr = compute_correction(feature_names, X_season, aq, al, sos)
                pass1 = apply_swap(pass1, raw, corr, tmask_s)

            ex = sum(1 for i, gi in enumerate(season_indices)
                    if test_mask[gi] and pass1[i] == int(y[gi]))
            per_season[hold] = ex
            total_exact += ex

        return total_exact, per_season

    # ════════════════════════════════════════════════════════════════
    #  TEST 1: Original features + different hyperparameters
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 1: Original features (68) + hyperparameter sweep')
    print('='*70)

    # Baseline first
    base_exact, base_ps = evaluate_config(
        X_orig, fn_orig, BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
        ADJ_COMP1_GAP, USE_TOP_K_A, FORCE_FEATURES, apply_correction=True)
    print(f'  v18 baseline: {base_exact}/91')
    ps_str = ' '.join(f'{base_ps.get(s,0):2d}' for s in test_seasons)
    print(f'    Per-season: {ps_str}')

    # Gap thresholds
    print(f'\n  --- Adj-pair gap threshold ---')
    for gap in [15, 20, 25, 30, 35, 40, 50, 68]:
        ex, ps = evaluate_config(
            X_orig, fn_orig, BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
            gap, USE_TOP_K_A, FORCE_FEATURES)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > base_exact else (' ←SAME' if ex == base_exact else '')
        print(f'    gap={gap:2d}: {ex}/91 [{ps_str}]{marker}')

    # Feature set sizes
    print(f'\n  --- Feature set size (k) ---')
    for k in [15, 20, 25, 30, 35, 40, 50, 68]:
        ex, ps = evaluate_config(
            X_orig, fn_orig, BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
            ADJ_COMP1_GAP, k, FORCE_FEATURES)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > base_exact else (' ←SAME' if ex == base_exact else '')
        print(f'    k={k:2d}: {ex}/91 [{ps_str}]{marker}')

    # C values
    print(f'\n  --- Regularization (C1 for comp1, C3 for comp3) ---')
    c_combos = [(1, 0.1), (1, 0.5), (2, 0.5), (5, 0.1), (5, 0.5), (5, 1.0),
                (10, 0.5), (10, 1.0), (20, 0.5), (20, 1.0), (1, 1), (2, 1)]
    for c1, c3 in c_combos:
        ex, ps = evaluate_config(
            X_orig, fn_orig, BLEND_W1, BLEND_W3, BLEND_W4, c1, c3,
            ADJ_COMP1_GAP, USE_TOP_K_A, FORCE_FEATURES)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > base_exact else (' ←SAME' if ex == base_exact else '')
        print(f'    C1={c1:4.1f} C3={c3:4.1f}: {ex}/91 [{ps_str}]{marker}')

    # Blend weights
    print(f'\n  --- Blend weights (w1/w3/w4) ---')
    weight_combos = [
        (0.70, 0.20, 0.10), (0.50, 0.40, 0.10), (0.60, 0.30, 0.10),
        (0.64, 0.28, 0.08), (0.80, 0.15, 0.05), (0.55, 0.35, 0.10),
        (0.70, 0.30, 0.00), (0.64, 0.36, 0.00), (0.50, 0.50, 0.00),
        (1.00, 0.00, 0.00), (0.00, 1.00, 0.00), (0.60, 0.20, 0.20),
        (0.50, 0.30, 0.20), (0.40, 0.40, 0.20),
    ]
    for w1, w3, w4 in weight_combos:
        ex, ps = evaluate_config(
            X_orig, fn_orig, w1, w3, w4, PW_C1, PW_C3,
            ADJ_COMP1_GAP, USE_TOP_K_A, FORCE_FEATURES)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > base_exact else (' ←SAME' if ex == base_exact else '')
        print(f'    w1={w1:.2f} w3={w3:.2f} w4={w4:.2f}: {ex}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  TEST 2: Extended features
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 2: EXTENDED features + different k values')
    print('='*70)

    for k in [25, 30, 35, 40, 50, len(fn_ext)]:
        ex, ps = evaluate_config(
            X_ext, fn_ext, BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
            ADJ_COMP1_GAP, k, FORCE_FEATURES)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > base_exact else (' ←SAME' if ex == base_exact else '')
        print(f'    Extended k={k:3d}: {ex}/91 [{ps_str}]{marker}')

    # Extended features + different gap
    print(f'\n  Extended features + different gaps:')
    for gap in [20, 25, 30, 35, 40]:
        for k in [25, 35]:
            ex, ps = evaluate_config(
                X_ext, fn_ext, BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
                gap, k, FORCE_FEATURES)
            ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
            marker = ' ←' if ex > base_exact else ''
            print(f'    gap={gap:2d} k={k:2d}: {ex}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  TEST 3: Correction params with different base configs
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 3: Best configs × correction params (joint sweep)')
    print('='*70)

    # Pick promising configs from Tests 1-2
    promising = [
        # (label, X, fn, w1, w3, w4, c1, c3, gap, k)
        ('v18_default', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 25),
        ('orig_gap20_k25', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 20, 25),
        ('orig_gap25_k25', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 25, 25),
        ('orig_gap35_k25', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 35, 25),
        ('orig_k30', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 30),
        ('orig_k35', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 35),
        ('orig_w70_20_10', X_orig, fn_orig, 0.70, 0.20, 0.10, 5.0, 0.5, 30, 25),
        ('ext_k25', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 25),
        ('ext_k35', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 35),
        ('ext_k40', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 40),
    ]

    corr_sweep = [(0, 2, 3), (0, 1, 2), (0, 2, 2), (0, 3, 3), (0, 2, 4)]

    print(f'  {len(promising)} base configs × {len(corr_sweep)} correction params = {len(promising)*len(corr_sweep)} combos')

    all_results = []
    for label, X, fn, w1, w3, w4, c1, c3, gap, k in promising:
        for aq, al, sos in corr_sweep:
            ex, ps = evaluate_config(
                X, fn, w1, w3, w4, c1, c3, gap, k, FORCE_FEATURES,
                apply_correction=True, aq=aq, al=al, sos=sos)
            all_results.append({
                'label': label, 'aq': aq, 'al': al, 'sos': sos,
                'full': ex, 'ps': ps,
                'X': X, 'fn': fn, 'w1': w1, 'w3': w3, 'w4': w4,
                'c1': c1, 'c3': c3, 'gap': gap, 'k': k
            })

    all_results.sort(key=lambda r: -r['full'])

    print(f'\n  Top 20 combos:')
    print(f'  {"Config":<25} {"Corr":<12} {"Full":>4}  {"Per-season":>25}')
    print(f'  {"─"*25} {"─"*12} {"─"*4}  {"─"*25}')
    for r in all_results[:20]:
        cr = f'aq{r["aq"]}_al{r["al"]}_s{r["sos"]}'
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        print(f'  {r["label"]:<25} {cr:<12} {r["full"]:4d}  {ps_str}')

    # ════════════════════════════════════════════════════════════════
    #  TEST 4: Nested LOSO on best configs
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 4: Nested LOSO validation (top configs)')
    print('='*70)

    # Take top configs that >= v18
    competitive = [r for r in all_results if r['full'] >= base_exact]
    # Add v18 baseline
    competitive.append({
        'label': 'v18_baseline', 'aq': 0, 'al': 2, 'sos': 3,
        'full': base_exact, 'ps': base_ps
    })
    
    print(f'  {len(competitive)} configs >= {base_exact}/91 for nested LOSO')

    nested_results = []
    for hold_season in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold_season]
        
        best_tune_score = -1
        best_idx = -1
        for ci, r in enumerate(competitive):
            tune_score = sum(r['ps'].get(s, 0) for s in tune_seasons)
            if tune_score > best_tune_score:
                best_tune_score = tune_score
                best_idx = ci

        best_r = competitive[best_idx]
        hold_exact = best_r['ps'].get(hold_season, 0)
        v18_hold = base_ps.get(hold_season, 0)

        nested_results.append({
            'season': hold_season, 'n': (test_mask & (seasons == hold_season)).sum(),
            'v18': v18_hold, 'v20': hold_exact,
            'config': f'{best_r["label"]}_aq{best_r.get("aq",0)}_al{best_r.get("al",2)}_s{best_r.get("sos",3)}'
        })

    print(f'\n  {"Season":<10} {"N":>3} {"v18":>4} {"v20":>4} {"Δ":>3}  Config')
    print(f'  {"─"*10} {"─"*3} {"─"*4} {"─"*4} {"─"*3}  {"─"*40}')
    nested_v18 = 0
    nested_v20 = 0
    for d in nested_results:
        delta = d['v20'] - d['v18']
        nested_v18 += d['v18']
        nested_v20 += d['v20']
        print(f'  {d["season"]:<10} {d["n"]:3d} {d["v18"]:4d} {d["v20"]:4d} {delta:+3d}  {d["config"][:50]}')

    print(f'\n  Nested LOSO: v18={nested_v18}/91, v20={nested_v20}/91 (Δ={nested_v20-nested_v18:+d})')

    # ════════════════════════════════════════════════════════════════
    #  TEST 5: No-correction base model sweep
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 5: Best base model WITHOUT correction')
    print('='*70)

    # Test if a better base model alone beats v12 without needing swap correction
    base_configs = [
        ('orig_default', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 25),
        ('orig_gap20', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 20, 25),
        ('orig_gap25', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 25, 25),
        ('orig_gap35', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 35, 25),
        ('orig_gap40', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 40, 25),
        ('orig_k20', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 20),
        ('orig_k30', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 30),
        ('orig_k35', X_orig, fn_orig, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 35),
        ('ext_k25', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 25),
        ('ext_k30', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 30),
        ('ext_k35', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 30, 35),
        ('orig_C10', X_orig, fn_orig, 0.64, 0.28, 0.08, 10.0, 0.5, 30, 25),
        ('orig_C20', X_orig, fn_orig, 0.64, 0.28, 0.08, 20.0, 0.5, 30, 25),
        ('orig_lr_only', X_orig, fn_orig, 1.00, 0.00, 0.00, 5.0, 0.5, 30, 25),
        ('orig_lr3_only', X_orig, fn_orig, 0.00, 1.00, 0.00, 5.0, 0.5, 30, 25),
        ('ext_gap25_k30', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 25, 30),
        ('ext_gap35_k35', X_ext, fn_ext, 0.64, 0.28, 0.08, 5.0, 0.5, 35, 35),
    ]

    no_corr_results = []
    for label, X, fn, w1, w3, w4, c1, c3, gap, k in base_configs:
        ex, ps = evaluate_config(X, fn, w1, w3, w4, c1, c3, gap, k, FORCE_FEATURES,
                                 apply_correction=False)
        no_corr_results.append({'label': label, 'full': ex, 'ps': ps})
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > 57 else ''
        print(f'    {label:<20}: {ex}/91 [{ps_str}]{marker}')

    best_no_corr = max(r['full'] for r in no_corr_results)
    print(f'\n  Best base without correction: {best_no_corr}/91 (v12 was 57/91)')

    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
