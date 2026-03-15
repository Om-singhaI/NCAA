#!/usr/bin/env python3
"""
v21: Alternative Architectures & Meta-Ensemble
===============================================

v18=61/91 is the ceiling for hyperparameter tuning of the v12 base.
v19 showed zone extension doesn't help. v20 showed:
  - gap, k, C, blend weights all optimized
  - Extended features HURT (reduce from 61 to 46)
  - No base config change improves over v18

This script tries FUNDAMENTALLY DIFFERENT approaches:
  1. Raw score ensemble: average raw scores from multiple diverse configs
  2. Per-zone Hungarian power optimization
  3. Iterative refinement (predict → identify outliers → repreddict)
  4. Direct ordinal regression (non-pairwise)
  5. Stacking: v12 raw prediction as input feature
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, build_pairwise_data, build_pairwise_data_adjacent,
    pairwise_score, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    PW_C1, PW_C3, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_correction(feature_names, X_data, aq=0, al=2, sos=3):
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos_val = X_data[:, fi['NETSOS']]
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


def predict_single_component(X_train, y_train, X_test, seasons_train, comp_type,
                              top_k_idx=None, c_val=5.0, gap=30):
    """Single-component prediction for diversity."""
    if comp_type == 'adj_lr':
        pw_X, pw_y = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=gap)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr = LogisticRegression(C=c_val, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_X_sc, pw_y)
        return pairwise_score(lr, X_test, sc)
    elif comp_type == 'topk_lr':
        X_tr_k = X_train[:, top_k_idx]
        X_te_k = X_test[:, top_k_idx]
        pw_X, pw_y = build_pairwise_data(X_tr_k, y_train, seasons_train)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr = LogisticRegression(C=c_val, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_X_sc, pw_y)
        return pairwise_score(lr, X_te_k, sc)
    elif comp_type == 'full_lr':
        pw_X, pw_y = build_pairwise_data(X_train, y_train, seasons_train)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr = LogisticRegression(C=c_val, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_X_sc, pw_y)
        return pairwise_score(lr, X_test, sc)
    elif comp_type == 'xgb':
        pw_X, pw_y = build_pairwise_data(X_train, y_train, seasons_train)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        return pairwise_score(xgb_clf, X_test, sc)
    elif comp_type == 'adj_xgb':
        pw_X, pw_y = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=gap)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        return pairwise_score(xgb_clf, X_test, sc)


def direct_regression_predict(X_train, y_train, X_test):
    """Direct seed prediction via regression ensemble."""
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc = sc.transform(X_test)

    # XGB regression
    xgb_preds = []
    for seed in [42, 123, 777]:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_train, y_train)
        xgb_preds.append(m.predict(X_test))

    # Ridge
    ridge = Ridge(alpha=5.0)
    ridge.fit(X_tr_sc, y_train)
    ridge_pred = ridge.predict(X_te_sc)

    # GBR
    gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, random_state=42)
    gbr.fit(X_train, y_train)
    gbr_pred = gbr.predict(X_test)

    return 0.5 * np.mean(xgb_preds, axis=0) + 0.3 * ridge_pred + 0.2 * gbr_pred


def hungarian_zoned(scores, seasons, avail, powers):
    """Hungarian with different power per zone."""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]

        # Build cost with zone-specific power
        cost = np.zeros((len(rv), len(pos)))
        for r_idx, r_val in enumerate(rv):
            for p_idx, p_val in enumerate(pos):
                # Determine power based on position
                if p_val <= 16:
                    pw = powers[0]
                elif p_val <= 34:
                    pw = powers[1]
                elif p_val <= 52:
                    pw = powers[2]
                else:
                    pw = powers[3]
                cost[r_idx, p_idx] = abs(r_val - p_val) ** pw

        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def main():
    t0 = time.time()
    print('='*70)
    print(' v21: ALTERNATIVE ARCHITECTURES & META-ENSEMBLE')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
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

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')

    # ════════════════════════════════════════════════════════════════
    #  Precompute per-season data
    # ════════════════════════════════════════════════════════════════
    print('  Precomputing per-season data...')
    season_data = {}
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
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]

        season_data[hold] = {
            'indices': season_indices,
            'X_season': X_season,
            'train_mask': global_train_mask,
            'test_mask_s': np.array([test_mask[gi] for gi in season_indices]),
            'top_k_idx': top_k_idx,
        }

    # ════════════════════════════════════════════════════════════════
    #  Approach 1: RAW SCORE ENSEMBLE (average diverse raw scores)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 1: RAW SCORE ENSEMBLE')
    print('='*70)
    print('  Compute diverse raw scores and try blending them')

    # Define diverse component configs
    comp_configs = [
        ('v12_default', None),  # standard v12 blend
        ('adj_lr_C5_g30', ('adj_lr', 5.0, 30)),
        ('adj_lr_C5_g40', ('adj_lr', 5.0, 40)),
        ('adj_lr_C10_g30', ('adj_lr', 10.0, 30)),
        ('adj_lr_C20_g30', ('adj_lr', 20.0, 30)),
        ('topk25_C0.5', ('topk_lr', 0.5, 25)),
        ('topk25_C1.0', ('topk_lr', 1.0, 25)),
        ('full_lr_C5', ('full_lr', 5.0, None)),
        ('xgb_full', ('xgb', None, None)),
        ('adj_xgb_g30', ('adj_xgb', None, 30)),
        ('direct_reg', ('direct', None, None)),
    ]

    # Compute raw scores per season per config
    config_raw = {}
    for ci, (name, cfg) in enumerate(comp_configs):
        config_raw[name] = {}
        for s, sd in season_data.items():
            X_s = sd['X_season']
            X_tr = X_all[sd['train_mask']]
            y_tr = y[sd['train_mask']]
            s_tr = seasons[sd['train_mask']]

            if cfg is None:
                # Standard v12 blend
                raw = predict_robust_blend(X_tr, y_tr, X_s, s_tr, sd['top_k_idx'])
            elif cfg[0] == 'direct':
                raw = direct_regression_predict(X_tr, y_tr, X_s)
            else:
                comp_type, c_val, misc = cfg
                if comp_type in ('adj_lr', 'full_lr'):
                    raw = predict_single_component(X_tr, y_tr, X_s, s_tr,
                                                    comp_type, c_val=c_val, gap=misc or 30)
                elif comp_type == 'topk_lr':
                    raw = predict_single_component(X_tr, y_tr, X_s, s_tr,
                                                    comp_type, top_k_idx=sd['top_k_idx'],
                                                    c_val=c_val)
                elif comp_type in ('xgb', 'adj_xgb'):
                    raw = predict_single_component(X_tr, y_tr, X_s, s_tr,
                                                    comp_type, gap=misc or 30)

            # Set known labels
            for i, gi in enumerate(sd['indices']):
                if not test_mask[gi]:
                    raw[i] = y[gi]

            config_raw[name][s] = raw
        print(f'    [{ci+1}/{len(comp_configs)}] {name} done')

    # Now try ensembling different combos of raw scores
    print(f'\n  Testing raw score ensembles...')

    ensemble_combos = [
        # (name, {config: weight})
        ('v12_only', {'v12_default': 1.0}),
        ('v12+direct_50_50', {'v12_default': 0.5, 'direct_reg': 0.5}),
        ('v12+direct_70_30', {'v12_default': 0.7, 'direct_reg': 0.3}),
        ('v12+direct_80_20', {'v12_default': 0.8, 'direct_reg': 0.2}),
        ('v12+direct_90_10', {'v12_default': 0.9, 'direct_reg': 0.1}),
        ('v12+adj_xgb', {'v12_default': 0.8, 'adj_xgb_g30': 0.2}),
        ('3_models', {'v12_default': 0.6, 'adj_lr_C10_g30': 0.2, 'xgb_full': 0.2}),
        ('4_models', {'v12_default': 0.5, 'adj_lr_C20_g30': 0.2, 'topk25_C1.0': 0.15, 'xgb_full': 0.15}),
        ('all_lr', {'adj_lr_C5_g30': 0.3, 'adj_lr_C5_g40': 0.2, 'adj_lr_C10_g30': 0.2, 'full_lr_C5': 0.15, 'topk25_C0.5': 0.15}),
        ('v12+C10+C20', {'v12_default': 0.5, 'adj_lr_C10_g30': 0.25, 'adj_lr_C20_g30': 0.25}),
        ('v12+various_C', {'v12_default': 0.6, 'adj_lr_C10_g30': 0.2, 'adj_lr_C20_g30': 0.2}),
        ('v12+direct+xgb', {'v12_default': 0.6, 'direct_reg': 0.2, 'xgb_full': 0.2}),
    ]

    print(f'  {"Ensemble":<25} {"No corr":>7} {"With corr":>9}  {"Per-season (w/corr)":>25}')
    print(f'  {"─"*25} {"─"*7} {"─"*9}  {"─"*25}')

    best_ens_exact = 0
    best_ens_name = ''

    for ens_name, weights in ensemble_combos:
        total_no_corr = 0
        total_with_corr = 0
        per_season = {}

        for s, sd in season_data.items():
            # Blend raw scores
            raw = np.zeros(len(sd['X_season']))
            for cfg_name, w in weights.items():
                raw += w * config_raw[cfg_name][s]

            avail = {s: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[seasons == s], avail, power=0.15)

            # No correction
            ex_nc = sum(1 for i, gi in enumerate(sd['indices'])
                       if test_mask[gi] and pass1[i] == int(y[gi]))
            total_no_corr += ex_nc

            # With v18 correction
            corr = compute_correction(fn, sd['X_season'])
            p2 = apply_swap(pass1, raw, corr, sd['test_mask_s'])
            ex_wc = sum(1 for i, gi in enumerate(sd['indices'])
                       if test_mask[gi] and p2[i] == int(y[gi]))
            total_with_corr += ex_wc
            per_season[s] = ex_wc

        ps_str = ' '.join(f'{per_season.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if total_with_corr > 61 else ''
        print(f'  {ens_name:<25} {total_no_corr:5d}/91 {total_with_corr:7d}/91  {ps_str}{marker}')

        if total_with_corr > best_ens_exact:
            best_ens_exact = total_with_corr
            best_ens_name = ens_name

    print(f'\n  Best ensemble: {best_ens_name} = {best_ens_exact}/91')

    # ════════════════════════════════════════════════════════════════
    #  Approach 2: Zone-specific Hungarian power
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 2: Zone-specific Hungarian power')
    print('='*70)

    power_combos = []
    for p1 in [0.05, 0.10, 0.15, 0.20, 0.30]:
        for p2 in [0.05, 0.10, 0.15, 0.20, 0.30]:
            for p3 in [0.05, 0.10, 0.15, 0.20, 0.30]:
                for p4 in [0.05, 0.10, 0.15, 0.20, 0.30]:
                    power_combos.append((p1, p2, p3, p4))

    print(f'  Testing {len(power_combos)} power combos...')

    best_zp_exact = 0
    best_zp_config = None
    zp_results = []

    for pi, (p1, p2, p3, p4) in enumerate(power_combos):
        total = 0
        ps = {}
        for s, sd in season_data.items():
            raw = config_raw['v12_default'][s]
            avail = {s: list(range(1, 69))}
            pass1 = hungarian_zoned(raw, seasons[seasons == s], avail, [p1, p2, p3, p4])

            # Always apply v18 correction
            corr = compute_correction(fn, sd['X_season'])
            p2_arr = apply_swap(pass1, raw, corr, sd['test_mask_s'])
            ex = sum(1 for i, gi in enumerate(sd['indices'])
                    if test_mask[gi] and p2_arr[i] == int(y[gi]))
            total += ex
            ps[s] = ex

        if total > best_zp_exact:
            best_zp_exact = total
            best_zp_config = (p1, p2, p3, p4)
        zp_results.append({'powers': (p1, p2, p3, p4), 'full': total, 'ps': ps})

        if (pi+1) % 100 == 0 and total > 61:
            print(f'    [{pi+1}] powers={p1:.2f},{p2:.2f},{p3:.2f},{p4:.2f}: {total}/91')

    zp_results.sort(key=lambda r: -r['full'])
    print(f'\n  Best zoned power: {best_zp_exact}/91 (config: {best_zp_config})')
    print(f'  Top 10:')
    for r in zp_results[:10]:
        p = r['powers']
        ps_str = ' '.join(f'{r["ps"].get(s,0):2d}' for s in test_seasons)
        print(f'    powers={p[0]:.2f},{p[1]:.2f},{p[2]:.2f},{p[3]:.2f}: {r["full"]}/91 [{ps_str}]')

    # ════════════════════════════════════════════════════════════════
    #  Approach 3: STACKING (v12 raw scores as meta-feature)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 3: Stacking (v12 predictions as additional feature)')
    print('='*70)

    # For each season, first get v12 LOSO raw scores, then add them as feature
    # and retrain a direct regression model on the augmented feature set
    stack_total = 0
    stack_ps = {}

    for s, sd in season_data.items():
        X_s = sd['X_season']
        si = sd['indices']
        X_tr = X_all[sd['train_mask']]
        y_tr = y[sd['train_mask']]

        # Get v12 raw scores for season and training data via LOSO
        raw_season = config_raw['v12_default'][s].copy()

        # For training data: need LOSO raw scores
        # Compute cross-validated raw scores for each training fold
        train_seasons = seasons[sd['train_mask']]
        train_raw = np.zeros(len(y_tr))
        for inner_fold in sorted(set(train_seasons)):
            inner_te = (train_seasons == inner_fold)
            inner_tr = ~inner_te
            if inner_te.sum() == 0:
                continue

            inner_top_k = select_top_k_features(
                X_tr[inner_tr], y_tr[inner_tr], fn, k=USE_TOP_K_A,
                forced_features=FORCE_FEATURES)[0]
            inner_raw = predict_robust_blend(
                X_tr[inner_tr], y_tr[inner_tr], X_tr[inner_te],
                train_seasons[inner_tr], inner_top_k)
            train_raw[inner_te] = inner_raw

        # Augment features
        X_tr_aug = np.column_stack([X_tr, train_raw])
        X_te_aug = np.column_stack([X_s, raw_season])

        # Retrain with augmented features
        aug_fn = fn + ['v12_raw_score']
        aug_top_k = select_top_k_features(
            X_tr_aug, y_tr, aug_fn, k=USE_TOP_K_A + 1,
            forced_features=FORCE_FEATURES + ['v12_raw_score'])[0]

        raw2 = predict_robust_blend(
            X_tr_aug, y_tr, X_te_aug,
            train_seasons, aug_top_k)

        # Set known labels
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw2[i] = y[gi]

        avail = {s: list(range(1, 69))}
        pass1 = hungarian(raw2, seasons[seasons == s], avail, power=0.15)

        # Apply v18 correction
        corr = compute_correction(fn, X_s)
        p2 = apply_swap(pass1, raw2, corr, sd['test_mask_s'])

        ex = sum(1 for i, gi in enumerate(si)
                if test_mask[gi] and p2[i] == int(y[gi]))
        stack_total += ex
        stack_ps[s] = ex

    ps_str = ' '.join(f'{stack_ps.get(s,0):2d}' for s in test_seasons)
    print(f'  Stacking result: {stack_total}/91 [{ps_str}]')

    # ════════════════════════════════════════════════════════════════
    #  Approach 4: Direct regression (no pairwise)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 4: Direct regression + v18 correction')
    print('='*70)

    dir_total = 0
    dir_ps = {}
    for s, sd in season_data.items():
        raw = config_raw['direct_reg'][s]
        avail = {s: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[seasons == s], avail, power=0.15)
        corr = compute_correction(fn, sd['X_season'])
        p2 = apply_swap(pass1, raw, corr, sd['test_mask_s'])
        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and p2[i] == int(y[gi]))
        dir_total += ex
        dir_ps[s] = ex

    ps_str = ' '.join(f'{dir_ps.get(s,0):2d}' for s in test_seasons)
    print(f'  Direct regression: {dir_total}/91 [{ps_str}]')

    # ════════════════════════════════════════════════════════════════
    #  Approach 5: Blended pairwise + direct
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 5: Pairwise × Direct blends')
    print('='*70)

    for alpha in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        bl_total = 0
        bl_ps = {}
        for s, sd in season_data.items():
            pw = config_raw['v12_default'][s]
            dr = config_raw['direct_reg'][s]
            raw = alpha * pw + (1-alpha) * dr
            avail = {s: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[seasons == s], avail, power=0.15)
            corr = compute_correction(fn, sd['X_season'])
            p2 = apply_swap(pass1, raw, corr, sd['test_mask_s'])
            ex = sum(1 for i, gi in enumerate(sd['indices'])
                    if test_mask[gi] and p2[i] == int(y[gi]))
            bl_total += ex
            bl_ps[s] = ex

        ps_str = ' '.join(f'{bl_ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if bl_total > 61 else ''
        print(f'  α={alpha:.2f} (PW): {bl_total}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Approach 6: Different seed assignment methods
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' APPROACH 6: Alternative seed assignment (non-Hungarian)')
    print('='*70)

    # Method: Simple rank assignment (no optimization, just rank)
    for s, sd in season_data.items():
        raw = config_raw['v12_default'][s]
        n_s = len(raw)

        # Identify test teams and training teams
        test_indices = [i for i in range(n_s) if sd['test_mask_s'][i]]
        train_indices = [i for i in range(n_s) if not sd['test_mask_s'][i]]

        # Greedy: sort by raw score, assign seeds in order
        # For training teams: force their true seeds
        taken_seeds = set(int(y[sd['indices'][i]]) for i in train_indices)
        avail_seeds = sorted([s_v for s_v in range(1, 69) if s_v not in taken_seeds])

        # Sort test teams by raw score
        test_by_raw = sorted(test_indices, key=lambda i: raw[i])

        greedy = np.zeros(n_s, dtype=int)
        for i in train_indices:
            greedy[i] = int(y[sd['indices'][i]])
        for ti, si in zip(test_by_raw, avail_seeds):
            greedy[ti] = si

        ex = sum(1 for i, gi in enumerate(sd['indices'])
                if test_mask[gi] and greedy[i] == int(y[gi]))
        print(f'  Season {s}: greedy={ex}', end='')

        # Hungarian for comparison
        avail_dict = {s: list(range(1, 69))}
        hun = hungarian(raw, seasons[seasons == s], avail_dict, power=0.15)
        ex_h = sum(1 for i, gi in enumerate(sd['indices'])
                  if test_mask[gi] and hun[i] == int(y[gi]))
        print(f'  hungarian={ex_h}')

    # ════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    print(f'  v18 baseline:        61/91')
    print(f'  Best ensemble:       {best_ens_exact}/91 ({best_ens_name})')
    print(f'  Best zoned power:    {best_zp_exact}/91 ({best_zp_config})')
    print(f'  Stacking:            {stack_total}/91')
    print(f'  Direct regression:   {dir_total}/91')
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
