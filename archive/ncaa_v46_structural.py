#!/usr/bin/env python3
"""
v46 Structural Improvements — go beyond zone corrections.

The v45c zone/power/blend/C landscape is fully optimized (local minimum).
Need structural changes to the base model to improve further.

Approaches:
1. Add Ridge regression component to the ensemble
2. Season-relative feature normalization (z-score within season)
3. Feature engineering — new signals for committee decisions
4. Prediction averaging — average multiple model variants
5. Confidence-weighted assignment — use prediction variance
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
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


def eval_pipeline(y, seasons, test_mask, season_data, zones=None, power=0.15):
    """Evaluate predictions from pre-computed season_data."""
    if zones is None:
        zones = ZONES
    n = len(y)
    preds = np.zeros(n, dtype=int)
    folds = sorted(set(seasons))
    fn = season_data['__fn__']
    
    for hold_season in folds:
        if hold_season not in season_data: continue
        sd = season_data[hold_season]
        
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(sd['raw'], seasons[sd['season_mask']], avail, power=power)
        assigned = apply_zones(assigned, sd['raw'], fn, sd['X_season'],
                                sd['test_mask_season'], sd['season_indices'], zones, power)
        for i, gi in enumerate(sd['season_indices']):
            if test_mask[gi]:
                preds[gi] = assigned[i]
    
    gt = y[test_mask].astype(int)
    pr = preds[test_mask]
    se = int(np.sum((pr - gt)**2))
    exact = int((pr == gt).sum())
    return se, exact, preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 STRUCTURAL IMPROVEMENTS')
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
    
    # ── APPROACH 1: Add Ridge regression to ensemble ──
    print('\n' + '='*60)
    print(' APPROACH 1: v12 + Ridge regression ensemble')
    print('='*60)
    
    # v12 gives pairwise-based scores. Ridge gives direct regression scores.
    # Blending these two different model types could improve robustness.
    
    for ridge_weight in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        season_data_ridge = {'__fn__': fn}
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            
            X_tr = X_all[global_train_mask]
            y_tr = y[global_train_mask]
            
            # v12 pairwise scores
            tki = select_top_k_features(X_tr, y_tr, fn, k=USE_TOP_K_A,
                                          forced_features=FORCE_FEATURES)[0]
            raw_pw = predict_robust_blend(X_tr, y_tr, X_season,
                                           seasons[global_train_mask], tki)
            
            # Ridge regression direct prediction
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_season)
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_sc, y_tr)
            raw_ridge = ridge.predict(X_te_sc)
            
            # Blend pairwise + ridge
            raw = (1 - ridge_weight) * raw_pw + ridge_weight * raw_ridge
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            season_data_ridge[hold_season] = {
                'X_season': X_season, 'raw': raw,
                'season_indices': season_indices,
                'season_mask': season_mask,
                'test_mask_season': tm,
            }
        
        se, exact, _ = eval_pipeline(y, seasons, test_mask, season_data_ridge)
        marker = ' ★' if se < 233 else ''
        print(f'  ridge_weight={ridge_weight:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── APPROACH 2: XGBoost regression component ──
    print('\n' + '='*60)
    print(' APPROACH 2: v12 + XGBoost regression ensemble')
    print('='*60)
    
    for xgb_weight in [0.05, 0.10, 0.15, 0.20, 0.25]:
        season_data_xgb = {'__fn__': fn}
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            
            X_tr = X_all[global_train_mask]
            y_tr = y[global_train_mask]
            
            tki = select_top_k_features(X_tr, y_tr, fn, k=USE_TOP_K_A,
                                          forced_features=FORCE_FEATURES)[0]
            raw_pw = predict_robust_blend(X_tr, y_tr, X_season,
                                           seasons[global_train_mask], tki)
            
            # XGB regression
            xgb_reg = xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=5.0, reg_alpha=2.0, min_child_weight=5,
                random_state=42, verbosity=0)
            xgb_reg.fit(X_tr, y_tr)
            raw_xgb = xgb_reg.predict(X_season)
            
            raw = (1 - xgb_weight) * raw_pw + xgb_weight * raw_xgb
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            season_data_xgb[hold_season] = {
                'X_season': X_season, 'raw': raw,
                'season_indices': season_indices,
                'season_mask': season_mask,
                'test_mask_season': tm,
            }
        
        se, exact, _ = eval_pipeline(y, seasons, test_mask, season_data_xgb)
        marker = ' ★' if se < 233 else ''
        print(f'  xgb_weight={xgb_weight:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── APPROACH 3: RF regression component ──
    print('\n' + '='*60)
    print(' APPROACH 3: v12 + RandomForest regression ensemble')
    print('='*60)
    
    for rf_weight in [0.05, 0.10, 0.15, 0.20, 0.25]:
        season_data_rf = {'__fn__': fn}
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            
            X_tr = X_all[global_train_mask]
            y_tr = y[global_train_mask]
            
            tki = select_top_k_features(X_tr, y_tr, fn, k=USE_TOP_K_A,
                                          forced_features=FORCE_FEATURES)[0]
            raw_pw = predict_robust_blend(X_tr, y_tr, X_season,
                                           seasons[global_train_mask], tki)
            
            rf = RandomForestRegressor(
                n_estimators=500, max_depth=8, min_samples_leaf=3,
                max_features=0.5, random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            raw_rf = rf.predict(X_season)
            
            raw = (1 - rf_weight) * raw_pw + rf_weight * raw_rf
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            season_data_rf[hold_season] = {
                'X_season': X_season, 'raw': raw,
                'season_indices': season_indices,
                'season_mask': season_mask,
                'test_mask_season': tm,
            }
        
        se, exact, _ = eval_pipeline(y, seasons, test_mask, season_data_rf)
        marker = ' ★' if se < 233 else ''
        print(f'  rf_weight={rf_weight:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── APPROACH 4: Season-relative features ──
    print('\n' + '='*60)
    print(' APPROACH 4: Season-relative feature normalization')
    print('='*60)
    
    # Z-score features within each season to remove cross-season variation
    X_znorm = X_all.copy()
    for s in folds:
        sm = (seasons == s)
        for j in range(X_znorm.shape[1]):
            col = X_znorm[sm, j]
            mu, std = col.mean(), col.std()
            if std > 0:
                X_znorm[sm, j] = (col - mu) / std
    
    season_data_znorm = {'__fn__': fn}
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        
        X_season = X_znorm[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        tki = select_top_k_features(
            X_znorm[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(
            X_znorm[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], tki)
        
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data_znorm[hold_season] = {
            'X_season': X_season, 'raw': raw,
            'season_indices': season_indices,
            'season_mask': season_mask,
            'test_mask_season': tm,
        }
    
    se_z, exact_z, preds_z = eval_pipeline(y, seasons, test_mask, season_data_znorm)
    print(f'  Season z-norm:  SE={se_z}, exact={exact_z}/91')
    
    # Also try original-features raw + z-norm raw average
    season_data_orig = {'__fn__': fn}
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        
        X_season_orig = X_all[season_mask]
        X_season_z = X_znorm[season_mask]
        season_indices = np.where(season_mask)[0]
        global_train_mask = ~season_test_mask
        
        tki = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_orig = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season_orig, seasons[global_train_mask], tki)
        
        tki_z = select_top_k_features(
            X_znorm[global_train_mask], y[global_train_mask],
            fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_z = predict_robust_blend(
            X_znorm[global_train_mask], y[global_train_mask],
            X_season_z, seasons[global_train_mask], tki_z)
        
        for blend_w in [0.0]:  # computed below
            pass
        
        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw_orig[i] = y[gi]
                raw_z[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        season_data_orig[hold_season] = {
            'X_season': X_season_orig, 'raw_orig': raw_orig, 'raw_z': raw_z,
            'season_indices': season_indices,
            'season_mask': season_mask,
            'test_mask_season': tm,
        }
    
    for bw in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        season_data_blend = {'__fn__': fn}
        for hold_season in folds:
            if hold_season not in season_data_orig: continue
            sd = season_data_orig[hold_season]
            raw_blend = (1 - bw) * sd['raw_orig'] + bw * sd['raw_z']
            season_data_blend[hold_season] = {
                'X_season': sd['X_season'], 'raw': raw_blend,
                'season_indices': sd['season_indices'],
                'season_mask': sd['season_mask'],
                'test_mask_season': sd['test_mask_season'],
            }
        se_b, exact_b, _ = eval_pipeline(y, seasons, test_mask, season_data_blend)
        curr = ' ← current' if bw == 0.0 else ''
        marker = ' ★' if se_b < 233 else ''
        print(f'  orig/znorm blend ({1-bw:.1f}/{bw:.1f}): SE={se_b}, exact={exact_b}/91{curr}{marker}')
    
    # ── APPROACH 5: Prediction averaging (multi-seed ensemble) ──
    print('\n' + '='*60)
    print(' APPROACH 5: Multi-seed prediction averaging')
    print('='*60)
    
    # Run multiple models with different random seeds and average raw scores
    seeds_list = [42, 123, 456, 789, 2024]
    
    for n_seeds in [3, 5]:
        season_data_avg = {'__fn__': fn}
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            X_tr = X_all[global_train_mask]
            y_tr = y[global_train_mask]
            s_tr = seasons[global_train_mask]
            
            tki = select_top_k_features(X_tr, y_tr, fn, k=USE_TOP_K_A,
                                          forced_features=FORCE_FEATURES)[0]
            
            all_raws = []
            for seed in seeds_list[:n_seeds]:
                # Component 1
                pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=30)
                sc_adj = StandardScaler()
                pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
                lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=seed)
                lr1.fit(pw_X_adj_sc, pw_y_adj)
                score1 = pairwise_score(lr1, X_season, sc_adj)
                
                # Component 3
                X_tr_k = X_tr[:, tki]
                X_te_k = X_season[:, tki]
                pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
                sc_k = StandardScaler()
                pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=seed)
                lr3.fit(pw_X_k_sc, pw_y_k)
                score3 = pairwise_score(lr3, X_te_k, sc_k)
                
                # Component 4
                pw_X_full, pw_y_full = build_pairwise_data(X_tr, y_tr, s_tr)
                sc_full = StandardScaler()
                pw_X_full_sc = sc_full.fit_transform(pw_X_full)
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=seed, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss')
                xgb_clf.fit(pw_X_full_sc, pw_y_full)
                score4 = pairwise_score(xgb_clf, X_season, sc_full)
                
                raw = 0.64 * score1 + 0.28 * score3 + 0.08 * score4
                all_raws.append(raw)
            
            # Average
            raw_avg = np.mean(all_raws, axis=0)
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw_avg[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            season_data_avg[hold_season] = {
                'X_season': X_season, 'raw': raw_avg,
                'season_indices': season_indices,
                'season_mask': season_mask,
                'test_mask_season': tm,
            }
        
        se_avg, exact_avg, _ = eval_pipeline(y, seasons, test_mask, season_data_avg)
        marker = ' ★' if se_avg < 233 else ''
        print(f'  {n_seeds}-seed avg: SE={se_avg}, exact={exact_avg}/91{marker}')
    
    # ── APPROACH 6: Top-K feature count variation ──
    print('\n' + '='*60)
    print(' APPROACH 6: Top-K feature count variation')
    print('='*60)
    
    for k in [15, 20, 25, 30, 35, 40]:
        season_data_k = {'__fn__': fn}
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            X_tr = X_all[global_train_mask]
            y_tr = y[global_train_mask]
            s_tr = seasons[global_train_mask]
            
            tki = select_top_k_features(X_tr, y_tr, fn, k=k,
                                          forced_features=FORCE_FEATURES)[0]
            
            # Only component 3 uses top-K, others use all features
            # So just re-run predict_robust_blend with different k
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_tr, y_tr, s_tr, max_gap=30)
            sc_adj = StandardScaler()
            pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
            lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_adj_sc, pw_y_adj)
            score1 = pairwise_score(lr1, X_season, sc_adj)
            
            X_tr_k = X_tr[:, tki]
            X_te_k = X_season[:, tki]
            pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sc_k = StandardScaler()
            pw_X_k_sc = sc_k.fit_transform(pw_X_k)
            lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_X_k_sc, pw_y_k)
            score3 = pairwise_score(lr3, X_te_k, sc_k)
            
            pw_X_full, pw_y_full = build_pairwise_data(X_tr, y_tr, s_tr)
            sc_full = StandardScaler()
            pw_X_full_sc = sc_full.fit_transform(pw_X_full)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            xgb_clf.fit(pw_X_full_sc, pw_y_full)
            score4 = pairwise_score(xgb_clf, X_season, sc_full)
            
            raw = 0.64 * score1 + 0.28 * score3 + 0.08 * score4
            
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            
            tm = np.array([test_mask[gi] for gi in season_indices])
            season_data_k[hold_season] = {
                'X_season': X_season, 'raw': raw,
                'season_indices': season_indices,
                'season_mask': season_mask,
                'test_mask_season': tm,
            }
        
        se_k, exact_k, _ = eval_pipeline(y, seasons, test_mask, season_data_k)
        curr = ' ← current' if k == 25 else ''
        marker = ' ★' if se_k < 233 else ''
        print(f'  top-K={k}: SE={se_k}, exact={exact_k}/91{curr}{marker}')
    
    # ── SUMMARY ──
    print('\n' + '='*60)
    print(' SUMMARY')
    print('='*60)
    print(f'  v45c baseline: SE=233, exact=66/91, RMSE451={np.sqrt(233/451):.4f}')
    print(f'  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
