#!/usr/bin/env python3
"""
v46 Smart Error Correction — Three approaches to fix remaining errors.

The remaining 25 errors (SE=233) fall into patterns:
1. BIG committee surprises (MurraySt=+8, NewMexico=-6, Northwestern=+5)
   → Committee valued team differently than any stat predicts
2. LOCAL swaps (SanDiegoSt↔Miami, Kentucky↔Wisconsin, etc.)
   → Teams very close in predicted quality, assigned to wrong neighbor

Smart approaches:
A. STACKED META-LEARNER: Use v45c predictions + features → second-stage model
B. RESIDUAL CORRECTION: Learn to predict v45c errors from features
C. PREDICTION SHRINKAGE: Blend v45c toward conf-bid historical mean
D. SOFT HUNGARIAN with confidence: Weight cost by prediction confidence
E. POST-HOC SWAP OPTIMIZATION: After assignment, try all pairwise swaps
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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


def get_v45c_predictions(X_all, y, seasons, test_mask, fn):
    """Run v45c pipeline and return raw scores + predictions per season."""
    folds = sorted(set(seasons))
    n = len(y)
    preds = np.zeros(n, dtype=int)
    raw_all = np.zeros(n)
    
    season_data = {}
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
            if not test_mask[gi]:
                raw[i] = y[gi]
        
        tm = np.array([test_mask[gi] for gi in season_indices])
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(raw, seasons[season_mask], avail, power=0.15)
        assigned = apply_zones(assigned, raw, fn, X_season, tm, season_indices, ZONES, 0.15)
        
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                preds[gi] = assigned[i]
                raw_all[gi] = raw[i]
        
        season_data[hold_season] = {
            'X_season': X_season, 'raw': raw,
            'season_indices': season_indices, 'season_mask': season_mask,
            'test_mask_season': tm,
        }
    
    return preds, raw_all, season_data


def compute_se(preds, y, test_mask):
    gt = y[test_mask].astype(int)
    pr = preds[test_mask]
    return int(np.sum((pr - gt)**2))


def main():
    t0 = time.time()
    print('='*60)
    print(' v46 SMART ERROR CORRECTION')
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
    
    # Get labeled data (seasons we know answers for)
    conferences = labeled['Conference'].fillna('Unknown').values
    bid_types = labeled['Bid Type'].fillna('Unknown').values
    
    print('\n  Computing v45c base predictions...')
    v45c_preds, v45c_raw, season_data = get_v45c_predictions(X_all, y, seasons, test_mask, fn)
    v45c_se = compute_se(v45c_preds, y, test_mask)
    print(f'  v45c baseline: SE={v45c_se}')
    
    # ── DEEP ERROR ANALYSIS ──
    print('\n' + '='*60)
    print(' ERROR ANALYSIS — What do the errors have in common?')
    print('='*60)
    
    gt = y[test_mask].astype(int)
    pr = v45c_preds[test_mask]
    rr = v45c_raw[test_mask]
    errors = pr - gt
    rids = record_ids[test_mask]
    confs = conferences[test_mask]
    bids = bid_types[test_mask]
    X_test = X_all[test_mask]
    fi = {f: i for i, f in enumerate(fn)}
    
    print(f'\n  {"Team":<28} {"Conf":<12} {"Bid":<4} {"GT":>3} {"Pred":>4} {"Raw":>6} {"NET":>4} {"SOS":>4} {"Err":>4}')
    print(f'  {"─"*28} {"─"*12} {"─"*4} {"─"*3} {"─"*4} {"─"*6} {"─"*4} {"─"*4} {"─"*4}')
    
    err_idx = np.where(errors != 0)[0]
    err_data = []
    for i in sorted(err_idx, key=lambda x: -errors[x]**2):
        net = X_test[i, fi['NET Rank']]
        sos = X_test[i, fi['NETSOS']]
        cb_mean = X_test[i, fi['cb_mean_seed']]
        tfr = X_test[i, fi['tourn_field_rank']]
        print(f'  {rids[i]:<28} {confs[i]:<12} {bids[i]:<4} {gt[i]:3d} {pr[i]:4d} {rr[i]:6.1f} '
              f'{net:4.0f} {sos:4.0f} {errors[i]:+4d}')
        err_data.append({
            'rid': rids[i], 'gt': gt[i], 'pred': pr[i], 'raw': rr[i],
            'err': errors[i], 'net': net, 'sos': sos, 'conf': confs[i],
            'bid': bids[i], 'cb_mean': cb_mean, 'tfr': tfr
        })
    
    # Check: is raw score closer to GT than assigned pred?
    print(f'\n  Raw vs Assigned comparison:')
    print(f'  {"Team":<28} {"GT":>3} {"Raw":>6} {"Pred":>4} {"RawErr":>7} {"PredErr":>8}')
    for ed in err_data:
        raw_err = ed['raw'] - ed['gt']
        pred_err = ed['pred'] - ed['gt']
        better = '← raw' if abs(raw_err) < abs(pred_err) else '← pred' if abs(pred_err) < abs(raw_err) else ''
        print(f'  {ed["rid"]:<28} {ed["gt"]:3d} {ed["raw"]:6.1f} {ed["pred"]:4d} {raw_err:+7.1f} {pred_err:+8d} {better}')
    
    # ── APPROACH A: STACKED META-LEARNER (Nested LOSO) ──
    print('\n' + '='*60)
    print(' APPROACH A: STACKED META-LEARNER')
    print(' (Use v12 raw + features → Ridge → refined raw → Hungarian)')
    print('='*60)
    
    # For each outer fold, train a meta-learner on inner folds' predictions
    # The meta-learner takes (v12_raw_score, features) and predicts seed
    
    for alpha in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        for meta_blend in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            preds_meta = np.zeros(n, dtype=int)
            
            for hold_season in folds:
                season_mask = (seasons == hold_season)
                season_test_mask = test_mask & season_mask
                if season_test_mask.sum() == 0: continue
                
                if hold_season not in season_data: continue
                sd = season_data[hold_season]
                
                # Training data for meta-learner: all OTHER seasons' teams
                train_seasons = [s for s in folds if s != hold_season]
                meta_train_mask = np.zeros(n, dtype=bool)
                for s in train_seasons:
                    meta_train_mask |= (seasons == s)
                meta_train_mask &= test_mask  # only tournament teams with GT
                
                # But we also need v45c raw scores for those teams, which we have
                meta_X_train = np.column_stack([
                    v45c_raw[meta_train_mask].reshape(-1, 1),
                    X_all[meta_train_mask]
                ])
                meta_y_train = y[meta_train_mask]
                
                # Scale
                sc = StandardScaler()
                meta_X_train_sc = sc.fit_transform(meta_X_train)
                
                # Train meta-learner
                ridge = Ridge(alpha=alpha)
                ridge.fit(meta_X_train_sc, meta_y_train)
                
                # Predict for this season's test teams
                X_season = sd['X_season']
                raw_orig = sd['raw'].copy()
                
                meta_X_test = np.column_stack([
                    raw_orig.reshape(-1, 1),
                    X_season
                ])
                meta_X_test_sc = sc.transform(meta_X_test)
                meta_pred = ridge.predict(meta_X_test_sc)
                
                # Blend: mostly original, some meta
                raw_refined = (1 - meta_blend) * raw_orig + meta_blend * meta_pred
                
                # Lock training teams
                for i, gi in enumerate(sd['season_indices']):
                    if not test_mask[gi]:
                        raw_refined[i] = y[gi]
                
                # Re-assign via Hungarian + zones
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(raw_refined, seasons[sd['season_mask']], avail, power=0.15)
                assigned = apply_zones(assigned, raw_refined, fn, X_season,
                                        sd['test_mask_season'], sd['season_indices'], ZONES, 0.15)
                
                for i, gi in enumerate(sd['season_indices']):
                    if test_mask[gi]:
                        preds_meta[gi] = assigned[i]
            
            se = compute_se(preds_meta, y, test_mask)
            exact = int((preds_meta[test_mask] == gt).sum())
            if se < v45c_se:
                print(f'  ★ α={alpha:5.1f}, blend={meta_blend:.2f}: SE={se:4d}, exact={exact}/91 (-{v45c_se-se})')
    
    print(f'  (if no ★, no config beat v45c)')
    
    # ── APPROACH B: RESIDUAL CORRECTION ──
    print('\n' + '='*60)
    print(' APPROACH B: RESIDUAL CORRECTION')
    print(' (Learn residuals from features, apply as post-hoc shift)')
    print('='*60)
    
    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
        for shrink in [0.1, 0.2, 0.3, 0.5]:
            preds_resid = np.zeros(n, dtype=int)
            
            for hold_season in folds:
                season_mask = (seasons == hold_season)
                season_test_mask = test_mask & season_mask
                if season_test_mask.sum() == 0: continue
                if hold_season not in season_data: continue
                sd = season_data[hold_season]
                
                # Training: compute residuals on other seasons
                train_seasons = [s for s in folds if s != hold_season]
                meta_train_mask = np.zeros(n, dtype=bool)
                for s in train_seasons:
                    meta_train_mask |= (seasons == s)
                meta_train_mask &= test_mask
                
                residuals = y[meta_train_mask] - v45c_preds[meta_train_mask]
                
                sc = StandardScaler()
                X_train_sc = sc.fit_transform(X_all[meta_train_mask])
                
                ridge = Ridge(alpha=alpha)
                ridge.fit(X_train_sc, residuals)
                
                # Predict residuals for this season's test teams
                X_season = sd['X_season']
                X_season_sc = sc.transform(X_season)
                pred_residuals = ridge.predict(X_season_sc)
                
                # Apply shrunken residual to raw scores
                raw_corrected = sd['raw'].copy()
                for i, gi in enumerate(sd['season_indices']):
                    if test_mask[gi]:
                        raw_corrected[i] -= shrink * pred_residuals[i]
                    else:
                        raw_corrected[i] = y[gi]
                
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(raw_corrected, seasons[sd['season_mask']], avail, power=0.15)
                assigned = apply_zones(assigned, raw_corrected, fn, X_season,
                                        sd['test_mask_season'], sd['season_indices'], ZONES, 0.15)
                
                for i, gi in enumerate(sd['season_indices']):
                    if test_mask[gi]:
                        preds_resid[gi] = assigned[i]
            
            se = compute_se(preds_resid, y, test_mask)
            exact = int((preds_resid[test_mask] == gt).sum())
            if se < v45c_se:
                print(f'  ★ α={alpha:5.1f}, shrink={shrink:.2f}: SE={se:4d}, exact={exact}/91 (-{v45c_se-se})')
    
    print(f'  (if no ★, no config beat v45c)')
    
    # ── APPROACH C: CONF-BID SHRINKAGE ──
    print('\n' + '='*60)
    print(' APPROACH C: SHRINKAGE TOWARD CONFERENCE-BID HISTORICAL MEAN')
    print(' (Blend prediction toward what committee historically gives this conf/bid combo)')
    print('='*60)
    
    for shrink_w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        preds_shrink = np.zeros(n, dtype=int)
        
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            if hold_season not in season_data: continue
            sd = season_data[hold_season]
            
            # Historical conf-bid means (from other seasons only!)
            train_seasons = [s for s in folds if s != hold_season]
            hist_mask = np.zeros(n, dtype=bool)
            for s in train_seasons:
                hist_mask |= (seasons == s)
            
            cb_means = {}
            for i in np.where(hist_mask)[0]:
                key = (conferences[i], bid_types[i])
                cb_means.setdefault(key, []).append(y[i])
            cb_mean_seeds = {k: np.mean(v) for k, v in cb_means.items()}
            
            raw_shrunk = sd['raw'].copy()
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    key = (conferences[gi], bid_types[gi])
                    hist_mean = cb_mean_seeds.get(key, raw_shrunk[i])
                    raw_shrunk[i] = (1 - shrink_w) * raw_shrunk[i] + shrink_w * hist_mean
                else:
                    raw_shrunk[i] = y[gi]
            
            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(raw_shrunk, seasons[sd['season_mask']], avail, power=0.15)
            assigned = apply_zones(assigned, raw_shrunk, fn, sd['X_season'],
                                    sd['test_mask_season'], sd['season_indices'], ZONES, 0.15)
            
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    preds_shrink[gi] = assigned[i]
        
        se = compute_se(preds_shrink, y, test_mask)
        exact = int((preds_shrink[test_mask] == gt).sum())
        marker = ' ★' if se < v45c_se else ''
        print(f'  shrink={shrink_w:.2f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── APPROACH D: POST-HOC PAIRWISE SWAP OPTIMIZATION ──
    print('\n' + '='*60)
    print(' APPROACH D: POST-HOC SWAP (try all 2-team swaps)')
    print(' (After v45c assignment, test all pairwise swaps using a learned cost)')
    print('='*60)
    
    # For each season, after v45c assignment, try swapping every pair of test teams
    # Keep the swap if a learned model says it's better
    
    for alpha in [1.0, 10.0, 50.0, 100.0]:
        preds_swap = np.zeros(n, dtype=int)
        total_swaps = 0
        
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            if hold_season not in season_data: continue
            sd = season_data[hold_season]
            
            # Train a seed predictor on other seasons
            train_mask = np.zeros(n, dtype=bool)
            for s in folds:
                if s != hold_season:
                    train_mask |= (seasons == s)
            train_mask &= test_mask
            
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_all[train_mask])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_sc, y[train_mask])
            
            # Direct predictions for this season
            X_season = sd['X_season']
            direct_pred = ridge.predict(sc.transform(X_season))
            
            # Get v45c assignments
            v45c_season = np.zeros(len(sd['season_indices']), dtype=int)
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    v45c_season[i] = v45c_preds[gi]
                else:
                    v45c_season[i] = int(y[gi])
            
            # Try pairwise swaps among test teams
            test_in_season = [(i, gi) for i, gi in enumerate(sd['season_indices']) if test_mask[gi]]
            best_assigned = v45c_season.copy()
            
            improved = True
            while improved:
                improved = False
                for a_idx in range(len(test_in_season)):
                    for b_idx in range(a_idx + 1, len(test_in_season)):
                        i_a, gi_a = test_in_season[a_idx]
                        i_b, gi_b = test_in_season[b_idx]
                        
                        seed_a = best_assigned[i_a]
                        seed_b = best_assigned[i_b]
                        
                        # Current cost (using direct prediction as "ideal")
                        cost_now = (direct_pred[i_a] - seed_a)**2 + (direct_pred[i_b] - seed_b)**2
                        cost_swap = (direct_pred[i_a] - seed_b)**2 + (direct_pred[i_b] - seed_a)**2
                        
                        if cost_swap < cost_now - 0.01:
                            best_assigned[i_a] = seed_b
                            best_assigned[i_b] = seed_a
                            improved = True
                            total_swaps += 1
            
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    preds_swap[gi] = best_assigned[i]
        
        se = compute_se(preds_swap, y, test_mask)
        exact = int((preds_swap[test_mask] == gt).sum())
        marker = ' ★' if se < v45c_se else ''
        print(f'  α={alpha:5.1f}: SE={se:4d}, exact={exact}/91, {total_swaps} swaps{marker}')
    
    # ── APPROACH E: SOFT HUNGARIAN (weighted cost) ──
    print('\n' + '='*60)
    print(' APPROACH E: CONFIDENCE-WEIGHTED HUNGARIAN')
    print(' (Weight cost matrix by prediction confidence)')
    print('='*60)
    
    for conf_power in [0.5, 1.0, 1.5, 2.0]:
        preds_conf = np.zeros(n, dtype=int)
        
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            if hold_season not in season_data: continue
            sd = season_data[hold_season]
            
            # Train a second model to estimate confidence (variance proxy)
            train_mask = np.zeros(n, dtype=bool)
            for s in folds:
                if s != hold_season:
                    train_mask |= (seasons == s)
            train_mask &= test_mask
            
            # Confidence = inverse of absolute residual from Ridge prediction
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_all[train_mask])
            ridge = Ridge(alpha=10.0)
            ridge.fit(X_tr_sc, y[train_mask])
            
            raw = sd['raw'].copy()
            X_season = sd['X_season']
            ridge_pred = ridge.predict(sc.transform(X_season))
            
            # Blend ridge and pairwise predictions
            raw_blend = 0.9 * raw + 0.1 * ridge_pred
            
            # Lock training
            for i, gi in enumerate(sd['season_indices']):
                if not test_mask[gi]:
                    raw_blend[i] = y[gi]
            
            # Confidence-weighted Hungarian:
            # For teams where v12 and ridge agree → high confidence → sharper cost
            # For teams where they disagree → low confidence → flatter cost
            si = [i for i, gi in enumerate(sd['season_indices'])]
            pos = list(range(1, 69))
            
            disagreement = np.abs(raw - ridge_pred)
            max_dis = disagreement.max() + 1e-6
            confidence = 1.0 - (disagreement / max_dis)  # 0 to 1
            
            # Build cost matrix with confidence weighting
            cost = np.zeros((len(si), len(pos)))
            for i_idx, i in enumerate(si):
                conf = confidence[i] ** conf_power
                for j, p in enumerate(pos):
                    cost[i_idx, j] = conf * abs(raw_blend[i] - p)**0.15
            
            ri, ci = linear_sum_assignment(cost)
            assigned = np.zeros(len(si), dtype=int)
            for r, c in zip(ri, ci):
                assigned[r] = pos[c]
            
            # Apply zones
            assigned = apply_zones(assigned, raw_blend, fn, X_season,
                                    sd['test_mask_season'], sd['season_indices'], ZONES, 0.15)
            
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    preds_conf[gi] = assigned[i]
        
        se = compute_se(preds_conf, y, test_mask)
        exact = int((preds_conf[test_mask] == gt).sum())
        marker = ' ★' if se < v45c_se else ''
        print(f'  conf_power={conf_power:.1f}: SE={se:4d}, exact={exact}/91{marker}')
    
    # ── APPROACH F: ENSEMBLE OF DIFFERENT ZONE ORDERINGS ──
    print('\n' + '='*60)
    print(' APPROACH F: RAW SCORE AVERAGING (multiple zone orderings)')
    print(' (Average raw scores from different zone application orders)')
    print('='*60)
    
    # Different zone orderings can produce different raw → seed mappings
    # Average the resulting integer predictions (as raw → multiple Hungarians → average)
    
    import itertools
    
    zone_perms = list(itertools.permutations(range(5)))[:20]  # first 20 permutations
    
    preds_ensemble = np.zeros(n, dtype=float)
    n_perms_used = 0
    
    for perm in zone_perms:
        reordered_zones = [ZONES[i] for i in perm]
        preds_perm = np.zeros(n, dtype=int)
        
        for hold_season in folds:
            if hold_season not in season_data: continue
            sd = season_data[hold_season]
            
            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(sd['raw'], seasons[sd['season_mask']], avail, power=0.15)
            assigned = apply_zones(assigned, sd['raw'], fn, sd['X_season'],
                                    sd['test_mask_season'], sd['season_indices'],
                                    reordered_zones, 0.15)
            
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    preds_perm[gi] = assigned[i]
        
        preds_ensemble += preds_perm
        n_perms_used += 1
    
    # Average and round
    preds_ensemble /= n_perms_used
    preds_avg_int = np.round(preds_ensemble).astype(int)
    
    # But we need valid 1-68 assignment — use Hungarian on averaged scores
    preds_final_ensemble = np.zeros(n, dtype=int)
    for hold_season in folds:
        if hold_season not in season_data: continue
        sd = season_data[hold_season]
        season_avg = preds_ensemble[sd['season_indices']]
        
        # Lock training teams
        for i, gi in enumerate(sd['season_indices']):
            if not test_mask[gi]:
                season_avg[i] = y[gi]
        
        avail = {hold_season: list(range(1, 69))}
        assigned = hungarian(season_avg, seasons[sd['season_mask']], avail, power=0.15)
        
        for i, gi in enumerate(sd['season_indices']):
            if test_mask[gi]:
                preds_final_ensemble[gi] = assigned[i]
    
    se = compute_se(preds_final_ensemble, y, test_mask)
    exact = int((preds_final_ensemble[test_mask] == gt).sum())
    marker = ' ★' if se < v45c_se else ''
    print(f'  {n_perms_used}-perm ensemble: SE={se:4d}, exact={exact}/91{marker}')
    
    # Also test: does the original zone ordering matter? Test all 120 permutations
    print(f'\n  Testing all zone orderings individually:')
    best_perm_se = v45c_se
    best_perm = None
    perm_results = {}
    for perm in itertools.permutations(range(5)):
        reordered = [ZONES[i] for i in perm]
        preds_p = np.zeros(n, dtype=int)
        for hold_season in folds:
            if hold_season not in season_data: continue
            sd = season_data[hold_season]
            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(sd['raw'], seasons[sd['season_mask']], avail, power=0.15)
            assigned = apply_zones(assigned, sd['raw'], fn, sd['X_season'],
                                    sd['test_mask_season'], sd['season_indices'],
                                    reordered, 0.15)
            for i, gi in enumerate(sd['season_indices']):
                if test_mask[gi]:
                    preds_p[gi] = assigned[i]
        se_p = compute_se(preds_p, y, test_mask)
        perm_results[perm] = se_p
        if se_p < best_perm_se:
            best_perm_se = se_p
            best_perm = perm
    
    perm_ses = list(perm_results.values())
    print(f'  120 permutations: min_SE={min(perm_ses)}, max_SE={max(perm_ses)}, '
          f'mean_SE={np.mean(perm_ses):.0f}, std_SE={np.std(perm_ses):.1f}')
    if best_perm_se < v45c_se:
        print(f'  ★ Best perm {best_perm}: SE={best_perm_se}')
    else:
        distinct = len(set(perm_ses))
        print(f'  {distinct} distinct SE values across permutations')
        for se_val in sorted(set(perm_ses)):
            cnt = perm_ses.count(se_val)
            print(f'    SE={se_val}: {cnt} permutations')
    
    # ── SUMMARY ──
    print('\n' + '='*60)
    print(' SUMMARY')
    print('='*60)
    print(f'  v45c baseline: SE={v45c_se}, RMSE451={np.sqrt(v45c_se/451):.4f}')
    print(f'  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
