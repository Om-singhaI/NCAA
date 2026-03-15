#!/usr/bin/env python3
"""
NCAA v9 — Novel Approaches (not parameter tuning!)
====================================================
v7 failed: MLP overfit local eval. v8 failed: all-LR overfit LOSO.
Both were just parameter/model tuning on the same framework.

v9 tries STRUCTURALLY different ideas:

 1. RECENCY-WEIGHTED TRAINING — weight recent seasons higher
    Rationale: 2025-26 will be most similar to 2024-25, not 2020-21
    
 2. FEATURE STABILITY SELECTION — only use features stable across ALL seasons
    Rationale: unstable features are noise, not signal
    
 3. MULTI-SEED-RANGE MODELS — separate models for 1-16, 17-32, 33-48, 49-68
    Rationale: what matters for a #1 seed is different from #60
    
 4. CONFIDENCE-WEIGHTED ASSIGNMENT — modify Hungarian power based on model confidence
    Rationale: when model is sure, use higher power; when uncertain, use lower
    
 5. LEAVE-TWO-OUT STACKING — train meta-model on genuine out-of-sample predictions
    Rationale: combine multiple models without overfitting to any one metric
    
 6. RANK AGGREGATION — Borda count / trimmed mean instead of score blending
    Rationale: different combination mechanism may be more robust

Evaluation: compute BOTH local Kaggle-style (locked seeds) AND LOSO.
v6 baseline: local=56/91 RMSE=2.474, LOSO=3.678
Only approaches that maintain or improve BOTH can be deployed.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, rankdata
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA_DIR)
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian
)


def compute_loso_score(fold_rmses):
    """LOSO score = mean + 0.5*std."""
    return np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)


def kaggle_eval(predict_fn, X_all, y, seasons, folds, test_mask,
                feature_names, power=0.15):
    """
    Kaggle-style eval: lock known training seeds, only predict test teams.
    Returns (exact, rmse, per_season_detail).
    """
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    
    for hold in folds:
        smask = (seasons == hold)
        test_in_season = test_mask & smask
        if test_in_season.sum() == 0:
            continue
        
        train_mask = ~test_in_season  # everything except this season's test
        X_season = X_all[smask]
        
        # Get pairwise scores for ALL teams in this season
        scores = predict_fn(X_all[train_mask], y[train_mask], X_season,
                          seasons[train_mask])
        
        # Lock training (non-test) teams to their known seeds
        season_idx = np.where(smask)[0]
        for i, gi in enumerate(season_idx):
            if not test_mask[gi]:
                scores[i] = y[gi]
        
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[smask], avail, power=power)
        
        for i, gi in enumerate(season_idx):
            if test_mask[gi]:
                test_assigned[gi] = assigned[i]
    
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    exact = int((pred == gt).sum())
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    return exact, rmse


def loso_eval(predict_fn, X_all, y, seasons, folds, power=0.15):
    """
    LOSO eval: hold out one full season at a time.
    Returns (loso_score, fold_rmses).
    """
    fold_rmses = []
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        
        scores = predict_fn(X_all[tr], y[tr], X_all[te], seasons[tr])
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[te], avail, power=power)
        rmse = np.sqrt(np.mean((assigned - y[te].astype(int)) ** 2))
        fold_rmses.append(rmse)
    
    return compute_loso_score(fold_rmses), fold_rmses


def main():
    t0 = time.time()
    print('=' * 60)
    print(' NCAA v9 — NOVEL STRUCTURAL APPROACHES')
    print(' (not parameter tuning)')
    print('=' * 60)
    
    # ── Load data ──
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    feat_A = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat_A.columns)
    n_feat = len(feature_names)
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat_A.values.astype(np.float64)), np.nan,
                     feat_A.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    # Test mask (Kaggle GT teams)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    print(f'  {n_labeled} teams, {n_feat} features, {len(folds)} folds')
    print(f'  Test teams: {test_mask.sum()}, Train teams: (~test_mask).sum()')
    
    # ────────────────────────────────────────────────────────
    #  BASELINE: v6
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' BASELINE: v6 (64%LR_C5 + 28%LRk25_C0.5 + 8%XGB_d4_300)')
    print('=' * 60)
    
    def v6_predict(X_tr, y_tr, X_te, s_tr):
        top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
        
        pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        
        lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        s1 = pairwise_score(lr1, X_te, sc)
        
        X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sck = StandardScaler()
        pw_Xk_sc = sck.fit_transform(pw_Xk)
        lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_Xk_sc, pw_yk)
        s3 = pairwise_score(lr3, X_te_k, sck)
        
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
            reg_alpha=1.0, min_child_weight=5, random_state=42,
            verbosity=0, use_label_encoder=False, eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        s4 = pairwise_score(xgb_clf, X_te, sc)
        
        return 0.64 * s1 + 0.28 * s3 + 0.08 * s4
    
    v6_exact, v6_rmse = kaggle_eval(v6_predict, X_all, y, seasons, folds,
                                      test_mask, feature_names)
    v6_loso, v6_fold_rmses = loso_eval(v6_predict, X_all, y, seasons, folds)
    print(f'  Kaggle: {v6_exact}/91 exact, RMSE={v6_rmse:.4f}')
    print(f'  LOSO:   {v6_loso:.4f} [{" ".join(f"{r:.2f}" for r in v6_fold_rmses)}]')
    
    results = [('v6 baseline', v6_exact, v6_rmse, v6_loso, v6_fold_rmses)]
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 1: RECENCY-WEIGHTED PAIRWISE TRAINING
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 1: Recency-weighted training')
    print('=' * 60)
    
    # Weight recent pairwise examples more heavily via oversampling
    for decay_name, decay_map in [
        ('linear', {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0}),
        ('sqrt', {0: 1.0, 1: 1.4, 2: 1.7, 3: 2.0, 4: 2.2}),
        ('mild', {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.5, 4: 2.0}),
    ]:
        def make_recency_predict(dm):
            def recency_predict(X_tr, y_tr, X_te, s_tr):
                top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
                
                # Build pairwise with recency weights
                sorted_seasons = sorted(set(s_tr))
                season_rank = {s: i for i, s in enumerate(sorted_seasons)}
                
                pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
                sc = StandardScaler()
                pw_X_sc = sc.fit_transform(pw_X)
                
                # Create sample weights for pairwise data
                # Each season generates n*(n-1) pairs (both directions)
                # We need to map pair index back to season
                sample_w = np.ones(len(pw_y))
                pair_idx = 0
                for s in sorted_seasons:
                    idx = np.where(s_tr == s)[0]
                    n_s = len(idx)
                    n_pairs = n_s * (n_s - 1)  # both directions
                    w = dm.get(season_rank[s], 1.0)
                    sample_w[pair_idx:pair_idx + n_pairs] = w
                    pair_idx += n_pairs
                
                lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
                lr1.fit(pw_X_sc, pw_y, sample_weight=sample_w)
                s1 = pairwise_score(lr1, X_te, sc)
                
                X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
                pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
                sck = StandardScaler()
                pw_Xk_sc = sck.fit_transform(pw_Xk)
                
                sample_wk = np.ones(len(pw_yk))
                pair_idx = 0
                for s in sorted_seasons:
                    idx = np.where(s_tr == s)[0]
                    n_s = len(idx)
                    n_pairs = n_s * (n_s - 1)
                    w = dm.get(season_rank[s], 1.0)
                    sample_wk[pair_idx:pair_idx + n_pairs] = w
                    pair_idx += n_pairs
                
                lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
                lr3.fit(pw_Xk_sc, pw_yk, sample_weight=sample_wk)
                s3 = pairwise_score(lr3, X_te_k, sck)
                
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                    reg_alpha=1.0, min_child_weight=5, random_state=42,
                    verbosity=0, use_label_encoder=False, eval_metric='logloss')
                xgb_clf.fit(pw_X_sc, pw_y, sample_weight=sample_w)
                s4 = pairwise_score(xgb_clf, X_te, sc)
                
                return 0.64 * s1 + 0.28 * s3 + 0.08 * s4
            return recency_predict
        
        pred_fn = make_recency_predict(decay_map)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'Recency-{decay_name}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 2: FEATURE STABILITY SELECTION
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 2: Feature stability selection')
    print('=' * 60)
    
    # Find features that rank consistently important across all 5 LOSO folds
    fold_importances = []
    for hold in folds:
        tr = seasons != hold
        X_tr, y_tr = X_all[tr], y[tr]
        
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_tr)
        ridge = Ridge(alpha=5.0)
        ridge.fit(X_sc, y_tr)
        ridge_imp = np.abs(ridge.coef_)
        
        rf = RandomForestRegressor(n_estimators=500, max_depth=10,
                                    min_samples_leaf=2, max_features=0.5,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        rf_imp = rf.feature_importances_
        
        xgb_m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   min_child_weight=3, reg_lambda=3.0,
                                   reg_alpha=1.0, random_state=42, verbosity=0)
        xgb_m.fit(X_tr, y_tr)
        xgb_imp = xgb_m.feature_importances_
        
        avg_rank = (rankdata(-ridge_imp) + rankdata(-rf_imp) + rankdata(-xgb_imp)) / 3
        fold_importances.append(avg_rank)
    
    # Stability: mean rank - std of ranks (lower = more consistently important)
    fold_imp_array = np.array(fold_importances)  # (5, n_feat)
    mean_rank = fold_imp_array.mean(axis=0)
    std_rank = fold_imp_array.std(axis=0)
    stability_score = mean_rank + std_rank  # penalize inconsistency
    
    # Try different numbers of stable features
    for n_stable in [15, 20, 25, 30, 35, 40]:
        stable_idx = np.argsort(stability_score)[:n_stable]
        stable_names = [feature_names[i] for i in stable_idx]
        
        def make_stable_predict(sidx, ns):
            def stable_predict(X_tr, y_tr, X_te, s_tr):
                # Use only stable features for a simpler model
                X_tr_s = X_tr[:, sidx]
                X_te_s = X_te[:, sidx]
                
                pw_X, pw_y = build_pairwise_data(X_tr_s, y_tr, s_tr)
                sc = StandardScaler()
                pw_X_sc = sc.fit_transform(pw_X)
                
                lr = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
                lr.fit(pw_X_sc, pw_y)
                s1 = pairwise_score(lr, X_te_s, sc)
                
                # Also do full-feature LR to maintain diversity
                pw_Xf, pw_yf = build_pairwise_data(X_tr, y_tr, s_tr)
                scf = StandardScaler()
                pw_Xf_sc = scf.fit_transform(pw_Xf)
                lrf = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
                lrf.fit(pw_Xf_sc, pw_yf)
                sf = pairwise_score(lrf, X_te, scf)
                
                # XGB on stable features (less overfit risk)
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                    reg_alpha=1.0, min_child_weight=5, random_state=42,
                    verbosity=0, use_label_encoder=False, eval_metric='logloss')
                xgb_clf.fit(pw_X_sc, pw_y)
                sx = pairwise_score(xgb_clf, X_te_s, sc)
                
                # v6-like weights
                return 0.64 * sf + 0.28 * s1 + 0.08 * sx
            return stable_predict
        
        pred_fn = make_stable_predict(stable_idx, n_stable)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'Stable-{n_stable}feat'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    if n_stable == 25:
        print(f'    Top-25 stable features: {stable_names[:10]}...')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 3: RANK AGGREGATION (Borda / trimmed mean)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 3: Rank aggregation (Borda / trimmed mean)')
    print('=' * 60)
    
    def make_rank_agg_predict(method='borda'):
        def rank_agg_predict(X_tr, y_tr, X_te, s_tr):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
            
            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            
            # Get individual model rankings
            lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_sc, pw_y)
            s1 = pairwise_score(lr1, X_te, sc)
            
            lr1b = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
            lr1b.fit(pw_X_sc, pw_y)
            s1b = pairwise_score(lr1b, X_te, sc)
            
            lr1c = LogisticRegression(C=10.0, penalty='l2', max_iter=2000, random_state=42)
            lr1c.fit(pw_X_sc, pw_y)
            s1c = pairwise_score(lr1c, X_te, sc)
            
            X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_Xk_sc, pw_yk)
            s3 = pairwise_score(lr3, X_te_k, sck)
            
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                reg_alpha=1.0, min_child_weight=5, random_state=42,
                verbosity=0, use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(pw_X_sc, pw_y)
            s4 = pairwise_score(xgb_clf, X_te, sc)
            
            all_scores = np.array([s1, s1b, s1c, s3, s4])  # (5, n_test)
            
            if method == 'borda':
                # Borda count: average ranks
                return np.mean(all_scores, axis=0)
            elif method == 'trimmed':
                # Trimmed mean: drop min/max rank per team, average rest
                n_models = all_scores.shape[0]
                trimmed = np.zeros(all_scores.shape[1])
                for j in range(all_scores.shape[1]):
                    ranks = all_scores[:, j]
                    sorted_r = np.sort(ranks)
                    trimmed[j] = np.mean(sorted_r[1:-1])  # drop highest and lowest
                return trimmed
            elif method == 'median':
                return np.median(all_scores, axis=0)
        return rank_agg_predict
    
    for method in ['borda', 'trimmed', 'median']:
        pred_fn = make_rank_agg_predict(method)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'RankAgg-{method}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 4: MULTI-SEED-RANGE MODELS
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 4: Multi-seed-range models')
    print('=' * 60)
    
    def seedrange_predict(X_tr, y_tr, X_te, s_tr):
        """Two-stage: first classify into seed groups, then rank within."""
        top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
        
        # First: get coarse rankings from v6-style model
        pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        
        lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        s1 = pairwise_score(lr1, X_te, sc)
        
        X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sck = StandardScaler()
        pw_Xk_sc = sck.fit_transform(pw_Xk)
        lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_Xk_sc, pw_yk)
        s3 = pairwise_score(lr3, X_te_k, sck)
        
        coarse = 0.7 * s1 + 0.3 * s3
        
        # Second: train specialized models for top/mid/bottom seeds
        # High seeds (1-16): need precise differentiation  
        # Use stronger regularization (C=1) to avoid overfitting the small group
        top_mask_tr = y_tr <= 20
        if top_mask_tr.sum() >= 20:
            pw_Xt, pw_yt = build_pairwise_data(X_tr[top_mask_tr], y_tr[top_mask_tr], s_tr[top_mask_tr])
            sct = StandardScaler()
            pw_Xt_sc = sct.fit_transform(pw_Xt)
            lrt = LogisticRegression(C=1.0, penalty='l2', max_iter=2000, random_state=42)
            lrt.fit(pw_Xt_sc, pw_yt)
            
            # Get scores for test teams that coarse model ranks in top ~20
            top_test_idx = np.argsort(coarse)[:20]
            if len(top_test_idx) > 1:
                X_top = X_te[top_test_idx]
                scores_top = pairwise_score(lrt, X_top, sct)
                # Blend coarse and fine rankings for top teams
                for i, ti in enumerate(top_test_idx):
                    coarse[ti] = 0.7 * coarse[ti] + 0.3 * scores_top[i]
        
        return coarse
    
    ex, rmse = kaggle_eval(seedrange_predict, X_all, y, seasons, folds,
                           test_mask, feature_names)
    loso, fr = loso_eval(seedrange_predict, X_all, y, seasons, folds)
    tag = 'SeedRange-2stage'
    results.append((tag, ex, rmse, loso, fr))
    d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
    d_loso = "↑" if loso < v6_loso else "↓"
    print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
          f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 5: MULTI-SEED ENSEMBLE (multiple random seeds)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 5: Multi-seed ensemble (variance reduction)')
    print('=' * 60)
    
    def make_multiseed_predict(n_seeds=5):
        def multiseed_predict(X_tr, y_tr, X_te, s_tr):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
            
            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            
            X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            
            all_s1, all_s3, all_s4 = [], [], []
            for seed in range(n_seeds):
                rs = 42 + seed * 1000
                lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=rs)
                lr1.fit(pw_X_sc, pw_y)
                all_s1.append(pairwise_score(lr1, X_te, sc))
                
                lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=rs)
                lr3.fit(pw_Xk_sc, pw_yk)
                all_s3.append(pairwise_score(lr3, X_te_k, sck))
                
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                    reg_alpha=1.0, min_child_weight=5, random_state=rs,
                    verbosity=0, use_label_encoder=False, eval_metric='logloss')
                xgb_clf.fit(pw_X_sc, pw_y)
                all_s4.append(pairwise_score(xgb_clf, X_te, sc))
            
            s1 = np.mean(all_s1, axis=0)
            s3 = np.mean(all_s3, axis=0)
            s4 = np.mean(all_s4, axis=0)
            
            return 0.64 * s1 + 0.28 * s3 + 0.08 * s4
        return multiseed_predict
    
    for n_seeds in [3, 5, 10]:
        pred_fn = make_multiseed_predict(n_seeds)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'MultiSeed-{n_seeds}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 6: DIFFERENT C VALUES (diversified regularization)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 6: Diversified regularization blend')
    print('=' * 60)
    
    # Instead of v6's C combo, try blending many C values for robustness
    def make_divreg_predict(c_list, weights):
        def divreg_predict(X_tr, y_tr, X_te, s_tr):
            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            
            blended = np.zeros(len(X_te))
            for C, w in zip(c_list, weights):
                lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
                lr.fit(pw_X_sc, pw_y)
                blended += w * pairwise_score(lr, X_te, sc)
            return blended
        return divreg_predict
    
    diverse_configs = [
        ('DivReg-3C-equal',  [0.5, 5.0, 50.0], [1/3, 1/3, 1/3]),
        ('DivReg-5C-equal',  [0.1, 0.5, 2.0, 5.0, 20.0], [0.2]*5),
        ('DivReg-3C-center', [1.0, 5.0, 10.0], [0.25, 0.50, 0.25]),
        ('DivReg-wide5',     [0.01, 0.1, 1.0, 10.0, 100.0], [0.2]*5),
    ]
    
    for tag, c_list, weights in diverse_configs:
        pred_fn = make_divreg_predict(c_list, weights)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 7: BOOTSTRAP AGGREGATION (Bagging pairwise models)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 7: Bagging pairwise models')
    print('=' * 60)
    
    def make_bagged_predict(n_bags=10, sample_frac=0.8):
        def bagged_predict(X_tr, y_tr, X_te, s_tr):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
            
            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            n_pairs = len(pw_y)
            
            X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            
            all_scores = []
            for bag in range(n_bags):
                rng = np.random.RandomState(42 + bag)
                idx = rng.choice(n_pairs, int(n_pairs * sample_frac), replace=True)
                
                lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
                lr1.fit(pw_X_sc[idx], pw_y[idx])
                s1 = pairwise_score(lr1, X_te, sc)
                
                lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
                lr3.fit(pw_Xk_sc[idx], pw_yk[idx])
                s3 = pairwise_score(lr3, X_te_k, sck)
                
                all_scores.append(0.64 * s1 + 0.28 * s3)
            
            # Add XGB without bagging (too slow)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                reg_alpha=1.0, min_child_weight=5, random_state=42,
                verbosity=0, use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(pw_X_sc, pw_y)
            s4 = pairwise_score(xgb_clf, X_te, sc)
            
            bagged = np.mean(all_scores, axis=0) + 0.08 * s4
            return bagged
        return bagged_predict
    
    for n_bags in [5, 10, 20]:
        pred_fn = make_bagged_predict(n_bags)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'Bagged-{n_bags}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 8: POWER SWEEP (v6 model, different powers)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 8: v6 model with different Hungarian powers')
    print('=' * 60)
    
    for power in [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.50]:
        ex, rmse = kaggle_eval(v6_predict, X_all, y, seasons, folds,
                               test_mask, feature_names, power=power)
        loso, fr = loso_eval(v6_predict, X_all, y, seasons, folds, power=power)
        tag = f'v6-power={power:.3f}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        marker = " ★" if d_kaggle == "↑" and d_loso == "↑" else ""
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}{marker}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 9: BLEND v6 + DIRECT REGRESSION
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 9: Blend pairwise v6 with direct regression')
    print('=' * 60)
    
    def make_hybrid_predict(pw_weight=0.85):
        def hybrid_predict(X_tr, y_tr, X_te, s_tr):
            # Pairwise component (v6)
            pw_scores = v6_predict(X_tr, y_tr, X_te, s_tr)
            
            # Direct regression component (XGB+Ridge)
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            
            ridge = Ridge(alpha=5.0)
            ridge.fit(X_tr_sc, y_tr)
            ridge_pred = ridge.predict(X_te_sc)
            
            xgb_preds = []
            for seed in [42, 123, 777]:
                m = xgb.XGBRegressor(
                    n_estimators=700, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
                m.fit(X_tr, y_tr)
                xgb_preds.append(m.predict(X_te))
            xgb_avg = np.mean(xgb_preds, axis=0)
            
            direct = 0.7 * xgb_avg + 0.3 * ridge_pred
            
            # Blend
            return pw_weight * pw_scores + (1 - pw_weight) * direct
        return hybrid_predict
    
    for pw_w in [0.95, 0.90, 0.85, 0.80, 0.70]:
        pred_fn = make_hybrid_predict(pw_w)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'Hybrid-pw{int(pw_w*100)}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        marker = " ★" if d_kaggle == "↑" and d_loso == "↑" else ""
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}{marker}')
    
    # ────────────────────────────────────────────────────────
    #  APPROACH 10: FEATURE NOISE INJECTION (regularization via noise)
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' APPROACH 10: Feature noise injection (implicit regularization)')
    print('=' * 60)
    
    def make_noisy_predict(noise_std=0.1, n_reps=5):
        def noisy_predict(X_tr, y_tr, X_te, s_tr):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=25)[0]
            all_scores = []
            
            for rep in range(n_reps):
                rng = np.random.RandomState(42 + rep)
                # Add small noise to training features before pairwise
                noise_tr = X_tr + rng.normal(0, noise_std, X_tr.shape)
                
                pw_X, pw_y = build_pairwise_data(noise_tr, y_tr, s_tr)
                sc = StandardScaler()
                pw_X_sc = sc.fit_transform(pw_X)
                
                lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
                lr1.fit(pw_X_sc, pw_y)
                s1 = pairwise_score(lr1, X_te, sc)
                
                X_tr_k, X_te_k = noise_tr[:, top_k_idx], X_te[:, top_k_idx]
                pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
                sck = StandardScaler()
                pw_Xk_sc = sck.fit_transform(pw_Xk)
                lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
                lr3.fit(pw_Xk_sc, pw_yk)
                s3 = pairwise_score(lr3, X_te_k, sck)
                
                all_scores.append(0.64 * s1 + 0.28 * s3)
            
            # XGB without noise
            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                reg_alpha=1.0, min_child_weight=5, random_state=42,
                verbosity=0, use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(pw_X_sc, pw_y)
            s4 = pairwise_score(xgb_clf, X_te, sc)
            
            return np.mean(all_scores, axis=0) + 0.08 * s4
        return noisy_predict
    
    for noise_std in [0.05, 0.1, 0.2]:
        pred_fn = make_noisy_predict(noise_std)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds,
                               test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'Noisy-σ={noise_std}'
        results.append((tag, ex, rmse, loso, fr))
        d_kaggle = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_loso = "↑" if loso < v6_loso else "↓"
        marker = " ★" if d_kaggle == "↑" and d_loso == "↑" else ""
        print(f'  {tag:30s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_kaggle}  '
              f'LOSO={loso:.4f} {d_loso}{marker}')
    
    # ────────────────────────────────────────────────────────
    #  FINAL SUMMARY
    # ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY — ALL APPROACHES')
    print('=' * 60)
    
    # Sort by combined rank of kaggle and loso
    for i, (tag, ex, rmse, loso, fr) in enumerate(results):
        results[i] = (tag, ex, rmse, loso, fr,
                      ex > v6_exact or (ex == v6_exact and rmse < v6_rmse),  # better kaggle
                      loso < v6_loso)  # better loso
    
    # Stars: ★★ = both improved, ★ = one improved, blank = neither
    print(f'\n  {"Approach":<32} {"Kaggle":>10} {"RMSE":>8} {"LOSO":>8} {"Status":>8}')
    print(f'  {"─"*32} {"─"*10} {"─"*8} {"─"*8} {"─"*8}')
    
    for tag, ex, rmse, loso, fr, kg_up, lo_up in results:
        if kg_up and lo_up:
            status = '★★'
        elif kg_up or lo_up:
            status = '★'
        else:
            status = ''
        bg = '→' if tag == 'v6 baseline' else ' '
        print(f' {bg}{tag:<32} {ex:2d}/91  {rmse:8.4f} {loso:8.4f} {status:>8}')
    
    # Identify winners
    print('\n  Candidates beating v6 on BOTH metrics:')
    both_better = [(tag, ex, rmse, loso) for tag, ex, rmse, loso, fr, kg, lo
                   in results if kg and lo]
    if both_better:
        for tag, ex, rmse, loso in sorted(both_better, key=lambda x: x[2]):
            print(f'    {tag}: Kaggle={ex}/91 RMSE={rmse:.4f} LOSO={loso:.4f}')
    else:
        print('    NONE — v6 remains optimal')
        print('\n  Closest approaches (better on ONE metric):')
        one_better = [(tag, ex, rmse, loso) for tag, ex, rmse, loso, fr, kg, lo
                      in results if kg or lo]
        for tag, ex, rmse, loso in sorted(one_better, key=lambda x: x[2])[:5]:
            print(f'    {tag}: Kaggle={ex}/91 RMSE={rmse:.4f} LOSO={loso:.4f}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
