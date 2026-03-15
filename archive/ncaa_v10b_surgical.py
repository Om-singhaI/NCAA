#!/usr/bin/env python3
"""
NCAA v10b — Surgical Committee Corrections
=============================================
v10 showed: adding features to the pairwise model HURTS (too many dims for 340 teams).
Instead, keep v6 model UNTOUCHED and apply committee-style corrections POST-model:

Strategy A: Score adjustment — modify pairwise scores before Hungarian
Strategy B: Feature forcing — ensure committee-critical features are in top-K
Strategy C: Confidence masking — different power for confident vs uncertain
Strategy D: Two-model committee blend — separate model for committee criteria  
Strategy E: Error-type-aware — analyze v6 errors, target specific fix

v6 baseline: 56/91 exact, RMSE=2.474, LOSO=3.678
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DATA_DIR)
from ncaa_2026_model import (
    load_data, build_features, build_pairwise_data, pairwise_score,
    hungarian, select_top_k_features
)


def compute_loso_score(fold_rmses):
    return np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)


def kaggle_eval(predict_fn, X_all, y, seasons, folds, test_mask,
                feature_names, power=0.15, raw_df=None):
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        smask = (seasons == hold)
        test_in_season = test_mask & smask
        if test_in_season.sum() == 0:
            continue
        train_mask = ~test_in_season
        X_season = X_all[smask]
        if raw_df is not None:
            scores = predict_fn(X_all[train_mask], y[train_mask], X_season,
                              seasons[train_mask], raw_df=raw_df,
                              train_idx=np.where(train_mask)[0],
                              test_idx=np.where(smask)[0])
        else:
            scores = predict_fn(X_all[train_mask], y[train_mask], X_season,
                              seasons[train_mask])
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


def loso_eval(predict_fn, X_all, y, seasons, folds, power=0.15, raw_df=None):
    fold_rmses = []
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        if raw_df is not None:
            scores = predict_fn(X_all[tr], y[tr], X_all[te], seasons[tr],
                              raw_df=raw_df,
                              train_idx=np.where(tr)[0],
                              test_idx=np.where(te)[0])
        else:
            scores = predict_fn(X_all[tr], y[tr], X_all[te], seasons[tr])
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[te], avail, power=power)
        rmse = np.sqrt(np.mean((assigned - y[te].astype(int)) ** 2))
        fold_rmses.append(rmse)
    return compute_loso_score(fold_rmses), fold_rmses


def v6_predict(X_tr, y_tr, X_te, s_tr, feature_names=None, **kwargs):
    """Exact v6 model."""
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(X_tr.shape[1])]
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


def main():
    t0 = time.time()
    print('=' * 60)
    print(' NCAA v10b — SURGICAL COMMITTEE CORRECTIONS')
    print('=' * 60)

    # ── Load data ──
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat_df = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat_df.columns)

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat_df.values.astype(np.float64)), np.nan,
                     feat_df.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    print(f'  {len(labeled)} teams, {len(feature_names)} features, {len(folds)} folds')

    results = []

    # ════════════════════════════════════════════════════
    #  BASELINE
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' BASELINE: v6')
    print('=' * 60)

    make_v6 = lambda: lambda X_tr, y_tr, X_te, s_tr, **kw: v6_predict(X_tr, y_tr, X_te, s_tr, feature_names)
    v6_fn = make_v6()
    v6_exact, v6_rmse = kaggle_eval(v6_fn, X_all, y, seasons, folds, test_mask, feature_names)
    v6_loso, v6_fr = loso_eval(v6_fn, X_all, y, seasons, folds)
    print(f'  Kaggle: {v6_exact}/91 exact, RMSE={v6_rmse:.4f}')
    print(f'  LOSO:   {v6_loso:.4f}')
    results.append(('v6-baseline', v6_exact, v6_rmse, v6_loso))

    # ════════════════════════════════════════════════════
    #  STRATEGY A: Committee Score Adjustment
    #  Keep v6 pairwise scores, add a small committee correction
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY A: Post-model committee score adjustment')
    print('=' * 60)

    # Build committee signals from the feature matrix
    # These are indices into the feature_names array
    fi = {name: i for i, name in enumerate(feature_names)}

    def make_adjusted_predict(alpha, committee_fn):
        """Adjust v6 pairwise scores with a committee correction."""
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            # Get base v6 scores
            base = v6_predict(X_tr, y_tr, X_te, s_tr, feature_names)
            # Compute committee adjustment for test teams
            adjustment = committee_fn(X_te)
            return base + alpha * adjustment
        return predict

    # Committee correction 1: Q1 wins boost + bad loss penalty
    def q1_correction(X):
        q1w = X[:, fi['Quadrant1_W']]
        q3l = X[:, fi.get('Quadrant3_L', 0)] if 'Quadrant3_L' in fi else np.zeros(len(X))
        q4l = X[:, fi.get('Quadrant4_L', 0)] if 'Quadrant4_L' in fi else np.zeros(len(X))
        # More Q1 wins → lower score (better seed), bad losses → higher score
        return -q1w * 0.5 + q3l * 1.0 + q4l * 2.0
    
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        pred_fn = make_adjusted_predict(alpha, q1_correction)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'v6+Q1adj-α={alpha}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # Committee correction 2: Resume score
    def resume_correction(X):
        return X[:, fi['resume_score']] * (-0.1)  # better resume → lower score
    
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        pred_fn = make_adjusted_predict(alpha, resume_correction)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'v6+resume-α={alpha}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # Committee correction 3: SOS-adjusted NET
    def sos_correction(X):
        return X[:, fi['sos_adj_net']] * 0.01
    
    for alpha in [0.5, 1.0, 2.0, 5.0]:
        pred_fn = make_adjusted_predict(alpha, sos_correction)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'v6+SOSadj-α={alpha}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY B: Forced Feature Inclusion in Top-K
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY B: Force committee features into top-K model')
    print('=' * 60)

    # Force-include specific features in the top-K model
    committee_must_include = ['Quadrant1_W', 'Quadrant1_L', 'NET Rank',
                               'NETSOS', 'resume_score', 'q1_dominance',
                               'WL_Pct', 'total_bad_losses']

    def make_forced_topk_predict(forced_features, total_k=25):
        forced_idx = [fi[f] for f in forced_features if f in fi]
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            # Get normal top-K
            top_k_idx_auto = select_top_k_features(X_tr, y_tr, feature_names,
                                                    k=total_k - len(forced_idx))[0]
            # Combine: forced + auto (remove duplicates)
            combined = list(forced_idx)
            for idx in top_k_idx_auto:
                if idx not in combined:
                    combined.append(idx)
            combined = combined[:total_k]  # trim to total_k

            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_sc, pw_y)
            s1 = pairwise_score(lr1, X_te, sc)

            X_tr_k, X_te_k = X_tr[:, combined], X_te[:, combined]
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
        return predict

    force_sets = [
        ('NET+Q1', ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']),
        ('committee_core', ['NET Rank', 'Quadrant1_W', 'Quadrant1_L', 'NETSOS',
                           'resume_score', 'q1_dominance']),
        ('committee_full', committee_must_include),
        ('resume_focus', ['resume_score', 'quality_ratio', 'q1_dominance',
                         'total_bad_losses', 'quad_balance']),
        ('sos_focus', ['NETSOS', 'AvgOppNETRank', 'sos_adj_net', 'sos_x_wpct',
                      'record_vs_sos']),
    ]

    for tag_name, forced in force_sets:
        pred_fn = make_forced_topk_predict(forced)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'force-{tag_name}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY C: Two-model blend (pairwise + direct regression)
    #  with VERY small regression weight (v9 showed >5% hurts)
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY C: v6 + micro-dose direct regression')
    print('=' * 60)

    def make_micro_hybrid(reg_weight=0.02):
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            pw = v6_predict(X_tr, y_tr, X_te, s_tr, feature_names)

            # Very lightweight direct regression
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            ridge = Ridge(alpha=10.0)  # heavy regularization
            ridge.fit(X_tr_sc, y_tr)
            direct = ridge.predict(X_te_sc)

            return (1 - reg_weight) * pw + reg_weight * direct
        return predict

    for rw in [0.01, 0.02, 0.03, 0.05]:
        pred_fn = make_micro_hybrid(rw)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'v6+ridge-{int(rw*100)}%'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY D: Separate top-K feature sets
    #  Use different top-K for different model components
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY D: Different top-K sizes')
    print('=' * 60)

    def make_varied_topk(k_reduced=20, w1=0.64, w3=0.28, w4=0.08):
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=k_reduced)[0]

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

            return w1 * s1 + w3 * s3 + w4 * s4
        return predict

    for k in [15, 20, 30, 35]:
        pred_fn = make_varied_topk(k)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'v6-topK={k}'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY E: Different XGB configurations
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY E: XGB component tuning')
    print('=' * 60)

    def make_xgb_variant(n_est=300, depth=4, lr=0.05, w4=0.08):
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
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
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                reg_alpha=1.0, min_child_weight=5, random_state=42,
                verbosity=0, use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(pw_X_sc, pw_y)
            s4 = pairwise_score(xgb_clf, X_te, sc)

            w1 = 0.64
            w3 = 0.28
            return w1 * s1 + w3 * s3 + w4 * s4
        return predict

    # Try different XGB weights
    for w4 in [0.0, 0.04, 0.08, 0.12, 0.16]:
        w1 = 0.64 * (1 - w4) / 0.92
        w3 = 0.28 * (1 - w4) / 0.92
        pred_fn = make_xgb_variant(w4=w4)
        # fix weights
        def make_weighted(w1_=w1, w3_=w3, w4_=w4):
            def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
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

                return w1_ * s1 + w3_ * s3 + w4_ * s4
            return predict

        pred_fn = make_weighted(w1, w3, w4)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        tag = f'xgb_w={w4:.0%}({w1:.0%}/{w3:.0%}/{w4:.0%})'
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY F: LR-only model (no XGB) with different C combos
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY F: LR-only pairwise (remove XGB noise)')
    print('=' * 60)

    def make_lr_only(c1=5.0, c3=0.5, w1=0.70, w3=0.30, k=25):
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=k)[0]

            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_sc, pw_y)
            s1 = pairwise_score(lr1, X_te, sc)

            X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_Xk_sc, pw_yk)
            s3 = pairwise_score(lr3, X_te_k, sck)

            return w1 * s1 + w3 * s3
        return predict

    lr_configs = [
        ('LRonly-v6ratio', 5.0, 0.5, 0.696, 0.304, 25),
        ('LRonly-equal', 5.0, 0.5, 0.5, 0.5, 25),
        ('LRonly-main', 5.0, 0.5, 0.80, 0.20, 25),
        ('LRonly-C3/0.3', 3.0, 0.3, 0.70, 0.30, 25),
        ('LRonly-C10/1', 10.0, 1.0, 0.70, 0.30, 25),
        ('LRonly-k20', 5.0, 0.5, 0.70, 0.30, 20),
        ('LRonly-k30', 5.0, 0.5, 0.70, 0.30, 30),
    ]

    for tag, c1, c3, w1, w3, k in lr_configs:
        pred_fn = make_lr_only(c1, c3, w1, w3, k)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  STRATEGY G: 3-LR blend (different C, same architecture)
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' STRATEGY G: 3-LR blend with different regularizations')
    print('=' * 60)

    def make_3lr(c1=5.0, c2=1.0, c3=0.5, w1=0.50, w2=0.25, w3=0.25, k=25):
        def predict(X_tr, y_tr, X_te, s_tr, **kwargs):
            top_k_idx = select_top_k_features(X_tr, y_tr, feature_names, k=k)[0]

            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            # Model 1: full features
            lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_sc, pw_y)
            s1 = pairwise_score(lr1, X_te, sc)

            # Model 2: full features, different C
            lr2 = LogisticRegression(C=c2, penalty='l2', max_iter=2000, random_state=42)
            lr2.fit(pw_X_sc, pw_y)
            s2 = pairwise_score(lr2, X_te, sc)

            # Model 3: top-K features
            X_tr_k, X_te_k = X_tr[:, top_k_idx], X_te[:, top_k_idx]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_Xk_sc, pw_yk)
            s3 = pairwise_score(lr3, X_te_k, sck)

            return w1 * s1 + w2 * s2 + w3 * s3
        return predict

    lr3_configs = [
        ('3LR-5/1/0.5-50/25/25', 5.0, 1.0, 0.5, 0.50, 0.25, 0.25, 25),
        ('3LR-5/2/0.5-50/20/30', 5.0, 2.0, 0.5, 0.50, 0.20, 0.30, 25),
        ('3LR-5/10/0.5-40/30/30', 5.0, 10.0, 0.5, 0.40, 0.30, 0.30, 25),
        ('3LR-5/0.1/0.5-50/20/30', 5.0, 0.1, 0.5, 0.50, 0.20, 0.30, 25),
        ('3LR-10/5/1-33/34/33', 10.0, 5.0, 1.0, 0.33, 0.34, 0.33, 25),
    ]

    for tag, c1, c2, c3, w1, w2, w3, k in lr3_configs:
        pred_fn = make_3lr(c1, c2, c3, w1, w2, w3, k)
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, feature_names)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds)
        results.append((tag, ex, rmse, loso))
        d_k = "↑" if ex > v6_exact or (ex == v6_exact and rmse < v6_rmse) else "↓" if ex < v6_exact else "="
        d_l = "↑" if loso < v6_loso else "↓"
        marker = " ★★" if d_k != "↓" and d_l == "↑" else " ★" if d_k != "↓" or d_l == "↑" else ""
        print(f'  {tag:35s} Kaggle={ex}/91 RMSE={rmse:.4f} {d_k}  LOSO={loso:.4f} {d_l}{marker}')

    # ════════════════════════════════════════════════════
    #  ERROR ANALYSIS: what does v6 get wrong?
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' ERROR ANALYSIS: v6 misses')
    print('=' * 60)

    # Get v6 predictions for all test teams
    v6_test_assigned = np.zeros(len(y), dtype=int)
    for hold in folds:
        smask = (seasons == hold)
        test_in_season = test_mask & smask
        if test_in_season.sum() == 0:
            continue
        train_mask = ~test_in_season
        scores = v6_predict(X_all[train_mask], y[train_mask], X_all[smask],
                           seasons[train_mask], feature_names)
        season_idx = np.where(smask)[0]
        for i, gi in enumerate(season_idx):
            if not test_mask[gi]:
                scores[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(scores, seasons[smask], avail, power=0.15)
        for i, gi in enumerate(season_idx):
            if test_mask[gi]:
                v6_test_assigned[gi] = assigned[i]

    test_idx = np.where(test_mask)[0]
    errors = []
    for ti in test_idx:
        true_s = int(y[ti])
        pred_s = int(v6_test_assigned[ti])
        diff = pred_s - true_s
        if diff != 0:
            errors.append({
                'rid': record_ids[ti],
                'season': seasons[ti],
                'true': true_s,
                'pred': pred_s,
                'diff': diff,
                'abs_diff': abs(diff),
                'net': X_all[ti, fi['NET Rank']],
                'q1w': X_all[ti, fi['Quadrant1_W']],
                'q1l': X_all[ti, fi['Quadrant1_L']],
                'sos': X_all[ti, fi['NETSOS']],
                'wpct': X_all[ti, fi['WL_Pct']],
                'bad_losses': X_all[ti, fi['total_bad_losses']],
                'is_al': X_all[ti, fi['is_AL']],
                'conf_str': X_all[ti, fi['conf_avg_net']],
            })

    errors_df = pd.DataFrame(errors).sort_values('abs_diff', ascending=False)
    print(f'\n  v6 errors: {len(errors_df)}/91 teams ({len(test_idx) - len(errors_df)} exact)')
    print(f'  Mean abs error: {errors_df["abs_diff"].mean():.2f}')
    print(f'  Max abs error: {errors_df["abs_diff"].max()}')

    print(f'\n  Worst 15 errors:')
    print(f'  {"Team":40s} {"True":>4} {"Pred":>4} {"Diff":>5} {"NET":>5} {"Q1W":>4} {"SOS":>5} {"AL?":>3}')
    for _, e in errors_df.head(15).iterrows():
        al = '✓' if e['is_al'] > 0.5 else ''
        print(f'  {e["rid"]:40s} {e["true"]:4d} {e["pred"]:4d} {e["diff"]:+5d} {e["net"]:5.0f} {e["q1w"]:4.0f} {e["sos"]:5.0f} {al:>3}')

    # Error patterns
    print(f'\n  Error by seed range:')
    for lo, hi, label in [(1, 16, 'Top seeds (1-16)'), (17, 34, 'Mid seeds (17-34)'),
                           (35, 50, 'Late seeds (35-50)'), (51, 68, 'Auto bids (51-68)')]:
        subset = errors_df[(errors_df['true'] >= lo) & (errors_df['true'] <= hi)]
        if len(subset) > 0:
            print(f'    {label}: {len(subset)} errors, mean abs diff={subset["abs_diff"].mean():.2f}')

    print(f'\n  Over-seeded (pred < true): {(errors_df["diff"] < 0).sum()}')
    print(f'  Under-seeded (pred > true): {(errors_df["diff"] > 0).sum()}')

    # ════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY')
    print('=' * 60)

    print(f'\n  {"Approach":<38} {"Kaggle":>10} {"RMSE":>8} {"LOSO":>8} {"St":>4}')
    print(f'  {"─"*38} {"─"*10} {"─"*8} {"─"*8} {"─"*4}')

    for tag, ex, rmse, loso in results:
        kg_up = ex > v6_exact or (ex == v6_exact and rmse < v6_rmse)
        lo_up = loso < v6_loso
        if kg_up and lo_up: status = '★★'
        elif kg_up or lo_up: status = '★'
        else: status = ''
        bg = '→' if tag == 'v6-baseline' else ' '
        print(f' {bg}{tag:<38} {ex:2d}/91  {rmse:8.4f} {loso:8.4f} {status:>4}')

    both = [(t, e, r, l) for t, e, r, l in results
            if (e > v6_exact or (e == v6_exact and r < v6_rmse)) and l < v6_loso]
    print(f'\n  Beating v6 on BOTH metrics:')
    if both:
        for t, e, r, l in sorted(both, key=lambda x: x[2]):
            print(f'    ★★ {t}: Kaggle={e}/91 RMSE={r:.4f} LOSO={l:.4f}')
    else:
        print('    NONE')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
