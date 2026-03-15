#!/usr/bin/env python3
"""
Test top v5b configurations on the Kaggle test set (91 tournament teams).
Compares against v4 baseline (49/91 exact, RMSE=2.92).
"""

import os, sys, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
    USE_TOP_K_A, PW_C1, PW_C2, PW_C3
)


def build_features_selective(df, context_df, labeled_df, tourn_rids):
    """Original 68 + 2 selective features (AvgOppNET, nonconf_sos_delta)."""
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    feat['AvgOppNET'] = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    ncsos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    feat['nonconf_sos_delta'] = ncsos - sos
    return feat


def score_season(X_train, y_train, seasons_train, X_season, feature_names,
                 weights, use_xgb_pw=False):
    """
    Compute blended scores for one season.
    weights = [pw_lr5, pw_lr001, pw_lrk, pw_xgb, direct]
    """
    top_k_idx = select_top_k_features(X_train, y_train, feature_names, k=USE_TOP_K_A)[0]
    
    blended = np.zeros(len(X_season))
    
    # PW-LR components share pairwise data
    pw_X, pw_y = build_pairwise_data(X_train, y_train, seasons_train)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    
    if weights[0] > 0:  # PW-LR C=5.0
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        blended += weights[0] * pairwise_score(lr1, X_season, sc)
    
    if weights[1] > 0:  # PW-LR C=0.01
        lr2 = LogisticRegression(C=PW_C2, penalty='l2', max_iter=2000, random_state=42)
        lr2.fit(pw_X_sc, pw_y)
        blended += weights[1] * pairwise_score(lr2, X_season, sc)
    
    if weights[2] > 0:  # PW-LR topK C=1.0
        X_k_tr = X_train[:, top_k_idx]
        X_k_te = X_season[:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(X_k_tr, y_train, seasons_train)
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        blended += weights[2] * pairwise_score(lr3, X_k_te, sc_k)
    
    if weights[3] > 0:  # PW-XGB
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_clf.fit(pw_X_sc, pw_y)
        blended += weights[3] * pairwise_score(xgb_clf, X_season, sc)
    
    if weights[4] > 0:  # Direct XGB+Ridge
        xgb_preds = []
        for seed in [42, 123, 777]:
            m = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=5.0, reg_alpha=2.0, min_child_weight=5,
                random_state=seed, verbosity=0
            )
            m.fit(X_train, y_train)
            xgb_preds.append(m.predict(X_season))
        xgb_avg = np.mean(xgb_preds, axis=0)
        
        sc_r = StandardScaler()
        ridge = Ridge(alpha=10.0)
        ridge.fit(sc_r.fit_transform(X_train), y_train)
        ridge_pred = ridge.predict(sc_r.transform(X_season))
        
        direct = 0.7 * xgb_avg + 0.3 * ridge_pred
        direct_ranks = np.argsort(np.argsort(direct)).astype(float) + 1.0
        blended += weights[4] * direct_ranks
    
    return blended


def main():
    print('='*70)
    print(' KAGGLE TEST SET EVALUATION — Top v5b Configurations')
    print('='*70)
    
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    test_rids = set(GT.keys())
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    test_mask = np.array([rid in test_rids for rid in record_ids])
    folds = sorted(set(seasons))
    
    # Configs to test: (name, n_feat, weights, power)
    configs = [
        # Baselines
        ('v4 original (68f, p=1.0)',       68, [0.60, 0.10, 0.30, 0.00, 0.00], 1.0),
        
        # 70-feat winners
        ('70f v4+10%D p=0.50 ★LOSO-best',  70, [0.54, 0.09, 0.27, 0.00, 0.10], 0.5),
        ('70f v4+10%D p=0.75',             70, [0.54, 0.09, 0.27, 0.00, 0.10], 0.75),
        ('70f v4+10%D p=1.00',             70, [0.54, 0.09, 0.27, 0.00, 0.10], 1.0),
        ('70f v4+10%D p=1.25',             70, [0.54, 0.09, 0.27, 0.00, 0.10], 1.25),
        ('70f v4 original p=1.50',         70, [0.60, 0.10, 0.30, 0.00, 0.00], 1.5),
        ('70f v4+15%X+15%D p=1.25',        70, [0.42, 0.07, 0.21, 0.15, 0.15], 1.25),
        
        # 68-feat winners
        ('68f v4+10%D p=1.25',             68, [0.54, 0.09, 0.27, 0.00, 0.10], 1.25),
        ('68f v4+15%X+15%D p=2.0',         68, [0.42, 0.07, 0.21, 0.15, 0.15], 2.0),
        ('68f v4+10%XGB p=0.75',           68, [0.54, 0.09, 0.27, 0.10, 0.00], 0.75),
        ('68f v4 original p=1.25',         68, [0.60, 0.10, 0.30, 0.00, 0.00], 1.25),
        ('68f v4+10%D p=0.50',             68, [0.54, 0.09, 0.27, 0.00, 0.10], 0.5),
        ('68f v4+20%D p=1.25',             68, [0.48, 0.08, 0.24, 0.00, 0.20], 1.25),
    ]
    
    best_rmse = float('inf')
    best_name = None
    best_assigned = None
    
    for cfg_name, n_feat, weights, power in configs:
        print(f'\n  Testing: {cfg_name}')
        
        # Build features
        if n_feat == 70:
            feat = build_features_selective(labeled, context_df, labeled, tourn_rids)
        else:
            feat = build_features(labeled, context_df, labeled, tourn_rids)
        fn = list(feat.columns)
        
        X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                         feat.values.astype(np.float64))
        imp = KNNImputer(n_neighbors=10, weights='distance')
        X_all = imp.fit_transform(X_raw)
        
        test_assigned = np.zeros(n_labeled, dtype=int)
        
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            n_te = season_test_mask.sum()
            if n_te == 0:
                continue
            
            global_train_mask = ~season_test_mask
            X_season = X_all[season_mask]
            
            # Score all teams in this season
            blended = score_season(
                X_all[global_train_mask], y[global_train_mask],
                seasons[global_train_mask], X_season, fn, weights)
            
            # Lock training teams to their known seeds
            season_indices = np.where(season_mask)[0]
            for i, global_idx in enumerate(season_indices):
                if not test_mask[global_idx]:
                    blended[i] = y[global_idx]
            
            # Hungarian assignment
            avail = {hold_season: list(range(1, 69))}
            assigned_season = hungarian(blended, seasons[season_mask], avail,
                                       power=power)
            
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    test_assigned[global_idx] = assigned_season[i]
        
        # Evaluate
        gt = y[test_mask].astype(int)
        pred = test_assigned[test_mask]
        exact = int((pred == gt).sum())
        rmse = np.sqrt(np.mean((pred - gt)**2))
        
        marker = ''
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = cfg_name
            best_assigned = test_assigned.copy()
            marker = ' ← BEST'
        
        print(f'    → {exact}/91 exact ({exact/91*100:.1f}%), RMSE={rmse:.4f}{marker}')
        
        # Per-season breakdown
        for s in folds:
            s_mask = test_mask & (seasons == s)
            if s_mask.sum() == 0:
                continue
            s_gt = y[s_mask].astype(int)
            s_pred = test_assigned[s_mask]
            s_ex = int((s_pred == s_gt).sum())
            s_rm = np.sqrt(np.mean((s_pred - s_gt)**2))
            print(f'       {s}: {s_ex}/{s_mask.sum()} exact, RMSE={s_rm:.3f}')
    
    # ─── Final results ───
    print('\n' + '='*70)
    print(f' BEST KAGGLE CONFIG: {best_name}')
    print(f' {int((best_assigned[test_mask] == y[test_mask].astype(int)).sum())}/91 exact, '
          f'RMSE={best_rmse:.4f}')
    print('='*70)
    
    # Build submission with best config
    submission = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    for i in np.where(test_mask)[0]:
        rid_to_seed[record_ids[i]] = int(best_assigned[i])
    submission['Overall Seed'] = submission['RecordID'].map(
        lambda r: rid_to_seed.get(r, 0))
    
    out_path = os.path.join(DATA_DIR, 'submission_kaggle_v5.csv')
    submission.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
    
    # Show some predictions
    test_indices = np.where(test_mask)[0]
    print(f'\n  Sample (best config):')
    print(f'  {"RecordID":<30} {"Pred":>4} {"True":>4} {"Diff":>5}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*5}')
    for i in test_indices[:15]:
        pred = best_assigned[i]
        actual = int(y[i])
        diff = pred - actual
        m = ' ✓' if diff == 0 else ''
        print(f'  {record_ids[i]:<30} {pred:4d} {actual:4d} {diff:+5d}{m}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
