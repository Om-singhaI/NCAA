#!/usr/bin/env python3
"""Quick LOSO validation for the Kaggle-winning v5 config."""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian,
    USE_TOP_K_A, PW_C1, PW_C3, BLEND_W1, BLEND_W2, BLEND_W3
)

# v5 winning config
V5_W_LR5  = 0.60
V5_W_XGB  = 0.20
V5_W_LRK  = 0.20
V5_POWER  = 0.5

def main():
    t0 = time.time()
    print('='*60)
    print(' LOSO VALIDATION: v4 vs v5 Winning Config')
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
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)
    
    # Test both configs
    for cfg_name, w_lr5, w_lr001, w_lrk, w_xgb, power in [
        ('v4 (60/10/30, p=1.0)',  0.60, 0.10, 0.30, 0.00, 1.0),
        ('v5 (60/0/20+20%XGB, p=0.5)', V5_W_LR5, 0.00, V5_W_LRK, V5_W_XGB, V5_POWER),
        ('v5 at p=0.75',         V5_W_LR5, 0.00, V5_W_LRK, V5_W_XGB, 0.75),
        ('v5 at p=1.0',          V5_W_LR5, 0.00, V5_W_LRK, V5_W_XGB, 1.0),
        ('v4 at p=0.5',          0.60, 0.10, 0.30, 0.00, 0.5),
    ]:
        assigned = np.zeros(n, dtype=int)
        fold_stats = []
        
        for hold in folds:
            tr = seasons != hold
            te = seasons == hold
            
            top_k_idx = select_top_k_features(X[tr], y[tr], fn, k=USE_TOP_K_A)[0]
            
            # Build pairwise data
            pw_X, pw_y = build_pairwise_data(X[tr], y[tr], seasons[tr])
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)
            
            blended = np.zeros(int(te.sum()))
            
            # PW-LR C=5.0
            if w_lr5 > 0:
                lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
                lr1.fit(pw_X_sc, pw_y)
                blended += w_lr5 * pairwise_score(lr1, X[te], sc)
            
            # PW-LR C=0.01
            if w_lr001 > 0:
                lr2 = LogisticRegression(C=0.01, penalty='l2', max_iter=2000, random_state=42)
                lr2.fit(pw_X_sc, pw_y)
                blended += w_lr001 * pairwise_score(lr2, X[te], sc)
            
            # PW-LR topK
            if w_lrk > 0:
                X_k_tr = X[tr][:, top_k_idx]
                X_k_te = X[te][:, top_k_idx]
                pw_X_k, pw_y_k = build_pairwise_data(X_k_tr, y[tr], seasons[tr])
                sc_k = StandardScaler()
                pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
                lr3.fit(pw_X_k_sc, pw_y_k)
                blended += w_lrk * pairwise_score(lr3, X_k_te, sc_k)
            
            # PW-XGB
            if w_xgb > 0:
                xgb_clf = xgb.XGBClassifier(
                    n_estimators=200, max_depth=3, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=42, verbosity=0, use_label_encoder=False,
                    eval_metric='logloss')
                xgb_clf.fit(pw_X_sc, pw_y)
                blended += w_xgb * pairwise_score(xgb_clf, X[te], sc)
            
            avail = {hold: list(range(1, 69))}
            a = hungarian(blended, seasons[te], avail, power=power)
            assigned[te] = a
            yte = y[te].astype(int)
            ex = int((a == yte).sum())
            rm = np.sqrt(np.mean((a - yte)**2))
            fold_stats.append((hold, int(te.sum()), ex, rm))
        
        total_exact = int((assigned == y.astype(int)).sum())
        total_rmse = np.sqrt(np.mean((assigned - y.astype(int))**2))
        rho, _ = spearmanr(assigned, y.astype(int))
        fold_rmses = [s[3] for s in fold_stats]
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        score = mean_rmse + 0.5 * std_rmse
        
        print(f'\n  --- {cfg_name} ---')
        print(f'  {"Season":>10} {"N":>3} {"Exact":>5} {"Pct":>6} {"RMSE":>8}')
        for s, n_f, ex, rm in fold_stats:
            print(f'  {s:>10} {n_f:3d} {ex:5d} {ex/n_f*100:5.1f}% {rm:8.3f}')
        print(f'  TOTAL: {total_exact}/{n} exact ({total_exact/n*100:.1f}%), '
              f'RMSE={total_rmse:.4f}, ρ={rho:.4f}')
        print(f'  Score: {score:.4f} (μ={mean_rmse:.4f}, σ={std_rmse:.4f})')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
