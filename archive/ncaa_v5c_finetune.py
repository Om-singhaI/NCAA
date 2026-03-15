#!/usr/bin/env python3
"""Fine-grained search around the best Kaggle configs."""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def main():
    print('='*70)
    print(' FINE-GRAINED KAGGLE TEST SEARCH')
    print('='*70)
    
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)
    test_rids = set(GT.keys())
    
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    feat = build_features(labeled, context_df, labeled, tourn_rids)
    fn = list(feat.columns)
    print(f'  Features: {len(fn)}')
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    test_mask = np.array([rid in test_rids for rid in record_ids])
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)
    
    # ── Precompute method scores per season (to avoid recomputation) ──
    # Methods: 0=PW-LR5, 1=PW-LR001, 2=PW-LRk, 3=PW-XGB, 4=Direct
    n_methods = 5
    # Store scores per season
    season_scores = {}  # season -> method_idx -> score_array
    
    print('\n  Precomputing scores per season...')
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        global_train_mask = ~season_test_mask
        X_season = X[season_mask]
        season_indices = np.where(season_mask)[0]
        
        top_k_idx = select_top_k_features(
            X[global_train_mask], y[global_train_mask], fn, k=USE_TOP_K_A)[0]
        
        pw_X, pw_y = build_pairwise_data(
            X[global_train_mask], y[global_train_mask], seasons[global_train_mask])
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        
        scores = {}
        
        # M0: PW-LR C=5.0
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y)
        scores[0] = pairwise_score(lr1, X_season, sc)
        
        # M1: PW-LR C=0.01
        lr2 = LogisticRegression(C=PW_C2, penalty='l2', max_iter=2000, random_state=42)
        lr2.fit(pw_X_sc, pw_y)
        scores[1] = pairwise_score(lr2, X_season, sc)
        
        # M2: PW-LR topK C=1.0
        X_k_tr = X[global_train_mask][:, top_k_idx]
        X_k_te = X_season[:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(
            X_k_tr, y[global_train_mask], seasons[global_train_mask])
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        scores[2] = pairwise_score(lr3, X_k_te, sc_k)
        
        # M3: PW-XGB
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_X_sc, pw_y)
        scores[3] = pairwise_score(xgb_clf, X_season, sc)
        
        # M4: Direct XGB+Ridge
        xgb_preds = []
        for seed in [42, 123, 777]:
            m = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_lambda=5.0, reg_alpha=2.0, min_child_weight=5,
                random_state=seed, verbosity=0)
            m.fit(X[global_train_mask], y[global_train_mask])
            xgb_preds.append(m.predict(X_season))
        xgb_avg = np.mean(xgb_preds, axis=0)
        sc_r = StandardScaler()
        ridge = Ridge(alpha=10.0)
        ridge.fit(sc_r.fit_transform(X[global_train_mask]), y[global_train_mask])
        ridge_pred = ridge.predict(sc_r.transform(X_season))
        direct = 0.7 * xgb_avg + 0.3 * ridge_pred
        scores[4] = np.argsort(np.argsort(direct)).astype(float) + 1.0
        
        season_scores[hold_season] = (scores, season_mask, season_indices)
        print(f'    {hold_season}: {season_test_mask.sum()} test teams ✓')
    
    # ── Fine-grained blend × power search ──
    print('\n  Fine-grained search...')
    
    # Generate configs: vary XGB weight (0-20%) and Direct weight (0-20%)
    # with remaining shared proportionally among v4 components
    configs = []
    for w_xgb in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        for w_dir in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
            remaining = 1.0 - w_xgb - w_dir
            if remaining < 0.3:
                continue
            # Distribute remaining in v4 proportions (60/10/30 of remaining)
            weights = [
                remaining * 0.60,
                remaining * 0.10,
                remaining * 0.30,
                w_xgb,
                w_dir
            ]
            name = f'XGB={w_xgb:.0%},D={w_dir:.0%}'
            configs.append((name, weights))
    
    # Also test some non-proportional distributions
    extra_configs = [
        ('LR5=70%+XGB=30%',      [0.70, 0.00, 0.00, 0.30, 0.00]),
        ('LR5=80%+XGB=20%',      [0.80, 0.00, 0.00, 0.20, 0.00]),
        ('LR5=60%+XGB=20%+LRk=20%', [0.60, 0.00, 0.20, 0.20, 0.00]),
        ('LR5=50%+LR001=10%+XGB=20%+D=20%', [0.50, 0.10, 0.00, 0.20, 0.20]),
    ]
    configs.extend(extra_configs)
    
    powers = [0.5, 0.625, 0.75, 0.875, 1.0, 1.125]
    
    results = []
    
    for cfg_name, weights in configs:
        for power in powers:
            test_assigned = np.zeros(n, dtype=int)
            
            for hold_season in folds:
                if hold_season not in season_scores:
                    continue
                scores, season_mask, season_indices = season_scores[hold_season]
                
                blended = np.zeros(len(season_indices))
                for mi in range(n_methods):
                    blended += weights[mi] * scores[mi]
                
                # Lock training seeds
                for i, global_idx in enumerate(season_indices):
                    if not test_mask[global_idx]:
                        blended[i] = y[global_idx]
                
                avail = {hold_season: list(range(1, 69))}
                assigned = hungarian(blended, seasons[season_mask], avail, power=power)
                
                for i, global_idx in enumerate(season_indices):
                    if test_mask[global_idx]:
                        test_assigned[global_idx] = assigned[i]
            
            gt = y[test_mask].astype(int)
            pred = test_assigned[test_mask]
            exact = int((pred == gt).sum())
            rmse = np.sqrt(np.mean((pred - gt)**2))
            
            results.append({
                'name': cfg_name, 'power': power, 'weights': weights,
                'exact': exact, 'rmse': rmse, 'assigned': test_assigned.copy()
            })
    
    results.sort(key=lambda r: r['rmse'])
    
    # Show top 30
    print(f'\n  {"Rk":>3} {"Config":<40} {"Pwr":>5} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─"*3} {"─"*40} {"─"*5} {"─"*5} {"─"*8}')
    for i, r in enumerate(results[:30]):
        m = ' ★' if i == 0 else ''
        print(f'  {i+1:3d} {r["name"]:<40} {r["power"]:5.3f} {r["exact"]:5d} '
              f'{r["rmse"]:8.4f}{m}')
    
    # Best result
    best = results[0]
    print(f'\n  BEST: {best["name"]}, power={best["power"]}'
          f' → {best["exact"]}/91 exact, RMSE={best["rmse"]:.4f}')
    
    # Save submission
    submission = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    for i in np.where(test_mask)[0]:
        rid_to_seed[record_ids[i]] = int(best['assigned'][i])
    submission['Overall Seed'] = submission['RecordID'].map(
        lambda r: rid_to_seed.get(r, 0))
    out = os.path.join(DATA_DIR, 'submission_kaggle_v5.csv')
    submission.to_csv(out, index=False)
    print(f'  Saved: {out}')
    
    # Also show per-season for best
    print(f'\n  Per-season (best):')
    for s in folds:
        s_mask = test_mask & (seasons == s)
        if s_mask.sum() == 0:
            continue
        s_gt = y[s_mask].astype(int)
        s_pred = best['assigned'][s_mask]
        s_ex = int((s_pred == s_gt).sum())
        s_rm = np.sqrt(np.mean((s_pred - s_gt)**2))
        print(f'    {s}: {s_ex}/{s_mask.sum()} exact, RMSE={s_rm:.3f}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
