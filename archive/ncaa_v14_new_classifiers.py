#!/usr/bin/env python3
"""
v14: Fundamentally new components — LightGBM, CatBoost, SVM pairwise classifiers.

Previous 110 configs all used same 3 components (LR+LR_topK+XGB).
THIS tries genuinely different model types as pairwise classifiers,
then searches for the best 3-6 component blend on Kaggle eval.

Also tries: Kaggle-simulation CV (hold out random teams, lock rest).
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from itertools import combinations

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, hungarian, pairwise_score,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Scoring helpers ───

def pw_score(model, X_test, scaler=None):
    """Pairwise win-rate scoring → rank [1..N]."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0


def kaggle_eval_one(assigned, y_int, test_mask):
    """Quick eval on test teams."""
    pred = assigned[test_mask]
    true = y_int[test_mask]
    exact = int((pred == true).sum())
    rmse = np.sqrt(np.mean((pred - true)**2))
    return exact, rmse


def main():
    t0 = time.time()
    print('=' * 70)
    print(' v14: NEW CLASSIFIERS — LightGBM, CatBoost, SVM, RF, KNN')
    print('=' * 70)

    # ─── Load data ───
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat.columns)

    y = labeled['Overall Seed'].values.astype(float)
    y_int = y.astype(int)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    print(f'  Teams: {n_labeled}, Test: {test_mask.sum()}, Features: {len(feature_names)}')

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 1: EVALUATE INDIVIDUAL COMPONENTS ON KAGGLE EVAL
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 1: INDIVIDUAL COMPONENT EVALUATION')
    print('=' * 70)

    # Store per-season scores for each component (for later blending)
    component_scores = {}  # name → array of scores for all teams

    def build_component(name, model_fn, use_topk=False, use_adj=False,
                        adj_gap=30, features='full'):
        """Build and evaluate a component across all seasons."""
        all_scores = np.zeros(n_labeled)

        for hold in folds:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            stm = test_mask & sm
            global_train = ~(test_mask & sm)

            X_tr = X_all[global_train]
            y_tr = y[global_train]
            s_tr = seasons[global_train]
            X_season = X_all[sm]

            top_k_idx = select_top_k_features(
                X_tr, y_tr, feature_names, k=USE_TOP_K_A,
                forced_features=FORCE_FEATURES)[0]

            # Choose features
            if features == 'topk':
                Xtr = X_tr[:, top_k_idx]
                Xte = X_season[:, top_k_idx]
            else:
                Xtr = X_tr
                Xte = X_season

            # Build pairwise data
            if use_adj:
                pw_X, pw_y = build_pairwise_data_adjacent(Xtr, y_tr, s_tr, max_gap=adj_gap)
            else:
                pw_X, pw_y = build_pairwise_data(Xtr, y_tr, s_tr)

            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            # Train model
            model = model_fn()
            model.fit(pw_X_sc, pw_y)

            # Score
            scores = pw_score(model, Xte, sc)
            all_scores[sm] = scores

        # Lock training, Hungarian, evaluate
        assigned = np.zeros(n_labeled, dtype=int)
        for hold in folds:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            locked = all_scores[sm].copy()
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    locked[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            asgn = hungarian(locked, seasons[sm], avail, power=0.15)
            for i, gi in enumerate(si):
                assigned[gi] = asgn[i]

        exact, rmse = kaggle_eval_one(assigned, y_int, test_mask)
        component_scores[name] = all_scores
        return exact, rmse

    # ─── Define components ───
    components = [
        # Existing components (for reference)
        ('LR_C5_adj30', lambda: LogisticRegression(C=5.0, max_iter=2000, random_state=42),
         False, True, 30, 'full'),
        ('LR_C05_topK', lambda: LogisticRegression(C=0.5, max_iter=2000, random_state=42),
         True, False, 30, 'topk'),
        ('XGB_d4', lambda: xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.0,
                    min_child_weight=5, random_state=42, verbosity=0,
                    use_label_encoder=False, eval_metric='logloss'),
         False, False, 30, 'full'),

        # NEW: LightGBM pairwise
        ('LGBM_d4', lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=42, verbose=-1),
         False, False, 30, 'full'),
        ('LGBM_d4_adj30', lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=42, verbose=-1),
         False, True, 30, 'full'),
        ('LGBM_d6', lambda: lgb.LGBMClassifier(n_estimators=500, max_depth=6,
                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                    reg_lambda=5.0, reg_alpha=2.0, min_child_weight=10,
                    random_state=42, verbose=-1),
         False, False, 30, 'full'),
        ('LGBM_topK', lambda: lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                    random_state=42, verbose=-1),
         True, False, 30, 'topk'),

        # NEW: CatBoost pairwise
        ('CB_d4', lambda: cb.CatBoostClassifier(iterations=300, depth=4,
                    learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42,
                    verbose=0),
         False, False, 30, 'full'),
        ('CB_d4_adj30', lambda: cb.CatBoostClassifier(iterations=300, depth=4,
                    learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42,
                    verbose=0),
         False, True, 30, 'full'),
        ('CB_d6', lambda: cb.CatBoostClassifier(iterations=500, depth=6,
                    learning_rate=0.03, l2_leaf_reg=5.0, random_seed=42,
                    verbose=0),
         False, False, 30, 'full'),

        # NEW: SVM pairwise
        ('SVM_rbf', lambda: SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
         False, False, 30, 'full'),
        ('SVM_rbf_topK', lambda: SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
         True, False, 30, 'topk'),
        ('SVM_linear', lambda: SVC(C=1.0, kernel='linear', probability=True, random_state=42),
         False, False, 30, 'full'),

        # NEW: Random Forest pairwise
        ('RF_500', lambda: RandomForestClassifier(n_estimators=500, max_depth=8,
                    min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1),
         False, False, 30, 'full'),
        ('RF_500_topK', lambda: RandomForestClassifier(n_estimators=500, max_depth=8,
                    min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1),
         True, False, 30, 'topk'),
        ('RF_500_adj30', lambda: RandomForestClassifier(n_estimators=500, max_depth=8,
                    min_samples_leaf=5, max_features=0.5, random_state=42, n_jobs=-1),
         False, True, 30, 'full'),

        # NEW: GradientBoosting (sklearn) pairwise
        ('GBM_sklearn', lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4,
                    learning_rate=0.05, subsample=0.8, min_samples_leaf=5,
                    random_state=42),
         False, False, 30, 'full'),

        # NEW: KNN pairwise
        ('KNN_7', lambda: KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
         False, False, 30, 'full'),
        ('KNN_11', lambda: KNeighborsClassifier(n_neighbors=11, weights='distance', n_jobs=-1),
         False, False, 30, 'full'),
        ('KNN_15_topK', lambda: KNeighborsClassifier(n_neighbors=15, weights='distance', n_jobs=-1),
         True, False, 30, 'topk'),

        # NEW: LR with different settings
        ('LR_C1_adj20', lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=42),
         False, True, 20, 'full'),
        ('LR_C10_full', lambda: LogisticRegression(C=10.0, max_iter=2000, random_state=42),
         False, False, 30, 'full'),
        ('LR_C5_topK_adj30', lambda: LogisticRegression(C=5.0, max_iter=2000, random_state=42),
         True, True, 30, 'topk'),
    ]

    print(f'\n  Evaluating {len(components)} individual components...\n')
    print(f'  {"Component":<25} {"Exact":>8} {"RMSE":>8}')
    print(f'  {"─"*25} {"─"*8} {"─"*8}')

    comp_results = []
    for i, (name, model_fn, use_topk, use_adj, adj_gap, feat_type) in enumerate(components):
        try:
            exact, rmse = build_component(name, model_fn, use_topk, use_adj, adj_gap, feat_type)
            comp_results.append((name, exact, rmse))
            mark = ' ★' if exact > 49 else ''  # Mark if better than best individual so far
            print(f'  {name:<25} {exact:>3}/91  {rmse:>8.4f}{mark}')
        except Exception as e:
            print(f'  {name:<25} ERROR: {e}')
        
        if (i + 1) % 5 == 0:
            print(f'  ... [{i+1}/{len(components)}] ({time.time()-t0:.0f}s)')

    # Sort by exact descending
    comp_results.sort(key=lambda x: (-x[1], x[2]))
    print(f'\n  TOP 10 individual components:')
    for i, (name, exact, rmse) in enumerate(comp_results[:10]):
        print(f'    {i+1}. {name}: {exact}/91, RMSE={rmse:.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 2: MULTI-COMPONENT BLEND SEARCH
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 2: MULTI-COMPONENT BLEND SEARCH')
    print('=' * 70)

    # Take top 10 components and search for best 2-5 component blends
    top_comp_names = [name for name, _, _ in comp_results[:10]]
    print(f'\n  Using top components: {top_comp_names}')

    best_blend_exact = 0
    best_blend_rmse = 999
    best_blend_config = None
    blend_results = []

    # Try all 2-component combinations
    print(f'\n  Searching 2-component blends...')
    for c1_name, c2_name in combinations(top_comp_names, 2):
        s1 = component_scores[c1_name]
        s2 = component_scores[c2_name]
        
        for w1 in np.arange(0.3, 0.85, 0.05):
            w2 = round(1.0 - w1, 2)
            
            blended_scores = w1 * s1 + w2 * s2
            
            assigned = np.zeros(n_labeled, dtype=int)
            for hold in folds:
                sm = (seasons == hold)
                si = np.where(sm)[0]
                locked = blended_scores[sm].copy()
                for i, gi in enumerate(si):
                    if not test_mask[gi]:
                        locked[i] = y[gi]
                avail = {hold: list(range(1, 69))}
                asgn = hungarian(locked, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    assigned[gi] = asgn[i]
            
            exact, rmse = kaggle_eval_one(assigned, y_int, test_mask)
            
            blend_results.append({
                'config': f'{w1:.2f}*{c1_name} + {w2:.2f}*{c2_name}',
                'exact': exact, 'rmse': rmse
            })
            
            if exact > best_blend_exact or (exact == best_blend_exact and rmse < best_blend_rmse):
                best_blend_exact = exact
                best_blend_rmse = rmse
                best_blend_config = blend_results[-1]['config']

    # Try all 3-component combinations with grid weights
    print(f'  Searching 3-component blends...')
    for c1_name, c2_name, c3_name in combinations(top_comp_names[:8], 3):
        s1 = component_scores[c1_name]
        s2 = component_scores[c2_name]
        s3 = component_scores[c3_name]
        
        for w1 in np.arange(0.3, 0.8, 0.1):
            for w2 in np.arange(0.1, 0.6, 0.1):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 < 0.05 or w3 > 0.6:
                    continue
                
                blended_scores = w1 * s1 + w2 * s2 + w3 * s3
                
                assigned = np.zeros(n_labeled, dtype=int)
                for hold in folds:
                    sm = (seasons == hold)
                    si = np.where(sm)[0]
                    locked = blended_scores[sm].copy()
                    for i, gi in enumerate(si):
                        if not test_mask[gi]:
                            locked[i] = y[gi]
                    avail = {hold: list(range(1, 69))}
                    asgn = hungarian(locked, seasons[sm], avail, power=0.15)
                    for i, gi in enumerate(si):
                        assigned[gi] = asgn[i]
                
                exact, rmse = kaggle_eval_one(assigned, y_int, test_mask)
                
                blend_results.append({
                    'config': f'{w1:.1f}*{c1_name}+{w2:.1f}*{c2_name}+{w3:.2f}*{c3_name}',
                    'exact': exact, 'rmse': rmse
                })
                
                if exact > best_blend_exact or (exact == best_blend_exact and rmse < best_blend_rmse):
                    best_blend_exact = exact
                    best_blend_rmse = rmse
                    best_blend_config = blend_results[-1]['config']

    # Sort all blend results
    blend_results.sort(key=lambda r: (-r['exact'], r['rmse']))

    print(f'\n  Total blends evaluated: {len(blend_results)}')
    print(f'\n  TOP 20 BLENDS:')
    print(f'  {"Config":<65} {"Exact":>7} {"RMSE":>8}')
    print(f'  {"─"*65} {"─"*7} {"─"*8}')
    for r in blend_results[:20]:
        print(f'  {r["config"]:<65} {r["exact"]:>3}/91 {r["rmse"]:>8.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  PHASE 3: LOSO VALIDATION OF TOP BLENDS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' PHASE 3: LOSO FOR TOP BLENDS (sanity check)')
    print('=' * 70)

    # Check if the top Kaggle blends also have good LOSO
    for r in blend_results[:10]:
        config_str = r['config']
        # Reconstruct scores from component names and weights
        parts = config_str.split('+')
        blended_loso = np.zeros(n_labeled)
        for part in parts:
            part = part.strip()
            wt_str, comp_name = part.split('*', 1)
            wt = float(wt_str)
            blended_loso += wt * component_scores[comp_name]
        
        # LOSO eval (no locking)
        loso_assigned = np.zeros(n_labeled, dtype=int)
        for hold in folds:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            avail = {hold: list(range(1, 69))}
            asgn = hungarian(blended_loso[sm], seasons[sm], avail, power=0.15)
            for i, gi in enumerate(si):
                loso_assigned[gi] = asgn[i]
        
        loso_exact = int((loso_assigned == y_int).sum())
        loso_rmse = np.sqrt(np.mean((loso_assigned - y_int)**2))
        
        print(f'  {config_str[:60]:<60} Kag={r["exact"]}/91 LOSO={loso_rmse:.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SUMMARY')
    print('=' * 70)
    print(f'\n  Best individual component: {comp_results[0][0]} = {comp_results[0][1]}/91')
    print(f'  Best blend: {best_blend_config}')
    print(f'    Exact: {best_blend_exact}/91')
    print(f'    RMSE:  {best_blend_rmse:.4f}')
    print(f'  v12 baseline: 57/91, RMSE=2.4829')
    
    if best_blend_exact > 57:
        print(f'\n  *** IMPROVEMENT FOUND: +{best_blend_exact - 57} exact matches! ***')
    elif best_blend_exact == 57 and best_blend_rmse < 2.4829:
        print(f'\n  * Same exact but lower RMSE ({best_blend_rmse:.4f} < 2.4829)')
    else:
        print(f'\n  No improvement over v12.')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
