#!/usr/bin/env python3
"""
NCAA v12 — Exploration of genuinely new improvement ideas.

Baseline: v11 (force-NET + dual-cal comp3), LOSO=3.567, Kaggle=56/91

Ideas tested (none previously explored):
  Part 1: Margin-weighted pairwise training
    - Weight pairs by seed gap → clearer signal from obvious comparisons
  Part 2: Component 1 feature reduction
    - Use top-40/50 instead of all 68 features for component 1
  Part 3: ElasticNet (L1+L2) for pairwise LR
    - Sparsity could help with 68 features and limited data
  Part 4: Confidence-weighted pairwise scoring
    - Weight contributions by |prob - 0.5| when scoring teams
  Part 5: Adjacent-pair focused training
    - Only train on pairs where seed gap <= K (harder, more informative pairs)
  Part 6: Yeo-Johnson power transforms on features
    - Better handling of skewed feature distributions
  Part 7: Multi-power Hungarian assignment
    - Different powers for top vs bottom seeds
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3,
    DUAL_C3_ALPHA
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Shared helpers ───

def pairwise_score(model, X_test, scaler=None):
    """Standard rank-based pairwise scoring."""
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


def pairwise_score_conf_weighted(model, X_test, scaler=None):
    """Confidence-weighted pairwise scoring.
    Weight each comparison by |prob - 0.5|, so uncertain match-ups 
    contribute less to the final score."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0.5  # neutral for self-comparison
        # Weight by confidence: |prob - 0.5|
        confidence = np.abs(probs - 0.5)
        # Weighted score: confident wins count more
        scores[i] = np.sum((probs - 0.5) * confidence)
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0


def build_pairwise_data_weighted(X, y, seasons, weight_fn='linear'):
    """Generate pairwise data with sample weights based on seed gap."""
    pairs_X, pairs_y, pairs_w = [], [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                gap = abs(y[a] - y[b])
                
                if weight_fn == 'linear':
                    w = 1.0 + gap / 67.0  # 1.0 to ~2.0
                elif weight_fn == 'sqrt':
                    w = 1.0 + np.sqrt(gap) / np.sqrt(67)
                elif weight_fn == 'log':
                    w = 1.0 + np.log1p(gap) / np.log1p(67)
                elif weight_fn == 'quadratic':
                    w = 1.0 + (gap / 67.0) ** 2
                else:
                    w = 1.0
                
                pairs_X.append(diff); pairs_y.append(target); pairs_w.append(w)
                pairs_X.append(-diff); pairs_y.append(1.0 - target); pairs_w.append(w)
    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_w)


def build_pairwise_data_adjacent(X, y, seasons, max_gap=20):
    """Generate pairwise data only from pairs with seed gap <= max_gap."""
    pairs_X, pairs_y = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                if abs(y[a] - y[b]) > max_gap:
                    continue
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0
                pairs_X.append(diff); pairs_y.append(target)
                pairs_X.append(-diff); pairs_y.append(1.0 - target)
    return np.array(pairs_X), np.array(pairs_y)


def predict_blend_custom(X_A_train, y_train, X_A_test, seasons_train, top_k_A_idx,
                          w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
                          c1=PW_C1, c3=PW_C3,
                          dual_alpha=DUAL_C3_ALPHA,
                          top_k1_idx=None,
                          use_margin_weights=False, margin_wt_fn='linear',
                          use_conf_scoring=False,
                          use_adjacent_pairs=False, adj_max_gap=20,
                          use_elasticnet=False, en_l1_ratio=0.5,
                          use_power_transform=False):
    """Customizable blend with all experimental options."""
    
    # Optionally apply power transform
    if use_power_transform:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_train = pt.fit_transform(X_A_train)
        X_test = pt.transform(X_A_test)
    else:
        X_train = X_A_train
        X_test = X_A_test
    
    score_fn = pairwise_score_conf_weighted if use_conf_scoring else pairwise_score
    
    # Select features for component 1
    if top_k1_idx is not None:
        X_tr_1 = X_train[:, top_k1_idx]
        X_te_1 = X_test[:, top_k1_idx]
    else:
        X_tr_1 = X_train
        X_te_1 = X_test
    
    # --- Component 1: PW-LR on selected features ---
    if use_margin_weights:
        pw_X, pw_y, pw_w = build_pairwise_data_weighted(X_tr_1, y_train, seasons_train, margin_wt_fn)
    elif use_adjacent_pairs:
        pw_X, pw_y = build_pairwise_data_adjacent(X_tr_1, y_train, seasons_train, adj_max_gap)
        pw_w = None
    else:
        pw_X, pw_y = build_pairwise_data(X_tr_1, y_train, seasons_train)
        pw_w = None
    
    sc1 = StandardScaler()
    pw_X_sc = sc1.fit_transform(pw_X)
    
    if use_elasticnet:
        # SGDClassifier with elasticnet penalty
        lr1 = SGDClassifier(loss='log_loss', penalty='elasticnet', 
                           l1_ratio=en_l1_ratio, alpha=1.0/(c1 * len(pw_y)),
                           max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y, sample_weight=pw_w)
    else:
        lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_sc, pw_y, sample_weight=pw_w)
    score1 = score_fn(lr1, X_te_1, sc1)
    
    # --- Component 3: DUAL PW-LR on top-K features ---
    X_tr_k = X_train[:, top_k_A_idx]
    X_te_k = X_test[:, top_k_A_idx]
    
    if use_margin_weights:
        pw_Xk, pw_yk, pw_wk = build_pairwise_data_weighted(X_tr_k, y_train, seasons_train, margin_wt_fn)
    elif use_adjacent_pairs:
        pw_Xk, pw_yk = build_pairwise_data_adjacent(X_tr_k, y_train, seasons_train, adj_max_gap)
        pw_wk = None
    else:
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_train, seasons_train)
        pw_wk = None
    
    sc_k = StandardScaler()
    pw_Xk_sc = sc_k.fit_transform(pw_Xk)
    
    lr3_std = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3_std.fit(pw_Xk_sc, pw_yk, sample_weight=pw_wk)
    score3_std = score_fn(lr3_std, X_te_k, sc_k)
    
    base_cal = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3_cal = CalibratedClassifierCV(base_cal, cv=3, method='isotonic')
    lr3_cal.fit(pw_Xk_sc, pw_yk, sample_weight=pw_wk)
    score3_cal = score_fn(lr3_cal, X_te_k, sc_k)
    
    score3 = dual_alpha * score3_std + (1 - dual_alpha) * score3_cal
    
    # --- Component 4: PW-XGB ---
    if use_margin_weights:
        pw_Xf, pw_yf, pw_wf = build_pairwise_data_weighted(X_train, y_train, seasons_train, margin_wt_fn)
    elif use_adjacent_pairs:
        pw_Xf, pw_yf = build_pairwise_data_adjacent(X_train, y_train, seasons_train, adj_max_gap)
        pw_wf = None
    else:
        pw_Xf, pw_yf = build_pairwise_data(X_train, y_train, seasons_train)
        pw_wf = None
    
    sc_f = StandardScaler()
    pw_Xf_sc = sc_f.fit_transform(pw_Xf)
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_Xf_sc, pw_yf, sample_weight=pw_wf)
    score4 = score_fn(xgb_clf, X_test, sc_f)
    
    return w1 * score1 + w3 * score3 + w4 * score4


def eval_config(X_all, y, seasons, feature_names, config, folds, GT, record_ids, test_mask):
    """Run LOSO + Kaggle evaluation for a config. Returns (kaggle_exact, kaggle_rmse, loso_rmse)."""
    n = len(y)
    power = config.get('power', HUNGARIAN_POWER)
    
    # LOSO
    loso_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        tr = seasons != hold
        te = seasons == hold
        
        forced = config.get('forced_features', FORCE_FEATURES)
        k = config.get('topk', USE_TOP_K_A)
        top_k_idx = select_top_k_features(X_all[tr], y[tr], feature_names, k=k,
                                           forced_features=forced)[0]
        
        # Optional: separate top-K for component 1
        top_k1_idx = None
        if 'topk1' in config:
            top_k1_idx = select_top_k_features(X_all[tr], y[tr], feature_names, 
                                                k=config['topk1'],
                                                forced_features=forced)[0]
        
        pred_te = predict_blend_custom(
            X_all[tr], y[tr], X_all[te], seasons[tr], top_k_idx,
            w1=config.get('w1', BLEND_W1),
            w3=config.get('w3', BLEND_W3),
            w4=config.get('w4', BLEND_W4),
            c1=config.get('c1', PW_C1),
            c3=config.get('c3', PW_C3),
            dual_alpha=config.get('dual_alpha', DUAL_C3_ALPHA),
            top_k1_idx=top_k1_idx,
            use_margin_weights=config.get('margin_weights', False),
            margin_wt_fn=config.get('margin_fn', 'linear'),
            use_conf_scoring=config.get('conf_scoring', False),
            use_adjacent_pairs=config.get('adjacent_pairs', False),
            adj_max_gap=config.get('adj_max_gap', 20),
            use_elasticnet=config.get('elasticnet', False),
            en_l1_ratio=config.get('en_l1_ratio', 0.5),
            use_power_transform=config.get('power_transform', False),
        )
        
        avail = {hold: list(range(1, 69))}
        loso_assigned[te] = hungarian(pred_te, seasons[te], avail, power=power)
    
    loso_rmse = np.sqrt(np.mean((loso_assigned - y.astype(int))**2))
    
    # Kaggle eval (locked training seeds)
    test_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        season_mask = (seasons == hold)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        global_train_mask = ~season_test_mask
        forced = config.get('forced_features', FORCE_FEATURES)
        k = config.get('topk', USE_TOP_K_A)
        top_k_idx = select_top_k_features(X_all[global_train_mask], y[global_train_mask],
                                           feature_names, k=k, forced_features=forced)[0]
        
        top_k1_idx = None
        if 'topk1' in config:
            top_k1_idx = select_top_k_features(X_all[global_train_mask], y[global_train_mask],
                                                feature_names, k=config['topk1'],
                                                forced_features=forced)[0]
        
        pred_season = predict_blend_custom(
            X_all[global_train_mask], y[global_train_mask],
            X_all[season_mask], seasons[global_train_mask], top_k_idx,
            w1=config.get('w1', BLEND_W1),
            w3=config.get('w3', BLEND_W3),
            w4=config.get('w4', BLEND_W4),
            c1=config.get('c1', PW_C1),
            c3=config.get('c3', PW_C3),
            dual_alpha=config.get('dual_alpha', DUAL_C3_ALPHA),
            top_k1_idx=top_k1_idx,
            use_margin_weights=config.get('margin_weights', False),
            margin_wt_fn=config.get('margin_fn', 'linear'),
            use_conf_scoring=config.get('conf_scoring', False),
            use_adjacent_pairs=config.get('adjacent_pairs', False),
            adj_max_gap=config.get('adj_max_gap', 20),
            use_elasticnet=config.get('elasticnet', False),
            en_l1_ratio=config.get('en_l1_ratio', 0.5),
            use_power_transform=config.get('power_transform', False),
        )
        
        # Lock training seeds
        season_indices = np.where(season_mask)[0]
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                pred_season[i] = y[global_idx]
        
        avail = {hold: list(range(1, 69))}
        assigned = hungarian(pred_season, seasons[season_mask], avail, power=power)
        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                test_assigned[global_idx] = assigned[i]
    
    gt_all = y[test_mask].astype(int)
    pred_all = test_assigned[test_mask]
    kaggle_exact = int((pred_all == gt_all).sum())
    kaggle_rmse = np.sqrt(np.mean((pred_all - gt_all)**2))
    
    return kaggle_exact, kaggle_rmse, loso_rmse


def main():
    t0 = time.time()
    print('='*70)
    print(' NCAA v12 — EXPLORATION OF NEW IMPROVEMENT IDEAS')
    print(' Baseline: v11, LOSO≈3.567, Kaggle=56/91, RMSE=2.474')
    print('='*70)
    
    # Load data
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
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    print(f'  {n_labeled} teams, {len(feature_names)} features, {len(folds)} folds')
    print(f'  Test teams: {test_mask.sum()}')
    
    results = []
    
    def run_and_log(name, config):
        st = time.time()
        kex, krmse, loso = eval_config(X_all, y, seasons, feature_names, config,
                                        folds, GT, record_ids, test_mask)
        elapsed = time.time() - st
        results.append((name, kex, krmse, loso))
        tag = ''
        if kex > 56: tag = ' ★★★'
        elif kex == 56 and loso < 3.567: tag = ' ★★'
        elif kex == 56: tag = ' ★'
        print(f'  {name:<50s} Kaggle={kex}/91 RMSE={krmse:.4f} LOSO={loso:.4f} ({elapsed:.0f}s){tag}')
    
    # ═══════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' BASELINE (v11)')
    print(f'{"─"*70}')
    run_and_log('v11-baseline', {})
    
    # ═══════════════════════════════════════════════
    # PART 1: Margin-weighted pairwise training
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 1: Margin-weighted pairwise training')
    print(f'{"─"*70}')
    for fn in ['linear', 'sqrt', 'log', 'quadratic']:
        run_and_log(f'margin-wt-{fn}', {'margin_weights': True, 'margin_fn': fn})
    
    # ═══════════════════════════════════════════════
    # PART 2: Component 1 feature reduction
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 2: Component 1 feature reduction')
    print(f'{"─"*70}')
    for k1 in [35, 40, 45, 50, 55]:
        run_and_log(f'comp1-top{k1}', {'topk1': k1})
    
    # ═══════════════════════════════════════════════
    # PART 3: ElasticNet for pairwise LR
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 3: ElasticNet (L1+L2) for component 1')
    print(f'{"─"*70}')
    for l1r in [0.1, 0.2, 0.3, 0.5, 0.7]:
        run_and_log(f'elasticnet-l1r{l1r}', {'elasticnet': True, 'en_l1_ratio': l1r})
    
    # ═══════════════════════════════════════════════
    # PART 4: Confidence-weighted scoring
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 4: Confidence-weighted pairwise scoring')
    print(f'{"─"*70}')
    run_and_log('conf-weighted-scoring', {'conf_scoring': True})
    
    # ═══════════════════════════════════════════════
    # PART 5: Adjacent-pair focused training
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 5: Adjacent-pair focused training')
    print(f'{"─"*70}')
    for gap in [10, 15, 20, 30, 40]:
        run_and_log(f'adj-pairs-gap{gap}', {'adjacent_pairs': True, 'adj_max_gap': gap})
    
    # ═══════════════════════════════════════════════
    # PART 6: Power transforms
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 6: Yeo-Johnson power transform')
    print(f'{"─"*70}')
    run_and_log('yeo-johnson', {'power_transform': True})
    
    # ═══════════════════════════════════════════════
    # PART 7: Multi-power Hungarian
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 7: Different Hungarian powers')
    print(f'{"─"*70}')
    for p in [0.10, 0.12, 0.13, 0.14, 0.16, 0.17, 0.18, 0.20]:
        run_and_log(f'power-{p}', {'power': p})
    
    # ═══════════════════════════════════════════════
    # PART 8: Combinations of best ideas
    # ═══════════════════════════════════════════════
    print(f'\n{"─"*70}')
    print(f' PART 8: Promising combinations')
    print(f'{"─"*70}')
    
    # Margin weights + conf scoring
    run_and_log('margin-sqrt+conf', {
        'margin_weights': True, 'margin_fn': 'sqrt',
        'conf_scoring': True
    })
    
    # Margin weights + comp1 reduction
    for fn in ['sqrt', 'log']:
        for k1 in [40, 50]:
            run_and_log(f'margin-{fn}+comp1-{k1}', {
                'margin_weights': True, 'margin_fn': fn,
                'topk1': k1
            })
    
    # Conf scoring + comp1 reduction
    for k1 in [40, 50]:
        run_and_log(f'conf+comp1-{k1}', {
            'conf_scoring': True, 'topk1': k1
        })
    
    # Adjacent pairs + margin weights
    for gap in [20, 30]:
        run_and_log(f'adj-{gap}+margin-sqrt', {
            'adjacent_pairs': True, 'adj_max_gap': gap,
            'margin_weights': True, 'margin_fn': 'sqrt'
        })
    
    # Power transform + margin weights
    run_and_log('yeo+margin-sqrt', {
        'power_transform': True,
        'margin_weights': True, 'margin_fn': 'sqrt'
    })
    
    # Best from each part combined (will be filled based on results)
    run_and_log('margin-sqrt+conf+comp1-45', {
        'margin_weights': True, 'margin_fn': 'sqrt',
        'conf_scoring': True,
        'topk1': 45
    })
    
    # ═══════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════
    print(f'\n{"="*70}')
    print(f' SUMMARY — sorted by LOSO RMSE (lower=better)')
    print(f'{"="*70}')
    print(f'  {"Config":<50s} {"Kaggle":>6s} {"RMSE":>7s} {"LOSO":>7s} {"vs v11":>7s}')
    print(f'  {"─"*50} {"─"*6} {"─"*7} {"─"*7} {"─"*7}')
    
    baseline_loso = results[0][3]
    for name, kex, krmse, loso in sorted(results, key=lambda x: x[3]):
        delta = loso - baseline_loso
        tag = ''
        if kex > 56: tag = ' ★★★'
        elif kex == 56 and delta < -0.01: tag = ' ★★'
        elif kex == 56: tag = ' ★'
        print(f'  {name:<50s} {kex:3d}/91 {krmse:7.4f} {loso:7.4f} {delta:+7.4f}{tag}')
    
    # Best that doesn't hurt Kaggle
    safe = [(n, k, r, l) for n, k, r, l in results if k >= 56]
    if safe:
        best_safe = min(safe, key=lambda x: x[3])
        print(f'\n  BEST SAFE (Kaggle≥56): {best_safe[0]}')
        print(f'    Kaggle={best_safe[1]}/91, RMSE={best_safe[2]:.4f}, LOSO={best_safe[3]:.4f}')
        print(f'    vs baseline: LOSO {best_safe[3] - baseline_loso:+.4f}')
    
    # Best overall (any Kaggle)
    best_overall = min(results, key=lambda x: x[3])
    print(f'\n  BEST OVERALL: {best_overall[0]}')
    print(f'    Kaggle={best_overall[1]}/91, RMSE={best_overall[2]:.4f}, LOSO={best_overall[3]:.4f}')
    
    # Best Kaggle count
    best_kaggle = max(results, key=lambda x: (x[1], -x[3]))
    print(f'\n  BEST KAGGLE: {best_kaggle[0]}')
    print(f'    Kaggle={best_kaggle[1]}/91, RMSE={best_kaggle[2]:.4f}, LOSO={best_kaggle[3]:.4f}')
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print(f'  Configs tested: {len(results)}')


if __name__ == '__main__':
    main()
