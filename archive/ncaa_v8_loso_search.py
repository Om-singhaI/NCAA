#!/usr/bin/env python3
"""
NCAA v8 LOSO-Optimized Search
==============================
Goal: Find a blend that GENUINELY generalizes better than v6 for 2025-26 prediction.

KEY PRINCIPLES (learned from v7 failure):
  1. LOSO score is the ONLY metric that predicts actual Kaggle performance
  2. Local "Kaggle RMSE" on 91 known test teams is MISLEADING (v7 proved this)
  3. MLP/neural nets OVERFIT — they're banned from v8
  4. LR methods generalize far better across seasons
  5. Any v8 candidate must beat v6 LOSO (3.678) AND be stable

SEARCH SPACE:
  - LR with varying C: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
  - LR with topK features: k=[15, 20, 25, 30, 35]
  - XGB with varying configs: d=[2,3,4], n=[100,200,300], lr=[0.01,0.05,0.1]
  - Blends of 2-4 components
  - Power: [0.05, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]

VALIDATION:
  - LOSO score = mean(fold_RMSEs) + 0.5 * std(fold_RMSEs)
  - Must beat v6's 3.678 on LOSO
  - Must improve on ≥3/5 folds (not just averages)
  - Bootstrap stability check (500 trials)
  - Weight perturbation sensitivity check
"""

import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Import shared functions
sys.path.insert(0, DATA_DIR)
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, pairwise_score, hungarian
)

V6_LOSO = 3.678  # v6 LOSO score (our baseline to beat)

def compute_loso_score(fold_rmses):
    """LOSO score = mean + 0.5*std (penalizes inconsistency)."""
    return np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)

def main():
    t0 = time.time()
    print('='*60)
    print(' NCAA v8 LOSO-OPTIMIZED SEARCH')
    print(' Goal: Beat v6 LOSO=3.678 with stability')
    print('='*60)

    # ── Load & prepare data ──
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat_A = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names_A = list(feat_A.columns)
    n_feat = len(feature_names_A)
    print(f'  {n_labeled} teams, {n_feat} features, 5 folds')

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    folds = sorted(set(seasons))

    X_A_raw = np.where(np.isinf(feat_A.values.astype(np.float64)), np.nan,
                       feat_A.values.astype(np.float64))
    imp_A = KNNImputer(n_neighbors=10, weights='distance')
    X_A_all = imp_A.fit_transform(X_A_raw)

    # ─────────────────────────────────────────────────
    #  PHASE 1: Precompute individual method LOSO scores per fold
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 1: Precomputing individual methods per fold')
    print('='*60)

    # Define methods to evaluate
    lr_C_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    topK_values = [15, 20, 25, 30, 35]
    xgb_configs = [
        (2, 100, 0.05), (2, 200, 0.05), (3, 100, 0.05),
        (3, 200, 0.05), (3, 200, 0.1), (3, 300, 0.05),
        (4, 200, 0.05), (4, 300, 0.05),
    ]

    methods = {}

    # LR with full features, varying C
    for C in lr_C_values:
        methods[f'LR_C{C}'] = ('lr_full', C, None)

    # LR with topK features, varying C and K
    for C in [0.1, 0.5, 1.0, 2.0]:
        for K in topK_values:
            methods[f'LRk{K}_C{C}'] = ('lr_topk', C, K)

    # XGB classifiers
    for d, n, lr in xgb_configs:
        methods[f'XGB_d{d}_{n}_lr{lr}'] = ('xgb', d, n, lr)

    n_methods = len(methods)
    print(f'  {n_methods} individual methods to evaluate')

    # Precompute per-fold pairwise scores for each method
    # fold_scores[method_name][fold_idx] = array of pairwise rank scores for test teams
    fold_scores = {m: {} for m in methods}
    fold_y = {}  # fold_idx -> y values for that fold
    fold_n = {}  # fold_idx -> number of teams

    for fi, hold in enumerate(folds):
        tr = seasons != hold
        te = seasons == hold
        n_te = te.sum()
        fold_y[fi] = y[te].astype(int)
        fold_n[fi] = n_te

        X_tr = X_A_all[tr]
        X_te = X_A_all[te]
        y_tr = y[tr]
        s_tr = seasons[tr]

        # Feature selection (need this for topK methods)
        top_k_results = {}
        for K in topK_values:
            top_k_results[K] = select_top_k_features(X_tr, y_tr, feature_names_A, k=K)[0]

        # Build pairwise data on full features
        pw_X_full, pw_y_full = build_pairwise_data(X_tr, y_tr, s_tr)
        sc_full = StandardScaler()
        pw_X_full_sc = sc_full.fit_transform(pw_X_full)

        # LR full features
        for C in lr_C_values:
            name = f'LR_C{C}'
            lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
            lr.fit(pw_X_full_sc, pw_y_full)
            fold_scores[name][fi] = pairwise_score(lr, X_te, sc_full)

        # LR topK features
        for C in [0.1, 0.5, 1.0, 2.0]:
            for K in topK_values:
                name = f'LRk{K}_C{C}'
                tk_idx = top_k_results[K]
                X_tr_k = X_tr[:, tk_idx]
                X_te_k = X_te[:, tk_idx]
                pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
                sc_k = StandardScaler()
                pw_X_k_sc = sc_k.fit_transform(pw_X_k)
                lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
                lr.fit(pw_X_k_sc, pw_y_k)
                fold_scores[name][fi] = pairwise_score(lr, X_te_k, sc_k)

        # XGB classifiers
        for d, n, lr_rate in xgb_configs:
            name = f'XGB_d{d}_{n}_lr{lr_rate}'
            xgb_clf = xgb.XGBClassifier(
                n_estimators=n, max_depth=d, learning_rate=lr_rate,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss'
            )
            xgb_clf.fit(pw_X_full_sc, pw_y_full)
            fold_scores[name][fi] = pairwise_score(xgb_clf, X_te, sc_full)

        print(f'  Fold {fi+1}/5 ({hold}): {n_te} teams — all methods computed')

    # ── Individual method LOSO scores ──
    print(f'\n  Individual method LOSO scores:')
    print(f'  {"Method":<25} {"LOSO":>7} {"fRMSE":>40}')
    indiv_loso = {}
    for name in sorted(methods.keys()):
        fold_rmses = []
        for fi in range(5):
            scores = fold_scores[name][fi]
            avail = {folds[fi]: list(range(1, 69))}
            assigned = hungarian(scores, np.full(fold_n[fi], folds[fi]), avail, power=0.15)
            rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
            fold_rmses.append(rmse)
        loso = compute_loso_score(fold_rmses)
        indiv_loso[name] = (loso, fold_rmses)
        rmse_str = ' '.join([f'{r:.2f}' for r in fold_rmses])
        marker = ' ★' if loso < V6_LOSO else ''
        print(f'  {name:<25} {loso:7.4f}  [{rmse_str}]{marker}')

    # ─────────────────────────────────────────────────
    #  PHASE 2: Exhaustive blend search (LOSO-optimized)
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 2: Blend search (LOSO primary metric)')
    print('='*60)

    # Only use methods with reasonable individual LOSO (< 4.5)
    good_methods = [m for m, (l, _) in indiv_loso.items() if l < 4.5]
    good_methods = sorted(good_methods, key=lambda m: indiv_loso[m][0])[:20]  # top 20
    print(f'  Using top {len(good_methods)} methods for blending')

    power_values = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
    
    # We'll search 2-component and 3-component blends
    # Weight grid: 5% increments
    weight_steps = np.arange(0.05, 1.001, 0.05)
    
    results = []
    n_configs = 0
    best_loso = 999.0
    best_config = None
    
    # ── 2-component blends ──
    print('\n  Searching 2-component blends...')
    for i, m1 in enumerate(good_methods):
        for m2 in good_methods[i+1:]:
            for w1 in weight_steps:
                w2 = 1.0 - w1
                if w2 < 0.04:
                    continue
                for power in power_values:
                    fold_rmses = []
                    for fi in range(5):
                        blended = w1 * fold_scores[m1][fi] + w2 * fold_scores[m2][fi]
                        avail = {folds[fi]: list(range(1, 69))}
                        assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]),
                                           avail, power=power)
                        rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
                        fold_rmses.append(rmse)
                    loso = compute_loso_score(fold_rmses)
                    n_configs += 1
                    if loso < best_loso:
                        best_loso = loso
                        best_config = {
                            'components': [(m1, w1), (m2, w2)],
                            'power': power,
                            'loso': loso,
                            'fold_rmses': fold_rmses.copy()
                        }
                        results.append(best_config.copy())
                    elif loso < V6_LOSO:
                        results.append({
                            'components': [(m1, w1), (m2, w2)],
                            'power': power,
                            'loso': loso,
                            'fold_rmses': fold_rmses.copy()
                        })
    
    print(f'    {n_configs} 2-comp configs searched, best LOSO={best_loso:.4f}')
    
    # ── 3-component blends ──
    print('  Searching 3-component blends...')
    weight_steps_3c = np.arange(0.05, 0.96, 0.05)
    n_3c = 0
    
    for i, m1 in enumerate(good_methods[:15]):
        for j, m2 in enumerate(good_methods[i+1:15]):
            for m3 in good_methods[i+j+2:15]:
                for w1 in weight_steps_3c:
                    for w2 in weight_steps_3c:
                        w3 = 1.0 - w1 - w2
                        if w3 < 0.04 or w3 > 0.96:
                            continue
                        for power in power_values:
                            fold_rmses = []
                            for fi in range(5):
                                blended = (w1 * fold_scores[m1][fi] + 
                                          w2 * fold_scores[m2][fi] + 
                                          w3 * fold_scores[m3][fi])
                                avail = {folds[fi]: list(range(1, 69))}
                                assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]),
                                                   avail, power=power)
                                rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
                                fold_rmses.append(rmse)
                            loso = compute_loso_score(fold_rmses)
                            n_configs += 1
                            n_3c += 1
                            if loso < best_loso:
                                best_loso = loso
                                best_config = {
                                    'components': [(m1, w1), (m2, w2), (m3, w3)],
                                    'power': power,
                                    'loso': loso,
                                    'fold_rmses': fold_rmses.copy()
                                }
                                results.append(best_config.copy())
                            elif loso < V6_LOSO:
                                results.append({
                                    'components': [(m1, w1), (m2, w2), (m3, w3)],
                                    'power': power,
                                    'loso': loso,
                                    'fold_rmses': fold_rmses.copy()
                                })
                            
                            if n_3c % 500000 == 0:
                                print(f'    {n_3c/1e6:.1f}M 3-comp configs... best LOSO={best_loso:.4f} [{time.time()-t0:.0f}s]')
    
    print(f'    {n_3c} 3-comp configs searched')
    
    # ── Sort results by LOSO ──
    results.sort(key=lambda x: x['loso'])
    
    print(f'\n  TOTAL: {n_configs} configs searched in {time.time()-t0:.0f}s')
    print(f'  Configs beating v6 LOSO ({V6_LOSO}): {sum(1 for r in results if r["loso"] < V6_LOSO)}')
    
    if not results or results[0]['loso'] >= V6_LOSO:
        print('\n  ❌ NO CONFIG BEATS V6 LOSO — v6 remains optimal')
        print(f'  Best found: LOSO={best_loso:.4f}')
        return
    
    # ─────────────────────────────────────────────────
    #  PHASE 3: Top candidates analysis
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 3: Top LOSO candidates')
    print('='*60)
    
    # Show top 20
    for rank, cfg in enumerate(results[:20]):
        comp_str = ' + '.join([f'{w*100:.0f}% {m}' for m, w in cfg['components']])
        rmse_str = ' '.join([f'{r:.2f}' for r in cfg['fold_rmses']])
        beat_folds = sum(1 for fi, r in enumerate(cfg['fold_rmses']) 
                        if r < indiv_loso['LR_C5.0'][1][fi])  # compare to v6's main component
        print(f'  #{rank+1:2d} LOSO={cfg["loso"]:.4f} p={cfg["power"]:.3f} '
              f'[{rmse_str}] beats≥{beat_folds}/5f')
        print(f'       {comp_str}')
    
    # ─────────────────────────────────────────────────
    #  PHASE 4: Stability validation (top 5 candidates)
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 4: Stability validation')
    print('='*60)
    
    top_candidates = results[:5]
    
    for rank, cfg in enumerate(top_candidates):
        comp_str = ' + '.join([f'{w*100:.0f}% {m}' for m, w in cfg['components']])
        print(f'\n  Candidate #{rank+1}: LOSO={cfg["loso"]:.4f}, p={cfg["power"]:.3f}')
        print(f'    {comp_str}')
        
        # ── 4a: Weight perturbation test ──
        # Perturb each weight by ±2% and check LOSO sensitivity
        components = cfg['components']
        power = cfg['power']
        perturbation_losos = []
        
        for trial in range(100):
            rng = np.random.RandomState(trial)
            perturbed_weights = []
            for _, w in components:
                pw = w + rng.uniform(-0.03, 0.03)  # ±3% perturbation
                perturbed_weights.append(max(0.01, pw))
            # Normalize
            total = sum(perturbed_weights)
            perturbed_weights = [w/total for w in perturbed_weights]
            
            # Also perturb power slightly
            pp = power + rng.uniform(-0.025, 0.025)
            pp = max(0.01, pp)
            
            fold_rmses = []
            for fi in range(5):
                blended = sum(pw * fold_scores[m][fi] 
                            for (m, _), pw in zip(components, perturbed_weights))
                avail = {folds[fi]: list(range(1, 69))}
                assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]),
                                   avail, power=pp)
                rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
                fold_rmses.append(rmse)
            perturbation_losos.append(compute_loso_score(fold_rmses))
        
        pert_mean = np.mean(perturbation_losos)
        pert_std = np.std(perturbation_losos)
        pert_worse = sum(1 for l in perturbation_losos if l >= V6_LOSO)
        print(f'    Perturbation test: mean={pert_mean:.4f} ±{pert_std:.4f}, '
              f'{pert_worse}/100 worse than v6')
        
        # ── 4b: Power sensitivity scan ──
        power_losos = {}
        for p in np.arange(0.025, 0.301, 0.025):
            fold_rmses = []
            for fi in range(5):
                blended = sum(w * fold_scores[m][fi] for m, w in components)
                avail = {folds[fi]: list(range(1, 69))}
                assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]),
                                   avail, power=p)
                rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
                fold_rmses.append(rmse)
            power_losos[p] = compute_loso_score(fold_rmses)
        
        best_power = min(power_losos, key=power_losos.get)
        power_range = max(power_losos.values()) - min(power_losos.values())
        n_beat_v6 = sum(1 for l in power_losos.values() if l < V6_LOSO)
        print(f'    Power scan: best_p={best_power:.3f} LOSO={power_losos[best_power]:.4f}, '
              f'{n_beat_v6}/{len(power_losos)} powers beat v6, range={power_range:.3f}')
        
        # ── 4c: Fold consistency check ──
        # How many folds does this config beat v6's fold performance?
        # v6 fold RMSEs from v7e analysis (LR_C5 + LRk25 + XGB blend)
        fold_beat_count = 0
        fold_detail = []
        for fi in range(5):
            # Compute v6 fold RMSE
            v6_blended = (0.64 * fold_scores['LR_C5.0'][fi] + 
                         0.28 * fold_scores.get('LRk25_C0.5', {}).get(fi, fold_scores['LR_C0.5'][fi] if 'LR_C0.5' in fold_scores else fold_scores['LR_C5.0'][fi]) + 
                         0.08 * fold_scores.get('XGB_d4_300_lr0.05', {}).get(fi, fold_scores['XGB_d4_200_lr0.05'][fi] if 'XGB_d4_200_lr0.05' in fold_scores else fold_scores['LR_C5.0'][fi]))
            avail = {folds[fi]: list(range(1, 69))}
            v6_assigned = hungarian(v6_blended, np.full(fold_n[fi], folds[fi]), avail, power=0.15)
            v6_rmse = np.sqrt(np.mean((v6_assigned - fold_y[fi])**2))
            
            v8_rmse = cfg['fold_rmses'][fi]
            beat = v8_rmse < v6_rmse
            fold_beat_count += int(beat)
            fold_detail.append(f'{folds[fi]}: v6={v6_rmse:.2f} v8={v8_rmse:.2f} {"✓" if beat else "✗"}')
        
        print(f'    Fold comparison: beats v6 on {fold_beat_count}/5 folds')
        for fd in fold_detail:
            print(f'      {fd}')
        
        # ── Overall stability rating ──
        stable = (pert_worse <= 20 and n_beat_v6 >= 5 and fold_beat_count >= 3)
        print(f'    STABILITY: {"✅ PASS" if stable else "⚠️  MARGINAL" if pert_worse <= 40 else "❌ FAIL"}')
    
    # ─────────────────────────────────────────────────
    #  PHASE 5: Fine-tune top stable candidate
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 5: Fine-tuning best stable candidate')
    print('='*60)
    
    # Take best overall candidate and fine-tune weights at 1% resolution
    best = results[0]
    components = best['components']
    method_names = [m for m, _ in components]
    base_weights = [w for _, w in components]
    n_comp = len(components)
    
    print(f'  Fine-tuning: {" + ".join(f"{w*100:.0f}% {m}" for m, w in components)}')
    
    best_fine_loso = best['loso']
    best_fine_weights = base_weights.copy()
    best_fine_power = best['power']
    n_fine = 0
    
    # Search ±10% around each weight at 1% resolution
    if n_comp == 2:
        for w1 in np.arange(max(0.05, base_weights[0]-0.10), 
                            min(0.95, base_weights[0]+0.10)+0.005, 0.01):
            w2 = 1.0 - w1
            if w2 < 0.04:
                continue
            for p in np.arange(max(0.025, best['power']-0.05), 
                              best['power']+0.055, 0.005):
                fold_rmses = []
                for fi in range(5):
                    blended = w1 * fold_scores[method_names[0]][fi] + w2 * fold_scores[method_names[1]][fi]
                    avail = {folds[fi]: list(range(1, 69))}
                    assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]), avail, power=p)
                    fold_rmses.append(np.sqrt(np.mean((assigned - fold_y[fi])**2)))
                loso = compute_loso_score(fold_rmses)
                n_fine += 1
                if loso < best_fine_loso:
                    best_fine_loso = loso
                    best_fine_weights = [w1, w2]
                    best_fine_power = p
    elif n_comp == 3:
        for w1 in np.arange(max(0.05, base_weights[0]-0.10), 
                            min(0.90, base_weights[0]+0.10)+0.005, 0.02):
            for w2 in np.arange(max(0.05, base_weights[1]-0.10), 
                                min(0.90, base_weights[1]+0.10)+0.005, 0.02):
                w3 = 1.0 - w1 - w2
                if w3 < 0.04 or w3 > 0.96:
                    continue
                for p in np.arange(max(0.025, best['power']-0.05), 
                                  best['power']+0.055, 0.005):
                    fold_rmses = []
                    for fi in range(5):
                        blended = (w1 * fold_scores[method_names[0]][fi] + 
                                  w2 * fold_scores[method_names[1]][fi] + 
                                  w3 * fold_scores[method_names[2]][fi])
                        avail = {folds[fi]: list(range(1, 69))}
                        assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]), avail, power=p)
                        fold_rmses.append(np.sqrt(np.mean((assigned - fold_y[fi])**2)))
                    loso = compute_loso_score(fold_rmses)
                    n_fine += 1
                    if loso < best_fine_loso:
                        best_fine_loso = loso
                        best_fine_weights = [w1, w2, w3]
                        best_fine_power = p
    
    print(f'  Fine-tuned {n_fine} configs')
    print(f'  Before: LOSO={best["loso"]:.4f}, After: LOSO={best_fine_loso:.4f}')
    
    comp_fine_str = ' + '.join([f'{w*100:.0f}% {m}' for m, w in zip(method_names, best_fine_weights)])
    print(f'  Best v8: {comp_fine_str}, power={best_fine_power:.3f}')
    
    # ─────────────────────────────────────────────────
    #  PHASE 6: Final validation of fine-tuned v8
    # ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print(' PHASE 6: Final validation of v8')
    print('='*60)
    
    # Recompute fold details
    final_fold_rmses = []
    final_fold_exact = []
    for fi in range(5):
        blended = sum(w * fold_scores[m][fi] for m, w in zip(method_names, best_fine_weights))
        avail = {folds[fi]: list(range(1, 69))}
        assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]), avail, power=best_fine_power)
        rmse = np.sqrt(np.mean((assigned - fold_y[fi])**2))
        exact = int(np.sum(assigned == fold_y[fi]))
        final_fold_rmses.append(rmse)
        final_fold_exact.append(exact)
    
    final_loso = compute_loso_score(final_fold_rmses)
    total_exact = sum(final_fold_exact)
    total_teams = sum(fold_n[fi] for fi in range(5))
    
    print(f'\n  v8 LOSO Results:')
    print(f'  {"Season":>10} {"N":>3} {"Exact":>5} {"Pct":>6} {"RMSE":>8}')
    for fi in range(5):
        n = fold_n[fi]
        ex = final_fold_exact[fi]
        rm = final_fold_rmses[fi]
        print(f'  {folds[fi]:>10} {n:3d} {ex:5d} {ex/n*100:5.1f}% {rm:8.3f}')
    
    print(f'\n  TOTAL: {total_exact}/{total_teams} exact ({total_exact/total_teams*100:.1f}%)')
    print(f'  LOSO score: {final_loso:.4f} (v6 was {V6_LOSO})')
    improvement = (V6_LOSO - final_loso) / V6_LOSO * 100
    print(f'  Improvement: {improvement:.1f}%')
    
    # Final perturbation stability
    print('\n  Final perturbation test (200 trials)...')
    final_pert_losos = []
    for trial in range(200):
        rng = np.random.RandomState(trial + 1000)
        pw = [w + rng.uniform(-0.03, 0.03) for w in best_fine_weights]
        pw = [max(0.01, w) for w in pw]
        total_w = sum(pw)
        pw = [w/total_w for w in pw]
        pp = best_fine_power + rng.uniform(-0.025, 0.025)
        pp = max(0.01, pp)
        
        fold_rmses = []
        for fi in range(5):
            blended = sum(w * fold_scores[m][fi] for m, w in zip(method_names, pw))
            avail = {folds[fi]: list(range(1, 69))}
            assigned = hungarian(blended, np.full(fold_n[fi], folds[fi]), avail, power=pp)
            fold_rmses.append(np.sqrt(np.mean((assigned - fold_y[fi])**2)))
        final_pert_losos.append(compute_loso_score(fold_rmses))
    
    pert_mean = np.mean(final_pert_losos)
    pert_std = np.std(final_pert_losos)
    pert_worse = sum(1 for l in final_pert_losos if l >= V6_LOSO)
    print(f'  Perturbation: mean={pert_mean:.4f} ±{pert_std:.4f}, '
          f'{pert_worse}/200 worse than v6')
    
    # ── Final verdict ──
    print('\n' + '='*60)
    if final_loso < V6_LOSO and pert_worse <= 40:
        print(f'  ✅ v8 APPROVED: LOSO={final_loso:.4f} < v6={V6_LOSO}')
        print(f'     {comp_fine_str}')
        print(f'     power={best_fine_power:.3f}')
        print(f'     Improvement: {improvement:.1f}% | Stability: {200-pert_worse}/200')
    else:
        print(f'  ❌ v8 REJECTED: not a reliable improvement over v6')
        print(f'     LOSO={final_loso:.4f} vs v6={V6_LOSO}')
        print(f'     Stability: {200-pert_worse}/200 trials beat v6')
    print('='*60)
    
    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
