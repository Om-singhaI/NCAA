#!/usr/bin/env python3
"""
v16: Hungarian Assignment Priors — Committee Seed Priors
========================================================

v15 LESSON: Adding features to the pairwise model HURTS (0/50 configs improved).
The pairwise model already captures feature relationships; more features = overfitting.

NEW INSIGHT: The pairwise model produces GREAT rankings (ρ=0.95-0.99) but the
ASSIGNMENT step (ranking → integer seeds) is where mid-tier teams go wrong.

Strategy: Modify the HUNGARIAN ASSIGNMENT step, not the pairwise model.

Current: cost[i][j] = |pw_score[i] - seed_j|^power
New:     cost[i][j] = α * |pw_score[i] - seed_j|^p1 + (1-α) * |prior[i] - seed_j|^p2

The "prior" is a committee-aware seed estimate that captures:
- AQ teams from weak conferences get penalized (seeded worse than NET)
- AL teams from power conferences get rewarded (seeded better than NET)
- Mid-range teams (seeds 17-34) need the most prior correction

Approaches tested:
1. Score blending: final_score = α * pw_score + (1-α) * regression_pred
2. Cost matrix priors: add prior term to Hungarian cost
3. Regression prior: Ridge/XGB regression seed prediction as prior
4. NET-conference prior: NET rank adjusted by conference strength
5. Adaptive blending: different α for different seed ranges
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# =================================================================
#  HUNGARIAN WITH PRIORS
# =================================================================

def hungarian_with_prior(pw_scores, prior_scores, seasons, avail,
                         alpha=0.8, power1=0.15, power2=0.3):
    """
    Hungarian assignment blending pairwise scores with committee priors.
    
    cost[i][j] = alpha * |pw_score[i] - seed_j|^power1 
               + (1-alpha) * |prior[i] - seed_j|^power2
    
    Args:
        pw_scores: raw pairwise ranking scores (continuous)
        prior_scores: committee prior seed estimates (continuous, ~1-68)
        alpha: weight on pairwise component (1.0 = pure pairwise = v12)
        power1: power for pairwise cost (default 0.15)
        power2: power for prior cost (default 0.3)
    """
    assigned = np.zeros(len(pw_scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        
        cost = np.zeros((len(si), len(pos)))
        for idx, i in enumerate(si):
            for jdx, p in enumerate(pos):
                pw_cost = abs(pw_scores[i] - p) ** power1
                prior_cost = abs(prior_scores[i] - p) ** power2
                cost[idx, jdx] = alpha * pw_cost + (1 - alpha) * prior_cost
        
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def hungarian_standard(scores, seasons, avail, power=0.15):
    """Standard Hungarian (v12 baseline)."""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[si[r]] = pos[c]
    return assigned


def hungarian_blended_score(pw_scores, prior_scores, seasons, avail,
                            alpha=0.8, power=0.15):
    """
    Blend scores BEFORE Hungarian: final = alpha*pw + (1-alpha)*prior
    Then run standard Hungarian on the blended scores.
    """
    blended = alpha * pw_scores + (1 - alpha) * prior_scores
    return hungarian_standard(blended, seasons, avail, power=power)


# =================================================================
#  PRIOR GENERATORS
# =================================================================

def compute_ridge_prior(X_train, y_train, X_test, alpha=5.0):
    """Ridge regression prior: predict seed directly from features."""
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_te_sc = sc.transform(X_test)
    model = Ridge(alpha=alpha)
    model.fit(X_tr_sc, y_train)
    prior = model.predict(X_te_sc)
    return np.clip(prior, 1, 68)


def compute_xgb_prior(X_train, y_train, X_test):
    """XGB regression prior: predict seed directly from features."""
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    prior = model.predict(X_test)
    return np.clip(prior, 1, 68)


def compute_rf_prior(X_train, y_train, X_test):
    """RF regression prior."""
    model = RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2,
        max_features=0.5, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    prior = model.predict(X_test)
    return np.clip(prior, 1, 68)


def compute_ensemble_prior(X_train, y_train, X_test):
    """Ensemble prior: average of Ridge + XGB + RF."""
    ridge_prior = compute_ridge_prior(X_train, y_train, X_test)
    xgb_prior = compute_xgb_prior(X_train, y_train, X_test)
    rf_prior = compute_rf_prior(X_train, y_train, X_test)
    return (ridge_prior + xgb_prior + rf_prior) / 3.0


def compute_net_conf_prior(X_data, feature_names, labeled_data=None):
    """
    Simple committee-formula prior based on NET rank and conference.
    
    For each team, estimate seed = NET_rank_among_tournament_teams,
    adjusted by conference strength. This captures the committee's
    tendency to penalize good-NET teams from weak conferences.
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    
    net = X_data[:, fi['NET Rank']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    sos = X_data[:, fi['NETSOS']]
    
    # Base: tournament field rank (NET-based ordering)
    if 'tourn_field_rank' in fi:
        base = X_data[:, fi['tourn_field_rank']]
    else:
        base = np.argsort(np.argsort(net)).astype(float) + 1.0
    
    # Conference adjustment: weak conference AQ teams get pushed down
    # Strong conference AL teams get pulled up  
    conf_weakness = (conf_avg - 100) / 200  # normalized: 0 = strong, 1 = weak
    
    # AQ penalty: weak conference = higher (worse) seed
    aq_adj = is_aq * conf_weakness * 8.0  # push AQ from weak conf down by up to 8
    
    # AL benefit: strong conference = lower (better) seed (already captured partially)
    al_adj = -is_al * (1 - conf_weakness) * 2.0  # slight pull up for strong conf AL
    
    prior = base + aq_adj + al_adj
    return np.clip(prior, 1, 68)


# =================================================================
#  PAIRWISE PREDICTION (v12 unchanged)
# =================================================================

def predict_pairwise_v12(X_train, y_train, X_test, seasons_train, 
                         feature_names, top_k_idx):
    """v12 pairwise blend, returns raw continuous scores."""
    # Component 1: PW-LR C=5.0, adj-pairs gap≤30 (64%)
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
        X_train, y_train, seasons_train, max_gap=ADJ_COMP1_GAP)
    sc_adj = StandardScaler()
    pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
    lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_adj_sc, pw_y_adj)
    score1 = pairwise_score(lr1, X_test, sc_adj)
    
    # Component 3: PW-LR C=0.5, top-25 features (28%)
    X_tr_k = X_train[:, top_k_idx]
    X_te_k = X_test[:, top_k_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    pw_X_k_sc = sc_k.fit_transform(pw_X_k)
    lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_X_k_sc, pw_y_k)
    score3 = pairwise_score(lr3, X_te_k, sc_k)
    
    # Component 4: PW-XGB d4/300/0.05 (8%)
    pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, seasons_train)
    sc_full = StandardScaler()
    pw_X_full_sc = sc_full.fit_transform(pw_X_full)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_full_sc, pw_y_full)
    score4 = pairwise_score(xgb_clf, X_test, sc_full)
    
    # v12 blend
    return BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4


# =================================================================
#  KAGGLE-STYLE EVALUATION
# =================================================================

def kaggle_eval(X_all, y, seasons, feature_names, test_mask, folds,
                prior_type='none', blend_method='score',
                alpha=0.8, power1=0.15, power2=0.3,
                prior_alpha_ridge=5.0):
    """
    Full Kaggle-style evaluation with committee priors.
    
    Args:
        prior_type: 'none' (v12), 'ridge', 'xgb', 'rf', 'ensemble', 'net_conf'
        blend_method: 'score' (blend before Hungarian), 'cost' (blend in cost matrix)
        alpha: weight on pairwise component (1.0 = v12 baseline)
        power1: power for pairwise cost in Hungarian
        power2: power for prior cost (only used if blend_method='cost')
    """
    test_assigned = np.zeros(len(y), dtype=int)
    
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
            continue
        
        global_train_mask = ~season_test_mask
        
        # Feature selection
        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        
        # Pairwise v12 scores for ALL teams in this season
        pw_scores = predict_pairwise_v12(
            X_all[global_train_mask], y[global_train_mask],
            X_all[season_mask], seasons[global_train_mask],
            feature_names, top_k_idx)
        
        # Lock training teams to known seeds
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                pw_scores[i] = y[global_idx]
        
        # Compute prior if needed
        if prior_type != 'none' and alpha < 1.0:
            if prior_type == 'ridge':
                prior = compute_ridge_prior(
                    X_all[global_train_mask], y[global_train_mask],
                    X_all[season_mask], alpha=prior_alpha_ridge)
            elif prior_type == 'xgb':
                prior = compute_xgb_prior(
                    X_all[global_train_mask], y[global_train_mask],
                    X_all[season_mask])
            elif prior_type == 'rf':
                prior = compute_rf_prior(
                    X_all[global_train_mask], y[global_train_mask],
                    X_all[season_mask])
            elif prior_type == 'ensemble':
                prior = compute_ensemble_prior(
                    X_all[global_train_mask], y[global_train_mask],
                    X_all[season_mask])
            elif prior_type == 'net_conf':
                prior = compute_net_conf_prior(
                    X_all[season_mask], feature_names)
            else:
                prior = pw_scores.copy()
            
            # Lock training teams in prior too
            for i, global_idx in enumerate(season_indices):
                if not test_mask[global_idx]:
                    prior[i] = y[global_idx]
            
            avail = {hold: list(range(1, 69))}
            
            if blend_method == 'score':
                assigned = hungarian_blended_score(
                    pw_scores, prior, seasons[season_mask], avail,
                    alpha=alpha, power=power1)
            elif blend_method == 'cost':
                assigned = hungarian_with_prior(
                    pw_scores, prior, seasons[season_mask], avail,
                    alpha=alpha, power1=power1, power2=power2)
            else:
                assigned = hungarian_standard(
                    pw_scores, seasons[season_mask], avail, power=power1)
        else:
            # Pure v12 (no prior)
            avail = {hold: list(range(1, 69))}
            assigned = hungarian_standard(
                pw_scores, seasons[season_mask], avail, power=power1)
        
        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                test_assigned[global_idx] = assigned[i]
    
    # Evaluate
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    exact = int((pred == gt).sum())
    rmse = np.sqrt(np.mean((pred - gt)**2))
    
    # Mid-range accuracy (seeds 17-34)
    mid_mask = (gt >= 17) & (gt <= 34)
    mid_exact = int((pred[mid_mask] == gt[mid_mask]).sum())
    mid_total = int(mid_mask.sum())
    
    return exact, rmse, mid_exact, mid_total, test_assigned


# =================================================================
#  MAIN
# =================================================================

def main():
    t0 = time.time()
    print('='*70)
    print(' v16: HUNGARIAN ASSIGNMENT PRIORS')
    print(' (Modify assignment step, NOT pairwise model)')
    print('='*70)

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
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    n_test = test_mask.sum()
    print(f'  Teams: {n_labeled}, Test: {n_test}')
    print(f'  Features: {len(feature_names)}')

    # ─── Verify v12 baseline ───
    print('\n  Verifying v12 baseline...')
    exact_v12, rmse_v12, mid_v12, mid_total, _ = kaggle_eval(
        X_all, y, seasons, feature_names, test_mask, folds,
        prior_type='none', alpha=1.0)
    print(f'  v12 baseline: {exact_v12}/{n_test} exact, RMSE={rmse_v12:.4f}, '
          f'mid={mid_v12}/{mid_total}')

    # ─── Test configurations ───
    print('\n' + '='*70)
    print(' TESTING HUNGARIAN PRIOR CONFIGURATIONS')
    print('='*70)

    configs = []

    # === APPROACH 1: Score blending (blend pw + regression before Hungarian) ===
    for prior_type in ['ridge', 'xgb', 'rf', 'ensemble', 'net_conf']:
        for alpha in [0.95, 0.90, 0.85, 0.80, 0.70]:
            configs.append({
                'name': f'score_{prior_type}_a{alpha:.2f}',
                'prior_type': prior_type,
                'blend_method': 'score',
                'alpha': alpha,
                'power1': 0.15,
            })

    # === APPROACH 2: Cost matrix blending (prior in Hungarian cost) ===
    for prior_type in ['ridge', 'xgb', 'ensemble']:
        for alpha in [0.90, 0.80, 0.70]:
            for power2 in [0.15, 0.30, 0.50]:
                configs.append({
                    'name': f'cost_{prior_type}_a{alpha:.2f}_p2={power2:.2f}',
                    'prior_type': prior_type,
                    'blend_method': 'cost',
                    'alpha': alpha,
                    'power1': 0.15,
                    'power2': power2,
                })
    
    # === APPROACH 3: Score blend with different Hungarian powers ===
    for prior_type in ['ridge', 'ensemble']:
        for alpha in [0.90, 0.85]:
            for power1 in [0.10, 0.20, 0.25]:
                configs.append({
                    'name': f'score_{prior_type}_a{alpha:.2f}_p={power1:.2f}',
                    'prior_type': prior_type,
                    'blend_method': 'score',
                    'alpha': alpha,
                    'power1': power1,
                })

    # === APPROACH 4: Pure regression through Hungarian (test baselines) ===
    for prior_type in ['ridge', 'xgb', 'rf', 'ensemble']:
        configs.append({
            'name': f'pure_{prior_type}',
            'prior_type': prior_type,
            'blend_method': 'score',
            'alpha': 0.0,
            'power1': 0.15,
        })

    n_configs = len(configs)
    print(f'\n  Running {n_configs} configs...\n')
    print(f'  {"Config":<50} {"Exact":>7} {"RMSE":>8} {"Mid":>12}')
    print(f'  {"─"*50} {"─"*7} {"─"*8} {"─"*12}')

    best_exact = exact_v12
    best_rmse = rmse_v12
    best_config = 'v12 baseline'
    results = []

    for ci, cfg in enumerate(configs):
        exact, rmse, mid_exact, mid_total, assigned = kaggle_eval(
            X_all, y, seasons, feature_names, test_mask, folds,
            prior_type=cfg['prior_type'],
            blend_method=cfg['blend_method'],
            alpha=cfg['alpha'],
            power1=cfg['power1'],
            power2=cfg.get('power2', 0.3))
        
        marker = ''
        if exact > best_exact or (exact == best_exact and rmse < best_rmse):
            best_exact = exact
            best_rmse = rmse
            best_config = cfg['name']
            marker = ' ★★★'
        elif exact == exact_v12:
            marker = ' ◆ (tied v12)'
        elif exact > exact_v12:
            marker = ' ★ IMPROVED!'
        
        print(f'  {cfg["name"]:<50} {exact:>3}/{n_test} {rmse:>8.4f} '
              f'{mid_exact}/{mid_total:<6}{marker}')
        
        results.append({
            'name': cfg['name'],
            'exact': exact,
            'rmse': rmse,
            'mid_exact': mid_exact,
            'mid_total': mid_total,
            **cfg,
        })
        
        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'  ... [{ci+1}/{n_configs}] ({elapsed:.0f}s)')

    # ─── Summary ───
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    
    print(f'\n  v12 baseline:     {exact_v12}/{n_test} exact, RMSE={rmse_v12:.4f}, '
          f'mid={mid_v12}/{mid_total}')
    print(f'  Best config:      {best_config}')
    print(f'  Best result:      {best_exact}/{n_test} exact, RMSE={best_rmse:.4f}')
    
    if best_exact > exact_v12:
        improvement = best_exact - exact_v12
        print(f'\n  ★★★ IMPROVEMENT: +{improvement} exact matches! ★★★')
    elif best_exact == exact_v12:
        print(f'\n  No exact match improvement found.')
    else:
        print(f'\n  All configs performed worse than v12.')

    # Top 10 configs
    results.sort(key=lambda x: (-x['exact'], x['rmse']))
    print(f'\n  Top 10 configs:')
    print(f'  {"Rank":>4} {"Config":<50} {"Exact":>7} {"RMSE":>8} {"Mid":>7}')
    print(f'  {"─"*4} {"─"*50} {"─"*7} {"─"*8} {"─"*7}')
    for i, r in enumerate(results[:10]):
        print(f'  {i+1:>4} {r["name"]:<50} {r["exact"]:>3}/{n_test} '
              f'{r["rmse"]:>8.4f} {r["mid_exact"]}/{r["mid_total"]}')

    # Group by approach
    print(f'\n  Best by approach:')
    approaches = {}
    for r in results:
        approach = r['name'].split('_')[0]
        if approach not in approaches or r['exact'] > approaches[approach]['exact'] or \
           (r['exact'] == approaches[approach]['exact'] and r['rmse'] < approaches[approach]['rmse']):
            approaches[approach] = r
    
    for approach, r in sorted(approaches.items()):
        print(f'    {approach:<12}: {r["name"]:<45} {r["exact"]}/{n_test} RMSE={r["rmse"]:.4f}')

    # Group by prior type
    print(f'\n  Best by prior type:')
    prior_types = {}
    for r in results:
        pt = r['prior_type']
        if pt not in prior_types or r['exact'] > prior_types[pt]['exact'] or \
           (r['exact'] == prior_types[pt]['exact'] and r['rmse'] < prior_types[pt]['rmse']):
            prior_types[pt] = r
    
    for pt, r in sorted(prior_types.items()):
        print(f'    {pt:<12}: {r["name"]:<45} {r["exact"]}/{n_test} RMSE={r["rmse"]:.4f}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
