#!/usr/bin/env python3
"""
v13b: Alternative scoring methods for Kaggle eval.

Key insight from v13: NO hyperparameter change breaks 57/91 ceiling.
The pairwise RANKING is already excellent (ρ>0.95), but the conversion
from ranking → Hungarian assignment loses information.

New ideas:
1. Raw probability-sum scoring (not rank-converted)
2. Calibrated probability scoring
3. Direct seed regression component
4. NET-anchored scoring for certain team types
5. Copeland-count scoring
6. Weighted pairwise scoring (weight by confidence)
7. Bradley-Terry model scoring
8. Ensemble of scoring methods
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def pairwise_score_raw(model, X_test, scaler=None):
    """Raw probability-sum scoring (NOT rank-converted).
    Returns the sum of win probabilities for each team.
    Higher = better team (lower seed)."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0
        scores[i] = probs.sum()
    return scores


def pairwise_score_rank(model, X_test, scaler=None):
    """Standard rank-based scoring (1=best, N=worst)."""
    raw = pairwise_score_raw(model, X_test, scaler)
    return np.argsort(np.argsort(-raw)).astype(float) + 1.0


def pairwise_score_copeland(model, X_test, scaler=None, threshold=0.5):
    """Copeland scoring: count definitive wins (prob > threshold)."""
    n = len(X_test)
    wins = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0.5
        wins[i] = (probs > threshold).sum()
    # Convert to rank
    return np.argsort(np.argsort(-wins)).astype(float) + 1.0


def pairwise_score_confidence_weighted(model, X_test, scaler=None):
    """Weight each pairwise comparison by its confidence.
    More confident predictions contribute more to the final score."""
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]
        probs[i] = 0.5
        # Weight by confidence: |p - 0.5| * 2
        confidence = np.abs(probs - 0.5) * 2
        weighted_probs = probs * confidence
        scores[i] = weighted_probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0


def raw_to_seed_scale(raw_scores, n_teams=68):
    """Convert raw probability sums to seed scale [1, n_teams].
    Linear mapping: highest raw → 1, lowest raw → n_teams."""
    ranked = np.argsort(np.argsort(-raw_scores))  # 0-based rank (0=best)
    return ranked.astype(float) + 1.0


def raw_to_seed_proportional(raw_scores, n_teams=68):
    """Proportional mapping: preserves relative distances.
    raw_max → 1, raw_min → n_teams, linear interpolation."""
    rmin, rmax = raw_scores.min(), raw_scores.max()
    if rmax == rmin:
        return np.full_like(raw_scores, n_teams / 2)
    # Higher raw = better = lower seed
    normalized = (rmax - raw_scores) / (rmax - rmin)  # 0=best, 1=worst
    return 1.0 + normalized * (n_teams - 1)


def main():
    t0 = time.time()
    print('=' * 70)
    print(' v13b: ALTERNATIVE SCORING METHODS')
    print('=' * 70)

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
    y_int = y.astype(int)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    teams = labeled['Team'].values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    print(f'  Teams: {n_labeled}, Test: {test_mask.sum()}')

    # ═══════════════════════════════════════════════════════════════
    #  Define scoring methods
    # ═══════════════════════════════════════════════════════════════
    
    scoring_methods = {
        'rank': pairwise_score_rank,
        'copeland_55': lambda m, X, s: pairwise_score_copeland(m, X, s, 0.55),
        'copeland_60': lambda m, X, s: pairwise_score_copeland(m, X, s, 0.60),
        'conf_weighted': pairwise_score_confidence_weighted,
        'raw_proportional': None,  # Handled specially
        'raw_rank': None,          # Handled specially  
    }
    
    # ═══════════════════════════════════════════════════════════════
    #  Run experiments
    # ═══════════════════════════════════════════════════════════════
    
    configs = []
    
    # For each scoring method, try with different powers and blend weights
    for score_method in ['rank', 'copeland_55', 'copeland_60', 'conf_weighted',
                          'raw_proportional', 'raw_rank']:
        for power in [0.15, 0.20]:
            for w1, w3, w4 in [(0.64, 0.28, 0.08), (0.70, 0.22, 0.08)]:
                configs.append({
                    'name': f'{score_method}_p{power}_w{w1}',
                    'score_method': score_method,
                    'power': power,
                    'w1': w1, 'w3': w3, 'w4': w4
                })
    
    # Also try direct regression blend
    for reg_weight in [0.0, 0.05, 0.10, 0.15, 0.20]:
        configs.append({
            'name': f'reg_blend_{reg_weight}',
            'score_method': 'rank',
            'power': 0.15,
            'w1': 0.64, 'w3': 0.28, 'w4': 0.08,
            'reg_weight': reg_weight
        })
    
    # Bradley-Terry inspired: use log-odds instead of probabilities
    configs.append({
        'name': 'log_odds_scoring',
        'score_method': 'log_odds',
        'power': 0.15,
        'w1': 0.64, 'w3': 0.28, 'w4': 0.08
    })
    
    # NET-anchored: blend pairwise score with NET rank
    for net_w in [0.05, 0.10, 0.15, 0.20, 0.30]:
        configs.append({
            'name': f'net_anchor_{net_w}',
            'score_method': 'rank',
            'power': 0.15,
            'w1': 0.64, 'w3': 0.28, 'w4': 0.08,
            'net_anchor': net_w
        })
    
    # Seed-line clipping: round to nearest seed line, then assign within
    configs.append({
        'name': 'seedline_round',
        'score_method': 'rank',
        'power': 0.15,
        'w1': 0.64, 'w3': 0.28, 'w4': 0.08,
        'seedline_clip': True
    })

    print(f'\n  Running {len(configs)} configs...')
    
    results = []
    
    for ci, cfg in enumerate(configs):
        assigned_all = np.zeros(n_labeled, dtype=int)
        
        for hold in folds:
            season_mask = (seasons == hold)
            season_indices = np.where(season_mask)[0]
            season_test_local = test_mask[season_mask]
            global_train_mask = ~(test_mask & season_mask)
            
            X_train = X_all[global_train_mask]
            y_train = y[global_train_mask]
            s_train = seasons[global_train_mask]
            X_season = X_all[season_mask]
            
            top_k_idx = select_top_k_features(
                X_train, y_train, feature_names, k=USE_TOP_K_A,
                forced_features=FORCE_FEATURES)[0]
            
            sm = cfg['score_method']
            
            # ── COMPONENT 1: LR C=5.0, adj-pairs ──
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
                X_train, y_train, s_train, max_gap=ADJ_COMP1_GAP)
            sc_adj = StandardScaler()
            pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
            lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_adj_sc, pw_y_adj)
            
            # ── COMPONENT 3: LR C=0.5, topK ──
            X_tr_k = X_train[:, top_k_idx]
            X_s_k = X_season[:, top_k_idx]
            pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, s_train)
            sc_k = StandardScaler()
            pw_X_k_sc = sc_k.fit_transform(pw_X_k)
            lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_X_k_sc, pw_y_k)
            
            # ── COMPONENT 4: XGB ──
            pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, s_train)
            sc_full = StandardScaler()
            pw_X_full_sc = sc_full.fit_transform(pw_X_full)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            xgb_clf.fit(pw_X_full_sc, pw_y_full)
            
            # ── Score with chosen method ──
            if sm == 'rank':
                s1 = pairwise_score_rank(lr1, X_season, sc_adj)
                s3 = pairwise_score_rank(lr3, X_s_k, sc_k)
                s4 = pairwise_score_rank(xgb_clf, X_season, sc_full)
            elif sm == 'copeland_55':
                s1 = pairwise_score_copeland(lr1, X_season, sc_adj, 0.55)
                s3 = pairwise_score_copeland(lr3, X_s_k, sc_k, 0.55)
                s4 = pairwise_score_copeland(xgb_clf, X_season, sc_full, 0.55)
            elif sm == 'copeland_60':
                s1 = pairwise_score_copeland(lr1, X_season, sc_adj, 0.60)
                s3 = pairwise_score_copeland(lr3, X_s_k, sc_k, 0.60)
                s4 = pairwise_score_copeland(xgb_clf, X_season, sc_full, 0.60)
            elif sm == 'conf_weighted':
                s1 = pairwise_score_confidence_weighted(lr1, X_season, sc_adj)
                s3 = pairwise_score_confidence_weighted(lr3, X_s_k, sc_k)
                s4 = pairwise_score_confidence_weighted(xgb_clf, X_season, sc_full)
            elif sm == 'raw_proportional':
                raw1 = pairwise_score_raw(lr1, X_season, sc_adj)
                raw3 = pairwise_score_raw(lr3, X_s_k, sc_k)
                raw4 = pairwise_score_raw(xgb_clf, X_season, sc_full)
                s1 = raw_to_seed_proportional(raw1)
                s3 = raw_to_seed_proportional(raw3)
                s4 = raw_to_seed_proportional(raw4)
            elif sm == 'raw_rank':
                raw1 = pairwise_score_raw(lr1, X_season, sc_adj)
                raw3 = pairwise_score_raw(lr3, X_s_k, sc_k)
                raw4 = pairwise_score_raw(xgb_clf, X_season, sc_full)
                s1 = raw_to_seed_scale(raw1)
                s3 = raw_to_seed_scale(raw3)
                s4 = raw_to_seed_scale(raw4)
            elif sm == 'log_odds':
                raw1 = pairwise_score_raw(lr1, X_season, sc_adj)
                raw3 = pairwise_score_raw(lr3, X_s_k, sc_k)
                raw4 = pairwise_score_raw(xgb_clf, X_season, sc_full)
                # Convert to log-odds scale, then to seed scale
                n_s = len(X_season)
                eps = 0.01
                for rr, out_name in [(raw1, 's1'), (raw3, 's3'), (raw4, 's4')]:
                    # Normalize to [0, 1]
                    pnorm = (rr - rr.min()) / (rr.max() - rr.min() + eps)
                    pnorm = np.clip(pnorm, eps, 1 - eps)
                    log_odds = np.log(pnorm / (1 - pnorm))
                    # Convert back to rank
                    rank_lo = np.argsort(np.argsort(-log_odds)).astype(float) + 1.0
                    if out_name == 's1': s1 = rank_lo
                    elif out_name == 's3': s3 = rank_lo
                    else: s4 = rank_lo
            else:
                raise ValueError(f'Unknown scoring method: {sm}')
            
            # Blend
            w1, w3, w4 = cfg['w1'], cfg['w3'], cfg['w4']
            blended = w1 * s1 + w3 * s3 + w4 * s4
            
            # Optional: blend with regression predictions
            if 'reg_weight' in cfg and cfg['reg_weight'] > 0:
                rw = cfg['reg_weight']
                # Ridge regression
                sc_reg = StandardScaler()
                X_tr_sc = sc_reg.fit_transform(X_train)
                ridge = Ridge(alpha=5.0)
                ridge.fit(X_tr_sc, y_train)
                reg_pred = ridge.predict(sc_reg.transform(X_season))
                blended = (1 - rw) * blended + rw * reg_pred
            
            # Optional: NET anchoring
            if 'net_anchor' in cfg:
                net_w = cfg['net_anchor']
                net_idx = feature_names.index('net_to_seed')
                net_seeds = X_season[:, net_idx]
                blended = (1 - net_w) * blended + net_w * net_seeds
            
            # Optional: seed-line clipping
            if cfg.get('seedline_clip', False):
                # Round to nearest seed-line midpoint (2.5, 6.5, 10.5, ...)
                seed_lines = np.ceil(blended / 4) * 4 - 2
                blended = seed_lines
            
            # Lock training teams
            for i, global_idx in enumerate(season_indices):
                if not test_mask[global_idx]:
                    blended[i] = y[global_idx]
            
            # Hungarian
            avail = {hold: list(range(1, 69))}
            assigned_s = hungarian(blended, seasons[season_mask], avail, power=cfg['power'])
            
            for i, global_idx in enumerate(season_indices):
                assigned_all[global_idx] = assigned_s[i]
        
        # Evaluate
        pred = assigned_all[test_mask]
        true = y_int[test_mask]
        exact = int((pred == true).sum())
        rmse = np.sqrt(np.mean((pred - true)**2))
        
        results.append({
            'name': cfg['name'],
            'exact': exact,
            'rmse': rmse,
        })
        
        if (ci + 1) % 10 == 0:
            print(f'    [{ci+1}/{len(configs)}]...', flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════════
    
    print('\n' + '=' * 70)
    print(' RESULTS (sorted by exact, then RMSE)')
    print('=' * 70)
    
    results.sort(key=lambda r: (-r['exact'], r['rmse']))
    
    print(f'\n  {"Config":<40} {"Exact":>8} {"RMSE":>8}')
    print(f'  {"─"*40} {"─"*8} {"─"*8}')
    
    for r in results:
        mark = ' ← baseline' if r['name'] == 'rank_p0.15_w0.64' else ''
        print(f'  {r["name"]:<40} {r["exact"]:>3}/91  {r["rmse"]:>8.4f}{mark}')

    # Show anything that beats 57
    better = [r for r in results if r['exact'] > 57]
    if better:
        print(f'\n  *** BEAT BASELINE (57/91): ***')
        for r in better:
            print(f'    {r["name"]}: {r["exact"]}/91, RMSE={r["rmse"]:.4f}')
    
    # Show best RMSE at 57 exact
    at_57 = [r for r in results if r['exact'] == 57]
    if at_57:
        best_57 = min(at_57, key=lambda r: r['rmse'])
        print(f'\n  Best at 57 exact: {best_57["name"]} RMSE={best_57["rmse"]:.4f}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
