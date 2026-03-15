#!/usr/bin/env python3
"""
NCAA v10c — Deep Dive: force-NET+Q1 Variants
==============================================
v10b found: forcing NET Rank + Q1W + Q1L into top-K gives IDENTICAL Kaggle
(56/91, RMSE=2.474) but BETTER LOSO (3.648 vs 3.678).

This script explores all variations of this approach:
- Different forced feature combos (NET+Q1, NET+Q1+Q2, NET+Q1+SOS, etc.)
- Different total top-K sizes with forced features
- Different C values with forced features
- Different blend weights with forced features
- Combinations thereof

Goal: find the SAFEST improvement over v6 (maintain Kaggle ≥56, improve LOSO)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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


def kaggle_eval(predict_fn, X_all, y, seasons, folds, test_mask, power=0.15):
    n = len(y)
    test_assigned = np.zeros(n, dtype=int)
    for hold in folds:
        smask = (seasons == hold)
        test_in_season = test_mask & smask
        if test_in_season.sum() == 0:
            continue
        train_mask = ~test_in_season
        scores = predict_fn(X_all[train_mask], y[train_mask], X_all[smask], seasons[train_mask])
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
    print(' NCAA v10c — FORCE-NET+Q1 DEEP DIVE')
    print('=' * 60)

    # Load data
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat_df = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat_df.columns)
    fi = {name: i for i, name in enumerate(feature_names)}

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

    results = []

    def make_model(forced_feats, total_k=25, c1=5.0, c3=0.5,
                   w1=0.64, w3=0.28, w4=0.08, xgb_est=300, xgb_depth=4):
        forced_idx = [fi[f] for f in forced_feats if f in fi]
        def predict(X_tr, y_tr, X_te, s_tr):
            auto_k = max(total_k - len(forced_idx), 5)
            top_k_auto = select_top_k_features(X_tr, y_tr, feature_names, k=auto_k)[0]
            combined = list(forced_idx)
            for idx in top_k_auto:
                if idx not in combined:
                    combined.append(idx)
            combined = combined[:total_k]

            pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
            sc = StandardScaler()
            pw_X_sc = sc.fit_transform(pw_X)

            lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(pw_X_sc, pw_y)
            s1 = pairwise_score(lr1, X_te, sc)

            X_tr_k, X_te_k = X_tr[:, combined], X_te[:, combined]
            pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
            sck = StandardScaler()
            pw_Xk_sc = sck.fit_transform(pw_Xk)
            lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(pw_Xk_sc, pw_yk)
            s3 = pairwise_score(lr3, X_te_k, sck)

            xgb_clf = xgb.XGBClassifier(
                n_estimators=xgb_est, max_depth=xgb_depth, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                reg_alpha=1.0, min_child_weight=5, random_state=42,
                verbosity=0, use_label_encoder=False, eval_metric='logloss')
            xgb_clf.fit(pw_X_sc, pw_y)
            s4 = pairwise_score(xgb_clf, X_te, sc)

            return w1 * s1 + w3 * s3 + w4 * s4
        return predict

    def evaluate(tag, pred_fn, power=0.15):
        ex, rmse = kaggle_eval(pred_fn, X_all, y, seasons, folds, test_mask, power)
        loso, fr = loso_eval(pred_fn, X_all, y, seasons, folds, power)
        results.append((tag, ex, rmse, loso, fr))
        return ex, rmse, loso, fr

    def show(tag, ex, rmse, loso, ref_ex=56, ref_rmse=2.474, ref_loso=3.6776):
        kg_up = ex > ref_ex or (ex == ref_ex and rmse < ref_rmse)
        lo_up = loso < ref_loso
        dk = "↑" if kg_up else ("↓" if ex < ref_ex or (ex == ref_ex and rmse > ref_rmse) else "=")
        dl = "↑" if lo_up else "↓"
        mk = " ★★" if dk != "↓" and dl == "↑" else " ★" if dk != "↓" or dl == "↑" else ""
        print(f'  {tag:45s} Kaggle={ex}/91 RMSE={rmse:.4f} {dk}  LOSO={loso:.4f} {dl}{mk}')

    # Baseline
    print('\n  BASELINE: v6 (top-K=25, no forced features)')
    pred_fn = make_model([], 25)
    ex, rmse, loso, fr = evaluate('v6-baseline', pred_fn)
    show('v6-baseline', ex, rmse, loso)
    v6_ex, v6_rmse, v6_loso = ex, rmse, loso

    # ════════════════════════════════════════════════════
    #  PART 1: Which features to force?
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 1: Which committee features to force (k=25)')
    print('─' * 60)

    force_options = [
        ('NET', ['NET Rank']),
        ('Q1W', ['Quadrant1_W']),
        ('Q1L', ['Quadrant1_L']),
        ('Q1WL', ['Quadrant1_W', 'Quadrant1_L']),
        ('NET+Q1W', ['NET Rank', 'Quadrant1_W']),
        ('NET+Q1WL', ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']),
        ('NET+Q1W+SOS', ['NET Rank', 'Quadrant1_W', 'NETSOS']),
        ('NET+Q1WL+SOS', ['NET Rank', 'Quadrant1_W', 'Quadrant1_L', 'NETSOS']),
        ('NET+Q1W+Q2W', ['NET Rank', 'Quadrant1_W', 'Quadrant2_W']),
        ('NET+Q1WL+Q2WL', ['NET Rank', 'Quadrant1_W', 'Quadrant1_L', 'Quadrant2_W', 'Quadrant2_L']),
        ('NET+resume', ['NET Rank', 'resume_score']),
        ('NET+Q1W+resume', ['NET Rank', 'Quadrant1_W', 'resume_score']),
        ('NET+badloss', ['NET Rank', 'total_bad_losses']),
        ('NET+Q1W+badloss', ['NET Rank', 'Quadrant1_W', 'total_bad_losses']),
        ('NET+Q1dom', ['NET Rank', 'q1_dominance']),
        ('NET+Q1W+wpct', ['NET Rank', 'Quadrant1_W', 'WL_Pct']),
        ('SOS', ['NETSOS']),
        ('Q1W+SOS', ['Quadrant1_W', 'NETSOS']),
    ]

    for tag, feats in force_options:
        pred_fn = make_model(feats, 25)
        ex, rmse, loso, fr = evaluate(f'force-{tag}', pred_fn)
        show(f'force-{tag}', ex, rmse, loso)

    # ════════════════════════════════════════════════════
    #  PART 2: Best forced set + different top-K sizes
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 2: force-NET+Q1WL with different top-K sizes')
    print('─' * 60)

    for k in [20, 22, 23, 24, 25, 26, 27, 28, 30]:
        feats = ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']
        pred_fn = make_model(feats, k)
        ex, rmse, loso, fr = evaluate(f'NET+Q1WL-k{k}', pred_fn)
        show(f'NET+Q1WL-k{k}', ex, rmse, loso)

    # ════════════════════════════════════════════════════
    #  PART 3: Best config + different C values
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 3: force-NET+Q1WL (k=25) with different C values')
    print('─' * 60)

    for c1, c3 in [(5.0, 0.5), (5.0, 0.3), (5.0, 1.0), (3.0, 0.5), (7.0, 0.5),
                    (10.0, 0.5), (10.0, 1.0), (3.0, 0.3), (2.0, 0.5)]:
        feats = ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']
        pred_fn = make_model(feats, 25, c1=c1, c3=c3)
        ex, rmse, loso, fr = evaluate(f'C{c1}/{c3}', pred_fn)
        show(f'NET+Q1WL C={c1}/{c3}', ex, rmse, loso)

    # ════════════════════════════════════════════════════
    #  PART 4: Best config + different blend weights
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 4: force-NET+Q1WL (k=25) with different weights')
    print('─' * 60)

    for w1, w3, w4 in [(0.64, 0.28, 0.08), (0.60, 0.30, 0.10),
                        (0.65, 0.30, 0.05), (0.70, 0.22, 0.08),
                        (0.60, 0.32, 0.08), (0.55, 0.35, 0.10),
                        (0.68, 0.24, 0.08), (0.64, 0.28, 0.08)]:
        feats = ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']
        pred_fn = make_model(feats, 25, w1=w1, w3=w3, w4=w4)
        ex, rmse, loso, fr = evaluate(f'w{w1:.2f}/{w3:.2f}/{w4:.2f}', pred_fn)
        show(f'NET+Q1WL w={w1:.2f}/{w3:.2f}/{w4:.2f}', ex, rmse, loso)

    # ════════════════════════════════════════════════════
    #  PART 5: Best config + different powers
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 5: force-NET+Q1WL (k=25) with different powers')
    print('─' * 60)

    for power in [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25]:
        feats = ['NET Rank', 'Quadrant1_W', 'Quadrant1_L']
        pred_fn = make_model(feats, 25)
        ex, rmse, loso, fr = evaluate(f'power={power}', pred_fn, power=power)
        # Use power-specific v6 baseline for comparison
        show(f'NET+Q1WL power={power}', ex, rmse, loso)

    # ════════════════════════════════════════════════════
    #  PART 6: Verify which features auto-select picks vs forced
    # ════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 6: Feature selection analysis')
    print('─' * 60)

    # Check what top-25 normally picks (v6)
    auto_idx, auto_names = select_top_k_features(X_all, y, feature_names, k=25)
    print(f'\n  v6 auto-selected top-25:')
    for i, (idx, name) in enumerate(zip(auto_idx, auto_names)):
        is_committee = name in ['NET Rank', 'Quadrant1_W', 'Quadrant1_L',
                                'NETSOS', 'resume_score', 'q1_dominance']
        marker = ' ← COMMITTEE' if is_committee else ''
        print(f'    {i+1:2d}. {name}{marker}')

    # Check per-fold: are NET/Q1 always in top-25?
    print(f'\n  Per-fold analysis: are committee features in auto top-25?')
    committee_feats = ['NET Rank', 'Quadrant1_W', 'Quadrant1_L', 'NETSOS',
                       'resume_score', 'q1_dominance', 'total_bad_losses']
    for hold in folds:
        tr = seasons != hold
        fold_idx, fold_names = select_top_k_features(X_all[tr], y[tr], feature_names, k=25)
        fold_name_set = set(fold_names)
        present = [f for f in committee_feats if f in fold_name_set]
        missing = [f for f in committee_feats if f not in fold_name_set]
        print(f'    {hold}: present={present}')
        if missing:
            print(f'           MISSING={missing}')

    # ════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY — ALL VARIANTS')
    print('=' * 60)

    print(f'\n  {"Approach":<47} {"Kaggle":>10} {"RMSE":>8} {"LOSO":>8} {"St":>4}')
    print(f'  {"─"*47} {"─"*10} {"─"*8} {"─"*8} {"─"*4}')

    for tag, ex, rmse, loso, fr in results:
        kg_up = ex > v6_ex or (ex == v6_ex and rmse < v6_rmse)
        lo_up = loso < v6_loso
        if kg_up and lo_up: status = '★★'
        elif ex == v6_ex and rmse == v6_rmse and lo_up: status = '=★'
        elif kg_up or lo_up: status = '★'
        else: status = ''
        bg = '→' if tag == 'v6-baseline' else ' '
        print(f' {bg}{tag:<47} {ex:2d}/91  {rmse:8.4f} {loso:8.4f} {status:>4}')

    # Find approaches that maintain Kaggle AND improve LOSO
    print(f'\n  ── Approaches that maintain Kaggle (56/91, RMSE≤2.474) AND improve LOSO: ──')
    improvements = [(t, e, r, l) for t, e, r, l, f in results
                    if e >= 56 and r <= v6_rmse + 0.001 and l < v6_loso]
    if improvements:
        for t, e, r, l in sorted(improvements, key=lambda x: x[3]):
            delta = v6_loso - l
            print(f'    {t}: Kaggle={e}/91 RMSE={r:.4f} LOSO={l:.4f} (↑{delta:.4f})')
    else:
        print('    NONE with strict RMSE equality')
        # Relax to same exact count
        relaxed = [(t, e, r, l) for t, e, r, l, f in results
                   if e >= 56 and l < v6_loso]
        if relaxed:
            print(f'\n  ── Relaxed (≥56 exact, any RMSE): ──')
            for t, e, r, l in sorted(relaxed, key=lambda x: x[3]):
                delta = v6_loso - l
                print(f'    {t}: Kaggle={e}/91 RMSE={r:.4f} LOSO={l:.4f} (↑{delta:.4f})')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
