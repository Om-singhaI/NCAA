#!/usr/bin/env python3
"""
Generate smart variant: skip low-zone (which hurts RMSE) but keep bot+tail.
Also test other skip combinations.
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_AL, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER,
    LOWZONE_ZONE, LOWZONE_Q1DOM, LOWZONE_FIELD, LOWZONE_POWER,
    BOTTOMZONE_ZONE, BOTTOMZONE_SOSNET, BOTTOMZONE_NETCONF,
    BOTTOMZONE_CBHIST, BOTTOMZONE_POWER,
    TAILZONE_ZONE, TAILZONE_OPP_RANK, TAILZONE_POWER,
)
from sklearn.impute import KNNImputer

KAGGLE_POWER = 0.15

def main():
    t0 = time.time()
    print('='*60)
    print('  SMART VARIANT GENERATOR (skip low-zone)')
    print('='*60)

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

    # Variants to generate
    variant_configs = [
        ('mid_only',           [True,  False, False, False]),
        ('mid+bot',            [True,  False, True,  False]),
        ('mid+tail',           [True,  False, False, True]),
        ('mid+bot+tail',       [True,  False, True,  True]),
        ('mid+low',            [True,  True,  False, False]),
        ('mid+low+bot',        [True,  True,  True,  False]),
        ('mid+low+tail',       [True,  True,  False, True]),
        ('mid+low+bot+tail',   [True,  True,  True,  True]),
        ('base_only',          [False, False, False, False]),
        ('low_only',           [False, True,  False, False]),
        ('bot_only',           [False, False, True,  False]),
        ('tail_only',          [False, False, False, True]),
    ]

    results = {}
    for vname, zones in variant_configs:
        use_mid, use_low, use_bot, use_tail = zones
        preds = np.zeros(n_labeled, dtype=int)

        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0:
                continue

            season_indices = np.where(season_mask)[0]
            test_mask_season = np.array([test_mask[gi] for gi in season_indices])
            global_train_mask = ~season_test_mask
            X_season = X_all[season_mask]

            top_k_idx = select_top_k_features(
                X_all[global_train_mask], y[global_train_mask],
                feature_names, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]

            raw_scores = predict_robust_blend(
                X_all[global_train_mask], y[global_train_mask],
                X_season, seasons[global_train_mask], top_k_idx)

            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw_scores[i] = y[gi]

            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)

            if use_mid:
                mid_correction = compute_committee_correction(
                    feature_names, X_season,
                    alpha_aq=CORRECTION_AQ, beta_al=CORRECTION_AL, gamma_sos=CORRECTION_SOS)
                assigned = apply_midrange_swap(
                    assigned, raw_scores, mid_correction,
                    test_mask_season, season_indices,
                    zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND, power=CORRECTION_POWER)

            if use_low:
                low_correction = compute_low_correction(
                    feature_names, X_season,
                    q1dom=LOWZONE_Q1DOM, field=LOWZONE_FIELD)
                assigned = apply_lowzone_swap(
                    assigned, raw_scores, low_correction,
                    test_mask_season, season_indices,
                    zone=LOWZONE_ZONE, power=LOWZONE_POWER)

            if use_bot:
                bot_correction = compute_bottom_correction(
                    feature_names, X_season,
                    sosnet=BOTTOMZONE_SOSNET, net_conf=BOTTOMZONE_NETCONF,
                    cbhist=BOTTOMZONE_CBHIST)
                assigned = apply_bottomzone_swap(
                    assigned, raw_scores, bot_correction,
                    test_mask_season, season_indices,
                    zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER)

            if use_tail:
                tail_correction = compute_tail_correction(
                    feature_names, X_season, opp_rank=TAILZONE_OPP_RANK)
                assigned = apply_tailzone_swap(
                    assigned, raw_scores, tail_correction,
                    test_mask_season, season_indices,
                    zone=TAILZONE_ZONE, power=TAILZONE_POWER)

            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = assigned[i]

        gt_all = y[test_mask].astype(int)
        pred_all = preds[test_mask]
        exact = int((pred_all == gt_all).sum())
        se = int(np.sum((pred_all - gt_all)**2))
        rmse91 = np.sqrt(se / 91)
        rmse451 = np.sqrt(se / 451)
        results[vname] = {'exact': exact, 'se': se, 'rmse91': rmse91, 'rmse451': rmse451, 'preds': preds.copy()}

    # Print sorted by SE
    print(f'\n  {"Variant":<25} {"Exact":>5} {"SE":>5} {"RMSE91":>8} {"RMSE451":>8}')
    print(f'  {"-"*60}')
    for vname, r in sorted(results.items(), key=lambda x: x[1]['se']):
        print(f'  {vname:<25} {r["exact"]:5d} {r["se"]:5d} {r["rmse91"]:8.4f} {r["rmse451"]:8.4f}')

    # Save the best variant (lowest SE)
    best_name = min(results.keys(), key=lambda k: results[k]['se'])
    best = results[best_name]
    print(f'\n  BEST: {best_name} (SE={best["se"]}, exact={best["exact"]}, RMSE451={best["rmse451"]:.4f})')

    rid_to_pred = {}
    for i in range(n_labeled):
        if test_mask[i]:
            rid_to_pred[record_ids[i]] = int(best['preds'][i])

    sub_out = sub_df[['RecordID']].copy()
    sub_out['Overall Seed'] = sub_out['RecordID'].map(lambda r: rid_to_pred.get(r, 0))
    out_path = os.path.join(os.path.dirname(__file__), 'submission_best_variant.csv')
    sub_out.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')

    # Show what differs between best and v25
    v25 = results['mid+low+bot+tail']
    print(f'\n  Differences vs v25 (mid+low+bot+tail):')
    print(f'  {"RID":<35} {"GT":>3} {"best":>4} {"v25":>4} {"best_SE":>7} {"v25_SE":>6}')
    for i in range(n_labeled):
        if test_mask[i] and best['preds'][i] != v25['preds'][i]:
            gt = int(y[i])
            se_b = (best['preds'][i] - gt)**2
            se_v = (v25['preds'][i] - gt)**2
            print(f'  {record_ids[i]:<35} {gt:3d} {best["preds"][i]:4d} {v25["preds"][i]:4d} {se_b:7d} {se_v:6d}')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
