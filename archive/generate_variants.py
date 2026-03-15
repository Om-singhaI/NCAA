#!/usr/bin/env python3
"""
Generate multiple submission variants with different zone correction combinations.
The user's best Kaggle score (1.09) may have come from fewer/different zones.
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
    print('  SUBMISSION VARIANT GENERATOR')
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

    # Generate raw predictions + pass 1 (Hungarian) per season
    raw_scores_all = np.zeros(n_labeled)
    pass1_all = np.zeros(n_labeled, dtype=int)

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
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
        pass1_assigned = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)

        for i, gi in enumerate(season_indices):
            raw_scores_all[gi] = raw_scores[i]
            pass1_all[gi] = pass1_assigned[i]

        # Compute all zone corrections for this season
        mid_correction = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=CORRECTION_AQ, beta_al=CORRECTION_AL, gamma_sos=CORRECTION_SOS)
        pass2_assigned = apply_midrange_swap(
            pass1_assigned, raw_scores, mid_correction,
            test_mask_season, season_indices,
            zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND, power=CORRECTION_POWER)

        low_correction = compute_low_correction(
            feature_names, X_season,
            q1dom=LOWZONE_Q1DOM, field=LOWZONE_FIELD)
        pass3_assigned = apply_lowzone_swap(
            pass2_assigned, raw_scores, low_correction,
            test_mask_season, season_indices,
            zone=LOWZONE_ZONE, power=LOWZONE_POWER)

        bot_correction = compute_bottom_correction(
            feature_names, X_season,
            sosnet=BOTTOMZONE_SOSNET, net_conf=BOTTOMZONE_NETCONF,
            cbhist=BOTTOMZONE_CBHIST)
        pass4_assigned = apply_bottomzone_swap(
            pass3_assigned, raw_scores, bot_correction,
            test_mask_season, season_indices,
            zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER)

        tail_correction = compute_tail_correction(
            feature_names, X_season, opp_rank=TAILZONE_OPP_RANK)
        pass5_assigned = apply_tailzone_swap(
            pass4_assigned, raw_scores, tail_correction,
            test_mask_season, season_indices,
            zone=TAILZONE_ZONE, power=TAILZONE_POWER)

        # Store all variant predictions
        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                variants[0][gi] = pass1_assigned[i]  # v12 base
                variants[1][gi] = pass2_assigned[i]  # + mid
                variants[2][gi] = pass3_assigned[i]  # + mid + low
                variants[3][gi] = pass4_assigned[i]  # + mid + low + bot
                variants[4][gi] = pass5_assigned[i]  # + mid + low + bot + tail (v25)

    # Evaluate all variants
    gt_all = y[test_mask].astype(int)

    print(f'\n  {"Variant":<35} {"Exact":>5} {"RMSE91":>8} {"RMSE451":>8} {"SE":>5}')
    print(f'  {"-"*70}')

    names = [
        'v12 base (no zones)',
        'v12 + mid-range',
        'v12 + mid + low',
        'v12 + mid + low + bot',
        'v12 + mid + low + bot + tail (v25)',
    ]

    for idx, name in enumerate(names):
        pred = variants[idx][test_mask]
        exact = int((pred == gt_all).sum())
        se = int(np.sum((pred - gt_all)**2))
        rmse91 = np.sqrt(se / 91)
        rmse451 = np.sqrt(se / 451)
        print(f'  {name:<35} {exact:5d} {rmse91:8.4f} {rmse451:8.4f} {se:5d}')

    # Save the best-scoring variants for Kaggle submission
    print(f'\n  Saving submission variants...')
    for idx, name in enumerate(names):
        rid_to_pred = {}
        for i in range(n_labeled):
            if test_mask[i]:
                rid_to_pred[record_ids[i]] = int(variants[idx][i])

        sub_out = sub_df[['RecordID']].copy()
        sub_out['Overall Seed'] = sub_out['RecordID'].map(lambda r: rid_to_pred.get(r, 0))
        out_path = f'submission_variant_{idx}.csv'
        sub_out.to_csv(os.path.join(os.path.dirname(__file__), out_path), index=False)
        print(f'    {out_path}: {name}')

    # Also compute differences between variants
    print(f'\n  Teams that DIFFER between v12 base and v25:')
    print(f'  {"RID":<35} {"GT":>3} {"v12":>4} {"v25":>4} {"v12_SE":>6} {"v25_SE":>6} {"impact":>8}')
    for i in range(n_labeled):
        if test_mask[i] and variants[0][i] != variants[4][i]:
            gt = int(y[i])
            se0 = (variants[0][i] - gt)**2
            se4 = (variants[4][i] - gt)**2
            impact = 'BETTER' if se4 < se0 else ('WORSE' if se4 > se0 else 'SAME')
            print(f'  {record_ids[i]:<35} {gt:3d} {variants[0][i]:4d} {variants[4][i]:4d} {se0:6d} {se4:6d} {impact:>8}')

    print(f'\n  Time: {time.time()-t0:.0f}s')
    print(f'\n  TRY submitting submission_variant_0.csv (v12 base, no zones)')
    print(f'  If it scores ~1.09 on Kaggle, the zone corrections are hurting.')


# Initialize variants storage
variants = {}
for i in range(5):
    variants[i] = np.zeros(500, dtype=int)  # Will be resized

if __name__ == '__main__':
    # Need to pre-allocate after knowing n_labeled
    # Hacky but works - use load_data to get size first
    all_df, labeled, _, _, _, _, _ = load_data()
    n = len(labeled)
    for i in range(5):
        variants[i] = np.zeros(n, dtype=int)
    main()
