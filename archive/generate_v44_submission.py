#!/usr/bin/env python3
"""
Generate v44 submission with RMSE-optimized params.
Changes from v27:
  - Mid-range: al=2 → al=0
  - Tail zone: (61,65) → (60,61)
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER,
    BOTTOMZONE_ZONE, BOTTOMZONE_SOSNET, BOTTOMZONE_NETCONF,
    BOTTOMZONE_CBHIST, BOTTOMZONE_POWER,
    TAILZONE_OPP_RANK, TAILZONE_POWER,
)
from sklearn.impute import KNNImputer

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15

# v44 overrides
V44_AL = 0          # was 2 in v27
V44_TAIL_ZONE = (60, 61)  # was (61, 65) in v27

def main():
    t0 = time.time()
    print('='*60)
    print(' v44 SUBMISSION GENERATOR')
    print(f' Changes: AL={V44_AL} (was 2), tail=({V44_TAIL_ZONE[0]},{V44_TAIL_ZONE[1]}) (was 61,65)')
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

    print(f'  Mid-range: AQ={CORRECTION_AQ}, AL={V44_AL}, SOS={CORRECTION_SOS}')
    print(f'  Bot-zone: {BOTTOMZONE_ZONE}, SOSNET={BOTTOMZONE_SOSNET}, '
          f'NETCONF={BOTTOMZONE_NETCONF}, CBHIST={BOTTOMZONE_CBHIST}')
    print(f'  Tail-zone: {V44_TAIL_ZONE}, OPP={TAILZONE_OPP_RANK}')

    test_assigned = np.zeros(n_labeled, dtype=int)

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X_all[season_mask]
        season_indices = np.where(season_mask)[0]

        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        raw_scores = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], top_k_idx)

        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                raw_scores[i] = y[global_idx]

        avail = {hold_season: list(range(1, 69))}
        pass1 = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)

        # Mid-range with AL=0
        corr = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=CORRECTION_AQ, beta_al=V44_AL,
            gamma_sos=CORRECTION_SOS)
        tm = np.array([test_mask[gi] for gi in season_indices])
        pass2 = apply_midrange_swap(pass1, raw_scores, corr, tm, season_indices,
                                     zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND,
                                     power=CORRECTION_POWER)

        # Bot-zone (unchanged)
        bot_corr = compute_bottom_correction(
            feature_names, X_season,
            sosnet=BOTTOMZONE_SOSNET, net_conf=BOTTOMZONE_NETCONF,
            cbhist=BOTTOMZONE_CBHIST)
        pass3 = apply_bottomzone_swap(pass2, raw_scores, bot_corr, tm, season_indices,
                                       zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER)

        # Tail-zone with (60,61)
        tail_corr = compute_tail_correction(
            feature_names, X_season, opp_rank=TAILZONE_OPP_RANK)
        pass4 = apply_tailzone_swap(pass3, raw_scores, tail_corr, tm, season_indices,
                                     zone=V44_TAIL_ZONE, power=TAILZONE_POWER)

        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                test_assigned[global_idx] = pass4[i]

        print(f'    {hold_season}: {n_te} test teams')

    # Evaluate
    gt_all = y[test_mask].astype(int)
    pred_all = test_assigned[test_mask]
    exact = int((pred_all == gt_all).sum())
    se = int(np.sum((pred_all - gt_all)**2))
    rmse91 = np.sqrt(np.mean((pred_all - gt_all)**2))

    print(f'\n  Exact: {exact}/91')
    print(f'  SE: {se}')
    print(f'  RMSE(91): {rmse91:.4f}')
    print(f'  RMSE(451): {np.sqrt(se/451):.4f}')

    # Build submission
    submission = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    for i in np.where(test_mask)[0]:
        rid_to_seed[record_ids[i]] = int(test_assigned[i])
    submission['Overall Seed'] = submission['RecordID'].map(lambda r: rid_to_seed.get(r, 0))

    out_path = os.path.join(DATA_DIR, 'submission_v44.csv')
    submission.to_csv(out_path, index=False)
    print(f'\n  Saved: {out_path}')

    # Compare to v27
    v27_path = os.path.join(DATA_DIR, 'submission_kaggle.csv')
    if os.path.exists(v27_path):
        v27 = pd.read_csv(v27_path)
        merged = submission.merge(v27, on='RecordID', suffixes=('_v44', '_v27'))
        diff = merged[merged['Overall Seed_v44'] != merged['Overall Seed_v27']]
        print(f'\n  Changes from v27: {len(diff)} teams')
        for _, row in diff.iterrows():
            rid = row['RecordID']
            v44_s = row['Overall Seed_v44']
            v27_s = row['Overall Seed_v27']
            gt = GT.get(rid, '?')
            print(f'    {rid:<30} v27={v27_s:3d} → v44={v44_s:3d} (GT={gt})')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
