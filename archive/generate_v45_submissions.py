#!/usr/bin/env python3
"""
Generate v45 submissions:
  v45a: mid(0,0,3) + uncorr(34,44)(-2,-3,-3) + bot + tail  → SE=307
  v45b: mid(0,0,3) + uncorr(34,44)(-2,-3,-4) + bot + tail  → SE=263
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    'v45a': {'mid': (0, 0, 3), 'uncorr_zone': (34, 44), 'uncorr': (-2, -3, -3),
             'desc': 'safe (sos=-3)'},
    'v45b': {'mid': (0, 0, 3), 'uncorr_zone': (34, 44), 'uncorr': (-2, -3, -4),
             'desc': 'aggressive (sos=-4)'},
}
BOT_ZONE = (50, 60)
BOT_PARAMS = (-4, 3, -1)
TAIL_ZONE = (60, 61)
TAIL_PARAMS = (-3,)


def main():
    t0 = time.time()
    print('='*60)
    print(' v45 SUBMISSION GENERATOR')
    print('='*60)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    feat = build_features(labeled, context_df, labeled, tourn_rids)
    fn = list(feat.columns)
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
    n = len(labeled)

    for vname, vcfg in VARIANTS.items():
        print(f'\n  ── {vname} ({vcfg["desc"]}) ──')
        print(f'  Mid: {vcfg["mid"]}, Uncorr zone: {vcfg["uncorr_zone"]}, '
              f'Uncorr params: {vcfg["uncorr"]}')

        preds = np.zeros(n, dtype=int)

        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0:
                continue

            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask

            tki = select_top_k_features(
                X_all[global_train_mask], y[global_train_mask],
                fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
            raw = predict_robust_blend(
                X_all[global_train_mask], y[global_train_mask],
                X_season, seasons[global_train_mask], tki)

            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw[i] = y[gi]

            avail = {hold_season: list(range(1, 69))}
            assigned = hungarian(raw, seasons[season_mask], avail, power=0.15)

            # Mid-range
            aq, al, sos = vcfg['mid']
            tm = np.array([test_mask[gi] for gi in season_indices])
            corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, season_indices,
                                            zone=(17, 34), blend=1.0, power=0.15)

            # Uncorr zone
            u_aq, u_al, u_sos = vcfg['uncorr']
            corr = compute_committee_correction(fn, X_season, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, season_indices,
                                            zone=vcfg['uncorr_zone'], blend=1.0, power=0.15)

            # Bot-zone
            sn, nc, cb = BOT_PARAMS
            corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, season_indices,
                                              zone=BOT_ZONE, power=0.15)

            # Tail-zone
            opp, = TAIL_PARAMS
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, season_indices,
                                            zone=TAIL_ZONE, power=0.15)

            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = assigned[i]

        # Evaluate
        gt_all = y[test_mask].astype(int)
        pred_all = preds[test_mask]
        exact = int((pred_all == gt_all).sum())
        se = int(np.sum((pred_all - gt_all)**2))
        print(f'  Exact: {exact}/91, SE={se}, RMSE451={np.sqrt(se/451):.4f}')

        # Build submission
        submission = sub_df[['RecordID']].copy()
        rid_to_seed = {record_ids[i]: int(preds[i]) for i in np.where(test_mask)[0]}
        submission['Overall Seed'] = submission['RecordID'].map(lambda r: rid_to_seed.get(r, 0))

        out_path = os.path.join(DATA_DIR, f'submission_{vname}.csv')
        submission.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')

        # Compare to v44
        v44_path = os.path.join(DATA_DIR, 'submission_v44.csv')
        if os.path.exists(v44_path):
            v44 = pd.read_csv(v44_path)
            merged = submission.merge(v44, on='RecordID', suffixes=(f'_{vname}', '_v44'))
            diff = merged[merged[f'Overall Seed_{vname}'] != merged['Overall Seed_v44']]
            print(f'  Changes from v44: {len(diff)} teams')
            for _, row in diff.iterrows():
                rid = row['RecordID']
                new_s = row[f'Overall Seed_{vname}']
                old_s = row['Overall Seed_v44']
                gt = GT.get(rid, '?')
                old_se = (old_s - gt)**2 if isinstance(gt, (int, float)) else '?'
                new_se = (new_s - gt)**2 if isinstance(gt, (int, float)) else '?'
                print(f'    {rid:<30} v44={old_s:3d} → {vname}={new_s:3d} '
                      f'(GT={gt}, SE: {old_se}→{new_se})')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
