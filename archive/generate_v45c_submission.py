#!/usr/bin/env python3
"""
Generate v45c submission — the combined best config.
Pipeline: mid(17-34) + uncorr(34-44) + midbot(48-52) + bot(52-60) + tail(60-63)
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

# v45c config
MID_ZONE = (17, 34)
MID_PARAMS = (0, 0, 3)         # aq, al, sos

UNCORR_ZONE = (34, 44)
UNCORR_PARAMS = (-2, -3, -4)   # aq, al, sos

MIDBOT_ZONE = (48, 52)
MIDBOT_PARAMS = (0, 2, -2)     # sosnet, net_conf, cbhist

BOT_ZONE = (52, 60)
BOT_PARAMS = (-4, 3, -1)       # sosnet, net_conf, cbhist

TAIL_ZONE = (60, 63)
TAIL_PARAMS = (1,)              # opp_rank


def main():
    t0 = time.time()
    print('='*60)
    print(' v45c SUBMISSION GENERATOR')
    print(' 6-zone pipeline: mid + uncorr + midbot + bot + tail')
    print('='*60)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n = len(labeled)
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

    print(f'\n  Zones:')
    print(f'  1. Mid {MID_ZONE}: aq={MID_PARAMS[0]}, al={MID_PARAMS[1]}, sos={MID_PARAMS[2]}')
    print(f'  2. Uncorr {UNCORR_ZONE}: aq={UNCORR_PARAMS[0]}, al={UNCORR_PARAMS[1]}, sos={UNCORR_PARAMS[2]}')
    print(f'  3. Midbot {MIDBOT_ZONE}: sn={MIDBOT_PARAMS[0]}, nc={MIDBOT_PARAMS[1]}, cb={MIDBOT_PARAMS[2]}')
    print(f'  4. Bot {BOT_ZONE}: sn={BOT_PARAMS[0]}, nc={BOT_PARAMS[1]}, cb={BOT_PARAMS[2]}')
    print(f'  5. Tail {TAIL_ZONE}: opp={TAIL_PARAMS[0]}')

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

        tm = np.array([test_mask[gi] for gi in season_indices])

        # 1. Mid-range
        aq, al, sos = MID_PARAMS
        corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, season_indices,
                                        zone=MID_ZONE, blend=1.0, power=0.15)

        # 2. Uncorr zone
        u_aq, u_al, u_sos = UNCORR_PARAMS
        corr = compute_committee_correction(fn, X_season, alpha_aq=u_aq, beta_al=u_al, gamma_sos=u_sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, season_indices,
                                        zone=UNCORR_ZONE, blend=1.0, power=0.15)

        # 3. Midbot zone
        sn, nc, cb = MIDBOT_PARAMS
        corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, season_indices,
                                          zone=MIDBOT_ZONE, power=0.15)

        # 4. Bot zone
        sn, nc, cb = BOT_PARAMS
        corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, season_indices,
                                          zone=BOT_ZONE, power=0.15)

        # 5. Tail zone
        opp = TAIL_PARAMS[0]
        corr = compute_tail_correction(fn, X_season, opp_rank=opp)
        assigned = apply_tailzone_swap(assigned, raw, corr, tm, season_indices,
                                        zone=TAIL_ZONE, power=0.15)

        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                preds[gi] = assigned[i]

        print(f'    {hold_season}: {season_test_mask.sum()} test teams')

    # Evaluate
    gt_all = y[test_mask].astype(int)
    pred_all = preds[test_mask]
    exact = int((pred_all == gt_all).sum())
    se = int(np.sum((pred_all - gt_all)**2))
    print(f'\n  Exact: {exact}/91, SE={se}, RMSE451={np.sqrt(se/451):.4f}')

    # Per-season
    print(f'\n  {"Season":<12} {"Test":>4} {"Exact":>5} {"SE":>5}')
    for s in folds:
        sm = test_mask & (seasons == s)
        if sm.sum() == 0: continue
        gt_s = y[sm].astype(int)
        pr_s = preds[sm]
        ex = int((pr_s == gt_s).sum())
        se_s = int(np.sum((pr_s - gt_s)**2))
        print(f'  {s:<12} {sm.sum():4d} {ex:5d} {se_s:5d}')

    # Build submission
    submission = sub_df[['RecordID']].copy()
    rid_to_seed = {record_ids[i]: int(preds[i]) for i in np.where(test_mask)[0]}
    submission['Overall Seed'] = submission['RecordID'].map(lambda r: rid_to_seed.get(r, 0))

    # Save as v45c
    out_path = os.path.join(DATA_DIR, 'submission_v45c.csv')
    submission.to_csv(out_path, index=False)
    print(f'\n  Saved: {out_path}')

    # Compare to v44 and v27
    for ref_name, ref_file in [('v44', 'submission_v44.csv'), ('v27', 'submission_kaggle.csv')]:
        ref_path = os.path.join(DATA_DIR, ref_file)
        if os.path.exists(ref_path):
            ref = pd.read_csv(ref_path)
            merged = submission.merge(ref, on='RecordID', suffixes=('_v45c', f'_{ref_name}'))
            diff = merged[merged[f'Overall Seed_v45c'] != merged[f'Overall Seed_{ref_name}']]
            print(f'\n  Changes vs {ref_name}: {len(diff)} teams')
            for _, row in diff.iterrows():
                rid = row['RecordID']
                new_s = row['Overall Seed_v45c']
                old_s = row[f'Overall Seed_{ref_name}']
                gt = GT.get(rid, '?')
                if isinstance(gt, (int, float)):
                    delta_se = (new_s - gt)**2 - (old_s - gt)**2
                    print(f'    {rid:<30} {ref_name}={old_s:3d} → v45c={new_s:3d} GT={gt:3d} ΔSE={delta_se:+4d}')
                else:
                    print(f'    {rid:<30} {ref_name}={old_s:3d} → v45c={new_s:3d}')

    # Remaining errors
    print(f'\n  Remaining errors (sorted by SE):')
    errors = []
    for i in np.where(test_mask)[0]:
        gt = int(y[i])
        pred = int(preds[i])
        err = pred - gt
        se = err**2
        if se > 0:
            errors.append((record_ids[i], gt, pred, err, se))
    errors.sort(key=lambda x: -x[4])
    for rid, gt, pred, err, se_i in errors:
        print(f'    {rid:<30} GT={gt:3d} Pred={pred:3d} Err={err:+3d} SE={se_i:3d}')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
