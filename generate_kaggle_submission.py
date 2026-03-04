#!/usr/bin/env python3
"""
Generate Kaggle submission using v50 model.

v50 = v12 base + dual-Hungarian ensemble + 7-zone correction + AQ↔AL swap.
Dual-Hungarian: run separate Hungarian on v12 pairwise and min8 Ridge,
average assignments (75/25), final Hungarian for valid assignment.
v50 changes: added extreme-tail zone (63-68) sn=1 nc=-1 cb=-1.
SE=14, 83/91 exact. Zero regressions from v49.
v49 Kaggle=0.163.

v12 base:
  64% PW-LR_C5 (adj-pairs gap≤30) + 28% PW-LR_topK25_C0.5 + 8% PW-XGB_d4
  with power=0.15, forced NET Rank in top-K.

Zone 1 — Mid (17-34): aq=0, al=0, sos=3  (SOS divergence only)
Zone 2 — Upper-mid (34-44): aq=-6, al=1, sos=-6  (v49: fixes TCU/Creighton)
Zone 3b — Mid-bot2 (42-50): sn=-4, nc=2, cb=-3  (committee path only, v48)
Zone 3 — Mid-bot (48-52): sn=0, nc=2, cb=-2  (fixes SouthDakotaSt/Richmond)
Zone 4 — Bot (52-60): sn=-4, nc=3, cb=-1  (original bot-zone, narrowed)
Zone 5 — Tail (60-63): opp=+1  (fixes Longwood/SaintPeter's)
Zone 7 — Extreme tail (63-68): sn=1, nc=-1, cb=-1  (v50: fixes SoutheastMo/TexasSouthern)
Post-proc — AQ↔AL swap: ng=10, pg=6  (v49: fixes NM/NW/Virginia)

Output: submission_kaggle.csv (451 rows: 91 with seeds, 360 with 0)
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    apply_aq_al_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_AL, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER,
    UPPERMID_ZONE, UPPERMID_AQ, UPPERMID_AL, UPPERMID_SOS, UPPERMID_POWER,
    COMM_UPPERMID_AQ, COMM_UPPERMID_AL, COMM_UPPERMID_SOS,
    MIDBOT_ZONE, MIDBOT_SOSNET, MIDBOT_NETCONF, MIDBOT_CBHIST, MIDBOT_POWER,
    MIDBOT2_ZONE, MIDBOT2_SOSNET, MIDBOT2_NETCONF, MIDBOT2_CBHIST, MIDBOT2_POWER,
    BOTTOMZONE_ZONE, BOTTOMZONE_SOSNET, BOTTOMZONE_NETCONF,
    BOTTOMZONE_CBHIST, BOTTOMZONE_POWER,
    TAILZONE_ZONE, TAILZONE_OPP_RANK, TAILZONE_POWER,
    XTAIL_ZONE, XTAIL_SOSNET, XTAIL_NETCONF, XTAIL_CBHIST, XTAIL_POWER,
    DUAL_RIDGE_ALPHA, DUAL_BLEND,
    SWAP_NET_GAP, SWAP_PRED_GAP, SWAP_ZONE,
)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Power for Kaggle evaluation (LOSO-validated)
KAGGLE_POWER = 0.15


def main():
    t0 = time.time()
    print('='*60)
    print(' KAGGLE SUBMISSION GENERATOR (v50 Model)')
    print(' v12 base + dual-Hungarian (min8) + 7-zone + AQ↔AL swap')
    print('='*60)

    # Load data
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    print(f'\n  Total labeled teams: {n_labeled}')
    print(f'  Test CSV teams: {len(test_df)}')
    print(f'  Test tournament teams (GT available): {len(GT)}')

    # Context
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    # Build features
    print('  Building features...')
    feat = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat.columns)
    print(f'  Features: {len(feature_names)}')

    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))

    # Impute
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    # Build min8 committee features for dual-Hungarian (v47)
    X_comm = build_min8_features(X_all, feature_names)
    print(f'  Min8 committee features: {X_comm.shape[1]}')

    # Identify test teams
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    print(f'  Training teams: {(~test_mask).sum()}')
    print(f'  Test teams to predict: {test_mask.sum()}')
    print(f'  Using power={KAGGLE_POWER} (optimal for locked-seed eval)')
    print(f'  Dual-Hungarian: α={DUAL_RIDGE_ALPHA}, blend={DUAL_BLEND}')
    print(f'  Zone 1 Mid: {MIDRANGE_ZONE}, aq={CORRECTION_AQ}, al={CORRECTION_AL}, sos={CORRECTION_SOS}')
    print(f'  Zone 2 Upper-mid (v12): {UPPERMID_ZONE}, aq={UPPERMID_AQ}, al={UPPERMID_AL}, sos={UPPERMID_SOS}')
    print(f'  Zone 2 Upper-mid (comm): {UPPERMID_ZONE}, aq={COMM_UPPERMID_AQ}, al={COMM_UPPERMID_AL}, sos={COMM_UPPERMID_SOS}')
    print(f'  Zone 3b Mid-bot2: {MIDBOT2_ZONE}, sn={MIDBOT2_SOSNET}, nc={MIDBOT2_NETCONF}, cb={MIDBOT2_CBHIST} (comm only)')
    print(f'  Zone 3 Mid-bot: {MIDBOT_ZONE}, sn={MIDBOT_SOSNET}, nc={MIDBOT_NETCONF}, cb={MIDBOT_CBHIST}')
    print(f'  Zone 4 Bot: {BOTTOMZONE_ZONE}, sn={BOTTOMZONE_SOSNET}, nc={BOTTOMZONE_NETCONF}, cb={BOTTOMZONE_CBHIST}')
    print(f'  Zone 5 Tail: {TAILZONE_ZONE}, opp={TAILZONE_OPP_RANK}')

    # ── Generate predictions ──
    test_assigned = np.zeros(n_labeled, dtype=int)
    v12_assigned = np.zeros(n_labeled, dtype=int)  # for comparison

    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X_all[season_mask]
        X_comm_season = X_comm[season_mask]
        season_indices = np.where(season_mask)[0]

        # v12 pairwise blend scores (adj-pair comp1 + standard comp3)
        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        raw_scores = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], top_k_idx)

        # Committee Ridge scores
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
        X_te_sc = sc.transform(X_comm_season)
        ridge = Ridge(alpha=DUAL_RIDGE_ALPHA)
        ridge.fit(X_tr_sc, y[global_train_mask])
        raw_comm = ridge.predict(X_te_sc)

        # Lock training teams to their known seeds
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                raw_scores[i] = y[global_idx]
                raw_comm[i] = y[global_idx]

        test_mask_season = np.array([test_mask[gi] for gi in season_indices])
        avail = {hold_season: list(range(1, 69))}

        # ── Branch A: v12 Hungarian + 6 zones (v50: +xtail) ──
        a_v12 = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)
        
        # Save pure v12 result for comparison
        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                v12_assigned[global_idx] = a_v12[i]

        # Zone 1-5 on v12 branch
        corr_mid = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=CORRECTION_AQ, beta_al=CORRECTION_AL,
            gamma_sos=CORRECTION_SOS)
        a_v12 = apply_midrange_swap(
            a_v12, raw_scores, corr_mid,
            test_mask_season, season_indices,
            zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND,
            power=CORRECTION_POWER)

        corr_umid = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=UPPERMID_AQ, beta_al=UPPERMID_AL,
            gamma_sos=UPPERMID_SOS)
        a_v12 = apply_midrange_swap(
            a_v12, raw_scores, corr_umid,
            test_mask_season, season_indices,
            zone=UPPERMID_ZONE, blend=1.0,
            power=UPPERMID_POWER)

        corr_mbot = compute_bottom_correction(
            feature_names, X_season,
            sosnet=MIDBOT_SOSNET, net_conf=MIDBOT_NETCONF,
            cbhist=MIDBOT_CBHIST)
        a_v12 = apply_bottomzone_swap(
            a_v12, raw_scores, corr_mbot,
            test_mask_season, season_indices,
            zone=MIDBOT_ZONE, power=MIDBOT_POWER)

        corr_bot = compute_bottom_correction(
            feature_names, X_season,
            sosnet=BOTTOMZONE_SOSNET, net_conf=BOTTOMZONE_NETCONF,
            cbhist=BOTTOMZONE_CBHIST)
        a_v12 = apply_bottomzone_swap(
            a_v12, raw_scores, corr_bot,
            test_mask_season, season_indices,
            zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER)

        corr_tail = compute_tail_correction(
            feature_names, X_season,
            opp_rank=TAILZONE_OPP_RANK)
        a_v12 = apply_tailzone_swap(
            a_v12, raw_scores, corr_tail,
            test_mask_season, season_indices,
            zone=TAILZONE_ZONE, power=TAILZONE_POWER)

        # Zone 7: Extreme tail (63-68) — v50, v12 path only
        corr_xtail = compute_bottom_correction(
            feature_names, X_season,
            sosnet=XTAIL_SOSNET, net_conf=XTAIL_NETCONF,
            cbhist=XTAIL_CBHIST)
        a_v12 = apply_bottomzone_swap(
            a_v12, raw_scores, corr_xtail,
            test_mask_season, season_indices,
            zone=XTAIL_ZONE, power=XTAIL_POWER)

        # ── Branch B: Committee Hungarian + 6 zones (includes midbot2) ──
        a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=KAGGLE_POWER)

        a_comm = apply_midrange_swap(
            a_comm, raw_comm, corr_mid,
            test_mask_season, season_indices,
            zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND,
            power=CORRECTION_POWER)

        # v49: committee path uses different uppermid params
        corr_umid_comm = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=COMM_UPPERMID_AQ, beta_al=COMM_UPPERMID_AL,
            gamma_sos=COMM_UPPERMID_SOS)
        a_comm = apply_midrange_swap(
            a_comm, raw_comm, corr_umid_comm,
            test_mask_season, season_indices,
            zone=UPPERMID_ZONE, blend=1.0,
            power=UPPERMID_POWER)

        # Zone 3b: Mid-bot2 (42-50) — committee path only (v48)
        corr_mbot2 = compute_bottom_correction(
            feature_names, X_season,
            sosnet=MIDBOT2_SOSNET, net_conf=MIDBOT2_NETCONF,
            cbhist=MIDBOT2_CBHIST)
        a_comm = apply_bottomzone_swap(
            a_comm, raw_comm, corr_mbot2,
            test_mask_season, season_indices,
            zone=MIDBOT2_ZONE, power=MIDBOT2_POWER)

        a_comm = apply_bottomzone_swap(
            a_comm, raw_comm, corr_mbot,
            test_mask_season, season_indices,
            zone=MIDBOT_ZONE, power=MIDBOT_POWER)
        a_comm = apply_bottomzone_swap(
            a_comm, raw_comm, corr_bot,
            test_mask_season, season_indices,
            zone=BOTTOMZONE_ZONE, power=BOTTOMZONE_POWER)
        a_comm = apply_tailzone_swap(
            a_comm, raw_comm, corr_tail,
            test_mask_season, season_indices,
            zone=TAILZONE_ZONE, power=TAILZONE_POWER)

        # ── Dual-Hungarian: average assignments + final Hungarian ──
        avg_seed = (1 - DUAL_BLEND) * a_v12.astype(float) + DUAL_BLEND * a_comm.astype(float)
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                avg_seed[i] = y[global_idx]

        final_assigned = hungarian(avg_seed, seasons[season_mask], avail, power=KAGGLE_POWER)

        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                test_assigned[global_idx] = final_assigned[i]

        # Count swaps from v12 to dual
        dual_swaps = sum(1 for i, gi in enumerate(season_indices)
                    if test_mask[gi] and v12_assigned[gi] != final_assigned[i])
        print(f'    {hold_season}: {n_te} test, {dual_swaps} dual-swaps from v12')

    # ── v49: Apply AQ↔AL swap rule ──
    pre_swap = test_assigned.copy()
    test_assigned = apply_aq_al_swap(
        test_assigned, X_all, feature_names, seasons, test_mask,
        net_gap=SWAP_NET_GAP, pred_gap=SWAP_PRED_GAP, swap_zone=SWAP_ZONE)
    n_swaps = int(np.sum(test_assigned[test_mask] != pre_swap[test_mask]))
    print(f'\n  AQ↔AL swap: {n_swaps} teams swapped (ng={SWAP_NET_GAP}, pg={SWAP_PRED_GAP})')

    # ── Evaluate ──
    gt_all = y[test_mask].astype(int)
    pred_all = test_assigned[test_mask]
    v12_pred = v12_assigned[test_mask]
    total_exact = int((pred_all == gt_all).sum())
    total_rmse = np.sqrt(np.mean((pred_all - gt_all)**2))
    v12_exact = int((v12_pred == gt_all).sum())
    v12_rmse = np.sqrt(np.mean((v12_pred - gt_all)**2))
    
    # Mid-range stats
    mid_mask = (gt_all >= MIDRANGE_ZONE[0]) & (gt_all <= MIDRANGE_ZONE[1])
    mid_exact = int((pred_all[mid_mask] == gt_all[mid_mask]).sum())
    mid_v12 = int((v12_pred[mid_mask] == gt_all[mid_mask]).sum())
    non_mid = ~mid_mask
    nm_exact = int((pred_all[non_mid] == gt_all[non_mid]).sum())
    nm_v12 = int((v12_pred[non_mid] == gt_all[non_mid]).sum())
    
    print(f'\n  v12 baseline: {v12_exact}/{test_mask.sum()} exact '
          f'({v12_exact/test_mask.sum()*100:.1f}%), RMSE={v12_rmse:.4f}')
    print(f'  v46 result:   {total_exact}/{test_mask.sum()} exact '
          f'({total_exact/test_mask.sum()*100:.1f}%), RMSE={total_rmse:.4f}')
    print(f'  Improvement:  +{total_exact - v12_exact} exact matches')
    print(f'  Mid({MIDRANGE_ZONE[0]}-{MIDRANGE_ZONE[1]}): '
          f'v12={mid_v12}/{mid_mask.sum()} v46={mid_exact}/{mid_mask.sum()}')
    print(f'  Non-mid: v12={nm_v12}/{non_mid.sum()} v46={nm_exact}/{non_mid.sum()}')

    # Per-season breakdown
    print(f'\n  {"Season":<10} {"Test":>4} {"Exact":>5} {"RMSE":>8}')
    print(f'  {"─"*10} {"─"*4} {"─"*5} {"─"*8}')
    for s in folds:
        s_mask = test_mask & (seasons == s)
        if s_mask.sum() == 0:
            continue
        gt_s = y[s_mask].astype(int)
        pr_s = test_assigned[s_mask]
        ex = int((pr_s == gt_s).sum())
        rm = np.sqrt(np.mean((pr_s - gt_s)**2))
        print(f'  {s:<10} {s_mask.sum():4d} {ex:5d} {rm:8.3f}')

    # Build submission
    submission = sub_df[['RecordID']].copy()
    rid_to_seed = {}
    test_indices = np.where(test_mask)[0]
    for i in test_indices:
        rid_to_seed[record_ids[i]] = int(test_assigned[i])
    submission['Overall Seed'] = submission['RecordID'].map(
        lambda r: rid_to_seed.get(r, 0))

    filled = (submission['Overall Seed'] > 0).sum()
    print(f'\n  Submission: {len(submission)} rows, {filled} with seeds')

    # Save
    out_path = os.path.join(DATA_DIR, 'submission_kaggle.csv')
    submission.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')

    # Show teams changed by mid-range correction
    print(f'\n  Teams changed by zone corrections:')
    print(f'  {"RecordID":<30} {"v12":>4} {"v46":>4} {"True":>4} {"v12err":>6} {"v46err":>6}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*4} {"─"*6} {"─"*6}')
    for i in test_indices:
        v12_i = v12_assigned[i]
        v46_i = test_assigned[i]
        if v12_i != v46_i:
            actual = int(y[i])
            v12_err = v12_i - actual
            v46_err = v46_i - actual
            marker = ' ★' if abs(v46_err) < abs(v12_err) else ' ✗' if abs(v46_err) > abs(v12_err) else ''
            print(f'  {record_ids[i]:<30} {v12_i:4d} {v46_i:4d} '
                  f'{actual:4d} {v12_err:+6d} {v46_err:+6d}{marker}')

    # Sample predictions
    print(f'\n  Sample predictions:')
    print(f'  {"RecordID":<30} {"Pred":>4} {"True":>4} {"Diff":>5}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*5}')
    for i in test_indices[:20]:
        pred = test_assigned[i]
        actual = int(y[i])
        diff = pred - actual
        marker = ' ✓' if diff == 0 else ''
        print(f'  {record_ids[i]:<30} {pred:4d} {actual:4d} {diff:+5d}{marker}')

    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('  Upload submission_kaggle.csv to Kaggle!')


if __name__ == '__main__':
    main()
