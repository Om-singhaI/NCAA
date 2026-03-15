#!/usr/bin/env python3
"""
Overfitting Analysis for v17 Mid-Range Swap Correction
========================================================

Tests:
1. How many configs hit 61+/91 out of 1241? (already know: 294)
2. Leave-One-Season-Out param selection: tune on 4 seasons, test on held-out
3. Permutation test: random corrections → how often beat v12?
4. Per-season stability: does the correction help in EVERY season or just some?
5. Training-team sanity check: apply correction to training teams too
6. Degrees of freedom analysis: correction affects only N teams, with M params
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'code'))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    compute_committee_correction, apply_midrange_swap,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_AL, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER
)
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


def get_v12_predictions(X_all, y, seasons, feature_names, test_mask, folds):
    """Run v12 baseline and return per-team predictions + raw scores per season."""
    n = len(y)
    v12_assigned = np.zeros(n, dtype=int)
    raw_scores_dict = {}  # season -> (season_indices, raw_scores, pass1_assigned)

    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X_all[season_mask]

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

        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)

        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                v12_assigned[global_idx] = pass1[i]

        test_mask_season = np.array([test_mask[gi] for gi in season_indices])
        raw_scores_dict[hold] = (season_indices, raw_scores.copy(), pass1.copy(), test_mask_season)

    return v12_assigned, raw_scores_dict


def apply_correction(raw_scores_dict, feature_names, X_all, test_mask,
                     aq, al, sos, zone=(17, 34), blend=1.0, power=0.15):
    """Apply swap correction with given params, return v17 predictions."""
    n = len(test_mask)
    v17_assigned = np.zeros(n, dtype=int)

    for hold, (season_indices, raw_scores, pass1, test_mask_season) in raw_scores_dict.items():
        correction = compute_committee_correction(
            feature_names, X_all[season_indices[0]:season_indices[-1]+1]
            if len(season_indices) > 0 else X_all[:0],
            alpha_aq=aq, beta_al=al, gamma_sos=sos)

        # Need to rebuild X_season properly
        X_season = X_all[season_indices]
        correction = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=aq, beta_al=al, gamma_sos=sos)

        pass2 = apply_midrange_swap(
            pass1.copy(), raw_scores.copy(), correction,
            test_mask_season, season_indices,
            zone=zone, blend=blend, power=power)

        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                v17_assigned[global_idx] = pass2[i]

    return v17_assigned


def main():
    t0 = time.time()
    print('='*70)
    print(' OVERFITTING ANALYSIS: v17 Mid-Range Swap Correction')
    print('='*70)

    # ── Load data ──
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

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')
    print(f'  Features: {len(feature_names)}')

    # ── Get v12 baseline ──
    print('\n  Running v12 baseline...')
    v12_assigned, raw_scores_dict = get_v12_predictions(
        X_all, y, seasons, feature_names, test_mask, folds)

    gt = y[test_mask].astype(int)
    v12_exact = int((v12_assigned[test_mask] == gt).sum())
    v12_rmse = np.sqrt(np.mean((v12_assigned[test_mask] - gt)**2))
    print(f'  v12: {v12_exact}/91 exact, RMSE={v12_rmse:.4f}')

    # ══════════════════════════════════════════════════════════════════
    # TEST 1: Per-Season Breakdown — How many swaps per season?
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 1: Per-Season Impact Analysis')
    print('='*70)

    v17_assigned = apply_correction(raw_scores_dict, feature_names, X_all, test_mask,
                                     aq=CORRECTION_AQ, al=CORRECTION_AL, sos=CORRECTION_SOS)

    for s in folds:
        s_mask = test_mask & (seasons == s)
        if s_mask.sum() == 0:
            continue
        gt_s = y[s_mask].astype(int)
        v12_s = v12_assigned[s_mask]
        v17_s = v17_assigned[s_mask]
        v12_ex = int((v12_s == gt_s).sum())
        v17_ex = int((v17_s == gt_s).sum())
        n_changed = int((v12_s != v17_s).sum())
        mid_mask_s = (gt_s >= MIDRANGE_ZONE[0]) & (gt_s <= MIDRANGE_ZONE[1])
        n_mid = mid_mask_s.sum()

        print(f'\n  {s}: {s_mask.sum()} test teams, {n_mid} mid-range (GT in {MIDRANGE_ZONE[0]}-{MIDRANGE_ZONE[1]})')
        print(f'    v12: {v12_ex}/{s_mask.sum()}, v17: {v17_ex}/{s_mask.sum()}, changed: {n_changed}')

        # Show each changed team
        s_indices = np.where(s_mask)[0]
        for idx in s_indices:
            if v12_assigned[idx] != v17_assigned[idx]:
                actual = int(y[idx])
                v12_err = abs(v12_assigned[idx] - actual)
                v17_err = abs(v17_assigned[idx] - actual)
                marker = '★' if v17_err < v12_err else '✗' if v17_err > v12_err else '='
                print(f'      {record_ids[idx]:<28} v12={v12_assigned[idx]:2d} v17={v17_assigned[idx]:2d} '
                      f'true={actual:2d}  v12err={v12_err:+d} v17err={v17_err:+d}  {marker}')

    # ══════════════════════════════════════════════════════════════════
    # TEST 2: Leave-One-Season-Out Param Validation
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 2: Leave-One-Season-Out Param Selection')
    print('  (Tune params on 4 seasons, evaluate on held-out season)')
    print('='*70)

    # Only seasons with test teams
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Param grid (smaller for speed)
    aq_vals = [0, 1, 2, 3, 4, 5]
    al_vals = [0, 1, 2, 3, 4, 5]
    sos_vals = [0, 1, 2, 3, 4]

    loso_total_v12 = 0
    loso_total_v17 = 0
    loso_total_teams = 0

    for hold_season in test_seasons:
        # Tune on other seasons
        tune_seasons = [s for s in test_seasons if s != hold_season]

        best_tune_exact = -1
        best_params = (0, 0, 0)

        for aq in aq_vals:
            for al in al_vals:
                for sos in sos_vals:
                    if aq == 0 and al == 0 and sos == 0:
                        continue  # skip no-correction

                    # Evaluate on tune seasons only
                    tune_exact = 0
                    tune_total = 0
                    for ts in tune_seasons:
                        ts_mask = test_mask & (seasons == ts)
                        if ts_mask.sum() == 0:
                            continue
                        if ts not in raw_scores_dict:
                            continue

                        v17_ts = apply_correction(
                            {ts: raw_scores_dict[ts]},
                            feature_names, X_all, test_mask,
                            aq=aq, al=al, sos=sos)

                        gt_ts = y[ts_mask].astype(int)
                        tune_exact += int((v17_ts[ts_mask] == gt_ts).sum())
                        tune_total += ts_mask.sum()

                    if tune_exact > best_tune_exact:
                        best_tune_exact = tune_exact
                        best_params = (aq, al, sos)

        # Now evaluate best params on held-out season
        hold_mask = test_mask & (seasons == hold_season)
        n_hold = hold_mask.sum()

        v12_hold = v12_assigned[hold_mask]
        gt_hold = y[hold_mask].astype(int)
        v12_ex = int((v12_hold == gt_hold).sum())

        v17_hold = apply_correction(
            {hold_season: raw_scores_dict[hold_season]},
            feature_names, X_all, test_mask,
            aq=best_params[0], al=best_params[1], sos=best_params[2])
        v17_ex = int((v17_hold[hold_mask] == gt_hold).sum())

        loso_total_v12 += v12_ex
        loso_total_v17 += v17_ex
        loso_total_teams += n_hold

        print(f'\n  Hold-out {hold_season} ({n_hold} teams):')
        print(f'    Best params from other seasons: aq={best_params[0]}, al={best_params[1]}, sos={best_params[2]}')
        print(f'    v12: {v12_ex}/{n_hold}, v17 (LOSO-tuned): {v17_ex}/{n_hold}, Δ={v17_ex - v12_ex:+d}')

    print(f'\n  LOSO TOTAL: v12={loso_total_v12}/{loso_total_teams}, '
          f'v17(LOSO)={loso_total_v17}/{loso_total_teams}, '
          f'Δ={loso_total_v17-loso_total_v12:+d}')

    # ══════════════════════════════════════════════════════════════════
    # TEST 3: Permutation Test — Random Corrections
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 3: Permutation Test')
    print('  (Random shuffle of mid-range test team seeds — how often ≥61?)')
    print('='*70)

    n_perms = 1000
    perm_results = []
    v12_pred = v12_assigned[test_mask]

    for perm_i in range(n_perms):
        rng = np.random.RandomState(perm_i)
        perm_pred = v12_pred.copy()

        # For each season, randomly shuffle the mid-range test team seeds
        for s in folds:
            s_global = np.where(test_mask & (seasons == s))[0]
            if len(s_global) == 0:
                continue

            # Find which test teams got mid-range seeds from v12
            mid_indices = []
            for gi in s_global:
                if MIDRANGE_ZONE[0] <= v12_assigned[gi] <= MIDRANGE_ZONE[1]:
                    mid_indices.append(gi)

            if len(mid_indices) > 1:
                # Extract their v12 seeds and randomly shuffle
                mid_seeds = [v12_assigned[gi] for gi in mid_indices]
                rng.shuffle(mid_seeds)
                for j, gi in enumerate(mid_indices):
                    # map global index to position in test_mask
                    test_pos = np.sum(test_mask[:gi+1]) - 1
                    perm_pred[test_pos] = mid_seeds[j]

        perm_exact = int((perm_pred == gt).sum())
        perm_results.append(perm_exact)

    perm_results = np.array(perm_results)
    v17_exact = int((v17_assigned[test_mask] == gt).sum())

    print(f'\n  Random shuffle results over {n_perms} permutations:')
    print(f'    Mean: {perm_results.mean():.1f}/91')
    print(f'    Std:  {perm_results.std():.1f}')
    print(f'    Min:  {perm_results.min()}/91')
    print(f'    Max:  {perm_results.max()}/91')
    print(f'    ≥57 (=v12): {(perm_results >= 57).sum()}/{n_perms} ({(perm_results >= 57).mean()*100:.1f}%)')
    print(f'    ≥61 (=v17): {(perm_results >= 61).sum()}/{n_perms} ({(perm_results >= 61).mean()*100:.1f}%)')
    print(f'    ≥63 (=best): {(perm_results >= 63).sum()}/{n_perms} ({(perm_results >= 63).mean()*100:.1f}%)')
    print(f'    p-value (≥61 by chance): {(perm_results >= 61).mean():.4f}')

    # ══════════════════════════════════════════════════════════════════
    # TEST 4: Degrees of Freedom Analysis
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 4: Degrees of Freedom Analysis')
    print('='*70)

    n_affected = int((v12_assigned[test_mask] != v17_assigned[test_mask]).sum())
    n_mid_test = 0
    for s in folds:
        s_mask = test_mask & (seasons == s)
        for gi in np.where(s_mask)[0]:
            if MIDRANGE_ZONE[0] <= v12_assigned[gi] <= MIDRANGE_ZONE[1]:
                n_mid_test += 1

    print(f'\n  Total test teams: 91')
    print(f'  Test teams in mid-range (v12 seed 17-34): {n_mid_test}')
    print(f'  Teams actually changed by correction: {n_affected}')
    print(f'  Free parameters: 3 (aq, al, sos)')
    print(f'  Parameter configs tested: 1241 (v17b)')
    print(f'  Ratio: {n_affected} teams changed / 3 params = {n_affected/3:.1f} teams per param')
    print(f'  Configs tested per affected team: {1241/max(n_affected,1):.0f}')
    print(f'  Overfitting risk: {"HIGH" if n_affected <= 5 else "MODERATE" if n_affected <= 15 else "LOW"}')

    # ══════════════════════════════════════════════════════════════════
    # TEST 5: Correction on TRAINING teams (sanity check)
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' TEST 5: Correction Applied to ALL Teams (Training + Test)')
    print('  (If genuinely captures committee bias, should help training too)')
    print('='*70)

    # Run full LOSO on ALL labeled teams (like our regular validation)
    all_assigned_v12 = np.zeros(n_labeled, dtype=int)
    all_assigned_v17 = np.zeros(n_labeled, dtype=int)

    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        n_season = season_mask.sum()

        train_mask_loso = ~season_mask
        X_season = X_all[season_mask]

        top_k_idx = select_top_k_features(
            X_all[train_mask_loso], y[train_mask_loso],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]

        raw = predict_robust_blend(
            X_all[train_mask_loso], y[train_mask_loso],
            X_season, seasons[train_mask_loso], top_k_idx)

        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=KAGGLE_POWER)

        for i, gi in enumerate(season_indices):
            all_assigned_v12[gi] = pass1[i]

        # v17: apply correction (to ALL teams, not just test)
        correction = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=CORRECTION_AQ, beta_al=CORRECTION_AL,
            gamma_sos=CORRECTION_SOS)

        # Apply swap on ALL season teams (no test_mask restriction)
        all_test_mask = np.ones(n_season, dtype=bool)  # treat all as "test"
        pass2 = apply_midrange_swap(
            pass1.copy(), raw.copy(), correction,
            all_test_mask, season_indices,
            zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND,
            power=CORRECTION_POWER)

        for i, gi in enumerate(season_indices):
            all_assigned_v17[gi] = pass2[i]

    gt_all = y.astype(int)
    all_v12_ex = int((all_assigned_v12 == gt_all).sum())
    all_v17_ex = int((all_assigned_v17 == gt_all).sum())
    all_v12_rmse = np.sqrt(np.mean((all_assigned_v12 - gt_all)**2))
    all_v17_rmse = np.sqrt(np.mean((all_assigned_v17 - gt_all)**2))

    # Mid-range on all teams
    all_mid = (gt_all >= MIDRANGE_ZONE[0]) & (gt_all <= MIDRANGE_ZONE[1])
    all_nm = ~all_mid
    mid_v12_all = int((all_assigned_v12[all_mid] == gt_all[all_mid]).sum())
    mid_v17_all = int((all_assigned_v17[all_mid] == gt_all[all_mid]).sum())
    nm_v12_all = int((all_assigned_v12[all_nm] == gt_all[all_nm]).sum())
    nm_v17_all = int((all_assigned_v17[all_nm] == gt_all[all_nm]).sum())

    print(f'\n  Full LOSO (all {n_labeled} teams):')
    print(f'    v12: {all_v12_ex}/{n_labeled} exact ({all_v12_ex/n_labeled*100:.1f}%), RMSE={all_v12_rmse:.4f}')
    print(f'    v17: {all_v17_ex}/{n_labeled} exact ({all_v17_ex/n_labeled*100:.1f}%), RMSE={all_v17_rmse:.4f}')
    print(f'    Δ: {all_v17_ex - all_v12_ex:+d}')
    print(f'    Mid({MIDRANGE_ZONE[0]}-{MIDRANGE_ZONE[1]}): v12={mid_v12_all}/{all_mid.sum()} '
          f'v17={mid_v17_all}/{all_mid.sum()} Δ={mid_v17_all - mid_v12_all:+d}')
    print(f'    Non-mid: v12={nm_v12_all}/{all_nm.sum()} v17={nm_v17_all}/{all_nm.sum()} '
          f'Δ={nm_v17_all - nm_v12_all:+d}')

    # Training teams only (exclude test)
    train_only = ~test_mask
    t_v12_ex = int((all_assigned_v12[train_only] == gt_all[train_only]).sum())
    t_v17_ex = int((all_assigned_v17[train_only] == gt_all[train_only]).sum())
    t_mid = all_mid & train_only
    tmid_v12 = int((all_assigned_v12[t_mid] == gt_all[t_mid]).sum())
    tmid_v17 = int((all_assigned_v17[t_mid] == gt_all[t_mid]).sum())
    
    print(f'\n  Training teams only ({train_only.sum()}/{n_labeled}):')
    print(f'    v12: {t_v12_ex}/{train_only.sum()} exact')
    print(f'    v17: {t_v17_ex}/{train_only.sum()} exact, Δ={t_v17_ex - t_v12_ex:+d}')
    print(f'    Mid-range training: v12={tmid_v12}/{t_mid.sum()} '
          f'v17={tmid_v17}/{t_mid.sum()} Δ={tmid_v17 - tmid_v12:+d}')

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' OVERFITTING VERDICT')
    print('='*70)
    
    print(f'''
  Evidence FOR overfitting:
    • Correction only changes {n_affected} teams across {len(test_seasons)} seasons
    • 1241 configs tested on same 91 test teams = data snooping
    • Parameters (aq={CORRECTION_AQ}, al={CORRECTION_AL}, sos={CORRECTION_SOS}) selected on test set
    • {4 if n_affected >= 4 else n_affected}/{n_affected} swaps are "perfect" (error→0) — suspiciously good

  Evidence AGAINST overfitting:
    • {(perm_results >= 61).sum()}/{n_perms} random shuffles hit 61+ (p={((perm_results >= 61).mean()):.4f})
    • 294/1241 configs (24%) hit 61+ — very broad optimum
    • Non-mid teams completely untouched (50/73 in both v12 and v17)
    • Formula has domain justification (committee bias is well-documented)
    • LOSO param selection result: {loso_total_v17}/{loso_total_teams} vs v12={loso_total_v12}/{loso_total_teams}
    • Training team correction: Δ={t_v17_ex - t_v12_ex:+d}

  CONCLUSION: {"The LOSO param selection GENERALIZES" if loso_total_v17 > loso_total_v12 else "LOSO shows NO generalization — OVERFITTING LIKELY" if loso_total_v17 <= loso_total_v12 else "Inconclusive"}
    LOSO-validated gain: {loso_total_v17 - loso_total_v12:+d} (vs +{v17_exact - v12_exact} when tuning on test)
    Random chance of 61+: {(perm_results >= 61).mean()*100:.1f}%
    Recommendation: {"SAFE to use" if loso_total_v17 >= loso_total_v12 + 2 else "Use with CAUTION" if loso_total_v17 > loso_total_v12 else "Consider reverting to v12"}
''')

    print(f'  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
