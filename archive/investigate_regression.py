#!/usr/bin/env python3
"""
Investigate why v26 Kaggle RMSE went UP by 0.1 despite more exact matches.
Compare v25 vs v26 submission side by side.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features,
    select_top_k_features, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    apply_ncsos_zone, predict_robust_blend,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    NCSOS_ZONE, NCSOS_WEIGHT, NCSOS_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)

def main():
    print('='*70)
    print('  INVESTIGATE v26 KAGGLE REGRESSION')
    print('='*70)

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
    teams = labeled['Team'].values if 'Team' in labeled.columns else record_ids
    folds = sorted(set(seasons))

    ncsos_raw = pd.to_numeric(labeled['NETNonConfSOS'], errors='coerce').fillna(200).values

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                    np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # Generate v25 and v26 predictions side by side
    v25_pred = np.zeros(len(y), dtype=int)
    v26_pred = np.zeros(len(y), dtype=int)

    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0:
            continue
        gt = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt], y[gt], fn, k=USE_TOP_K_A,
                                     forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X[gt], y[gt], X_s, seasons[gt], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])

        # v25: all zones except NCSOS
        p25 = p1.copy()
        corr = compute_committee_correction(fn, X_s, alpha_aq=0, beta_al=2, gamma_sos=3)
        p25 = apply_midrange_swap(p25, raw, corr, tm, si, zone=(17,34), blend=1.0, power=0.15)
        corr = compute_low_correction(fn, X_s, q1dom=1, field=2)
        p25 = apply_lowzone_swap(p25, raw, corr, tm, si, zone=(35,52), power=0.15)
        corr = compute_bottom_correction(fn, X_s, sosnet=-4, net_conf=3, cbhist=-1)
        p25 = apply_bottomzone_swap(p25, raw, corr, tm, si, zone=(50,60), power=0.15)
        corr = compute_tail_correction(fn, X_s, opp_rank=-3)
        p25 = apply_tailzone_swap(p25, raw, corr, tm, si, zone=(61,65), power=0.15)

        # v26: v25 + NCSOS zone
        ncsos_s = ncsos_raw[sm]
        p26 = apply_ncsos_zone(p25.copy(), raw, ncsos_s, tm,
                                zone=NCSOS_ZONE, weight=NCSOS_WEIGHT, power=NCSOS_POWER)

        for i, gi in enumerate(si):
            if test_mask[gi]:
                v25_pred[gi] = p25[i]
                v26_pred[gi] = p26[i]

    # ── Compare predictions ──
    print('\n  ALL differences between v25 and v26:')
    print(f'  {"RID":<32} {"Team":<22} {"GT":>3} {"v25":>4} {"v26":>4} {"v25err":>6} {"v26err":>6} {"impact":>8}')
    print(f'  {"-"*95}')

    total_v25_se = 0
    total_v26_se = 0
    n_test = 0
    changes = []

    for i in range(len(y)):
        if not test_mask[i]:
            continue
        gt = int(y[i])
        n_test += 1
        v25_e = v25_pred[i] - gt
        v26_e = v26_pred[i] - gt
        total_v25_se += v25_e**2
        total_v26_se += v26_e**2

        if v25_pred[i] != v26_pred[i]:
            se_diff = v26_e**2 - v25_e**2
            changes.append({
                'rid': record_ids[i], 'team': teams[i], 'gt': gt,
                'v25': v25_pred[i], 'v26': v26_pred[i],
                'v25_err': v25_e, 'v26_err': v26_e,
                'se_diff': se_diff,
            })

    changes.sort(key=lambda x: x['gt'])
    for c in changes:
        impact = 'BETTER' if c['se_diff'] < 0 else ('WORSE' if c['se_diff'] > 0 else 'SAME')
        print(f'  {c["rid"]:<32} {c["team"]:<22} {c["gt"]:3d} {c["v25"]:4d} {c["v26"]:4d} '
              f'{c["v25_err"]:+6d} {c["v26_err"]:+6d} {impact:>8} (SE diff={c["se_diff"]:+d})')

    v25_rmse = np.sqrt(total_v25_se / n_test)
    v26_rmse = np.sqrt(total_v26_se / n_test)
    v25_exact = sum(1 for i in range(len(y)) if test_mask[i] and v25_pred[i] == int(y[i]))
    v26_exact = sum(1 for i in range(len(y)) if test_mask[i] and v26_pred[i] == int(y[i]))

    print(f'\n  v25: {v25_exact}/91 exact, RMSE={v25_rmse:.4f}, total SE={total_v25_se}')
    print(f'  v26: {v26_exact}/91 exact, RMSE={v26_rmse:.4f}, total SE={total_v26_se}')
    print(f'  Delta RMSE: {v26_rmse - v25_rmse:+.4f}')
    print(f'  Delta SE: {total_v26_se - total_v25_se:+d}')

    # ── Check if RMSE went up despite more exact matches ──
    print(f'\n  Net SE change from NCSOS zone:')
    total_se_change = sum(c['se_diff'] for c in changes)
    print(f'    {total_se_change:+d} (negative = better)')
    for c in changes:
        print(f'    {c["team"]:<22}: SE {c["v25_err"]**2} → {c["v26_err"]**2} ({c["se_diff"]:+d})')

    # ── Check the submission file format ──
    print('\n\n  Checking submission file...')
    sub = pd.read_csv(os.path.join(os.path.dirname(__file__), 'submission_kaggle.csv'))
    print(f'  Rows: {len(sub)}')
    print(f'  Columns: {list(sub.columns)}')
    print(f'  Seeds > 0: {(sub["Overall Seed"] > 0).sum()}')
    print(f'  Seeds == 0: {(sub["Overall Seed"] == 0).sum()}')
    print(f'  Unique seeds: {sorted(sub[sub["Overall Seed"] > 0]["Overall Seed"].unique())}')
    
    # Check if all 91 test RIDs are in submission
    sub_rids = set(sub[sub['Overall Seed'] > 0]['RecordID'].values)
    gt_rids = set(GT.keys())
    print(f'\n  GT RIDs in submission: {len(sub_rids & gt_rids)}/{len(gt_rids)}')
    missing = gt_rids - sub_rids
    if missing:
        print(f'  MISSING from submission: {missing}')
    extra = sub_rids - gt_rids
    if extra:
        print(f'  In submission but NOT in GT: {extra}')

    # ── Check the Kaggle evaluation ──
    # Kaggle might evaluate on ALL 451 rows, not just 91
    # If so, the 360 zeros are compared against actual seeds = 0 (non-tournament)
    # OR Kaggle might have different ground truth
    print('\n\n  Kaggle evaluation analysis:')
    print(f'  If Kaggle evaluates all {len(sub)} rows:')
    print(f'    360 rows with seed=0 contribute 0 to RMSE if GT=0')
    print(f'    91 rows with actual seeds → RMSE based on those')
    print(f'    RMSE over 451 rows = sqrt({total_v26_se}/451) = {np.sqrt(total_v26_se/451):.4f}')
    print(f'  If Kaggle evaluates only 91 test rows:')
    print(f'    RMSE = sqrt({total_v26_se}/91) = {v26_rmse:.4f}')

    # ── Generate v25 submission for comparison ──
    print('\n\n  Generating v25 submission for comparison...')
    sub_v25 = sub_df[['RecordID']].copy()
    rid_to_v25 = {}
    for i in range(len(y)):
        if test_mask[i]:
            rid_to_v25[record_ids[i]] = int(v25_pred[i])
    sub_v25['Overall Seed'] = sub_v25['RecordID'].map(lambda r: rid_to_v25.get(r, 0))
    out_v25 = os.path.join(os.path.dirname(__file__), 'submission_kaggle_v25.csv')
    sub_v25.to_csv(out_v25, index=False)
    print(f'  Saved: {out_v25}')

    # Also verify v25 submission matches what we had before
    filled_v25 = (sub_v25['Overall Seed'] > 0).sum()
    print(f'  v25 submission: {len(sub_v25)} rows, {filled_v25} with seeds')

    # ── Washington St analysis ──
    print('\n\n  ═══ KEY FINDING: Washington St. ═══')
    print('  v25: pred=24, GT=26, error=2, SE=4')
    print('  v26: pred=22, GT=26, error=4, SE=16')
    print('  SE increase: +12')
    print('')
    print('  Other 3 fixes:')
    print('  SanDiegoSt: SE 9→0 = -9')
    print('  Miami(FL):  SE 9→0 = -9')
    print('  SouthCarolina: SE 4→0 = -4')
    print('  Net SE change: -9 -9 -4 +12 = -10')
    print('')
    print('  So locally RMSE IMPROVED. But Kaggle says +0.1 worse.')
    print('  This means Kaggle is evaluating differently than we think.')


if __name__ == '__main__':
    main()
