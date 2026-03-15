#!/usr/bin/env python3
"""
Deep analysis of v18's 30 remaining errors.
Find patterns we can exploit for v19.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    USE_TOP_K_A, FORCE_FEATURES,
    MIDRANGE_ZONE, CORRECTION_AQ, CORRECTION_AL, CORRECTION_SOS,
    CORRECTION_BLEND, CORRECTION_POWER
)
from sklearn.impute import KNNImputer
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


def main():
    t0 = time.time()
    print('='*70)
    print(' REMAINING ERROR ANALYSIS (v18: 30 misses out of 91)')
    print('='*70)

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
    fi = {f: i for i, f in enumerate(feature_names)}

    # Run v18 to get predictions
    test_assigned = np.zeros(n_labeled, dtype=int)
    v12_assigned = np.zeros(n_labeled, dtype=int)
    raw_scores_all = np.zeros(n_labeled)

    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue

        global_train_mask = ~season_test_mask
        X_season = X_all[season_mask]

        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]

        raw = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_season, seasons[global_train_mask], top_k_idx)

        for i, gi in enumerate(season_indices):
            if not test_mask[gi]:
                raw[i] = y[gi]
            else:
                raw_scores_all[gi] = raw[i]

        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw, seasons[season_mask], avail, power=KAGGLE_POWER)

        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                v12_assigned[gi] = pass1[i]

        correction = compute_committee_correction(
            feature_names, X_season,
            alpha_aq=CORRECTION_AQ, beta_al=CORRECTION_AL, gamma_sos=CORRECTION_SOS)
        test_mask_season = np.array([test_mask[gi] for gi in season_indices])
        pass2 = apply_midrange_swap(
            pass1.copy(), raw.copy(), correction, test_mask_season, season_indices,
            zone=MIDRANGE_ZONE, blend=CORRECTION_BLEND, power=CORRECTION_POWER)

        for i, gi in enumerate(season_indices):
            if test_mask[gi]:
                test_assigned[gi] = pass2[i]

    # Analyze errors
    test_indices = np.where(test_mask)[0]
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]

    correct = pred == gt
    errors = ~correct
    n_correct = correct.sum()
    n_errors = errors.sum()

    print(f'\n  Total: {n_correct}/91 correct, {n_errors} errors')

    # Detailed error table
    print(f'\n  {"RecordID":<30} {"Pred":>4} {"True":>4} {"Err":>4} {"AbsE":>4} '
          f'{"Raw":>6} {"NET":>4} {"Conf":<8} {"Bid":<3} {"Pwr":>3} {"Zone":<6}')
    print(f'  {"─"*30} {"─"*4} {"─"*4} {"─"*4} {"─"*4} {"─"*6} {"─"*4} {"─"*8} {"─"*3} {"─"*3} {"─"*6}')

    error_details = []
    for idx in test_indices:
        p = test_assigned[idx]
        t = int(y[idx])
        err = p - t
        if err == 0:
            continue

        net = X_all[idx, fi['NET Rank']]
        is_aq = int(X_all[idx, fi['is_AQ']])
        is_al = int(X_all[idx, fi['is_AL']])
        is_power = int(X_all[idx, fi['is_power_conf']])
        conf_avg = X_all[idx, fi['conf_avg_net']]
        sos = X_all[idx, fi['NETSOS']]
        raw = raw_scores_all[idx]
        bid = 'AQ' if is_aq else 'AL'

        # Seed zone
        if t <= 16:
            zone = 'top'
        elif t <= 34:
            zone = 'mid'
        elif t <= 52:
            zone = 'low'
        else:
            zone = 'bottom'

        error_details.append({
            'rid': record_ids[idx], 'pred': p, 'true': t, 'err': err,
            'abs_err': abs(err), 'raw': raw, 'net': net,
            'bid': bid, 'power': is_power, 'zone': zone,
            'conf_avg': conf_avg, 'sos': sos,
            'v12_pred': v12_assigned[idx]
        })

        print(f'  {record_ids[idx]:<30} {p:4d} {t:4d} {err:+4d} {abs(err):4d} '
              f'{raw:6.1f} {net:4.0f} {"PWR" if is_power else "MID":8s} {bid:<3} '
              f'{is_power:3d} {zone:<6}')

    # Summary stats
    print(f'\n' + '='*70)
    print(' ERROR PATTERNS')
    print('='*70)

    errs = error_details
    abs_errs = [e['abs_err'] for e in errs]
    print(f'\n  Absolute error distribution:')
    for ae in sorted(set(abs_errs)):
        count = sum(1 for e in errs if e['abs_err'] == ae)
        teams = [e['rid'].split('-')[-1] for e in errs if e['abs_err'] == ae]
        print(f'    |err|={ae}: {count} teams — {", ".join(teams[:5])}{"..." if len(teams)>5 else ""}')

    print(f'\n  By zone (true seed):')
    for zone in ['top', 'mid', 'low', 'bottom']:
        zone_errs = [e for e in errs if e['zone'] == zone]
        total_in_zone = sum(1 for i in test_indices if
                           (int(y[i]) <= 16 and zone == 'top') or
                           (17 <= int(y[i]) <= 34 and zone == 'mid') or
                           (35 <= int(y[i]) <= 52 and zone == 'low') or
                           (int(y[i]) >= 53 and zone == 'bottom'))
        correct_in_zone = total_in_zone - len(zone_errs)
        print(f'    {zone:6s}: {correct_in_zone}/{total_in_zone} correct, '
              f'{len(zone_errs)} errors, avg|err|={np.mean([e["abs_err"] for e in zone_errs]):.1f}' if zone_errs else f'    {zone:6s}: {correct_in_zone}/{total_in_zone} correct, 0 errors')

    print(f'\n  By bid type:')
    for bid in ['AQ', 'AL']:
        bid_errs = [e for e in errs if e['bid'] == bid]
        total_bid = sum(1 for i in test_indices if
                       (int(X_all[i, fi['is_AQ']]) == 1 and bid == 'AQ') or
                       (int(X_all[i, fi['is_AL']]) == 1 and bid == 'AL'))
        correct_bid = total_bid - len(bid_errs)
        if bid_errs:
            print(f'    {bid}: {correct_bid}/{total_bid} correct, {len(bid_errs)} errors, '
                  f'avg|err|={np.mean([e["abs_err"] for e in bid_errs]):.1f}')

    print(f'\n  By conference type:')
    for pwr in [1, 0]:
        pwr_errs = [e for e in errs if e['power'] == pwr]
        label = 'Power' if pwr else 'Mid-maj'
        total_pwr = sum(1 for i in test_indices if int(X_all[i, fi['is_power_conf']]) == pwr)
        correct_pwr = total_pwr - len(pwr_errs)
        if pwr_errs:
            print(f'    {label:7s}: {correct_pwr}/{total_pwr} correct, {len(pwr_errs)} errors, '
                  f'avg|err|={np.mean([e["abs_err"] for e in pwr_errs]):.1f}')

    print(f'\n  Direction of errors:')
    over = [e for e in errs if e['err'] > 0]
    under = [e for e in errs if e['err'] < 0]
    print(f'    Overseeded (pred > true): {len(over)} teams, avg={np.mean([e["err"] for e in over]):+.1f}')
    print(f'    Underseeded (pred < true): {len(under)} teams, avg={np.mean([e["err"] for e in under]):+.1f}')

    print(f'\n  By season:')
    for s in folds:
        s_errs = [e for e in errs if e['rid'].startswith(s)]
        s_total = sum(1 for i in test_indices if seasons[i] == s)
        s_correct = s_total - len(s_errs)
        if s_errs:
            print(f'    {s}: {s_correct}/{s_total} correct, {len(s_errs)} errors, '
                  f'avg|err|={np.mean([e["abs_err"] for e in s_errs]):.1f}')
        else:
            print(f'    {s}: {s_total}/{s_total} correct')

    # Swap analysis: which error pairs could be fixed by swapping?
    print(f'\n  Swap-fixable errors (pairs where swapping seeds fixes both):')
    for s in folds:
        s_errs = [e for e in errs if e['rid'].startswith(s)]
        for i in range(len(s_errs)):
            for j in range(i+1, len(s_errs)):
                e1, e2 = s_errs[i], s_errs[j]
                # If swapping predictions would fix both
                if e1['pred'] == e2['true'] and e2['pred'] == e1['true']:
                    print(f'    PERFECT SWAP: {e1["rid"]} (pred={e1["pred"]},true={e1["true"]}) '
                          f'<-> {e2["rid"]} (pred={e2["pred"]},true={e2["true"]})')
                # If swapping reduces total error
                elif (abs(e1['pred'] - e2['true']) + abs(e2['pred'] - e1['true']) <
                      abs(e1['err']) + abs(e2['err'])):
                    new_err1 = e1['pred'] - e2['true']
                    new_err2 = e2['pred'] - e1['true']
                    print(f'    PARTIAL SWAP: {e1["rid"]} (pred={e1["pred"]},true={e1["true"]}) '
                          f'<-> {e2["rid"]} (pred={e2["pred"]},true={e2["true"]}) '
                          f'→ reduces error by {abs(e1["err"]) + abs(e2["err"]) - abs(new_err1) - abs(new_err2)}')

    # Raw score analysis: how far were the raw scores from the true seeds?
    print(f'\n  Raw score vs true seed (for errors):')
    print(f'  {"RecordID":<30} {"Raw":>6} {"True":>4} {"RawErr":>6} {"Pred":>4} {"PredErr":>6} {"RawBetter":>9}')
    for e in sorted(errs, key=lambda x: -x['abs_err']):
        raw_err = e['raw'] - e['true']
        pred_better = abs(e['raw'] - e['true']) < abs(e['pred'] - e['true'])
        print(f'  {e["rid"]:<30} {e["raw"]:6.1f} {e["true"]:4d} {raw_err:+6.1f} '
              f'{e["pred"]:4d} {e["err"]:+6d} {"YES" if pred_better else "no":>9}')

    # Feature importance for errors vs correct
    print(f'\n  Feature differences (errors vs correct):')
    err_indices = [i for i in test_indices if test_assigned[i] != int(y[i])]
    cor_indices = [i for i in test_indices if test_assigned[i] == int(y[i])]

    for fname in ['NET Rank', 'NETSOS', 'conf_avg_net', 'is_AQ', 'is_AL',
                   'is_power_conf', 'cb_mean_seed', 'Quadrant1_W', 'Quadrant3_L',
                   'Quadrant4_L', 'tourn_field_rank', 'W_pct']:
        if fname in fi:
            err_vals = X_all[err_indices, fi[fname]]
            cor_vals = X_all[cor_indices, fi[fname]]
            print(f'    {fname:20s}: errors={np.mean(err_vals):6.1f}±{np.std(err_vals):5.1f}, '
                  f'correct={np.mean(cor_vals):6.1f}±{np.std(cor_vals):5.1f}')

    # What would oracle do? 
    print(f'\n  Oracle analysis (if we had perfect knowledge):')
    print(f'    30 errors × avg |err|={np.mean(abs_errs):.1f}')
    print(f'    |err|=1: {sum(1 for e in errs if e["abs_err"]==1)} (adjacent, possibly noise)')
    print(f'    |err|=2: {sum(1 for e in errs if e["abs_err"]==2)} (within region, fixable?)')
    print(f'    |err|≥3: {sum(1 for e in errs if e["abs_err"]>=3)} (structural errors)')
    big_errs = [e for e in errs if e['abs_err'] >= 3]
    if big_errs:
        print(f'\n    Big errors (|err|≥3):')
        for e in sorted(big_errs, key=lambda x: -x['abs_err']):
            print(f'      {e["rid"]:<28} pred={e["pred"]:2d} true={e["true"]:2d} err={e["err"]:+d} '
                  f'NET={e["net"]:.0f} bid={e["bid"]} pwr={e["power"]}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
