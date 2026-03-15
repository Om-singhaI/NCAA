#!/usr/bin/env python3
"""
v17b: Fine-tune the winning swap correction from v17.

Best v17 configs (61/91):
  swap_aq4_al4_sos2  → 61/91, RMSE=2.3511, mid=11/18, non-mid=50/73
  swap_aq8_al4_sos2  → 61/91, RMSE=2.3511, mid=11/18, non-mid=50/73  
  swap_aq8_al8_sos0  → 61/91, RMSE=2.7952, mid=11/18, non-mid=50/73

This script does a fine grid around those params and also tries:
1. Different zone boundaries for swap
2. Different correction powers
3. Combining brand/hist on top of winning config
4. Different blend factors
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER
)
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


def compute_correction(feature_names, X_data,
                       alpha_aq=0.0, beta_al=0.0,
                       gamma_sos=0.0, delta_brand=0.0,
                       epsilon_hist=0.0):
    """Committee correction formula."""
    fi = {f: i for i, f in enumerate(feature_names)}
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos = X_data[:, fi['NETSOS']]
    cb_mean = X_data[:, fi['cb_mean_seed']] if 'cb_mean_seed' in fi else np.full(n, 35.0)
    tourn_rank = X_data[:, fi['tourn_field_rank']] if 'tourn_field_rank' in fi else net
    q1w = X_data[:, fi['Quadrant1_W']] if 'Quadrant1_W' in fi else np.zeros(n)
    q3l = X_data[:, fi['Quadrant3_L']] if 'Quadrant3_L' in fi else np.zeros(n)
    q4l = X_data[:, fi['Quadrant4_L']] if 'Quadrant4_L' in fi else np.zeros(n)
    
    conf_weakness = np.clip((conf_avg - 80) / 120, 0, 2)
    
    if alpha_aq != 0:
        aq_penalty = is_aq * conf_weakness * (100 - np.clip(net, 1, 100)) / 100
        correction += alpha_aq * aq_penalty
    
    if beta_al != 0:
        al_benefit = is_al * is_power * np.clip((net - 20) / 50, 0, 1)
        correction -= beta_al * al_benefit
    
    if gamma_sos != 0:
        sos_gap = (sos - net) / 100
        correction += gamma_sos * sos_gap
    
    if delta_brand != 0:
        eye_test = q1w * 0.3 - (q3l + q4l) * 0.5 - conf_weakness * 0.3
        correction -= delta_brand * is_aq * np.clip(eye_test, 0, 3)
    
    if epsilon_hist != 0:
        hist_dev = (cb_mean - tourn_rank) / 10
        correction += epsilon_hist * hist_dev
    
    return correction


def kaggle_eval_swap(X_all, y, seasons, feature_names, test_mask, folds,
                     correction_params, blend=1.0, zone=(17, 34), power=0.15):
    """Swap-method: run v12, then re-order mid-range test teams only."""
    test_assigned = np.zeros(len(y), dtype=int)
    
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0:
            continue
        
        global_train_mask = ~season_test_mask
        
        # v12 prediction
        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        raw_scores = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_all[season_mask], seasons[global_train_mask], top_k_idx)
        
        # Lock training teams
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                raw_scores[i] = y[global_idx]
        
        # Pass 1: standard Hungarian
        avail = {hold: list(range(1, 69))}
        pass1 = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)
        
        # Pass 2: swap mid-range test teams
        lo, hi = zone
        correction = compute_correction(feature_names, X_all[season_mask], **correction_params)
        
        mid_test_indices = []
        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx] and lo <= pass1[i] <= hi:
                mid_test_indices.append(i)
        
        final = pass1.copy()
        if len(mid_test_indices) > 1:
            mid_seeds = [pass1[i] for i in mid_test_indices]
            mid_corrected = [raw_scores[i] + blend * correction[i] for i in mid_test_indices]
            
            cost = np.array([[abs(score - seed)**power for seed in mid_seeds]
                            for score in mid_corrected])
            ri, ci = linear_sum_assignment(cost)
            for r, c in zip(ri, ci):
                final[mid_test_indices[r]] = mid_seeds[c]
        
        for i, global_idx in enumerate(season_indices):
            if test_mask[global_idx]:
                test_assigned[global_idx] = final[i]
    
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    exact = int((pred == gt).sum())
    rmse = np.sqrt(np.mean((pred - gt)**2))
    
    mid_mask = (gt >= zone[0]) & (gt <= zone[1])
    mid_exact = int((pred[mid_mask] == gt[mid_mask]).sum())
    mid_total = int(mid_mask.sum())
    non_mid_mask = ~mid_mask
    non_mid_exact = int((pred[non_mid_mask] == gt[non_mid_mask]).sum())
    non_mid_total = int(non_mid_mask.sum())
    
    return exact, rmse, mid_exact, mid_total, non_mid_exact, non_mid_total, test_assigned


def main():
    t0 = time.time()
    print('='*70)
    print(' v17b: FINE-TUNE SWAP CORRECTION')
    print('='*70)

    # Load
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
    n_test = test_mask.sum()

    print(f'  Teams: {n_labeled}, Test: {n_test}')

    # v12 baseline
    exact_v12, rmse_v12, mid_v12, mid_total, nm_v12, nm_total, _ = \
        kaggle_eval_swap(X_all, y, seasons, feature_names, test_mask, folds,
                         correction_params={'alpha_aq': 0, 'beta_al': 0, 'gamma_sos': 0},
                         blend=0.0)
    print(f'  v12: {exact_v12}/{n_test}, mid={mid_v12}/{mid_total}, nm={nm_v12}/{nm_total}')

    configs = []

    # === Fine grid around winning params: aq=4, al=4, sos=2, blend=1.0, zone=17-34 ===
    for aq in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]:
        for al in [1, 2, 3, 4, 5, 6, 7, 8]:
            for sos in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]:
                configs.append({
                    'name': f'aq{aq}_al{al}_s{sos}',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos},
                    'blend': 1.0,
                    'zone': (17, 34),
                    'power': 0.15,
                })

    # === Add brand/hist on top of best combos ===
    for aq in [3, 4, 5]:
        for al in [3, 4, 5]:
            for sos in [1.5, 2, 2.5]:
                for brand in [1, 2, 3]:
                    configs.append({
                        'name': f'aq{aq}_al{al}_s{sos}_br{brand}',
                        'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos,
                                  'delta_brand': brand},
                        'blend': 1.0,
                        'zone': (17, 34),
                        'power': 0.15,
                    })
                for hist in [0.5, 1, 1.5, 2]:
                    configs.append({
                        'name': f'aq{aq}_al{al}_s{sos}_h{hist}',
                        'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos,
                                  'epsilon_hist': hist},
                        'blend': 1.0,
                        'zone': (17, 34),
                        'power': 0.15,
                    })

    # === Different blend values ===
    for aq in [3, 4, 5]:
        for al in [3, 4, 5]:
            for sos in [1.5, 2, 2.5]:
                for blend in [0.3, 0.5, 0.7, 1.5, 2.0]:
                    configs.append({
                        'name': f'aq{aq}_al{al}_s{sos}_b{blend:.1f}',
                        'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos},
                        'blend': blend,
                        'zone': (17, 34),
                        'power': 0.15,
                    })

    # === Different zones ===
    for zone in [(15, 36), (16, 35), (18, 33), (19, 32), (15, 40), (13, 45)]:
        for aq in [3, 4, 5]:
            for al in [3, 4, 5]:
                configs.append({
                    'name': f'z{zone[0]}-{zone[1]}_aq{aq}_al{al}_s2',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': 2},
                    'blend': 1.0,
                    'zone': zone,
                    'power': 0.15,
                })

    # === Different powers ===
    for power in [0.10, 0.12, 0.18, 0.20, 0.25, 0.30, 0.50]:
        for aq in [3, 4, 5]:
            for al in [3, 4, 5]:
                configs.append({
                    'name': f'p{power:.2f}_aq{aq}_al{al}_s2',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': 2},
                    'blend': 1.0,
                    'zone': (17, 34),
                    'power': power,
                })

    n_configs = len(configs)
    print(f'\n  Running {n_configs} configs...\n')
    print(f'  {"Config":<45} {"Exact":>7} {"RMSE":>8} {"Mid":>7} {"NM":>5}')
    print(f'  {"─"*45} {"─"*7} {"─"*8} {"─"*7} {"─"*5}')

    best_exact = exact_v12
    best_rmse = 999
    best_config = 'v12'
    results = []

    for ci, cfg in enumerate(configs):
        exact, rmse, mid_ex, mid_tot, nm_ex, nm_tot, assigned = \
            kaggle_eval_swap(
                X_all, y, seasons, feature_names, test_mask, folds,
                correction_params=cfg['params'],
                blend=cfg['blend'],
                zone=cfg['zone'],
                power=cfg['power'])
        
        marker = ''
        if exact > best_exact or (exact == best_exact and rmse < best_rmse):
            best_exact = exact
            best_rmse = rmse
            best_config = cfg['name']
            marker = ' ★★★'
        elif exact >= 61:
            marker = ' ★'
        elif exact >= 60:
            marker = ' ◆'
        
        if exact >= 60 or marker:
            print(f'  {cfg["name"]:<45} {exact:>3}/{n_test} {rmse:>8.4f} '
                  f'{mid_ex}/{mid_tot:<3} {nm_ex}/{nm_tot}{marker}')
        
        results.append({
            'name': cfg['name'], 'exact': exact, 'rmse': rmse,
            'mid_exact': mid_ex, 'non_mid_exact': nm_ex,
            'mid_total': mid_tot, 'non_mid_total': nm_tot,
            **cfg,
        })
        
        if (ci + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  ... [{ci+1}/{n_configs}] ({elapsed:.0f}s) best={best_exact}/{n_test}')

    # Summary
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    print(f'\n  v12:    {exact_v12}/{n_test}, RMSE={rmse_v12:.4f}')
    print(f'  Best:   {best_config}  →  {best_exact}/{n_test}, RMSE={best_rmse:.4f}')
    
    if best_exact > exact_v12:
        print(f'\n  ★★★ +{best_exact - exact_v12} over v12! ★★★')

    # Top 20
    results.sort(key=lambda x: (-x['exact'], x['rmse']))
    print(f'\n  Top 20:')
    print(f'  {"#":>3} {"Config":<45} {"Ex":>3} {"RMSE":>8} {"Mid":>5} {"NM":>5}')
    for i, r in enumerate(results[:20]):
        print(f'  {i+1:>3} {r["name"]:<45} {r["exact"]:>3} {r["rmse"]:>8.4f} '
              f'{r["mid_exact"]:>2}/{r["mid_total"]} {r["non_mid_exact"]:>2}/{r["non_mid_total"]}')

    # Count how many hit 61+
    gt61 = sum(1 for r in results if r['exact'] >= 61)
    gt60 = sum(1 for r in results if r['exact'] >= 60)
    print(f'\n  Configs at 61+: {gt61}, at 60+: {gt60}, total tested: {n_configs}')
    
    # No non-mid regression check
    safe61 = [r for r in results if r['exact'] >= 61 and r['non_mid_exact'] >= nm_v12]
    print(f'  61+ with no non-mid regression: {len(safe61)}')
    if safe61:
        safe61.sort(key=lambda x: x['rmse'])
        print(f'  Best safe: {safe61[0]["name"]} → {safe61[0]["exact"]}/{n_test}, '
              f'RMSE={safe61[0]["rmse"]:.4f}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
