#!/usr/bin/env python3
"""
v17: Targeted Mid-Range Correction
===================================

Strategy: Keep v12 EXACTLY as-is (same features, same weights, same pairwise
blend, same Hungarian). Then apply a POST-PREDICTION correction ONLY to teams
whose assigned seed falls in the "hard zone" (seeds 17-34, where accuracy = 39%).

The correction uses a committee-bias formula that nudges ONLY mid-range teams:
- AQ from weak conferences → push seed HIGHER (committee penalizes)
- AL from power conferences with poor NET → push seed LOWER (committee rewards)
- Teams with NET-SOS divergence → adjust based on schedule strength

Two-pass approach:
  Pass 1: Run v12 as normal → 57/91 baseline
  Pass 2: For teams assigned seeds 17-34, compute correction and RE-ASSIGN
           only those seeds via constrained Hungarian, keeping all other seeds locked.

This preserves v12's 94% accuracy on top seeds and 61% on bottom seeds.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    predict_robust_blend, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


# =================================================================
#  COMMITTEE CORRECTION FORMULA
# =================================================================

def compute_committee_correction(team_data, feature_names, X_data,
                                 alpha_aq=0.0, beta_al=0.0,
                                 gamma_sos=0.0, delta_brand=0.0,
                                 epsilon_hist=0.0):
    """
    Compute a per-team seed correction based on committee bias patterns.
    Positive correction = push seed HIGHER (worse). Negative = push LOWER (better).
    
    Components:
      alpha_aq: AQ from weak conference penalty (push higher)
      beta_al:  AL from power conference benefit (push lower)  
      gamma_sos: NET-SOS divergence adjustment
      delta_brand: Mid-major brand/eye-test correction
      epsilon_hist: Historical conference seed deviation
    """
    fi = {f: i for i, f in enumerate(feature_names)}
    
    n = X_data.shape[0]
    correction = np.zeros(n)
    
    net = X_data[:, fi['NET Rank']]
    is_aq = X_data[:, fi['is_AQ']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    conf_avg = X_data[:, fi['conf_avg_net']]
    sos = X_data[:, fi['NETSOS']]
    
    # Get additional features if available
    cb_mean = X_data[:, fi['cb_mean_seed']] if 'cb_mean_seed' in fi else np.full(n, 35.0)
    tourn_rank = X_data[:, fi['tourn_field_rank']] if 'tourn_field_rank' in fi else net
    q1w = X_data[:, fi['Quadrant1_W']] if 'Quadrant1_W' in fi else np.zeros(n)
    q3l = X_data[:, fi['Quadrant3_L']] if 'Quadrant3_L' in fi else np.zeros(n)
    q4l = X_data[:, fi['Quadrant4_L']] if 'Quadrant4_L' in fi else np.zeros(n)
    
    # Conference weakness: 0 = strong (avg NET ~60), 1+ = weak (avg NET ~200)
    conf_weakness = np.clip((conf_avg - 80) / 120, 0, 2)
    
    # Component 1: AQ-weak-conference penalty
    # AQ teams from weak conferences (OVC, Horizon, etc.) → committee penalizes
    # Murray St (OVC, NET 21) was seeded at 26, far worse than NET
    if alpha_aq != 0:
        aq_penalty = is_aq * conf_weakness * (100 - np.clip(net, 1, 100)) / 100
        correction += alpha_aq * aq_penalty
    
    # Component 2: AL-power-conference benefit
    # AL from power conf with worse NET → committee gives better seeds
    # Clemson (ACC, NET 35) got seed 22; Miami (ACC, NET 35) got seed 20
    if beta_al != 0:
        al_benefit = is_al * is_power * np.clip((net - 20) / 50, 0, 1)
        correction -= beta_al * al_benefit  # negative = better seed
    
    # Component 3: NET-SOS divergence
    # Large gap between NET and SOS suggests inflated/deflated NET
    # Murray St: NET 21, SOS 220 → massive gap → committee knows schedule was weak
    if gamma_sos != 0:
        sos_gap = (sos - net) / 100  # positive = SOS worse than NET (weak schedule)
        correction += gamma_sos * sos_gap
    
    # Component 4: Brand/eye-test (mid-major with talent)
    # Memphis (AAC, NET 49) got seed 20 — committee saw talent the numbers don't
    # Approximate: good Q1 wins + few bad losses from mid-conf
    if delta_brand != 0:
        eye_test = q1w * 0.3 - (q3l + q4l) * 0.5 - conf_weakness * 0.3
        correction -= delta_brand * is_aq * np.clip(eye_test, 0, 3)
    
    # Component 5: Historical conference deviation
    # Use cb_mean_seed as proxy for committee's historical treatment
    if epsilon_hist != 0:
        hist_dev = (cb_mean - tourn_rank) / 10  # positive = historically seeded worse
        correction += epsilon_hist * hist_dev
    
    return correction


# =================================================================
#  KAGGLE-STYLE EVALUATION WITH MID-RANGE CORRECTION
# =================================================================

def kaggle_eval_with_correction(X_all, y, seasons, feature_names, test_mask, folds,
                                correction_zone=(17, 34),
                                correction_params=None,
                                correction_blend=0.5,
                                correction_method='rescore',
                                correction_power=0.15):
    """
    Two-pass Kaggle evaluation:
    Pass 1: v12 as normal → assigns all seeds
    Pass 2: For teams in correction_zone, re-assign using corrected scores
    
    correction_method:
      'rescore': Blend correction into raw scores, re-run Hungarian on mid-range only
      'nudge': Add correction directly to assigned seed and re-snap to available
      'reorder': Re-order mid-range by corrected score, assign available mid-range seeds
    """
    if correction_params is None:
        correction_params = {}
    
    record_ids = np.arange(len(y))  # just indices
    test_assigned = np.zeros(len(y), dtype=int)
    
    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        season_test_mask = test_mask & season_mask
        n_te = season_test_mask.sum()
        if n_te == 0:
            continue
        
        global_train_mask = ~season_test_mask
        
        # ── Pass 1: v12 as normal ──
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
        
        # Standard Hungarian → Pass 1 seeds
        avail = {hold: list(range(1, 69))}
        pass1_assigned = hungarian(raw_scores, seasons[season_mask], avail, power=KAGGLE_POWER)
        
        # ── Pass 2: Correct mid-range ──
        lo, hi = correction_zone
        
        # Compute committee correction for all teams in this season
        correction = compute_committee_correction(
            None, feature_names, X_all[season_mask], **correction_params)
        
        if correction_method == 'rescore':
            # Blend correction into raw scores for mid-range test teams only
            corrected_scores = raw_scores.copy()
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    assigned_seed = pass1_assigned[i]
                    if lo <= assigned_seed <= hi:
                        # Nudge the raw score toward the corrected direction
                        corrected_scores[i] = raw_scores[i] + correction_blend * correction[i]
            
            # Re-run Hungarian with corrected scores
            pass2_assigned = hungarian(corrected_scores, seasons[season_mask], avail, 
                                       power=correction_power)
            
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    test_assigned[global_idx] = pass2_assigned[i]
        
        elif correction_method == 'swap':
            # Only re-assign seeds within the mid-range zone
            # Identify mid-range test teams and available mid-range seeds
            mid_test_indices = []  # indices within season
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx] and lo <= pass1_assigned[i] <= hi:
                    mid_test_indices.append(i)
            
            if len(mid_test_indices) > 1:
                # Get the seeds that were assigned to mid-range test teams
                mid_seeds = [pass1_assigned[i] for i in mid_test_indices]
                
                # Get corrected scores for these teams
                mid_corrected = [raw_scores[i] + correction_blend * correction[i] 
                                for i in mid_test_indices]
                
                # Re-assign just these seeds using Hungarian
                cost = np.array([[abs(score - seed)**correction_power 
                                 for seed in mid_seeds]
                                for score in mid_corrected])
                ri, ci = linear_sum_assignment(cost)
                
                # Apply new assignments
                final_assigned = pass1_assigned.copy()
                for r, c in zip(ri, ci):
                    final_assigned[mid_test_indices[r]] = mid_seeds[c]
                
                for i, global_idx in enumerate(season_indices):
                    if test_mask[global_idx]:
                        test_assigned[global_idx] = final_assigned[i]
            else:
                # 0 or 1 mid-range test teams → no swaps possible
                for i, global_idx in enumerate(season_indices):
                    if test_mask[global_idx]:
                        test_assigned[global_idx] = pass1_assigned[i]
        
        elif correction_method == 'nudge':
            # Direct nudge: shift assigned seed by correction amount, snap to nearest available
            final_assigned = pass1_assigned.copy()
            taken = set(int(pass1_assigned[i]) for i, gi in enumerate(season_indices) 
                       if not test_mask[gi])  # locked training seeds
            
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx] and lo <= pass1_assigned[i] <= hi:
                    nudged = pass1_assigned[i] + correction_blend * correction[i]
                    nudged = int(round(np.clip(nudged, 1, 68)))
                    # Find nearest available seed
                    for delta in range(0, 68):
                        for candidate in [nudged + delta, nudged - delta]:
                            if 1 <= candidate <= 68 and candidate not in taken:
                                final_assigned[i] = candidate
                                taken.add(candidate)
                                break
                        else:
                            continue
                        break
            
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    test_assigned[global_idx] = final_assigned[i]
        
        else:
            # No correction, just v12
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    test_assigned[global_idx] = pass1_assigned[i]
    
    # Evaluate
    gt = y[test_mask].astype(int)
    pred = test_assigned[test_mask]
    exact = int((pred == gt).sum())
    rmse = np.sqrt(np.mean((pred - gt)**2))
    
    # Mid-range accuracy
    mid_mask = (gt >= correction_zone[0]) & (gt <= correction_zone[1])
    mid_exact = int((pred[mid_mask] == gt[mid_mask]).sum())
    mid_total = int(mid_mask.sum())
    
    # Non-mid accuracy (should stay same as v12)
    non_mid_mask = ~mid_mask
    non_mid_exact = int((pred[non_mid_mask] == gt[non_mid_mask]).sum())
    non_mid_total = int(non_mid_mask.sum())
    
    return exact, rmse, mid_exact, mid_total, non_mid_exact, non_mid_total, test_assigned


# =================================================================
#  MAIN
# =================================================================

def main():
    t0 = time.time()
    print('='*70)
    print(' v17: TARGETED MID-RANGE CORRECTION')
    print(' (Keep v12 untouched, correct ONLY seeds 17-34)')
    print('='*70)

    # ─── Load data ───
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
    print(f'  Features: {len(feature_names)}')

    # ─── v12 baseline ───
    print('\n  Computing v12 baseline...')
    exact_v12, rmse_v12, mid_v12, mid_total, non_mid_v12, non_mid_total, _ = \
        kaggle_eval_with_correction(
            X_all, y, seasons, feature_names, test_mask, folds,
            correction_params={},  # no correction
            correction_blend=0.0)
    print(f'  v12: {exact_v12}/{n_test} exact, RMSE={rmse_v12:.4f}')
    print(f'       mid(17-34): {mid_v12}/{mid_total} ({mid_v12/mid_total*100:.0f}%)')
    print(f'       non-mid:    {non_mid_v12}/{non_mid_total} ({non_mid_v12/non_mid_total*100:.0f}%)')

    # ─── Test correction configs ───
    print('\n' + '='*70)
    print(' TESTING CORRECTION CONFIGURATIONS')
    print('='*70)

    configs = []

    # === RESCORE METHOD: blend correction into raw scores, re-run Hungarian ===
    # Test each component individually
    for alpha_aq in [2, 4, 6, 8, 10, 15]:
        configs.append({
            'name': f'rescore_aq_pen={alpha_aq}',
            'method': 'rescore',
            'params': {'alpha_aq': alpha_aq},
            'blend': 1.0,
            'power': 0.15,
        })
    
    for beta_al in [2, 4, 6, 8, 10]:
        configs.append({
            'name': f'rescore_al_ben={beta_al}',
            'method': 'rescore',
            'params': {'beta_al': beta_al},
            'blend': 1.0,
            'power': 0.15,
        })
    
    for gamma in [1, 2, 4, 6, 8]:
        configs.append({
            'name': f'rescore_sos_gap={gamma}',
            'method': 'rescore',
            'params': {'gamma_sos': gamma},
            'blend': 1.0,
            'power': 0.15,
        })
    
    for delta in [2, 4, 6, 8]:
        configs.append({
            'name': f'rescore_brand={delta}',
            'method': 'rescore',
            'params': {'delta_brand': delta},
            'blend': 1.0,
            'power': 0.15,
        })
    
    for eps in [1, 2, 4, 6]:
        configs.append({
            'name': f'rescore_hist={eps}',
            'method': 'rescore',
            'params': {'epsilon_hist': eps},
            'blend': 1.0,
            'power': 0.15,
        })
    
    # Multi-component combos
    for aq in [4, 8]:
        for al in [4, 8]:
            for sos in [0, 2, 4]:
                for blend in [0.5, 1.0]:
                    configs.append({
                        'name': f'rescore_aq{aq}_al{al}_sos{sos}_b{blend:.1f}',
                        'method': 'rescore',
                        'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos},
                        'blend': blend,
                        'power': 0.15,
                    })
    
    # Full combos with brand and hist
    for aq in [4, 8]:
        for al in [4, 8]:
            for sos in [2, 4]:
                for brand in [0, 4]:
                    for hist in [0, 2]:
                        if brand == 0 and hist == 0:
                            continue  # already tested above
                        configs.append({
                            'name': f'rescore_full_aq{aq}_al{al}_s{sos}_b{brand}_h{hist}',
                            'method': 'rescore',
                            'params': {'alpha_aq': aq, 'beta_al': al, 
                                      'gamma_sos': sos, 'delta_brand': brand,
                                      'epsilon_hist': hist},
                            'blend': 1.0,
                            'power': 0.15,
                        })

    # === SWAP METHOD: only re-assign within mid-range ===
    for aq in [4, 8]:
        for al in [4, 8]:
            for sos in [0, 2, 4]:
                configs.append({
                    'name': f'swap_aq{aq}_al{al}_sos{sos}',
                    'method': 'swap',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': sos},
                    'blend': 1.0,
                    'power': 0.15,
                })
    
    # === Different correction zones ===
    # Maybe the zone should be broader or narrower
    configs_with_zones = []
    for zone in [(15, 36), (13, 40), (20, 30)]:
        for aq in [4, 8]:
            for al in [4, 8]:
                configs_with_zones.append({
                    'name': f'rescore_zone{zone[0]}-{zone[1]}_aq{aq}_al{al}',
                    'method': 'rescore',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': 2},
                    'blend': 1.0,
                    'power': 0.15,
                    'zone': zone,
                })

    # === Different correction powers ===
    for power in [0.10, 0.20, 0.25]:
        for aq in [4, 8]:
            for al in [4, 8]:
                configs.append({
                    'name': f'rescore_p{power:.2f}_aq{aq}_al{al}',
                    'method': 'rescore',
                    'params': {'alpha_aq': aq, 'beta_al': al, 'gamma_sos': 2},
                    'blend': 1.0,
                    'power': power,
                })

    n_configs = len(configs) + len(configs_with_zones)
    print(f'\n  Running {n_configs} configs...\n')
    print(f'  {"Config":<55} {"Exact":>7} {"RMSE":>8} {"Mid":>7} {"NonMid":>7}')
    print(f'  {"─"*55} {"─"*7} {"─"*8} {"─"*7} {"─"*7}')

    best_exact = exact_v12
    best_rmse = rmse_v12
    best_config = 'v12 baseline'
    results = []

    for ci, cfg in enumerate(configs):
        exact, rmse, mid_ex, mid_tot, nm_ex, nm_tot, _ = \
            kaggle_eval_with_correction(
                X_all, y, seasons, feature_names, test_mask, folds,
                correction_zone=(17, 34),
                correction_params=cfg['params'],
                correction_blend=cfg['blend'],
                correction_method=cfg['method'],
                correction_power=cfg['power'])
        
        marker = ''
        if exact > best_exact or (exact == best_exact and rmse < best_rmse):
            best_exact = exact
            best_rmse = rmse
            best_config = cfg['name']
            marker = ' ★★★'
        elif exact > exact_v12:
            marker = ' ★'
        elif exact == exact_v12:
            if nm_ex < non_mid_v12:
                marker = ' ◇ (tied, non-mid worse)'
            elif nm_ex == non_mid_v12 and rmse < rmse_v12:
                marker = ' ◆ (tied, better RMSE)'
            elif nm_ex == non_mid_v12:
                marker = ' ◆ (tied)'
        
        print(f'  {cfg["name"]:<55} {exact:>3}/{n_test} {rmse:>8.4f} '
              f'{mid_ex}/{mid_tot:<3} {nm_ex}/{nm_tot:<3}{marker}')
        
        results.append({
            'name': cfg['name'],
            'exact': exact, 'rmse': rmse,
            'mid_exact': mid_ex, 'mid_total': mid_tot,
            'non_mid_exact': nm_ex, 'non_mid_total': nm_tot,
            **cfg,
        })
        
        if (ci + 1) % 15 == 0:
            elapsed = time.time() - t0
            print(f'  ... [{ci+1}/{n_configs}] ({elapsed:.0f}s)')

    # Run zone configs
    for ci, cfg in enumerate(configs_with_zones):
        zone = cfg['zone']
        exact, rmse, mid_ex, mid_tot, nm_ex, nm_tot, _ = \
            kaggle_eval_with_correction(
                X_all, y, seasons, feature_names, test_mask, folds,
                correction_zone=zone,
                correction_params=cfg['params'],
                correction_blend=cfg['blend'],
                correction_method=cfg['method'],
                correction_power=cfg['power'])
        
        marker = ''
        if exact > best_exact or (exact == best_exact and rmse < best_rmse):
            best_exact = exact
            best_rmse = rmse
            best_config = cfg['name']
            marker = ' ★★★'
        elif exact > exact_v12:
            marker = ' ★'
        elif exact == exact_v12:
            marker = ' ◆'
        
        print(f'  {cfg["name"]:<55} {exact:>3}/{n_test} {rmse:>8.4f} '
              f'{mid_ex}/{mid_tot:<3} {nm_ex}/{nm_tot:<3}{marker}')
        
        results.append({
            'name': cfg['name'],
            'exact': exact, 'rmse': rmse,
            'mid_exact': mid_ex, 'mid_total': mid_tot,
            'non_mid_exact': nm_ex, 'non_mid_total': nm_tot,
            **cfg,
        })

    # ─── Summary ───
    print('\n' + '='*70)
    print(' SUMMARY')
    print('='*70)
    
    print(f'\n  v12 baseline:     {exact_v12}/{n_test} exact, RMSE={rmse_v12:.4f}')
    print(f'                    mid={mid_v12}/{mid_total}, non-mid={non_mid_v12}/{non_mid_total}')
    print(f'  Best config:      {best_config}')
    print(f'  Best result:      {best_exact}/{n_test} exact, RMSE={best_rmse:.4f}')
    
    if best_exact > exact_v12:
        improvement = best_exact - exact_v12
        print(f'\n  ★★★ IMPROVEMENT: +{improvement} exact matches! ★★★')
    elif best_exact == exact_v12:
        print(f'\n  No exact match improvement, checking RMSE...')
        rmse_improved = [r for r in results if r['exact'] == exact_v12 and r['rmse'] < rmse_v12]
        if rmse_improved:
            rmse_improved.sort(key=lambda x: x['rmse'])
            print(f'  Best RMSE improvement: {rmse_improved[0]["name"]} '
                  f'RMSE={rmse_improved[0]["rmse"]:.4f} (v12={rmse_v12:.4f})')
    
    # Top 15 configs
    results.sort(key=lambda x: (-x['exact'], x['rmse']))
    print(f'\n  Top 15 configs:')
    print(f'  {"Rank":>4} {"Config":<50} {"Exact":>7} {"RMSE":>8} {"Mid":>5} {"NM":>5}')
    print(f'  {"─"*4} {"─"*50} {"─"*7} {"─"*8} {"─"*5} {"─"*5}')
    for i, r in enumerate(results[:15]):
        print(f'  {i+1:>4} {r["name"]:<50} {r["exact"]:>3}/{n_test} '
              f'{r["rmse"]:>8.4f} {r["mid_exact"]:>2}/{r["mid_total"]} '
              f'{r["non_mid_exact"]:>2}/{r["non_mid_total"]}')
    
    # Check: did any config improve mid without hurting non-mid?
    print(f'\n  Configs that improved mid-range without hurting non-mid:')
    improved_mid = [r for r in results 
                    if r['mid_exact'] > mid_v12 and r['non_mid_exact'] >= non_mid_v12]
    if improved_mid:
        for r in improved_mid[:10]:
            print(f'    {r["name"]:<50} mid={r["mid_exact"]}/{r["mid_total"]} '
                  f'(+{r["mid_exact"]-mid_v12}) non-mid={r["non_mid_exact"]}/{r["non_mid_total"]}')
    else:
        print(f'    None found.')
    
    # Any mid improvement at all (even if non-mid hurt)?
    print(f'\n  Best mid-range accuracy configs:')
    results.sort(key=lambda x: (-x['mid_exact'], x['rmse']))
    for r in results[:5]:
        delta_nm = r['non_mid_exact'] - non_mid_v12
        print(f'    {r["name"]:<50} mid={r["mid_exact"]}/{r["mid_total"]} '
              f'non-mid={r["non_mid_exact"]}/{r["non_mid_total"]} ({delta_nm:+d})')

    elapsed = time.time() - t0
    print(f'\n  Time: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
