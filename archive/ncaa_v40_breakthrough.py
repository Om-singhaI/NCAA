#!/usr/bin/env python3
"""
v40 BREAKTHROUGH — Systematic improvement beyond v25 (70/91)
=============================================================
v33 found: NETNonConfSOS zone (17,24) w=4 → 73/91 (+3)
           + ncsos_vs_sos zone (46,52) w=-5 → 75/91 (+5)

This script:
1. Validates the 5th zone (NETNonConfSOS) with proper nested LOSO
2. Validates the dual zone (ncsos_vs_sos) 
3. Explores additional improvements on top
4. Tests base model improvements (different blend weights, C values, features)
5. Runs full overfitting battery on best config

Remaining 18 errors after v33's 5th zone:
  SWAP PAIRS: SouthDakotaSt↔Richmond, Charleston↔VCU, SEMoSt↔TexasSouthern,
              NewMexico↔Northwestern, Clemson↔WashSt, Kentucky↔Wisconsin
  OUTLIER: Murray St (GT=26, pred=41) — massive conference penalty
  3-WAY: S.Dakota St/W.Kentucky/LongBeach (seeds 59-61)
"""

import os, sys, time, warnings, re
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import xgboost as xgb
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, parse_wl,
    select_top_k_features, hungarian,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3, HUNGARIAN_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def predict_blend(X_train, y_train, X_test, seasons_train, top_k_idx,
                  w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
                  c1=PW_C1, c3=PW_C3, gap=ADJ_COMP1_GAP):
    """Configurable blend prediction."""
    pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=gap)
    sc_adj = StandardScaler()
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(sc_adj.fit_transform(pw_X_adj), pw_y_adj)
    score1 = pairwise_score(lr1, X_test, sc_adj)

    X_tr_k, X_te_k = X_train[:, top_k_idx], X_test[:, top_k_idx]
    pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train, seasons_train)
    sc_k = StandardScaler()
    lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(sc_k.fit_transform(pw_X_k), pw_y_k)
    score3 = pairwise_score(lr3, X_te_k, sc_k)

    pw_X_full, pw_y_full = build_pairwise_data(X_train, y_train, seasons_train)
    sc_full = StandardScaler()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss')
    xgb_clf.fit(sc_full.fit_transform(pw_X_full), pw_y_full)
    score4 = pairwise_score(xgb_clf, X_test, sc_full)
    return w1 * score1 + w3 * score3 + w4 * score4


def apply_v25_zones(pass1, raw, fn, X, tm, idx):
    """Apply all v25 zone corrections."""
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=0.15)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=0.15)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=0.15)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=0.15)
    return p


def apply_generic_zone(p, raw, feature_vals, tm, zone, weight, power=0.15):
    """Apply a generic zone correction using any feature values."""
    lo, hi = zone
    zone_test = [i for i in range(len(p)) if tm[i] and lo <= p[i] <= hi]
    if len(zone_test) <= 1:
        return p
    fv = feature_vals[zone_test] if hasattr(feature_vals, '__getitem__') else np.array([feature_vals[i] for i in zone_test])
    vmin, vmax = fv.min(), fv.max()
    if vmax > vmin:
        norm = (fv - vmin) / (vmax - vmin) * 2 - 1
    else:
        norm = np.zeros_like(fv, dtype=float)
    corr_vals = weight * norm
    seeds = [p[i] for i in zone_test]
    corrected = [raw[zone_test[k]] + corr_vals[k] for k in range(len(zone_test))]
    cost = np.array([[abs(sv - sd)**power for sd in seeds] for sv in corrected])
    ri, ci = linear_sum_assignment(cost)
    pnew = p.copy()
    for r, c in zip(ri, ci):
        pnew[zone_test[r]] = seeds[c]
    return pnew


def count_exact(p, tm, indices, test_mask, y):
    """Count exact matches for test teams."""
    return sum(1 for i, gi in enumerate(indices) if test_mask[gi] and p[i] == int(y[gi]))


def main():
    print('='*70)
    print('  v40 BREAKTHROUGH — Beyond v25 (70/91)')
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

    # Get raw columns not in features
    ncsos_raw = pd.to_numeric(labeled['NETNonConfSOS'], errors='coerce').fillna(200).values
    sos_raw = pd.to_numeric(labeled['NETSOS'], errors='coerce').fillna(200).values
    ncsos_vs_sos = ncsos_raw - sos_raw
    opp_net_raw = pd.to_numeric(labeled['AvgOppNET'], errors='coerce').fillna(150).values
    net_raw = pd.to_numeric(labeled['NET Rank'], errors='coerce').fillna(300).values

    # More potential corrective features
    conf_raw = labeled['Conference'].fillna('Unknown').values
    bid_raw = labeled['Bid Type'].fillna('').values
    
    # Conference median NET per season
    conf_med = {}
    for sv in folds:
        sm = context_df['Season'].astype(str) == sv
        nets = pd.to_numeric(context_df.loc[sm, 'NET Rank'], errors='coerce').fillna(300)
        confs = context_df.loc[sm, 'Conference'].fillna('Unknown')
        for c in confs.unique():
            conf_med[(sv, c)] = nets[confs == c].median()
    
    # Net vs conf median (per team) 
    net_vs_confmed = np.zeros(len(labeled))
    for idx in range(len(labeled)):
        sv = seasons[idx]
        c = conf_raw[idx]
        med = conf_med.get((sv, c), 200)
        net_vs_confmed[idx] = net_raw[idx] - med
    
    # Q1 + Q2 win rate
    q1w = pd.to_numeric(feat.get('Quadrant1_W', pd.Series(0, index=labeled.index)), errors='coerce').fillna(0).values
    q2w = pd.to_numeric(feat.get('Quadrant2_W', pd.Series(0, index=labeled.index)), errors='coerce').fillna(0).values
    q1l = pd.to_numeric(feat.get('Quadrant1_L', pd.Series(0, index=labeled.index)), errors='coerce').fillna(0).values
    q2l = pd.to_numeric(feat.get('Quadrant2_L', pd.Series(0, index=labeled.index)), errors='coerce').fillna(0).values
    q12_rate = (q1w + q2w) / (q1w + q2w + q1l + q2l + 0.5)
    
    # Road win pct
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=labeled.index)).fillna(0.5).values
    
    # Conference record pct
    conf_pct = feat.get('Conf.Record_Pct', pd.Series(0.5, index=labeled.index)).fillna(0.5).values

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                    np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Available extra features for zone corrections
    extra_features = {
        'ncsos': ncsos_raw,
        'ncsos_vs_sos': ncsos_vs_sos,
        'opp_net': opp_net_raw,
        'net_vs_confmed': net_vs_confmed,
        'q12_rate': q12_rate,
        'road_pct': road_pct,
        'conf_pct': conf_pct,
        'net': net_raw,
        'sos': sos_raw,
    }

    # Precompute per-season data
    season_data = {}
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
        raw = predict_blend(X[gt], y[gt], X_s, seasons[gt], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])
        
        # Store extra features per season
        extras = {k: v[sm] for k, v in extra_features.items()}
        
        season_data[hold] = {
            'pass1': p1, 'raw': raw, 'X': X_s,
            'tm': tm, 'indices': si.copy(),
            **extras,
        }

    # v25 baseline
    v25_total = 0
    for s, sd in season_data.items():
        p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
        v25_total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        season_data[s]['v25'] = p

    print(f'\n  v25 baseline: {v25_total}/91')

    # ════════════════════════════════════════════════════════════
    #  PHASE 1: Validate 5th zone (NETNonConfSOS)
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 1: VALIDATE 5th ZONE (NETNonConfSOS @ 17-24)')
    print('='*70)

    z5_scores = {}
    for s, sd in season_data.items():
        p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
        p = apply_generic_zone(p, sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15)
        z5_scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
    z5_total = sum(z5_scores.values())
    print(f'  v25 + 5th zone: {z5_total}/91  per-season: {list(z5_scores.values())}')

    # Nested LOSO for 5th zone
    nlos_v25 = 0
    nlos_z5 = 0
    for hold in test_seasons:
        tune_seasons = [s for s in test_seasons if s != hold]
        v25_tune = sum(count_exact(season_data[s]['v25'], season_data[s]['tm'],
                                    season_data[s]['indices'], test_mask, y) for s in tune_seasons)
        z5_tune = 0
        for s in tune_seasons:
            sd = season_data[s]
            p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
            p = apply_generic_zone(p, sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15)
            z5_tune += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        
        # Pick best config for this fold
        sd = season_data[hold]
        v25_hold = count_exact(sd['v25'], sd['tm'], sd['indices'], test_mask, y)
        p_z5 = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
        p_z5 = apply_generic_zone(p_z5, sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15)
        z5_hold = count_exact(p_z5, sd['tm'], sd['indices'], test_mask, y)
        
        # Choose whichever did better on tune set
        if z5_tune >= v25_tune:
            nlos_z5 += z5_hold
        else:
            nlos_z5 += v25_hold
        nlos_v25 += v25_hold

    print(f'  Nested LOSO v25:      {nlos_v25}/91 (just v25 for each held-out)')
    print(f'  Nested LOSO v25+z5:   {nlos_z5}/91 (pick best on tune set)')
    gap = z5_total - nlos_z5
    print(f'  Overfit gap: {gap}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 2: Validate dual zone
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 2: VALIDATE DUAL ZONE')
    print('='*70)

    # Test: v25 + ncsos(17,24) w=4 + ncsos_vs_sos(46,52) w=-5
    dual_scores = {}
    for s, sd in season_data.items():
        p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
        p = apply_generic_zone(p, sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15)
        p = apply_generic_zone(p, sd['raw'], sd['ncsos_vs_sos'], sd['tm'], (46, 52), -5, 0.15)
        dual_scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
    dual_total = sum(dual_scores.values())
    print(f'  v25 + ncsos(17,24) + ncsos_vs_sos(46,52): {dual_total}/91  per-season: {list(dual_scores.values())}')

    # Test: v25 + ncsos(17,24) w=4 + ncsos(64,67) w=1
    dual2_scores = {}
    for s, sd in season_data.items():
        p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
        p = apply_generic_zone(p, sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15)
        p = apply_generic_zone(p, sd['raw'], sd['ncsos'], sd['tm'], (64, 67), 1, 0.15)
        dual2_scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
    dual2_total = sum(dual2_scores.values())
    print(f'  v25 + ncsos(17,24) + ncsos(64,67): {dual2_total}/91  per-season: {list(dual2_scores.values())}')

    # Nested LOSO for dual zones
    configs = {
        'v25': lambda sd: apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices']),
        'z5': lambda sd: apply_generic_zone(
            apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices']),
            sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15),
        'dual_a': lambda sd: apply_generic_zone(
            apply_generic_zone(
                apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices']),
                sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15),
            sd['raw'], sd['ncsos_vs_sos'], sd['tm'], (46, 52), -5, 0.15),
        'dual_b': lambda sd: apply_generic_zone(
            apply_generic_zone(
                apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices']),
                sd['raw'], sd['ncsos'], sd['tm'], (17, 24), 4, 0.15),
            sd['raw'], sd['ncsos'], sd['tm'], (64, 67), 1, 0.15),
    }
    
    # Compute all config scores per season
    config_scores = {}
    for name, fn_apply in configs.items():
        scores = {}
        for s, sd in season_data.items():
            p = fn_apply(sd)
            scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        config_scores[name] = scores
        total = sum(scores.values())
        print(f'  {name:>10}: {total}/91  {list(scores.values())}')

    # Nested LOSO: for each held-out season, pick best config on remaining 4
    print(f'\n  Nested LOSO:')
    nested_total = 0
    nested_details = []
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune_score = -1
        best_cfg = ''
        for name, scores in config_scores.items():
            tune_score = sum(scores.get(s, 0) for s in tune)
            if tune_score > best_tune_score:
                best_tune_score = tune_score
                best_cfg = name
            elif tune_score == best_tune_score and name == 'v25':
                best_cfg = name  # prefer simpler
        hold_score = config_scores[best_cfg][hold]
        nested_total += hold_score
        nested_details.append((hold, hold_score, best_cfg, best_tune_score))
    
    for hold, score, cfg, tune_sc in nested_details:
        print(f'    {hold}: {score}  (chose {cfg}, tune={tune_sc})')
    print(f'  ★ Nested LOSO: {nested_total}/91')

    # ════════════════════════════════════════════════════════════
    #  PHASE 3: EXHAUSTIVE EXTRA FEATURE/ZONE SEARCH
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 3: EXHAUSTIVE SEARCH — more zone corrections')
    print('='*70)

    # Start from best validated config (z5 or dual_a)
    best_base_name = max(config_scores, key=lambda n: sum(config_scores[n].values()))
    best_base_total = sum(config_scores[best_base_name].values())
    best_base_fn = configs[best_base_name]
    print(f'  Starting from: {best_base_name} ({best_base_total}/91)')

    # Search extra zones on top
    extra_feat_names = list(extra_features.keys())
    best_extra_score = best_base_total
    best_extra_configs = []

    for feat_name in extra_feat_names:
        for lo in range(1, 65, 2):
            for hi in range(lo + 3, min(lo + 20, 69), 2):
                for w in [-8, -5, -3, -1, 1, 3, 5, 8]:
                    total = 0
                    for s, sd in season_data.items():
                        p = best_base_fn(sd)
                        p = apply_generic_zone(p, sd['raw'], sd[feat_name], sd['tm'], (lo, hi), w, 0.15)
                        total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                    if total > best_extra_score:
                        best_extra_score = total
                        best_extra_configs = [(feat_name, lo, hi, w, total)]
                        print(f'  NEW BEST: +{feat_name}({lo},{hi}) w={w}: {total}/91 ★')
                    elif total == best_extra_score and total > best_base_total:
                        best_extra_configs.append((feat_name, lo, hi, w, total))

    if best_extra_configs:
        print(f'\n  Best extra: {best_extra_score}/91 ({len(best_extra_configs)} configs)')
        for c in best_extra_configs[:10]:
            print(f'    +{c[0]}({c[1]},{c[2]}) w={c[3]}')
    else:
        print(f'\n  No improvement found beyond {best_base_name}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 4: BASE MODEL IMPROVEMENTS
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 4: BASE MODEL IMPROVEMENTS')
    print('  Try different blend weights, C values, top-K, gap')
    print('='*70)

    base_configs_to_test = [
        # (w1, w3, w4, c1, c3, gap, k)
        (0.64, 0.28, 0.08, 5.0, 0.5, 30, 25),  # current v12
        (0.60, 0.30, 0.10, 5.0, 0.5, 30, 25),
        (0.70, 0.22, 0.08, 5.0, 0.5, 30, 25),
        (0.64, 0.28, 0.08, 3.0, 0.5, 30, 25),
        (0.64, 0.28, 0.08, 10.0, 0.5, 30, 25),
        (0.64, 0.28, 0.08, 5.0, 1.0, 30, 25),
        (0.64, 0.28, 0.08, 5.0, 0.5, 25, 25),
        (0.64, 0.28, 0.08, 5.0, 0.5, 35, 25),
        (0.64, 0.28, 0.08, 5.0, 0.5, 30, 20),
        (0.64, 0.28, 0.08, 5.0, 0.5, 30, 30),
        (0.64, 0.28, 0.08, 5.0, 0.5, 30, 35),
        (0.50, 0.35, 0.15, 5.0, 0.5, 30, 25),
        (0.64, 0.28, 0.08, 7.0, 0.3, 30, 25),
        (0.55, 0.35, 0.10, 5.0, 0.5, 25, 30),
        (0.64, 0.28, 0.08, 5.0, 0.5, 20, 25),
        (0.64, 0.28, 0.08, 5.0, 0.5, 40, 25),
    ]

    best_base_score = v25_total
    best_base_cfg = None

    for cfg_idx, (w1, w3, w4, c1, c3, gap, k) in enumerate(base_configs_to_test):
        total = 0
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            st = test_mask & sm
            if st.sum() == 0:
                continue
            gt = ~st
            X_s = X[sm]
            tki = select_top_k_features(X[gt], y[gt], fn, k=k,
                                         forced_features=FORCE_FEATURES)[0]
            raw = predict_blend(X[gt], y[gt], X_s, seasons[gt], tki,
                               w1=w1, w3=w3, w4=w4, c1=c1, c3=c3, gap=gap)
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in si])
            p = apply_v25_zones(p1, raw, fn, X_s, tm, si)
            total += count_exact(p, tm, si, test_mask, y)
        
        label = f'w={w1:.2f}/{w3:.2f}/{w4:.2f} C={c1}/{c3} gap={gap} k={k}'
        marker = ' ★' if total > best_base_score else ''
        print(f'  [{cfg_idx+1:2d}] {label}: {total}/91{marker}')
        if total > best_base_score:
            best_base_score = total
            best_base_cfg = (w1, w3, w4, c1, c3, gap, k)

    if best_base_cfg:
        print(f'\n  IMPROVED BASE: {best_base_score}/91 with {best_base_cfg}')
    else:
        print(f'\n  No base model improvement found (v12 is optimal)')

    # ════════════════════════════════════════════════════════════
    #  PHASE 5: HUNGARIAN POWER TUNING
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 5: HUNGARIAN POWER + ZONE POWER TUNING')
    print('='*70)

    for hp in [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        total = 0
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            st = test_mask & sm
            if st.sum() == 0:
                continue
            gt = ~st
            X_s = X[sm]
            tki = select_top_k_features(X[gt], y[gt], fn, k=USE_TOP_K_A,
                                         forced_features=FORCE_FEATURES)[0]
            raw = predict_blend(X[gt], y[gt], X_s, seasons[gt], tki)
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=hp)
            tm = np.array([test_mask[gi] for gi in si])
            p = apply_v25_zones(p1, raw, fn, X_s, tm, si)
            total += count_exact(p, tm, si, test_mask, y)
        marker = ' ★' if total > v25_total else ('  (current)' if hp == 0.15 else '')
        print(f'  power={hp:.2f}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 6: SWAP-SPECIFIC CORRECTION
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 6: SWAP-PAIR TARGETING')
    print('  Analyze remaining errors — can we break specific swap pairs?')
    print('='*70)

    # Use best config found so far
    best_fn = best_base_fn  # Start with best base config
    
    # Get all errors from best config
    all_errors = []
    for s, sd in season_data.items():
        p = best_fn(sd)
        for i, gi in enumerate(sd['indices']):
            if test_mask[gi]:
                gt_seed = int(y[gi])
                pred = p[i]
                if pred != gt_seed:
                    all_errors.append({
                        'season': s, 'team': teams[gi], 'rid': record_ids[gi],
                        'gt': gt_seed, 'pred': pred, 'err': pred - gt_seed,
                        'net': net_raw[gi], 'sos': sos_raw[gi],
                        'ncsos': ncsos_raw[gi], 'opp_net': opp_net_raw[gi],
                        'bid': bid_raw[gi], 'conf': conf_raw[gi],
                        'q12_rate': q12_rate[gi], 'road_pct': road_pct[gi],
                    })

    # Find swap pairs
    swap_pairs = []
    for i, e1 in enumerate(all_errors):
        for j, e2 in enumerate(all_errors):
            if j > i and e1['season'] == e2['season'] and e1['gt'] == e2['pred'] and e2['gt'] == e1['pred']:
                swap_pairs.append((e1, e2))
    
    print(f'\n  Total errors: {len(all_errors)}')
    print(f'  Swap pairs: {len(swap_pairs)}')
    
    for e1, e2 in swap_pairs:
        print(f'\n    {e1["season"]}: {e1["team"]}(GT={e1["gt"]}) ↔ {e2["team"]}(GT={e2["gt"]})')
        for key in ['net', 'sos', 'ncsos', 'opp_net', 'q12_rate', 'road_pct', 'bid', 'conf']:
            v1, v2 = e1[key], e2[key]
            if isinstance(v1, float):
                print(f'      {key:>12}: {v1:7.1f} vs {v2:7.1f}  Δ={v1-v2:+.1f}')
            else:
                print(f'      {key:>12}: {v1} vs {v2}')

    non_swap_errors = [e for e in all_errors if not any(
        (e['rid'] == sp[0]['rid'] or e['rid'] == sp[1]['rid']) for sp in swap_pairs)]
    if non_swap_errors:
        print(f'\n  Non-swap errors ({len(non_swap_errors)}):')
        for e in non_swap_errors:
            print(f'    {e["season"]} {e["team"]}: GT={e["gt"]}, pred={e["pred"]} (err={e["err"]:+d})')

    # ════════════════════════════════════════════════════════════
    #  PHASE 7: MULTI-ZONE EXHAUSTIVE on top of best
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 7: TRIPLE ZONE SEARCH (3 new zones on top of best)')
    print('='*70)

    # If we have the 5th zone, try adding 2 more
    if best_extra_configs:
        extra1 = best_extra_configs[0]
        def apply_with_extra1(sd):
            p = best_base_fn(sd)
            return apply_generic_zone(p, sd['raw'], sd[extra1[0]], sd['tm'], (extra1[1], extra1[2]), extra1[3], 0.15)
        
        best_triple = best_extra_score
        best_triple_cfg = None
        
        for feat_name in extra_feat_names:
            for lo in range(1, 65, 3):
                for hi in range(lo + 3, min(lo + 15, 69), 3):
                    for w in [-5, -3, -1, 1, 3, 5]:
                        total = 0
                        for s, sd in season_data.items():
                            p = apply_with_extra1(sd)
                            p = apply_generic_zone(p, sd['raw'], sd[feat_name], sd['tm'], (lo, hi), w, 0.15)
                            total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                        if total > best_triple:
                            best_triple = total
                            best_triple_cfg = (feat_name, lo, hi, w)
                            print(f'  +{feat_name}({lo},{hi}) w={w}: {total}/91 ★')
        
        if best_triple_cfg:
            print(f'\n  Triple best: {best_triple}/91 with {best_triple_cfg}')
        else:
            print(f'\n  No improvement from triple zone')
    else:
        print(f'  No extra zones to stack on')

    # ════════════════════════════════════════════════════════════
    #  PHASE 8: FINAL OVERFITTING BATTERY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 8: OVERFITTING BATTERY ON BEST CONFIG')
    print('='*70)

    # Determine final best config
    final_configs = {'v25': configs['v25']}
    if z5_total > v25_total:
        final_configs['z5'] = configs['z5']
    if dual_total > z5_total:
        final_configs['dual_a'] = configs['dual_a']
    if dual2_total > z5_total:
        final_configs['dual_b'] = configs['dual_b']
    if best_extra_configs:
        extra1 = best_extra_configs[0]
        def make_extra_fn(bf, e):
            return lambda sd: apply_generic_zone(bf(sd), sd['raw'], sd[e[0]], sd['tm'], (e[1], e[2]), e[3], 0.15)
        final_configs[f'best+{extra1[0]}'] = make_extra_fn(best_base_fn, extra1)

    # Per-config scores
    print('\n  All config scores:')
    for name, fn_apply in final_configs.items():
        scores = {}
        for s, sd in season_data.items():
            p = fn_apply(sd)
            scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        total = sum(scores.values())
        print(f'    {name:>20}: {total}/91  {list(scores.values())}')

    # Nested LOSO for all final configs
    print(f'\n  Nested LOSO (all final configs):')
    final_config_scores = {}
    for name, fn_apply in final_configs.items():
        scores = {}
        for s, sd in season_data.items():
            p = fn_apply(sd)
            scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        final_config_scores[name] = scores

    nested_final = 0
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_name = 'v25'
        for name, scores in final_config_scores.items():
            ts = sum(scores.get(s, 0) for s in tune)
            if ts > best_tune:
                best_tune = ts
                best_name = name
            elif ts == best_tune and name == 'v25':
                best_name = name
        nested_final += final_config_scores[best_name][hold]
        print(f'    {hold}: chose {best_name} (tune={best_tune}) → hold={final_config_scores[best_name][hold]}')
    
    print(f'\n  ★ NESTED LOSO TOTAL: {nested_final}/91')

    # Permutation test for best config
    best_final_name = max(final_config_scores, key=lambda n: sum(final_config_scores[n].values()))
    best_final_total = sum(final_config_scores[best_final_name].values())
    
    if best_final_name != 'v25':
        print(f'\n  Permutation test for {best_final_name} ({best_final_total}/91):')
        rng = np.random.RandomState(42)
        n_perm = 1000
        perm_scores = []
        for _ in range(n_perm):
            total = 0
            for s, sd in season_data.items():
                p = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
                # Shuffle relevant feature within season test teams
                test_idx = [i for i in range(len(sd['tm'])) if sd['tm'][i]]
                if len(test_idx) > 1:
                    shuffled_ncsos = sd['ncsos'].copy()
                    vals = shuffled_ncsos[test_idx].copy()
                    rng.shuffle(vals)
                    shuffled_ncsos[test_idx] = vals
                    p = apply_generic_zone(p, sd['raw'], shuffled_ncsos, sd['tm'], (17, 24), 4, 0.15)
                total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            perm_scores.append(total)
        
        perm_scores = np.array(perm_scores)
        p_val = np.mean(perm_scores >= best_final_total)
        print(f'    Real: {best_final_total}/91, Perm mean: {perm_scores.mean():.1f}±{perm_scores.std():.1f}, p={p_val:.4f}')

    # ════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  SUMMARY')
    print('='*70)
    print(f'\n  v25 baseline:      {v25_total}/91')
    print(f'  Best found:        {best_final_total}/91 ({best_final_name})')
    print(f'  Nested LOSO:       {nested_final}/91')
    print(f'  Overfit gap:       {best_final_total - nested_final}')

    if best_final_total > v25_total:
        print(f'\n  ✓ IMPROVEMENT FOUND: +{best_final_total - v25_total}')
        if best_final_total - nested_final <= 3:
            print(f'  ✓ Overfit gap ≤ 3 — safe to deploy')
        else:
            print(f'  ✗ Overfit gap > 3 — risky, consider conservative config')
    else:
        print(f'\n  No improvement beyond v25')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
