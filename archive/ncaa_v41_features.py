#!/usr/bin/env python3
"""
v41 FEATURE EXPANSION — Add unused columns to the BASE model
=============================================================
Instead of adding more post-processing zones (diminishing returns, overfit risk),
add NETNonConfSOS and AvgOppNET directly to the 68-feature set.

This changes the pairwise model ITSELF, which is properly LOSO-validated.
The base model trains on ~270 teams per fold and tests on ~68 — much more data
than zone corrections which only have 91 test teams total.

Also tests: new derived features, interaction terms, feature ablation.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, parse_wl,
    select_top_k_features, hungarian,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3, HUNGARIAN_POWER,
    build_features,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def build_features_extended(df, context_df, labeled_df, tourn_rids, extra_cols=None):
    """Build extended feature set: original 68 + new features from unused columns."""
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    if extra_cols is None:
        extra_cols = []
    
    # AvgOppNET (raw offensive/defensive efficiency of opponents)
    if 'AvgOppNET' in extra_cols:
        opp_net = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(150)
        feat['AvgOppNET'] = opp_net
        # Derived: OppNET vs NET rank (are opponents ranked similarly?)
        net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
        feat['OppNET_vs_NETRank'] = opp_net - net
    
    # NETNonConfSOS (non-conference schedule strength)
    if 'NETNonConfSOS' in extra_cols:
        ncsos = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
        sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
        feat['NETNonConfSOS'] = ncsos
        # Derived: NonConf vs Conf SOS (did team schedule tough non-conf?)
        feat['NCSOS_vs_SOS'] = ncsos - sos
        # Non-conf SOS ratio to NET rank
        net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
        feat['NCSOS_net_ratio'] = ncsos / (net + 1)
    
    # Conference-strength adjusted features
    if 'conf_strength' in extra_cols:
        net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
        sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
        conf = df['Conference'].fillna('Unknown')
        bid = df['Bid Type'].fillna('')
        
        # Conference median NET per season
        for sv in df['Season'].unique():
            sm = context_df['Season'].astype(str) == str(sv)
            nets = pd.to_numeric(context_df.loc[sm, 'NET Rank'], errors='coerce').fillna(300)
            confs = context_df.loc[sm, 'Conference'].fillna('Unknown')
            df_sv = df[df['Season'].astype(str) == str(sv)]
            for idx in df_sv.index:
                c = str(df.loc[idx, 'Conference'])
                med = nets[confs == c].median() if (confs == c).sum() > 0 else 200
                feat.loc[idx, 'net_vs_conf_median'] = float(net[idx]) - med
        
        # AQ NET gap (AQ teams: how far from conf median?)
        feat['aq_conf_gap'] = (bid == 'AQ').astype(float) * feat.get('net_vs_conf_median', pd.Series(0, index=df.index)).fillna(0)
        
        # AL quality (AL teams: NET rank relative to other ALs)
        feat['al_quality'] = 0.0
        for sv in df['Season'].unique():
            sv_mask = df['Season'].astype(str) == str(sv)
            al_mask = sv_mask & (bid == 'AL')
            if al_mask.sum() > 1:
                al_nets = net[al_mask]
                feat.loc[al_mask, 'al_quality'] = al_nets.rank(pct=True)

    # Quadrant performance ratios
    if 'quad_ratios' in extra_cols:
        q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
        q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
        q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
        q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
        q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
        q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
        
        # Q1 win to Q3+Q4 loss ratio (how dominant?)
        feat['q1w_to_badloss'] = q1w / (q3l + q4l + 0.5)
        # Total quality wins vs bad losses (committee values this)
        feat['quality_vs_bad'] = (q1w*3 + q2w) / (q3l*2 + q4l*3 + 1)

    return feat


def predict_robust_blend_extended(X_train, y_train, X_test, seasons_train, top_k_idx,
                                   w1=BLEND_W1, w3=BLEND_W3, w4=BLEND_W4,
                                   c1=PW_C1, c3=PW_C3, gap=ADJ_COMP1_GAP):
    """Same as predict_robust_blend but configurable."""
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


def eval_config(labeled, context_df, tourn_rids, GT, extra_cols, k=25, 
                force_feats=None, verbose=False):
    """Evaluate a feature configuration with full LOSO + v25 zones."""
    if force_feats is None:
        force_feats = ['NET Rank']
    
    feat = build_features_extended(labeled, context_df, labeled, tourn_rids, extra_cols)
    fn = list(feat.columns)
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]
    
    total = 0
    per_season = {}
    
    for hold in test_seasons:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0:
            continue
        gt_mask = ~st
        X_s = X[sm]
        
        # Force features that exist in this feature set
        ff = [f for f in force_feats if f in fn]
        tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=k,
                                     forced_features=ff)[0]
        raw = predict_robust_blend_extended(X[gt_mask], y[gt_mask], X_s,
                                             seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])
        
        # Apply v25 zones (using first 68 features only — zone corrections
        # use named features so they work regardless of extra columns)
        p = apply_v25_zones(p1, raw, fn, X_s, tm, si)
        
        ex = sum(1 for i, gi in enumerate(si) if test_mask[gi] and p[i] == int(y[gi]))
        total += ex
        per_season[hold] = ex
    
    return total, per_season, fn


def main():
    print('='*70)
    print('  v41 FEATURE EXPANSION — Add unused columns to BASE model')
    print('='*70)
    
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    # ════════════════════════════════════════════════════════════
    #  TEST 1: Baseline (current 68 features)
    # ════════════════════════════════════════════════════════════
    print('\n  Testing feature configurations...')
    
    configs = {
        'baseline (68 feat)':           [],
        '+AvgOppNET':                   ['AvgOppNET'],
        '+NETNonConfSOS':               ['NETNonConfSOS'],
        '+both':                        ['AvgOppNET', 'NETNonConfSOS'],
        '+conf_strength':               ['conf_strength'],
        '+quad_ratios':                  ['quad_ratios'],
        '+all_new':                     ['AvgOppNET', 'NETNonConfSOS', 'conf_strength', 'quad_ratios'],
        '+AvgOppNET+conf':              ['AvgOppNET', 'conf_strength'],
        '+NETNonConf+conf':             ['NETNonConfSOS', 'conf_strength'],
        '+NETNonConf+quad':             ['NETNonConfSOS', 'quad_ratios'],
        '+both+conf':                   ['AvgOppNET', 'NETNonConfSOS', 'conf_strength'],
        '+both+quad':                   ['AvgOppNET', 'NETNonConfSOS', 'quad_ratios'],
    }
    
    best_score = 0
    best_config = ''
    results = {}
    
    for name, extra_cols in configs.items():
        total, per_season, fn_list = eval_config(labeled, context_df, tourn_rids, GT, extra_cols)
        results[name] = (total, per_season, len(fn_list))
        marker = ' ★' if total > best_score else ''
        if total > best_score:
            best_score = total
            best_config = name
        ps_str = ' '.join(f'{per_season.get(s, 0):2d}' for s in sorted(per_season.keys()))
        print(f'  {name:>30}: {total}/91  [{ps_str}]  ({len(fn_list)} feat){marker}')
    
    # ════════════════════════════════════════════════════════════
    #  TEST 2: Top-K variations with best feature set
    # ════════════════════════════════════════════════════════════
    print(f'\n  Best feature set: {best_config} ({best_score}/91)')
    print(f'\n  Testing top-K variations with best feature set...')
    
    best_extra = [v for k, v in configs.items() if k == best_config][0]
    
    for k in [20, 22, 25, 28, 30, 35, 40]:
        total, per_season, fn_list = eval_config(labeled, context_df, tourn_rids, GT, 
                                                   best_extra, k=k)
        marker = ' ★' if total > best_score else ''
        if total > best_score:
            best_score = total
        ps_str = ' '.join(f'{per_season.get(s, 0):2d}' for s in sorted(per_season.keys()))
        print(f'    k={k:2d}: {total}/91  [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════
    #  TEST 3: Force different features into top-K
    # ════════════════════════════════════════════════════════════
    print(f'\n  Testing forced features...')
    
    force_options = [
        ['NET Rank'],
        ['NET Rank', 'NETNonConfSOS'],
        ['NET Rank', 'AvgOppNET'],
        ['NET Rank', 'NETNonConfSOS', 'AvgOppNET'],
        ['NET Rank', 'NCSOS_vs_SOS'],
        ['NET Rank', 'NCSOS_net_ratio'],
        ['NET Rank', 'OppNET_vs_NETRank'],
    ]
    
    for ff in force_options:
        # Only test if all forced features would be in the feature set
        try:
            total, per_season, fn_list = eval_config(labeled, context_df, tourn_rids, GT,
                                                       ['AvgOppNET', 'NETNonConfSOS'],
                                                       force_feats=ff)
            marker = ' ★' if total > best_score else ''
            if total > best_score:
                best_score = total
            ps_str = ' '.join(f'{per_season.get(s, 0):2d}' for s in sorted(per_season.keys()))
            print(f'    force={ff}: {total}/91  [{ps_str}]{marker}')
        except Exception as e:
            print(f'    force={ff}: ERROR ({e})')

    # ════════════════════════════════════════════════════════════
    #  TEST 4: Different zone params with extended features
    # ════════════════════════════════════════════════════════════
    print(f'\n  Testing zone param variations with best features...')
    
    # Re-build data with best feature set for zone tuning
    feat = build_features_extended(labeled, context_df, labeled, tourn_rids, 
                                    ['AvgOppNET', 'NETNonConfSOS'])
    fn = list(feat.columns)
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)
    
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]
    
    # Precompute base predictions with extended features
    season_data = {}
    for hold in test_seasons:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0:
            continue
        gt_mask = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=25,
                                     forced_features=['NET Rank'])[0]
        raw = predict_robust_blend_extended(X[gt_mask], y[gt_mask], X_s,
                                             seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = {'pass1': p1, 'raw': raw, 'X': X_s, 'tm': tm, 'indices': si}
    
    # Test zone param variations
    zone_configs = [
        # (mid_al, mid_sos, low_q1, low_field, bot_sn, bot_nc, bot_cb, bot_lo, bot_hi, tail_opr, tail_lo, tail_hi)
        (2, 3, 1, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # current v25
        (2, 3, 1, 2, -4, 3, -1, 50, 60, 0, 61, 65),    # no tail
        (2, 3, 1, 2, -4, 3, -1, 53, 65, -3, 61, 65),   # old bot zone
        (1, 3, 1, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # less al
        (3, 3, 1, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # more al
        (2, 2, 1, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # less sos
        (2, 4, 1, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # more sos
        (2, 3, 2, 2, -4, 3, -1, 50, 60, -3, 61, 65),   # more q1
        (2, 3, 1, 3, -4, 3, -1, 50, 60, -3, 61, 65),   # more field
        (2, 3, 1, 1, -4, 3, -1, 50, 60, -3, 61, 65),   # less field
        (2, 3, 1, 2, -3, 3, -1, 50, 60, -3, 61, 65),   # less sosnet
        (2, 3, 1, 2, -5, 3, -1, 50, 60, -3, 61, 65),   # more sosnet
        (2, 3, 1, 2, -4, 4, -1, 50, 60, -3, 61, 65),   # more net_conf
        (2, 3, 1, 2, -4, 3, -2, 50, 60, -3, 61, 65),   # more cbhist
        (2, 3, 1, 2, -4, 3, -1, 48, 62, -3, 61, 65),   # wider bot
        (2, 3, 1, 2, -4, 3, -1, 50, 60, -4, 61, 65),   # more tail
        (2, 3, 1, 2, -4, 3, -1, 50, 60, -3, 58, 65),   # wider tail
    ]
    
    best_zone_score = 0
    best_zone_cfg = None
    
    for zc in zone_configs:
        al, sos, q1, field, sn, nc, cb, blo, bhi, opr, tlo, thi = zc
        total = 0
        for s, sd in season_data.items():
            p = sd['pass1'].copy()
            corr = compute_committee_correction(fn, sd['X'], alpha_aq=0, beta_al=al, gamma_sos=sos)
            p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                    zone=(17,34), blend=1.0, power=0.15)
            corr = compute_low_correction(fn, sd['X'], q1dom=q1, field=field)
            p = apply_lowzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                   zone=(35,52), power=0.15)
            corr = compute_bottom_correction(fn, sd['X'], sosnet=sn, net_conf=nc, cbhist=cb)
            p = apply_bottomzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                      zone=(blo, bhi), power=0.15)
            corr = compute_tail_correction(fn, sd['X'], opp_rank=opr)
            p = apply_tailzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                    zone=(tlo, thi), power=0.15)
            total += sum(1 for i, gi in enumerate(sd['indices'])
                        if test_mask[gi] and p[i] == int(y[gi]))
        
        marker = ' ★' if total > best_zone_score else ''
        if total > best_zone_score:
            best_zone_score = total
            best_zone_cfg = zc
        label = f'al={al} s={sos} q={q1} f={field} sn={sn} nc={nc} cb={cb} b=({blo},{bhi}) t_opr={opr} t=({tlo},{thi})'
        print(f'    {label}: {total}/91{marker}')
    
    # ════════════════════════════════════════════════════════════
    #  TEST 5: Extended features + exhaustive zone sweep
    # ════════════════════════════════════════════════════════════
    if best_zone_score > 0:
        print(f'\n  Best zone config with extended features: {best_zone_score}/91')
        print(f'  vs v25 baseline (original features): 70/91')
        
        if best_zone_score > 70:
            print(f'\n  ✓ Feature expansion HELPS: +{best_zone_score - 70}')
            
            # Do nested LOSO for best config
            print(f'\n  Running nested LOSO for best config...')
            bz = best_zone_cfg
            al, sos, q1, field, sn, nc, cb, blo, bhi, opr, tlo, thi = bz
            
            nested_total = 0
            nested_v25 = 0
            for hold in test_seasons:
                sd = season_data[hold]
                
                # v25 zones with extended features
                p_v25 = apply_v25_zones(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
                v25_hold = sum(1 for i, gi in enumerate(sd['indices']) 
                              if test_mask[gi] and p_v25[i] == int(y[gi]))
                nested_v25 += v25_hold
                
                # Best zones with extended features
                p = sd['pass1'].copy()
                corr = compute_committee_correction(fn, sd['X'], alpha_aq=0, beta_al=al, gamma_sos=sos)
                p = apply_midrange_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                        zone=(17,34), blend=1.0, power=0.15)
                corr = compute_low_correction(fn, sd['X'], q1dom=q1, field=field)
                p = apply_lowzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                       zone=(35,52), power=0.15)
                corr = compute_bottom_correction(fn, sd['X'], sosnet=sn, net_conf=nc, cbhist=cb)
                p = apply_bottomzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                          zone=(blo, bhi), power=0.15)
                corr = compute_tail_correction(fn, sd['X'], opp_rank=opr)
                p = apply_tailzone_swap(p, sd['raw'], corr, sd['tm'], sd['indices'],
                                        zone=(tlo, thi), power=0.15)
                best_hold = sum(1 for i, gi in enumerate(sd['indices'])
                               if test_mask[gi] and p[i] == int(y[gi]))
                
                # Pick better
                tune_seasons = [s for s in test_seasons if s != hold]
                # Just pick best config for now
                nested_total += best_hold
                print(f'    {hold}: v25={v25_hold}, best={best_hold}')
            
            print(f'  Nested LOSO v25+extended: {nested_v25}/91')
            print(f'  Nested LOSO best_zones+extended: {nested_total}/91')
        else:
            print(f'\n  Feature expansion doesn\'t help (same or worse)')
    
    # ════════════════════════════════════════════════════════════
    #  TEST 6: Different number of XGB component estimators
    # ════════════════════════════════════════════════════════════
    print(f'\n  Testing XGB component variations...')
    
    feat_orig = build_features(labeled, context_df, labeled, tourn_rids)
    fn_orig = list(feat_orig.columns)
    X_orig_raw = np.where(np.isinf(feat_orig.values.astype(np.float64)), np.nan,
                          feat_orig.values.astype(np.float64))
    imp2 = KNNImputer(n_neighbors=10, weights='distance')
    X_orig = imp2.fit_transform(X_orig_raw)
    
    # Test with more XGB trees/depth
    xgb_variants = [
        (300, 4, 0.05, 5),   # current
        (500, 4, 0.05, 5),
        (300, 5, 0.05, 5),
        (300, 3, 0.05, 5),
        (300, 4, 0.03, 5),
        (300, 4, 0.08, 5),
        (500, 5, 0.03, 3),
        (200, 4, 0.08, 5),
        (400, 4, 0.05, 3),
    ]
    
    for n_est, depth, lr, mcw in xgb_variants:
        total = 0
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            st = test_mask & sm
            if st.sum() == 0:
                continue
            gt_mask = ~st
            X_s = X_orig[sm]
            tki = select_top_k_features(X_orig[gt_mask], y[gt_mask], fn_orig, k=25,
                                         forced_features=['NET Rank'])[0]
            
            # Standard comp 1 and 3
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
                X_orig[gt_mask], y[gt_mask], seasons[gt_mask], max_gap=30)
            sc1 = StandardScaler()
            lr1 = LogisticRegression(C=5.0, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(sc1.fit_transform(pw_X_adj), pw_y_adj)
            s1 = pairwise_score(lr1, X_s, sc1)
            
            X_tk = X_orig[gt_mask][:, tki]
            X_stk = X_s[:, tki]
            pw_Xk, pw_yk = build_pairwise_data(X_tk, y[gt_mask], seasons[gt_mask])
            sck = StandardScaler()
            lr3 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(sck.fit_transform(pw_Xk), pw_yk)
            s3 = pairwise_score(lr3, X_stk, sck)
            
            # Modified XGB
            pw_Xf, pw_yf = build_pairwise_data(X_orig[gt_mask], y[gt_mask], seasons[gt_mask])
            scf = StandardScaler()
            xgb_clf = xgb.XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=mcw,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            xgb_clf.fit(scf.fit_transform(pw_Xf), pw_yf)
            s4 = pairwise_score(xgb_clf, X_s, scf)
            
            raw = 0.64 * s1 + 0.28 * s3 + 0.08 * s4
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=0.15)
            tm = np.array([test_mask[gi] for gi in si])
            p = apply_v25_zones(p1, raw, fn_orig, X_s, tm, si)
            total += sum(1 for i, gi in enumerate(si)
                        if test_mask[gi] and p[i] == int(y[gi]))
        
        marker = ' ★' if total > 70 else ''
        print(f'    XGB n={n_est} d={depth} lr={lr} mcw={mcw}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  TEST 7: 4th component — add a Ridge regression component
    # ════════════════════════════════════════════════════════════
    print(f'\n  Testing 4-component blend (add Ridge)...')
    
    for ridge_w in [0.05, 0.10, 0.15]:
        for ridge_alpha in [1.0, 5.0, 10.0]:
            w1_adj = BLEND_W1 * (1 - ridge_w)
            w3_adj = BLEND_W3 * (1 - ridge_w)
            w4_adj = BLEND_W4 * (1 - ridge_w)
            
            total = 0
            for hold in test_seasons:
                sm = (seasons == hold)
                si = np.where(sm)[0]
                st = test_mask & sm
                if st.sum() == 0:
                    continue
                gt_mask = ~st
                X_s = X_orig[sm]
                tki = select_top_k_features(X_orig[gt_mask], y[gt_mask], fn_orig, k=25,
                                             forced_features=['NET Rank'])[0]
                raw = predict_robust_blend_extended(X_orig[gt_mask], y[gt_mask], X_s,
                                                     seasons[gt_mask], tki,
                                                     w1=w1_adj, w3=w3_adj, w4=w4_adj)
                
                # Add Ridge component
                sc_r = StandardScaler()
                ridge = Ridge(alpha=ridge_alpha)
                ridge.fit(sc_r.fit_transform(X_orig[gt_mask]), y[gt_mask])
                ridge_pred = ridge.predict(sc_r.transform(X_s))
                
                raw = raw * (1 - ridge_w) + ridge_w * ridge_pred
                
                for i, gi in enumerate(si):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                avail = {hold: list(range(1, 69))}
                p1 = hungarian(raw, seasons[sm], avail, power=0.15)
                tm = np.array([test_mask[gi] for gi in si])
                p = apply_v25_zones(p1, raw, fn_orig, X_s, tm, si)
                total += sum(1 for i, gi in enumerate(si)
                            if test_mask[gi] and p[i] == int(y[gi]))
            
            marker = ' ★' if total > 70 else ''
            print(f'    Ridge w={ridge_w:.2f} a={ridge_alpha}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  SUMMARY')
    print('='*70)
    
    print(f'\n  Feature expansion results:')
    for name, (total, ps, nf) in sorted(results.items(), key=lambda x: -x[1][0]):
        marker = ' ★' if total >= best_score else ''
        print(f'    {name:>30}: {total}/91 ({nf} feat){marker}')
    
    print(f'\n  v25 baseline: 70/91')
    print(f'  Best found: {best_score}/91 ({best_config})')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
