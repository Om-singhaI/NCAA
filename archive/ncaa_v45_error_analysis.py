#!/usr/bin/env python3
"""
Comprehensive error analysis for v44 config.
Identify the biggest SE contributors and look for patterns.
"""

import os, sys, time, warnings
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

warnings.filterwarnings('ignore')
np.random.seed(42)

# v44 config
MID_PARAMS = (0, 0, 3)     # aq, al, sos
BOT_ZONE = (50, 60)
BOT_PARAMS = (-4, 3, -1)   # sosnet, net_conf, cbhist
TAIL_ZONE = (60, 61)
TAIL_PARAMS = (-3,)         # opp_rank


def run_pipeline(season_data, test_mask, y, folds, fn):
    """Run v44 pipeline, return per-team predictions."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    raw_scores_all = np.zeros(n)
    pass1_all = np.zeros(n, dtype=int)
    
    for hold in folds:
        if hold not in season_data:
            continue
        si, tm, X_s, raw, p1 = season_data[hold]
        assigned = p1.copy()
        
        # Mid-range
        aq, al, sos = MID_PARAMS
        corr = compute_committee_correction(fn, X_s, alpha_aq=aq, beta_al=al, gamma_sos=sos)
        assigned = apply_midrange_swap(assigned, raw, corr, tm, si,
                                       zone=(17, 34), blend=1.0, power=0.15)
        # Bot-zone
        sn, nc, cb = BOT_PARAMS
        corr = compute_bottom_correction(fn, X_s, sosnet=sn, net_conf=nc, cbhist=cb)
        assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si,
                                          zone=BOT_ZONE, power=0.15)
        # Tail-zone
        opp, = TAIL_PARAMS
        corr = compute_tail_correction(fn, X_s, opp_rank=opp)
        assigned = apply_tailzone_swap(assigned, raw, corr, tm, si,
                                        zone=TAIL_ZONE, power=0.15)
        
        for i, gi in enumerate(si):
            preds[gi] = assigned[i]
            raw_scores_all[gi] = raw[i]
            pass1_all[gi] = p1[i]
    
    return preds, raw_scores_all, pass1_all


def main():
    t0 = time.time()
    print('='*70)
    print('  COMPREHENSIVE ERROR ANALYSIS — v44 CONFIG')
    print('='*70)

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
    X = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # Pre-compute season data
    season_data = {}
    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        if st.sum() == 0: continue
        gt_mask = ~st
        X_s = X[sm]
        tki = select_top_k_features(X[gt_mask], y[gt_mask], fn, k=USE_TOP_K_A,
                                     forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X[gt_mask], y[gt_mask], X_s, seasons[gt_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=0.15)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = (si, tm, X_s, raw, p1.copy())

    preds, raw_scores, pass1_preds = run_pipeline(season_data, test_mask, y, folds, fn)

    # ── Error table ──
    print('\n  ══ ALL ERRORS (sorted by SE, descending) ══')
    print(f'  {"Rank":>4} {"RecordID":<30} {"GT":>4} {"Pred":>4} {"Err":>5} {"SE":>5} '
          f'{"Raw":>7} {"P1":>4} {"Zone":>10}')
    print(f'  {"─"*4} {"─"*30} {"─"*4} {"─"*4} {"─"*5} {"─"*5} {"─"*7} {"─"*4} {"─"*10}')
    
    errors = []
    for i in np.where(test_mask)[0]:
        gt = int(y[i])
        pred = int(preds[i])
        err = pred - gt
        se = err**2
        raw = raw_scores[i]
        p1 = int(pass1_preds[i])
        
        # Determine zone
        if 17 <= pred <= 34 or 17 <= gt <= 34:
            zone = 'mid'
        elif 50 <= pred <= 60 or 50 <= gt <= 60:
            zone = 'bot'
        elif 60 <= pred <= 61 or 60 <= gt <= 61:
            zone = 'tail'
        elif pred <= 16 or gt <= 16:
            zone = 'top'
        elif 35 <= pred <= 49 or 35 <= gt <= 49:
            zone = 'uncorrected'
        else:
            zone = 'far-tail'
        
        errors.append((record_ids[i], gt, pred, err, se, raw, p1, zone, seasons[i]))
    
    errors.sort(key=lambda x: -x[4])
    total_se = sum(e[4] for e in errors)
    cumulative = 0
    
    for rank, (rid, gt, pred, err, se, raw, p1, zone, season) in enumerate(errors, 1):
        if se == 0 and rank > 30:
            continue
        cumulative += se
        pct = cumulative / total_se * 100
        if se > 0:
            print(f'  {rank:4d} {rid:<30} {gt:4d} {pred:4d} {err:+5d} {se:5d} '
                  f'{raw:7.1f} {p1:4d} {zone:<10} [{pct:5.1f}%]')

    n_exact = sum(1 for e in errors if e[4] == 0)
    n_errors = sum(1 for e in errors if e[4] > 0)
    print(f'\n  Total: {n_exact} exact, {n_errors} errors, SE={total_se}')

    # ── Error by zone ──
    print(f'\n  ══ SE BY ZONE ══')
    zone_se = {}
    zone_count = {}
    for rid, gt, pred, err, se, raw, p1, zone, season in errors:
        zone_se[zone] = zone_se.get(zone, 0) + se
        zone_count[zone] = zone_count.get(zone, 0) + 1
    for zone in sorted(zone_se, key=lambda z: -zone_se[z]):
        print(f'  {zone:<15} SE={zone_se[zone]:5d} ({zone_se[zone]/total_se*100:5.1f}%) '
              f'teams={zone_count[zone]}')

    # ── Error by season ──
    print(f'\n  ══ SE BY SEASON ══')
    season_se = {}
    season_count = {}
    for rid, gt, pred, err, se, raw, p1, zone, season in errors:
        season_se[season] = season_se.get(season, 0) + se
        season_count[season] = season_count.get(season, 0) + 1
    for season in sorted(season_se):
        print(f'  {season:<12} SE={season_se[season]:5d} teams={season_count[season]:2d}')

    # ── Biggest errors deep dive ──
    print(f'\n  ══ DEEP DIVE: TOP ERRORS ══')
    for rank, (rid, gt, pred, err, se, raw, p1, zone, season) in enumerate(errors[:10], 1):
        if se == 0:
            break
        print(f'\n  {rank}. {rid} (Season {season})')
        print(f'     GT={gt}, Pred={pred}, Error={err:+d}, SE={se}')
        print(f'     Raw score={raw:.2f}, Pass1 (Hungarian)={p1}')
        print(f'     Zone: {zone}')
        
        # Find this team's features
        idx = np.where(record_ids == rid)[0][0]
        sm = (seasons == season)
        si = np.where(sm)[0]
        local_idx = np.where(si == idx)[0][0]
        X_team = X[idx]
        
        # Show key features
        key_feats = ['NET Rank', 'Adjusted Q1', 'Adjusted Losses', 'SOS Rank',
                     'Opp. Rank', 'Conf. Bid History', 'SOS NET Rank']
        for f in key_feats:
            if f in fn:
                fi = fn.index(f)
                print(f'     {f}: {X_team[fi]:.1f}')

    # ── Analysis: What if we could fix the biggest errors? ──
    print(f'\n\n  ══ IMPACT ANALYSIS ══')
    print(f'  Current: SE={total_se}, RMSE451={np.sqrt(total_se/451):.4f}')
    for fix_count in [1, 2, 3, 5, 10]:
        fixable_se = sum(e[4] for e in errors[:fix_count])
        new_se = total_se - fixable_se
        print(f'  Fix top {fix_count:2d} errors: SE={new_se:4d}, '
              f'RMSE451={np.sqrt(new_se/451):.4f} (save {fixable_se} SE)')

    # ── Uncorrected zone analysis (seeds 35-49) ──
    print(f'\n\n  ══ UNCORRECTED ZONE ANALYSIS (seeds 35-49) ══')
    print(f'  These seeds have NO zone correction applied.')
    uncorr_errors = [(rid, gt, pred, err, se, raw, p1, zone, s) 
                     for rid, gt, pred, err, se, raw, p1, zone, s in errors
                     if zone == 'uncorrected' and se > 0]
    for rid, gt, pred, err, se, raw, p1, zone, season in uncorr_errors:
        idx = np.where(record_ids == rid)[0][0]
        X_team = X[idx]
        feats_str = []
        for f in ['NET Rank', 'SOS Rank', 'SOS NET Rank', 'Adjusted Q1', 'Adjusted Losses']:
            if f in fn:
                fi = fn.index(f)
                feats_str.append(f'{f}={X_team[fi]:.0f}')
        print(f'  {rid:<30} GT={gt:3d} Pred={pred:3d} Err={err:+3d} SE={se:3d} '
              f'P1={p1:3d} ({", ".join(feats_str)})')

    # ── Far-tail analysis ──
    print(f'\n\n  ══ FAR-TAIL ANALYSIS (seeds > 61) ══')
    far_errors = [(rid, gt, pred, err, se, raw, p1, zone, s) 
                  for rid, gt, pred, err, se, raw, p1, zone, s in errors
                  if (gt > 61 or pred > 61) and se > 0]
    for rid, gt, pred, err, se, raw, p1, zone, season in far_errors:
        print(f'  {rid:<30} GT={gt:3d} Pred={pred:3d} Err={err:+3d} SE={se:3d} P1={p1:3d}')

    # ── Raw score vs GT analysis: are raw scores better than Hungarian for any teams? ──
    print(f'\n\n  ══ RAW SCORE ANALYSIS ══')
    print(f'  For errors > 1, check if raw score was closer to GT:')
    for rid, gt, pred, err, se, raw, p1, zone, season in errors:
        if se <= 1: continue
        raw_err = raw - gt
        p1_err = p1 - gt
        pred_err = pred - gt
        closer = abs(raw_err) < abs(pred_err)
        if closer:
            print(f'  {rid:<28} GT={gt:3d} Raw={raw:5.1f}(err={raw_err:+5.1f}) '
                  f'P1={p1:3d}(err={p1_err:+3d}) Final={pred:3d}(err={pred_err:+3d}) '
                  f'← RAW BETTER')

    print(f'\n  Time: {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
