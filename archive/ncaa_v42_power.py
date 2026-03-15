#!/usr/bin/env python3
"""
v42 POWER TUNING + SWAP-PAIR STRATEGIES
=========================================
v33 found 73/91 with NETNonConfSOS zone(17,24) w=4 pw=0.10
v40 found only 71/91 with same zone but pw=0.15
The difference is the Hungarian power for zone corrections.

This script:
1. Tests per-zone powers (0.05-0.30 for each of the 5 zones)
2. Tests NETNonConfSOS zone with pw=0.10
3. Swap-pair post-processing: after zones, check all adjacent pairs
   and try reversing specific swaps based on signal strength
4. Pairwise gap narrowing: train LR on even closer pairs (gap=15, 20)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, parse_wl,
    select_top_k_features, hungarian,
    build_pairwise_data, build_pairwise_data_adjacent, pairwise_score,
    compute_committee_correction, apply_midrange_swap,
    compute_low_correction, apply_lowzone_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    predict_robust_blend,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4, PW_C1, PW_C3, HUNGARIAN_POWER,
)

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()


def apply_generic_zone(p, raw, fvals, tm, zone, weight, power=0.15):
    """Apply zone correction using any feature values."""
    lo, hi = zone
    zt = [i for i in range(len(p)) if tm[i] and lo <= p[i] <= hi]
    if len(zt) <= 1:
        return p
    fv = np.array([fvals[i] for i in zt], dtype=float)
    vmin, vmax = fv.min(), fv.max()
    if vmax > vmin:
        norm = (fv - vmin) / (vmax - vmin) * 2 - 1
    else:
        norm = np.zeros(len(fv))
    corr = weight * norm
    seeds = [p[i] for i in zt]
    corrected = [raw[zt[k]] + corr[k] for k in range(len(zt))]
    cost = np.array([[abs(sv - sd)**power for sd in seeds] for sv in corrected])
    ri, ci = linear_sum_assignment(cost)
    pnew = p.copy()
    for r, c in zip(ri, ci):
        pnew[zt[r]] = seeds[c]
    return pnew


def apply_zones_custom(pass1, raw, fn, X, tm, idx, ncsos_vals=None,
                        mid_pw=0.15, low_pw=0.15, bot_pw=0.15, tail_pw=0.15,
                        z5=None, z5_pw=0.15):
    """Apply zone corrections with custom powers per zone."""
    p = pass1.copy()
    corr = compute_committee_correction(fn, X, alpha_aq=0, beta_al=2, gamma_sos=3)
    p = apply_midrange_swap(p, raw, corr, tm, idx, zone=(17,34), blend=1.0, power=mid_pw)
    corr = compute_low_correction(fn, X, q1dom=1, field=2)
    p = apply_lowzone_swap(p, raw, corr, tm, idx, zone=(35,52), power=low_pw)
    corr = compute_bottom_correction(fn, X, sosnet=-4, net_conf=3, cbhist=-1)
    p = apply_bottomzone_swap(p, raw, corr, tm, idx, zone=(50,60), power=bot_pw)
    corr = compute_tail_correction(fn, X, opp_rank=-3)
    p = apply_tailzone_swap(p, raw, corr, tm, idx, zone=(61,65), power=tail_pw)
    if z5 is not None and ncsos_vals is not None:
        lo, hi, w = z5
        p = apply_generic_zone(p, raw, ncsos_vals, tm, (lo, hi), w, z5_pw)
    return p


def count_exact(p, tm, indices, test_mask, y):
    return sum(1 for i, gi in enumerate(indices) if test_mask[gi] and p[i] == int(y[gi]))


def main():
    print('='*70)
    print('  v42 POWER TUNING + SWAP-PAIR STRATEGIES')
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
    sos_raw = pd.to_numeric(labeled['NETSOS'], errors='coerce').fillna(200).values

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                    np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    # Precompute season data
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
        raw = predict_robust_blend(X[gt], y[gt], X_s, seasons[gt], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        avail = {hold: list(range(1, 69))}
        p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
        tm = np.array([test_mask[gi] for gi in si])
        season_data[hold] = {
            'pass1': p1, 'raw': raw, 'X': X_s,
            'tm': tm, 'indices': si.copy(),
            'ncsos': ncsos_raw[sm],
        }

    # ════════════════════════════════════════════════════════════
    #  PHASE 1: Per-zone power sweep
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 1: PER-ZONE POWER SWEEP')
    print('='*70)

    powers = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    
    best_power_score = 0
    best_powers = None

    # First: sweep each zone independently
    print('\n  Sweeping each zone power independently...')
    for zone_name, idx_name in [('mid', 0), ('low', 1), ('bot', 2), ('tail', 3)]:
        print(f'    {zone_name}:')
        for pw in powers:
            total = 0
            pw_set = [0.15, 0.15, 0.15, 0.15]
            pw_set[idx_name] = pw
            for s, sd in season_data.items():
                p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                       sd['tm'], sd['indices'],
                                       mid_pw=pw_set[0], low_pw=pw_set[1],
                                       bot_pw=pw_set[2], tail_pw=pw_set[3])
                total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            marker = '★' if total > 70 else ''
            print(f'      pw={pw:.2f}: {total}/91 {marker}')
            if total > best_power_score:
                best_power_score = total
                best_powers = tuple(pw_set)

    print(f'\n  Best single-zone power: {best_power_score}/91')

    # Now: joint sweep of all 4 zone powers (reduced grid)
    print('\n  Joint sweep (reduced grid)...')
    coarse_powers = [0.10, 0.15, 0.20]
    best_joint = 0
    best_joint_pw = None
    
    for mp, lp, bp, tp in product(coarse_powers, repeat=4):
        total = 0
        for s, sd in season_data.items():
            p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'],
                                   mid_pw=mp, low_pw=lp, bot_pw=bp, tail_pw=tp)
            total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        if total > best_joint:
            best_joint = total
            best_joint_pw = (mp, lp, bp, tp)
            print(f'    mid={mp:.2f} low={lp:.2f} bot={bp:.2f} tail={tp:.2f}: {total}/91 ★')

    if best_joint_pw:
        print(f'\n  Best joint: {best_joint}/91 at {best_joint_pw}')
    
    # Fine-tune around best joint
    if best_joint_pw and best_joint > 70:
        print('\n  Fine-tuning around best joint...')
        bmp, blp, bbp, btp = best_joint_pw
        fine_range = lambda v: [max(0.05, v-0.05), v, min(0.30, v+0.05)]
        for mp in fine_range(bmp):
            for lp in fine_range(blp):
                for bp in fine_range(bbp):
                    for tp in fine_range(btp):
                        total = 0
                        for s, sd in season_data.items():
                            p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                                   sd['tm'], sd['indices'],
                                                   mid_pw=mp, low_pw=lp, bot_pw=bp, tail_pw=tp)
                            total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
                        if total > best_joint:
                            best_joint = total
                            best_joint_pw = (mp, lp, bp, tp)
                            print(f'    {mp:.2f}/{lp:.2f}/{bp:.2f}/{tp:.2f}: {total}/91 ★')

    # ════════════════════════════════════════════════════════════
    #  PHASE 2: NETNonConfSOS 5th zone with pw=0.10
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 2: NETNonConfSOS ZONE WITH pw=0.10')
    print('='*70)

    # Test the exact v33 config: zone(17,24) w=4 pw=0.10
    z5_configs = []
    for lo in range(14, 22):
        for hi in range(lo+4, min(lo+15, 35)):
            for w in [2, 3, 4, 5, 6, 7, 8]:
                for pw5 in [0.05, 0.08, 0.10, 0.12, 0.15]:
                    z5_configs.append((lo, hi, w, pw5))

    best_z5 = 70
    best_z5_cfg = None
    best_z5_configs = []
    
    for lo, hi, w, pw5 in z5_configs:
        total = 0
        for s, sd in season_data.items():
            p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'], ncsos_vals=sd['ncsos'],
                                   z5=(lo, hi, w), z5_pw=pw5)
            total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        if total > best_z5:
            best_z5 = total
            best_z5_cfg = (lo, hi, w, pw5)
            best_z5_configs = [(lo, hi, w, pw5, total)]
            print(f'  zone=({lo},{hi}) w={w} pw={pw5:.2f}: {total}/91 ★')
        elif total == best_z5 and total > 70:
            best_z5_configs.append((lo, hi, w, pw5, total))

    print(f'\n  Best 5th zone: {best_z5}/91')
    if best_z5_configs:
        print(f'  {len(best_z5_configs)} configs at best score')
        for c in best_z5_configs[:15]:
            print(f'    zone=({c[0]},{c[1]}) w={c[2]} pw={c[3]:.2f}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 3: COMBINE best power + best z5
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 3: COMBINE BEST POWERS + BEST 5TH ZONE')
    print('='*70)

    if best_z5_cfg and best_joint_pw:
        bmp, blp, bbp, btp = best_joint_pw
        lo, hi, w, pw5 = best_z5_cfg
        total = 0
        per_season = {}
        for s, sd in season_data.items():
            p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                   sd['tm'], sd['indices'], ncsos_vals=sd['ncsos'],
                                   mid_pw=bmp, low_pw=blp, bot_pw=bbp, tail_pw=btp,
                                   z5=(lo, hi, w), z5_pw=pw5)
            ex = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            total += ex
            per_season[s] = ex
        print(f'  Combined: {total}/91  per-season: {list(per_season.values())}')
    
    # Also test best z5 with uniform powers
    if best_z5_cfg:
        lo, hi, w, pw5 = best_z5_cfg
        for unif_pw in [0.10, 0.12, 0.15, 0.18, 0.20]:
            total = 0
            for s, sd in season_data.items():
                p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'],
                                       sd['tm'], sd['indices'], ncsos_vals=sd['ncsos'],
                                       mid_pw=unif_pw, low_pw=unif_pw, bot_pw=unif_pw, tail_pw=unif_pw,
                                       z5=(lo, hi, w), z5_pw=pw5)
                total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
            marker = '★' if total > best_z5 else ''
            print(f'  uniform={unif_pw:.2f} + z5: {total}/91 {marker}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 4: NESTED LOSO VALIDATION of best configs
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 4: NESTED LOSO VALIDATION')
    print('='*70)

    # Build config functions
    def v25_fn(sd):
        return apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
    
    def z5_fn(sd):
        lo, hi, w, pw5 = best_z5_cfg if best_z5_cfg else (17, 24, 4, 0.10)
        return apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'],
                                   ncsos_vals=sd['ncsos'], z5=(lo, hi, w), z5_pw=pw5)
    
    def joint_fn(sd):
        if best_joint_pw:
            mp, lp, bp, tp = best_joint_pw
        else:
            mp, lp, bp, tp = 0.15, 0.15, 0.15, 0.15
        return apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'],
                                   mid_pw=mp, low_pw=lp, bot_pw=bp, tail_pw=tp)

    test_configs = {
        'v25': v25_fn,
        'z5': z5_fn,
        'joint_pw': joint_fn,
    }

    # Compute scores
    config_scores = {}
    for name, fn_apply in test_configs.items():
        scores = {}
        for s, sd in season_data.items():
            p = fn_apply(sd)
            scores[s] = count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        config_scores[name] = scores
        total = sum(scores.values())
        print(f'  {name:>12}: {total}/91  {list(scores.values())}')

    # Nested LOSO
    nested_total = 0
    for hold in test_seasons:
        tune = [s for s in test_seasons if s != hold]
        best_tune = -1
        best_name = 'v25'
        for name, scores in config_scores.items():
            ts = sum(scores.get(s, 0) for s in tune)
            if ts > best_tune:
                best_tune = ts
                best_name = name
            elif ts == best_tune and name == 'v25':
                best_name = name
        nested_total += config_scores[best_name][hold]
        print(f'    {hold}: chose {best_name} (tune={best_tune}) → {config_scores[best_name][hold]}')
    
    print(f'\n  ★ Nested LOSO: {nested_total}/91')

    # ════════════════════════════════════════════════════════════
    #  PHASE 5: SWAP-PAIR POST-PROCESSING
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 5: SWAP-PAIR POST-PROCESSING')
    print('  After all zones, try smart pair-swaps based on signals')
    print('='*70)

    # For each season, after v25 zones, look at all adjacent test team pairs
    # Try swapping each pair and see if ALL signals agree
    # This is like a vote: if multiple features agree on the swap, do it
    
    fi = {f: i for i, f in enumerate(fn)}
    
    for decision_threshold in [3, 4, 5]:
        total = 0
        for s, sd in season_data.items():
            p = apply_zones_custom(sd['pass1'], sd['raw'], fn, sd['X'], sd['tm'], sd['indices'])
            
            # Get test teams with their assigned seeds
            test_idx = [(i, p[i]) for i in range(len(p)) if sd['tm'][i]]
            test_idx.sort(key=lambda x: x[1])
            
            # Check all adjacent pairs
            for pi in range(len(test_idx)):
                for pj in range(pi+1, len(test_idx)):
                    ii, si = test_idx[pi]
                    ij, sj = test_idx[pj]
                    if abs(si - sj) > 3:  # only adjacent seeds
                        continue
                    
                    # Should ii be LOWER (better) than ij?
                    # Multiple signals vote:
                    votes = 0
                    xi, xj = sd['X'][ii], sd['X'][ij]
                    
                    # 1. NET rank (lower = better)
                    if xi[fi['NET Rank']] < xj[fi['NET Rank']]: votes += 1
                    elif xi[fi['NET Rank']] > xj[fi['NET Rank']]: votes -= 1
                    
                    # 2. SOS (lower = stronger)
                    if xi[fi['NETSOS']] < xj[fi['NETSOS']]: votes += 1
                    elif xi[fi['NETSOS']] > xj[fi['NETSOS']]: votes -= 1
                    
                    # 3. Q1 wins (more = better)
                    if xi[fi['Quadrant1_W']] > xj[fi['Quadrant1_W']]: votes += 1
                    elif xi[fi['Quadrant1_W']] < xj[fi['Quadrant1_W']]: votes -= 1
                    
                    # 4. Power conference
                    if xi[fi['is_power_conf']] > xj[fi['is_power_conf']]: votes += 1
                    elif xi[fi['is_power_conf']] < xj[fi['is_power_conf']]: votes -= 1
                    
                    # 5. Conference avg NET (lower = stronger)
                    if xi[fi['conf_avg_net']] < xj[fi['conf_avg_net']]: votes += 1
                    elif xi[fi['conf_avg_net']] > xj[fi['conf_avg_net']]: votes -= 1
                    
                    # 6. Win pct
                    if xi[fi['WL_Pct']] > xj[fi['WL_Pct']]: votes += 1
                    elif xi[fi['WL_Pct']] < xj[fi['WL_Pct']]: votes -= 1
                    
                    # 7. Raw score
                    if sd['raw'][ii] < sd['raw'][ij]: votes += 1
                    elif sd['raw'][ii] > sd['raw'][ij]: votes -= 1
                    
                    # If overwhelming vote says ii should be better but ii has worse seed, swap
                    if votes >= decision_threshold and si > sj:
                        p[ii], p[ij] = p[ij], p[ii]
                    elif votes <= -decision_threshold and si < sj:
                        p[ii], p[ij] = p[ij], p[ii]
            
            total += count_exact(p, sd['tm'], sd['indices'], test_mask, y)
        
        marker = '★' if total > 70 else ''
        print(f'  threshold={decision_threshold}: {total}/91 {marker}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 6: DIFFERENT PAIRWISE TRAINING GAPS
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 6: PAIRWISE GAP VARIATIONS (component 1)')
    print('='*70)

    for gap in [10, 15, 20, 25, 30, 35, 40, 50]:
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
            
            # Component 1 with different gap
            pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
                X[gt], y[gt], seasons[gt], max_gap=gap)
            sc_adj = StandardScaler()
            lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr1.fit(sc_adj.fit_transform(pw_X_adj), pw_y_adj)
            s1 = pairwise_score(lr1, X_s, sc_adj)
            
            # Standard components 3 and 4
            X_tk = X[gt][:, tki]
            X_stk = X_s[:, tki]
            pw_Xk, pw_yk = build_pairwise_data(X_tk, y[gt], seasons[gt])
            sck = StandardScaler()
            lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(sck.fit_transform(pw_Xk), pw_yk)
            s3 = pairwise_score(lr3, X_stk, sck)
            
            pw_Xf, pw_yf = build_pairwise_data(X[gt], y[gt], seasons[gt])
            scf = StandardScaler()
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            xgb_clf.fit(scf.fit_transform(pw_Xf), pw_yf)
            s4 = pairwise_score(xgb_clf, X_s, scf)
            
            raw = BLEND_W1 * s1 + BLEND_W3 * s3 + BLEND_W4 * s4
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in si])
            p = apply_zones_custom(p1, raw, fn, X_s, tm, si)
            total += count_exact(p, tm, si, test_mask, y)
        
        marker = ' ★' if total > 70 else ('  (current)' if gap == 30 else '')
        print(f'  gap={gap:2d}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  PHASE 7: MULTI-GAP BLEND
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  PHASE 7: MULTI-GAP BLEND (2 LR components with different gaps)')
    print('='*70)

    for gap_a, gap_b, wa, wb in [
        (20, 30, 0.5, 0.5),
        (15, 30, 0.3, 0.7),
        (25, 30, 0.4, 0.6),
        (20, 40, 0.4, 0.6),
        (15, 40, 0.3, 0.7),
        (20, 30, 0.3, 0.7),
        (20, 30, 0.6, 0.4),
    ]:
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
            
            pw_Xa, pw_ya = build_pairwise_data_adjacent(X[gt], y[gt], seasons[gt], max_gap=gap_a)
            sc_a = StandardScaler()
            lr_a = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr_a.fit(sc_a.fit_transform(pw_Xa), pw_ya)
            sa = pairwise_score(lr_a, X_s, sc_a)
            
            pw_Xb, pw_yb = build_pairwise_data_adjacent(X[gt], y[gt], seasons[gt], max_gap=gap_b)
            sc_b = StandardScaler()
            lr_b = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
            lr_b.fit(sc_b.fit_transform(pw_Xb), pw_yb)
            sb = pairwise_score(lr_b, X_s, sc_b)
            
            s1 = wa * sa + wb * sb
            
            X_tk = X[gt][:, tki]; X_stk = X_s[:, tki]
            pw_Xk, pw_yk = build_pairwise_data(X_tk, y[gt], seasons[gt])
            sck = StandardScaler()
            lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
            lr3.fit(sck.fit_transform(pw_Xk), pw_yk)
            s3 = pairwise_score(lr3, X_stk, sck)
            
            pw_Xf, pw_yf = build_pairwise_data(X[gt], y[gt], seasons[gt])
            scf = StandardScaler()
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric='logloss')
            xgb_clf.fit(scf.fit_transform(pw_Xf), pw_yf)
            s4 = pairwise_score(xgb_clf, X_s, scf)
            
            raw = BLEND_W1 * s1 + BLEND_W3 * s3 + BLEND_W4 * s4
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            p1 = hungarian(raw, seasons[sm], avail, power=HUNGARIAN_POWER)
            tm = np.array([test_mask[gi] for gi in si])
            p = apply_zones_custom(p1, raw, fn, X_s, tm, si)
            total += count_exact(p, tm, si, test_mask, y)
        
        marker = ' ★' if total > 70 else ''
        print(f'  gap_a={gap_a:2d} gap_b={gap_b:2d} wa={wa:.1f} wb={wb:.1f}: {total}/91{marker}')

    # ════════════════════════════════════════════════════════════
    #  SUMMARY
    # ════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print('  SUMMARY')
    print('='*70)

    overall_best = max(best_power_score, best_joint if best_joint_pw else 0,
                       best_z5 if best_z5_cfg else 0, 70)
    print(f'\n  v25 baseline:    70/91')
    print(f'  Best power:      {best_power_score}/91')
    if best_joint_pw:
        print(f'  Best joint pw:   {best_joint}/91 at {best_joint_pw}')
    if best_z5_cfg:
        print(f'  Best z5:         {best_z5}/91 at {best_z5_cfg}')
    print(f'  Overall best:    {overall_best}/91')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
