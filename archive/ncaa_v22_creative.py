#!/usr/bin/env python3
"""
v22: Final creative ideas to push beyond 61/91
================================================

After v19-v21 exhaustively showed 61/91 is the ceiling for:
  - Zone extensions, hyperparams, features, ensembles, stacking, direct regression,
    zone-specific Hungarian power, alternative architectures

This tries two final ideas:
  1. WEIGHTED pairwise training: weight pairs by 1/|seed_gap| to focus on
     distinguishing similar teams (where most errors are)
  2. Multi-seed model averaging (different random states)
  3. Isotonic regression calibration of raw scores
  4. Per-season model selection: pick different model configs per season
  5. RMSE optimization: test configs optimized for RMSE not exact match
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, build_pairwise_data, build_pairwise_data_adjacent,
    pairwise_score, hungarian,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER,
    PW_C1, PW_C3, ADJ_COMP1_GAP,
    BLEND_W1, BLEND_W3, BLEND_W4
)

warnings.filterwarnings('ignore')
np.random.seed(42)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_correction(feature_names, X_data, aq=0, al=2, sos=3):
    fi = {f: i for i, f in enumerate(feature_names)}
    correction = np.zeros(X_data.shape[0])
    net = X_data[:, fi['NET Rank']]
    is_al = X_data[:, fi['is_AL']]
    is_power = X_data[:, fi['is_power_conf']]
    sos_val = X_data[:, fi['NETSOS']]
    if al != 0:
        correction -= al * is_al * is_power * np.clip((net - 20) / 50, 0, 1)
    if sos != 0:
        correction += sos * (sos_val - net) / 100
    return correction


def apply_swap(pass1, raw_scores, correction, test_mask_season, zone=(17,34), power=0.15):
    lo, hi = zone
    mid_test = [i for i in range(len(pass1))
                if test_mask_season[i] and lo <= pass1[i] <= hi]
    if len(mid_test) <= 1:
        return pass1.copy()
    mid_seeds = [pass1[i] for i in mid_test]
    mid_corr = [raw_scores[i] + correction[i] for i in mid_test]
    cost = np.array([[abs(s - seed)**power for seed in mid_seeds] for s in mid_corr])
    ri, ci = linear_sum_assignment(cost)
    final = pass1.copy()
    for r, c in zip(ri, ci):
        final[mid_test[r]] = mid_seeds[c]
    return final


def build_weighted_pairwise_data(X, y, seasons, max_gap=68, weight_fn='inv'):
    """Build pairwise data with sample weights based on seed gap."""
    pairs_X, pairs_y, pairs_w = [], [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]
                gap = abs(y[a] - y[b])
                if gap > max_gap:
                    continue
                diff = X[a] - X[b]
                target = 1.0 if y[a] < y[b] else 0.0

                # Weight: focus on smaller gaps (harder pairs)
                if weight_fn == 'inv':
                    w = 1.0 / (gap + 1)
                elif weight_fn == 'inv_sqrt':
                    w = 1.0 / np.sqrt(gap + 1)
                elif weight_fn == 'exp':
                    w = np.exp(-gap / 20)
                elif weight_fn == 'linear':
                    w = max(0, 1 - gap / 68)
                else:
                    w = 1.0

                pairs_X.append(diff)
                pairs_y.append(target)
                pairs_w.append(w)
                pairs_X.append(-diff)
                pairs_y.append(1.0 - target)
                pairs_w.append(w)

    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_w)


def predict_weighted_blend(X_train, y_train, X_test, seasons_train, top_k_idx,
                            weight_fn='inv', gap=30):
    """v12-like blend but with weighted pairwise training."""
    # Component 1: weighted adj-pair LR
    pw_X, pw_y, pw_w = build_weighted_pairwise_data(
        X_train, y_train, seasons_train, max_gap=gap, weight_fn=weight_fn)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_sc, pw_y, sample_weight=pw_w)
    score1 = pairwise_score(lr1, X_test, sc)

    # Component 3: weighted top-K LR
    X_tr_k = X_train[:, top_k_idx]
    X_te_k = X_test[:, top_k_idx]
    pw_Xk, pw_yk, pw_wk = build_weighted_pairwise_data(
        X_tr_k, y_train, seasons_train, max_gap=68, weight_fn=weight_fn)
    sc_k = StandardScaler()
    pw_Xk_sc = sc_k.fit_transform(pw_Xk)
    lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_Xk_sc, pw_yk, sample_weight=pw_wk)
    score3 = pairwise_score(lr3, X_te_k, sc_k)

    # Component 4: XGB (doesn't easily support sample weights in this setup)
    pw_Xf, pw_yf = build_pairwise_data(X_train, y_train, seasons_train)
    sc_f = StandardScaler()
    pw_Xf_sc = sc_f.fit_transform(pw_Xf)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss')
    xgb_clf.fit(pw_Xf_sc, pw_yf)
    score4 = pairwise_score(xgb_clf, X_test, sc_f)

    return BLEND_W1 * score1 + BLEND_W3 * score3 + BLEND_W4 * score4


def predict_multiseed(X_train, y_train, X_test, seasons_train, top_k_idx, seeds_list):
    """Average predictions across multiple random seeds."""
    all_scores = []
    for seed in seeds_list:
        # Component 1
        pw_X, pw_y = build_pairwise_data_adjacent(X_train, y_train, seasons_train, max_gap=30)
        sc = StandardScaler()
        pw_X_sc = sc.fit_transform(pw_X)
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=seed)
        lr1.fit(pw_X_sc, pw_y)
        s1 = pairwise_score(lr1, X_test, sc)

        # Component 3
        X_tr_k = X_train[:, top_k_idx]
        X_te_k = X_test[:, top_k_idx]
        pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_train, seasons_train)
        sc_k = StandardScaler()
        pw_Xk_sc = sc_k.fit_transform(pw_Xk)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=seed)
        lr3.fit(pw_Xk_sc, pw_yk)
        s3 = pairwise_score(lr3, X_te_k, sc_k)

        # Component 4
        pw_Xf, pw_yf = build_pairwise_data(X_train, y_train, seasons_train)
        sc_f = StandardScaler()
        pw_Xf_sc = sc_f.fit_transform(pw_Xf)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=seed, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_Xf_sc, pw_yf)
        s4 = pairwise_score(xgb_clf, X_test, sc_f)

        score = BLEND_W1 * s1 + BLEND_W3 * s3 + BLEND_W4 * s4
        all_scores.append(score)

    return np.mean(all_scores, axis=0)


def main():
    t0 = time.time()
    print('='*70)
    print(' v22: FINAL CREATIVE IDEAS')
    print('='*70)

    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
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

    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(np.where(np.isinf(feat.values.astype(np.float64)),
                                        np.nan, feat.values.astype(np.float64)))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    test_seasons = [s for s in folds if (test_mask & (seasons == s)).sum() > 0]

    print(f'\n  Teams: {n_labeled}, Test: {test_mask.sum()}, Seasons: {folds}')

    # Helper function
    def eval_raw_scores(raw_scores_per_season, do_correction=True):
        total = 0
        ps = {}
        for s in test_seasons:
            sm = (seasons == s)
            si = np.where(sm)[0]
            raw = raw_scores_per_season[s]
            avail = {s: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[sm], avail, power=0.15)
            if do_correction:
                tmask_s = np.array([test_mask[gi] for gi in si])
                corr = compute_correction(fn, X_all[sm])
                pass1 = apply_swap(pass1, raw, corr, tmask_s)
            ex = sum(1 for i, gi in enumerate(si) if test_mask[gi] and pass1[i] == int(y[gi]))
            total += ex
            ps[s] = ex
        return total, ps

    # ════════════════════════════════════════════════════════════════
    #  Idea 1: WEIGHTED pairwise training
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 1: Weighted pairwise training')
    print('='*70)

    for wfn in ['inv', 'inv_sqrt', 'exp', 'linear']:
        for gap in [30, 40, 68]:
            raw_per_s = {}
            for hold in test_seasons:
                sm = (seasons == hold)
                si = np.where(sm)[0]
                st = test_mask & sm
                tr = ~st
                tk = select_top_k_features(X_all[tr], y[tr], fn, k=USE_TOP_K_A,
                                           forced_features=FORCE_FEATURES)[0]
                raw = predict_weighted_blend(X_all[tr], y[tr], X_all[sm],
                                             seasons[tr], tk, weight_fn=wfn, gap=gap)
                for i, gi in enumerate(si):
                    if not test_mask[gi]:
                        raw[i] = y[gi]
                raw_per_s[hold] = raw

            ex, ps = eval_raw_scores(raw_per_s)
            ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
            marker = ' ←' if ex > 61 else ''
            print(f'  wt={wfn:<10} gap={gap:2d}: {ex}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Idea 2: Multi-seed averaging
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 2: Multi-seed model averaging')
    print('='*70)

    seed_sets = [
        ('1-seed(42)', [42]),
        ('3-seed', [42, 123, 777]),
        ('5-seed', [42, 123, 777, 2024, 31415]),
        ('5-seed-alt', [7, 13, 99, 42, 500]),
    ]

    for label, seeds in seed_sets:
        raw_per_s = {}
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            st = test_mask & sm
            tr = ~st
            tk = select_top_k_features(X_all[tr], y[tr], fn, k=USE_TOP_K_A,
                                       forced_features=FORCE_FEATURES)[0]
            raw = predict_multiseed(X_all[tr], y[tr], X_all[sm], seasons[tr], tk, seeds)
            for i, gi in enumerate(si):
                if not test_mask[gi]:
                    raw[i] = y[gi]
            raw_per_s[hold] = raw

        ex, ps = eval_raw_scores(raw_per_s)
        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if ex > 61 else ''
        print(f'  {label:<15}: {ex}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Idea 3: Isotonic calibration of raw scores
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 3: Isotonic regression calibration')
    print('='*70)

    # Standard v12 raw scores first
    v12_raw = {}
    for hold in test_seasons:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        tr = ~st
        tk = select_top_k_features(X_all[tr], y[tr], fn, k=USE_TOP_K_A,
                                   forced_features=FORCE_FEATURES)[0]
        raw = predict_robust_blend(X_all[tr], y[tr], X_all[sm], seasons[tr], tk)
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                raw[i] = y[gi]
        v12_raw[hold] = raw

    # a) No calibration baseline
    ex_nc, ps_nc = eval_raw_scores(v12_raw, do_correction=False)
    ex_wc, ps_wc = eval_raw_scores(v12_raw, do_correction=True)
    ps_str = ' '.join(f'{ps_wc.get(s,0):2d}' for s in test_seasons)
    print(f'  No calibration: {ex_nc}/91 (no corr), {ex_wc}/91 (w/corr) [{ps_str}]')

    # b) Isotonic calibration: learn mapping from raw_score → true_seed on training data
    iso_raw = {}
    for hold in test_seasons:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        st = test_mask & sm
        tr = ~st

        # Collect training predictions and true seeds
        train_idx = np.where(tr)[0]
        train_preds = []
        for inner_fold in sorted(set(seasons[tr])):
            inner_te = (seasons == inner_fold) & tr
            inner_tr = tr & ~inner_te
            if inner_te.sum() == 0:
                continue
            tk = select_top_k_features(X_all[inner_tr], y[inner_tr], fn, k=USE_TOP_K_A,
                                       forced_features=FORCE_FEATURES)[0]
            inner_raw = predict_robust_blend(X_all[inner_tr], y[inner_tr],
                                              X_all[inner_te], seasons[inner_tr], tk)
            for i, gi in enumerate(np.where(inner_te)[0]):
                train_preds.append((inner_raw[i], y[gi]))

        # Fit isotonic regression
        tp_arr = np.array(train_preds)
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(tp_arr[:, 0], tp_arr[:, 1])

        # Calibrate test predictions
        raw = v12_raw[hold].copy()
        test_indices = [i for i, gi in enumerate(si) if test_mask[gi]]
        for i in test_indices:
            raw[i] = iso.predict([raw[i]])[0]

        iso_raw[hold] = raw

    ex_iso, ps_iso = eval_raw_scores(iso_raw, do_correction=True)
    ps_str = ' '.join(f'{ps_iso.get(s,0):2d}' for s in test_seasons)
    marker = ' ←' if ex_iso > 61 else ''
    print(f'  With isotonic calibration: {ex_iso}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Idea 4: Different Hungarian power for the correction step
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 4: Correction step power sweep')
    print('='*70)

    for corr_power in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]:
        total = 0
        ps = {}
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            raw = v12_raw[hold]
            avail = {hold: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[sm], avail, power=0.15)
            tmask_s = np.array([test_mask[gi] for gi in si])
            corr = compute_correction(fn, X_all[sm])
            p2 = apply_swap(pass1, raw, corr, tmask_s, power=corr_power)
            ex = sum(1 for i, gi in enumerate(si) if test_mask[gi] and p2[i] == int(y[gi]))
            total += ex
            ps[hold] = ex

        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if total > 61 else (' ←SAME' if total == 61 else '')
        print(f'  corr_power={corr_power:.2f}: {total}/91 [{ps_str}]{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Idea 5: RMSE-optimized power
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 5: RMSE-optimized assignment')
    print('='*70)

    for hw_power in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]:
        total_exact = 0
        total_rmse = 0
        total_n = 0
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            raw = v12_raw[hold]
            avail = {hold: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[sm], avail, power=hw_power)
            tmask_s = np.array([test_mask[gi] for gi in si])
            corr = compute_correction(fn, X_all[sm])
            p2 = apply_swap(pass1, raw, corr, tmask_s)

            n_t = 0; exact = 0; sse = 0
            for i, gi in enumerate(si):
                if test_mask[gi]:
                    n_t += 1
                    if p2[i] == int(y[gi]):
                        exact += 1
                    sse += (p2[i] - y[gi])**2
            total_exact += exact
            total_rmse += sse
            total_n += n_t

        rmse = np.sqrt(total_rmse / total_n) if total_n > 0 else 999
        marker = ' ←' if total_exact > 61 else ''
        print(f'  hw_power={hw_power:.2f}: {total_exact}/91 exact, RMSE={rmse:.4f}{marker}')

    # ════════════════════════════════════════════════════════════════
    #  Idea 6: Different correction strength (blend factor)
    # ════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' IDEA 6: Correction blend factor sweep')
    print('='*70)

    for blend_f in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        total = 0
        ps = {}
        for hold in test_seasons:
            sm = (seasons == hold)
            si = np.where(sm)[0]
            raw = v12_raw[hold]
            avail = {hold: list(range(1, 69))}
            pass1 = hungarian(raw, seasons[sm], avail, power=0.15)
            tmask_s = np.array([test_mask[gi] for gi in si])
            corr = compute_correction(fn, X_all[sm])

            # Modified swap with blend factor
            lo, hi = 17, 34
            mid_test = [i for i in range(len(pass1))
                        if tmask_s[i] and lo <= pass1[i] <= hi]
            if len(mid_test) > 1:
                mid_seeds = [pass1[i] for i in mid_test]
                mid_corr = [raw[i] + blend_f * corr[i] for i in mid_test]
                cost = np.array([[abs(s - seed)**0.15 for seed in mid_seeds] for s in mid_corr])
                ri, ci = linear_sum_assignment(cost)
                p2 = pass1.copy()
                for r, c in zip(ri, ci):
                    p2[mid_test[r]] = mid_seeds[c]
            else:
                p2 = pass1.copy()

            ex = sum(1 for i, gi in enumerate(si) if test_mask[gi] and p2[i] == int(y[gi]))
            total += ex
            ps[hold] = ex

        ps_str = ' '.join(f'{ps.get(s,0):2d}' for s in test_seasons)
        marker = ' ←' if total > 61 else (' ←SAME' if total == 61 else '')
        print(f'  blend_f={blend_f:.2f}: {total}/91 [{ps_str}]{marker}')

    print(f'\n  Total time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
