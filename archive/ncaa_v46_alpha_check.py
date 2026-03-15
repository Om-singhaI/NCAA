#!/usr/bin/env python3
"""Quick α sensitivity check near 15 for dual-Hungarian."""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

ZONES = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

def apply_zones(assigned, raw, fn, X_season, tm, si, zones, power=0.15):
    for name, ztype, zone, params in zones:
        if ztype == 'committee':
            aq, al, sos = params
            corr = compute_committee_correction(fn, X_season, alpha_aq=aq, beta_al=al, gamma_sos=sos)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            sn, nc, cb = params
            corr = compute_bottom_correction(fn, X_season, sosnet=sn, net_conf=nc, cbhist=cb)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            opp = params[0]
            corr = compute_tail_correction(fn, X_season, opp_rank=opp)
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned

def build_committee_features(X, fn):
    fi = {f: i for i, f in enumerate(fn)}
    net = X[:, fi['NET Rank']]
    sos = X[:, fi['NETSOS']]
    opp = X[:, fi['AvgOppNETRank']]
    is_al = X[:, fi['is_AL']]
    is_aq = X[:, fi['is_AQ']]
    is_power = X[:, fi['is_power_conf']]
    conf_avg = X[:, fi['conf_avg_net']]
    q1w = X[:, fi['Quadrant1_W']]
    q1l = X[:, fi['Quadrant1_L']]
    q3l = X[:, fi['Quadrant3_L']]
    q4l = X[:, fi['Quadrant4_L']]
    wpct = X[:, fi['WL_Pct']]
    cb_mean = X[:, fi['cb_mean_seed']]
    tfr = X[:, fi['tourn_field_rank']]
    feats = [net, sos, opp, is_al, is_power, q1w, q3l+q4l, wpct, cb_mean, tfr,
             is_aq*(1-is_power)*net, is_al*is_power*(200-net), net-0.3*sos,
             net-conf_avg, is_aq*np.maximum(0,net-50), is_power*np.maximum(0,100-sos),
             q1w/(q1w+q1l+0.5), is_power*(q3l+q4l), tfr, cb_mean*is_aq, cb_mean*is_al]
    return np.column_stack(feats)

def main():
    t0 = time.time()
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
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    X_comm = build_committee_features(X_all, fn)

    print('α sensitivity near 15, blend=0.15:')
    for alpha in [11, 12, 13, 14, 14.5, 15, 15.5, 16, 17, 18, 19, 20, 25, 30]:
        preds = np.zeros(n, dtype=int)
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            tki = select_top_k_features(X_all[global_train_mask], y[global_train_mask],
                                        fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
            raw_v12 = predict_robust_blend(X_all[global_train_mask], y[global_train_mask],
                                           X_season, seasons[global_train_mask], tki)
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
            X_te_sc = sc.transform(X_comm[season_mask])
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_tr_sc, y[global_train_mask])
            raw_comm = ridge.predict(X_te_sc)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw_v12[i] = y[gi]
                    raw_comm[i] = y[gi]
            tm = np.array([test_mask[gi] for gi in season_indices])
            avail = {hold_season: list(range(1, 69))}
            a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
            a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, season_indices, ZONES, 0.15)
            a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, season_indices, ZONES, 0.15)
            avg = 0.85 * a_v12.astype(float) + 0.15 * a_comm.astype(float)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    avg[i] = y[gi]
            a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = a_final[i]
        gt = y[test_mask].astype(int)
        pr = preds[test_mask]
        se = int(np.sum((pr - gt)**2))
        exact = int((pr == gt).sum())
        marker = ' ★' if se < 188 else ''
        print(f'  α={alpha:5.1f}: SE={se:4d}, exact={exact}/91{marker}')

    # Also try some blend variations at α=15
    print('\nBlend sensitivity at α=15:')
    for blend_pct in [5, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20, 25]:
        blend = blend_pct / 100.0
        preds = np.zeros(n, dtype=int)
        for hold_season in folds:
            season_mask = (seasons == hold_season)
            season_test_mask = test_mask & season_mask
            if season_test_mask.sum() == 0: continue
            X_season = X_all[season_mask]
            season_indices = np.where(season_mask)[0]
            global_train_mask = ~season_test_mask
            tki = select_top_k_features(X_all[global_train_mask], y[global_train_mask],
                                        fn, k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
            raw_v12 = predict_robust_blend(X_all[global_train_mask], y[global_train_mask],
                                           X_season, seasons[global_train_mask], tki)
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_comm[global_train_mask])
            X_te_sc = sc.transform(X_comm[season_mask])
            ridge = Ridge(alpha=15)
            ridge.fit(X_tr_sc, y[global_train_mask])
            raw_comm = ridge.predict(X_te_sc)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    raw_v12[i] = y[gi]
                    raw_comm[i] = y[gi]
            tm = np.array([test_mask[gi] for gi in season_indices])
            avail = {hold_season: list(range(1, 69))}
            a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
            a_v12 = apply_zones(a_v12, raw_v12, fn, X_season, tm, season_indices, ZONES, 0.15)
            a_comm = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, season_indices, ZONES, 0.15)
            avg = (1-blend) * a_v12.astype(float) + blend * a_comm.astype(float)
            for i, gi in enumerate(season_indices):
                if not test_mask[gi]:
                    avg[i] = y[gi]
            a_final = hungarian(avg, seasons[season_mask], avail, power=0.15)
            for i, gi in enumerate(season_indices):
                if test_mask[gi]:
                    preds[gi] = a_final[i]
        gt = y[test_mask].astype(int)
        pr = preds[test_mask]
        se = int(np.sum((pr - gt)**2))
        exact = int((pr == gt).sum())
        marker = ' ★' if se < 132 else (' ◆' if se == 132 else '')
        print(f'  blend={blend:.2f}: SE={se:4d}, exact={exact}/91{marker}')

    print(f'\nDone in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
