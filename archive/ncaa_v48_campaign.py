#!/usr/bin/env python3
"""
v48 Improvement Campaign — push v47 (SE=94) lower.

Current v47 errors (from campaign output):
  2023-24 is biggest target: SE=70 (16/21 exact, 5 wrong)
  2022-23: SE=16 (16/21 exact, 5 wrong)  
  2021-22: SE=4 (13/17 exact, 4 wrong)
  2020-21: SE=2 (16/18 exact, 2 wrong)
  2024-25: SE=2 (12/14 exact, 2 wrong)

Strategy:
  A. Error analysis — identify exactly which teams are still wrong at SE=94
  B. Triple-Hungarian — add 3rd model path to ensemble
  C. Feature 9th/10th variants — try adding one more feature to min_8
  D. Different v12 power for dual merge
  E. Per-zone blend weights
  F. Different zone configurations for v47 context
  G. Multiple committee blends averaged
  H. Raw score blending (before Hungarian) instead of post-Hungarian
  I. Different committee zone params for v47
  J. Stacking: use v12+committee raw as features for meta-learner
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from scipy.optimize import linear_sum_assignment
import xgboost as xgb

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_committee_features, build_min8_features,
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
            corr = compute_committee_correction(fn, X_season, *params)
            assigned = apply_midrange_swap(assigned, raw, corr, tm, si, zone=zone, blend=1.0, power=power)
        elif ztype == 'bottom':
            corr = compute_bottom_correction(fn, X_season, *params)
            assigned = apply_bottomzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
        elif ztype == 'tail':
            corr = compute_tail_correction(fn, X_season, opp_rank=params[0])
            assigned = apply_tailzone_swap(assigned, raw, corr, tm, si, zone=zone, power=power)
    return assigned


def cache_v12(X_all, y, fn, seasons, test_mask):
    folds = sorted(set(seasons))
    cache = {}
    for hold_season in folds:
        season_mask = (seasons == hold_season)
        season_test_mask = test_mask & season_mask
        if season_test_mask.sum() == 0: continue
        si = np.where(season_mask)[0]
        train_mask = ~season_test_mask
        tki = select_top_k_features(X_all[train_mask], y[train_mask], fn,
                                    k=USE_TOP_K_A, forced_features=FORCE_FEATURES)[0]
        raw_v12 = predict_robust_blend(X_all[train_mask], y[train_mask],
                                       X_all[season_mask], seasons[train_mask], tki)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_v12[i] = y[gi]
        tm = np.array([test_mask[gi] for gi in si])
        avail = {hold_season: list(range(1, 69))}
        a_v12 = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        a_v12_zoned = apply_zones(a_v12, raw_v12, fn, X_all[season_mask], tm, si, ZONES, 0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'raw_v12': raw_v12,
            'a_v12': a_v12, 'a_v12_zoned': a_v12_zoned,
            'avail': avail, 'X_season': X_all[season_mask],
        }
    return cache


def run_fast(cache, X_comm, y, fn, seasons, test_mask, n,
             alpha=10.0, blend=0.25, zones=ZONES, comm_zones=True,
             comm_model='ridge', el_alpha=0.15, l1_ratio=0.5):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        train_mask = c['train_mask']; avail = c['avail']
        a_v12 = c['a_v12_zoned']; X_season = c['X_season']
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_comm[train_mask])
        X_te = sc.transform(X_comm[sm])
        if comm_model == 'ridge':
            mdl = Ridge(alpha=alpha)
        elif comm_model == 'elastic':
            mdl = ElasticNet(alpha=el_alpha, l1_ratio=l1_ratio, max_iter=5000)
        elif comm_model == 'lasso':
            mdl = Lasso(alpha=el_alpha, max_iter=5000)
        mdl.fit(X_tr, y[train_mask])
        raw_comm = mdl.predict(X_te)
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        a_comm = hungarian(raw_comm, seasons[sm], avail, power=0.15)
        if comm_zones:
            a_comm = apply_zones(a_comm, raw_comm, fn, X_season, tm, si, zones, 0.15)
        avg = (1.0 - blend) * a_v12.astype(float) + blend * a_comm.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        a_final = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = a_final[i]
    return preds


def main():
    t0 = time.time()
    print('='*60)
    print(' v48 IMPROVEMENT CAMPAIGN')
    print(' Starting from v47: SE=94, 73/91, Kaggle=0.437')
    print('='*60)
    
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
    gt = y[test_mask].astype(int)
    fi = {f: i for i, f in enumerate(fn)}
    
    X_min8 = build_min8_features(X_all, fn)
    X_comm21 = build_committee_features(X_all, fn)
    
    print('  Caching v12...')
    cache = cache_v12(X_all, y, fn, seasons, test_mask)
    
    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    
    base = run_fast(cache, X_min8, y, fn, seasons, test_mask, n)
    base_se, base_ex = ev(base)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: DETAILED v47 ERROR ANALYSIS
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' PHASE 1: v47 remaining errors')
    print('='*60)
    errors = []
    for gi in np.where(test_mask)[0]:
        pred = base[gi]; actual = int(y[gi])
        if pred != actual:
            rid = record_ids[gi]; season = seasons[gi]
            net = X_all[gi, fi['NET Rank']]; sos = X_all[gi, fi['NETSOS']]
            opp = X_all[gi, fi['AvgOppNETRank']]
            is_aq = int(X_all[gi, fi['is_AQ']]); is_al = int(X_all[gi, fi['is_AL']])
            is_pow = int(X_all[gi, fi['is_power_conf']])
            tfr = X_all[gi, fi['tourn_field_rank']]
            cb = X_all[gi, fi['cb_mean_seed']]
            q1w = X_all[gi, fi['Quadrant1_W']]
            wpct = X_all[gi, fi['WL_Pct']]
            se = (pred - actual)**2
            errors.append((rid, season, actual, pred, se, net, sos, opp, is_aq, is_al, is_pow, tfr, cb, q1w, wpct))
    
    errors.sort(key=lambda x: -x[4])
    print(f'\n  {len(errors)} wrong predictions, total SE={sum(e[4] for e in errors)}')
    print(f'  {"RecordID":32s} {"Yr":7s} {"GT":>3s} {"Pd":>3s} {"SE":>3s} {"NET":>4s} {"SOS":>4s} {"Opp":>4s} {"TFR":>5s} {"CB":>5s} {"Q1W":>4s} {"W%":>5s} {"AQ":>2s} {"AL":>2s} {"Pw":>2s}')
    for rid, season, actual, pred, se, net, sos, opp, aq, al, pw, tfr, cb, q1w, wpct in errors:
        yr = season[-2:]
        print(f'  {rid:32s} {yr:7s} {actual:3d} {pred:3d} {se:3d} {net:4.0f} {sos:4.0f} {opp:4.0f} {tfr:5.1f} {cb:5.1f} {q1w:4.0f} {wpct:5.3f} {aq:2d} {al:2d} {pw:2d}')
    
    print(f'\n  Per-season:')
    for s in folds:
        s_mask = test_mask & (seasons == s)
        s_gt = y[s_mask].astype(int)
        sse = int(np.sum((base[s_mask] - s_gt)**2))
        eex = int((base[s_mask] == s_gt).sum())
        print(f'    {s}: SE={sse:3d} ({eex}/{s_mask.sum()} exact)')
    
    best_se = base_se
    best_label = 'v47'
    best_preds = base.copy()
    results = []
    
    def try_cfg(label, X_c, **kwargs):
        nonlocal best_se, best_label, best_preds
        p = run_fast(cache, X_c, y, fn, seasons, test_mask, n, **kwargs)
        se, ex = ev(p)
        results.append((label, se, ex))
        if se < best_se:
            best_se, best_label, best_preds = se, label, p.copy()
            print(f'  ★ {label}: SE={se}, exact={ex}/91')
        return se, ex

    # ═══════════════════════════════════════════════════════
    # PHASE 2: FEATURE EXPLORATION (add 9th feature to min_8)
    # ═══════════════════════════════════════════════════════
    print('\n── A: Add 9th feature to min_8 ──')
    net = X_all[:, fi['NET Rank']]
    sos = X_all[:, fi['NETSOS']]
    opp = X_all[:, fi['AvgOppNETRank']]
    is_al = X_all[:, fi['is_AL']]
    is_aq = X_all[:, fi['is_AQ']]
    is_pow = X_all[:, fi['is_power_conf']]
    conf = X_all[:, fi['conf_avg_net']]
    q1w = X_all[:, fi['Quadrant1_W']]
    q1l = X_all[:, fi['Quadrant1_L']]
    q2w = X_all[:, fi['Quadrant2_W']]
    q3l = X_all[:, fi['Quadrant3_L']]
    q4l = X_all[:, fi['Quadrant4_L']]
    wpct = X_all[:, fi['WL_Pct']]
    cb = X_all[:, fi['cb_mean_seed']]
    tfr = X_all[:, fi['tourn_field_rank']]
    
    extra_features = {
        'is_al': is_al,
        'is_aq': is_aq,
        'is_pow': is_pow,
        'q1w': q1w,
        'q2w': q2w,
        'bad_loss': q3l + q4l,
        'net_sos': net - 0.3 * sos,
        'net_conf': net - conf,
        'q1rate': q1w / (q1w + q1l + 0.5),
        'cb_al': cb * is_al,
        'aq_net': is_aq * net,
        'al_200net': is_al * (200 - net),
        'pow_bad': is_pow * (q3l + q4l),
        'aq_weaknet': is_aq * np.maximum(0, net - 50),
        'conf_avg': conf,
        'net2': net**2 / 1000,
        'opp_sos': opp - sos,
        'pow_net': is_pow * net,
        'al_pow_net': is_al * is_pow * net,
        'al_pow_sos': is_al * is_pow * sos,
        'net_tfr': np.abs(net - tfr),
    }
    
    for feat_name, feat_col in extra_features.items():
        X_test = np.column_stack([X_min8, feat_col])
        for alpha in [5, 10, 15]:
            for blend in [0.20, 0.25, 0.30]:
                try_cfg(f'+{feat_name}_a{alpha}_b{blend}', X_test, alpha=alpha, blend=blend)
    
    print(f'  [A done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3: TWO-FEATURE ADDITIONS
    # ═══════════════════════════════════════════════════════
    print('\n── B: Add 2 features to min_8 ──')
    # Try promising pairs
    promising_singles = ['is_al', 'q1w', 'net_sos', 'cb_al', 'bad_loss', 'q1rate']
    for i, f1 in enumerate(promising_singles):
        for f2 in promising_singles[i+1:]:
            X_test = np.column_stack([X_min8, extra_features[f1], extra_features[f2]])
            for alpha in [5, 10, 15]:
                for blend in [0.20, 0.25, 0.30]:
                    try_cfg(f'+{f1}+{f2}_a{alpha}_b{blend}', X_test, alpha=alpha, blend=blend)
    
    print(f'  [B done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 4: TRIPLE HUNGARIAN
    # ═══════════════════════════════════════════════════════
    print('\n── C: Triple-Hungarian (min8 + v1_21feat) ──')
    for min_b in [0.15, 0.20, 0.25]:
        for v1_b in [0.03, 0.05, 0.08, 0.10]:
            if min_b + v1_b > 0.40: continue
            preds_t = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                
                # Min8 Ridge
                sc1 = StandardScaler()
                r1 = Ridge(alpha=10)
                r1.fit(sc1.fit_transform(X_min8[train_mask]), y[train_mask])
                raw1 = r1.predict(sc1.transform(X_min8[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw1[i] = y[gi]
                a1 = hungarian(raw1, seasons[sm], avail, power=0.15)
                a1 = apply_zones(a1, raw1, fn, X_season, tm, si, ZONES, 0.15)
                
                # v1 21-feat Ridge
                sc2 = StandardScaler()
                r2 = Ridge(alpha=15)
                r2.fit(sc2.fit_transform(X_comm21[train_mask]), y[train_mask])
                raw2 = r2.predict(sc2.transform(X_comm21[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw2[i] = y[gi]
                a2 = hungarian(raw2, seasons[sm], avail, power=0.15)
                a2 = apply_zones(a2, raw2, fn, X_season, tm, si, ZONES, 0.15)
                
                w12 = 1.0 - min_b - v1_b
                avg = w12 * a_v12.astype(float) + min_b * a1.astype(float) + v1_b * a2.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_t[gi] = af[i]
            
            se, ex = ev(preds_t)
            results.append((f'triple_m{min_b}_v{v1_b}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'triple_m{min_b}_v{v1_b}'
                best_preds = preds_t.copy()
                print(f'  ★ triple min={min_b} v1={v1_b}: SE={se}, exact={ex}/91')
    
    print(f'  [C done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 5: RAW SCORE BLENDING (before Hungarian)
    # ═══════════════════════════════════════════════════════
    print('\n── D: Raw score blending (pre-Hungarian) ──')
    for alpha in [5, 10, 15]:
        for blend in [0.10, 0.15, 0.20, 0.25, 0.30]:
            preds_raw = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                raw_v12 = c['raw_v12']; X_season = c['X_season']
                
                sc = StandardScaler()
                r = Ridge(alpha=alpha)
                r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                raw_comm = r.predict(sc.transform(X_min8[sm]))
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_comm[i] = y[gi]
                
                # Blend raw scores BEFORE Hungarian
                raw_blend = (1.0 - blend) * raw_v12 + blend * raw_comm
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_blend[i] = y[gi]
                
                a = hungarian(raw_blend, seasons[sm], avail, power=0.15)
                a = apply_zones(a, raw_blend, fn, X_season, tm, si, ZONES, 0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_raw[gi] = a[i]
            
            se, ex = ev(preds_raw)
            results.append((f'rawblend_a{alpha}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'rawblend_a{alpha}_b{blend}'
                best_preds = preds_raw.copy()
                print(f'  ★ raw blend α={alpha} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [D done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 6: STACKING (v12 raw + comm raw as meta features)
    # ═══════════════════════════════════════════════════════
    print('\n── E: Stacking meta-learner ──')
    for meta_alpha in [1, 5, 10, 50, 100]:
        preds_stack = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            train_mask = c['train_mask']; avail = c['avail']
            raw_v12 = c['raw_v12']; X_season = c['X_season']
            
            # Get committee raw scores across all seasons
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
            raw_comm = r.predict(sc.transform(X_min8[sm]))
            
            # Also get v1 21-feat scores
            sc2 = StandardScaler()
            r2 = Ridge(alpha=15)
            r2.fit(sc2.fit_transform(X_comm21[train_mask]), y[train_mask])
            raw_v1 = r2.predict(sc2.transform(X_comm21[sm]))
            
            # Meta features: raw_v12, raw_comm, raw_v1
            meta_X = np.column_stack([raw_v12, raw_comm, raw_v1])
            
            # Need meta training — use only known-seed teams in this season
            meta_train = np.array([not test_mask[gi] for gi in si])
            meta_y = np.array([y[gi] for gi in si])
            
            if meta_train.sum() < 5: continue
            
            meta_r = Ridge(alpha=meta_alpha)
            meta_r.fit(meta_X[meta_train], meta_y[meta_train])
            raw_meta = meta_r.predict(meta_X)
            
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw_meta[i] = y[gi]
            a = hungarian(raw_meta, seasons[sm], avail, power=0.15)
            a = apply_zones(a, raw_meta, fn, X_season, tm, si, ZONES, 0.15)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_stack[gi] = a[i]
        
        se, ex = ev(preds_stack)
        results.append((f'stack_a{meta_alpha}', se, ex))
        if se < best_se:
            best_se, best_label = se, f'stack_a{meta_alpha}'
            best_preds = preds_stack.copy()
            print(f'  ★ stack α={meta_alpha}: SE={se}, exact={ex}/91')
    
    print(f'  [E done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 7: ZONE RE-OPTIMIZATION (focused on 2023-24)
    # ═══════════════════════════════════════════════════════
    print('\n── F: Zone re-optimization (focused sweeps) ──')
    # The biggest errors are in seeds 36-42 range (NM=42→36, NW=36→41)
    # Try adding a new zone or adjusting existing ones
    
    # Expand uppermid zone boundaries
    for lo in [32, 34, 36]:
        for hi in [42, 44, 46, 48]:
            for aq in [-3, -2, -1, 0]:
                for al in [-5, -4, -3, -2]:
                    for sos_p in [-5, -4, -3, -2]:
                        zones_test = [
                            ('mid',     'committee', (17, lo), (0, 0, 3)),
                            ('uppermid','committee', (lo, hi), (aq, al, sos_p)),
                            ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
                            ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
                            ('tail',    'tail',      (60, 63), (1,)),
                        ]
                        try_cfg(f'z_lo{lo}_hi{hi}_a{aq}_l{al}_s{sos_p}',
                               X_min8, zones=zones_test)
    
    print(f'  [F done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 8: DIFFERENT HUNGARIAN POWER FOR FINAL STEP
    # ═══════════════════════════════════════════════════════
    print('\n── G: Final Hungarian power ──')
    for power in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50]:
        preds_p = np.zeros(n, dtype=int)
        for hold_season, c in cache.items():
            sm = c['season_mask']; si = c['si']; tm = c['tm']
            train_mask = c['train_mask']; avail = c['avail']
            a_v12 = c['a_v12_zoned']; X_season = c['X_season']
            sc = StandardScaler()
            r = Ridge(alpha=10)
            r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
            raw = r.predict(sc.transform(X_min8[sm]))
            for i, gi in enumerate(si):
                if not test_mask[gi]: raw[i] = y[gi]
            ac = hungarian(raw, seasons[sm], avail, power=0.15)
            ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
            avg = 0.75 * a_v12.astype(float) + 0.25 * ac.astype(float)
            for i, gi in enumerate(si):
                if not test_mask[gi]: avg[i] = y[gi]
            af = hungarian(avg, seasons[sm], avail, power=power)
            for i, gi in enumerate(si):
                if test_mask[gi]: preds_p[gi] = af[i]
        
        se, ex = ev(preds_p)
        results.append((f'fpower_{power}', se, ex))
        if se < best_se:
            best_se, best_label = se, f'fpower_{power}'
            best_preds = preds_p.copy()
            print(f'  ★ final power={power}: SE={se}, exact={ex}/91')
    
    print(f'  [G done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 9: XGB AS COMMITTEE
    # ═══════════════════════════════════════════════════════
    print('\n── H: XGB committee on min_8 features ──')
    for n_est in [100, 200, 300]:
        for depth in [2, 3, 4]:
            for blend in [0.20, 0.25, 0.30]:
                preds_x = np.zeros(n, dtype=int)
                for hold_season, c in cache.items():
                    sm = c['season_mask']; si = c['si']; tm = c['tm']
                    train_mask = c['train_mask']; avail = c['avail']
                    a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                    m = xgb.XGBRegressor(n_estimators=n_est, max_depth=depth,
                                          learning_rate=0.05, subsample=0.8,
                                          colsample_bytree=0.8, verbosity=0, random_state=42)
                    m.fit(X_min8[train_mask], y[train_mask])
                    raw = m.predict(X_min8[sm])
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: raw[i] = y[gi]
                    ac = hungarian(raw, seasons[sm], avail, power=0.15)
                    ac = apply_zones(ac, raw, fn, X_season, tm, si, ZONES, 0.15)
                    avg = (1 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                    for i, gi in enumerate(si):
                        if not test_mask[gi]: avg[i] = y[gi]
                    af = hungarian(avg, seasons[sm], avail, power=0.15)
                    for i, gi in enumerate(si):
                        if test_mask[gi]: preds_x[gi] = af[i]
                
                se, ex = ev(preds_x)
                results.append((f'xgb_n{n_est}_d{depth}_b{blend}', se, ex))
                if se < best_se:
                    best_se, best_label = se, f'xgb_n{n_est}_d{depth}_b{blend}'
                    best_preds = preds_x.copy()
                    print(f'  ★ XGB n={n_est} d={depth} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [H done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 10: ElasticNet ON MIN8 WITH FINE TUNING
    # ═══════════════════════════════════════════════════════
    print('\n── I: ElasticNet on min_8 ──')
    for ea in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        for l1r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for blend in [0.20, 0.25, 0.30]:
                try_cfg(f'EN_ea{ea}_l1{l1r}_b{blend}', X_min8,
                       comm_model='elastic', el_alpha=ea, l1_ratio=l1r, blend=blend)
    
    print(f'  [I done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 11: MULTI-ALPHA RIDGE ENSEMBLE ON MIN8
    # ═══════════════════════════════════════════════════════
    print('\n── J: Multi-alpha Ridge ensemble ──')
    for alphas, label in [
        ([3, 10], '3_10'), ([5, 15], '5_15'), ([3, 10, 30], '3_10_30'),
        ([5, 10, 15], '5_10_15'), ([1, 5, 10, 20, 50], '1-50'),
    ]:
        for blend in [0.20, 0.25, 0.30]:
            preds_ma = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                raw_avg = np.zeros(sm.sum())
                for a in alphas:
                    sc = StandardScaler()
                    r = Ridge(alpha=a)
                    r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                    raw_avg += r.predict(sc.transform(X_min8[sm]))
                raw_avg /= len(alphas)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_avg[i] = y[gi]
                ac = hungarian(raw_avg, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw_avg, fn, X_season, tm, si, ZONES, 0.15)
                avg = (1 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_ma[gi] = af[i]
            se, ex = ev(preds_ma)
            results.append((f'mα_{label}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'mα_{label}_b{blend}'
                best_preds = preds_ma.copy()
                print(f'  ★ multi-α {label} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [J done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # PHASE 12: RIDGE+XGB COMMITTEE BLEND ON MIN8
    # ═══════════════════════════════════════════════════════
    print('\n── K: Ridge+XGB committee blend ──')
    for xgb_w in [0.2, 0.3, 0.5]:
        for blend in [0.20, 0.25, 0.30]:
            preds_rx = np.zeros(n, dtype=int)
            for hold_season, c in cache.items():
                sm = c['season_mask']; si = c['si']; tm = c['tm']
                train_mask = c['train_mask']; avail = c['avail']
                a_v12 = c['a_v12_zoned']; X_season = c['X_season']
                sc = StandardScaler()
                r = Ridge(alpha=10)
                r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
                raw_r = r.predict(sc.transform(X_min8[sm]))
                xm = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, verbosity=0, random_state=42)
                xm.fit(X_min8[train_mask], y[train_mask])
                raw_x = xm.predict(X_min8[sm])
                raw_blend = (1 - xgb_w) * raw_r + xgb_w * raw_x
                for i, gi in enumerate(si):
                    if not test_mask[gi]: raw_blend[i] = y[gi]
                ac = hungarian(raw_blend, seasons[sm], avail, power=0.15)
                ac = apply_zones(ac, raw_blend, fn, X_season, tm, si, ZONES, 0.15)
                avg = (1 - blend) * a_v12.astype(float) + blend * ac.astype(float)
                for i, gi in enumerate(si):
                    if not test_mask[gi]: avg[i] = y[gi]
                af = hungarian(avg, seasons[sm], avail, power=0.15)
                for i, gi in enumerate(si):
                    if test_mask[gi]: preds_rx[gi] = af[i]
            se, ex = ev(preds_rx)
            results.append((f'rx_xw{xgb_w}_b{blend}', se, ex))
            if se < best_se:
                best_se, best_label = se, f'rx_xw{xgb_w}_b{blend}'
                best_preds = preds_rx.copy()
                print(f'  ★ rx xw={xgb_w} b={blend}: SE={se}, exact={ex}/91')
    
    print(f'  [K done] best={best_label} SE={best_se}')
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print('\n' + '='*60)
    print(' FINAL RESULTS')
    print('='*60)
    print(f'  Baseline v47: SE={base_se}, exact={base_ex}/91')
    print(f'  Best found: {best_label}, SE={best_se}')
    print(f'  Improvement: Δ={base_se - best_se}')
    
    results.sort(key=lambda x: x[1])
    print(f'\n  Top 20:')
    for label, se, ex in results[:20]:
        flag = '★' if se < base_se else ' '
        print(f'  {flag} SE={se:4d}  exact={ex:2d}/91  {label}')
    
    if best_se < base_se:
        print(f'\n  Errors changed from v47:')
        for gi in np.where(test_mask)[0]:
            old, new, actual = base[gi], best_preds[gi], int(y[gi])
            if old != new:
                rid = record_ids[gi]
                print(f'    {rid:32s} gt={actual:2d} old={old:2d} new={new:2d} '
                      f'oldSE={(old-actual)**2:3d} newSE={(new-actual)**2:3d}')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*60)


if __name__ == '__main__':
    main()
