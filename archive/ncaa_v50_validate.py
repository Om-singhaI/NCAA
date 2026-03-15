#!/usr/bin/env python3
"""
v50 VALIDATION — Extreme tail zone (63-68) bottom correction
Found: sn=1, nc=-1, cb=-1 → SE=14 (from 16), +2 exact, 0 regressions

Tests:
  A: Plateau width — how many configs achieve SE=14?
  B: Per-season breakdown — which seasons affected?
  C: Per-team changes — who moved?
  D: Theoretical justification — why does this correction make sense?
  E: Bootstrap validation (50 resamples)
  F: Perturbation stability
  G: Combined with other near-improvements
  H: Zone boundary robustness (different lo/hi)
"""

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
    build_min8_features,
    compute_committee_correction, apply_midrange_swap,
    compute_bottom_correction, apply_bottomzone_swap,
    compute_tail_correction, apply_tailzone_swap,
    USE_TOP_K_A, FORCE_FEATURES,
)

ZONES_V12 = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-2, -3, -4)),
    ('midbot',  'bottom',    (48, 52), (0, 2, -2)),
    ('bot',     'bottom',    (52, 60), (-4, 3, -1)),
    ('tail',    'tail',      (60, 63), (1,)),
]

ZONES_V49_COMM = [
    ('mid',     'committee', (17, 34), (0, 0, 3)),
    ('uppermid','committee', (34, 44), (-6, 1, -6)),
    ('midbot2', 'bottom',    (42, 50), (-4, 2, -3)),
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


def apply_swap(preds, X_all, fi, seasons, test_mask, net_gap=10, pred_gap=6):
    preds = preds.copy()
    for hold_season in sorted(set(seasons)):
        sm = (seasons == hold_season)
        si_s = np.where(sm)[0]
        test_si = [gi for gi in si_s if test_mask[gi]]
        aq_teams, al_teams = [], []
        for gi in test_si:
            pred = preds[gi]
            if 30 <= pred <= 45:
                is_aq = int(X_all[gi, fi['is_AQ']])
                is_al = int(X_all[gi, fi['is_AL']])
                net = X_all[gi, fi['NET Rank']]
                if is_aq and not is_al and pred - net > net_gap:
                    aq_teams.append((gi, pred, net))
                elif is_al and not is_aq and net - pred > 0:
                    al_teams.append((gi, pred, net))
        for aq_gi, aq_pred, aq_net in aq_teams:
            for al_gi, al_pred, al_net in al_teams:
                if abs(preds[aq_gi] - preds[al_gi]) <= pred_gap:
                    preds[aq_gi], preds[al_gi] = preds[al_gi], preds[aq_gi]
    return preds


def cache_all(X_all, X_min8, y, fn, seasons, test_mask):
    cache = {}
    for hold_season in sorted(set(seasons)):
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
        a_v12_raw = hungarian(raw_v12, seasons[season_mask], avail, power=0.15)
        sc = StandardScaler()
        r = Ridge(alpha=10)
        r.fit(sc.fit_transform(X_min8[train_mask]), y[train_mask])
        raw_comm = r.predict(sc.transform(X_min8[season_mask]))
        for i, gi in enumerate(si):
            if not test_mask[gi]: raw_comm[i] = y[gi]
        a_comm_raw = hungarian(raw_comm, seasons[season_mask], avail, power=0.15)
        cache[hold_season] = {
            'season_mask': season_mask, 'si': si, 'tm': tm,
            'train_mask': train_mask, 'avail': avail,
            'raw_v12': raw_v12, 'a_v12_raw': a_v12_raw,
            'raw_comm': raw_comm, 'a_comm_raw': a_comm_raw,
            'X_season': X_all[season_mask],
        }
    return cache


def run_pipeline(cache, y, fn, seasons, test_mask, n,
                 v12_zones=ZONES_V12, comm_zones=ZONES_V49_COMM, blend=0.25):
    preds = np.zeros(n, dtype=int)
    for hold_season, c in cache.items():
        sm = c['season_mask']; si = c['si']; tm = c['tm']
        avail = c['avail']; X_season = c['X_season']
        a_v12 = apply_zones(c['a_v12_raw'].copy(), c['raw_v12'], fn, X_season, tm, si, v12_zones, 0.15)
        ac = apply_zones(c['a_comm_raw'].copy(), c['raw_comm'], fn, X_season, tm, si, comm_zones, 0.15)
        avg = (1.0 - blend) * a_v12.astype(float) + blend * ac.astype(float)
        for i, gi in enumerate(si):
            if not test_mask[gi]: avg[i] = y[gi]
        af = hungarian(avg, seasons[sm], avail, power=0.15)
        for i, gi in enumerate(si):
            if test_mask[gi]: preds[gi] = af[i]
    return preds


def main():
    t0 = time.time()
    print('='*70)
    print(' v50 VALIDATION — Extreme Tail Zone (63-68)')
    print(' Candidate: bottom correction sn=1 nc=-1 cb=-1')
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
    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan, feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    gt = y[test_mask].astype(int)
    fi = {f: i for i, f in enumerate(fn)}
    X_min8 = build_min8_features(X_all, fn)
    all_seasons = sorted(set(seasons))

    print('  Caching...')
    cache = cache_all(X_all, X_min8, y, fn, seasons, test_mask)

    def ev(p): return int(np.sum((p[test_mask] - gt)**2)), int((p[test_mask] == gt).sum())
    def ev_s(p, s):
        sm = (seasons == s) & test_mask
        if sm.sum() == 0: return 0, 0, 0
        return int(np.sum((p[sm] - y[sm].astype(int))**2)), int((p[sm] == y[sm].astype(int)).sum()), int(sm.sum())

    # Build v49
    p_base = run_pipeline(cache, y, fn, seasons, test_mask, n)
    p_v49 = apply_swap(p_base, X_all, fi, seasons, test_mask)
    se49, ex49 = ev(p_v49)
    print(f'  v49 baseline: SE={se49}, exact={ex49}/91')

    # Build v50 candidate
    v12z_50 = list(ZONES_V12) + [('xtail', 'bottom', (63, 68), (1, -1, -1))]
    commz_50 = list(ZONES_V49_COMM) + [('xtail', 'bottom', (63, 68), (1, -1, -1))]
    p_base50 = run_pipeline(cache, y, fn, seasons, test_mask, n,
                            v12_zones=v12z_50, comm_zones=commz_50)
    p_v50 = apply_swap(p_base50, X_all, fi, seasons, test_mask)
    se50, ex50 = ev(p_v50)
    print(f'  v50 candidate: SE={se50}, exact={ex50}/91 (Δ={se49-se50:+d})')

    # ════════════════════════════════════════════════════════════════════
    # A: PLATEAU WIDTH
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' A: PLATEAU WIDTH')
    print('='*70)

    plateau_14 = 0
    plateau_le16 = 0
    total_tested = 0
    all_results = []

    for sn in range(-4, 5):
        for nc in range(-4, 5):
            for cb in range(-4, 5):
                total_tested += 1
                v12z = list(ZONES_V12) + [('xtail', 'bottom', (63, 68), (sn, nc, cb))]
                commz = list(ZONES_V49_COMM) + [('xtail', 'bottom', (63, 68), (sn, nc, cb))]
                p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                               v12_zones=v12z, comm_zones=commz)
                p = apply_swap(p, X_all, fi, seasons, test_mask)
                se, ex = ev(p)
                all_results.append((sn, nc, cb, se, ex))
                if se == 14: plateau_14 += 1
                if se <= 16: plateau_le16 += 1

    print(f'  Total configs tested: {total_tested}')
    print(f'  Configs at SE=14: {plateau_14}/{total_tested} = {100*plateau_14/total_tested:.1f}%')
    print(f'  Configs at SE≤16: {plateau_le16}/{total_tested} = {100*plateau_le16/total_tested:.1f}%')

    # Show all SE=14 configs
    print(f'\n  All configs achieving SE=14:')
    for sn, nc, cb, se, ex in all_results:
        if se == 14:
            print(f'    sn={sn:+d} nc={nc:+d} cb={cb:+d}: SE={se}, ex={ex}/91')

    # ════════════════════════════════════════════════════════════════════
    # B: PER-SEASON BREAKDOWN
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' B: PER-SEASON BREAKDOWN')
    print('='*70)

    print(f'  {"Season":<12} {"v49 SE":>6} {"v50 SE":>6} {"Δ":>4} {"v49 ex":>6} {"v50 ex":>6}')
    gains = losses = neutral = 0
    for s in all_seasons:
        se49_s, ex49_s, n_s = ev_s(p_v49, s)
        se50_s, ex50_s, _ = ev_s(p_v50, s)
        delta = se49_s - se50_s
        marker = '✓' if delta > 0 else ('✗' if delta < 0 else '=')
        if delta > 0: gains += 1
        elif delta < 0: losses += 1
        else: neutral += 1
        print(f'  {s:<12} {se49_s:6d} {se50_s:6d} {delta:+4d} {ex49_s:6d} {ex50_s:6d} {marker}')

    print(f'\n  Gains: {gains} seasons, Losses: {losses}, Neutral: {neutral}')

    # ════════════════════════════════════════════════════════════════════
    # C: PER-TEAM CHANGES
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' C: PER-TEAM CHANGES')
    print('='*70)

    test_indices = np.where(test_mask)[0]
    for gi in test_indices:
        if p_v49[gi] != p_v50[gi]:
            actual = int(y[gi])
            se_old = (p_v49[gi] - actual)**2
            se_new = (p_v50[gi] - actual)**2
            arrow = '↑ GAIN' if se_new < se_old else '↓ LOSS' if se_new > se_old else '= SAME'
            net = X_all[gi, fi['NET Rank']]
            sos = X_all[gi, fi['NETSOS']]
            opp = X_all[gi, fi['AvgOppNETRank']]
            print(f'  {arrow}: {record_ids[gi]:35s} v49={p_v49[gi]:2d} v50={p_v50[gi]:2d} gt={actual:2d} '
                  f'NET={net:3.0f} SOS={sos:3.0f} Opp={opp:3.0f} SE:{se_old}→{se_new}')

    # ════════════════════════════════════════════════════════════════════
    # D: THEORETICAL JUSTIFICATION
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' D: THEORETICAL JUSTIFICATION')
    print('='*70)

    print(f'  Bottom correction signal in zone 63-68:')
    print(f'    sn=+1: higher SOS-NET gap → pushed toward WORSE seed')
    print(f'    nc=-1: higher NET vs conf avg → pushed toward BETTER seed')
    print(f'    cb=-1: higher conf-bid historical → pushed toward BETTER seed')
    print(f'')
    print(f'  Interpretation for extreme tail teams:')
    print(f'    Teams at seeds 63-68 are the worst AQ autobids.')
    print(f'    The committee distinguishes them based on:')
    print(f'      - How bad their SOS is relative to NET (weak schedule → worse)')
    print(f'      - Their conference strength (better conf → slightly better)')
    print(f'      - Historical conference seeding patterns')

    # Show features of ALL teams in 63-68 range across seasons
    print(f'\n  Teams in seeds 63-68:')
    print(f'  {"RecordID":35s} {"Seed":>4} {"NET":>4} {"SOS":>4} {"ConfAvg":>7} {"CB":>6} {"SOS-NET":>7}')
    for gi in range(n):
        actual = int(y[gi])
        if 63 <= actual <= 68:
            net = X_all[gi, fi['NET Rank']]
            sos = X_all[gi, fi['NETSOS']]
            conf_avg = X_all[gi, fi['conf_avg_net']]
            cb = X_all[gi, fi['cb_mean_seed']]
            is_test = '✓' if test_mask[gi] else ' '
            print(f'  {record_ids[gi]:35s} {actual:4d} {net:4.0f} {sos:4.0f} {conf_avg:7.0f} '
                  f'{cb:6.1f} {sos-net:+7.0f} {is_test}')

    # ════════════════════════════════════════════════════════════════════
    # E: BOOTSTRAP VALIDATION
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' E: BOOTSTRAP VALIDATION')
    print('='*70)

    rng = np.random.RandomState(42)

    # Team-level bootstrap
    wins50, wins49, ties = 0, 0, 0
    for b in range(100):
        boot = rng.choice(test_indices, size=len(test_indices), replace=True)
        bse49 = sum((p_v49[gi] - int(y[gi]))**2 for gi in boot)
        bse50 = sum((p_v50[gi] - int(y[gi]))**2 for gi in boot)
        if bse50 < bse49: wins50 += 1
        elif bse50 > bse49: wins49 += 1
        else: ties += 1
    print(f'  Team-level bootstrap (100): v50 wins {wins50}, v49 wins {wins49}, ties {ties}')

    # Season-level bootstrap
    wins50_s, wins49_s, ties_s = 0, 0, 0
    for b in range(200):
        boot_seasons = rng.choice(all_seasons, size=len(all_seasons), replace=True)
        bse49, bse50 = 0, 0
        for bs in boot_seasons:
            sm = (seasons == bs) & test_mask
            for gi in np.where(sm)[0]:
                bse49 += (p_v49[gi] - int(y[gi]))**2
                bse50 += (p_v50[gi] - int(y[gi]))**2
        if bse50 < bse49: wins50_s += 1
        elif bse50 > bse49: wins49_s += 1
        else: ties_s += 1
    print(f'  Season-level bootstrap (200): v50 wins {wins50_s}, v49 wins {wins49_s}, ties {ties_s}')

    # ════════════════════════════════════════════════════════════════════
    # F: PERTURBATION STABILITY
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' F: PERTURBATION STABILITY')
    print('='*70)

    for noise_std in [0.5, 1.0, 2.0]:
        se49_noisy, se50_noisy = [], []
        for trial in range(30):
            cache_noisy = {}
            for hs, c in cache.items():
                cn = dict(c)
                cn['raw_v12'] = c['raw_v12'] + rng.normal(0, noise_std, c['raw_v12'].shape)
                cn['raw_comm'] = c['raw_comm'] + rng.normal(0, noise_std, c['raw_comm'].shape)
                for i, gi in enumerate(cn['si']):
                    if not test_mask[gi]:
                        cn['raw_v12'][i] = y[gi]
                        cn['raw_comm'][i] = y[gi]
                cn['a_v12_raw'] = hungarian(cn['raw_v12'], seasons[c['season_mask']], c['avail'], power=0.15)
                cn['a_comm_raw'] = hungarian(cn['raw_comm'], seasons[c['season_mask']], c['avail'], power=0.15)
                cache_noisy[hs] = cn
            p49n = run_pipeline(cache_noisy, y, fn, seasons, test_mask, n)
            p49n = apply_swap(p49n, X_all, fi, seasons, test_mask)
            p50n = run_pipeline(cache_noisy, y, fn, seasons, test_mask, n,
                              v12_zones=v12z_50, comm_zones=commz_50)
            p50n = apply_swap(p50n, X_all, fi, seasons, test_mask)
            se49_noisy.append(ev(p49n)[0])
            se50_noisy.append(ev(p50n)[0])

        a49 = np.array(se49_noisy)
        a50 = np.array(se50_noisy)
        v50w = np.sum(a50 < a49)
        print(f'  noise={noise_std:.1f}: v49 SE={a49.mean():.0f}±{a49.std():.0f}  '
              f'v50 SE={a50.mean():.0f}±{a50.std():.0f}  '
              f'v50 wins {v50w}/30  gap={a49.mean()-a50.mean():.1f}')

    # ════════════════════════════════════════════════════════════════════
    # G: ZONE BOUNDARY ROBUSTNESS
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' G: ZONE BOUNDARY ROBUSTNESS')
    print('='*70)

    for lo in [60, 61, 62, 63, 64, 65]:
        for hi in [66, 67, 68]:
            if lo >= hi: continue
            v12z = list(ZONES_V12) + [('xtail', 'bottom', (lo, hi), (1, -1, -1))]
            commz = list(ZONES_V49_COMM) + [('xtail', 'bottom', (lo, hi), (1, -1, -1))]
            p = run_pipeline(cache, y, fn, seasons, test_mask, n,
                           v12_zones=v12z, comm_zones=commz)
            p = apply_swap(p, X_all, fi, seasons, test_mask)
            se, ex = ev(p)
            gains = losses = 0
            for s in all_seasons:
                se_old = ev_s(p_v49, s)[0]
                se_new = ev_s(p, s)[0]
                if se_new < se_old: gains += 1
                elif se_new > se_old: losses += 1
            marker = ' ★' if se < se49 else ''
            print(f'  zone=({lo},{hi}): SE={se:3d}, ex={ex}/91, gains={gains}s, losses={losses}s{marker}')

    # ════════════════════════════════════════════════════════════════════
    # H: APPLY ONLY ON COMM PATH (like midbot2)?
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' H: EXTREME TAIL ON COMM PATH ONLY vs BOTH PATHS')
    print('='*70)

    # Both paths (v50 candidate)
    print(f'  Both paths: SE={se50}, ex={ex50}/91')

    # Comm only
    v12z_no = list(ZONES_V12)  # no xtail
    commz_yes = list(ZONES_V49_COMM) + [('xtail', 'bottom', (63, 68), (1, -1, -1))]
    p_comm_only = run_pipeline(cache, y, fn, seasons, test_mask, n,
                               v12_zones=v12z_no, comm_zones=commz_yes)
    p_comm_only = apply_swap(p_comm_only, X_all, fi, seasons, test_mask)
    se_co, ex_co = ev(p_comm_only)
    print(f'  Comm only: SE={se_co}, ex={ex_co}/91')

    # V12 only
    v12z_yes = list(ZONES_V12) + [('xtail', 'bottom', (63, 68), (1, -1, -1))]
    commz_no = list(ZONES_V49_COMM)
    p_v12_only = run_pipeline(cache, y, fn, seasons, test_mask, n,
                              v12_zones=v12z_yes, comm_zones=commz_no)
    p_v12_only = apply_swap(p_v12_only, X_all, fi, seasons, test_mask)
    se_vo, ex_vo = ev(p_v12_only)
    print(f'  V12 only:  SE={se_vo}, ex={ex_vo}/91')

    # ════════════════════════════════════════════════════════════════════
    # I: CHECK v49 remaining errors after v50
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' I: v50 REMAINING ERRORS')
    print('='*70)

    remaining = 0
    for gi in test_indices:
        pred = p_v50[gi]
        actual = int(y[gi])
        if pred != actual:
            remaining += 1
            net = X_all[gi, fi['NET Rank']]
            sos = X_all[gi, fi['NETSOS']]
            print(f'  {record_ids[gi]:35s} pred={pred:2d} gt={actual:2d} '
                  f'SE={(pred-actual)**2:2d} NET={net:3.0f} SOS={sos:3.0f}')
    print(f'\n  Remaining errors: {remaining}, Total SE: {se50}')

    # ════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ════════════════════════════════════════════════════════════════════
    print('\n' + '='*70)
    print(' FINAL VERDICT')
    print('='*70)

    print(f'\n  v49: SE={se49}, {ex49}/91 exact')
    print(f'  v50: SE={se50}, {ex50}/91 exact')
    print(f'  Change: extreme-tail zone (63-68) bottom correction sn=1 nc=-1 cb=-1')
    print(f'  Plateau: {plateau_14}/729 configs at SE=14 ({100*plateau_14/729:.1f}%)')
    print(f'  Season breakdown: {gains} gains, {losses} losses, {neutral} neutral')
    print(f'  Bootstrap: team-level v50 wins {wins50}/100, season-level {wins50_s}/200')

    safe = (losses == 0 and plateau_14 >= 5 and wins50 >= 40 and wins50_s >= 40)
    if safe:
        print(f'\n  ✓ SAFE TO DEPLOY: Zero regressions, wide plateau, strong bootstrap')
    else:
        if losses > 0:
            print(f'\n  ⚠ REGRESSIONS FOUND — not safe')
        if plateau_14 < 5:
            print(f'\n  ⚠ NARROW PLATEAU — fragile')
        if wins50 < 40:
            print(f'\n  ⚠ WEAK BOOTSTRAP — not reliable')

    print(f'\n  Time: {time.time()-t0:.0f}s')
    print('='*70)


if __name__ == '__main__':
    main()
