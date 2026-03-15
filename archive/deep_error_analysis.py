#!/usr/bin/env python3
"""
Deep error analysis: WHY do LOSO improvements not translate to Kaggle?

Key structural difference:
  LOSO: predict ALL 68 teams in a season → Hungarian assigns all seeds
  Kaggle: predict only TEST teams (~18/season), LOCK training teams to known seeds
          → Hungarian assigns all seeds but training teams are anchored

This creates fundamentally different optimization landscapes:
  - In LOSO, every team competes for every seed
  - In Kaggle, test teams must find seeds NOT occupied by (correctly) locked training teams
  
Analysis:
  1. Per-team errors: which teams are wrong, by how much?
  2. Seed-range patterns: where do errors cluster?
  3. Locked-seed effect: does locking distort nearby predictions?
  4. Season-level difficulty: which seasons are hardest?
  5. Feature analysis: what distinguishes right vs wrong teams?
  6. LOSO vs Kaggle disagreement: which teams change between eval modes?
  7. Ceiling analysis: theoretical maximum given our approach
  8. Error direction analysis: do we predict too high or too low?
"""

import os, sys, time
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    build_pairwise_data, predict_robust_blend, hungarian, pairwise_score,
    build_pairwise_data_adjacent,
    USE_TOP_K_A, FORCE_FEATURES, HUNGARIAN_POWER, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_POWER = 0.15


def main():
    t0 = time.time()
    print('=' * 70)
    print(' DEEP ERROR ANALYSIS: LOSO vs KAGGLE DISCONNECT')
    print('=' * 70)

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
    teams = labeled['Team'].values.astype(str)
    conferences = labeled['Conference'].values.astype(str)
    bid_types = labeled['Bid Type'].fillna('').values.astype(str)
    folds = sorted(set(seasons))

    X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                     feat.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_all = imp.fit_transform(X_raw)

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    train_mask = ~test_mask

    print(f'\n  Total teams: {n_labeled}')
    print(f'  Training teams: {train_mask.sum()} (seeds locked in Kaggle)')
    print(f'  Test teams: {test_mask.sum()} (predicted in Kaggle)')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 1: GENERATE BOTH LOSO AND KAGGLE PREDICTIONS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 1: GENERATING LOSO vs KAGGLE PREDICTIONS')
    print('=' * 70)

    loso_assigned = np.zeros(n_labeled, dtype=int)
    kaggle_assigned = np.zeros(n_labeled, dtype=int)
    # Also store raw pairwise scores before Hungarian
    loso_raw_scores = np.zeros(n_labeled)
    kaggle_raw_scores = np.zeros(n_labeled)

    for hold in folds:
        season_mask = (seasons == hold)
        season_indices = np.where(season_mask)[0]
        n_season = season_mask.sum()

        # --- LOSO mode: train on other seasons, predict ALL teams in held-out season ---
        tr_loso = ~season_mask
        top_k_idx = select_top_k_features(
            X_all[tr_loso], y[tr_loso], feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        loso_pred = predict_robust_blend(
            X_all[tr_loso], y[tr_loso], X_all[season_mask],
            seasons[tr_loso], top_k_idx)
        loso_raw_scores[season_mask] = loso_pred
        avail = {hold: list(range(1, 69))}
        loso_season = hungarian(loso_pred, seasons[season_mask], avail, power=HUNGARIAN_POWER)
        loso_assigned[season_mask] = loso_season

        # --- Kaggle mode: train on everything except THIS season's test teams ---
        season_test_mask = test_mask & season_mask
        global_train_mask = ~season_test_mask  # includes training teams from this season!
        top_k_idx_k = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        kaggle_pred = predict_robust_blend(
            X_all[global_train_mask], y[global_train_mask],
            X_all[season_mask], seasons[global_train_mask], top_k_idx_k)

        # Lock training teams to known seeds
        for i, global_idx in enumerate(season_indices):
            if not test_mask[global_idx]:
                kaggle_pred[i] = y[global_idx]
        
        kaggle_raw_scores[season_mask] = kaggle_pred
        kaggle_season = hungarian(kaggle_pred, seasons[season_mask], avail, power=KAGGLE_POWER)
        
        for i, global_idx in enumerate(season_indices):
            kaggle_assigned[global_idx] = kaggle_season[i]

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 2: OVERALL COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 2: OVERALL LOSO vs KAGGLE COMPARISON')
    print('=' * 70)

    y_int = y.astype(int)

    # LOSO stats (all teams)
    loso_exact_all = int((loso_assigned == y_int).sum())
    loso_rmse_all = np.sqrt(np.mean((loso_assigned - y_int)**2))

    # LOSO stats (test teams only)
    loso_exact_test = int((loso_assigned[test_mask] == y_int[test_mask]).sum())
    loso_rmse_test = np.sqrt(np.mean((loso_assigned[test_mask] - y_int[test_mask])**2))

    # LOSO stats (train teams only)
    loso_exact_train = int((loso_assigned[train_mask] == y_int[train_mask]).sum())
    loso_rmse_train = np.sqrt(np.mean((loso_assigned[train_mask] - y_int[train_mask])**2))

    # Kaggle stats (test teams)
    kaggle_exact = int((kaggle_assigned[test_mask] == y_int[test_mask]).sum())
    kaggle_rmse = np.sqrt(np.mean((kaggle_assigned[test_mask] - y_int[test_mask])**2))

    # Kaggle stats (train teams — should be perfect since locked)
    kaggle_exact_train = int((kaggle_assigned[train_mask] == y_int[train_mask]).sum())

    print(f'\n  {"Metric":<30} {"LOSO(all)":<14} {"LOSO(test)":<14} {"LOSO(train)":<14} {"Kaggle(test)":<14}')
    print(f'  {"─"*30} {"─"*14} {"─"*14} {"─"*14} {"─"*14}')
    print(f'  {"Exact matches":<30} {loso_exact_all:>4}/{n_labeled:<8} {loso_exact_test:>4}/{test_mask.sum():<8} {loso_exact_train:>4}/{train_mask.sum():<8} {kaggle_exact:>4}/{test_mask.sum():<8}')
    print(f'  {"Exact %":<30} {loso_exact_all/n_labeled*100:>6.1f}%{"":6} {loso_exact_test/test_mask.sum()*100:>6.1f}%{"":6} {loso_exact_train/train_mask.sum()*100:>6.1f}%{"":6} {kaggle_exact/test_mask.sum()*100:>6.1f}%{"":6}')
    print(f'  {"RMSE":<30} {loso_rmse_all:>8.4f}{"":5} {loso_rmse_test:>8.4f}{"":5} {loso_rmse_train:>8.4f}{"":5} {kaggle_rmse:>8.4f}{"":5}')
    print(f'  {"Train teams perfect?":<30} {"":14} {"":14} {kaggle_exact_train:>4}/{train_mask.sum():<8} {"(locked)":14}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 3: ARE TEST TEAMS HARDER THAN TRAINING TEAMS?
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 3: ARE TEST TEAMS SYSTEMATICALLY DIFFERENT?')
    print('=' * 70)

    # Compare seed distributions
    test_seeds = y_int[test_mask]
    train_seeds = y_int[train_mask]

    print(f'\n  Seed statistics:')
    print(f'  {"Metric":<20} {"Train teams":<15} {"Test teams":<15}')
    print(f'  {"─"*20} {"─"*15} {"─"*15}')
    print(f'  {"Mean seed":<20} {train_seeds.mean():>8.1f}{"":6} {test_seeds.mean():>8.1f}')
    print(f'  {"Median seed":<20} {np.median(train_seeds):>8.1f}{"":6} {np.median(test_seeds):>8.1f}')
    print(f'  {"Std seed":<20} {train_seeds.std():>8.1f}{"":6} {test_seeds.std():>8.1f}')

    # Seed range distribution
    ranges = [(1, 4, 'Top 4 (1-4)'), (5, 16, 'Upper (5-16)'), (17, 34, 'Mid (17-34)'),
              (35, 51, 'Lower-mid (35-51)'), (52, 68, 'Bottom (52-68)')]
    print(f'\n  Seed range distribution:')
    print(f'  {"Range":<20} {"Train":<10} {"Test":<10} {"Test %":<10}')
    print(f'  {"─"*20} {"─"*10} {"─"*10} {"─"*10}')
    for lo, hi, name in ranges:
        n_tr = int(((train_seeds >= lo) & (train_seeds <= hi)).sum())
        n_te = int(((test_seeds >= lo) & (test_seeds <= hi)).sum())
        pct_te = n_te / test_mask.sum() * 100
        print(f'  {name:<20} {n_tr:>5}{"":4} {n_te:>5}{"":4} {pct_te:>5.1f}%')

    # Feature comparison
    print(f'\n  Key feature differences (test vs train, among tournament teams):')
    key_feats = ['NET Rank', 'NETSOS', 'WL_Pct', 'power_rating', 'resume_score',
                 'is_power_conf', 'is_AL', 'is_AQ', 'conf_avg_net']
    print(f'  {"Feature":<20} {"Train mean":<12} {"Test mean":<12} {"Diff":<10}')
    print(f'  {"─"*20} {"─"*12} {"─"*12} {"─"*10}')
    for fname in key_feats:
        fidx = feature_names.index(fname) if fname in feature_names else None
        if fidx is not None:
            tr_mean = X_all[train_mask, fidx].mean()
            te_mean = X_all[test_mask, fidx].mean()
            diff = te_mean - tr_mean
            print(f'  {fname:<20} {tr_mean:>10.2f}  {te_mean:>10.2f}  {diff:>+8.2f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 4: PER-SEASON ERROR BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 4: PER-SEASON DETAILED BREAKDOWN')
    print('=' * 70)

    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        n_train_s = (train_mask & sm).sum()

        if n_test_s == 0:
            continue

        loso_ex = int((loso_assigned[stm] == y_int[stm]).sum())
        kaggle_ex = int((kaggle_assigned[stm] == y_int[stm]).sum())
        loso_rm = np.sqrt(np.mean((loso_assigned[stm] - y_int[stm])**2))
        kaggle_rm = np.sqrt(np.mean((kaggle_assigned[stm] - y_int[stm])**2))

        # How many training teams were displaced by locking?
        kaggle_train_correct = int((kaggle_assigned[train_mask & sm] == y_int[train_mask & sm]).sum())
        
        print(f'\n  ─── {hold} ({n_train_s} train + {n_test_s} test = {sm.sum()} total) ───')
        print(f'  LOSO on test teams:   {loso_ex}/{n_test_s} exact ({loso_ex/n_test_s*100:.0f}%), RMSE={loso_rm:.3f}')
        print(f'  Kaggle on test teams: {kaggle_ex}/{n_test_s} exact ({kaggle_ex/n_test_s*100:.0f}%), RMSE={kaggle_rm:.3f}')
        print(f'  Kaggle train locked:  {kaggle_train_correct}/{n_train_s} exact (should be {n_train_s}/{n_train_s})')

        # Detail each test team
        print(f'\n  {"Team":<25} {"Conf":<10} {"Bid":<4} {"NET":>4} {"True":>5} {"LOSO":>5} {"Kaggle":>6} {"LOSO_e":>6} {"Kag_e":>6}')
        print(f'  {"─"*25} {"─"*10} {"─"*4} {"─"*4} {"─"*5} {"─"*5} {"─"*6} {"─"*6} {"─"*6}')
        
        test_indices_s = np.where(stm)[0]
        # Sort by true seed
        order = np.argsort(y_int[test_indices_s])
        for idx in test_indices_s[order]:
            true_s = y_int[idx]
            loso_s = loso_assigned[idx]
            kaggle_s = kaggle_assigned[idx]
            loso_e = loso_s - true_s
            kaggle_e = kaggle_s - true_s
            net_val = X_all[idx, feature_names.index('NET Rank')]
            l_mark = '✓' if loso_e == 0 else f'{loso_e:+d}'
            k_mark = '✓' if kaggle_e == 0 else f'{kaggle_e:+d}'
            print(f'  {teams[idx]:<25} {conferences[idx]:<10} {bid_types[idx]:<4} {net_val:>4.0f} {true_s:>5} {loso_s:>5} {kaggle_s:>6} {l_mark:>6} {k_mark:>6}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 5: ERROR PATTERNS BY SEED RANGE
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 5: ERROR PATTERNS BY SEED RANGE (TEST TEAMS ONLY)')
    print('=' * 70)

    for lo, hi, name in ranges:
        mask = test_mask & (y_int >= lo) & (y_int <= hi)
        n = mask.sum()
        if n == 0:
            continue
        k_ex = int((kaggle_assigned[mask] == y_int[mask]).sum())
        k_rmse = np.sqrt(np.mean((kaggle_assigned[mask] - y_int[mask])**2))
        k_mean_err = (kaggle_assigned[mask] - y_int[mask]).mean()
        l_ex = int((loso_assigned[mask] == y_int[mask]).sum())
        l_rmse = np.sqrt(np.mean((loso_assigned[mask] - y_int[mask])**2))
        l_mean_err = (loso_assigned[mask] - y_int[mask]).mean()
        print(f'\n  {name} (n={n}):')
        print(f'    Kaggle: {k_ex}/{n} exact ({k_ex/n*100:.0f}%), RMSE={k_rmse:.3f}, mean_err={k_mean_err:+.2f}')
        print(f'    LOSO:   {l_ex}/{n} exact ({l_ex/n*100:.0f}%), RMSE={l_rmse:.3f}, mean_err={l_mean_err:+.2f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 6: LOSO vs KAGGLE DISAGREEMENT ON TEST TEAMS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 6: LOSO vs KAGGLE DISAGREEMENT (test teams only)')
    print('=' * 70)

    # Teams where LOSO and Kaggle give different seeds
    test_indices = np.where(test_mask)[0]
    agree = 0
    loso_right_kaggle_wrong = []
    kaggle_right_loso_wrong = []
    both_wrong = []
    both_right = 0

    for idx in test_indices:
        loso_correct = (loso_assigned[idx] == y_int[idx])
        kaggle_correct = (kaggle_assigned[idx] == y_int[idx])
        loso_kaggle_same = (loso_assigned[idx] == kaggle_assigned[idx])
        
        if loso_kaggle_same:
            agree += 1
        
        if loso_correct and kaggle_correct:
            both_right += 1
        elif loso_correct and not kaggle_correct:
            loso_right_kaggle_wrong.append(idx)
        elif not loso_correct and kaggle_correct:
            kaggle_right_loso_wrong.append(idx)
        else:
            both_wrong.append(idx)

    n_test = test_mask.sum()
    print(f'\n  Agreement: {agree}/{n_test} test teams get same seed in both modes ({agree/n_test*100:.0f}%)')
    print(f'  Both correct: {both_right}')
    print(f'  LOSO correct, Kaggle wrong: {len(loso_right_kaggle_wrong)}')
    print(f'  Kaggle correct, LOSO wrong: {len(kaggle_right_loso_wrong)}')
    print(f'  Both wrong: {len(both_wrong)}')

    if loso_right_kaggle_wrong:
        print(f'\n  Teams where LOSO is right but Kaggle is WRONG (locking hurts):')
        print(f'  {"Team":<25} {"Season":<10} {"True":>5} {"LOSO":>5} {"Kaggle":>6}')
        for idx in loso_right_kaggle_wrong:
            print(f'  {teams[idx]:<25} {seasons[idx]:<10} {y_int[idx]:>5} {loso_assigned[idx]:>5} {kaggle_assigned[idx]:>6}')

    if kaggle_right_loso_wrong:
        print(f'\n  Teams where Kaggle is right but LOSO is WRONG (locking helps):')
        print(f'  {"Team":<25} {"Season":<10} {"True":>5} {"LOSO":>5} {"Kaggle":>6}')
        for idx in kaggle_right_loso_wrong:
            print(f'  {teams[idx]:<25} {seasons[idx]:<10} {y_int[idx]:>5} {loso_assigned[idx]:>5} {kaggle_assigned[idx]:>6}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 7: THE LOCKED-SEED CONSTRAINT EFFECT
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 7: LOCKED-SEED CONSTRAINT ANALYSIS')
    print('=' * 70)
    
    # For each season, analyze what happens when we lock training seeds
    # Key question: do locked seeds "push" test teams to wrong seeds?
    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        if n_test_s == 0:
            continue
        
        # Which seeds are occupied by training teams?
        train_seeds_s = set(y_int[train_mask & sm].tolist())
        test_true_seeds = y_int[stm]
        available_for_test = set(range(1, 69)) - train_seeds_s
        
        # How many test teams have their TRUE seed available?
        true_available = sum(1 for s in test_true_seeds if s in available_for_test)
        
        # How many test teams' true seeds are BLOCKED by training teams?
        true_blocked = n_test_s - true_available
        
        # For blocked teams, what's the nearest available seed?
        blocked_info = []
        for idx in np.where(stm)[0]:
            true_s = y_int[idx]
            if true_s not in available_for_test:
                # This team's true seed is occupied by a training team
                nearest = min(available_for_test, key=lambda x: abs(x - true_s))
                blocked_info.append((teams[idx], true_s, nearest, abs(nearest - true_s)))
        
        print(f'\n  ─── {hold}: {n_test_s} test teams, {len(train_seeds_s)} training seeds locked ───')
        print(f'  Available seeds for test: {len(available_for_test)}/{68}')
        print(f'  Test teams with true seed available: {true_available}/{n_test_s}')
        print(f'  Test teams with true seed BLOCKED: {true_blocked}/{n_test_s}')
        
        if blocked_info:
            print(f'  Blocked teams:')
            for team, true_s, nearest, dist in sorted(blocked_info, key=lambda x: -x[3]):
                print(f'    {team:<25} true_seed={true_s:>3} nearest_available={nearest:>3} (gap={dist})')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 8: RAW PAIRWISE SCORE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 8: RAW SCORE vs HUNGARIAN ASSIGNMENT')
    print('=' * 70)

    # In LOSO, raw scores represent relative ordering within the season
    # Let's see how well raw scores correlate with true seeds
    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        if n_test_s == 0:
            continue
        
        # LOSO raw scores for ALL teams in season
        season_loso_raw = loso_raw_scores[sm]
        season_y = y_int[sm]
        rho_loso, _ = spearmanr(season_loso_raw, season_y)
        
        # Kaggle raw scores for test teams only (after locking)
        # The scores we used include locked values, so test team scores
        # are the pairwise ranking among ALL teams including locked ones
        test_kaggle_raw = kaggle_raw_scores[stm]
        test_y = y_int[stm]
        rho_kaggle, _ = spearmanr(test_kaggle_raw, test_y)
        
        print(f'\n  {hold}: Spearman(raw_score, true_seed)')
        print(f'    LOSO (all teams):  ρ = {rho_loso:.4f}')
        print(f'    Kaggle (test only): ρ = {rho_kaggle:.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 9: COMPONENT-LEVEL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 9: COMPONENT-LEVEL ANALYSIS ON TEST TEAMS')
    print('=' * 70)
    print('  Evaluating each component independently on Kaggle-style eval...')

    comp_results = {}
    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        if n_test_s == 0:
            continue
        
        season_indices = np.where(sm)[0]
        season_test_mask_local = test_mask[sm]  # within season
        global_train_mask = ~(test_mask & (seasons == hold))
        
        top_k_idx = select_top_k_features(
            X_all[global_train_mask], y[global_train_mask],
            feature_names, k=USE_TOP_K_A,
            forced_features=FORCE_FEATURES)[0]
        
        X_train_comp = X_all[global_train_mask]
        y_train_comp = y[global_train_mask]
        seasons_train_comp = seasons[global_train_mask]
        X_season = X_all[sm]
        
        # Component 1: LR C=5.0 adj-pairs
        pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
            X_train_comp, y_train_comp, seasons_train_comp, max_gap=ADJ_COMP1_GAP)
        sc_adj = StandardScaler()
        pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
        lr1 = LogisticRegression(C=PW_C1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_adj_sc, pw_y_adj)
        score1 = pairwise_score(lr1, X_season, sc_adj)
        
        # Component 3: LR C=0.5 topK
        X_tr_k = X_train_comp[:, top_k_idx]
        X_s_k = X_season[:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_train_comp, seasons_train_comp)
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=PW_C3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        score3 = pairwise_score(lr3, X_s_k, sc_k)
        
        # Component 4: XGB
        pw_X_full, pw_y_full = build_pairwise_data(X_train_comp, y_train_comp, seasons_train_comp)
        sc_full = StandardScaler()
        pw_X_full_sc = sc_full.fit_transform(pw_X_full)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric='logloss')
        xgb_clf.fit(pw_X_full_sc, pw_y_full)
        score4 = pairwise_score(xgb_clf, X_season, sc_full)

        # Evaluate each component separately (with locking)
        for comp_name, comp_scores in [('comp1_LR_C5_adj', score1),
                                        ('comp3_LR_C05_topK', score3),
                                        ('comp4_XGB', score4),
                                        ('blend_v12', BLEND_W1*score1 + BLEND_W3*score3 + BLEND_W4*score4)]:
            locked_scores = comp_scores.copy()
            for i, global_idx in enumerate(season_indices):
                if not test_mask[global_idx]:
                    locked_scores[i] = y[global_idx]
            
            avail = {hold: list(range(1, 69))}
            assigned_s = hungarian(locked_scores, seasons[sm], avail, power=KAGGLE_POWER)
            
            test_pred = assigned_s[season_test_mask_local]
            test_true = y_int[stm]
            ex = int((test_pred == test_true).sum())
            rm = np.sqrt(np.mean((test_pred - test_true)**2))
            
            if comp_name not in comp_results:
                comp_results[comp_name] = {'exact': 0, 'total': 0, 'errors': []}
            comp_results[comp_name]['exact'] += ex
            comp_results[comp_name]['total'] += n_test_s
            comp_results[comp_name]['errors'].extend((test_pred - test_true).tolist())

    print(f'\n  {"Component":<25} {"Exact":>10} {"RMSE":>8}')
    print(f'  {"─"*25} {"─"*10} {"─"*8}')
    for cname in ['comp1_LR_C5_adj', 'comp3_LR_C05_topK', 'comp4_XGB', 'blend_v12']:
        r = comp_results[cname]
        rmse = np.sqrt(np.mean(np.array(r['errors'])**2))
        print(f'  {cname:<25} {r["exact"]:>4}/{r["total"]:<5} {rmse:>8.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 10: CEILING ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 10: CEILING ANALYSIS — THEORETICAL BEST')
    print('=' * 70)

    # Given the locked-seed constraint, what's the BEST possible score?
    # If we had a PERFECT ordering of test teams, would locking still cause errors?
    
    total_perfect_possible = 0
    total_blocked = 0
    
    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        if n_test_s == 0:
            continue
        
        train_seeds_s = set(y_int[train_mask & sm].tolist())
        test_true_seeds = y_int[stm].tolist()
        
        # With perfect ordering, can Hungarian assign all test teams correctly?
        # Only if their true seed isn't taken by a training team
        available = set(range(1, 69)) - train_seeds_s
        perfect = sum(1 for s in test_true_seeds if s in available)
        total_perfect_possible += perfect
        total_blocked += (n_test_s - perfect)
        
        print(f'  {hold}: {perfect}/{n_test_s} test teams CAN be exact with perfect model (ceiling)')
        print(f'         {n_test_s - perfect} teams have their true seed blocked by training team')

    print(f'\n  OVERALL CEILING: {total_perfect_possible}/{test_mask.sum()} '
          f'({total_perfect_possible/test_mask.sum()*100:.1f}%) — absolute maximum possible')
    print(f'  Currently achieving: {kaggle_exact}/{test_mask.sum()} '
          f'({kaggle_exact/test_mask.sum()*100:.1f}%)')
    print(f'  Remaining gap: {total_perfect_possible - kaggle_exact} teams '
          f'(could improve by {(total_perfect_possible - kaggle_exact)/test_mask.sum()*100:.1f}%)')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 11: ERROR DIRECTION & MAGNITUDE
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 11: ERROR DIRECTION & MAGNITUDE')
    print('=' * 70)

    errors = kaggle_assigned[test_mask] - y_int[test_mask]
    
    print(f'\n  Error distribution (Kaggle, test teams):')
    print(f'    Mean error:      {errors.mean():+.3f} (positive = predicted too low = seed too high)')
    print(f'    Median error:    {np.median(errors):+.1f}')
    print(f'    Mean abs error:  {np.abs(errors).mean():.3f}')
    print(f'    Max overpredict: {errors.min():+d} (predicted much better than actual)')
    print(f'    Max underpredict:{errors.max():+d} (predicted much worse than actual)')

    # Error histogram
    err_counts = Counter(errors.tolist())
    print(f'\n  Error histogram:')
    print(f'  {"Error":>6} {"Count":>6} {"Bar":<40}')
    for e in sorted(err_counts.keys()):
        bar = '█' * err_counts[e]
        print(f'  {e:>+6d} {err_counts[e]:>6d} {bar}')

    # Biggest absolute errors
    abs_errors = np.abs(errors)
    worst_indices = test_indices[np.argsort(-abs_errors)[:15]]
    
    print(f'\n  15 WORST predictions (Kaggle):')
    print(f'  {"Team":<25} {"Season":<10} {"NET":>4} {"True":>5} {"Pred":>5} {"Err":>5}')
    print(f'  {"─"*25} {"─"*10} {"─"*4} {"─"*5} {"─"*5} {"─"*5}')
    for idx in worst_indices:
        net_val = X_all[idx, feature_names.index('NET Rank')]
        print(f'  {teams[idx]:<25} {seasons[idx]:<10} {net_val:>4.0f} {y_int[idx]:>5} {kaggle_assigned[idx]:>5} {kaggle_assigned[idx]-y_int[idx]:>+5d}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 12: BID TYPE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 12: ERROR BY BID TYPE')
    print('=' * 70)

    for bt in ['AL', 'AQ']:
        bt_mask = test_mask & (bid_types == bt)
        n_bt = bt_mask.sum()
        if n_bt == 0:
            continue
        ex = int((kaggle_assigned[bt_mask] == y_int[bt_mask]).sum())
        rm = np.sqrt(np.mean((kaggle_assigned[bt_mask] - y_int[bt_mask])**2))
        me = (kaggle_assigned[bt_mask] - y_int[bt_mask]).mean()
        print(f'\n  {bt} teams (n={n_bt}): {ex}/{n_bt} exact ({ex/n_bt*100:.0f}%), RMSE={rm:.3f}, mean_err={me:+.2f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 13: CONFERENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 13: ERROR BY CONFERENCE (test teams)')
    print('=' * 70)

    conf_stats = defaultdict(lambda: {'exact': 0, 'total': 0, 'errors': []})
    for idx in test_indices:
        conf = conferences[idx]
        correct = kaggle_assigned[idx] == y_int[idx]
        conf_stats[conf]['total'] += 1
        if correct:
            conf_stats[conf]['exact'] += 1
        conf_stats[conf]['errors'].append(kaggle_assigned[idx] - y_int[idx])

    sorted_confs = sorted(conf_stats.keys(), key=lambda c: -conf_stats[c]['total'])
    print(f'\n  {"Conference":<20} {"Total":>6} {"Exact":>6} {"RMSE":>8} {"MeanErr":>8}')
    print(f'  {"─"*20} {"─"*6} {"─"*6} {"─"*8} {"─"*8}')
    for c in sorted_confs:
        s = conf_stats[c]
        errs = np.array(s['errors'])
        rmse = np.sqrt(np.mean(errs**2))
        me = errs.mean()
        print(f'  {c:<20} {s["total"]:>6} {s["exact"]:>6} {rmse:>8.3f} {me:>+8.2f}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 14: HUNGARIAN POWER SENSITIVITY ON KAGGLE
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 14: HUNGARIAN POWER SENSITIVITY (Kaggle eval)')
    print('=' * 70)

    powers_to_test = [0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.50, 1.0, 1.5, 2.0]
    
    print(f'\n  {"Power":>6} {"Exact":>8} {"RMSE":>8}')
    print(f'  {"─"*6} {"─"*8} {"─"*8}')
    
    for power in powers_to_test:
        test_assigned_p = np.zeros(n_labeled, dtype=int)
        for hold in folds:
            sm = (seasons == hold)
            stm = test_mask & sm
            if stm.sum() == 0:
                continue
            
            season_indices = np.where(sm)[0]
            # Use pre-computed kaggle raw scores
            locked_scores = kaggle_raw_scores[sm].copy()
            for i, global_idx in enumerate(season_indices):
                if not test_mask[global_idx]:
                    locked_scores[i] = y[global_idx]
            
            avail = {hold: list(range(1, 69))}
            assigned_p = hungarian(locked_scores, seasons[sm], avail, power=power)
            for i, global_idx in enumerate(season_indices):
                if test_mask[global_idx]:
                    test_assigned_p[global_idx] = assigned_p[i]
        
        ex = int((test_assigned_p[test_mask] == y_int[test_mask]).sum())
        rm = np.sqrt(np.mean((test_assigned_p[test_mask] - y_int[test_mask])**2))
        mark = ' ← current' if power == KAGGLE_POWER else ''
        print(f'  {power:>6.2f} {ex:>4}/{test_mask.sum():<3} {rm:>8.4f}{mark}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 15: WHAT-IF: NO HUNGARIAN (direct rounding)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 15: RAW SCORE QUALITY (test teams, Kaggle mode)')
    print('=' * 70)
    
    # How well do raw pairwise scores (before Hungarian) rank the test teams?
    for hold in folds:
        sm = (seasons == hold)
        stm = test_mask & sm
        n_test_s = stm.sum()
        if n_test_s == 0:
            continue
        
        # Raw Kaggle scores for test teams
        raw_test = kaggle_raw_scores[stm]
        true_test = y_int[stm]
        
        # What if we just round raw scores?
        rounded = np.round(raw_test).astype(int)
        rounded = np.clip(rounded, 1, 68)
        ex_round = int((rounded == true_test).sum())
        rm_round = np.sqrt(np.mean((rounded - true_test)**2))
        
        rho, _ = spearmanr(raw_test, true_test)
        
        print(f'\n  {hold} (test={n_test_s}):')
        print(f'    Spearman(raw, true): {rho:.4f}')
        print(f'    If rounded directly: {ex_round}/{n_test_s} exact, RMSE={rm_round:.3f}')
        print(f'    After Hungarian:     {int((kaggle_assigned[stm] == true_test).sum())}/{n_test_s} exact, RMSE={np.sqrt(np.mean((kaggle_assigned[stm] - true_test)**2)):.3f}')
        
        # Show raw scores vs true for test teams
        order = np.argsort(true_test)
        test_idx_s = np.where(stm)[0]
        print(f'    {"Team":<22} {"True":>5} {"Raw":>7} {"Round":>5} {"Hung":>5}')
        for ii in order:
            idx = test_idx_s[ii]
            print(f'    {teams[idx]:<22} {true_test[ii]:>5} {raw_test[ii]:>7.2f} {rounded[ii]:>5} {kaggle_assigned[idx]:>5}')

    # ═══════════════════════════════════════════════════════════════
    #  SECTION 16: KEY INSIGHTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' SECTION 16: KEY INSIGHTS SUMMARY')
    print('=' * 70)
    
    print(f'\n  1. LOSO evaluates ALL 68 teams per season')
    print(f'     Kaggle evaluates only {test_mask.sum()} test teams ({test_mask.sum()/n_labeled*100:.0f}% of data)')
    print(f'     with {train_mask.sum()} training teams locked to known seeds.')
    
    print(f'\n  2. Kaggle ceiling (with perfect model): {total_perfect_possible}/{test_mask.sum()} '
          f'({total_perfect_possible/test_mask.sum()*100:.1f}%)')
    print(f'     Current Kaggle: {kaggle_exact}/{test_mask.sum()} ({kaggle_exact/test_mask.sum()*100:.1f}%)')
    
    print(f'\n  3. LOSO-to-Kaggle transfer on test teams:')
    print(f'     LOSO accuracy on test teams:   {loso_exact_test}/{test_mask.sum()} ({loso_exact_test/test_mask.sum()*100:.1f}%)')
    print(f'     Kaggle accuracy on test teams: {kaggle_exact}/{test_mask.sum()} ({kaggle_exact/test_mask.sum()*100:.1f}%)')
    
    print(f'\n  4. Kaggle has MORE training data (locked seeds from same season),')
    print(f'     but the Hungarian assignment is constrained by locked seeds,')
    print(f'     so improvements in ordering test teams may not translate to')
    print(f'     better seed assignments due to blocking.')
    
    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
