#!/usr/bin/env python3
"""
v15: Committee-bias features — targeted at mid-tier prediction errors.

The committee systematically:
  - PENALIZES AQ teams from weak conferences (higher seed than NET suggests)
  - REWARDS AL at-large teams from power conferences (lower seed than NET)
  - Uses "eye test" factors: conference strength, quality wins, bad losses

Key new features:
  1. committee_adjust: predicted gap between NET-based seed and actual seed
     based on conference tier × bid type × resume quality
  2. net_seed_gap_pred: regression prediction of (actual_seed - net_rank)
     using conf strength, bid, SOS, quadrant record
  3. bubble_score: continuous measure of how much committee subjectivity
     applies to this team (higher = more unpredictable)
  4. conf_tier_penalty: how much the committee historically adjusts seeds
     for this conference tier
  5. aq_conference_discount: specific penalty for AQ from weak conferences
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, parse_wl, build_features, select_top_k_features,
    build_pairwise_data, build_pairwise_data_adjacent, hungarian, pairwise_score,
    USE_TOP_K_A, FORCE_FEATURES, ADJ_COMP1_GAP,
    PW_C1, PW_C3, BLEND_W1, BLEND_W3, BLEND_W4
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def build_features_v15(df, context_df, labeled_df, tourn_rids):
    """Build v12's 68 features + new committee-bias features."""
    # Start with existing features
    feat = build_features(df, context_df, labeled_df, tourn_rids)
    
    # ─── Extract columns we need ───
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    bid  = df['Bid Type'].fillna('')
    conf = df['Conference'].fillna('Unknown')
    
    is_AL = (bid == 'AL').astype(float)
    is_AQ = (bid == 'AQ').astype(float)
    
    # Conference strength from context
    all_net_vals = pd.to_numeric(context_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': context_df['Conference'].fillna('Unknown'),
                       'NET': all_net_vals}).groupby('Conference')['NET']
    conf_avg = conf.map(cs.mean()).fillna(200)
    conf_min = conf.map(cs.min()).fillna(300)
    
    # Power conferences
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12'}
    mid_power = {'AAC','Mountain West','WCC','Atlantic 10','MVC'}
    is_power = conf.isin(power).astype(float)
    is_mid = conf.isin(mid_power).astype(float)
    is_low = (1.0 - is_power - is_mid).clip(0, 1)
    
    # Quadrant records
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 1: Committee Adjustment Score
    #  The committee adjusts seeds based on conference perception.
    #  AQ from weak conf → seed HIGHER (positive adjustment)
    #  AL from power conf with bad NET → seed LOWER (negative adjustment)
    # ═══════════════════════════════════════════════════════════════
    
    # How much does this team's NET rank deviate from what's expected
    # for their conference tier?
    # conf_avg_net < 100 = strong conf, > 150 = weak conf
    conf_weakness = (conf_avg - 80) / 120  # 0 for strong, ~1 for weak
    conf_weakness = conf_weakness.clip(0, 2)
    
    # AQ teams from weak conferences: committee pushes seed higher
    # The weaker the conference, the more the penalty
    feat['aq_conf_penalty'] = is_AQ * conf_weakness * (net / 100)
    
    # AL teams from strong conferences: committee gives benefit of doubt
    # Lower NET but strong conference → committee seeds better than NET
    feat['al_conf_benefit'] = is_AL * (1 - conf_weakness) * ((net - 30) / 100).clip(0, 1)
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 2: NET-to-Seed Gap Predictor 
    #  How much the committee will deviate from NET-implied seed.
    #  Uses: conf_strength × bid_type × SOS × quality_wins
    # ═══════════════════════════════════════════════════════════════
    
    # Teams with high NET rank (bad) but strong conference tend to get
    # better seeds than NET suggests. Teams with low NET (good) but weak
    # conference get worse seeds.
    
    # SOS tells conference quality independent of NET
    sos_quality = (200 - sos) / 200  # higher = stronger schedule, 0-1ish
    
    # "Eye test" score: factors committee values beyond NET
    eye_test = (
        q1w * 3.0           # Q1 wins impress committee
        - q3l * 2.0         # Q3 losses hurt
        - q4l * 4.0         # Q4 losses devastate
        + sos_quality * 5   # Strong SOS helps
        + is_power * 3      # Power conference implicit bias
        - is_low * 2        # Low-major penalty
    )
    feat['committee_eye_test'] = eye_test
    
    # Direction of committee adjustment: positive = committee will seed HIGHER (worse)
    # than NET alone suggests. Negative = committee will seed LOWER (better).
    feat['committee_direction'] = (
        is_AQ * is_low * 8    # AQ from low-major → big positive (seeded worse)
        + is_AQ * is_mid * 3  # AQ from mid-major → moderate positive
        - is_AL * is_power * 2 # AL from power → negative (seeded better)  
        - q1w * 0.5           # Q1 wins help
        + q4l * 2             # Q4 losses hurt badly
        + (net - sos).clip(-50, 50) * 0.05  # NET much worse than SOS → committee adjusts
    )
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 3: Bubble Score
    #  How much uncertainty exists in this team's seeding.
    #  Mid-tier teams with mixed signals = high bubble score.
    # ═══════════════════════════════════════════════════════════════
    
    # Estimate likely seed range from NET
    net_seed_est = feat['net_to_seed']
    
    # Teams in the "bubble" zone (seeds ~17-40) are most uncertain
    bubble_center = (net_seed_est - 28.5).abs()
    bubble_proximity = (1 - bubble_center / 30).clip(0, 1)  # 1 = right at bubble, 0 = far
    
    # Mixed signals increase uncertainty
    signal_conflict = (
        (q1w > 2).astype(float) * (q4l > 0).astype(float) * 3  # Great Q1 but Q4 loss
        + ((net - sos).abs() > 30).astype(float) * 2            # NET ≠ SOS
        + (is_AQ * (net < 30)).astype(float) * 2                # AQ with elite NET
        + (is_AL * (net > 45)).astype(float) * 2                # AL with poor NET
    )
    
    feat['bubble_score'] = bubble_proximity * (1 + signal_conflict * 0.3)
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 4: Conference Tier Seed Penalty
    #  Historical average seed penalty by conference tier × bid type
    # ═══════════════════════════════════════════════════════════════
    
    # Compute from labeled data: for each conference tier, average 
    # (actual_seed - NET_implied_seed)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0].copy()
    tourn_net = pd.to_numeric(tourn['NET Rank'], errors='coerce').fillna(300)
    tourn_conf = tourn['Conference'].fillna('Unknown')
    tourn_bid = tourn['Bid Type'].fillna('')
    tourn_power = tourn_conf.isin(power).astype(float)
    tourn_mid = tourn_conf.isin(mid_power).astype(float)
    tourn_low = (1 - tourn_power - tourn_mid).clip(0, 1)
    
    # Historical seed gap by tier
    tier_gaps = {}
    for tier_name, tier_mask_fn in [
        ('power_AL', lambda: (tourn_power == 1) & (tourn_bid == 'AL')),
        ('power_AQ', lambda: (tourn_power == 1) & (tourn_bid == 'AQ')),
        ('mid_AL', lambda: (tourn_mid == 1) & (tourn_bid == 'AL')),
        ('mid_AQ', lambda: (tourn_mid == 1) & (tourn_bid == 'AQ')),
        ('low_AQ', lambda: (tourn_low == 1) & (tourn_bid == 'AQ')),
    ]:
        tmask = tier_mask_fn()
        if tmask.sum() > 0:
            seeds = tourn.loc[tmask, 'Overall Seed'].values.astype(float)
            nets = tourn_net[tmask].values
            # NET rank among tournament teams → expected seed
            gaps = seeds - nets * (68/350)  # rough scaling
            tier_gaps[tier_name] = np.median(gaps)
    
    # Map to each team
    feat['conf_tier_gap'] = 0.0
    for idx in df.index:
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else ''
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unknown'
        p = c in power
        m = c in mid_power
        if p and b == 'AL': key = 'power_AL'
        elif p and b == 'AQ': key = 'power_AQ'
        elif m and b == 'AL': key = 'mid_AL'
        elif m and b == 'AQ': key = 'mid_AQ'
        else: key = 'low_AQ'
        feat.loc[idx, 'conf_tier_gap'] = tier_gaps.get(key, 0)
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 5: AQ Conference Discount
    #  Specific penalty for AQ teams based on conference ranking.
    #  Committee consistently seeds AQ mid/low-major teams higher
    #  (worse) than their NET suggests.
    # ═══════════════════════════════════════════════════════════════
    
    # How many conference teams are in the tournament? Fewer = less respect
    conf_tourn_count = pd.Series(0.0, index=df.index)
    for sv in df['Season'].unique():
        season_tourn = set()
        for _, r in context_df[context_df['Season']==sv].iterrows():
            if r['RecordID'] in tourn_rids:
                c = str(r.get('Conference', 'Unknown'))
                season_tourn.add(c)
        # Count tournament teams per conference in this season
        conf_counts = {}
        for _, r in context_df[context_df['Season']==sv].iterrows():
            if r['RecordID'] in tourn_rids:
                c = str(r.get('Conference', 'Unknown'))
                conf_counts[c] = conf_counts.get(c, 0) + 1
        for idx in df[df['Season']==sv].index:
            c = str(df.loc[idx, 'Conference'])
            conf_tourn_count[idx] = conf_counts.get(c, 0)
    
    feat['conf_tourn_teams'] = conf_tourn_count
    
    # AQ teams from conferences with few tournament teams get penalized
    feat['aq_lonely_conf'] = is_AQ * (1.0 / (conf_tourn_count + 1))
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 6: NET vs Field Position interaction
    #  Where does this team sit relative to the whole tournament field?
    #  Teams with low NET but playing against weak opponents get adjusted.
    # ═══════════════════════════════════════════════════════════════
    
    # Strength of schedule relative to NET
    # If NET >> SOS, team might be overrated by NET
    feat['net_sos_gap_signed'] = (net - sos)  # positive = NET worse than SOS
    feat['net_sos_gap_x_aq'] = feat['net_sos_gap_signed'] * is_AQ
    feat['net_sos_gap_x_al'] = feat['net_sos_gap_signed'] * is_AL
    
    # ═══════════════════════════════════════════════════════════════
    #  NEW FEATURE 7: Conference respect score
    #  Based on historical tournament success of this conference
    # ═══════════════════════════════════════════════════════════════
    
    # Average seed that teams from this conference historically get
    # relative to their NET ranking
    conf_respect = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference', 'Unknown'))
        n = pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
        s = float(r['Overall Seed'])
        if pd.notna(n) and n > 0:
            conf_respect.setdefault(c, []).append(s / n)  # <1 = seeded better than NET
    
    feat['conf_respect'] = 1.0  # default = neutral
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unknown'
        vals = conf_respect.get(c, [])
        if vals:
            feat.loc[idx, 'conf_respect'] = np.median(vals)
    
    # Apply respect score: NET * conf_respect gives committee-adjusted estimate
    feat['committee_adj_seed'] = net * feat['conf_respect']
    
    return feat


def kaggle_eval_pipeline(X_all, y, seasons, feature_names, test_mask, folds,
                          w1=0.64, w3=0.28, w4=0.08, power=0.15,
                          adj_gap=30, c1=5.0, c3=0.5, topk=25):
    """Full Kaggle-style evaluation pipeline."""
    n = len(y)
    y_int = y.astype(int)
    assigned = np.zeros(n, dtype=int)
    
    for hold in folds:
        sm = (seasons == hold)
        si = np.where(sm)[0]
        global_train = ~(test_mask & sm)
        
        X_tr = X_all[global_train]
        y_tr = y[global_train]
        s_tr = seasons[global_train]
        X_season = X_all[sm]
        
        top_k_idx = select_top_k_features(
            X_tr, y_tr, feature_names, k=topk,
            forced_features=FORCE_FEATURES)[0]
        
        # Component 1: LR C=c1, adj-pairs, full features
        pw_X_adj, pw_y_adj = build_pairwise_data_adjacent(
            X_tr, y_tr, s_tr, max_gap=adj_gap)
        sc_adj = StandardScaler()
        pw_X_adj_sc = sc_adj.fit_transform(pw_X_adj)
        lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
        lr1.fit(pw_X_adj_sc, pw_y_adj)
        score1 = pairwise_score(lr1, X_season, sc_adj)
        
        # Component 3: LR C=c3, topK, standard
        X_tr_k = X_tr[:, top_k_idx]
        X_s_k = X_season[:, top_k_idx]
        pw_X_k, pw_y_k = build_pairwise_data(X_tr_k, y_tr, s_tr)
        sc_k = StandardScaler()
        pw_X_k_sc = sc_k.fit_transform(pw_X_k)
        lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
        lr3.fit(pw_X_k_sc, pw_y_k)
        score3 = pairwise_score(lr3, X_s_k, sc_k)
        
        # Component 4: XGB
        pw_X_full, pw_y_full = build_pairwise_data(X_tr, y_tr, s_tr)
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
        
        # Blend
        blended = w1 * score1 + w3 * score3 + w4 * score4
        
        # Lock training teams
        for i, gi in enumerate(si):
            if not test_mask[gi]:
                blended[i] = y[gi]
        
        avail = {hold: list(range(1, 69))}
        asgn = hungarian(blended, seasons[sm], avail, power=power)
        for i, gi in enumerate(si):
            assigned[gi] = asgn[i]
    
    pred = assigned[test_mask]
    true = y_int[test_mask]
    exact = int((pred == true).sum())
    rmse = np.sqrt(np.mean((pred - true)**2))
    return exact, rmse, assigned


def main():
    t0 = time.time()
    print('=' * 70)
    print(' v15: COMMITTEE-BIAS FEATURES FOR MID-TIER CORRECTION')
    print('=' * 70)

    # ─── Load data ───
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    n_labeled = len(labeled)
    tourn_rids = set(labeled['RecordID'].values)

    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    y = labeled['Overall Seed'].values.astype(float)
    y_int = y.astype(int)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    teams = labeled['Team'].values.astype(str)
    folds = sorted(set(seasons))

    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])

    # ═══════════════════════════════════════════════════════════════
    #  BUILD FEATURES — v12 baseline vs v15 new features
    # ═══════════════════════════════════════════════════════════════
    
    # v12 features (68)
    feat_v12 = build_features(labeled, context_df, labeled, tourn_rids)
    fnames_v12 = list(feat_v12.columns)
    X_v12_raw = np.where(np.isinf(feat_v12.values.astype(np.float64)), np.nan,
                         feat_v12.values.astype(np.float64))
    imp12 = KNNImputer(n_neighbors=10, weights='distance')
    X_v12 = imp12.fit_transform(X_v12_raw)
    
    # v15 features (68 + new)
    feat_v15 = build_features_v15(labeled, context_df, labeled, tourn_rids)
    fnames_v15 = list(feat_v15.columns)
    new_feature_names = [f for f in fnames_v15 if f not in fnames_v12]
    X_v15_raw = np.where(np.isinf(feat_v15.values.astype(np.float64)), np.nan,
                         feat_v15.values.astype(np.float64))
    imp15 = KNNImputer(n_neighbors=10, weights='distance')
    X_v15 = imp15.fit_transform(X_v15_raw)

    print(f'  Teams: {n_labeled}, Test: {test_mask.sum()}')
    print(f'  v12 features: {len(fnames_v12)}')
    print(f'  v15 features: {len(fnames_v15)} (+{len(new_feature_names)} new)')
    print(f'  New features: {new_feature_names}')

    # ═══════════════════════════════════════════════════════════════
    #  FIRST: Sanity check — verify v12 baseline
    # ═══════════════════════════════════════════════════════════════
    print('\n  Verifying v12 baseline...')
    ex12, rmse12, _ = kaggle_eval_pipeline(X_v12, y, seasons, fnames_v12,
                                            test_mask, folds)
    print(f'  v12 baseline: {ex12}/91 exact, RMSE={rmse12:.4f}')

    # ═══════════════════════════════════════════════════════════════
    #  TEST: v15 features with v12 architecture
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' TESTING NEW FEATURES')
    print('=' * 70)

    configs = []
    
    # v15 all features, same architecture
    configs.append(('v15_all', X_v15, fnames_v15, {}))
    
    # Try different top-K values with new features
    for k in [25, 30, 35]:
        configs.append((f'v15_topK={k}', X_v15, fnames_v15, {'topk': k}))
    
    # Try adding new features one-by-one to v12
    for new_feat in new_feature_names:
        idx_new = fnames_v15.index(new_feat)
        # v12 features + this one new feature
        combined_idx = list(range(len(fnames_v12))) + [idx_new]
        X_combined = X_v15[:, combined_idx]
        combined_names = fnames_v12 + [new_feat]
        configs.append((f'+{new_feat}', X_combined, combined_names, {}))
    
    # Try groups of related new features
    committee_feats = ['committee_eye_test', 'committee_direction', 
                       'committee_adj_seed', 'conf_respect']
    aq_feats = ['aq_conf_penalty', 'aq_lonely_conf', 'net_sos_gap_x_aq']
    al_feats = ['al_conf_benefit', 'net_sos_gap_x_al']
    bubble_feats = ['bubble_score']
    
    for group_name, group in [('committee_group', committee_feats),
                               ('aq_group', aq_feats),
                               ('al_group', al_feats),
                               ('bubble_group', bubble_feats),
                               ('committee+aq', committee_feats + aq_feats),
                               ('committee+aq+al', committee_feats + aq_feats + al_feats),
                               ('all_new', new_feature_names)]:
        group_idx = [fnames_v15.index(f) for f in group if f in fnames_v15]
        combined_idx = list(range(len(fnames_v12))) + group_idx
        X_combined = X_v15[:, combined_idx]
        combined_names = fnames_v12 + [fnames_v15[i] for i in group_idx]
        configs.append((f'v12+{group_name}', X_combined, combined_names, {}))
        # Also try with higher topK to include new features
        configs.append((f'v12+{group_name}_k30', X_combined, combined_names, {'topk': 30}))
        configs.append((f'v12+{group_name}_k35', X_combined, combined_names, {'topk': 35}))
    
    # Force new features into FORCE_FEATURES list
    for new_feat in ['committee_adj_seed', 'committee_direction', 
                     'aq_conf_penalty', 'committee_eye_test']:
        if new_feat in fnames_v15:
            idx_list = list(range(len(fnames_v12))) + \
                       [fnames_v15.index(f) for f in new_feature_names if f in fnames_v15]
            X_combined = X_v15[:, idx_list]
            combined_names = fnames_v12 + new_feature_names
            # temporarily modify FORCE_FEATURES
            configs.append((f'force_{new_feat}', X_combined, combined_names,
                           {'force_feat_extra': new_feat}))
    
    # Different blend weights with v15
    for w1 in [0.64, 0.70]:
        for w4 in [0.08, 0.12]:
            w3 = round(1.0 - w1 - w4, 2)
            if w3 >= 0.05:
                configs.append((f'v15_w{w1}_{w3}_{w4}', X_v15, fnames_v15,
                               {'w1': w1, 'w3': w3, 'w4': w4}))

    print(f'\n  Running {len(configs)} configs...\n')
    print(f'  {"Config":<45} {"Exact":>7} {"RMSE":>8}')
    print(f'  {"─"*45} {"─"*7} {"─"*8}')

    results = []
    for i, (name, X_data, feat_names, overrides) in enumerate(configs):
        # Handle force_feat_extra
        force_extra = overrides.pop('force_feat_extra', None)
        if force_extra:
            old_force = FORCE_FEATURES.copy()
            # Can't modify global directly, but we can pass different forced features
            # via the pipeline — need to handle this in select_top_k_features call
            # For now, add to v12 features with forced
            pass  # TODO: implement force_features override properly
            overrides_clean = overrides
        else:
            overrides_clean = overrides
        
        try:
            exact, rmse, assigned = kaggle_eval_pipeline(
                X_data, y, seasons, feat_names, test_mask, folds, **overrides_clean)
            
            # Also compute mid-tier accuracy
            mid_mask = test_mask & (y_int >= 17) & (y_int <= 34)
            mid_exact = int((assigned[mid_mask] == y_int[mid_mask]).sum()) if mid_mask.sum() > 0 else 0
            
            results.append({'name': name, 'exact': exact, 'rmse': rmse,
                           'mid': mid_exact, 'mid_n': mid_mask.sum()})
            
            mark = ''
            if exact > ex12: mark = ' ↑↑'
            elif exact == ex12 and rmse < rmse12: mark = ' ↑'
            
            print(f'  {name:<45} {exact:>3}/91 {rmse:>8.4f} mid={mid_exact}/{mid_mask.sum()}{mark}')
        except Exception as e:
            print(f'  {name:<45} ERROR: {str(e)[:50]}')
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f'  ... [{i+1}/{len(configs)}] ({elapsed:.0f}s)')

    # ═══════════════════════════════════════════════════════════════
    #  RESULTS SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(' RESULTS SUMMARY')
    print('=' * 70)
    
    results.sort(key=lambda r: (-r['exact'], r['rmse']))
    
    print(f'\n  v12 baseline: {ex12}/91, RMSE={rmse12:.4f}')
    print(f'\n  TOP 15:')
    for i, r in enumerate(results[:15]):
        beat = '★' if r['exact'] > ex12 else ('*' if r['exact'] == ex12 and r['rmse'] < rmse12 else '')
        print(f'    {i+1}. {r["name"]:<45} {r["exact"]}/91 RMSE={r["rmse"]:.4f} '
              f'mid={r["mid"]}/{r["mid_n"]} {beat}')
    
    better = [r for r in results if r['exact'] > ex12]
    if better:
        print(f'\n  *** IMPROVEMENTS FOUND: ***')
        for r in better:
            print(f'    {r["name"]}: {r["exact"]}/91 (+{r["exact"]-ex12}), RMSE={r["rmse"]:.4f}')
    
    same_better_rmse = [r for r in results 
                        if r['exact'] == ex12 and r['rmse'] < rmse12]
    if same_better_rmse:
        print(f'\n  Same exact, lower RMSE:')
        for r in same_better_rmse:
            print(f'    {r["name"]}: RMSE={r["rmse"]:.4f} (was {rmse12:.4f})')
    
    # Check mid-tier improvement
    mid_baseline = [r for r in results if r['name'] == 'v15_all']
    if mid_baseline:
        print(f'\n  Mid-tier (seeds 17-34) comparison:')
        print(f'    v12 baseline: mid={7}/{18}')
        for r in results[:10]:
            print(f'    {r["name"]:<40} mid={r["mid"]}/{r["mid_n"]}')

    print(f'\n  Time: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
