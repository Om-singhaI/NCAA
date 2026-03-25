"""
Compare our model's predictions across 7 weekly test sets (Feb 6 - Mar 15).

Each test set has all 365 teams' stats at a different point in the season, but
Bid Type is NaN for all teams. We use our reference NCAA_2026_Data.csv to
identify the 68 tournament teams and their bid types, then run the full v50
pipeline (same as run_prediction) on each week's stats.

This shows how seed predictions stabilize as the season progresses.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ncaa_2026_model import (
    load_data, build_features, select_top_k_features,
    predict_robust_blend, hungarian,
    build_min8_features,
    USE_TOP_K_A, FORCE_FEATURES,
    DUAL_RIDGE_ALPHA, DUAL_BLEND,
    DATA_DIR,
)
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

POWER = 0.15

# Name mapping: our reference data has (AQ) suffixes that test sets don't
AQ_NAME_MAP = {
    'Gonzaga(AQ)': 'Gonzaga',
    'High Point(AQ)': 'High Point',
    'Hofstra(AQ)': 'Hofstra',
    'LIU(AQ)': 'LIU',
    'Lehigh(AQ)': 'Lehigh',
    'North Dakota St.(AQ)': 'North Dakota St.',
    'Queens (NC)(AQ)': 'Queens (NC)',
    'Tennessee St.(AQ)': 'Tennessee St.',
    'Troy(AQ)': 'Troy',
    'UMBC(AQ)': 'UMBC',
    'UNI(AQ)': 'UNI',
    'Wright St.(AQ)': 'Wright St.',
}
REVERSE_MAP = {v: k for k, v in AQ_NAME_MAP.items()}


def predict_one_test_set(test_set_path, ref_tourn, all_df_hist, labeled, tourn_rids):
    """
    Run full v50 pipeline on one weekly test set.

    Args:
        test_set_path: path to weekly CSV (365 teams, Bid Type all NaN)
        ref_tourn: DataFrame of 68 reference tournament teams with Bid Type
        all_df_hist: historical training data (all rows, no seed col)
        labeled: historical labeled tournament teams (with Overall Seed)
        tourn_rids: set of historical tournament RecordIDs

    Returns:
        dict {team_name: predicted_seed} for the 68 tournament teams
    """
    new_df = pd.read_csv(test_set_path)

    # Build a mapping: test-set team -> row, using reference tournament teams
    ref_team_to_bid = {}
    ref_team_to_rid = {}
    for _, r in ref_tourn.iterrows():
        name = r['Team']
        ts_name = AQ_NAME_MAP.get(name, name)  # map to test set name
        ref_team_to_bid[ts_name] = r['Bid Type']
        ref_team_to_rid[ts_name] = r['RecordID']

    # Assign Bid Type and RecordID to matching teams in test set
    bid_types = []
    record_ids = []
    for _, r in new_df.iterrows():
        team = r['Team']
        if team in ref_team_to_bid:
            bid_types.append(ref_team_to_bid[team])
            record_ids.append(ref_team_to_rid[team])
        else:
            bid_types.append(np.nan)
            record_ids.append(r['RecordID'])
    new_df['Bid Type'] = bid_types
    new_df['RecordID'] = record_ids

    # Filter to tournament teams
    new_tourn = new_df[new_df['Bid Type'].isin(['AL', 'AQ'])].copy()
    n_tourn = len(new_tourn)
    if n_tourn == 0:
        return {}

    # Context: all historical teams + all new teams
    context_all = pd.concat([all_df_hist, new_df], ignore_index=True)

    # All tournament RIDs (historic + new)
    all_tourn_rids = tourn_rids.copy()
    for _, r in new_tourn.iterrows():
        all_tourn_rids.add(r['RecordID'])

    n_labeled = len(labeled)

    # Build features
    feat_train = build_features(labeled, context_all, labeled, all_tourn_rids)
    feat_new = build_features(new_tourn, context_all, labeled, all_tourn_rids)
    feature_names = list(feat_train.columns)

    y_train = labeled['Overall Seed'].values.astype(float)

    # Impute jointly
    X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan,
                        feat_train.values.astype(np.float64))
    X_new_raw = np.where(np.isinf(feat_new.values.astype(np.float64)), np.nan,
                         feat_new.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X_comb = imp.fit_transform(np.vstack([X_tr_raw, X_new_raw]))
    X_train = X_comb[:n_labeled]
    X_new = X_comb[n_labeled:]

    # Feature selection
    top_k_idx, _ = select_top_k_features(
        X_train, y_train, feature_names, k=USE_TOP_K_A,
        forced_features=FORCE_FEATURES)

    # v12 pairwise blend scores
    seasons_train = labeled['Season'].values.astype(str)
    raw_v12 = predict_robust_blend(X_train, y_train, X_new,
                                    seasons_train, top_k_idx)

    # Committee Ridge scores (min8 features)
    X_comm_train = build_min8_features(X_train, feature_names)
    X_comm_new = build_min8_features(X_new, feature_names)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_comm_train)
    X_te_sc = sc.transform(X_comm_new)
    ridge = Ridge(alpha=DUAL_RIDGE_ALPHA)
    ridge.fit(X_tr_sc, y_train)
    raw_comm = ridge.predict(X_te_sc)

    # Dual-Hungarian: run both, blend assignments
    new_season = str(new_tourn['Season'].iloc[0])
    new_seasons = np.array([new_season] * n_tourn)
    avail = {new_season: list(range(1, 69))}

    seeds_v12 = hungarian(raw_v12, new_seasons, avail, power=POWER)
    seeds_comm = hungarian(raw_comm, new_seasons, avail, power=POWER)

    # Blend: 75% v12 + 25% committee
    blended = (1 - DUAL_BLEND) * seeds_v12.astype(float) + DUAL_BLEND * seeds_comm.astype(float)
    seeds_final = hungarian(blended, new_seasons, avail, power=0.01)

    # Build result dict: use reference team names (with AQ suffix)
    result = {}
    for i, (_, row) in enumerate(new_tourn.iterrows()):
        ts_name = row['Team']
        # Map back to reference name if needed
        ref_name = REVERSE_MAP.get(ts_name, ts_name)
        result[ref_name] = int(seeds_final[i])

    return result


def main():
    print('=' * 70)
    print(' COMPARING PREDICTIONS ACROSS 7 WEEKLY TEST SETS')
    print(' (Feb 6 -> Mar 15, 2026)')
    print('=' * 70)

    # Load training data once
    all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    n_labeled = len(labeled)
    print(f'\n  Training data: {n_labeled} labeled teams')

    # Historical context (all teams, no seed col)
    all_df_hist = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)

    # Reference tournament teams from our final 2026 data
    ref_df = pd.read_csv(os.path.join(DATA_DIR, 'data', 'NCAA_2026_Data.csv'))
    ref_tourn = ref_df[ref_df['Bid Type'].isin(['AL', 'AQ'])].copy()
    print(f'  Reference tournament teams: {len(ref_tourn)}')

    dates = ['20260206', '20260208', '20260215', '20260222', '20260301', '20260308', '20260315']
    labels = ['Feb 6', 'Feb 8', 'Feb 15', 'Feb 22', 'Mar 1', 'Mar 8', 'Mar 15']

    all_predictions = {}

    for date, label in zip(dates, labels):
        fpath = os.path.join(DATA_DIR, 'data', 'test_sets', f'NCAA_Seed_Test_Set_2026_{date}.csv')
        print(f'\n--- Processing {label} ({date}) ---')

        t0 = time.time()
        preds = predict_one_test_set(fpath, ref_tourn, all_df_hist, labeled, tourn_rids)
        elapsed = time.time() - t0
        print(f'  Predicted {len(preds)} seeds in {elapsed:.1f}s')

        all_predictions[label] = preds

    # ================================================================
    #  COMPARISON TABLE
    # ================================================================
    print('\n' + '=' * 70)
    print(' PREDICTION COMPARISON TABLE')
    print('=' * 70)

    # Sort by latest prediction
    latest = all_predictions.get('Mar 15', {})
    if not latest:
        latest = all_predictions[labels[-1]]
    teams_by_seed = sorted(latest.items(), key=lambda x: x[1])

    # Header
    header = f'{"Team":<28s}'
    for label in labels:
        header += f'{label:>7s}'
    header += f'{"Spread":>8s}'
    print(header)
    print('-' * len(header))

    # Body
    for team, _ in teams_by_seed:
        row = f'{team:<28s}'
        seeds = []
        for label in labels:
            s = all_predictions[label].get(team, None)
            if s is not None:
                row += f'{s:>7d}'
                seeds.append(s)
            else:
                row += f'{"---":>7s}'
        if len(seeds) >= 2:
            spread = max(seeds) - min(seeds)
            row += f'{spread:>8d}'
        else:
            row += f'{"---":>8s}'
        print(row)

    # ================================================================
    #  STABILITY ANALYSIS
    # ================================================================
    print('\n' + '=' * 70)
    print(' STABILITY ANALYSIS')
    print('=' * 70)

    spreads = []
    for team, _ in teams_by_seed:
        seeds = [all_predictions[l].get(team) for l in labels
                 if all_predictions[l].get(team) is not None]
        if len(seeds) >= 2:
            spreads.append(max(seeds) - min(seeds))

    spreads = np.array(spreads)
    print(f'  Teams tracked: {len(spreads)}')
    print(f'  Mean spread (max-min across dates): {spreads.mean():.1f}')
    print(f'  Median spread: {np.median(spreads):.1f}')
    print(f'  Teams with spread=0 (perfectly stable): {(spreads == 0).sum()}')
    print(f'  Teams with spread<=2: {(spreads <= 2).sum()}')
    print(f'  Teams with spread>5 (volatile): {(spreads > 5).sum()}')

    # Most volatile teams
    volatile = []
    for team, _ in teams_by_seed:
        seeds = [all_predictions[l].get(team) for l in labels
                 if all_predictions[l].get(team) is not None]
        if len(seeds) >= 2:
            spread = max(seeds) - min(seeds)
            if spread > 3:
                volatile.append((team, min(seeds), max(seeds), spread))

    if volatile:
        volatile.sort(key=lambda x: -x[3])
        print(f'\n  MOST VOLATILE TEAMS (spread > 3):')
        for team, lo, hi, sp in volatile:
            print(f'    {team:<25s}  range: {lo}-{hi} (spread={sp})')

    # Pairwise agreement between consecutive dates
    print(f'\n  EXACT AGREEMENT BETWEEN CONSECUTIVE DATES:')
    for i in range(len(labels) - 1):
        p1 = all_predictions[labels[i]]
        p2 = all_predictions[labels[i + 1]]
        common = set(p1.keys()) & set(p2.keys())
        if len(common) > 0:
            exact = sum(1 for t in common if p1[t] == p2[t])
            print(f'    {labels[i]:>7s} -> {labels[i+1]:<7s}: '
                  f'{exact}/{len(common)} exact ({100*exact/len(common):.1f}%)')
        else:
            print(f'    {labels[i]:>7s} -> {labels[i+1]:<7s}: no common teams')

    # Agreement with final (Mar 15) predictions
    print(f'\n  AGREEMENT WITH FINAL (Mar 15):')
    for label in labels[:-1]:
        p1 = all_predictions[label]
        p2 = all_predictions['Mar 15']
        common = set(p1.keys()) & set(p2.keys())
        if len(common) > 0:
            exact = sum(1 for t in common if p1[t] == p2[t])
            within2 = sum(1 for t in common if abs(p1[t] - p2[t]) <= 2)
            print(f'    {label:>7s}: {exact}/{len(common)} exact ({100*exact/len(common):.1f}%), '
                  f'{within2}/{len(common)} within 2 ({100*within2/len(common):.1f}%)')

    print(f'\n  Done!')


if __name__ == '__main__':
    main()
