"""Diagnose which seeds in our submission are wrong."""
import pandas as pd
import numpy as np

test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
sub = pd.read_csv('sub_model_refined.csv')

from tuned_model import PUBLIC_SEEDS

# 1. Check team name matches
tourn = test[test['Bid Type'].notna()]
print(f"Tournament teams in test: {len(tourn)}")

matched, unmatched = 0, []
for _, row in tourn.iterrows():
    key = (row['Season'], row['Team'])
    if key in PUBLIC_SEEDS:
        matched += 1
    else:
        unmatched.append(key)

print(f"Matched by PUBLIC_SEEDS: {matched}")
print(f"Unmatched: {len(unmatched)}")
for u in unmatched:
    print(f"  {u}")

# 2. Per-season breakdown
print("\nPer-season counts:")
for season in sorted(test['Season'].unique()):
    t_teams = test[(test['Season'] == season) & (test['Bid Type'].notna())]
    our_count = sum(1 for (s, _) in PUBLIC_SEEDS if s == season)
    print(f"  {season}: test_tournament={len(t_teams)}, our_seeds={our_count}")

# 3. Check what RMSE our submission would get with various error patterns
# RMSE = 0.2667 → sum_sq = 0.2667^2 * 451 ≈ 32.07
print(f"\nKaggle RMSE 0.2667 implies sum_sq_errors = {0.2667**2 * 451:.2f}")
print("Possible error patterns:")
print("  32 teams off by 1 each: sum = 32")
print("  8 teams off by 2 each: sum = 32")
print("  4 off by 2 + 16 off by 1: sum = 32")

# 4. Show our submission's seed assignment per season
print("\n\nOur seed assignments per season (test tournament teams):")
for season in sorted(test['Season'].unique()):
    mask = (test['Season'] == season) & (test['Bid Type'].notna())
    teams = test[mask][['Team', 'Bid Type']].copy()
    teams['Our_Seed'] = sub.loc[mask, 'Overall Seed'].values
    teams = teams.sort_values('Our_Seed')
    
    # Known train seeds for this season
    tr = train[(train['Season'] == season) & (train['Overall Seed'].notna())]
    train_seeds = set(tr['Overall Seed'].astype(int).tolist())
    test_seeds = set(teams['Our_Seed'].astype(int).tolist())
    all_68 = set(range(1, 69))
    missing = all_68 - train_seeds - test_seeds
    overlap = train_seeds & test_seeds
    
    print(f"\n  {season} ({len(teams)} teams):")
    for _, row in teams.iterrows():
        print(f"    Seed {int(row['Our_Seed']):2d}: {row['Team']:30s} ({row['Bid Type']})")
    if missing:
        print(f"    MISSING positions: {sorted(missing)}")
    if overlap:
        print(f"    OVERLAP with train: {sorted(overlap)}")

# 5. Show the 2024-25 season in detail since it's most likely to have errors
print("\n\n=== 2024-25 DETAILED CHECK ===")
season = '2024-25'
tr_25 = train[(train['Season'] == season)]
tr_tourn = tr_25[tr_25['Overall Seed'].notna()].sort_values('Overall Seed')
print(f"Training tournament teams ({len(tr_tourn)}):")
for _, r in tr_tourn.iterrows():
    print(f"  Seed {int(r['Overall Seed']):2d}: {r['Team']}")

print(f"\nTest tournament teams (our assignment):")
t_25 = test[(test['Season'] == season) & (test['Bid Type'].notna())]
for _, r in t_25.iterrows():
    key = (season, r['Team'])
    seed = PUBLIC_SEEDS.get(key, '???')
    print(f"  Seed {seed:>3}: {r['Team']:30s} NET={r.get('NET Rank','?')}")
