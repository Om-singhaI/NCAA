import pandas as pd

test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')

# Get tournament teams from test (those with Bid Type)
tourney_test = test[test['Bid Type'].notna() & (test['Bid Type'] != '')]

print("TOURNAMENT TEAMS IN TEST SET (need actual seeds):")
print("="*60)
for s in sorted(tourney_test['Season'].unique()):
    subset = tourney_test[tourney_test['Season'] == s].sort_values('NET Rank')
    print(f"\n{s} ({len(subset)} teams):")
    for _, row in subset.iterrows():
        print(f"  {row['Team']:30s} NET={str(row['NET Rank']):>4s} Bid={row['Bid Type']}")

# Also show training seeds per season for reference (to find gaps)
print("\n\n" + "="*60)
print("TRAINING SEEDS (for finding missing seed positions):")
print("="*60)
for s in sorted(train['Season'].unique()):
    subset = train[(train['Season'] == s) & (train['Overall Seed'].notna()) & (train['Overall Seed'] != '')]
    subset = subset.sort_values('Overall Seed')
    seeds_used = set(subset['Overall Seed'].astype(float).astype(int).tolist())
    all_seeds = set(range(1, 69))
    missing = sorted(all_seeds - seeds_used)
    print(f"\n{s}: {len(subset)} seeds in training, {len(missing)} missing (in test)")
    print(f"  Missing seed positions: {missing}")
