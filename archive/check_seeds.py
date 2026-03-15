import pandas as pd

train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')

# Check seed values
seeds = train[train['Overall Seed'].notna() & (train['Overall Seed'] != '')]
print("Seed value range:")
print(f"  Min: {seeds['Overall Seed'].min()}")
print(f"  Max: {seeds['Overall Seed'].max()}")
print(f"  Count: {len(seeds)}")
print(f"  Mean: {seeds['Overall Seed'].astype(float).mean():.1f}")
print()

# Distribution
print("Seed distribution:")
print(seeds['Overall Seed'].astype(float).astype(int).value_counts().sort_index())
print()

# Show some specific teams with known seeds for validation
print("Sample teams with seeds (for verification):")
for s in sorted(seeds['Season'].unique()):
    subset = seeds[seeds['Season'] == s].sort_values('Overall Seed')
    top5 = subset.head(5)[['Team', 'Overall Seed', 'NET Rank']].to_string(index=False)
    print(f"\n  {s} (top 5):")
    print(f"  {top5}")
    
    # Count per season
    print(f"  Total seeded: {len(subset)}")
