import pandas as pd

test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')

print("TEST SET:")
print(f"  Total rows: {len(test)}")
print(f"  Seasons: {sorted(test['Season'].unique())}")
for s in sorted(test['Season'].unique()):
    subset = test[test['Season'] == s]
    print(f"    {s}: {len(subset)} teams")

print()
print("TRAIN SET:")
print(f"  Total rows: {len(train)}")
print(f"  Seasons: {sorted(train['Season'].unique())}")
for s in sorted(train['Season'].unique()):
    subset = train[train['Season'] == s]
    has_seed = subset['Overall Seed'].notna() & (subset['Overall Seed'] != '')
    n_seeded = has_seed.sum()
    print(f"    {s}: {len(subset)} teams ({n_seeded} seeded)")

print()
print("TEST COLUMNS:", list(test.columns))
print("TRAIN COLUMNS:", list(train.columns))

print()
print("TEST - Teams by season:")
for s in sorted(test['Season'].unique()):
    teams = sorted(test[test['Season'] == s]['Team'].tolist())
    print(f"  {s} ({len(teams)} teams):")
    for t in teams[:15]:
        print(f"    {t}")
    if len(teams) > 15:
        print(f"    ... and {len(teams)-15} more")

# Check Bid Types in test
print()
print("TEST Bid Types:")
print(test['Bid Type'].value_counts())
