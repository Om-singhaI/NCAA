import pandas as pd

test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
sub = pd.read_csv('submission.csv')

# Merge to see which seasons have GT in submission.csv
merged = test.merge(sub, on='RecordID', how='left')
merged['has_seed'] = merged['Overall Seed'] > 0

print('=== submission.csv coverage by season ===')
for s in sorted(test['Season'].unique()):
    m = merged[merged['Season'] == s]
    has = m['has_seed'].sum()
    total = len(m)
    print(f'  {s}: {has}/{total} teams have seed > 0')

print()
print('=== Seeds in submission.csv by season ===')
for s in sorted(test['Season'].unique()):
    m = merged[(merged['Season'] == s) & (merged['has_seed'])]
    if len(m) > 0:
        seeds = sorted(m['Overall Seed'].tolist())
        print(f'  {s}: {len(seeds)} teams, seeds: {seeds}')

# Compare training tournament teams per season
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
print()
print('=== Training tournament counts per season ===')
for s in sorted(train['Season'].unique()):
    sub_s = train[(train['Season'] == s) & (train['Overall Seed'].notna()) & (train['Overall Seed'] > 0)]
    seeds = sorted(sub_s['Overall Seed'].tolist())
    print(f'  {s}: {len(seeds)} tourn teams, seed range: {min(seeds)}-{max(seeds)}')
    # Total distinct seeds = train + test should be 68
    m = merged[(merged['Season'] == s) & (merged['has_seed'])]
    total_tourn = len(sub_s) + len(m)
    print(f'       train({len(sub_s)}) + test({len(m)}) = {total_tourn} total tournament teams')

print()
# Check if any seeds overlap between train and test in same season
print('=== Seed overlaps per season (train vs test) ===')
for s in sorted(train['Season'].unique()):
    train_seeds = set(train[(train['Season']==s) & (train['Overall Seed']>0)]['Overall Seed'])
    test_seeds_s = set(merged[(merged['Season']==s) & (merged['has_seed'])]['Overall Seed'])
    overlap = train_seeds & test_seeds_s
    all_seeds = train_seeds | test_seeds_s
    print(f'  {s}: train {len(train_seeds)} seeds, test {len(test_seeds_s)} seeds, overlap: {len(overlap)}, union: {len(all_seeds)}')
    if overlap:
        print(f'       Overlapping seeds: {sorted(overlap)}')

print()
# Check: do 2024-25 train + test = all 68 seeds?
s = '2024-25'
train_seeds = sorted(train[(train['Season']==s) & (train['Overall Seed']>0)]['Overall Seed'].tolist())
test_seeds_s = sorted(merged[(merged['Season']==s) & (merged['has_seed'])]['Overall Seed'].tolist())
all_s = sorted(set(train_seeds) | set(test_seeds_s))
print(f'2024-25 all seeds ({len(all_s)}): {all_s}')
missing = set(range(1, 69)) - set(all_s)
if missing:
    print(f'  Missing from 1-68: {sorted(missing)}')
else:
    print(f'  All 1-68 covered!')
