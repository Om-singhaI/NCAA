import pandas as pd

sub = pd.read_csv('sub_perfect_actual.csv')
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')

for season in sorted(test['Season'].unique()):
    train_s = train[train['Season'] == season]
    train_seeds = set(train_s['Overall Seed'].dropna().astype(int).tolist())
    
    test_ids = test[test['Season'] == season]['RecordID'].tolist()
    sub_s = sub[sub['RecordID'].isin(test_ids)]
    test_seeds = set(sub_s[sub_s['Overall Seed'] > 0]['Overall Seed'].astype(int).tolist())
    
    combined = train_seeds | test_seeds
    all_68 = set(range(1, 69))
    missing = all_68 - combined
    overlap = train_seeds & test_seeds
    
    print(f'{season}: train={len(train_seeds)}, test={len(test_seeds)}, combined={len(combined)}/68')
    if missing:
        print(f'  MISSING from 1-68: {sorted(missing)}')
    if overlap:
        print(f'  OVERLAP (in both): {sorted(overlap)}')
    if not missing and not overlap:
        print(f'  PERFECT: All 68 positions covered, no overlaps')
