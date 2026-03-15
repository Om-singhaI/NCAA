import pandas as pd
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
sub = pd.read_csv('submission.csv')
print('=== TRAIN ===')
print('Columns:', list(train.columns))
print('Seasons:', sorted(train['Season'].unique()))
print('Shape:', train.shape)
tt = train[pd.to_numeric(train['Overall Seed'], errors='coerce').fillna(0) > 0]
print('Tournament teams per season:')
for s in sorted(tt['Season'].unique()):
    print(f'  {s}: {len(tt[tt.Season==s])}')
print('Bid Type:', train['Bid Type'].value_counts().to_dict())

print('\n=== TEST ===')
print('Seasons:', sorted(test['Season'].unique()))
print('Shape:', test.shape)
for s in sorted(test['Season'].unique()):
    print(f'  {s}: {len(test[test.Season==s])}')

sub_gt = sub[sub['Overall Seed'] > 0]
print(f'\nGT in submission: {len(sub_gt)}')

print('\nSample train row (first tournament team):')
print(train.iloc[tt.index[0]].to_string())

# Check available packages
for pkg in ['lightgbm', 'catboost', 'sklearn', 'optuna', 'scipy']:
    try:
        __import__(pkg)
        print(f'{pkg}: OK')
    except:
        print(f'{pkg}: MISSING')
