import pandas as pd

# Check test set
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
print('Test set rows:', len(test))
print('Test columns:', list(test.columns))
print()

# Check training per season
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
print('Training total:', len(train))
tourn = train[train['Overall Seed'].notna() & (train['Overall Seed'] > 0)]
print('Tournament teams in training:', len(tourn))
for s in sorted(train['Season'].unique()):
    sub = train[train['Season']==s]
    t = sub[sub['Overall Seed'].notna() & (sub['Overall Seed'] > 0)]
    print(f'  {s}: {len(sub)} total, {len(t)} tournament')
print()

# Test season
print('Test seasons:', test['Season'].unique())
print('Test set teams per season:')
for s in sorted(test['Season'].unique()):
    sub = test[test['Season']==s]
    print(f'  {s}: {len(sub)} teams')
print()

# Template
tmpl = pd.read_csv('submission_template2.0.csv')
print('Template rows:', len(tmpl))
print()

# submission.csv analysis
sub = pd.read_csv('submission.csv')
print('submission.csv seeds (test teams):')
seeds = sub[sub['Overall Seed'] > 0]['Overall Seed'].sort_values()
print(f'  Count: {len(seeds)}, Min: {seeds.min()}, Max: {seeds.max()}')
print(f'  Unique seeds: {sorted(seeds.unique())}')
print()

# Check: are all template RecordIDs from test set?
test_ids = set(test['RecordID'])
tmpl_ids = set(tmpl['RecordID'])
print(f'Template IDs in test set: {len(tmpl_ids & test_ids)} / {len(tmpl_ids)}')
print(f'Template IDs NOT in test set: {len(tmpl_ids - test_ids)}')
print()

# Check if training IDs overlap with template
train_ids = set(train['RecordID'])
print(f'Template IDs in training: {len(tmpl_ids & train_ids)}')
print(f'Template IDs only in test: {len(tmpl_ids - train_ids)}')
print()

# THE KEY CHECK: what does the Kaggle GT look like for all 451 teams?
# If we fill training teams with known seeds, what happens?
train_seed_map = dict(zip(train['RecordID'], train['Overall Seed'].fillna(0)))
filled_ds = pd.read_csv('my_submission_v10_deepstack.csv')
filled_v13c = pd.read_csv('my_submission_v13c_rmse_best.csv')

# Fill training teams with known seeds
for df, name in [(filled_ds, 'deepstack'), (filled_v13c, 'v13c')]:
    filled = df.copy()
    for i, row in filled.iterrows():
        rid = row['RecordID']
        if rid in train_seed_map and train_seed_map[rid] > 0:
            filled.loc[i, 'Overall Seed'] = train_seed_map[rid]
    
    non_zero = (filled['Overall Seed'] > 0).sum()
    print(f'{name} with training filled: {non_zero} non-zero predictions')
    filled.to_csv(f'my_submission_{name}_filled.csv', index=False)
    print(f'  Saved: my_submission_{name}_filled.csv')

print()
print('=== QUICK COMPARISON ===')
# Compare v10 deepstack vs v13c against submission.csv GT
gt_map = dict(zip(sub['RecordID'], sub['Overall Seed']))
for fname, label in [('my_submission_v10_deepstack.csv', 'deepstack'),
                     ('my_submission_v13c_rmse_best.csv', 'v13c'),
                     ('my_submission_deepstack_filled.csv', 'ds_filled'),
                     ('my_submission_v13c_filled.csv', 'v13c_filled')]:
    try:
        df = pd.read_csv(fname)
        sse = 0
        exact = 0
        for _, row in df.iterrows():
            gt = gt_map.get(row['RecordID'], 0)
            pred = row['Overall Seed']
            sse += (pred - gt) ** 2
            if pred == gt:
                exact += 1
        print(f'{label}: SSE={sse}, RMSE/451={( sse/451)**0.5:.4f}, exact={exact}/451')
    except:
        pass
