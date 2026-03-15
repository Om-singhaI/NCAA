import pandas as pd
import numpy as np

sub = pd.read_csv('submission.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
template = pd.read_csv('submission_template2.0.csv')

print('submission.csv:', len(sub), 'rows, non-zero:', (sub['Overall Seed'] > 0).sum())
print('Test set:', len(test), 'rows')
print('Template:', len(template), 'rows')

gt_sub = {}
for _, r in sub.iterrows():
    s = int(r['Overall Seed'])
    if s > 0:
        gt_sub[r['RecordID']] = s

# Test set has NO Overall Seed column — GT only from submission.csv
print(f'\nGT from submission.csv: {len(gt_sub)} teams')
print(f'Test set columns: {test.columns.tolist()[:5]}... (no Overall Seed)')

# Check: which teams in GT are in train vs test set?
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
train_rids = set(train['RecordID'])
test_rids = set(test['RecordID'])
gt_in_train = sum(1 for r in gt_sub if r in train_rids)
gt_in_test = sum(1 for r in gt_sub if r in test_rids)
print(f'  GT teams in training set: {gt_in_train}')
print(f'  GT teams in test set: {gt_in_test}')

# Compare training GT with submission GT
train_gt = {}
for _, r in train.iterrows():
    s = pd.to_numeric(r.get('Overall Seed'), errors='coerce')
    if pd.notna(s) and s > 0:
        train_gt[r['RecordID']] = int(s)
print(f'  Training GT: {len(train_gt)} teams')

mismatches = 0
for rid in sorted(train_gt.keys()):
    if rid in gt_sub and train_gt[rid] != gt_sub[rid]:
        print(f'  TRAIN/SUB MISMATCH: {rid}: train={train_gt[rid]}, sub={gt_sub[rid]}')
        mismatches += 1
print(f'  Train vs Sub mismatches: {mismatches}')

# Also check submission format difference  
ds = pd.read_csv('my_submission_v10_deepstack.csv')
v13 = pd.read_csv('my_submission_v13c_rmse_best.csv')
print(f'\nDeepstack rows: {len(ds)}, template: {ds.columns.tolist()}')
print(f'v13c rows: {len(v13)}, template: {v13.columns.tolist()}')
print(f'Deepstack RecordIDs match template: {set(ds["RecordID"]) == set(template["RecordID"])}')
print(f'v13c RecordIDs match template: {set(v13["RecordID"]) == set(template["RecordID"])}')
print(f'Deepstack RecordIDs match test: {set(ds["RecordID"]) == set(test["RecordID"])}')

# Check RMSE with correct GT (submission.csv)
ds_preds = dict(zip(ds['RecordID'], ds['Overall Seed']))
v13_preds = dict(zip(v13['RecordID'], v13['Overall Seed']))

print('\n--- RMSE with submission.csv GT ---')
for name, preds in [('deepstack', ds_preds), ('v13c', v13_preds)]:
    sse = 0
    exact = 0
    n = 0
    for rid, gt in gt_sub.items():
        pred = preds.get(rid, 0)
        sse += (pred - gt) ** 2
        if pred == gt:
            exact += 1
        n += 1
    rmse = np.sqrt(sse / n)
    r451 = np.sqrt(sse / 451)
    print(f'  {name}: {exact}/{n} exact, RMSE={rmse:.4f}, RMSE/451={r451:.4f}, SSE={sse}')
