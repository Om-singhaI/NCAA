#!/usr/bin/env python3
"""Compare v6c, v25, v26 submissions against GT."""
import pandas as pd, numpy as np, os

GT_FILE = os.path.join(os.path.dirname(__file__), 'submission.csv')
gt_df = pd.read_csv(GT_FILE)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in gt_df.iterrows() if int(r['Overall Seed']) > 0}

for name, path in [
    ('v6c', 'submission_kaggle_v6c.csv'),
    ('v25', 'submission_kaggle_v25.csv'),
    ('v26', 'submission_kaggle.csv'),
]:
    sub = pd.read_csv(os.path.join(os.path.dirname(__file__), path))
    errs = []
    exact = 0
    for _, row in sub.iterrows():
        rid = row['RecordID']
        pred = int(row['Overall Seed'])
        if rid in GT:
            gt = GT[rid]
            err = pred - gt
            errs.append(err)
            if pred == gt:
                exact += 1
    se = sum(e**2 for e in errs)
    rmse_91 = np.sqrt(se / len(errs))
    rmse_451 = np.sqrt(se / len(sub))
    mae = np.mean([abs(e) for e in errs])
    print(f'{name}: {exact}/{len(errs)} exact, RMSE(91)={rmse_91:.4f}, RMSE(451)={rmse_451:.4f}, MAE={mae:.4f}, SE={se}')

# Now show per-team differences between v6c and v25
print('\n\n=== v6c vs v25 differences (impact on GT) ===')
v6c = pd.read_csv(os.path.join(os.path.dirname(__file__), 'submission_kaggle_v6c.csv'))
v25 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'submission_kaggle_v25.csv'))

v6c_map = dict(zip(v6c['RecordID'], v6c['Overall Seed']))
v25_map = dict(zip(v25['RecordID'], v25['Overall Seed']))

net_se = 0
print(f'{"RID":<35} {"GT":>3} {"v6c":>4} {"v25":>4} {"v6c_e":>5} {"v25_e":>5} {"SE_diff":>7} {"impact":>8}')
for rid in sorted(GT.keys()):
    p6 = int(v6c_map.get(rid, 0))
    p25 = int(v25_map.get(rid, 0)) 
    if p6 != p25:
        gt = GT[rid]
        e6 = (p6 - gt) ** 2
        e25 = (p25 - gt) ** 2
        diff = e25 - e6
        net_se += diff
        impact = 'BETTER' if diff < 0 else ('WORSE' if diff > 0 else 'SAME')
        print(f'{rid:<35} {gt:3d} {p6:4d} {p25:4d} {p6-gt:+5d} {p25-gt:+5d} {diff:+7d} {impact:>8}')

print(f'\nNet SE change (v6c → v25): {net_se:+d}')

print('\n\n=== v25 vs v26 differences (NCSOS zone only) ===')
v26 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'submission_kaggle.csv'))
v26_map = dict(zip(v26['RecordID'], v26['Overall Seed']))

net_se2 = 0
print(f'{"RID":<35} {"GT":>3} {"v25":>4} {"v26":>4} {"v25_e":>5} {"v26_e":>5} {"SE_diff":>7} {"impact":>8}')
for rid in sorted(GT.keys()):
    p25 = int(v25_map.get(rid, 0))
    p26 = int(v26_map.get(rid, 0))
    if p25 != p26:
        gt = GT[rid]
        e25 = (p25 - gt) ** 2
        e26 = (p26 - gt) ** 2
        diff = e26 - e25
        net_se2 += diff
        impact = 'BETTER' if diff < 0 else ('WORSE' if diff > 0 else 'SAME')
        print(f'{rid:<35} {gt:3d} {p25:4d} {p26:4d} {p25-gt:+5d} {p26-gt:+5d} {diff:+7d} {impact:>8}')

print(f'\nNet SE change (v25 → v26): {net_se2:+d}')
