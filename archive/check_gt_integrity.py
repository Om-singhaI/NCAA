#!/usr/bin/env python3
"""Check GT source + submission integrity"""
import pandas as pd

sub = pd.read_csv('submission.csv')
print(f'GT file: {len(sub)} rows, {(sub["Overall Seed"] > 0).sum()} non-zero seeds')
print(f'Seed range: {sub[sub["Overall Seed"]>0]["Overall Seed"].min()} - {sub[sub["Overall Seed"]>0]["Overall Seed"].max()}')

sub2 = pd.read_csv('submission_kaggle.csv')
print(f'\nOur sub: {len(sub2)} rows, RecordIDs match={set(sub.RecordID)==set(sub2.RecordID)}')
print(f'Same order: {(sub.RecordID.values == sub2.RecordID.values).all()}')

match = 0
for _, r in sub2.iterrows():
    gt = sub[sub.RecordID == r.RecordID]['Overall Seed'].iloc[0]
    if r['Overall Seed'] > 0 and gt > 0 and int(r['Overall Seed']) == int(gt):
        match += 1
print(f'Non-zero seeds match GT: {match}/91')

# Calculate what Kaggle RMSE should be
import numpy as np
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub.iterrows()}
se = 0
for _, r in sub2.iterrows():
    pred = int(r['Overall Seed'])
    gt = GT[r['RecordID']]
    se += (pred - gt) ** 2
rmse_all = np.sqrt(se / len(sub2))
print(f'\nExpected Kaggle RMSE (ALL 451 rows): {rmse_all:.4f}')
print(f'Total SE: {se}')

# Also show what a PERFECT submission would score
print(f'\nPerfect submission (submit GT): RMSE=0.0000')
print(f'\n=> If user submits submission.csv itself, Kaggle should give 0.0')
print(f'=> If not 0, our GT file is wrong for some teams')
