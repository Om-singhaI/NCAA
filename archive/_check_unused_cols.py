#!/usr/bin/env python3
"""Quick check of unused data columns"""
import pandas as pd
import numpy as np

train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
test = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
all_df = pd.concat([train, test])

print('AvgOppNET sample:')
labeled = all_df[all_df['Overall Seed']>0].head(10)
for _, r in labeled.iterrows():
    print(f"  {r['Team']:25s} NET={r['NET Rank']:5.0f} OppRank={r['AvgOppNETRank']:6.0f} OppNET={r['AvgOppNET']:8.2f} SOS={r['NETSOS']:6.0f} NCS={r['NETNonConfSOS']:6.0f}")

print()
labeled_full = all_df[all_df['Overall Seed']>0].copy()
for col in ['AvgOppNET', 'NETNonConfSOS', 'NET Rank', 'NETSOS', 'AvgOppNETRank']:
    vals = pd.to_numeric(labeled_full[col], errors='coerce')
    seeds = labeled_full['Overall Seed']
    corr = vals.corr(seeds)
    print(f'{col:20s} corr with seed: {corr:.4f}')

print()
# Check the error teams specifically
sub = pd.read_csv('submission.csv')
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub.iterrows() if int(r['Overall Seed'])>0}
errors_rids = [
    '2021-22-IowaSt.', '2021-22-MurraySt.', '2021-22-Richmond', '2021-22-SouthDakotaSt.',
    '2021-22-SouthernCalifornia', '2022-23-Col.ofCharleston', '2022-23-Miami(FL)',
    '2022-23-SanDiegoSt.', '2022-23-SoutheastMo.St.', '2022-23-TexasSouthern',
    '2022-23-VCU', '2023-24-Clemson', '2023-24-LongBeachSt.', '2023-24-NewMexico',
    '2023-24-Northwestern', '2023-24-SouthCarolina', '2023-24-SouthDakotaSt.',
    '2023-24-WashingtonSt.', '2023-24-WesternKy.', '2024-25-Kentucky', '2024-25-Wisconsin'
]
print("Error team features (unused cols):")
merged = pd.concat([train, test])
for rid in errors_rids:
    row = merged[merged['RecordID'] == rid]
    if len(row) > 0:
        r = row.iloc[0]
        gt = GT.get(rid, 0)
        oppnet = pd.to_numeric(r.get('AvgOppNET', np.nan), errors='coerce')
        ncsos = pd.to_numeric(r.get('NETNonConfSOS', np.nan), errors='coerce')
        net = pd.to_numeric(r.get('NET Rank', np.nan), errors='coerce')
        sos = pd.to_numeric(r.get('NETSOS', np.nan), errors='coerce')
        print(f"  {rid:35s} GT={gt:2d} NET={net:3.0f} OppNET={oppnet:6.1f} NCS={ncsos:3.0f} SOS={sos:3.0f}")
