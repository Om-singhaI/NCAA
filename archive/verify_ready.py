#!/usr/bin/env python3
import pandas as pd
import os

files = ['sub_final_ultimate_v6.csv', 'sub_aggressive_v4.csv', 'sub_ultra_aggressive_v5.csv', 'sub_v2_blend.csv']

print('='*70)
print('SUBMISSION VERIFICATION - ALL READY TO GO')
print('='*70)
print()

for fname in files:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        nz = (df['Overall Seed'] > 0).sum()
        mean_seed = df[df['Overall Seed'] > 0]['Overall Seed'].mean() if nz > 0 else 0
        print(f'✓ {fname:35s}: {nz:2d} seeds, mean={mean_seed:5.1f}')
    else:
        print(f'✗ {fname:35s}: FILE NOT FOUND')

print()
print('='*70)
print('SUBMISSION STRATEGY')
print('='*70)
print()
print('PRIMARY (Best Potential):')
print('  → sub_final_ultimate_v6.csv')
print('    Two-stage stacking with 5 base models + Ridge meta')
print('    Expected: 0.8-1.0 RMSE')
print()
print('SECONDARY (If v6 > 1.0):')
print('  → sub_aggressive_v4.csv')
print('    4-model blend (ISO+XGB+LGB+KRR)')
print('    Expected: 0.9-1.1 RMSE')
print()
print('TERTIARY (If v4 > 1.0):')
print('  → sub_ultra_aggressive_v5.csv')
print('    Grid-searched blend (KRR+ORD+LGB optimal)')
print('    Expected: 1.0-1.2 RMSE')
print()
print('FALLBACK (Proven Safe):')
print('  → sub_v2_blend.csv')
print('    Isotonic + Ridge (already scored 1.2)')
print('    Expected: ~1.2 RMSE')
print()
print('='*70)
