#!/usr/bin/env python3
"""
Quick verification of my_submission.csv
"""
import pandas as pd

sub = pd.read_csv('my_submission.csv')
print("✅ SUBMISSION VERIFICATION")
print(f"Rows: {len(sub)}")
print(f"Columns: {list(sub.columns)}")
print(f"\nSeed statistics:")
print(f"  Mean: {sub['Overall Seed'].mean():.2f}")
print(f"  Std: {sub['Overall Seed'].std():.2f}")
print(f"  Min: {sub['Overall Seed'].min():.2f}")
print(f"  Max: {sub['Overall Seed'].max():.2f}")
print(f"  Teams with seed > 0: {(sub['Overall Seed'] > 0).sum()}")
print(f"  Teams with seed = 0: {(sub['Overall Seed'] == 0).sum()}")

print(f"\nNo NaNs: {sub.isna().sum().sum() == 0}")
print(f"All seeds in [0, 68]: {((sub['Overall Seed'] >= 0) & (sub['Overall Seed'] <= 68)).all()}")

print("\nTop 10 teams (best seeds):")
top = sub.nsmallest(10, 'Overall Seed')[['RecordID', 'Overall Seed']]
print(top.to_string(index=False))

print("\n✅ Ready for Kaggle submission!")
