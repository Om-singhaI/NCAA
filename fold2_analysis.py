"""
Analyze what made Fold 2 special (2.17 RMSE) and replicate it
Key insight: Fold 2 might contain specific seasons - analyze season patterns
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    m = re.search(r"(\d+)[^\d]+(\d+)", str(s))
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (np.nan, np.nan)

def parse_quad(s):
    if pd.isna(s) or str(s).strip() == '':
        return np.nan
    m = re.search(r"(\d+)[^\d]+(\d+)", str(s))
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)", str(s))
    if m2:
        return int(m2.group(1))
    return np.nan

# Load data
df_train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
df_train['Overall Seed'] = pd.to_numeric(df_train['Overall Seed'], errors='coerce').fillna(0)

# Get fold splits
gkf = GroupKFold(n_splits=5)
groups = df_train['Season'].values

print("=" * 80)
print("FOLD ANALYSIS - What makes Fold 2 special?")
print("=" * 80)

for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(df_train, groups=groups)):
    val_data = df_train.iloc[val_idx]
    seasons = sorted(val_data['Season'].unique())
    selected_count = (val_data['Overall Seed'] > 0).sum()
    
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Validation seasons: {seasons}")
    print(f"  Samples: {len(val_data)}")
    print(f"  Selected teams: {selected_count}/{len(val_data)}")
    print(f"  Seed range: {val_data['Overall Seed'].min():.0f} - {val_data['Overall Seed'].max():.0f}")
    print(f"  Mean seed: {val_data[val_data['Overall Seed']>0]['Overall Seed'].mean():.2f}")
    print(f"  Difficulty: {'EASIER (lower variance)' if val_data[val_data['Overall Seed']>0]['Overall Seed'].std() < 8 else 'HARDER (higher variance)'}")

# FOLD 2 SPECIFIC ANALYSIS (assuming 5-fold)
print("\n" + "=" * 80)
print("FOLD 2 DEEP DIVE")
print("=" * 80)

fold_idx = 1  # 0-indexed, so fold 2 is index 1
folds = list(gkf.split(df_train, groups=groups))
_, fold2_val_idx = folds[fold_idx]
fold2_data = df_train.iloc[fold2_val_idx]

print(f"\nFold 2 contains seasons: {sorted(fold2_data['Season'].unique())}")
print(f"Total samples: {len(fold2_data)}")
print(f"Selected: {(fold2_data['Overall Seed'] > 0).sum()}")

# Analyze by season within fold 2
print("\nFold 2 breakdown by season:")
for season in sorted(fold2_data['Season'].unique()):
    season_data = fold2_data[fold2_data['Season'] == season]
    sel = (season_data['Overall Seed'] > 0).sum()
    print(f"  Season {season}: {len(season_data)} teams, {sel} selected")

# Seed difficulty in fold2
print("\nFold 2 seed characteristics:")
selected_seeds = fold2_data[fold2_data['Overall Seed'] > 0]['Overall Seed']
print(f"  Mean seed: {selected_seeds.mean():.2f}")
print(f"  Std dev: {selected_seeds.std():.2f}")
print(f"  Seeds 1-16: {(selected_seeds <= 16).sum()}")
print(f"  Seeds 17-40: {((selected_seeds > 16) & (selected_seeds <= 40)).sum()}")
print(f"  Seeds 41+: {(selected_seeds > 40).sum()}")
