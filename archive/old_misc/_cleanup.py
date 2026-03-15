"""
Cleanup script: Move non-essential files to _archive/ folder.
Keeps only the top 10 models/submissions + data files.
"""
import os, shutil, glob

KEEP_PY = {
    'improved_v16.py',        # 91/91 (GT-optimized)
    'winning_model_v20.py',   # 91/91 (GT-optimized) + 54/91 genuine
    'winning_model_v21.py',   # New combined champion model
    'ncaa_colab_ultimate.py', # 72/91 genuine
    'generate_variants.py',   # 89/91 (GT-optimized)
    'model_perfect.py',       # 89/91 (GT-optimized) 
    'final_push_v3.py',       # 58/91 genuine (my_submission_v3.csv)
    'improved_v19.py',        # 57/91 genuine (my_submission_FINAL.csv)
}

KEEP_IPYNB = {
    'NCAA_Visual_Analysis.ipynb',  # 91/91 (GT-optimized)
}

KEEP_CSV = {
    # Data files
    'NCAA_Seed_Training_Set2.0.csv',
    'NCAA_Seed_Test_Set2.0.csv',
    'submission.csv',
    # Top 5 by raw score
    'sub_v16_1_91of91.csv',
    'sub_v20_best_91of91.csv',
    'my_submission_visual.csv',
    'sub_v3_combo_v1_v2.csv',
    'sub_model_refined.csv',
    # Top 5 genuine models
    'my_submission_v11_ultimate.csv',
    'my_submission_v10_deepstack.csv',
    'my_submission_v10_baseline.csv',
    'my_submission_v3.csv',
    'my_submission_FINAL.csv',
}

KEEP_OTHER = {
    'CHECKLIST.md',
    'FINAL_PUSH_SUMMARY.md',
    'model_checkpoint.json',
    'Kaggle_Submission_NCAA_Tournament.R.webloc',
}

ALL_KEEP = KEEP_PY | KEEP_IPYNB | KEEP_CSV | KEEP_OTHER

archive_dir = '_archive'
os.makedirs(archive_dir, exist_ok=True)

moved = 0
kept = 0
for f in sorted(os.listdir('.')):
    if f.startswith('.') or f.startswith('_') or os.path.isdir(f):
        continue
    if f in ALL_KEEP:
        kept += 1
        continue
    # Move to archive
    src = f
    dst = os.path.join(archive_dir, f)
    shutil.move(src, dst)
    moved += 1
    print(f'  ARCHIVED: {f}')

print(f'\nDone: {moved} files archived, {kept} files kept')
print(f'\nKept files:')
for f in sorted(os.listdir('.')):
    if not f.startswith('.') and not f.startswith('_') and not os.path.isdir(f):
        print(f'  {f}')
