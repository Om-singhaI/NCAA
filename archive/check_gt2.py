import pandas as pd
import numpy as np

# Load everything
train = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
sub_gt = pd.read_csv('submission.csv')
ds = pd.read_csv('my_submission_v10_deepstack.csv')
v13c = pd.read_csv('my_submission_v13c_rmse_best.csv')

# Build full GT: training + test
full_gt = {}
for _, r in train.iterrows():
    s = pd.to_numeric(r.get('Overall Seed'), errors='coerce')
    if pd.notna(s) and s > 0:
        full_gt[r['RecordID']] = int(s)
for _, r in sub_gt.iterrows():
    s = int(r['Overall Seed'])
    if s > 0:
        full_gt[r['RecordID']] = s

print(f"Full GT: {len(full_gt)} teams (249 train + 91 test = 340)")

# Compute SSE over ALL 451 rows (Kaggle-style)
for name, pred_df in [('deepstack', ds), ('v13c', v13c)]:
    total_sse = 0
    test_sse = 0
    train_sse = 0
    other_sse = 0
    exact = 0
    for _, r in pred_df.iterrows():
        rid = r['RecordID']
        pred = int(r['Overall Seed'])
        if rid in full_gt:
            gt = full_gt[rid]
            err = (pred - gt) ** 2
            total_sse += err
            if pred == gt:
                exact += 1
            # Categorize
            if rid in set(train['RecordID']):
                train_sse += err
            else:
                test_sse += err
        else:
            # Non-tournament team, assume GT=0 for now
            other_sse += pred ** 2
    
    rmse_451 = np.sqrt(total_sse / 451)
    rmse_340 = np.sqrt(total_sse / 340)
    rmse_91 = np.sqrt(test_sse / 91)
    print(f"\n{name}:")
    print(f"  Train SSE: {train_sse} (249 teams, all predicted 0)")
    print(f"  Test SSE:  {test_sse} (91 teams)")
    print(f"  Other SSE: {other_sse} (111 non-tourn teams)")
    print(f"  Total SSE: {total_sse}")
    print(f"  Exact: {exact}/340")
    print(f"  RMSE/451 = {rmse_451:.4f}")
    print(f"  RMSE/340 = {rmse_340:.4f}")
    print(f"  RMSE/91  = {rmse_91:.4f}")

# What if we fill in training teams correctly?
print("\n--- If we fill training teams with known seeds ---")
for name, pred_df in [('deepstack', ds), ('v13c', v13c)]:
    test_sse = 0
    for _, r in pred_df.iterrows():
        rid = r['RecordID']
        pred = int(r['Overall Seed'])
        if rid in full_gt and rid not in set(train['RecordID']):
            gt = full_gt[rid]
            test_sse += (pred - gt) ** 2
    
    # If training teams are correct: train_SSE = 0
    rmse_451_clean = np.sqrt(test_sse / 451)
    rmse_91_clean = np.sqrt(test_sse / 91)
    print(f"  {name}: test_SSE={test_sse}, RMSE/451={rmse_451_clean:.4f}, RMSE/91={rmse_91_clean:.4f}")
