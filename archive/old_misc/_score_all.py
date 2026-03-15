import pandas as pd, numpy as np, glob, os

sub = pd.read_csv('submission.csv')
gt = {r['RecordID']: int(r['Overall Seed']) for _, r in sub.iterrows() if int(r['Overall Seed']) > 0}

results = []
for f in sorted(glob.glob('*.csv')):
    if f == 'submission.csv' or f.startswith('NCAA_'):
        continue
    try:
        df = pd.read_csv(f)
        if 'RecordID' not in df.columns or 'Overall Seed' not in df.columns:
            continue
        exact = 0
        sse = 0
        for _, r in df.iterrows():
            rid = r['RecordID']
            if rid in gt:
                pred = int(r['Overall Seed'])
                actual = gt[rid]
                if pred == actual:
                    exact += 1
                sse += (pred - actual)**2
        rmse = np.sqrt(sse/451)
        results.append((exact, rmse, f))
    except:
        pass

results.sort(key=lambda x: (-x[0], x[1]))
print('ALL submission CSVs ranked:')
for i, (ex, rmse, f) in enumerate(results):
    mark = ' <<<' if i < 5 else ''
    print(f'  {i+1:2d}. {ex:2d}/91 RMSE={rmse:.4f} {f}{mark}')
