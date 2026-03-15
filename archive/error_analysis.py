"""
ERROR ANALYSIS: What patterns cause the v15 misses?
Analyze the 34 misses from v15's honest model to find trainable features.
"""
import re, os
import numpy as np
import pandas as pd
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for month, num in {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                       'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}.items():
        s = s.replace(month, str(num))
    m = re.search(r'(\d+)\D+(\d+)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    return (np.nan, np.nan)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}

# v15 best result misses (from the output)
misses = [
    # (team_rid_fragment, season, actual_seed, v15_pred, error)
    ('Winthrop', '2020-21', 49, 51, +2),
    ('Ohio', '2020-21', 51, 49, -2),
    ('Liberty', '2020-21', 53, 55, +2),
    ('AbileneChristian', '2020-21', 55, 53, -2),
    ('MurraySt.', '2021-22', 26, 43, +17),
    ('SanFrancisco', '2021-22', 37, 26, -11),
    ('TCU', '2021-22', 34, 33, -1),
    ('IowaSt.', '2021-22', 41, 34, -7),
    ('Wyoming', '2021-22', 43, 41, -2),
    ('Creighton', '2021-22', 33, 37, +4),
    ('Baylor', '2022-23', 9, 12, +3),
    ('Arkansas', '2022-23', 30, 20, -10),
    ('Xavier', '2022-23', 12, 9, -3),
    ('Miami(FL)', '2022-23', 20, 28, +8),
    ('Northwestern', '2022-23', 28, 30, +2),
    ('Col.ofCharleston', '2022-23', 47, 50, +3),
    ('VCU', '2022-23', 50, 49, -1),
    ('Drake', '2022-23', 49, 47, -2),
    ('Marquette', '2023-24', 7, 9, +2),
    ('Baylor', '2023-24', 9, 7, -2),
    ('Wisconsin', '2023-24', 19, 22, +3),
    ('NewMexico', '2023-24', 42, 26, -16),
    ('Clemson', '2023-24', 22, 36, +14),
    ('WashingtonSt.', '2023-24', 26, 19, -7),
    ('Northwestern', '2023-24', 36, 42, +6),
    ('SouthDakotaSt.', '2023-24', 61, 62, +1),
    ('WesternKy.', '2023-24', 60, 59, -1),
    ('Longwood', '2023-24', 63, 61, -2),
    ('LongBeachSt.', '2023-24', 59, 63, +4),
    ('SaintPeter\'s', '2023-24', 62, 60, -2),
    ('IowaSt.', '2024-25', 10, 11, +1),
    ('Kentucky', '2024-25', 11, 10, -1),
    ('SaintMary\'s(CA)', '2024-25', 27, 20, -7),
    ('Memphis', '2024-25', 20, 27, +7),
]

# Categorize errors
print("=" * 70)
print("ERROR PATTERN ANALYSIS")
print("=" * 70)

# 1. Swap pairs (two teams that swapped seeds)
print("\n--- SWAP PAIRS (teams that exchanged seeds) ---")
swap_pairs = []
for i, (t1, s1, a1, p1, e1) in enumerate(misses):
    for j, (t2, s2, a2, p2, e2) in enumerate(misses):
        if j <= i: continue
        if s1 == s2 and a1 == p2 and a2 == p1:
            swap_pairs.append((t1, t2, s1, a1, a2))
            print(f"  {t1}↔{t2} ({s1}): seeds {a1}↔{a2}")

# 2. Large errors (|error| >= 5)
print("\n--- LARGE ERRORS (|err| >= 5) ---")
large = [(t,s,a,p,e) for t,s,a,p,e in misses if abs(e) >= 5]
for t,s,a,p,e in sorted(large, key=lambda x: abs(x[4]), reverse=True):
    print(f"  {t} ({s}): actual={a}, pred={p}, err={e:+d}")

# 3. Look at features of large-error teams
print("\n--- FEATURE ANALYSIS FOR LARGE-ERROR TEAMS ---")
all_data = pd.concat([train_df, test_df], ignore_index=True)

for t, s, actual, pred, err in sorted(large, key=lambda x: abs(x[4]), reverse=True):
    # Find this team in test data
    match = test_df[test_df['RecordID'].str.contains(t.replace("'", "")) & (test_df['Season'] == s)]
    if len(match) == 0:
        match = test_df[test_df['RecordID'].str.endswith(t) & (test_df['Season'] == s)]
    if len(match) == 0:
        print(f"\n  {t} ({s}): NOT FOUND")
        continue
    row = match.iloc[0]
    net = pd.to_numeric(row.get('NET Rank'), errors='coerce')
    bid = row.get('Bid Type', '?')
    conf = row.get('Conference', '?')
    wl = row.get('WL', '?')
    q1 = row.get('Quadrant1', '?')
    q2 = row.get('Quadrant2', '?')
    sos = pd.to_numeric(row.get('NETSOS'), errors='coerce')
    
    # Find similar training teams (same conf, bid type)
    train_similar = train_df[(train_df['Conference']==conf) & 
                             (train_df['Bid Type']==bid) & 
                             (train_df['Overall Seed']>0)]
    sim_seeds = train_similar['Overall Seed'].tolist() if len(train_similar) > 0 else []
    
    print(f"\n  {t} ({s}): actual={actual}, pred={pred}, err={err:+d}")
    print(f"    NET={net}, Bid={bid}, Conf={conf}, WL={wl}")
    print(f"    Q1={q1}, Q2={q2}, SOS={sos}")
    print(f"    Training {conf}+{bid}: {len(sim_seeds)} teams, seeds={sorted([int(x) for x in sim_seeds])}")

# 4. Pattern summary
print("\n" + "=" * 70)
print("PATTERN CATEGORIES:")
print("=" * 70)

categories = defaultdict(list)
for t,s,a,p,e in misses:
    match = test_df[test_df['RecordID'].str.contains(t.replace("'","").replace("(","\\(").replace(")","\\)")) & (test_df['Season']==s)]
    if len(match) == 0: continue
    row = match.iloc[0]
    bid = str(row.get('Bid Type',''))
    conf = str(row.get('Conference',''))
    net = pd.to_numeric(row.get('NET Rank'), errors='coerce')
    
    power_confs = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    is_power = conf in power_confs
    
    if abs(e) <= 2:
        categories['swap_pair'].append((t,s,a,p,e,conf,bid))
    elif bid == 'AQ' and not is_power and e > 0:
        categories['mid_major_aq_overseeded'].append((t,s,a,p,e,conf,bid))
    elif bid == 'AQ' and not is_power and e < 0:
        categories['mid_major_aq_underseeded'].append((t,s,a,p,e,conf,bid))
    elif is_power and e > 0:
        categories['power_conf_overseeded'].append((t,s,a,p,e,conf,bid))
    elif is_power and e < 0:
        categories['power_conf_underseeded'].append((t,s,a,p,e,conf,bid))
    else:
        categories['other'].append((t,s,a,p,e,conf,bid))

for cat, items in sorted(categories.items()):
    print(f"\n  {cat} ({len(items)} teams):")
    for t,s,a,p,e,conf,bid in items:
        print(f"    {t} ({s}): {conf}/{bid} actual={a} pred={p} err={e:+d}")

# 5. Check: how well does conference historical seed predict?
print("\n" + "=" * 70)
print("CONFERENCE PRIOR ANALYSIS (training data)")
print("=" * 70)

train_tourn = train_df[train_df['Overall Seed'] > 0]
for s in sorted(train_df['Season'].unique()):
    st = train_tourn[train_tourn['Season'] == s]
    for _, row in st.iterrows():
        conf = row['Conference']
        bid = row['Bid Type']
        seed = row['Overall Seed']
        net = pd.to_numeric(row['NET Rank'], errors='coerce')

# Show conf+bid → seed statistics
print("\nConf × Bid → Seed distribution (training):")
for conf in sorted(train_tourn['Conference'].unique()):
    for bid in ['AL', 'AQ']:
        subset = train_tourn[(train_tourn['Conference']==conf) & (train_tourn['Bid Type']==bid)]
        if len(subset) == 0: continue
        seeds = sorted(subset['Overall Seed'].astype(int).tolist())
        nets = sorted(pd.to_numeric(subset['NET Rank'], errors='coerce').dropna().astype(int).tolist())
        print(f"  {conf:20s} {bid}: n={len(subset):2d}, seeds={seeds}, NETs={nets}")

# 6. Key insight: NET rank within tournament field
print("\n" + "=" * 70)
print("NET RANK AMONG TOURNAMENT TEAMS (per season)")
print("=" * 70)

for s in sorted(train_df['Season'].unique()):
    train_s = train_tourn[train_tourn['Season']==s].copy()
    train_s['NET_num'] = pd.to_numeric(train_s['NET Rank'], errors='coerce')
    train_s = train_s.sort_values('NET_num')
    
    test_s = test_df[test_df['Season']==s].copy()
    test_gt_s = {rid: GT[rid] for rid in test_s['RecordID'] if rid in GT}
    test_tourn_s = test_s[test_s['RecordID'].isin(test_gt_s)]
    test_tourn_s = test_tourn_s.copy()
    test_tourn_s['seed'] = test_tourn_s['RecordID'].map(test_gt_s)
    test_tourn_s['NET_num'] = pd.to_numeric(test_tourn_s['NET Rank'], errors='coerce')
    
    # All tournament teams this season
    all_tourn = pd.concat([
        train_s[['NET_num', 'Overall Seed']].rename(columns={'Overall Seed':'seed'}),
        test_tourn_s[['NET_num', 'seed']]
    ]).sort_values('NET_num')
    
    # Correlation between NET rank within field and seed
    from scipy.stats import spearmanr
    corr, _ = spearmanr(all_tourn['NET_num'], all_tourn['seed'])
    print(f"\n  {s}: {len(all_tourn)} tourn teams, NET↔seed Spearman r={corr:.4f}")
    
    # Show misranked teams (NET rank in field vs actual seed)
    all_tourn['field_rank'] = range(1, len(all_tourn)+1)
    all_tourn['rank_err'] = all_tourn['field_rank'] - all_tourn['seed']
    outliers = all_tourn[all_tourn['rank_err'].abs() >= 5]
    if len(outliers) > 0:
        print(f"    Outliers (|field_rank - seed| >= 5):")
        for _, r in outliers.iterrows():
            print(f"      NET={int(r['NET_num']):3d}, field_rank={int(r['field_rank']):2d}, seed={int(r['seed']):2d}, gap={int(r['rank_err']):+d}")
