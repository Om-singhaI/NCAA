#!/usr/bin/env python3
"""Extract mid-range (17-34) test team details."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ncaa_2026_model import load_data, build_features
from sklearn.impute import KNNImputer

all_df, labeled, unlabeled, train_df, test_df, sub_df, GT = load_data()
tourn_rids = set(labeled['RecordID'].values)
context_df = pd.concat([
    train_df.drop(columns=['Overall Seed'], errors='ignore'),
    test_df.drop(columns=['Overall Seed'], errors='ignore')
], ignore_index=True)
feat = build_features(labeled, context_df, labeled, tourn_rids)
feature_names = list(feat.columns)
y = labeled['Overall Seed'].values.astype(float)
y_int = y.astype(int)
seasons = labeled['Season'].values.astype(str)
teams = labeled['Team'].values.astype(str)
conferences = labeled['Conference'].values.astype(str)
bid_types = labeled['Bid Type'].fillna('').values.astype(str)
test_rids = set(GT.keys())
record_ids = labeled['RecordID'].values.astype(str)
test_mask = np.array([rid in test_rids for rid in record_ids])

X_raw = np.where(np.isinf(feat.values.astype(np.float64)), np.nan,
                 feat.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(X_raw)

net_idx = feature_names.index('NET Rank')
sos_idx = feature_names.index('NETSOS')

power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12'}
mid_power = {'AAC','Mountain West','WCC','Atlantic 10','MVC','The American'}

# Mid-range test teams
mid_mask = test_mask & (y_int >= 17) & (y_int <= 34) 
mid_indices = np.where(mid_mask)[0]

print('='*100)
print(' MID-RANGE TEST TEAMS (Seeds 17-34)')
print('='*100)
print(f'  Total: {mid_mask.sum()}')
print()
fmt = '  {:<25s} {:<10s} {:<15s} {:<4s} {:>4s} {:>4s} {:>9s} {:<10s}'
print(fmt.format('Team','Season','Conference','Bid','NET','SOS','TrueSeed','ConfType'))
print(fmt.format('-'*25,'-'*10,'-'*15,'-'*4,'-'*4,'-'*4,'-'*9,'-'*10))

order = np.argsort(y_int[mid_indices])
for idx in order:
    i = mid_indices[idx]
    c = conferences[i]
    ct = 'POWER' if c in power else ('MID' if c in mid_power else 'LOW')
    print(fmt.format(
        teams[i], seasons[i], c, bid_types[i],
        f'{X_all[i,net_idx]:.0f}', f'{X_all[i,sos_idx]:.0f}',
        str(y_int[i]), ct
    ))

al_mid = sum(1 for i in mid_indices if bid_types[i] == 'AL')
aq_mid = sum(1 for i in mid_indices if bid_types[i] == 'AQ')
print(f'\n  By bid: AL={al_mid}, AQ={aq_mid}')

power_ct = sum(1 for i in mid_indices if conferences[i] in power)
mid_ct = sum(1 for i in mid_indices if conferences[i] in mid_power)
low_ct = len(mid_indices) - power_ct - mid_ct
print(f'  By conf: POWER={power_ct}, MID={mid_ct}, LOW={low_ct}')
