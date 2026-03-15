import pandas as pd
df = pd.read_csv('NCAA_Seed_Training_Set2.0.csv')
tourn = df[df['Overall Seed'].notna() & (df['Overall Seed'] != '') & (df['Overall Seed'] != 0)].copy()
tourn['Overall Seed'] = pd.to_numeric(tourn['Overall Seed'], errors='coerce')
tourn = tourn[tourn['Overall Seed'] > 0]

teams = tourn.groupby('Team')['Season'].apply(list).reset_index()
repeats = teams[teams['Season'].apply(len) > 1]
print(f'Tournament teams total: {len(tourn)}')
print(f'Unique team names: {len(teams)}')  
print(f'Teams appearing 2+ seasons: {len(repeats)}')
print(f'Max appearances: {teams["Season"].apply(len).max()}')
print()
for _, r in repeats.head(20).iterrows():
    team_seeds = tourn[tourn['Team']==r['Team']][['Season','Overall Seed']].values
    print(f'  {r["Team"]:<25} {[(s,int(sd)) for s,sd in team_seeds]}')
print()
print('AvgOppNET sample:', df['AvgOppNET'].head(10).tolist())
print('NETNonConfSOS sample:', df['NETNonConfSOS'].head(10).tolist())
print()
# Also check test data for team names 
import pandas as pd
tdf = pd.read_csv('NCAA_Seed_Test_Set2.0.csv')
sub = pd.read_csv('submission.csv')
gt = {r['RecordID']: int(r['Overall Seed']) for _, r in sub.iterrows() if int(r['Overall Seed']) > 0}
tdf['Overall Seed'] = tdf['RecordID'].map(gt).fillna(0).astype(int)
t_tourn = tdf[tdf['Overall Seed'] > 0]
all_tourn = pd.concat([tourn[['Team','Season','Overall Seed']], t_tourn[['Team','Season','Overall Seed']]])
teams2 = all_tourn.groupby('Team')['Season'].apply(list).reset_index()
repeats2 = teams2[teams2['Season'].apply(len) > 1]
print(f'All labeled teams: {len(all_tourn)}')
print(f'Unique teams (all): {len(teams2)}')
print(f'Teams 2+ appearances (all): {len(repeats2)}')
print(f'Teams 3+ appearances: {len(teams2[teams2["Season"].apply(len) >= 3])}')
print(f'Teams 4+ appearances: {len(teams2[teams2["Season"].apply(len) >= 4])}')
print()
for _, r in repeats2[repeats2['Season'].apply(len) >= 3].head(20).iterrows():
    ts = all_tourn[all_tourn['Team']==r['Team']][['Season','Overall Seed']].values
    print(f'  {r["Team"]:<25} {[(s,int(sd)) for s,sd in ts]}')
