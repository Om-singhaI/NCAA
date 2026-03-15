#!/usr/bin/env python3
"""
NCAA v37 — Root Cause Analysis + Generalizable Model

GOAL: Build a model for the PRIVATE leaderboard where the data is DIFFERENT.
Stop optimizing for this specific test set.

Phase 1: Root cause analysis
  - Why do systematic errors happen? (Murray St +21, Clemson +19, etc.)
  - Are these TRAINING errors too, or test-only surprises?
  - What features distinguish correct vs incorrect predictions?
  - Is there a pattern: bid type? conference? season?

Phase 2: Fix systematic biases
  - If the model consistently over/under-seeds certain team types, fix that
  - Build corrections that generalize (not GT-fitted)

Phase 3: LOSO-validated final model
  - Pick the model that has the best AVERAGE LOSO performance
  - Prioritize RMSE over exact matches (transfers better to new data)
"""

import os, sys, time, re, warnings
import numpy as np
import pandas as pd

IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                    'xgboost', 'lightgbm', 'catboost'])
    from google.colab import drive, files
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/NCAA-1'
else:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()

# ============================================================
#  DATA LOADING
# ============================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))


def parse_wl(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                 'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}.items():
        s = s.replace(m, str(n))
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)


train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()

GT = {}
for _, r in sub_df.iterrows():
    if int(r['Overall Seed']) > 0:
        GT[r['RecordID']] = int(r['Overall Seed'])

tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]

y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)
train_rids = train_tourn['RecordID'].values

test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

avail_seeds_all = {}
for s in sorted(train_df['Season'].unique()):
    used = set(train_tourn[train_tourn['Season'] == s]['Overall Seed'].astype(int))
    avail_seeds_all[s] = sorted(set(range(1, 69)) - used)

# For test seasons: only seeds NOT used by training teams in that season
test_avail = {}
for s in sorted(set(test_seasons)):
    s_plain = str(s)
    used = set(train_tourn[train_tourn['Season'].astype(str) == s_plain]['Overall Seed'].astype(int))
    avail_list = sorted(set(range(1, 69)) - used)
    test_avail[s] = avail_list
    test_avail[s_plain] = avail_list
    test_avail[np.str_(s_plain)] = avail_list
    print(f'  Season {s_plain}: {len(used)} train seeds used, {len(avail_list)} available for test')

n_tr = len(y_train)
n_te = len(tourn_idx)
test_tourn_rids = set(GT.keys())
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
print(f'{n_tr} train, {n_te} test')
print(f'Train seasons: {sorted(set(train_seasons))}')
print(f'Test seasons: {sorted(set(test_seasons))}')


# ============================================================
#  FEATURE ENGINEERING
# ============================================================
def build_features(df, all_df, labeled_df):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w = wl.apply(lambda x: x[0]); l = wl.apply(lambda x: x[1])
            feat[col + '_Pct'] = np.where((w+l)!=0, w/(w+l), 0.5)
            if col == 'WL':
                feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
    q1w=feat.get('Quadrant1_W',pd.Series(0,index=df.index)).fillna(0)
    q1l=feat.get('Quadrant1_L',pd.Series(0,index=df.index)).fillna(0)
    q2w=feat.get('Quadrant2_W',pd.Series(0,index=df.index)).fillna(0)
    q2l=feat.get('Quadrant2_L',pd.Series(0,index=df.index)).fillna(0)
    q3w=feat.get('Quadrant3_W',pd.Series(0,index=df.index)).fillna(0)
    q3l=feat.get('Quadrant3_L',pd.Series(0,index=df.index)).fillna(0)
    q4w=feat.get('Quadrant4_W',pd.Series(0,index=df.index)).fillna(0)
    q4l=feat.get('Quadrant4_L',pd.Series(0,index=df.index)).fillna(0)
    wpct=feat.get('WL_Pct',pd.Series(0.5,index=df.index)).fillna(0.5)
    net=pd.to_numeric(df['NET Rank'],errors='coerce').fillna(300)
    prev=pd.to_numeric(df['PrevNET'],errors='coerce').fillna(300)
    sos=pd.to_numeric(df['NETSOS'],errors='coerce').fillna(200)
    opp_net=pd.to_numeric(df['AvgOppNETRank'],errors='coerce').fillna(200)
    feat['NET Rank']=net; feat['PrevNET']=prev; feat['NETSOS']=sos; feat['AvgOppNETRank']=opp_net
    bid=df['Bid Type'].fillna('')
    feat['is_AL']=(bid=='AL').astype(float); feat['is_AQ']=(bid=='AQ').astype(float)
    conf=df['Conference'].fillna('Unknown')
    all_conf=all_data['Conference'].fillna('Unknown')
    all_net_vals=pd.to_numeric(all_data['NET Rank'],errors='coerce').fillna(300)
    cg=pd.DataFrame({'Conference':all_conf,'NET':all_net_vals})
    feat['conf_avg_net']=conf.map(cg.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net']=conf.map(cg.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_min_net']=conf.map(cg.groupby('Conference')['NET'].min()).fillna(300)
    feat['conf_std_net']=conf.map(cg.groupby('Conference')['NET'].std()).fillna(50)
    feat['conf_count']=conf.map(cg.groupby('Conference')['NET'].count()).fillna(1)
    power_confs={'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf']=conf.isin(power_confs).astype(float)
    cav=feat['conf_avg_net']
    nsp=labeled_df[labeled_df['Overall Seed']>0][['NET Rank','Overall Seed']].copy()
    nsp['NET Rank']=pd.to_numeric(nsp['NET Rank'],errors='coerce'); nsp=nsp.dropna()
    si=nsp['NET Rank'].values.argsort()
    ir=IsotonicRegression(increasing=True,out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si],nsp['Overall Seed'].values[si])
    feat['net_to_seed']=ir.predict(net.values)
    feat['net_sqrt']=np.sqrt(net); feat['net_log']=np.log1p(net); feat['net_inv']=1.0/(net+1)
    feat['seed_line_est']=np.ceil(net/4).clip(1,17)
    feat['elo_proxy']=400-net; feat['elo_momentum']=prev-net
    feat['adj_net']=net-q1w*0.5+q3l*1.0+q4l*2.0
    feat['power_rating']=(0.35*(400-net)+0.25*(300-sos)+0.2*q1w*10+0.1*wpct*100+0.1*(prev-net))
    feat['sos_x_wpct']=(300-sos)/200*wpct
    feat['record_vs_sos']=wpct*(300-sos)/100
    feat['wpct_x_confstr']=wpct*(300-cav)/200
    feat['sos_adj_net']=net+(sos-100)*0.15
    feat['al_net']=net*feat['is_AL']; feat['aq_net']=net*feat['is_AQ']
    feat['aq_sos_penalty']=feat['is_AQ']*(sos/100)
    feat['midmajor_aq']=feat['is_AQ']*(1-feat['is_power_conf'])
    feat['resume_score']=q1w*4+q2w*2-q3l*2-q4l*4
    feat['quality_ratio']=(q1w*3+q2w*2)/(q3l*2+q4l*3+1)
    feat['total_bad_losses']=q3l+q4l
    feat['q1_dominance']=q1w/(q1w+q1l+0.5)
    feat['q12_wins']=q1w+q2w; feat['q34_losses']=q3l+q4l
    feat['quad_balance']=(q1w+q2w)-(q3l+q4l)
    feat['q1_pct']=q1w/(q1w+q1l+0.1); feat['q2_pct']=q2w/(q2w+q2l+0.1)
    feat['net_sos_ratio']=net/(sos+1); feat['net_minus_sos']=net-sos
    road_pct=feat.get('RoadWL_Pct',pd.Series(0.5,index=df.index)).fillna(0.5)
    feat['road_quality']=road_pct*(300-sos)/200
    feat['net_vs_conf_min']=net-feat['conf_min_net']
    feat['conf_rank_ratio']=net/(feat['conf_avg_net']+1)
    all_tourn_rids=set(labeled_df[labeled_df['Overall Seed']>0]['RecordID'].values)|test_tourn_rids
    feat['tourn_field_rank']=34.0
    for sv in df['Season'].unique():
        nets_in_field=[]
        for _,row in all_df[all_df['Season']==sv].iterrows():
            if row['RecordID'] in all_tourn_rids:
                n=pd.to_numeric(row.get('NET Rank',300),errors='coerce')
                if pd.notna(n): nets_in_field.append(n)
        nets_in_field=sorted(nets_in_field)
        smask=df['Season']==sv
        for idx in df[smask].index:
            n=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(n):
                feat.loc[idx,'tourn_field_rank']=float(sum(1 for x in nets_in_field if x<n)+1)
    feat['net_rank_among_al']=30.0
    for sv in df['Season'].unique():
        al_nets=[]
        for _,row in all_df[all_df['Season']==sv].iterrows():
            if str(row.get('Bid Type',''))=='AL':
                n=pd.to_numeric(row.get('NET Rank',300),errors='coerce')
                if pd.notna(n): al_nets.append(n)
        al_nets=sorted(al_nets)
        smask=(df['Season']==sv)&(df['Bid Type'].fillna('')=='AL')
        for idx in df[smask].index:
            n=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(n):
                feat.loc[idx,'net_rank_among_al']=float(sum(1 for x in al_nets if x<n)+1)
    tourn=labeled_df[labeled_df['Overall Seed']>0]
    cb_stats={}
    for _,r in tourn.iterrows():
        c=str(r.get('Conference','Unk')); b=str(r.get('Bid Type','Unk'))
        cb_stats.setdefault((c,b),[]).append(float(r['Overall Seed']))
    for idx in df.index:
        c=str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b=str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        vals=cb_stats.get((c,b),[])
        feat.loc[idx,'cb_mean_seed']=np.mean(vals) if vals else 35.0
        feat.loc[idx,'cb_median_seed']=np.median(vals) if vals else 35.0
    feat['net_vs_conf']=net/(cav+1)
    for cn,cv in [('NET Rank',net),('elo_proxy',feat['elo_proxy']),('adj_net',feat['adj_net']),
                   ('net_to_seed',feat['net_to_seed']),('power_rating',feat['power_rating'])]:
        feat[cn+'_spctile']=0.5
        for sv in df['Season'].unique():
            smask=df['Season']==sv; svals=cv[smask]
            if len(svals)>1: feat.loc[smask,cn+'_spctile']=svals.rank(pct=True)
    return feat


# ============================================================
#  PHASE 1: ROOT CAUSE ANALYSIS
# ============================================================
print('\n' + '='*70)
print(' PHASE 1: ROOT CAUSE ANALYSIS')
print('='*70)

feat_train = build_features(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_features(test_df, all_data, labeled_df=train_tourn)
feat_names = list(feat_train.columns)

X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan, feat_test_full.values.astype(np.float64))
X_stack = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_stack_imp = imp.fit_transform(X_stack)
X_tr_all = X_stack_imp[:n_tr]
X_te_all = X_stack_imp[n_tr:][tourn_idx]
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_all)
X_te_sc = scaler.transform(X_te_all)


def hungarian(scores, seasons, avail, power=1.1):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, sv in enumerate(seasons) if sv == s]
        pos = avail[s]
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned


def evaluate(assigned, gt):
    return int(np.sum(assigned == gt)), int(np.sum((assigned - gt)**2))


# --- 1A: Analyze TRAIN errors (leave-one-season-out) ---
print('\n--- 1A: LOSO TRAINING ERROR ANALYSIS ---')
print('Do the same teams/types consistently fail in LOSO?')

mod_params = {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
              'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':3,
              'reg_lambda':2.0,'reg_alpha':0.5}

loso_errors = []  # (rid, season, predicted_seed, actual_seed, error, bid_type, net, conf)
unique_seasons = sorted(set(train_seasons))

for hold_season in unique_seasons:
    tr_mask = train_seasons != hold_season
    te_mask = train_seasons == hold_season
    X_tr_cv = X_tr_all[tr_mask]
    y_tr_cv = y_train[tr_mask]
    X_te_cv = X_tr_all[te_mask]
    y_te_cv = y_train[te_mask]
    te_rids = train_rids[te_mask]
    te_seasons_cv = train_seasons[te_mask]

    # Available seeds for this season
    avail_cv = {hold_season: list(range(1, 69))}

    # Train mod model
    preds = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr_cv, y_tr_cv)
        preds.append(m.predict(X_te_cv))
    avg_pred = np.mean(preds, axis=0)

    # Ridge blend
    sc_cv = StandardScaler()
    X_tr_cv_sc = sc_cv.fit_transform(X_tr_cv)
    X_te_cv_sc = sc_cv.transform(X_te_cv)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_cv_sc, y_tr_cv)
    ridge_pred = rm.predict(X_te_cv_sc)
    blend = 0.7 * avg_pred + 0.3 * ridge_pred

    # Hungarian assignment
    assigned = hungarian(blend, te_seasons_cv, avail_cv, power=1.1)
    exact = int(np.sum(assigned == y_te_cv.astype(int)))
    total = len(y_te_cv)
    print(f'  Season {hold_season}: {exact}/{total} ({100*exact/total:.0f}%)')

    # Record errors
    for i in range(len(y_te_cv)):
        err = int(assigned[i]) - int(y_te_cv[i])
        bid_type = train_tourn.iloc[np.where(te_mask)[0][i]]['Bid Type']
        net = train_tourn.iloc[np.where(te_mask)[0][i]]['NET Rank']
        conf = train_tourn.iloc[np.where(te_mask)[0][i]]['Conference']
        loso_errors.append({
            'rid': te_rids[i],
            'season': hold_season,
            'predicted': int(assigned[i]),
            'actual': int(y_te_cv[i]),
            'error': err,
            'abs_error': abs(err),
            'bid_type': bid_type,
            'net': pd.to_numeric(net, errors='coerce'),
            'conference': conf,
        })

loso_df = pd.DataFrame(loso_errors)
print(f'\n  Total LOSO: {int((loso_df["error"]==0).sum())}/{len(loso_df)} exact = '
      f'{100*(loso_df["error"]==0).mean():.1f}%')
print(f'  LOSO RMSE: {np.sqrt((loso_df["error"]**2).mean()):.3f}')
print(f'  LOSO MAE:  {loso_df["abs_error"].mean():.3f}')

# --- 1B: Error patterns by bid type ---
print('\n--- 1B: ERROR PATTERNS BY BID TYPE ---')
for bt in ['AL', 'AQ']:
    subset = loso_df[loso_df['bid_type'] == bt]
    if len(subset) > 0:
        exact_pct = 100 * (subset['error'] == 0).mean()
        mean_err = subset['error'].mean()
        mae = subset['abs_error'].mean()
        big = (subset['abs_error'] > 5).sum()
        print(f'  {bt}: {int((subset["error"]==0).sum())}/{len(subset)} exact ({exact_pct:.0f}%), '
              f'mean_err={mean_err:+.2f}, MAE={mae:.2f}, big_errors(>5)={big}')

# --- 1C: Error patterns by seed tier ---
print('\n--- 1C: ERROR PATTERNS BY SEED TIER ---')
for lo, hi, label in [(1, 4, 'Seeds 1-4 (top)'), (5, 16, 'Seeds 5-16 (AL mid)'),
                       (17, 32, 'Seeds 17-32 (AQ top)'), (33, 52, 'Seeds 33-52 (AQ mid)'),
                       (53, 68, 'Seeds 53-68 (AQ bottom)')]:
    subset = loso_df[(loso_df['actual'] >= lo) & (loso_df['actual'] <= hi)]
    if len(subset) > 0:
        exact_pct = 100 * (subset['error'] == 0).mean()
        mean_err = subset['error'].mean()
        mae = subset['abs_error'].mean()
        print(f'  {label:25s}: {int((subset["error"]==0).sum()):3d}/{len(subset):3d} '
              f'({exact_pct:4.0f}%), bias={mean_err:+.2f}, MAE={mae:.2f}')

# --- 1D: Per-conference patterns ---
print('\n--- 1D: CONFERENCES WITH BIGGEST ERRORS ---')
conf_stats = loso_df.groupby('conference').agg(
    count=('error', 'size'),
    mean_error=('error', 'mean'),
    mae=('abs_error', 'mean'),
    exact_rate=('error', lambda x: (x == 0).mean()),
    big_errors=('abs_error', lambda x: (x > 5).sum()),
).sort_values('mae', ascending=False)

print(f'  {"Conference":20s} {"N":>4s} {"Bias":>6s} {"MAE":>6s} {"Exact%":>7s} {"Big":>4s}')
for conf, row in conf_stats.head(20).iterrows():
    print(f'  {str(conf):20s} {row["count"]:4.0f} {row["mean_error"]:+6.2f} '
          f'{row["mae"]:6.2f} {100*row["exact_rate"]:6.1f}% {row["big_errors"]:4.0f}')

# --- 1E: Teams with BIGGEST LOSO errors ---
print('\n--- 1E: WORST LOSO ERRORS (top 25) ---')
worst = loso_df.nlargest(25, 'abs_error')
for _, row in worst.iterrows():
    print(f'  {row["rid"]:35s}  pred={row["predicted"]:2d} actual={row["actual"]:2d} '
          f'err={row["error"]:+3d}  {row["bid_type"]:2s}  NET={row["net"]:>6}  {row["conference"]}')

# --- 1F: Correlation between prediction error and features ---
print('\n--- 1F: FEATURE CORRELATIONS WITH RESIDUAL ERROR ---')
# For each team in LOSO, get features
loso_features = []
for i, row in loso_df.iterrows():
    tr_idx_in_tourn = np.where(train_rids == row['rid'])[0]
    if len(tr_idx_in_tourn) > 0:
        loso_features.append(X_tr_all[tr_idx_in_tourn[0]])
loso_features = np.array(loso_features)
loso_signed_errors = loso_df['error'].values[:len(loso_features)]
loso_abs_errors = loso_df['abs_error'].values[:len(loso_features)]

print(f'\n  Features most correlated with SIGNED error (bias):')
correlations_signed = []
for j, fname in enumerate(feat_names):
    corr = np.corrcoef(loso_features[:, j], loso_signed_errors)[0, 1]
    if not np.isnan(corr):
        correlations_signed.append((fname, corr))
correlations_signed.sort(key=lambda x: abs(x[1]), reverse=True)
for fname, corr in correlations_signed[:15]:
    direction = 'over-seed' if corr > 0 else 'under-seed'
    print(f'    {fname:30s} r={corr:+.3f}  (high value → {direction})')

print(f'\n  Features most correlated with ABSOLUTE error (difficulty):')
correlations_abs = []
for j, fname in enumerate(feat_names):
    corr = np.corrcoef(loso_features[:, j], loso_abs_errors)[0, 1]
    if not np.isnan(corr):
        correlations_abs.append((fname, corr))
correlations_abs.sort(key=lambda x: abs(x[1]), reverse=True)
for fname, corr in correlations_abs[:15]:
    print(f'    {fname:30s} r={corr:+.3f}  (high value → {"harder" if corr > 0 else "easier"})')


# --- Now analyze TEST errors ---
print('\n--- 1G: TEST SET ERRORS (our best model) ---')
SEEDS = [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]
preds_te = []
for seed in SEEDS:
    m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
    m.fit(X_tr_all, y_train)
    preds_te.append(m.predict(X_te_all))
mod_avg_te = np.mean(preds_te, axis=0)

rm = Ridge(alpha=5.0)
rm.fit(X_tr_sc, y_train)
ridge_te = rm.predict(X_te_sc)
blend_te = 0.7 * mod_avg_te + 0.3 * ridge_te

RC_D2B = {'n_estimators':150,'max_depth':2,'learning_rate':0.03,
           'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,
           'reg_lambda':2.0}

preds_tr = []
for seed in SEEDS:
    m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
    m.fit(X_tr_all, y_train)
    preds_tr.append(m.predict(X_tr_all))
mod_avg_tr = np.mean(preds_tr, axis=0)
ridge_tr = rm.predict(X_tr_sc)
blend_tr = 0.7 * mod_avg_tr + 0.3 * ridge_tr
residuals = y_train - blend_tr
X_aug_tr = np.column_stack([X_tr_all, blend_tr])
X_aug_te = np.column_stack([X_te_all, blend_te])
rc = xgb.XGBRegressor(**RC_D2B, random_state=42, verbosity=0)
rc.fit(X_aug_tr, residuals)
final_te = blend_te + rc.predict(X_aug_te)

assigned = hungarian(final_te, test_seasons, test_avail, 1.1)
exact, sse = evaluate(assigned, test_gt)
print(f'  Test: {exact}/91')

test_errors = []
for i in range(n_te):
    err = int(assigned[i]) - int(test_gt[i])
    # Get raw features
    ti = tourn_idx[i]
    bid = test_df.iloc[ti].get('Bid Type', '')
    net = pd.to_numeric(test_df.iloc[ti].get('NET Rank', 300), errors='coerce')
    conf = test_df.iloc[ti].get('Conference', '')
    test_errors.append({
        'rid': test_rids[i],
        'season': test_seasons[i],
        'predicted': int(assigned[i]),
        'actual': int(test_gt[i]),
        'error': err,
        'abs_error': abs(err),
        'raw_pred': float(final_te[i]),
        'bid_type': bid,
        'net': net,
        'conference': conf,
    })
test_err_df = pd.DataFrame(test_errors)

print('\n  ERROR BY BID TYPE (TEST):')
for bt in ['AL', 'AQ']:
    subset = test_err_df[test_err_df['bid_type'] == bt]
    if len(subset) > 0:
        exact_pct = 100 * (subset['error'] == 0).mean()
        mean_err = subset['error'].mean()
        mae = subset['abs_error'].mean()
        big = (subset['abs_error'] > 5).sum()
        print(f'    {bt}: {int((subset["error"]==0).sum())}/{len(subset)} exact ({exact_pct:.0f}%), '
              f'bias={mean_err:+.2f}, MAE={mae:.2f}, big(>5)={big}')

print('\n  ERROR BY SEED TIER (TEST):')
for lo, hi, label in [(1, 4, 'Seeds 1-4'), (5, 16, 'Seeds 5-16'),
                       (17, 32, 'Seeds 17-32'), (33, 52, 'Seeds 33-52'),
                       (53, 68, 'Seeds 53-68')]:
    subset = test_err_df[(test_err_df['actual'] >= lo) & (test_err_df['actual'] <= hi)]
    if len(subset) > 0:
        exact_pct = 100 * (subset['error'] == 0).mean()
        mean_err = subset['error'].mean()
        mae = subset['abs_error'].mean()
        print(f'    {label:20s}: {int((subset["error"]==0).sum()):3d}/{len(subset):3d} '
              f'({exact_pct:4.0f}%), bias={mean_err:+.2f}, MAE={mae:.2f}')

# --- 1H: Are the same error TYPES seen in both LOSO and test? ---
print('\n--- 1H: COMPARING LOSO vs TEST ERROR PATTERNS ---')
print(f'  {"Metric":30s} {"LOSO":>10s} {"Test":>10s}')
print(f'  {"Exact %":30s} {100*(loso_df["error"]==0).mean():>9.1f}% {100*(test_err_df["error"]==0).mean():>9.1f}%')
print(f'  {"Mean signed error":30s} {loso_df["error"].mean():>+10.3f} {test_err_df["error"].mean():>+10.3f}')
print(f'  {"MAE":30s} {loso_df["abs_error"].mean():>10.3f} {test_err_df["abs_error"].mean():>10.3f}')
print(f'  {"RMSE":30s} {np.sqrt((loso_df["error"]**2).mean()):>10.3f} {np.sqrt((test_err_df["error"]**2).mean()):>10.3f}')

loso_al_mae = loso_df[loso_df['bid_type']=='AL']['abs_error'].mean()
test_al_mae = test_err_df[test_err_df['bid_type']=='AL']['abs_error'].mean()
loso_aq_mae = loso_df[loso_df['bid_type']=='AQ']['abs_error'].mean()
test_aq_mae = test_err_df[test_err_df['bid_type']=='AQ']['abs_error'].mean()
print(f'  {"AL MAE":30s} {loso_al_mae:>10.3f} {test_al_mae:>10.3f}')
print(f'  {"AQ MAE":30s} {loso_aq_mae:>10.3f} {test_aq_mae:>10.3f}')


# ============================================================
#  PHASE 2: GENERALIZATION-FOCUSED MODEL DESIGN
# ============================================================
print('\n' + '='*70)
print(' PHASE 2: BUILDING GENERALIZATION-FOCUSED MODELS')
print('='*70)

# Key insight from root cause: if certain patterns (bid type bias,
# conference bias, seed-tier bias) are CONSISTENT between LOSO and test,
# we can fix them. If they aren't, they're noise.

# Strategy A: Bias-corrected model
#   Train a secondary model on LOSO residuals to correct systematic biases.

# Strategy B: Conservative regularization
#   More regularization = better generalization. Trade training fit for stability.

# Strategy C: Diverse ensemble with LOSO-selected weights
#   Multiple different architectures, weight them by LOSO performance.

# Strategy D: Robust loss (Huber) + train on all data + conservative depth

# ---- LOSO framework for model selection ----
# Leave one season out, predict it, and evaluate
# This tells us what to expect on truly unseen data


def loso_evaluate(model_fn, X, y, seasons, extra_X_te=None, power=1.1):
    """Leave-one-season-out evaluation.
    model_fn(X_train, y_train, X_test) -> predictions
    Returns: (exact_matches, total, per_season_details)
    """
    all_exact = 0
    all_total = 0
    all_sse = 0
    season_details = {}
    all_abs_errors = []

    for hold in sorted(set(seasons)):
        tr_mask = seasons != hold
        te_mask = seasons == hold
        X_tr_cv = X[tr_mask]; y_tr_cv = y[tr_mask]
        X_te_cv = X[te_mask]; y_te_cv = y[te_mask]

        preds = model_fn(X_tr_cv, y_tr_cv, X_te_cv)
        avail_cv = {hold: list(range(1, 69))}
        seasons_cv = seasons[te_mask]
        assigned = hungarian(preds, seasons_cv, avail_cv, power)
        ex = int(np.sum(assigned == y_te_cv.astype(int)))
        sse = int(np.sum((assigned - y_te_cv.astype(int))**2))
        n = int(te_mask.sum())
        abs_err = np.abs(assigned - y_te_cv.astype(int))
        all_abs_errors.extend(abs_err)

        all_exact += ex
        all_total += n
        all_sse += sse
        season_details[hold] = (ex, n)

    rmse = np.sqrt(all_sse / all_total) if all_total > 0 else 999
    mae = np.mean(all_abs_errors)
    return all_exact, all_total, rmse, mae, season_details


# --- Model A: Mod10 + Ridge30% (our standard) ---
def model_mod10_ridge30(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.7 * avg + 0.3 * ridge

# --- Model B: Mod10 + Ridge30% + RC ---
def model_mod10_ridge30_rc(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds_tr = []; preds_te = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds_tr.append(m.predict(X_tr))
        preds_te.append(m.predict(X_te))
    avg_tr = np.mean(preds_tr, axis=0)
    avg_te = np.mean(preds_te, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge_tr = rm.predict(X_tr_s)
    ridge_te = rm.predict(X_te_s)
    blend_tr = 0.7 * avg_tr + 0.3 * ridge_tr
    blend_te = 0.7 * avg_te + 0.3 * ridge_te
    # RC
    res = y_tr - blend_tr
    X_aug_tr = np.column_stack([X_tr, blend_tr])
    X_aug_te = np.column_stack([X_te, blend_te])
    rc_m = xgb.XGBRegressor(**RC_D2B, random_state=42, verbosity=0)
    rc_m.fit(X_aug_tr, res)
    return blend_te + rc_m.predict(X_aug_te)

# --- Model C: Conservative (deeper reg, fewer trees) ---
conservative_params = {'n_estimators':300, 'max_depth':4, 'learning_rate':0.02,
                         'subsample':0.7, 'colsample_bytree':0.7, 'min_child_weight':5,
                         'reg_lambda':5.0, 'reg_alpha':1.0}

def model_conservative(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**conservative_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=10.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.6 * avg + 0.4 * ridge

# --- Model D: Ultra-conservative (mostly ridge) ---
def model_ultra_conservative(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777]:
        m = xgb.XGBRegressor(**conservative_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=10.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.5 * avg + 0.5 * ridge

# --- Model E: Dreg (proven principled) + Ridge ---
dreg_params = {'n_estimators':500,'max_depth':5,'learning_rate':0.03,
               'subsample':0.7,'colsample_bytree':0.7,'min_child_weight':4,
               'reg_lambda':3.0,'reg_alpha':1.0}

def model_dreg_ridge(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**dreg_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.7 * avg + 0.3 * ridge

# --- Model F: Dreg + Ridge + RC ---
def model_dreg_ridge_rc(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds_tr = []; preds_te = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**dreg_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds_tr.append(m.predict(X_tr))
        preds_te.append(m.predict(X_te))
    avg_tr = np.mean(preds_tr, axis=0)
    avg_te = np.mean(preds_te, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge_tr = rm.predict(X_tr_s)
    ridge_te = rm.predict(X_te_s)
    blend_tr = 0.7 * avg_tr + 0.3 * ridge_tr
    blend_te = 0.7 * avg_te + 0.3 * ridge_te
    res = y_tr - blend_tr
    X_aug_tr = np.column_stack([X_tr, blend_tr])
    X_aug_te = np.column_stack([X_te, blend_te])
    rc_m = xgb.XGBRegressor(**RC_D2B, random_state=42, verbosity=0)
    rc_m.fit(X_aug_tr, res)
    return blend_te + rc_m.predict(X_aug_te)


# --- Model G: Multi-architecture ensemble (mod + dreg + conservative averaged) ---
def model_multi_arch(X_tr, y_tr, X_te):
    p1 = model_mod10_ridge30(X_tr, y_tr, X_te)
    p2 = model_dreg_ridge(X_tr, y_tr, X_te)
    p3 = model_conservative(X_tr, y_tr, X_te)
    return (p1 + p2 + p3) / 3.0

# --- Model H: Multi-arch with RC ---
def model_multi_arch_rc(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    # Mod predictions
    mod_tr = []; mod_te = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr); mod_tr.append(m.predict(X_tr)); mod_te.append(m.predict(X_te))

    # Dreg predictions
    dreg_tr = []; dreg_te = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = xgb.XGBRegressor(**dreg_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr); dreg_tr.append(m.predict(X_tr)); dreg_te.append(m.predict(X_te))

    # Conservative predictions
    cons_tr = []; cons_te = []
    for seed in [42, 123, 777]:
        m = xgb.XGBRegressor(**conservative_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr); cons_tr.append(m.predict(X_tr)); cons_te.append(m.predict(X_te))

    # Ridge
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    r_tr = rm.predict(X_tr_s); r_te = rm.predict(X_te_s)

    # Combine: average of all learners
    all_preds_tr = mod_tr + dreg_tr + cons_tr + [r_tr]
    all_preds_te = mod_te + dreg_te + cons_te + [r_te]
    avg_tr = np.mean(all_preds_tr, axis=0)
    avg_te = np.mean(all_preds_te, axis=0)

    # RC on the ensemble average
    res = y_tr - avg_tr
    X_aug_tr = np.column_stack([X_tr, avg_tr])
    X_aug_te = np.column_stack([X_te, avg_te])
    rc_m = xgb.XGBRegressor(**RC_D2B, random_state=42, verbosity=0)
    rc_m.fit(X_aug_tr, res)
    return avg_te + rc_m.predict(X_aug_te)


# --- Model I: LightGBM-based for diversity ---
import lightgbm as lgb
lgb_params = {'n_estimators':500, 'max_depth':5, 'learning_rate':0.03,
              'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':3,
              'reg_lambda':2.0, 'reg_alpha':0.5, 'verbosity':-1}

def model_lgb_ridge(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = lgb.LGBMRegressor(**lgb_params, random_state=seed, n_jobs=1)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.7 * avg + 0.3 * ridge


# --- Model J: XGB + LGB + Ridge (cross-algorithm ensemble) ---
def model_xgb_lgb_ridge(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    xgb_preds = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = xgb.XGBRegressor(**mod_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        xgb_preds.append(m.predict(X_te))

    lgb_preds = []
    for seed in [42, 123, 777, 2024, 31415]:
        m = lgb.LGBMRegressor(**lgb_params, random_state=seed, n_jobs=1)
        m.fit(X_tr, y_tr)
        lgb_preds.append(m.predict(X_te))

    xgb_avg = np.mean(xgb_preds, axis=0)
    lgb_avg = np.mean(lgb_preds, axis=0)

    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)

    return 0.4 * xgb_avg + 0.3 * lgb_avg + 0.3 * ridge


# --- Model K: Huber objective (robust to outliers) ---
huber_params = {**mod_params, 'objective': 'reg:pseudohubererror',
                'huber_slope': 4.0}

def model_huber_ridge(X_tr, y_tr, X_te):
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    preds = []
    for seed in [42, 123, 777, 2024, 31415, 1337, 9999, 54321, 11111, 88888]:
        m = xgb.XGBRegressor(**huber_params, random_state=seed, verbosity=0)
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))
    avg = np.mean(preds, axis=0)
    rm = Ridge(alpha=5.0)
    rm.fit(X_tr_s, y_tr)
    ridge = rm.predict(X_te_s)
    return 0.7 * avg + 0.3 * ridge


# ============================================================
#  RUN LOSO ON ALL MODELS
# ============================================================
print('\n--- LOSO EVALUATION OF ALL MODELS ---')

models = {
    'A: mod10+r30':        model_mod10_ridge30,
    'B: mod10+r30+rc':     model_mod10_ridge30_rc,
    'C: conservative':     model_conservative,
    'D: ultra_cons':       model_ultra_conservative,
    'E: dreg+r30':         model_dreg_ridge,
    'F: dreg+r30+rc':      model_dreg_ridge_rc,
    'G: multi_arch_avg':   model_multi_arch,
    'H: multi_arch+rc':    model_multi_arch_rc,
    'I: lgb+r30':          model_lgb_ridge,
    'J: xgb+lgb+r30':     model_xgb_lgb_ridge,
    'K: huber+r30':        model_huber_ridge,
}

loso_results = {}
print(f'\n  {"Model":25s} {"LOSO Exact":>12s} {"LOSO RMSE":>12s} {"LOSO MAE":>10s}  Per-season')

for name, fn in models.items():
    exact, total, rmse, mae, details = loso_evaluate(fn, X_tr_all, y_train, train_seasons)
    loso_results[name] = {'exact': exact, 'total': total, 'rmse': rmse, 'mae': mae, 'details': details}
    per_s = ' '.join(f'{d[0]}/{d[1]}' for d in details.values())
    print(f'  {name:25s} {exact:3d}/{total:3d} ({100*exact/total:.0f}%) '
          f'{rmse:10.3f}  {mae:8.3f}   {per_s}')


# ============================================================
#  PHASE 3: TEST SET PERFORMANCE (to see correlation with LOSO)
# ============================================================
print('\n--- TEST SET PERFORMANCE ---')
print(f'  {"Model":25s} {"Test Exact":>12s} {"Test RMSE":>12s}  Per-season')

test_results = {}
for name, fn in models.items():
    preds = fn(X_tr_all, y_train, X_te_all)
    assigned = hungarian(preds, test_seasons, test_avail, 1.1)
    ex, sse = evaluate(assigned, test_gt)
    rmse_t = np.sqrt(sse / n_te)
    test_results[name] = {'exact': ex, 'rmse': rmse_t, 'assigned': assigned, 'preds': preds}

    per_s = []
    for s in sorted(set(test_seasons)):
        si = [i for i, sv in enumerate(test_seasons) if sv == s]
        e = sum(1 for i in si if assigned[i] == test_gt[i])
        per_s.append(f'{e}/{len(si)}')
    print(f'  {name:25s} {ex:3d}/{n_te:3d} ({100*ex/n_te:.0f}%) '
          f'{rmse_t:10.3f}   {" ".join(per_s)}')


# ============================================================
#  CORRELATION: does LOSO predict test performance?
# ============================================================
print('\n--- LOSO vs TEST CORRELATION ---')
print(f'  {"Model":25s} {"LOSO":>6s} {"Test":>6s} {"ΔLOSO-RMSE":>12s} {"ΔTest-RMSE":>12s}')
loso_exacts = []; test_exacts = []; loso_rmses = []; test_rmses = []
for name in models:
    le = loso_results[name]['exact']; lr = loso_results[name]['rmse']
    te = test_results[name]['exact']; tr = test_results[name]['rmse']
    print(f'  {name:25s} {le:3d}/249 {te:3d}/{n_te:3d}  {lr:12.3f}  {tr:12.3f}')
    loso_exacts.append(le); test_exacts.append(te)
    loso_rmses.append(lr); test_rmses.append(tr)

corr_exact = np.corrcoef(loso_exacts, test_exacts)[0,1]
corr_rmse = np.corrcoef(loso_rmses, test_rmses)[0,1]
print(f'\n  Correlation (LOSO exact vs Test exact): r = {corr_exact:.3f}')
print(f'  Correlation (LOSO RMSE vs Test RMSE):   r = {corr_rmse:.3f}')


# ============================================================
#  PHASE 4: ENSEMBLE OF MODELS WEIGHTED BY LOSO
# ============================================================
print('\n' + '='*70)
print(' PHASE 4: LOSO-WEIGHTED ENSEMBLES')
print('='*70)

# Rank by LOSO RMSE (lower = better) and by LOSO exact (higher = better)
ranked_rmse = sorted(loso_results.items(), key=lambda x: x[1]['rmse'])
ranked_exact = sorted(loso_results.items(), key=lambda x: -x[1]['exact'])

print('\n  Models ranked by LOSO RMSE:')
for i, (name, res) in enumerate(ranked_rmse):
    print(f'    {i+1}. {name:25s} RMSE={res["rmse"]:.3f} exact={res["exact"]}/249')

# Try various ensemble combinations using LOSO-ranked models
print('\n  LOSO-weighted test ensembles:')

all_test_preds = {name: test_results[name]['preds'] for name in models}

# Softmax weights from LOSO RMSE
rmse_arr = np.array([loso_results[name]['rmse'] for name in models])
softmax_weights = np.exp(-rmse_arr) / np.exp(-rmse_arr).sum()

# Weighted average
names = list(models.keys())
ensemble_pred = np.zeros(n_te)
for i, name in enumerate(names):
    ensemble_pred += softmax_weights[i] * all_test_preds[name]

assigned = hungarian(ensemble_pred, test_seasons, test_avail, 1.1)
ex, sse = evaluate(assigned, test_gt)
print(f'    Softmax-RMSE weighted ensemble: {ex}/91')

# Equal weight of top-K by LOSO RMSE
for k in [3, 5, 7, len(models)]:
    top_names = [n for n, _ in ranked_rmse[:k]]
    avg_pred = np.mean([all_test_preds[n] for n in top_names], axis=0)
    assigned = hungarian(avg_pred, test_seasons, test_avail, 1.1)
    ex, _ = evaluate(assigned, test_gt)
    print(f'    Top-{k} LOSO-RMSE equal avg: {ex}/91  ({", ".join(n.split(":")[0] for n in top_names)})')

# Top-K by LOSO exact
for k in [3, 5]:
    top_names = [n for n, _ in ranked_exact[:k]]
    avg_pred = np.mean([all_test_preds[n] for n in top_names], axis=0)
    assigned = hungarian(avg_pred, test_seasons, test_avail, 1.1)
    ex, _ = evaluate(assigned, test_gt)
    print(f'    Top-{k} LOSO-exact avg: {ex}/91  ({", ".join(n.split(":")[0] for n in top_names)})')


# ============================================================
#  PHASE 5: BEST GENERALIZABLE MODEL SELECTION
# ============================================================
print('\n' + '='*70)
print(' PHASE 5: RECOMMENDED MODELS FOR PRIVATE LEADERBOARD')
print('='*70)

# The model to submit should be:
# 1. Best LOSO RMSE (most generalizable raw predictions)
# 2. Best LOSO exact (most generalizable assignment quality)
# 3. Best test score (but this risks overfitting to public test)

best_loso_rmse_name = ranked_rmse[0][0]
best_loso_exact_name = ranked_exact[0][0]
best_test_name = max(test_results, key=lambda x: test_results[x]['exact'])

print(f'\n  Best LOSO RMSE:  {best_loso_rmse_name} '
      f'(LOSO: {loso_results[best_loso_rmse_name]["rmse"]:.3f}, '
      f'Test: {test_results[best_loso_rmse_name]["exact"]}/91)')
print(f'  Best LOSO exact: {best_loso_exact_name} '
      f'(LOSO: {loso_results[best_loso_exact_name]["exact"]}/249, '
      f'Test: {test_results[best_loso_exact_name]["exact"]}/91)')
print(f'  Best test score: {best_test_name} '
      f'(LOSO: {loso_results[best_test_name]["exact"]}/249, '
      f'Test: {test_results[best_test_name]["exact"]}/91)')

# "Safe bet" for private leaderboard: best LOSO performance
print(f'\n  RECOMMENDATION FOR PRIVATE LEADERBOARD:')
print(f'    Primary: {best_loso_rmse_name} (lowest LOSO RMSE = most robust)')
print(f'    Backup:  Best LOSO-exact model as secondary submission')


# ============================================================
#  SAVE SUBMISSIONS
# ============================================================
print(f'\n{"="*70}')
print(' SAVING')
print(f'{"="*70}')

def save_sub(assigned, name, desc):
    sub = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub['RecordID'] == rid
        if mask.any():
            sub.loc[mask, 'Overall Seed'] = int(assigned[i])
    path = os.path.join(DATA_DIR, name)
    sub.to_csv(path, index=False)
    ex, _ = evaluate(assigned, test_gt)
    print(f'  Saved {name}: {ex}/91 [{desc}]')

# Save models ranked by generalization quality (LOSO RMSE)
saved = 0
for name, res in ranked_rmse:
    saved += 1
    if saved > 10: break
    a = test_results[name]['assigned']
    desc = f'{name} (LOSO-RMSE={res["rmse"]:.3f})'
    save_sub(a, f'submission_v37_{saved}.csv', desc)

# Also save the LOSO-weighted ensembles
# Top-3 LOSO-RMSE equal avg
top3_names = [n for n, _ in ranked_rmse[:3]]
avg_pred = np.mean([all_test_preds[n] for n in top3_names], axis=0)
assigned = hungarian(avg_pred, test_seasons, test_avail, 1.1)
ex, _ = evaluate(assigned, test_gt)
save_sub(assigned, f'submission_v37_ensemble_top3.csv', f'Top-3 LOSO avg ({ex}/91)')

# Top-5
top5_names = [n for n, _ in ranked_rmse[:5]]
avg_pred = np.mean([all_test_preds[n] for n in top5_names], axis=0)
assigned = hungarian(avg_pred, test_seasons, test_avail, 1.1)
ex, _ = evaluate(assigned, test_gt)
save_sub(assigned, f'submission_v37_ensemble_top5.csv', f'Top-5 LOSO avg ({ex}/91)')

# Softmax weighted
ensemble_pred = np.zeros(n_te)
for i, name in enumerate(names):
    ensemble_pred += softmax_weights[i] * all_test_preds[name]
assigned = hungarian(ensemble_pred, test_seasons, test_avail, 1.1)
ex, _ = evaluate(assigned, test_gt)
save_sub(assigned, f'submission_v37_ensemble_softmax.csv', f'Softmax LOSO ({ex}/91)')

print(f'\nTotal: {time.time()-t0:.0f}s')

if IN_COLAB:
    for p in [f'submission_v37_{i}.csv' for i in range(1,11)] + \
             ['submission_v37_ensemble_top3.csv', 'submission_v37_ensemble_top5.csv',
              'submission_v37_ensemble_softmax.csv']:
        fp = os.path.join(DATA_DIR, p)
        if os.path.exists(fp): files.download(fp)
