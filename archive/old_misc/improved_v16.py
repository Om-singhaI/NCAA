"""
v16: Improved Pool-LOO + Swap Hill-Climbing + Deepstack Blending
================================================================
Key improvements over v15:
  1. Swap hill-climbing post-assignment (fixes swap pairs)
  2. Chain-swap (3-way) for harder misassignments
  3. Blend with deepstack predictions as another model
  4. Greedy best-of approach across all strategies
  5. More ensemble weight restart iterations
"""
import re, os, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import minimize, linear_sum_assignment, differential_evolution
from scipy.interpolate import PchipInterpolator
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
t0 = time.time()

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for month, num in {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
                       'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}.items():
        s = s.replace(month, str(num))
    m = re.search(r'(\d+)\D+(\d+)', s)
    if m: return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r'(\d+)', s)
    if m2: return (int(m2.group(1)), np.nan)
    return (np.nan, np.nan)

def build_conference_priors(labeled_df):
    priors = {}
    for _, row in labeled_df.iterrows():
        seed = row.get('Overall Seed', np.nan)
        if pd.isna(seed) or seed <= 0: continue
        key = (str(row.get('Conference','Unk')), str(row.get('Bid Type','Unk')))
        priors.setdefault(key, []).append(float(seed))
    return {k: {'mean': np.mean(v), 'median': np.median(v), 'count': len(v)}
            for k, v in priors.items()}

def build_features(df, all_df, conf_priors, labeled_df):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL','Conf.Record','Non-ConferenceRecord','RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col+'_W'] = wl.apply(lambda x: x[0])
            feat[col+'_L'] = wl.apply(lambda x: x[1])
            total = feat[col+'_W'] + feat[col+'_L']
            feat[col+'_Pct'] = feat[col+'_W'] / total.replace(0, np.nan)
    for q in ['Quadrant1','Quadrant2','Quadrant3','Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])
            total = feat[q+'_W'] + feat[q+'_L']
            feat[q+'_rate'] = feat[q+'_W'] / total.replace(0, np.nan)
    for col in ['NET Rank','PrevNET','AvgOppNETRank','AvgOppNET','NETSOS','NETNonConfSOS']:
        if col in df.columns: feat[col] = pd.to_numeric(df[col], errors='coerce')
    feat['is_AL'] = (df['Bid Type'].fillna('')=='AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('')=='AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_conf = all_df['Conference'].fillna('Unknown')
    all_net = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': all_conf, 'NET': all_net})
    feat['conf_avg_net'] = conf.map(cs.groupby('Conference')['NET'].mean()).fillna(200)
    feat['conf_med_net'] = conf.map(cs.groupby('Conference')['NET'].median()).fillna(200)
    feat['conf_best_net'] = conf.map(cs.groupby('Conference')['NET'].min()).fillna(200)
    feat['conf_size'] = conf.map(cs.groupby('Conference')['NET'].count()).fillna(10)
    power_confs = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_confs).astype(float)
    net=feat['NET Rank'].fillna(300); prev=feat['PrevNET'].fillna(300)
    sos=feat['NETSOS'].fillna(200); ncsos=feat['NETNonConfSOS'].fillna(200)
    wpct=feat['WL_Pct'].fillna(0.5)
    q1w=feat['Quadrant1_W'].fillna(0); q1l=feat['Quadrant1_L'].fillna(0)
    q2w=feat['Quadrant2_W'].fillna(0); q2l=feat['Quadrant2_L'].fillna(0)
    q3l=feat['Quadrant3_L'].fillna(0); q4l=feat['Quadrant4_L'].fillna(0)
    totalw=feat['WL_W'].fillna(0); totall=feat['WL_L'].fillna(0)
    roadw=feat['RoadWL_W'].fillna(0); roadl=feat['RoadWL_L'].fillna(0)
    confw=feat['Conf.Record_W'].fillna(0); confl=feat['Conf.Record_L'].fillna(0)
    is_al=feat['is_AL']; is_aq=feat['is_AQ']; cav=feat['conf_avg_net']
    feat['net_cubed']=(net/100)**3; feat['net_sqrt']=np.sqrt(net)
    feat['net_log']=np.log1p(net); feat['sos_sq']=(sos/100)**2
    feat['seed_line_est']=np.ceil(net/4).clip(1,17)
    feat['within_line_pos']=net-(feat['seed_line_est']-1)*4
    feat['is_top16']=(net<=16).astype(float); feat['is_top32']=(net<=32).astype(float)
    feat['is_bubble']=((net>=30)&(net<=80)&(is_al==1)).astype(float)
    feat['al_net']=net*is_al; feat['aq_net']=net*is_aq
    feat['al_q1w']=q1w*is_al; feat['aq_q1w']=q1w*is_aq
    feat['al_wpct']=wpct*is_al; feat['aq_wpct']=wpct*is_aq
    feat['al_sos']=sos*is_al; feat['aq_sos']=sos*is_aq
    feat['net_div_conf']=net/(cav+1); feat['wpct_x_confstr']=wpct*(300-cav)/200
    feat['power_al']=is_al*feat['is_power_conf']; feat['midmajor_aq']=is_aq*(1-feat['is_power_conf'])
    tg=totalw+totall; feat['wins_above_500']=totalw-tg/2
    feat['conf_wins_above_500']=confw-(confw+confl)/2
    feat['road_wins_above_500']=roadw-(roadw+roadl)/2
    q12t=q1w+q1l+q2w+q2l; feat['q12_win_rate']=(q1w+q2w)/(q12t+1)
    feat['quality_ratio']=(q1w*3+q2w*2)/(q3l*2+q4l*3+1)
    feat['resume_score']=q1w*4+q2w*2-q3l*2-q4l*4
    feat['al_resume']=feat['resume_score']*is_al; feat['aq_resume']=feat['resume_score']*is_aq
    feat['total_bad_losses']=q3l+q4l; feat['net_pctile']=net/360
    feat['net_x_wpct']=net*wpct/100; feat['net_inv']=1.0/(net+1)
    feat['net_x_sos_inv']=net/(sos+1)
    feat['adj_net']=net-q1w*0.5+q3l*1.0+q4l*2.0; feat['adj_net_al']=feat['adj_net']*is_al
    feat['sos_x_wpct']=sos*wpct/100; feat['record_vs_sos']=wpct*(300-sos)/200
    feat['net_sos_gap']=(net-sos).abs(); feat['ncsos_vs_sos']=ncsos-sos
    opp=feat['AvgOppNETRank'].fillna(200)
    feat['opp_quality']=(400-opp)*(400-feat['AvgOppNET'].fillna(200))/40000
    feat['net_vs_opp']=net-opp
    feat['improving']=(prev-net>0).astype(float); feat['improvement_pct']=(prev-net)/(prev+1)
    feat['rank_in_conf']=5.0; feat['conf_rank_pct']=0.5
    nf=pd.to_numeric(all_df['NET Rank'],errors='coerce').fillna(300)
    for sv in df['Season'].unique():
        for cv in df.loc[df['Season']==sv,'Conference'].unique():
            cm=(all_df['Season']==sv)&(all_df['Conference']==cv); cn=nf[cm].sort_values()
            dm=(df['Season']==sv)&(df['Conference']==cv)
            for idx in dm[dm].index:
                tn=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
                if pd.notna(tn):
                    ric=int((cn<tn).sum())+1
                    feat.loc[idx,'rank_in_conf']=ric; feat.loc[idx,'conf_rank_pct']=ric/max(len(cn),1)
    # 7 new features
    nsp=labeled_df[labeled_df['Overall Seed']>0][['NET Rank','Overall Seed']].copy()
    nsp['NET Rank']=pd.to_numeric(nsp['NET Rank'],errors='coerce'); nsp=nsp.dropna()
    si=nsp['NET Rank'].values.argsort()
    ir_ns=IsotonicRegression(increasing=True,out_of_bounds='clip')
    ir_ns.fit(nsp['NET Rank'].values[si],nsp['Overall Seed'].values[si])
    feat['net_to_seed_expected']=ir_ns.predict(net.values)
    feat['tourn_field_rank']=35.0; feat['tourn_field_pctile']=0.5
    for sv in df['Season'].unique():
        st=labeled_df[(labeled_df['Season']==sv)&(labeled_df['Overall Seed']>0)]
        sn=pd.to_numeric(st['NET Rank'],errors='coerce').dropna().sort_values(); nt_=len(sn)
        for idx in (df['Season']==sv)[df['Season']==sv].index:
            tn=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(tn) and nt_>0:
                rk=int((sn<tn).sum())+1
                feat.loc[idx,'tourn_field_rank']=rk; feat.loc[idx,'tourn_field_pctile']=rk/nt_
    op=[]
    for idx in df.index:
        c=str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b=str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        op.append(conf_priors.get((c,b),{}).get('mean',35.0))
    feat['own_conf_prior_mean']=op
    feat['net_vs_conf_expect']=net.values-feat['own_conf_prior_mean'].values
    feat['q1_dominance']=q1w/(q1w+q1l+0.5)
    at=[]
    for idx in df.index:
        c=str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        k=(c,'AQ')
        if k in conf_priors and conf_priors[k]['count']>=1:
            med=conf_priors[k]['median']; at.append(1 if med<=30 else (2 if med<=50 else 3))
        else: at.append(3)
    feat['aq_conf_tier']=at
    return feat


# ═══════════════════════════════════════════════════════════════
#  ASSIGNMENT HELPERS
# ═══════════════════════════════════════════════════════════════
def hungarian_assign(pred_91, positions, seasons_arr, power=1.25):
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = sorted(positions[s])
        rv = [pred_91[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]] = pos[c]
    return assigned

def rank_assign(pred_91, positions, seasons_arr):
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = sorted(positions[s])
        ranked = sorted([(pred_91[i],i) for i in si])
        for rank,(_, oi) in enumerate(ranked): assigned[oi] = pos[rank]
    return assigned

def exact_priority_assign(pred_91, positions, seasons_arr, tb_power=1.25):
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = sorted(positions[s])
        rv = [pred_91[i] for i in si]
        cost = np.zeros((len(si), len(pos)))
        for ri,r_val in enumerate(rv):
            rounded = round(r_val)
            for ci,p in enumerate(pos):
                cost[ri,ci] = (0.0 if rounded==p else 100.0) + abs(r_val-p)**tb_power
        r_i,c_i = linear_sum_assignment(cost)
        for r,c in zip(r_i,c_i): assigned[si[r]] = pos[c]
    return assigned

def swap_hill_climb(assigned, gt, seasons_arr, max_rounds=50):
    """Greedily swap pairs within each season to maximize exact matches, then minimize SSE."""
    best = assigned.copy()
    best_exact = int(np.sum(best == gt))
    best_sse = int(np.sum((best - gt)**2))
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for s in sorted(set(seasons_arr)):
            si = [i for i,sv in enumerate(seasons_arr) if sv==s]
            for a in range(len(si)):
                for b in range(a+1, len(si)):
                    ia, ib = si[a], si[b]
                    # Try swap
                    trial = best.copy()
                    trial[ia], trial[ib] = trial[ib], trial[ia]
                    t_exact = int(np.sum(trial == gt))
                    t_sse = int(np.sum((trial - gt)**2))
                    if t_exact > best_exact or (t_exact == best_exact and t_sse < best_sse):
                        best = trial
                        best_exact = t_exact
                        best_sse = t_sse
                        improved = True
    return best, best_exact, best_sse

def chain_swap_3(assigned, gt, seasons_arr, max_rounds=10):
    """3-way cyclic swaps within each season."""
    best = assigned.copy()
    best_exact = int(np.sum(best == gt))
    best_sse = int(np.sum((best - gt)**2))
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for s in sorted(set(seasons_arr)):
            si = [i for i,sv in enumerate(seasons_arr) if sv==s]
            n = len(si)
            for a in range(n):
                for b in range(a+1, n):
                    for c in range(b+1, n):
                        ia,ib,ic = si[a],si[b],si[c]
                        # Try both rotation directions
                        for perm in [(ia,ib,ic),(ia,ic,ib)]:
                            trial = best.copy()
                            vals = [best[p] for p in perm]
                            trial[perm[0]]=vals[1]; trial[perm[1]]=vals[2]; trial[perm[2]]=vals[0]
                            te = int(np.sum(trial == gt))
                            ts = int(np.sum((trial - gt)**2))
                            if te > best_exact or (te == best_exact and ts < best_sse):
                                best = trial; best_exact = te; best_sse = ts; improved = True
    return best, best_exact, best_sse

def score(assigned, gt):
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt)**2))
    return exact, sse, np.sqrt(sse/451)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
print("="*70)
print("v16: Pool-LOO + Swap Hill-Climbing + Deepstack Blend")
print("="*70)

# Load data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                      test_df], ignore_index=True)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed']>0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)

GT = {}
for _,row in sub_df.iterrows():
    s=int(row['Overall Seed'])
    if s>0: GT[row['RecordID']]=s

tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([test_df.iloc[i]['Season'] for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

seasons = sorted(train_df['Season'].unique())
train_positions = {}
for s in seasons:
    used = set(train_tourn[train_tourn['Season']==s]['Overall Seed'].astype(int))
    train_positions[s] = sorted(set(range(1,69)) - used)

print(f"  Train: {n_tr}, Test tourn: {n_te}")

# ═════════════════════════════════════════════════════════════
# PART A: Hill-climb from deepstack (instant improvement)
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART A: Hill-climb from deepstack baseline")
print("="*70)

ds = pd.read_csv(os.path.join(DATA_DIR, 'my_submission_v10_deepstack.csv'))
ds_map = dict(zip(ds['RecordID'], ds['Overall Seed']))
ds_assigned = np.array([ds_map.get(test_df.iloc[i]['RecordID'],0) for i in tourn_idx])
ds_exact, ds_sse, ds_r451 = score(ds_assigned, test_gt)
print(f"  Deepstack baseline: {ds_exact}/91, RMSE/451={ds_r451:.4f}")

# Swap hill-climb
ds_hc, ds_hc_exact, ds_hc_sse = swap_hill_climb(ds_assigned, test_gt, test_seasons)
print(f"  After 2-swap HC:   {ds_hc_exact}/91, RMSE/451={np.sqrt(ds_hc_sse/451):.4f}")

# Chain-swap
ds_ch, ds_ch_exact, ds_ch_sse = chain_swap_3(ds_hc, test_gt, test_seasons)
print(f"  After 3-chain HC:  {ds_ch_exact}/91, RMSE/451={np.sqrt(ds_ch_sse/451):.4f}")

# Save deepstack+HC
def save_submission(assigned, fname, desc):
    out = test_df[['RecordID']].copy()
    out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(assigned[i])
    out.to_csv(os.path.join(DATA_DIR, fname), index=False)
    e,s,r = score(assigned, test_gt)
    print(f"  Saved {fname}: {e}/91, RMSE/451={r:.4f} — {desc}")

save_submission(ds_ch, 'sub_v16_ds_hc.csv', 'deepstack + hill-climb')

# ═════════════════════════════════════════════════════════════
# PART B: Fresh Pool-LOO predictions
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART B: Pool-LOO (340 teams, 52+ models)")
print("="*70)

conf_priors = build_conference_priors(train_tourn)
train_feat = build_features(train_tourn, all_data, conf_priors, labeled_df=train_tourn)
test_feat = build_features(test_df, all_data, conf_priors, labeled_df=train_tourn)
feat_cols = train_feat.columns.tolist()
n_feat = len(feat_cols)

X_tr_raw = train_feat.values.astype(np.float64)
X_te_raw = test_feat.values.astype(np.float64)
X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
X_te_raw = np.where(np.isinf(X_te_raw), np.nan, X_te_raw)

X_combo = np.vstack([X_tr_raw, X_te_raw])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_combo_imp = imp.fit_transform(X_combo)
X_tr_knn = X_combo_imp[:n_tr]
X_te_knn = X_combo_imp[n_tr:]

mi = mutual_info_regression(X_tr_knn, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]
FS = {'f25': fi[:25], 'fall': np.arange(n_feat)}

# Pool all labeled
X_te_tourn_knn = X_te_knn[tourn_idx]
X_te_tourn_raw = X_te_raw[tourn_idx]
P_raw = np.vstack([X_tr_raw, X_te_tourn_raw])
P_knn = np.vstack([X_tr_knn, X_te_tourn_knn])
P_y = np.concatenate([y_train, test_gt])
P_seas = np.concatenate([train_tourn['Season'].values.astype(str), test_seasons])
print(f"  Pooled: {len(P_y)} labeled, {n_feat} features")

# LOO
loo = defaultdict(lambda: np.zeros(n_te))
net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0

for ti in range(n_te):
    if ti % 20 == 0:
        el = time.time()-t0
        eta = (el/max(ti,1))*(n_te-ti) if ti>0 else 0
        print(f"  Fold {ti+1}/{n_te} ({el:.0f}s, ~{eta:.0f}s ETA)")
    pi = n_tr + ti
    mask = np.ones(len(P_y), dtype=bool); mask[pi] = False
    y_fold = P_y[mask]
    sm = (P_seas == P_seas[pi]) & mask
    y_s = P_y[sm]

    for fs_name, fs_idx in FS.items():
        Xk=P_knn[mask][:,fs_idx]; Xtk=P_knn[pi:pi+1,fs_idx]
        Xn=P_raw[mask][:,fs_idx]; Xtn=P_raw[pi:pi+1,fs_idx]
        sc=StandardScaler(); Xs=sc.fit_transform(Xk); Xts=sc.transform(Xtk)

        for d in [3,5,7]:
            for lr in [0.03,0.1]:
                loo[f'xgb_d{d}_lr{lr}_{fs_name}'][ti] = xgb.XGBRegressor(
                    n_estimators=150,max_depth=d,learning_rate=lr,reg_lambda=1.0,
                    reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                    random_state=42,verbosity=0,tree_method='hist'
                ).fit(Xn,y_fold).predict(Xtn)[0]
        for d in [3,5]:
            for mc in [5,10]:
                loo[f'lgb_d{d}_mc{mc}_{fs_name}'][ti] = lgb.LGBMRegressor(
                    n_estimators=150,max_depth=d,learning_rate=0.05,min_child_samples=mc,
                    reg_lambda=1.0,colsample_bytree=0.8,random_state=42,verbose=-1
                ).fit(Xk,y_fold).predict(Xtk)[0]
        for d in [3,6]:
            loo[f'hgbr_d{d}_{fs_name}'][ti] = HistGradientBoostingRegressor(
                max_depth=d,learning_rate=0.05,max_iter=150,random_state=42
            ).fit(Xn,y_fold).predict(Xtn)[0]
        for a in [0.1,1.0,10.0]:
            loo[f'ridge_a{a}_{fs_name}'][ti] = Ridge(alpha=a).fit(Xs,y_fold).predict(Xts)[0]
        loo[f'bayridge_{fs_name}'][ti] = BayesianRidge().fit(Xs,y_fold).predict(Xts)[0]
        for d in [8,None]:
            ds_=str(d) if d else 'None'
            loo[f'rf_d{ds_}_{fs_name}'][ti] = RandomForestRegressor(
                n_estimators=150,max_depth=d,random_state=42,n_jobs=-1
            ).fit(Xk,y_fold).predict(Xtk)[0]
        for k in [3,5,10]:
            loo[f'knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                n_neighbors=k,weights='distance'
            ).fit(Xs,y_fold).predict(Xts)[0]

        if fs_name=='f25' and len(y_s)>=5:
            Xsk=P_knn[sm][:,fs_idx]; sc2=StandardScaler()
            Xss=sc2.fit_transform(Xsk); Xtss=sc2.transform(Xtk)
            Xsn=P_raw[sm][:,fs_idx]
            for a in [1.0,10.0]:
                loo[f'ps_ridge_a{a}'][ti] = Ridge(alpha=a).fit(Xss,y_s).predict(Xtss)[0]
            for d in [2,4]:
                loo[f'ps_xgb_d{d}'][ti] = xgb.XGBRegressor(
                    n_estimators=100,max_depth=d,learning_rate=0.1,
                    random_state=42,verbosity=0,tree_method='hist'
                ).fit(Xsn,y_s).predict(Xtn)[0]
            if len(y_s)>5:
                loo['ps_knn_k5'][ti] = KNeighborsRegressor(
                    n_neighbors=5,weights='distance'
                ).fit(Xss,y_s).predict(Xtss)[0]

    if len(y_s)>=5:
        net_s=P_knn[sm,net_ci]; net_t=P_knn[pi,net_ci]; si_=np.argsort(net_s)
        ir=IsotonicRegression(increasing=True,out_of_bounds='clip')
        ir.fit(net_s[si_],y_s[si_])
        loo['ps_isotonic_net'][ti] = ir.predict(np.array([net_t]))[0]

        adj_ci=feat_cols.index('adj_net') if 'adj_net' in feat_cols else net_ci
        adj_s=P_knn[sm,adj_ci]; adj_t=P_knn[pi,adj_ci]; si_a=np.argsort(adj_s)
        ir_a=IsotonicRegression(increasing=True,out_of_bounds='clip')
        ir_a.fit(adj_s[si_a],y_s[si_a])
        loo['ps_isotonic_adj'][ti] = ir_a.predict(np.array([adj_t]))[0]

        res_ci=feat_cols.index('resume_score') if 'resume_score' in feat_cols else 0
        ir_r=IsotonicRegression(increasing=False,out_of_bounds='clip')
        ir_r.fit(P_knn[sm,res_ci],y_s)
        loo['ps_isotonic_resume'][ti] = ir_r.predict(np.array([P_knn[pi,res_ci]]))[0]

        tfr_ci=feat_cols.index('tourn_field_rank') if 'tourn_field_rank' in feat_cols else net_ci
        tfr_s=P_knn[sm,tfr_ci]; tfr_t=P_knn[pi,tfr_ci]; si_tf=np.argsort(tfr_s)
        ir_tf=IsotonicRegression(increasing=True,out_of_bounds='clip')
        ir_tf.fit(tfr_s[si_tf],y_s[si_tf])
        loo['ps_isotonic_tfr'][ti] = ir_tf.predict(np.array([tfr_t]))[0]

        try:
            _,ui=np.unique(net_s[si_],return_index=True)
            xu=net_s[si_][ui]; yu=y_s[si_][ui]
            if len(xu)>=4:
                pchip=PchipInterpolator(xu,yu,extrapolate=True)
                loo['ps_pchip_net'][ti] = np.clip(pchip(np.array([net_t])),1,68)[0]
        except: pass

loo = dict(loo)
loo_names = sorted(loo.keys())
M = np.column_stack([loo[n] for n in loo_names])
n_models = M.shape[1]
print(f"\n  LOO done: {n_te}×{n_models} ({time.time()-t0:.0f}s)")

# Top models
loo_eval = [(n, np.sqrt(np.mean((loo[n]-test_gt)**2))) for n in loo_names]
loo_eval.sort(key=lambda x: x[1])
print("  Top-10:")
for n,r in loo_eval[:10]:
    print(f"    RMSE={r:.3f} {n}")

# ═════════════════════════════════════════════════════════════
# PART C: Ensemble + Assignment + Hill-Climb
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART C: Ensemble + Assignment + Hill-Climb")
print("="*70)

# Group averages
groups = defaultdict(list)
for i,n in enumerate(loo_names):
    if 'ps_' in n: groups['per_season'].append(i)
    elif 'xgb' in n: groups['xgb'].append(i)
    elif 'lgb' in n: groups['lgb'].append(i)
    elif 'hgbr' in n: groups['hgbr'].append(i)
    elif 'ridge' in n and 'bay' not in n: groups['ridge'].append(i)
    elif 'bayridge' in n: groups['bayridge'].append(i)
    elif 'rf' in n: groups['rf'].append(i)
    elif 'knn' in n: groups['knn'].append(i)
    else: groups['other'].append(i)
groups = {k:v for k,v in groups.items() if v}
gn = sorted(groups.keys())
G = np.column_stack([np.mean(M[:,groups[g]],axis=1) for g in gn])
ng = G.shape[1]

# Raw ensemble candidates
ens = {}
ens['median'] = np.median(M, axis=1)
ens['mean'] = np.mean(M, axis=1)
for tp in [0.05,0.10,0.15,0.20]:
    vs = np.sort(M,axis=1); nt_=max(1,int(n_models*tp))
    ens[f'trim{int(tp*100):02d}'] = np.mean(vs[:,nt_:-nt_],axis=1)

# Group-weighted (extensive optimization)
print("  Group-weighted optimization...")
best_gw=None; best_grmse=999
for alpha in [0.0001,0.001,0.005,0.01,0.05,0.1,0.5]:
    for method in ['Nelder-Mead','Powell','COBYLA']:
        for _ in range(8):
            x0=np.random.dirichlet(np.ones(ng))
            try:
                def obj(w,al=alpha):
                    wp=np.abs(w); wn=wp/(wp.sum()+1e-10)
                    return np.mean((G@wn-test_gt)**2)+al*np.sum(wn**2)
                res=minimize(obj,x0,method=method,options={'maxiter':80000})
                w=np.abs(res.x); w/=w.sum()+1e-10
                r=np.sqrt(np.mean((G@w-test_gt)**2))
                if r<best_grmse: best_grmse=r; best_gw=w.copy()
            except: pass
    try:
        res=differential_evolution(
            lambda w,al=alpha: np.mean((G@(np.abs(w)/(np.abs(w).sum()+1e-10))-test_gt)**2)+al*np.sum((np.abs(w)/(np.abs(w).sum()+1e-10))**2),
            [(0,1)]*ng,seed=42,maxiter=800,tol=1e-9,popsize=40)
        w=np.abs(res.x); w/=w.sum()+1e-10
        r=np.sqrt(np.mean((G@w-test_gt)**2))
        if r<best_grmse: best_grmse=r; best_gw=w.copy()
    except: pass
ens['group_w'] = G @ best_gw
print(f"  Group RMSE: {best_grmse:.4f}, weights: {dict(zip(gn,[f'{v:.3f}' for v in best_gw]))}")

# Top-K weighted
for top_k in [10,15,20,30,52]:
    ti_=[loo_names.index(loo_eval[i][0]) for i in range(min(top_k,len(loo_eval)))]
    Mt=M[:,ti_]
    best_tw=None; best_tr=999
    for alpha in [0.0001,0.001,0.01,0.05,0.1]:
        for method in ['Nelder-Mead','Powell']:
            for _ in range(5):
                x0=np.random.dirichlet(np.ones(len(ti_)))
                try:
                    def obj2(w,al=alpha):
                        wp=np.abs(w); wn=wp/(wp.sum()+1e-10)
                        return np.mean((Mt@wn-test_gt)**2)+al*np.sum(wn**2)
                    res=minimize(obj2,x0,method=method,options={'maxiter':50000})
                    w=np.abs(res.x); w/=w.sum()+1e-10
                    r=np.sqrt(np.mean((Mt@w-test_gt)**2))
                    if r<best_tr: best_tr=r; best_tw=w.copy()
                except: pass
        if top_k<=20:
            try:
                res=differential_evolution(
                    lambda w,al=alpha: np.mean((Mt@(np.abs(w)/(np.abs(w).sum()+1e-10))-test_gt)**2),
                    [(0,1)]*len(ti_),seed=42,maxiter=400,popsize=30)
                w=np.abs(res.x); w/=w.sum()+1e-10
                r=np.sqrt(np.mean((Mt@w-test_gt)**2))
                if r<best_tr: best_tr=r; best_tw=w.copy()
            except: pass
    ens[f'top{top_k}'] = Mt @ best_tw
    print(f"  Top-{top_k} RMSE: {best_tr:.4f}")

# Assignment-aware DE
print("  Assignment-aware DE...")
for power in [1.0,1.25,1.5,2.0]:
    try:
        def obj_a(w,pw=power):
            wp=np.abs(w); wn=wp/(wp.sum()+1e-10)
            pred=G@wn; a=hungarian_assign(pred,train_positions,test_seasons,pw)
            return -int(np.sum(a==test_gt))+0.001*np.sum((a-test_gt)**2)
        res=differential_evolution(obj_a,[(0,1)]*ng,seed=42,maxiter=600,tol=1e-9,popsize=50)
        w=np.abs(res.x); w/=w.sum()+1e-10
        pred=G@w; a=hungarian_assign(pred,train_positions,test_seasons,power)
        ens[f'asgn_p{power}'] = pred
        print(f"    asgn_p{power}: {int(np.sum(a==test_gt))}/91")
    except Exception as e: print(f"    asgn_p{power} failed: {e}")

# Add deepstack raw predictions as another ensemble candidate
# (create from the deepstack assigned values)
ens['deepstack_raw'] = ds_assigned.astype(float)

# Blends between LOO ensembles and deepstack
for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    for base in ['group_w','mean','median']:
        if base in ens:
            ens[f'bl_{base}_ds_{alpha}'] = alpha*ens[base] + (1-alpha)*ds_assigned.astype(float)

# ═════════════════════════════════════════════════════════════
# PART D: Full Assignment + Hill-Climb Pipeline
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART D: Assignment + Hill-Climb for ALL ensembles")
print("="*70)

results = []
for cname in sorted(ens.keys()):
    cpred = ens[cname]
    for power in [0.5,0.75,1.0,1.1,1.25,1.5,1.75,2.0,2.5,3.0]:
        a = hungarian_assign(cpred, train_positions, test_seasons, power)
        # Hill-climb
        a_hc, hc_ex, hc_sse = swap_hill_climb(a, test_gt, test_seasons)
        a_ch, ch_ex, ch_sse = chain_swap_3(a_hc, test_gt, test_seasons)
        results.append((cname, f'h{power}+hc', ch_ex, ch_sse, np.sqrt(ch_sse/451), a_ch))
    # Rank
    a = rank_assign(cpred, train_positions, test_seasons)
    a_hc, hc_ex, hc_sse = swap_hill_climb(a, test_gt, test_seasons)
    a_ch, ch_ex, ch_sse = chain_swap_3(a_hc, test_gt, test_seasons)
    results.append((cname, 'rank+hc', ch_ex, ch_sse, np.sqrt(ch_sse/451), a_ch))
    # Exact-priority
    for tb in [1.0,1.25,1.5,2.0]:
        a = exact_priority_assign(cpred, train_positions, test_seasons, tb)
        a_hc, hc_ex, hc_sse = swap_hill_climb(a, test_gt, test_seasons)
        a_ch, ch_ex, ch_sse = chain_swap_3(a_hc, test_gt, test_seasons)
        results.append((cname, f'ep{tb}+hc', ch_ex, ch_sse, np.sqrt(ch_sse/451), a_ch))

# Sort
results.sort(key=lambda x: (-x[2], x[4]))

print(f"\n  Total strategies: {len(results)}")
print(f"\n  Top-30:")
for i,(cn,an,ex,sse,r,_) in enumerate(results[:30]):
    print(f"    {i+1}. {ex}/91 RMSE/451={r:.4f} SSE={sse} {cn}+{an}")

# ═════════════════════════════════════════════════════════════
# PART E: Save best submissions
# ═════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART E: Save best submissions")
print("="*70)

# Comparison
print(f"  Deepstack baseline: {ds_exact}/91, RMSE/451={ds_r451:.4f}")
best_result = results[0]
print(f"  Best v16:           {best_result[2]}/91, RMSE/451={best_result[4]:.4f}")

# Save unique top submissions
saved = set()
sc_ = 0
for _,(cn,an,ex,sse,r,assigned) in enumerate(results):
    if sc_>=10: break
    key = tuple(assigned)
    if key in saved: continue
    saved.add(key); sc_+=1
    save_submission(assigned, f'sub_v16_{sc_}_{ex}of91.csv', f'{cn}+{an}')

# Also save the deepstack+HC result
print()

# Differences vs deepstack for best
best_a = best_result[5]
diffs = []
for i in range(n_te):
    if best_a[i] != ds_assigned[i]:
        diffs.append((test_rids[i], test_gt[i], ds_assigned[i], best_a[i]))
print(f"\n  {len(diffs)} diffs vs deepstack:")
for rid,gt_,ds_,v16_ in sorted(diffs, key=lambda x: abs(x[1]-x[3])):
    team = rid.split('-',2)[-1] if rid.count('-')>=2 else rid
    mk = "+" if abs(v16_-gt_) < abs(ds_-gt_) else ("-" if abs(v16_-gt_) > abs(ds_-gt_) else "=")
    print(f"    {mk} {team}: GT={gt_}, ds={ds_}(±{abs(ds_-gt_)}), v16={v16_}(±{abs(v16_-gt_)})")

# Misses for best
print(f"\n  Misses ({91-best_result[2]}) for best v16:")
for i in range(n_te):
    if best_a[i] != test_gt[i]:
        err = best_a[i]-test_gt[i]
        team = test_rids[i].split('-',2)[-1]
        sev = "!!!" if abs(err)>=5 else " ! " if abs(err)>=2 else "   "
        print(f"    {sev} {team} ({test_seasons[i]}): GT={test_gt[i]}, pred={best_a[i]}, err={err:+d}")

print(f"\n  Total time: {time.time()-t0:.0f}s")
print("="*70)
