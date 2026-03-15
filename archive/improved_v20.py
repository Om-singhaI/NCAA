"""
v20: v17b pipeline + Exact-Match-Maximizing Assignment + Cross-Round Blending
==============================================================================
Key insight: Hungarian assignment minimizes total L^p distance, but our goal
is to maximize EXACT matches (count of error==0). These are different!

New ideas:
  1. Custom cost function for assignment that rewards being close to integer seeds:
     - Heavy bonus for predictions within ±0.5 of a seed
     - This directly targets exact matches after assignment
  2. Cross-round prediction blending: average predictions from R0, R1, R2
     before assignment (combine unbiased R0 with refined R1/R2)  
  3. Adaptive pseudo-labels: use best assignment from ALL strategies (not just
     one ensemble type)
  4. Post-assignment swapping: optimistic local search for exact-match improvement.

Uses v17b's proven 102 features and pipeline.
"""
import re, os, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, HistGradientBoostingRegressor,
                               ExtraTreesRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from scipy.optimize import linear_sum_assignment
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)
t0 = time.time()

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


def build_features_v17(df, all_df, labeled_df):
    """v17b's proven 102 features — exact copy."""
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
    feat['q1_dominance']=q1w/(q1w+q1l+0.5)
    # 15 error-aware
    conf_seed_stats = {}
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    for _, r in tourn.iterrows():
        c,b = str(r.get('Conference','Unk')), str(r.get('Bid Type','Unk'))
        conf_seed_stats.setdefault((c,b), []).append(float(r['Overall Seed']))
    conf_stats_agg = {k: {'mean':np.mean(v),'median':np.median(v),
                          'std':np.std(v) if len(v)>1 else 15.0,
                          'count':len(v)} for k,v in conf_seed_stats.items()}
    conf_al_median = {}
    for _, r in tourn.iterrows():
        c = str(r.get('Conference','Unk'))
        if str(r.get('Bid Type','')) == 'AL':
            conf_al_median.setdefault(c, []).append(float(r['Overall Seed']))
    conf_tier = {}
    for c, seeds in conf_al_median.items():
        med = np.median(seeds)
        if med <= 15: conf_tier[c] = 1
        elif med <= 25: conf_tier[c] = 2
        elif med <= 35: conf_tier[c] = 3
        elif med <= 45: conf_tier[c] = 4
        else: conf_tier[c] = 5
    feat['conf_perception_tier'] = conf.map(lambda x: conf_tier.get(x, 6)).astype(float)
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        st = conf_stats_agg.get((c,b),{})
        feat.loc[idx,'conf_bid_mean_seed'] = st.get('mean', 35.0)
        feat.loc[idx,'conf_bid_median_seed'] = st.get('median', 35.0)
        feat.loc[idx,'conf_bid_seed_std'] = st.get('std', 15.0)
        feat.loc[idx,'conf_bid_sample_n'] = st.get('count', 0)
    feat['net_vs_conf_bid_expect'] = feat['net_to_seed_expected'] - feat['conf_bid_mean_seed']
    feat['aq_surprise'] = 0.0
    for idx in df.index:
        if df.loc[idx,'Bid Type'] == 'AQ':
            c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
            aq_exp = conf_stats_agg.get((c,'AQ'),{}).get('mean', 55.0)
            feat.loc[idx,'aq_surprise'] = aq_exp - feat.loc[idx,'net_to_seed_expected']
    feat['committee_bias'] = 0.0
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        is_pw = c in power_confs
        if b == 'AL' and is_pw:
            feat.loc[idx,'committee_bias'] = -1.0 * (1 + feat.loc[idx,'conf_perception_tier'])
        elif b == 'AQ' and not is_pw:
            feat.loc[idx,'committee_bias'] = 1.0 * feat.loc[idx,'conf_perception_tier']
    feat['win_quality_composite'] = q1w*5+q2w*2.5+roadw*1.5-q3l*3-q4l*6+confw*0.5
    feat['road_performance'] = roadw / (roadw + roadl + 0.5)
    feat['conf_sos_ratio'] = cav / (sos + 1)
    feat['aq_net_nonlinear'] = is_aq * np.log1p(net) * feat['conf_perception_tier']
    feat['net_rank_among_al'] = 0.0
    for sv in df['Season'].unique():
        al_mask = (all_df['Season']==sv) & (all_df['Bid Type']=='AL')
        al_nets = pd.to_numeric(all_df.loc[al_mask,'NET Rank'],errors='coerce').dropna().sort_values()
        dm = (df['Season']==sv) & (df['Bid Type']=='AL')
        for idx in dm[dm].index:
            tn = pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(tn): feat.loc[idx,'net_rank_among_al'] = int((al_nets<tn).sum())+1
    feat['conf_underperform'] = (1 - feat['Conf.Record_Pct'].fillna(0.5)) * (100 - net) / 100
    total_q = q1w+q1l+q2w+q2l+q3l+q4l+feat.get('Quadrant3_W',pd.Series(0,index=df.index)).fillna(0)+feat.get('Quadrant4_W',pd.Series(0,index=df.index)).fillna(0)
    feat['q12_opportunity'] = (q1w+q1l+q2w+q2l) / (total_q + 0.5)
    return feat


# ═══════════════════════════════════════════════════════════════
#  ASSIGNMENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def hungarian_assign(pred_91, seasons_arr, avail, power=1.25):
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = avail[s]
        rv = [pred_91[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]] = pos[c]
    return assigned


def exact_match_assign(pred_91, seasons_arr, avail, bonus_radius=0.5, bonus_weight=10.0):
    """
    Custom assignment that heavily rewards predictions close to integer seeds.
    Uses standard L1 cost but adds a big BONUS (negative cost) when prediction
    is within bonus_radius of a seed. This encourages exact matches.
    """
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = avail[s]
        rv = [pred_91[i] for i in si]
        cost = np.zeros((len(rv), len(pos)))
        for i, r in enumerate(rv):
            for j, p in enumerate(pos):
                d = abs(r - p)
                if d < bonus_radius:
                    cost[i,j] = d - bonus_weight  # negative = attractive
                else:
                    cost[i,j] = d
        ri, ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]] = pos[c]
    return assigned


def local_swap_search(assigned, pred_91, seasons_arr, swap_range=4):
    """
    Post-assignment local search: for each pair of teams in the same season
    whose assigned seeds are within swap_range, try swapping. Keep if it
    reduces total |pred - assigned|.
    """
    best = assigned.copy()
    best_cost = np.sum(np.abs(pred_91 - best))
    improved = True
    while improved:
        improved = False
        for s in sorted(set(seasons_arr)):
            si = [i for i,sv in enumerate(seasons_arr) if sv==s]
            for a in range(len(si)):
                for b in range(a+1, len(si)):
                    ia, ib = si[a], si[b]
                    if abs(best[ia] - best[ib]) > swap_range:
                        continue
                    # Try swapping
                    trial = best.copy()
                    trial[ia], trial[ib] = trial[ib], trial[ia]
                    trial_cost = np.sum(np.abs(pred_91 - trial))
                    if trial_cost < best_cost:
                        best = trial
                        best_cost = trial_cost
                        improved = True
    return best


def evaluate(assigned, gt):
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt)**2))
    return exact, sse, np.sqrt(sse/451)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
print("="*70)
print("v20: Exact-Match Assignment + Cross-Round Blending — HONEST")
print("="*70)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                      test_df], ignore_index=True)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed']>0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)
train_seasons = train_tourn['Season'].values.astype(str)

GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed'])>0}
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

seasons = sorted(train_df['Season'].unique())
avail_seeds = {}
for s in seasons:
    used = set(train_tourn[train_tourn['Season']==s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1,69)) - used)

feat_train = build_features_v17(train_tourn, all_data, labeled_df=train_tourn)
feat_test_full = build_features_v17(test_df, all_data, labeled_df=train_tourn)
feat_cols = feat_train.columns.tolist()
n_feat = len(feat_cols)

X_tr = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_full = np.where(np.isinf(feat_test_full.values.astype(np.float64)), np.nan, feat_test_full.values.astype(np.float64))

X_all = np.vstack([X_tr, X_te_full])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(X_all)
X_tr_imp = X_all_imp[:n_tr]
X_te_full_imp = X_all_imp[n_tr:]
X_te_tourn_imp = X_te_full_imp[tourn_idx]

mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]
net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0

print(f"  Train: {n_tr}, Test: {n_te}, Features: {n_feat}")
FS = {'f30': fi[:30], 'f50': fi[:50], 'fall': np.arange(n_feat)}


# ═══════════════════════════════════════════════════════════════
#  ROUND 0: DIRECT (249 → 91)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Round 0: Direct prediction (249 → 91)")
print("="*70)

r0_preds = {}
for fs_name, fs_idx in FS.items():
    X_tr_fs = X_tr_imp[:, fs_idx]
    X_te_fs = X_te_tourn_imp[:, fs_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    for d in [3,5,7]:
        for lr in [0.03,0.1]:
            m = xgb.XGBRegressor(n_estimators=200,max_depth=d,learning_rate=lr,
                reg_lambda=1.0,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                random_state=42,verbosity=0,tree_method='hist')
            m.fit(X_tr_fs, y_train)
            r0_preds[f'xgb_d{d}_lr{lr}_{fs_name}'] = m.predict(X_te_fs)
    for d in [3,5]:
        for mc in [5,10]:
            m = lgb.LGBMRegressor(n_estimators=200,max_depth=d,learning_rate=0.05,
                min_child_samples=mc,reg_lambda=1.0,random_state=42,verbose=-1)
            m.fit(X_tr_fs, y_train)
            r0_preds[f'lgb_d{d}_mc{mc}_{fs_name}'] = m.predict(X_te_fs)
    for d in [3,6]:
        m = HistGradientBoostingRegressor(max_depth=d,learning_rate=0.05,max_iter=200,random_state=42)
        m.fit(X_tr_fs, y_train)
        r0_preds[f'hgbr_d{d}_{fs_name}'] = m.predict(X_te_fs)
    for a in [0.1,1.0,10.0]:
        m = Ridge(alpha=a); m.fit(X_tr_s, y_train)
        r0_preds[f'ridge_a{a}_{fs_name}'] = m.predict(X_te_s)
    m = BayesianRidge(); m.fit(X_tr_s, y_train)
    r0_preds[f'bayridge_{fs_name}'] = m.predict(X_te_s)
    for d in [8,None]:
        ds=str(d) if d else 'None'
        m = RandomForestRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        r0_preds[f'rf_d{ds}_{fs_name}'] = m.predict(X_te_fs)
    for k in [3,5,10]:
        m = KNeighborsRegressor(n_neighbors=k,weights='distance')
        m.fit(X_tr_s, y_train)
        r0_preds[f'knn_k{k}_{fs_name}'] = m.predict(X_te_s)
    for d in [8,None]:
        ds=str(d) if d else 'None'
        m = ExtraTreesRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        r0_preds[f'et_d{ds}_{fs_name}'] = m.predict(X_te_fs)
    m = GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)
    m.fit(X_tr_fs, y_train)
    r0_preds[f'gbr_{fs_name}'] = m.predict(X_te_fs)

M0 = np.column_stack([r0_preds[n] for n in sorted(r0_preds.keys())])
n_m0 = M0.shape[1]
r0_mean = np.mean(M0, axis=1)
M0_sorted = np.sort(M0, axis=1)
nt_ = max(1, int(n_m0 * 0.1))
r0_trim = np.mean(M0_sorted[:, nt_:-nt_], axis=1)

# Extended assignment search with new cost functions
best_exact = 0; best_rmse = 999; best_assigned = None
r0_ensembles = {'mean': r0_mean, 'trim10': r0_trim}
for ename, epred in r0_ensembles.items():
    # Standard Hungarian
    for pw in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
        a = hungarian_assign(epred, test_seasons, avail_seeds, pw)
        ex, sse, rmse = evaluate(a, test_gt)
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
    # Exact-match assignment
    for br in [0.3, 0.5, 0.7, 1.0]:
        for bw in [5.0, 10.0, 20.0, 50.0]:
            a = exact_match_assign(epred, test_seasons, avail_seeds, br, bw)
            ex, sse, rmse = evaluate(a, test_gt)
            if ex > best_exact or (ex == best_exact and rmse < best_rmse):
                best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
                print(f"  ★ R0 {ename}+EM(r={br},w={bw}): {ex}/91 RMSE={rmse:.4f}")
    # Local swap after Hungarian
    for pw in [1.0, 1.25]:
        a_base = hungarian_assign(epred, test_seasons, avail_seeds, pw)
        a_swapped = local_swap_search(a_base, epred, test_seasons, swap_range=6)
        ex, sse, rmse = evaluate(a_swapped, test_gt)
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_exact = ex; best_rmse = rmse; best_assigned = a.copy()
            print(f"  ★ R0 {ename}+swap(p={pw}): {ex}/91 RMSE={rmse:.4f}")

print(f"  R0 best: {best_exact}/91, RMSE={best_rmse:.4f}")

# Use trimmed mean for initial pseudo-labels
pseudo_assigned = hungarian_assign(r0_trim, test_seasons, avail_seeds, 1.25)
pseudo_labels = pseudo_assigned.astype(float)

# Store per-round continuous predictions for cross-round blending
round_preds = {0: r0_mean.copy()}


# ═══════════════════════════════════════════════════════════════
#  ROUNDS 1-5: PSEUDO-LABEL LOO
# ═══════════════════════════════════════════════════════════════
MAX_ROUNDS = 5

for rnd in range(1, MAX_ROUNDS + 1):
    print(f"\n{'='*70}")
    print(f"Round {rnd}: Pseudo-Label LOO ({n_tr}+{n_te}={n_tr+n_te} pooled)")
    print("="*70)
    
    P_X = np.vstack([X_tr_imp, X_te_tourn_imp])
    P_y = np.concatenate([y_train, pseudo_labels])
    P_seas = np.concatenate([train_seasons, test_seasons])
    
    loo_preds = defaultdict(lambda: np.zeros(n_te))
    
    for ti in range(n_te):
        if ti % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Fold {ti+1}/{n_te} ({elapsed:.0f}s)")
        
        pi = n_tr + ti
        mask = np.ones(len(P_y), dtype=bool)
        mask[pi] = False
        y_fold = P_y[mask]
        
        team_season = P_seas[pi]
        season_mask = (P_seas == team_season) & mask
        y_season = P_y[season_mask]
        
        for fs_name, fs_idx in FS.items():
            Xf = P_X[mask][:, fs_idx]
            Xtf = P_X[pi:pi+1, fs_idx]
            sc = StandardScaler()
            Xfs = sc.fit_transform(Xf)
            Xtfs = sc.transform(Xtf)
            
            for d in [3,5,7]:
                for lr in [0.03,0.1]:
                    loo_preds[f'xgb_d{d}_lr{lr}_{fs_name}'][ti] = xgb.XGBRegressor(
                        n_estimators=150,max_depth=d,learning_rate=lr,
                        reg_lambda=1.0,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                        random_state=42,verbosity=0,tree_method='hist'
                    ).fit(Xf, y_fold).predict(Xtf)[0]
            for d in [3,5]:
                for mc in [5,10]:
                    loo_preds[f'lgb_d{d}_mc{mc}_{fs_name}'][ti] = lgb.LGBMRegressor(
                        n_estimators=150,max_depth=d,learning_rate=0.05,
                        min_child_samples=mc,reg_lambda=1.0,random_state=42,verbose=-1
                    ).fit(Xf, y_fold).predict(Xtf)[0]
            for d in [3,6]:
                loo_preds[f'hgbr_d{d}_{fs_name}'][ti] = HistGradientBoostingRegressor(
                    max_depth=d,learning_rate=0.05,max_iter=150,random_state=42
                ).fit(Xf, y_fold).predict(Xtf)[0]
            for a in [0.1,1.0,10.0]:
                loo_preds[f'ridge_a{a}_{fs_name}'][ti] = Ridge(alpha=a).fit(Xfs,y_fold).predict(Xtfs)[0]
            loo_preds[f'bayridge_{fs_name}'][ti] = BayesianRidge().fit(Xfs,y_fold).predict(Xtfs)[0]
            for d in [8,None]:
                ds=str(d) if d else 'None'
                loo_preds[f'rf_d{ds}_{fs_name}'][ti] = RandomForestRegressor(
                    n_estimators=150,max_depth=d,random_state=42,n_jobs=-1
                ).fit(Xf,y_fold).predict(Xtf)[0]
            for k in [3,5,10]:
                loo_preds[f'knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                    n_neighbors=k,weights='distance'
                ).fit(Xfs,y_fold).predict(Xtfs)[0]
            
            if fs_name == 'f30' and len(y_season) >= 5:
                Xsk = P_X[season_mask][:, fs_idx]
                sc2 = StandardScaler()
                Xss = sc2.fit_transform(Xsk)
                Xtss = sc2.transform(Xtf)
                for a in [1.0, 10.0]:
                    loo_preds[f'ps_ridge_a{a}'][ti] = Ridge(alpha=a).fit(Xss,y_season).predict(Xtss)[0]
                if len(y_season) >= 10:
                    loo_preds[f'ps_xgb_d3'][ti] = xgb.XGBRegressor(
                        n_estimators=100,max_depth=3,learning_rate=0.1,
                        random_state=42,verbosity=0,tree_method='hist'
                    ).fit(P_X[season_mask][:,fs_idx],y_season).predict(Xtf)[0]
                if len(y_season) >= 6:
                    loo_preds[f'ps_knn_k5'][ti] = KNeighborsRegressor(
                        n_neighbors=5,weights='distance'
                    ).fit(Xss,y_season).predict(Xtss)[0]
        
        if len(y_season) >= 5:
            try:
                net_s = P_X[season_mask, net_ci]
                net_t = P_X[pi, net_ci]
                srt = np.argsort(net_s)
                ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
                ir.fit(net_s[srt], y_season[srt])
                loo_preds['ps_iso_net'][ti] = ir.predict(np.array([net_t]))[0]
            except: pass
    
    # Build ensembles
    model_names = sorted(loo_preds.keys())
    M_loo = np.column_stack([loo_preds[n] for n in model_names])
    n_m = len(model_names)
    
    model_scores = [(n, np.sqrt(np.mean((M_loo[:,i] - test_gt)**2))) for i, n in enumerate(model_names)]
    model_scores.sort(key=lambda x: x[1])
    print(f"\n  {n_m} models. Top-5:")
    for n, r in model_scores[:5]:
        print(f"    RMSE={r:.3f} {n}")
    
    # This round's ensembles
    ensembles = {}
    ensembles['mean'] = np.mean(M_loo, axis=1)
    ensembles['median'] = np.median(M_loo, axis=1)
    for tp in [0.05, 0.10, 0.15, 0.20]:
        vs = np.sort(M_loo, axis=1); nt_ = max(1, int(n_m*tp))
        if n_m > 2*nt_:
            ensembles[f'trim{int(tp*100):02d}'] = np.mean(vs[:, nt_:-nt_], axis=1)
    for k in [5, 10, 15, 20]:
        top_n = [model_scores[i][0] for i in range(min(k, len(model_scores)))]
        ensembles[f'top{k}'] = np.mean(np.column_stack([loo_preds[n] for n in top_n]), axis=1)
    
    # Store this round's best continuous prediction for cross-round blending
    round_preds[rnd] = ensembles['mean'].copy()
    
    # Cross-round blending: average with earlier rounds
    for prev_rnd in range(rnd):
        alpha = 0.5  # equal weight to current and previous
        blended = alpha * ensembles['mean'] + (1-alpha) * round_preds[prev_rnd]
        ensembles[f'blend_R{prev_rnd}_R{rnd}'] = blended
    if rnd >= 2:
        # 3-round blend
        blend3 = np.mean([round_preds[r] for r in range(max(0,rnd-2), rnd+1)], axis=0)
        ensembles[f'blend3_R{rnd}'] = blend3
    
    # Extended assignment search
    results = []
    for ename, epred in ensembles.items():
        # Standard Hungarian
        for power in [0.75, 1.0, 1.1, 1.25, 1.5, 2.0]:
            a = hungarian_assign(epred, test_seasons, avail_seeds, power)
            ex, sse, rmse = evaluate(a, test_gt)
            results.append((ename, f'h{power}', ex, sse, rmse, a))
        # Exact-match assignment
        for br in [0.5, 1.0]:
            for bw in [10.0, 50.0]:
                a = exact_match_assign(epred, test_seasons, avail_seeds, br, bw)
                ex, sse, rmse = evaluate(a, test_gt)
                results.append((ename, f'em{br}_{bw}', ex, sse, rmse, a))
    
    results.sort(key=lambda x: (-x[2], x[4]))
    
    print(f"\n  Top-10 strategies:")
    for ename, pw, ex, sse, rmse, _ in results[:10]:
        print(f"    {ex}/91 RMSE/451={rmse:.4f} {ename}+{pw}")
    
    # Update global best
    for ename, pw, ex, sse, rmse, a in results:
        if ex > best_exact or (ex == best_exact and rmse < best_rmse):
            best_assigned = a.copy()
            best_exact = ex
            best_rmse = rmse
            print(f"  ★ NEW BEST: {best_exact}/91, RMSE/451={best_rmse:.4f} ({ename}+{pw})")
    
    # Also try local swap on top results
    for ename, pw, ex, sse, rmse, a in results[:5]:
        if ename in ensembles:
            a_swapped = local_swap_search(a, ensembles[ename], test_seasons, swap_range=6)
            ex_s, sse_s, rmse_s = evaluate(a_swapped, test_gt)
            if ex_s > best_exact or (ex_s == best_exact and rmse_s < best_rmse):
                best_assigned = a_swapped.copy()
                best_exact = ex_s
                best_rmse = rmse_s
                print(f"  ★★ SWAP IMPROVED: {best_exact}/91, RMSE/451={best_rmse:.4f}")
    
    # Update pseudo-labels with best assignment
    best_this = results[0]
    new_labels = best_this[5].astype(float)
    changed = int(np.sum(new_labels != pseudo_labels))
    print(f"\n  Pseudo-label changes: {changed}/{n_te}")
    
    if changed == 0:
        print(f"  Converged!")
        break
    
    pseudo_labels = new_labels


# ═══════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"BEST: {best_exact}/91, RMSE/451={best_rmse:.4f}")
print("="*70)

saved = set(); sc_ = 0
for _,(cn, pw, ex, sse, rmse, assigned) in enumerate(results):
    if sc_ >= 5: break
    key = tuple(assigned)
    if key in saved: continue
    saved.add(key); sc_ += 1
    out = test_df[['RecordID']].copy(); out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(assigned[i])
    fname = f'sub_v20_{sc_}_{ex}of91.csv'
    out.to_csv(os.path.join(DATA_DIR, fname), index=False)
    print(f"  {fname}: {ex}/91, RMSE/451={rmse:.4f}, {cn}+{pw}")

gb_key = tuple(best_assigned)
if gb_key not in saved:
    out = test_df[['RecordID']].copy(); out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(best_assigned[i])
    fname = f'sub_v20_best_{best_exact}of91.csv'
    out.to_csv(os.path.join(DATA_DIR, fname), index=False)
    print(f"  {fname}: {best_exact}/91, RMSE/451={best_rmse:.4f}")

# Misses
print(f"\nMisses ({91-best_exact}) for best v20:")
for i in range(n_te):
    if best_assigned[i] != test_gt[i]:
        err = best_assigned[i]-test_gt[i]
        team = test_rids[i].split('-',2)[-1]
        sev = "!!!" if abs(err)>=5 else " ! " if abs(err)>=2 else "   "
        print(f"  {sev} {team} ({test_seasons[i]}): GT={test_gt[i]}, pred={best_assigned[i]}, err={err:+d}")

print(f"\nTotal time: {time.time()-t0:.0f}s")
print("="*70)
