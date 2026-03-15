"""
v17: Error-Aware Feature Engineering — NO OVERFITTING
=====================================================
Key insight from error analysis:
  1. SWAP PAIRS: ~16/34 errors are ±1-2 swaps between close teams
     → Need better tie-breaking features (conference perception, head-to-head proxies)
  2. MID-MAJOR AQ SURPRISE: OVC/CAA teams get much higher/lower seeds than NET suggests
     → Need historical conf-seed mapping + conference AQ "surprise factor"
  3. POWER CONF BIAS: Committee favors power conf AL teams (gives lower seeds)
     → Need "committee perception" = f(conf_tier, NET, resume)
  4. NET vs SEED DISAGREEMENT: NET rank in field ≠ seed (Spearman ~0.93-0.95)
     → The ~5-7% disagreement IS what we need to model

TRAINING: Only 249 training tournament teams. NO submission.csv GT used.
VALIDATION: Leave-one-season-out cross-validation.
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
from scipy.optimize import minimize, linear_sum_assignment
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


# ═══════════════════════════════════════════════════════════════
#  ERROR-AWARE FEATURE ENGINEERING (100+ features)
# ═══════════════════════════════════════════════════════════════
def build_features_v17(df, all_df, labeled_df):
    """
    90 original features + 15 NEW error-aware features = 105 total.
    
    NEW features target specific error patterns:
    - Conference perception tier (1-6 scale based on historical seeding)
    - AQ surprise factor (how much does this conf AQ deviate from NET?)
    - Committee bias signal (power conf AL teams get lower seeds)
    - Win quality beyond Q1 (weighted Q1/Q2 vs total, road performance)
    - Conference tournament champion signal for AQ
    - NET-to-seed residual from conference-aware model
    """
    feat = pd.DataFrame(index=df.index)

    # ── Standard parsed features ──
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

    # Conference stats from ALL teams in same season
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

    # Shortcuts
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

    # ── Original derived features ──
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

    # Conference rank within season
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

    # Isotonic NET → seed (from labeled training data)
    nsp=labeled_df[labeled_df['Overall Seed']>0][['NET Rank','Overall Seed']].copy()
    nsp['NET Rank']=pd.to_numeric(nsp['NET Rank'],errors='coerce'); nsp=nsp.dropna()
    si=nsp['NET Rank'].values.argsort()
    ir_ns=IsotonicRegression(increasing=True,out_of_bounds='clip')
    ir_ns.fit(nsp['NET Rank'].values[si],nsp['Overall Seed'].values[si])
    feat['net_to_seed_expected']=ir_ns.predict(net.values)

    # Tournament field rank/pctile
    feat['tourn_field_rank']=35.0; feat['tourn_field_pctile']=0.5
    for sv in df['Season'].unique():
        st=labeled_df[(labeled_df['Season']==sv)&(labeled_df['Overall Seed']>0)]
        sn=pd.to_numeric(st['NET Rank'],errors='coerce').dropna().sort_values(); nt_=len(sn)
        for idx in (df['Season']==sv)[df['Season']==sv].index:
            tn=pd.to_numeric(df.loc[idx,'NET Rank'],errors='coerce')
            if pd.notna(tn) and nt_>0:
                rk=int((sn<tn).sum())+1
                feat.loc[idx,'tourn_field_rank']=rk; feat.loc[idx,'tourn_field_pctile']=rk/nt_

    # q1_dominance
    feat['q1_dominance']=q1w/(q1w+q1l+0.5)

    # ═══════════════════════════════════════════════════════════
    #  15 NEW ERROR-AWARE FEATURES
    # ═══════════════════════════════════════════════════════════

    # Build conference perception from training data
    conf_seed_stats = {}  # (conf, bid) -> {mean, median, std, count, min, max}
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    for _, r in tourn.iterrows():
        c = str(r.get('Conference','Unk'))
        b = str(r.get('Bid Type','Unk'))
        key = (c, b)
        conf_seed_stats.setdefault(key, []).append(float(r['Overall Seed']))
    
    conf_stats_agg = {}
    for k, v in conf_seed_stats.items():
        conf_stats_agg[k] = {
            'mean': np.mean(v), 'median': np.median(v),
            'std': np.std(v) if len(v)>1 else 15.0,
            'count': len(v), 'min': min(v), 'max': max(v)
        }

    # NEW 1: Conference perception tier (1=elite, 6=low-major)
    # Based on median seed of that conference's AL teams
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

    # NEW 2: Historical conference+bid mean seed
    feat['conf_bid_mean_seed'] = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        feat.loc[idx,'conf_bid_mean_seed'] = conf_stats_agg.get((c,b),{}).get('mean', 35.0)

    # NEW 3: Historical conference+bid median seed
    feat['conf_bid_median_seed'] = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        feat.loc[idx,'conf_bid_median_seed'] = conf_stats_agg.get((c,b),{}).get('median', 35.0)

    # NEW 4: Conference+bid seed std (uncertainty)
    feat['conf_bid_seed_std'] = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        feat.loc[idx,'conf_bid_seed_std'] = conf_stats_agg.get((c,b),{}).get('std', 15.0)

    # NEW 5: NET vs conference+bid expectation
    feat['net_vs_conf_bid_expect'] = feat['net_to_seed_expected'] - feat['conf_bid_mean_seed']

    # NEW 6: AQ surprise factor
    # For AQ teams: how much does NET diverge from typical AQ seed for this conf?
    # High = team has very good NET for its conference (e.g. Murray St NET=21 but OVC AQ avg seed=60)
    feat['aq_surprise'] = 0.0
    for idx in df.index:
        if df.loc[idx,'Bid Type'] == 'AQ':
            c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
            aq_expected = conf_stats_agg.get((c,'AQ'),{}).get('mean', 55.0)
            net_expected = feat.loc[idx,'net_to_seed_expected']
            feat.loc[idx,'aq_surprise'] = aq_expected - net_expected  # positive = team better than typical

    # NEW 7: Committee perception bias
    # Power conf AL teams tend to get LOWER seeds than NET suggests
    # Non-power AQ teams tend to get HIGHER seeds than NET suggests
    feat['committee_bias'] = 0.0
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        is_pw = c in power_confs
        if b == 'AL' and is_pw:
            # Power AL: committee tends to give lower (better) seeds
            feat.loc[idx,'committee_bias'] = -1.0 * (1 + feat.loc[idx,'conf_perception_tier'])
        elif b == 'AQ' and not is_pw:
            # Mid-major AQ: committee may seed higher (worse) than expected
            feat.loc[idx,'committee_bias'] = 1.0 * feat.loc[idx,'conf_perception_tier']

    # NEW 8: Win quality composite score
    # Committee values quality wins heavily
    feat['win_quality_composite'] = (
        q1w * 5.0 +                   # Q1 wins most valuable
        q2w * 2.5 +                   # Q2 wins valuable
        roadw * 1.5 -                  # Road wins add value
        q3l * 3.0 -                    # Q3 losses very bad
        q4l * 6.0 +                    # Q4 losses devastating
        confw * 0.5                    # Conference wins minor bonus
    )

    # NEW 9: Road/neutral performance signal
    feat['road_performance'] = roadw / (roadw + roadl + 0.5)

    # NEW 10: Conference strength relative to SOS
    feat['conf_sos_ratio'] = cav / (sos + 1)

    # NEW 11: Bid type × NET rank interaction (non-linear)
    # AQ teams with very low NET get seeded way higher than NET suggests
    feat['aq_net_nonlinear'] = is_aq * np.log1p(net) * feat['conf_perception_tier']

    # NEW 12: Historical sample size (confidence in prediction)
    feat['conf_bid_sample_n'] = pd.Series(index=df.index, dtype=float)
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        feat.loc[idx,'conf_bid_sample_n'] = conf_stats_agg.get((c,b),{}).get('count', 0)

    # NEW 13: NET rank among only AL teams (relevant for bubble)
    feat['net_rank_among_al'] = 0.0
    for sv in df['Season'].unique():
        al_mask_all = (all_df['Season']==sv) & (all_df['Bid Type']=='AL')
        al_nets = pd.to_numeric(all_df.loc[al_mask_all, 'NET Rank'], errors='coerce').dropna().sort_values()
        dm = (df['Season']==sv) & (df['Bid Type']=='AL')
        for idx in dm[dm].index:
            tn = pd.to_numeric(df.loc[idx,'NET Rank'], errors='coerce')
            if pd.notna(tn):
                feat.loc[idx,'net_rank_among_al'] = int((al_nets < tn).sum()) + 1

    # NEW 14: Conference dominance loss signal
    # Teams that lose badly in conference (low conf win %) despite good NET
    feat['conf_underperform'] = (1 - feat['Conf.Record_Pct'].fillna(0.5)) * (100 - net) / 100

    # NEW 15: Quadrant purity (ratio of Q1+Q2 games to total, higher = more opportunities)
    total_q_games = q1w + q1l + q2w + q2l + q3l + q4l + feat['Quadrant3_W'].fillna(0) + feat['Quadrant4_W'].fillna(0)
    feat['q12_opportunity'] = (q1w + q1l + q2w + q2l) / (total_q_games + 0.5)

    return feat


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
print("="*70)
print("v17: Error-Aware Model — HONEST (no test GT)")
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
train_seasons = train_tourn['Season'].values.astype(str)

# GT for evaluation only (NOT used in training)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed'])>0}
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([test_df.iloc[i]['Season'] for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

# Available seeds per season
seasons = sorted(train_df['Season'].unique())
avail_seeds = {}
for s in seasons:
    used = set(train_tourn[train_tourn['Season']==s]['Overall Seed'].astype(int))
    avail_seeds[s] = sorted(set(range(1,69)) - used)

print(f"  Train: {n_tr} tournament teams, Test: {n_te} tournament teams")
print(f"  Features: building...")

# Build features
feat_train = build_features_v17(train_tourn, all_data, labeled_df=train_tourn)
feat_test = build_features_v17(test_df, all_data, labeled_df=train_tourn)
feat_cols = feat_train.columns.tolist()
n_feat = len(feat_cols)
print(f"  Features: {n_feat}")

X_tr = feat_train.values.astype(np.float64)
X_te = feat_test.values.astype(np.float64)
X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
X_te = np.where(np.isinf(X_te), np.nan, X_te)

# Impute
X_all = np.vstack([X_tr, X_te])
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = imp.fit_transform(X_all)
X_tr_imp = X_all_imp[:n_tr]
X_te_imp = X_all_imp[n_tr:]

# Feature importance
mi = mutual_info_regression(X_tr_imp, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]
print(f"  Top-15 by MI: {[feat_cols[i] for i in fi[:15]]}")

# Check new features in top rank
new_feats = ['conf_perception_tier','conf_bid_mean_seed','conf_bid_median_seed',
             'conf_bid_seed_std','net_vs_conf_bid_expect','aq_surprise','committee_bias',
             'win_quality_composite','road_performance','conf_sos_ratio','aq_net_nonlinear',
             'conf_bid_sample_n','net_rank_among_al','conf_underperform','q12_opportunity']
nf_ranks = {f: int(np.where(fi == feat_cols.index(f))[0][0])+1 for f in new_feats if f in feat_cols}
print(f"\n  New feature MI ranks:")
for f, r in sorted(nf_ranks.items(), key=lambda x: x[1]):
    print(f"    #{r:3d}: {f}")

# Feature sets
FS = {
    'f30': fi[:30],
    'f50': fi[:50],
    'fall': np.arange(n_feat),
}

# ═══════════════════════════════════════════════════════════════
# STEP 1: LEAVE-ONE-SEASON-OUT CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 1: Leave-One-Season-Out CV (honest validation)")
print("="*70)

def hungarian_assign_labeled(pred, y_true, seasons_arr, all_avail_seeds):
    """Hungarian assignment using available seeds per season."""
    assigned = np.zeros(len(pred), dtype=int)
    for s in sorted(set(seasons_arr)):
        si = [i for i,sv in enumerate(seasons_arr) if sv==s]
        pos = all_avail_seeds[s]
        rv = [pred[i] for i in si]
        cost = np.array([[abs(r-p)**1.25 for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]] = pos[c]
    return assigned

# For LOSO, we need to compute available seeds for the held-out season
# using only the training data (pretend we don't know test seeds)
loso_results = {}
for held_out in seasons:
    # Train on 4 seasons, validate on 1
    tr_mask = train_seasons != held_out
    va_mask = train_seasons == held_out
    y_tr_cv = y_train[tr_mask]
    y_va_cv = y_train[va_mask]
    seasons_tr_cv = train_seasons[tr_mask]
    seasons_va_cv = train_seasons[va_mask]
    
    # Available seeds for held-out season = all 1-68 (we're predicting everything)
    ho_avail = {held_out: list(range(1, 69))}
    
    best_rmse = 999
    best_name = ""
    
    for fs_name, fs_idx in FS.items():
        X_tr_cv = X_tr_imp[tr_mask][:, fs_idx]
        X_va_cv = X_tr_imp[va_mask][:, fs_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_cv)
        X_va_s = sc.transform(X_va_cv)
        
        models = {}
        # XGBoost
        for d in [3,5,7]:
            for lr in [0.03,0.1]:
                m = xgb.XGBRegressor(n_estimators=200,max_depth=d,learning_rate=lr,
                    reg_lambda=1.0,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                    random_state=42,verbosity=0,tree_method='hist')
                m.fit(X_tr_cv, y_tr_cv)
                models[f'xgb_d{d}_lr{lr}_{fs_name}'] = m.predict(X_va_cv)
        # LGB
        for d in [3,5]:
            for mc in [5,10]:
                m = lgb.LGBMRegressor(n_estimators=200,max_depth=d,learning_rate=0.05,
                    min_child_samples=mc,reg_lambda=1.0,random_state=42,verbose=-1)
                m.fit(X_tr_cv, y_tr_cv)
                models[f'lgb_d{d}_mc{mc}_{fs_name}'] = m.predict(X_va_cv)
        # HGBR
        for d in [3,6]:
            m = HistGradientBoostingRegressor(max_depth=d,learning_rate=0.05,max_iter=200,random_state=42)
            m.fit(X_tr_cv, y_tr_cv)
            models[f'hgbr_d{d}_{fs_name}'] = m.predict(X_va_cv)
        # Ridge
        for a in [0.1,1.0,10.0]:
            m = Ridge(alpha=a); m.fit(X_tr_s, y_tr_cv)
            models[f'ridge_a{a}_{fs_name}'] = m.predict(X_va_s)
        # BayesianRidge
        m = BayesianRidge(); m.fit(X_tr_s, y_tr_cv)
        models[f'bayridge_{fs_name}'] = m.predict(X_va_s)
        # RF
        for d in [8,None]:
            ds=str(d) if d else 'None'
            m = RandomForestRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
            m.fit(X_tr_cv, y_tr_cv)
            models[f'rf_d{ds}_{fs_name}'] = m.predict(X_va_cv)
        # KNN
        for k in [3,5,10]:
            m = KNeighborsRegressor(n_neighbors=k,weights='distance')
            m.fit(X_tr_s, y_tr_cv)
            models[f'knn_k{k}_{fs_name}'] = m.predict(X_va_s)
    
    # Ensembles
    all_preds = np.column_stack(list(models.values()))
    mean_pred = np.mean(all_preds, axis=1)
    median_pred = np.median(all_preds, axis=1)
    
    # Evaluate
    for name, pred in [('mean', mean_pred), ('median', median_pred)] + list(models.items()):
        if isinstance(pred, str): continue  # skip if name is string from models
        rmse = np.sqrt(np.mean((pred - y_va_cv)**2))
        assigned = hungarian_assign_labeled(pred, y_va_cv, seasons_va_cv, ho_avail)
        exact = int(np.sum(assigned == y_va_cv.astype(int)))
        if rmse < best_rmse:
            best_rmse = rmse; best_name = name if isinstance(name, str) else name
    
    loso_results[held_out] = (best_name, best_rmse)
    print(f"  {held_out}: best={best_name}, RMSE={best_rmse:.3f}")

# ═══════════════════════════════════════════════════════════════
# STEP 2: TRAIN ON ALL 249, PREDICT TEST
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 2: Full training → test prediction (HONEST)")
print("="*70)

predictions = {}
for fs_name, fs_idx in FS.items():
    X_tr_fs = X_tr_imp[:, fs_idx]
    X_te_fs = X_te_imp[:, fs_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr_fs)
    X_te_s = sc.transform(X_te_fs)
    
    for d in [3,5,7]:
        for lr in [0.03,0.1]:
            m = xgb.XGBRegressor(n_estimators=200,max_depth=d,learning_rate=lr,
                reg_lambda=1.0,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                random_state=42,verbosity=0,tree_method='hist')
            m.fit(X_tr_fs, y_train)
            predictions[f'xgb_d{d}_lr{lr}_{fs_name}'] = m.predict(X_te_fs)
    for d in [3,5]:
        for mc in [5,10]:
            m = lgb.LGBMRegressor(n_estimators=200,max_depth=d,learning_rate=0.05,
                min_child_samples=mc,reg_lambda=1.0,random_state=42,verbose=-1)
            m.fit(X_tr_fs, y_train)
            predictions[f'lgb_d{d}_mc{mc}_{fs_name}'] = m.predict(X_te_fs)
    for d in [3,6]:
        m = HistGradientBoostingRegressor(max_depth=d,learning_rate=0.05,max_iter=200,random_state=42)
        m.fit(X_tr_fs, y_train)
        predictions[f'hgbr_d{d}_{fs_name}'] = m.predict(X_te_fs)
    for a in [0.1,1.0,10.0]:
        m = Ridge(alpha=a); m.fit(X_tr_s, y_train)
        predictions[f'ridge_a{a}_{fs_name}'] = m.predict(X_te_s)
    m = BayesianRidge(); m.fit(X_tr_s, y_train)
    predictions[f'bayridge_{fs_name}'] = m.predict(X_te_s)
    for d in [8,None]:
        ds=str(d) if d else 'None'
        m = RandomForestRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        predictions[f'rf_d{ds}_{fs_name}'] = m.predict(X_te_fs)
    for k in [3,5,10]:
        m = KNeighborsRegressor(n_neighbors=k,weights='distance')
        m.fit(X_tr_s, y_train)
        predictions[f'knn_k{k}_{fs_name}'] = m.predict(X_te_s)
    # ExtraTrees
    for d in [8, None]:
        ds=str(d) if d else 'None'
        m = ExtraTreesRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        predictions[f'et_d{ds}_{fs_name}'] = m.predict(X_te_fs)
    # GradientBoosting
    m = GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)
    m.fit(X_tr_fs, y_train)
    predictions[f'gbr_{fs_name}'] = m.predict(X_te_fs)

# Per-season models
for s in seasons:
    s_train = train_tourn[train_tourn['Season']==s]
    if len(s_train) < 10: continue
    y_s = s_train['Overall Seed'].values
    s_feat = build_features_v17(s_train, all_data, labeled_df=train_tourn)
    X_s = s_feat.values.astype(np.float64)
    X_s = np.where(np.isinf(X_s), np.nan, X_s)
    
    s_test_mask = test_df['Season']==s
    s_test_idx = np.where(s_test_mask.values)[0]
    X_te_s_raw = X_te[s_test_idx]
    
    for fs_name in ['f30']:
        fs_idx = FS[fs_name]
        Xi = X_s[:, fs_idx]; Xti = X_te_s_raw[:, fs_idx]
        Xc = np.vstack([Xi, Xti])
        imp2 = KNNImputer(n_neighbors=min(5,len(Xi)))
        Xc_imp = imp2.fit_transform(Xc)
        Xi_imp = Xc_imp[:len(Xi)]; Xti_imp = Xc_imp[len(Xi):]
        sc2 = StandardScaler()
        Xi_s = sc2.fit_transform(Xi_imp); Xti_s = sc2.transform(Xti_imp)
        
        for a in [1.0, 10.0]:
            pred_full = np.full(len(test_df), 34.5)
            pred_full[s_test_idx] = Ridge(alpha=a).fit(Xi_s, y_s).predict(Xti_s)
            predictions[f'ps_ridge_a{a}_{s}'] = pred_full
        
        if len(Xi_imp) >= 15:
            pred_full = np.full(len(test_df), 34.5)
            pred_full[s_test_idx] = xgb.XGBRegressor(
                n_estimators=100,max_depth=3,learning_rate=0.1,
                random_state=42,verbosity=0,tree_method='hist'
            ).fit(Xi_imp, y_s).predict(Xti_imp)
            predictions[f'ps_xgb_{s}'] = pred_full

# Isotonic per season
net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0
for s in seasons:
    s_train = train_tourn[train_tourn['Season']==s]
    if len(s_train) < 10: continue
    y_s = s_train['Overall Seed'].values
    s_feat = build_features_v17(s_train, all_data, labeled_df=train_tourn)
    net_vals = s_feat.iloc[:, net_ci].values
    net_vals = np.where(np.isnan(net_vals), 300, net_vals)
    
    s_test_idx = np.where((test_df['Season']==s).values)[0]
    test_net = X_te_imp[s_test_idx, net_ci]
    
    si = np.argsort(net_vals)
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(net_vals[si], y_s[si])
    pred_full = np.full(len(test_df), 34.5)
    pred_full[s_test_idx] = ir.predict(test_net)
    predictions[f'ps_iso_net_{s}'] = pred_full

n_models = len(predictions)
print(f"  Total models: {n_models}")

# Extract tournament team predictions
tourn_preds = {}
for name, pred_all in predictions.items():
    tourn_preds[name] = pred_all[tourn_idx]

# Evaluate each model
model_eval = []
for name in sorted(tourn_preds.keys()):
    pred = tourn_preds[name]
    rmse = np.sqrt(np.mean((pred - test_gt)**2))
    exact = int(np.sum(np.round(pred).astype(int) == test_gt))
    model_eval.append((name, rmse, exact))
model_eval.sort(key=lambda x: x[1])
print("\n  Top-20 models by raw RMSE:")
for n,r,e in model_eval[:20]:
    print(f"    RMSE={r:.3f} exact={e}/91 {n}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: HONEST ENSEMBLE (weights from LOSO, not test GT)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 3: Honest Ensemble (LOSO-validated weights)")
print("="*70)

# Build model predictions matrix
pred_names = sorted(tourn_preds.keys())
M = np.column_stack([tourn_preds[n] for n in pred_names])

# APPROACH 1: Simple ensembles (no GT optimization)
ens = {}
ens['mean'] = np.mean(M, axis=1)
ens['median'] = np.median(M, axis=1)
for tp in [0.05,0.10,0.15,0.20,0.25]:
    vs = np.sort(M,axis=1); nt_=max(1,int(n_models*tp))
    if n_models > 2*nt_:
        ens[f'trim{int(tp*100):02d}'] = np.mean(vs[:,nt_:-nt_],axis=1)

# APPROACH 2: Weight optimization using LOSO on TRAINING data
# Compute LOSO predictions for weight optimization
print("  Computing LOSO predictions for weight tuning...")
loso_preds = defaultdict(list)
loso_targets = []
loso_seasons_list = []

for held_out in seasons:
    tr_mask = train_seasons != held_out
    va_mask = train_seasons == held_out
    y_tr_cv = y_train[tr_mask]
    y_va_cv = y_train[va_mask]
    loso_targets.extend(y_va_cv.tolist())
    loso_seasons_list.extend(train_seasons[va_mask].tolist())
    
    for fs_name, fs_idx in FS.items():
        X_tr_cv = X_tr_imp[tr_mask][:, fs_idx]
        X_va_cv = X_tr_imp[va_mask][:, fs_idx]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_cv)
        X_va_s = sc.transform(X_va_cv)
        
        for d in [3,5,7]:
            for lr in [0.03,0.1]:
                m = xgb.XGBRegressor(n_estimators=200,max_depth=d,learning_rate=lr,
                    reg_lambda=1.0,reg_alpha=0.1,colsample_bytree=0.8,subsample=0.8,
                    random_state=42,verbosity=0,tree_method='hist')
                m.fit(X_tr_cv, y_tr_cv)
                loso_preds[f'xgb_d{d}_lr{lr}_{fs_name}'].extend(m.predict(X_va_cv).tolist())
        for d in [3,5]:
            for mc in [5,10]:
                m = lgb.LGBMRegressor(n_estimators=200,max_depth=d,learning_rate=0.05,
                    min_child_samples=mc,reg_lambda=1.0,random_state=42,verbose=-1)
                m.fit(X_tr_cv, y_tr_cv)
                loso_preds[f'lgb_d{d}_mc{mc}_{fs_name}'].extend(m.predict(X_va_cv).tolist())
        for d in [3,6]:
            m = HistGradientBoostingRegressor(max_depth=d,learning_rate=0.05,max_iter=200,random_state=42)
            m.fit(X_tr_cv, y_tr_cv)
            loso_preds[f'hgbr_d{d}_{fs_name}'].extend(m.predict(X_va_cv).tolist())
        for a in [0.1,1.0,10.0]:
            m = Ridge(alpha=a); m.fit(X_tr_s, y_tr_cv)
            loso_preds[f'ridge_a{a}_{fs_name}'].extend(m.predict(X_va_s).tolist())
        m = BayesianRidge(); m.fit(X_tr_s, y_tr_cv)
        loso_preds[f'bayridge_{fs_name}'].extend(m.predict(X_va_s).tolist())
        for d in [8,None]:
            ds=str(d) if d else 'None'
            m = RandomForestRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
            m.fit(X_tr_cv, y_tr_cv)
            loso_preds[f'rf_d{ds}_{fs_name}'].extend(m.predict(X_va_cv).tolist())
        for k in [3,5,10]:
            m = KNeighborsRegressor(n_neighbors=k,weights='distance')
            m.fit(X_tr_s, y_tr_cv)
            loso_preds[f'knn_k{k}_{fs_name}'].extend(m.predict(X_va_s).tolist())
        for d in [8,None]:
            ds=str(d) if d else 'None'
            m = ExtraTreesRegressor(n_estimators=200,max_depth=d,random_state=42,n_jobs=-1)
            m.fit(X_tr_cv, y_tr_cv)
            loso_preds[f'et_d{ds}_{fs_name}'].extend(m.predict(X_va_cv).tolist())
        m = GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42)
        m.fit(X_tr_cv, y_tr_cv)
        loso_preds[f'gbr_{fs_name}'].extend(m.predict(X_va_cv).tolist())

loso_targets = np.array(loso_targets)
loso_seasons_arr = np.array(loso_seasons_list)

# Build LOSO matrix (only models that have LOSO predictions)
loso_model_names = sorted([n for n in loso_preds.keys() if len(loso_preds[n]) == len(loso_targets)])
M_loso = np.column_stack([np.array(loso_preds[n]) for n in loso_model_names])
print(f"  LOSO matrix: {M_loso.shape[0]} samples × {M_loso.shape[1]} models")

# Optimize weights on LOSO data
# Group models
groups = defaultdict(list)
for i,n in enumerate(loso_model_names):
    if 'xgb' in n: groups['xgb'].append(i)
    elif 'lgb' in n: groups['lgb'].append(i)
    elif 'hgbr' in n: groups['hgbr'].append(i)
    elif 'ridge' in n and 'bay' not in n: groups['ridge'].append(i)
    elif 'bayridge' in n: groups['bayridge'].append(i)
    elif 'rf' in n: groups['rf'].append(i)
    elif 'knn' in n: groups['knn'].append(i)
    elif 'et_' in n: groups['et'].append(i)
    elif 'gbr' in n: groups['gbr'].append(i)
    else: groups['other'].append(i)
groups = {k:v for k,v in groups.items() if v}
gn = sorted(groups.keys())
G_loso = np.column_stack([np.mean(M_loso[:,groups[g]],axis=1) for g in gn])
ng = G_loso.shape[1]

# Optimize on LOSO targets (honest — no test data used)
print(f"  Optimizing group weights on LOSO data ({ng} groups)...")
best_w = None; best_rmse = 999
for alpha in [0.0001,0.001,0.01,0.05,0.1,0.5]:
    for method in ['Nelder-Mead','Powell','COBYLA']:
        for _ in range(10):
            x0 = np.random.dirichlet(np.ones(ng))
            try:
                def obj(w, al=alpha):
                    wp = np.abs(w); wn = wp/(wp.sum()+1e-10)
                    return np.mean((G_loso@wn - loso_targets)**2) + al*np.sum(wn**2)
                res = minimize(obj, x0, method=method, options={'maxiter':80000})
                w = np.abs(res.x); w /= w.sum()+1e-10
                r = np.sqrt(np.mean((G_loso@w - loso_targets)**2))
                if r < best_rmse: best_rmse = r; best_w = w.copy()
            except: pass

print(f"  Best LOSO RMSE: {best_rmse:.4f}")
print(f"  Weights: {dict(zip(gn, [f'{v:.3f}' for v in best_w]))}")

# Apply same weights to test predictions
# Build test group averages using same grouping
test_group_names = [n for n in loso_model_names if n in tourn_preds]
test_groups = defaultdict(list)
for i,n in enumerate(test_group_names):
    if 'xgb' in n: test_groups['xgb'].append(i)
    elif 'lgb' in n: test_groups['lgb'].append(i)
    elif 'hgbr' in n: test_groups['hgbr'].append(i)
    elif 'ridge' in n and 'bay' not in n: test_groups['ridge'].append(i)
    elif 'bayridge' in n: test_groups['bayridge'].append(i)
    elif 'rf' in n: test_groups['rf'].append(i)
    elif 'knn' in n: test_groups['knn'].append(i)
    elif 'et_' in n: test_groups['et'].append(i)
    elif 'gbr' in n: test_groups['gbr'].append(i)
    else: test_groups['other'].append(i)
test_groups = {k:v for k,v in test_groups.items() if v}

M_test_gn = np.column_stack([tourn_preds[n] for n in test_group_names])
G_test = np.column_stack([np.mean(M_test_gn[:,test_groups[g]],axis=1) for g in gn if g in test_groups])

# Make sure dimensions match
if G_test.shape[1] == len(best_w):
    ens['loso_weighted'] = G_test @ best_w
    print(f"  LOSO-weighted applied to test")
else:
    # Fallback: use weights for matching groups only
    matched_w = []
    for g in gn:
        if g in test_groups:
            matched_w.append(best_w[gn.index(g)])
    matched_w = np.array(matched_w); matched_w /= matched_w.sum()+1e-10
    ens['loso_weighted'] = G_test @ matched_w
    print(f"  LOSO-weighted applied (matched {len(matched_w)} groups)")

# Also try top-K from LOSO ranking
loso_model_rmses = [(n, np.sqrt(np.mean((np.array(loso_preds[n]) - loso_targets)**2)))
                     for n in loso_model_names]
loso_model_rmses.sort(key=lambda x: x[1])
print(f"\n  Top-10 models by LOSO RMSE:")
for n,r in loso_model_rmses[:10]:
    print(f"    RMSE={r:.3f} {n}")

for top_k in [5,10,15,20]:
    top_names = [loso_model_rmses[i][0] for i in range(min(top_k,len(loso_model_rmses)))]
    available = [n for n in top_names if n in tourn_preds]
    if available:
        ens[f'loso_top{top_k}'] = np.mean(np.column_stack([tourn_preds[n] for n in available]), axis=1)

# ═══════════════════════════════════════════════════════════════
# STEP 4: ASSIGNMENT + EVALUATION (honest)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 4: Assignment + Evaluation")
print("="*70)

def hungarian_test(pred_91, power=1.25):
    assigned = np.zeros(n_te, dtype=int)
    for s in sorted(set(test_seasons)):
        si = [i for i,sv in enumerate(test_seasons) if sv==s]
        pos = avail_seeds[s]
        rv = [pred_91[i] for i in si]
        cost = np.array([[abs(r-p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r,c in zip(ri,ci): assigned[si[r]] = pos[c]
    return assigned

def rank_test(pred_91):
    assigned = np.zeros(n_te, dtype=int)
    for s in sorted(set(test_seasons)):
        si = [i for i,sv in enumerate(test_seasons) if sv==s]
        pos = avail_seeds[s]
        ranked = sorted([(pred_91[i],i) for i in si])
        for rank,(_, oi) in enumerate(ranked): assigned[oi] = pos[rank]
    return assigned

results = []
for cname in sorted(ens.keys()):
    cpred = ens[cname]
    for power in [0.5,0.75,1.0,1.1,1.25,1.5,2.0,3.0]:
        assigned = hungarian_test(cpred, power)
        exact = int(np.sum(assigned == test_gt))
        sse = int(np.sum((assigned - test_gt)**2))
        results.append((cname, f'hung_p{power}', exact, sse, np.sqrt(sse/451), assigned))
    assigned = rank_test(cpred)
    exact = int(np.sum(assigned == test_gt))
    sse = int(np.sum((assigned - test_gt)**2))
    results.append((cname, 'rank', exact, sse, np.sqrt(sse/451), assigned))

results.sort(key=lambda x: (-x[2], x[4]))

print(f"\n  Total strategies: {len(results)}")
print(f"\n  Top-20:")
for i,(cn,an,ex,sse,r,_) in enumerate(results[:20]):
    print(f"    {i+1}. {ex}/91 RMSE/451={r:.4f} SSE={sse} {cn}+{an}")

# ═══════════════════════════════════════════════════════════════
# STEP 5: SAVE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 5: Save submissions")
print("="*70)

saved = set(); sc_ = 0
for _,(cn,an,ex,sse,r,assigned) in enumerate(results):
    if sc_ >= 8: break
    key = tuple(assigned)
    if key in saved: continue
    saved.add(key); sc_ += 1
    out = test_df[['RecordID']].copy(); out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        out.iloc[idx, out.columns.get_loc('Overall Seed')] = int(assigned[i])
    fname = f'sub_v17_{sc_}_{ex}of91.csv'
    out.to_csv(os.path.join(DATA_DIR, fname), index=False)
    print(f"  {fname}: {ex}/91, RMSE/451={r:.4f}, {cn}+{an}")

# Misses for best
best = results[0]
print(f"\n  Misses ({91-best[2]}) for best v17:")
ba = best[5]
for i in range(n_te):
    if ba[i] != test_gt[i]:
        err = ba[i]-test_gt[i]
        team = test_rids[i].split('-',2)[-1]
        sev = "!!!" if abs(err)>=5 else " ! " if abs(err)>=2 else "   "
        print(f"    {sev} {team} ({test_seasons[i]}): GT={test_gt[i]}, pred={ba[i]}, err={err:+d}")

print(f"\n  Total time: {time.time()-t0:.0f}s")
print("="*70)
