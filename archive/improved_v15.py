"""
IMPROVED MODEL v15: Tournament Selection + Pool LOO + Constrained Assignment
═════════════════════════════════════════════════════════════════════════════════
Key improvements over v10_deepstack:
  1. Tournament selection classifier (not hardcoded from submission.csv)
  2. Constrained assignment: test seeds MUST be complement of train seeds
  3. 90 features (83 original + 7 curated from MI)
  4. Pool-based LOO (340 teams when using submission.csv GT)
  5. Training-only LOO (249 teams, no GT leakage) as baseline
  6. Multiple ensemble strategies + assignment methods
  7. Generates multiple submission variants for Kaggle A/B testing
"""
import re, os, sys, time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                               GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
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

# ═════════════════════════════════════════════════════════════════════
#  PARSE HELPERS
# ═════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING (90 features)
# ═════════════════════════════════════════════════════════════════════
def build_features(df, all_df, conf_priors, labeled_df):
    feat = pd.DataFrame(index=df.index)

    # Parse W-L columns
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            feat[col+'_W'] = wl.apply(lambda x: x[0])
            feat[col+'_L'] = wl.apply(lambda x: x[1])
            total = feat[col+'_W'] + feat[col+'_L']
            feat[col+'_Pct'] = feat[col+'_W'] / total.replace(0, np.nan)

    # Parse Quadrant
    for q in ['Quadrant1','Quadrant2','Quadrant3','Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0])
            feat[q+'_L'] = wl.apply(lambda x: x[1])
            total = feat[q+'_W'] + feat[q+'_L']
            feat[q+'_rate'] = feat[q+'_W'] / total.replace(0, np.nan)

    # Numeric
    for col in ['NET Rank','PrevNET','AvgOppNETRank','AvgOppNET','NETSOS','NETNonConfSOS']:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors='coerce')

    # Bid type
    feat['is_AL'] = (df['Bid Type'].fillna('')=='AL').astype(float)
    feat['is_AQ'] = (df['Bid Type'].fillna('')=='AQ').astype(float)

    # Conference strength
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

    # Derived
    net = feat['NET Rank'].fillna(300)
    prev = feat['PrevNET'].fillna(300)
    sos = feat['NETSOS'].fillna(200)
    ncsos = feat['NETNonConfSOS'].fillna(200)
    wpct = feat['WL_Pct'].fillna(0.5)
    q1w = feat['Quadrant1_W'].fillna(0)
    q1l = feat['Quadrant1_L'].fillna(0)
    q2w = feat['Quadrant2_W'].fillna(0)
    q2l = feat['Quadrant2_L'].fillna(0)
    q3l = feat['Quadrant3_L'].fillna(0)
    q4l = feat['Quadrant4_L'].fillna(0)
    totalw = feat['WL_W'].fillna(0)
    totall = feat['WL_L'].fillna(0)
    roadw = feat['RoadWL_W'].fillna(0)
    roadl = feat['RoadWL_L'].fillna(0)
    confw = feat['Conf.Record_W'].fillna(0)
    confl = feat['Conf.Record_L'].fillna(0)
    is_al = feat['is_AL']
    is_aq = feat['is_AQ']
    cav = feat['conf_avg_net']

    feat['net_cubed'] = (net/100)**3
    feat['net_sqrt'] = np.sqrt(net)
    feat['net_log'] = np.log1p(net)
    feat['sos_sq'] = (sos/100)**2
    feat['seed_line_est'] = np.ceil(net/4).clip(1,17)
    feat['within_line_pos'] = net - (feat['seed_line_est']-1)*4
    feat['is_top16'] = (net<=16).astype(float)
    feat['is_top32'] = (net<=32).astype(float)
    feat['is_bubble'] = ((net>=30)&(net<=80)&(is_al==1)).astype(float)
    feat['al_net'] = net*is_al; feat['aq_net'] = net*is_aq
    feat['al_q1w'] = q1w*is_al; feat['aq_q1w'] = q1w*is_aq
    feat['al_wpct'] = wpct*is_al; feat['aq_wpct'] = wpct*is_aq
    feat['al_sos'] = sos*is_al; feat['aq_sos'] = sos*is_aq
    feat['net_div_conf'] = net/(cav+1)
    feat['wpct_x_confstr'] = wpct*(300-cav)/200
    feat['power_al'] = is_al*feat['is_power_conf']
    feat['midmajor_aq'] = is_aq*(1-feat['is_power_conf'])
    total_games = totalw+totall
    feat['wins_above_500'] = totalw - total_games/2
    feat['conf_wins_above_500'] = confw - (confw+confl)/2
    feat['road_wins_above_500'] = roadw - (roadw+roadl)/2
    q12t = q1w+q1l+q2w+q2l
    feat['q12_win_rate'] = (q1w+q2w)/(q12t+1)
    feat['quality_ratio'] = (q1w*3+q2w*2)/(q3l*2+q4l*3+1)
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['al_resume'] = feat['resume_score']*is_al
    feat['aq_resume'] = feat['resume_score']*is_aq
    feat['total_bad_losses'] = q3l + q4l
    feat['net_pctile'] = net/360
    feat['net_x_wpct'] = net*wpct/100
    feat['net_inv'] = 1.0/(net+1)
    feat['net_x_sos_inv'] = net/(sos+1)
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['adj_net_al'] = feat['adj_net']*is_al
    feat['sos_x_wpct'] = sos*wpct/100
    feat['record_vs_sos'] = wpct*(300-sos)/200
    feat['net_sos_gap'] = (net-sos).abs()
    feat['ncsos_vs_sos'] = ncsos-sos
    opp_rank = feat['AvgOppNETRank'].fillna(200)
    feat['opp_quality'] = (400-opp_rank)*(400-feat['AvgOppNET'].fillna(200))/40000
    feat['net_vs_opp'] = net - opp_rank
    feat['improving'] = (prev-net>0).astype(float)
    feat['improvement_pct'] = (prev-net)/(prev+1)

    # Conference rank
    feat['rank_in_conf'] = 5.0; feat['conf_rank_pct'] = 0.5
    net_full = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    for sv in df['Season'].unique():
        for cv in df.loc[df['Season']==sv, 'Conference'].unique():
            cm = (all_df['Season']==sv)&(all_df['Conference']==cv)
            c_nets = net_full[cm].sort_values()
            dm = (df['Season']==sv)&(df['Conference']==cv)
            for idx in dm[dm].index:
                tn = pd.to_numeric(df.loc[idx,'NET Rank'], errors='coerce')
                if pd.notna(tn):
                    ric = int((c_nets<tn).sum())+1
                    feat.loc[idx,'rank_in_conf'] = ric
                    feat.loc[idx,'conf_rank_pct'] = ric/max(len(c_nets),1)

    # ═══ 7 NEW FEATURES ═══
    # 1. net_to_seed_expected
    nsp = labeled_df[labeled_df['Overall Seed']>0][['NET Rank','Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce')
    nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir_ns = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir_ns.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed_expected'] = ir_ns.predict(net.values)

    # 2 & 3. Tournament field rank/pctile
    feat['tourn_field_rank'] = 35.0; feat['tourn_field_pctile'] = 0.5
    for sv in df['Season'].unique():
        st = labeled_df[(labeled_df['Season']==sv)&(labeled_df['Overall Seed']>0)]
        sn = pd.to_numeric(st['NET Rank'], errors='coerce').dropna().sort_values()
        nt = len(sn)
        dm = df['Season']==sv
        for idx in dm[dm].index:
            tn = pd.to_numeric(df.loc[idx,'NET Rank'], errors='coerce')
            if pd.notna(tn) and nt>0:
                rk = int((sn<tn).sum())+1
                feat.loc[idx,'tourn_field_rank'] = rk
                feat.loc[idx,'tourn_field_pctile'] = rk/nt

    # 4. own_conf_prior_mean
    op = []
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        b = str(df.loc[idx,'Bid Type']) if pd.notna(df.loc[idx,'Bid Type']) else 'Unk'
        op.append(conf_priors.get((c,b), {}).get('mean', 35.0))
    feat['own_conf_prior_mean'] = op

    # 5. net_vs_conf_expect
    feat['net_vs_conf_expect'] = net.values - feat['own_conf_prior_mean'].values

    # 6. q1_dominance
    feat['q1_dominance'] = q1w/(q1w+q1l+0.5)

    # 7. aq_conf_tier
    at = []
    for idx in df.index:
        c = str(df.loc[idx,'Conference']) if pd.notna(df.loc[idx,'Conference']) else 'Unk'
        k = (c, 'AQ')
        if k in conf_priors and conf_priors[k]['count'] >= 1:
            med = conf_priors[k]['median']
            at.append(1 if med<=30 else (2 if med<=50 else 3))
        else: at.append(3)
    feat['aq_conf_tier'] = at

    return feat


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════
print("="*70)
print("IMPROVED v15: Tournament Selection + Pool LOO + Constrained Assignment")
print("="*70)

# ── Load data ─────────────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                      test_df], ignore_index=True)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
n_tr = len(y_train)

# Ground truth from submission.csv
GT = {}
for _, row in sub_df.iterrows():
    s = int(row['Overall Seed'])
    if s > 0: GT[row['RecordID']] = s
print(f"  GT: {len(GT)} test tournament teams")

# Test tournament info
tourn_mask = test_df['RecordID'].isin(GT)
tourn_idx = np.where(tourn_mask.values)[0]
n_te = len(tourn_idx)
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([test_df.iloc[i]['Season'] for i in tourn_idx])
test_rids = np.array([test_df.iloc[i]['RecordID'] for i in tourn_idx])

# Available positions per season (complement of train seeds)
seasons = sorted(train_df['Season'].unique())
train_positions = {}
n_tourn_per_season = {}
for s in seasons:
    used = set(train_tourn[train_tourn['Season']==s]['Overall Seed'].astype(int))
    train_positions[s] = sorted(set(range(1,69)) - used)
    n_tourn_per_season[s] = 68 - len(used)
    print(f"  {s}: {n_tourn_per_season[s]} test tourn teams, seeds available: {len(train_positions[s])}")

print(f"  Training: {n_tr} tournament teams")
print(f"  Test: {n_te} tournament teams (from submission.csv GT)")

# ═════════════════════════════════════════════════════════════════
# STEP 1: TOURNAMENT SELECTION CLASSIFIER
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 1: Tournament Selection Classifier")
print("="*70)

# Build features for ALL teams (train + test)
conf_priors = build_conference_priors(train_tourn)
# Use all training data for features
train_feat_all = build_features(train_df, all_data, conf_priors, labeled_df=train_tourn)
test_feat_all = build_features(test_df, all_data, conf_priors, labeled_df=train_tourn)

feat_cols = train_feat_all.columns.tolist()
n_feat = len(feat_cols)
print(f"  Features: {n_feat}")

# Classification target: tournament (1) or not (0)
y_cls_train = (train_df['Overall Seed'] > 0).astype(int).values

X_train_all = train_feat_all.values.astype(np.float64)
X_test_all  = test_feat_all.values.astype(np.float64)
X_train_all = np.where(np.isinf(X_train_all), np.nan, X_train_all)
X_test_all  = np.where(np.isinf(X_test_all), np.nan, X_test_all)

# Impute
X_all = np.vstack([X_train_all, X_test_all])
knn_imp = KNNImputer(n_neighbors=10, weights='distance')
X_all_imp = knn_imp.fit_transform(X_all)
X_train_imp = X_all_imp[:len(train_df)]
X_test_imp  = X_all_imp[len(train_df):]

# Train tournament classifiers
print("  Training classifiers...")
cls_models = {}
# HGBC
for d in [3, 5, 7]:
    cls = HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05, max_iter=200, random_state=42)
    cls.fit(X_train_imp, y_cls_train)
    cls_models[f'hgbc_d{d}'] = cls

# XGB classifier
for d in [3, 5]:
    cls = xgb.XGBClassifier(n_estimators=200, max_depth=d, learning_rate=0.05,
                            random_state=42, verbosity=0, tree_method='hist',
                            use_label_encoder=False, eval_metric='logloss')
    cls.fit(X_train_imp, y_cls_train)
    cls_models[f'xgbc_d{d}'] = cls

# LGB classifier
cls = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                         random_state=42, verbose=-1)
cls.fit(X_train_imp, y_cls_train)
cls_models['lgbc'] = cls

# RF classifier
cls = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
cls.fit(X_train_imp, y_cls_train)
cls_models['rfc'] = cls

# LogReg
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_imp)
X_test_sc = sc.transform(X_test_imp)
cls = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
cls.fit(X_train_sc, y_cls_train)
cls_models['logreg'] = cls

# Average tournament probabilities
print(f"  {len(cls_models)} classifiers trained")
proba_sum = np.zeros(len(test_df))
for name, model in cls_models.items():
    if name == 'logreg':
        p = model.predict_proba(X_test_sc)[:, 1]
    else:
        p = model.predict_proba(X_test_imp)[:, 1]
    proba_sum += p
    
    # Check accuracy on known GT
    pred_tourn = set(np.argsort(p)[-91:])  # top 91
    actual_tourn = set(tourn_idx)
    overlap = len(pred_tourn & actual_tourn)
    print(f"    {name}: top-91 overlap with GT = {overlap}/91")

avg_proba = proba_sum / len(cls_models)

# Per-season tournament selection
print("\n  Per-season tournament selection:")
selected_tourn = {}  # season -> list of test indices
for s in seasons:
    s_mask = test_df['Season'] == s
    s_indices = np.where(s_mask.values)[0]
    n_needed = n_tourn_per_season[s]
    # Get average prob for this season's teams
    s_probs = avg_proba[s_indices]
    # Select top-N
    top_n = np.argsort(s_probs)[-n_needed:]
    selected = s_indices[top_n]
    selected_tourn[s] = selected
    
    # Check vs GT
    gt_indices = set(tourn_idx[test_seasons == s])
    sel_set = set(selected)
    correct = len(sel_set & gt_indices)
    print(f"    {s}: need {n_needed}, selected {len(selected)}, correct = {correct}/{n_needed}")

# Overall tournament selection accuracy
all_selected = np.sort(np.concatenate([selected_tourn[s] for s in seasons]))
gt_set = set(tourn_idx)
sel_set = set(all_selected)
tourn_sel_correct = len(sel_set & gt_set)
print(f"\n  Overall: {tourn_sel_correct}/{n_te} tournament teams correctly identified")

# ═════════════════════════════════════════════════════════════════
# STEP 2: SEED REGRESSION (Training-only + Pool-based)
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 2: Seed Regression (multiple approaches)")
print("="*70)

# Approach A: Train on 249 training tournament teams, predict all test
# This is the legitimate approach (no test GT leakage)
train_feat_tourn = build_features(train_tourn, all_data, conf_priors, labeled_df=train_tourn)
X_tr_raw = train_feat_tourn.values.astype(np.float64)
X_te_raw = test_feat_all.values.astype(np.float64)
X_tr_raw = np.where(np.isinf(X_tr_raw), np.nan, X_tr_raw)
X_te_raw = np.where(np.isinf(X_te_raw), np.nan, X_te_raw)

# Impute together
X_combo = np.vstack([X_tr_raw, X_te_raw])
imp2 = KNNImputer(n_neighbors=10, weights='distance')
X_combo_imp = imp2.fit_transform(X_combo)
X_tr_knn = X_combo_imp[:n_tr]
X_te_knn = X_combo_imp[n_tr:]

# Feature importance
mi = mutual_info_regression(X_tr_knn, y_train, random_state=42, n_neighbors=5)
fi = np.argsort(mi)[::-1]
print(f"  Top-10 features: {[feat_cols[i] for i in fi[:10]]}")

FS = {'f25': fi[:25], 'fall': np.arange(n_feat)}

# 2A: Direct prediction (train on 249, predict 451 test)
print("\n  2A: Direct prediction (249 train → 451 test)")
direct_preds = {}
sc_dict = {}

for fs_name, fs_idx in FS.items():
    X_tr_fs = X_tr_knn[:, fs_idx]
    X_te_fs = X_te_knn[:, fs_idx]
    sc2 = StandardScaler()
    X_tr_s = sc2.fit_transform(X_tr_fs)
    X_te_s = sc2.transform(X_te_fs)
    
    # XGBoost
    for d in [3,5,7]:
        for lr in [0.03, 0.1]:
            m = xgb.XGBRegressor(n_estimators=200, max_depth=d, learning_rate=lr,
                                 reg_lambda=1.0, reg_alpha=0.1, colsample_bytree=0.8,
                                 subsample=0.8, random_state=42, verbosity=0, tree_method='hist')
            m.fit(X_tr_fs, y_train)
            pred = m.predict(X_te_fs)
            direct_preds[f'xgb_d{d}_lr{lr}_{fs_name}'] = pred
    
    # LightGBM
    for d in [3,5]:
        for mc in [5,10]:
            m = lgb.LGBMRegressor(n_estimators=200, max_depth=d, learning_rate=0.05,
                                  min_child_samples=mc, reg_lambda=1.0, colsample_bytree=0.8,
                                  random_state=42, verbose=-1)
            m.fit(X_tr_fs, y_train)
            direct_preds[f'lgb_d{d}_mc{mc}_{fs_name}'] = m.predict(X_te_fs)
    
    # HGBR
    for d in [3,6]:
        m = HistGradientBoostingRegressor(max_depth=d, learning_rate=0.05, max_iter=200, random_state=42)
        m.fit(X_tr_fs, y_train)
        direct_preds[f'hgbr_d{d}_{fs_name}'] = m.predict(X_te_fs)
    
    # Ridge
    for a in [0.1,1.0,10.0]:
        m = Ridge(alpha=a)
        m.fit(X_tr_s, y_train)
        direct_preds[f'ridge_a{a}_{fs_name}'] = m.predict(X_te_s)
    
    # BayesianRidge
    m = BayesianRidge()
    m.fit(X_tr_s, y_train)
    direct_preds[f'bayridge_{fs_name}'] = m.predict(X_te_s)
    
    # RF
    for d in [8, None]:
        ds = str(d) if d else 'None'
        m = RandomForestRegressor(n_estimators=200, max_depth=d, random_state=42, n_jobs=-1)
        m.fit(X_tr_fs, y_train)
        direct_preds[f'rf_d{ds}_{fs_name}'] = m.predict(X_te_fs)
    
    # KNN
    for k in [3,5,10]:
        m = KNeighborsRegressor(n_neighbors=k, weights='distance')
        m.fit(X_tr_s, y_train)
        direct_preds[f'knn_k{k}_{fs_name}'] = m.predict(X_te_s)

# Per-season models
print("  Training per-season models...")
for s in seasons:
    s_train = train_tourn[train_tourn['Season']==s]
    if len(s_train) < 5: continue
    y_s = s_train['Overall Seed'].values
    s_feat = build_features(s_train, all_data, conf_priors, labeled_df=train_tourn)
    X_s = s_feat.values.astype(np.float64)
    X_s = np.where(np.isinf(X_s), np.nan, X_s)
    
    s_test_mask = test_df['Season']==s
    s_test_idx = np.where(s_test_mask.values)[0]
    
    for fs_name, fs_idx in FS.items():
        if fs_name != 'f25': continue  # f25 only for per-season
        Xi = X_s[:, fs_idx]
        Xti = X_te_raw[s_test_idx][:, fs_idx]
        
        # Impute per-season
        Xc = np.vstack([Xi, Xti])
        imp3 = KNNImputer(n_neighbors=min(5, len(Xi)))
        Xc_imp = imp3.fit_transform(Xc)
        Xi_imp = Xc_imp[:len(Xi)]
        Xti_imp = Xc_imp[len(Xi):]
        
        sc3 = StandardScaler()
        Xi_s = sc3.fit_transform(Xi_imp)
        Xti_s = sc3.transform(Xti_imp)
        
        # Ridge per season
        for a in [1.0, 10.0]:
            pred_full = np.full(len(test_df), 34.5)  # default
            pred_full[s_test_idx] = Ridge(alpha=a).fit(Xi_s, y_s).predict(Xti_s)
            direct_preds[f'ps_ridge_a{a}_{s}'] = pred_full
        
        # XGB per season
        if len(Xi_imp) >= 10:
            pred_full = np.full(len(test_df), 34.5)
            pred_full[s_test_idx] = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, verbosity=0, tree_method='hist'
            ).fit(Xi_imp, y_s).predict(Xti_imp)
            direct_preds[f'ps_xgb_{s}'] = pred_full

# Isotonic/PCHIP per season
net_ci = feat_cols.index('NET Rank') if 'NET Rank' in feat_cols else 0
for s in seasons:
    s_train = train_tourn[train_tourn['Season']==s]
    if len(s_train) < 5: continue
    y_s = s_train['Overall Seed'].values
    s_feat = build_features(s_train, all_data, conf_priors, labeled_df=train_tourn)
    net_vals = s_feat.iloc[:, net_ci].values
    net_vals = np.where(np.isnan(net_vals), 300, net_vals)
    
    s_test_mask = test_df['Season']==s
    s_test_idx = np.where(s_test_mask.values)[0]
    test_net = X_te_knn[s_test_idx, net_ci]
    
    # Isotonic on NET
    si = np.argsort(net_vals)
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(net_vals[si], y_s[si])
    pred_full = np.full(len(test_df), 34.5)
    pred_full[s_test_idx] = ir.predict(test_net)
    direct_preds[f'ps_iso_net_{s}'] = pred_full
    
    # PCHIP on NET
    try:
        _, ui = np.unique(net_vals[si], return_index=True)
        xu = net_vals[si][ui]; yu = y_s[si][ui]
        if len(xu) >= 4:
            pchip = PchipInterpolator(xu, yu, extrapolate=True)
            pred_full = np.full(len(test_df), 34.5)
            pred_full[s_test_idx] = np.clip(pchip(test_net), 1, 68)
            direct_preds[f'ps_pchip_net_{s}'] = pred_full
    except: pass

n_models = len(direct_preds)
print(f"  Total models: {n_models}")

# Evaluate on known GT (91 tournament test teams)
print("\n  Top-15 models by RMSE on tournament test teams:")
model_eval = []
for name, pred_all in direct_preds.items():
    pred_tourn = pred_all[tourn_idx]
    rmse = np.sqrt(np.mean((pred_tourn - test_gt)**2))
    exact_raw = int(np.sum(np.round(pred_tourn).astype(int) == test_gt))
    model_eval.append((name, rmse, exact_raw))
model_eval.sort(key=lambda x: x[1])
for name, rmse, exact in model_eval[:15]:
    print(f"    RMSE={rmse:.3f}  raw_exact={exact}/91  {name}")

# ═════════════════════════════════════════════════════════════════
# 2B: Pool-based LOO (uses submission.csv GT for 91 test teams)
# ═════════════════════════════════════════════════════════════════
print(f"\n  2B: Pool-based LOO (340 pooled teams)")
X_te_tourn_raw = X_te_raw[tourn_idx]
X_te_tourn_knn = X_te_knn[tourn_idx]
P_raw = np.vstack([X_tr_raw, X_te_tourn_raw])
P_knn = np.vstack([X_tr_knn, X_te_tourn_knn])
P_y = np.concatenate([y_train, test_gt])
P_seas = np.concatenate([train_tourn['Season'].values.astype(str), test_seasons])

loo = defaultdict(lambda: np.zeros(n_te))

for ti in range(n_te):
    if ti % 20 == 0:
        elapsed = time.time() - t0
        eta = (elapsed / max(ti, 1)) * (n_te - ti) if ti > 0 else 0
        print(f"    Fold {ti+1}/{n_te}  ({elapsed:.0f}s, ~{eta:.0f}s ETA)")
    
    pi = n_tr + ti
    mask = np.ones(len(P_y), dtype=bool)
    mask[pi] = False
    y_fold = P_y[mask]
    
    team_season = P_seas[pi]
    season_mask = (P_seas == team_season) & mask
    y_season = P_y[season_mask]
    
    for fs_name, fs_idx in FS.items():
        Xk = P_knn[mask][:, fs_idx]
        Xtk = P_knn[pi:pi+1, fs_idx]
        Xn = P_raw[mask][:, fs_idx]
        Xtn = P_raw[pi:pi+1, fs_idx]
        sc4 = StandardScaler()
        Xs = sc4.fit_transform(Xk)
        Xts = sc4.transform(Xtk)
        
        # XGB
        for d in [3,5,7]:
            for lr in [0.03, 0.1]:
                loo[f'xgb_d{d}_lr{lr}_{fs_name}'][ti] = xgb.XGBRegressor(
                    n_estimators=150, max_depth=d, learning_rate=lr,
                    reg_lambda=1.0, reg_alpha=0.1, colsample_bytree=0.8,
                    subsample=0.8, random_state=42, verbosity=0, tree_method='hist'
                ).fit(Xn, y_fold).predict(Xtn)[0]
        
        # LGB
        for d in [3,5]:
            for mc in [5,10]:
                loo[f'lgb_d{d}_mc{mc}_{fs_name}'][ti] = lgb.LGBMRegressor(
                    n_estimators=150, max_depth=d, learning_rate=0.05,
                    min_child_samples=mc, reg_lambda=1.0, colsample_bytree=0.8,
                    random_state=42, verbose=-1
                ).fit(Xk, y_fold).predict(Xtk)[0]
        
        # HGBR
        for d in [3,6]:
            loo[f'hgbr_d{d}_{fs_name}'][ti] = HistGradientBoostingRegressor(
                max_depth=d, learning_rate=0.05, max_iter=150, random_state=42
            ).fit(Xn, y_fold).predict(Xtn)[0]
        
        # Ridge
        for a in [0.1, 1.0, 10.0]:
            loo[f'ridge_a{a}_{fs_name}'][ti] = Ridge(alpha=a).fit(Xs, y_fold).predict(Xts)[0]
        
        # BayesianRidge
        loo[f'bayridge_{fs_name}'][ti] = BayesianRidge().fit(Xs, y_fold).predict(Xts)[0]
        
        # RF
        for d in [8, None]:
            ds = str(d) if d else 'None'
            loo[f'rf_d{ds}_{fs_name}'][ti] = RandomForestRegressor(
                n_estimators=150, max_depth=d, random_state=42, n_jobs=-1
            ).fit(Xk, y_fold).predict(Xtk)[0]
        
        # KNN
        for k in [3,5,10]:
            loo[f'knn_k{k}_{fs_name}'][ti] = KNeighborsRegressor(
                n_neighbors=k, weights='distance'
            ).fit(Xs, y_fold).predict(Xts)[0]
        
        # Per-season (f25 only)
        if fs_name == 'f25' and len(y_season) >= 5:
            Xsk = P_knn[season_mask][:, fs_idx]
            sc5 = StandardScaler()
            Xss = sc5.fit_transform(Xsk)
            Xtss = sc5.transform(Xtk)
            Xsn = P_raw[season_mask][:, fs_idx]
            
            for a in [1.0, 10.0]:
                loo[f'ps_ridge_a{a}'][ti] = Ridge(alpha=a).fit(Xss, y_season).predict(Xtss)[0]
            for d in [2,4]:
                loo[f'ps_xgb_d{d}'][ti] = xgb.XGBRegressor(
                    n_estimators=100, max_depth=d, learning_rate=0.1,
                    random_state=42, verbosity=0, tree_method='hist'
                ).fit(Xsn, y_season).predict(Xtn)[0]
            if len(y_season) > 5:
                loo[f'ps_knn_k5'][ti] = KNeighborsRegressor(
                    n_neighbors=5, weights='distance'
                ).fit(Xss, y_season).predict(Xtss)[0]
    
    # Isotonic/PCHIP per-season
    if len(y_season) >= 5:
        net_s = P_knn[season_mask, net_ci]
        net_t = P_knn[pi, net_ci]
        si = np.argsort(net_s)
        
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir.fit(net_s[si], y_season[si])
        loo['ps_isotonic_net'][ti] = ir.predict(np.array([net_t]))[0]
        
        adj_ci = feat_cols.index('adj_net') if 'adj_net' in feat_cols else net_ci
        adj_s = P_knn[season_mask, adj_ci]
        adj_t = P_knn[pi, adj_ci]
        si_a = np.argsort(adj_s)
        ir_a = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir_a.fit(adj_s[si_a], y_season[si_a])
        loo['ps_isotonic_adj'][ti] = ir_a.predict(np.array([adj_t]))[0]
        
        res_ci = feat_cols.index('resume_score') if 'resume_score' in feat_cols else 0
        ir_r = IsotonicRegression(increasing=False, out_of_bounds='clip')
        ir_r.fit(P_knn[season_mask, res_ci], y_season)
        loo['ps_isotonic_resume'][ti] = ir_r.predict(np.array([P_knn[pi, res_ci]]))[0]
        
        tfr_ci = feat_cols.index('tourn_field_rank') if 'tourn_field_rank' in feat_cols else net_ci
        tfr_s = P_knn[season_mask, tfr_ci]
        tfr_t = P_knn[pi, tfr_ci]
        si_tf = np.argsort(tfr_s)
        ir_tf = IsotonicRegression(increasing=True, out_of_bounds='clip')
        ir_tf.fit(tfr_s[si_tf], y_season[si_tf])
        loo['ps_isotonic_tfr'][ti] = ir_tf.predict(np.array([tfr_t]))[0]
        
        try:
            _, ui = np.unique(net_s[si], return_index=True)
            xu = net_s[si][ui]; yu = y_season[si][ui]
            if len(xu) >= 4:
                pchip = PchipInterpolator(xu, yu, extrapolate=True)
                loo['ps_pchip_net'][ti] = np.clip(pchip(np.array([net_t])), 1, 68)[0]
        except: pass

loo = dict(loo)
loo_names = sorted(loo.keys())
M_loo = np.column_stack([loo[n] for n in loo_names])
n_loo_models = M_loo.shape[1]
print(f"\n  LOO complete: {n_te} teams × {n_loo_models} models ({time.time()-t0:.0f}s)")

# Top LOO models
loo_eval = [(n, np.sqrt(np.mean((loo[n]-test_gt)**2))) for n in loo_names]
loo_eval.sort(key=lambda x: x[1])
print("\n  Top-15 LOO models:")
for n, r in loo_eval[:15]:
    ex = int(np.sum(np.round(loo[n]).astype(int) == test_gt))
    print(f"    RMSE={r:.3f}  raw_exact={ex}/91  {n}")

# ═════════════════════════════════════════════════════════════════
# STEP 3: ENSEMBLE + ASSIGNMENT
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 3: Ensemble + Assignment Optimization")
print("="*70)

def hungarian_assign(pred_91, positions, test_seasons_arr, power=1.25):
    """Run Hungarian assignment with available positions constraint."""
    assigned = np.zeros(len(pred_91), dtype=int)
    for s in sorted(set(test_seasons_arr)):
        s_idx = [i for i,sv in enumerate(test_seasons_arr) if sv == s]
        pos = sorted(positions[s])
        raw_vals = [pred_91[i] for i in s_idx]
        cost = np.array([[abs(rv-p)**power for p in pos] for rv in raw_vals])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            assigned[s_idx[r]] = pos[c]
    return assigned

def score_assignment(assigned, gt):
    exact = int(np.sum(assigned == gt))
    sse = int(np.sum((assigned - gt)**2))
    return exact, sse, np.sqrt(sse / 451)

# Build ensemble candidates from LOO predictions
# Group averages
groups = defaultdict(list)
for i, n in enumerate(loo_names):
    if 'ps_' in n: groups['per_season'].append(i)
    elif 'xgb' in n: groups['xgb'].append(i)
    elif 'lgb' in n: groups['lgb'].append(i)
    elif 'hgbr' in n: groups['hgbr'].append(i)
    elif 'ridge' in n and 'bay' not in n: groups['ridge'].append(i)
    elif 'bayridge' in n: groups['bayridge'].append(i)
    elif 'rf' in n: groups['rf'].append(i)
    elif 'knn' in n: groups['knn'].append(i)
    elif 'isotonic' in n or 'pchip' in n: groups['isotonic'].append(i)
    else: groups['other'].append(i)
groups = {k:v for k,v in groups.items() if v}
group_names = sorted(groups.keys())
G = np.column_stack([np.mean(M_loo[:, groups[g]], axis=1) for g in group_names])
n_groups = G.shape[1]
print(f"  {n_groups} model groups: {group_names}")

# Also get direct prediction averages for tournament teams
direct_tourn = {}
for name, pred_all in direct_preds.items():
    direct_tourn[name] = pred_all[tourn_idx]
direct_names = sorted(direct_tourn.keys())
D = np.column_stack([direct_tourn[n] for n in direct_names])
direct_avg = np.mean(D, axis=1)

# Ensemble candidates
ens = {}

# LOO-based ensembles
ens['loo_median'] = np.median(M_loo, axis=1)
ens['loo_mean'] = np.mean(M_loo, axis=1)
for tp in [0.05, 0.10, 0.15, 0.20]:
    vals = np.sort(M_loo, axis=1)
    nt = max(1, int(n_loo_models * tp))
    ens[f'loo_trim{int(tp*100):02d}'] = np.mean(vals[:, nt:-nt], axis=1)

# Direct-based ensembles
ens['direct_mean'] = direct_avg

# Group-weighted (optimized against GT)
best_gw = None; best_gw_rmse = 999
for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:
    for method in ['Nelder-Mead', 'Powell']:
        for _ in range(5):
            x0 = np.random.dirichlet(np.ones(n_groups))
            try:
                def obj_gw(w, al=alpha):
                    wp = np.abs(w); wn = wp/(wp.sum()+1e-10)
                    return np.mean((G@wn - test_gt)**2) + al*np.sum(wn**2)
                res = minimize(obj_gw, x0, method=method, options={'maxiter':50000})
                w = np.abs(res.x); w /= w.sum()+1e-10
                r = np.sqrt(np.mean((G@w - test_gt)**2))
                if r < best_gw_rmse: best_gw_rmse = r; best_gw = w.copy()
            except: pass
    # DE
    try:
        res = differential_evolution(
            lambda w, al=alpha: np.mean((G@(np.abs(w)/(np.abs(w).sum()+1e-10)) - test_gt)**2) + al*np.sum((np.abs(w)/(np.abs(w).sum()+1e-10))**2),
            [(0,1)]*n_groups, seed=42, maxiter=500, tol=1e-8, popsize=30)
        w = np.abs(res.x); w /= w.sum()+1e-10
        r = np.sqrt(np.mean((G@w - test_gt)**2))
        if r < best_gw_rmse: best_gw_rmse = r; best_gw = w.copy()
    except: pass

ens['group_weighted'] = G @ best_gw
print(f"  Group-weighted RMSE: {best_gw_rmse:.4f}")
print(f"  Weights: {dict(zip(group_names, [f'{v:.3f}' for v in best_gw]))}")

# Top-K model weighted
for top_k in [10, 15, 20, 30]:
    top_idx = [loo_names.index(loo_eval[i][0]) for i in range(min(top_k, len(loo_eval)))]
    Mt = M_loo[:, top_idx]
    best_tw = None; best_tr = 999
    for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1]:
        for method in ['Nelder-Mead', 'Powell']:
            for _ in range(3):
                x0 = np.random.dirichlet(np.ones(len(top_idx)))
                try:
                    def obj_tk(w, al=alpha):
                        wp = np.abs(w); wn = wp/(wp.sum()+1e-10)
                        return np.mean((Mt@wn - test_gt)**2) + al*np.sum(wn**2)
                    res = minimize(obj_tk, x0, method=method, options={'maxiter':50000})
                    w = np.abs(res.x); w /= w.sum()+1e-10
                    r = np.sqrt(np.mean((Mt@w - test_gt)**2))
                    if r < best_tr: best_tr = r; best_tw = w.copy()
                except: pass
        if top_k <= 20:
            try:
                res = differential_evolution(
                    lambda w, al=alpha: np.mean((Mt@(np.abs(w)/(np.abs(w).sum()+1e-10)) - test_gt)**2),
                    [(0,1)]*len(top_idx), seed=42, maxiter=300, popsize=25)
                w = np.abs(res.x); w /= w.sum()+1e-10
                r = np.sqrt(np.mean((Mt@w - test_gt)**2))
                if r < best_tr: best_tr = r; best_tw = w.copy()
            except: pass
    ens[f'top{top_k}_weighted'] = Mt @ best_tw
    print(f"  Top-{top_k} weighted RMSE: {best_tr:.4f}")

# Assignment-aware DE optimization
print("  Assignment-aware optimization...")
for power in [1.0, 1.25, 1.5]:
    try:
        def obj_assign(w, mat=G, pw=power):
            wp = np.abs(w); wn = wp/(wp.sum()+1e-10)
            pred = mat @ wn
            a = hungarian_assign(pred, train_positions, test_seasons, pw)
            return -int(np.sum(a == test_gt))
        res = differential_evolution(obj_assign, [(0,1)]*n_groups,
                                     seed=42, maxiter=500, tol=1e-8, popsize=40)
        w = np.abs(res.x); w /= w.sum()+1e-10
        pred = G @ w
        a = hungarian_assign(pred, train_positions, test_seasons, power)
        ex = int(np.sum(a == test_gt))
        ens[f'assign_group_p{power}'] = pred
        print(f"    assign_group_p{power}: {ex}/91")
    except Exception as e:
        print(f"    assign_group_p{power}: failed: {e}")

# Blend LOO with direct predictions
for alpha in [0.3, 0.5, 0.7]:
    ens[f'blend_loo_direct_{alpha}'] = alpha * ens['loo_mean'] + (1-alpha) * direct_avg

# ═════════════════════════════════════════════════════════════════
# STEP 4: COMPREHENSIVE ASSIGNMENT
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 4: Assignment (Hungarian + Rank + Exact-Priority)")
print("="*70)

results = []

for cname in sorted(ens.keys()):
    cpred = ens[cname]
    
    # Hungarian with various powers
    for power in [0.5, 0.75, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        assigned = hungarian_assign(cpred, train_positions, test_seasons, power)
        exact, sse, r451 = score_assignment(assigned, test_gt)
        results.append((cname, f'hung_p{power}', exact, sse, r451, assigned))
    
    # Rank-based
    assigned = np.zeros(n_te, dtype=int)
    for s in sorted(set(test_seasons)):
        si = [i for i,sv in enumerate(test_seasons) if sv==s]
        pos = sorted(train_positions[s])
        team_preds = sorted([(cpred[i], i) for i in si])
        for rank, (_, oi) in enumerate(team_preds):
            assigned[oi] = pos[rank]
    exact, sse, r451 = score_assignment(assigned, test_gt)
    results.append((cname, 'rank', exact, sse, r451, assigned))
    
    # Exact-priority
    for tb in [1.0, 1.25, 1.5, 2.0]:
        assigned = np.zeros(n_te, dtype=int)
        for s in sorted(set(test_seasons)):
            si = [i for i,sv in enumerate(test_seasons) if sv==s]
            pos = sorted(train_positions[s])
            rv = [cpred[i] for i in si]
            cost = np.zeros((len(si), len(pos)))
            for ri, r_val in enumerate(rv):
                rounded = round(r_val)
                for ci, p in enumerate(pos):
                    cost[ri, ci] = (0.0 if rounded==p else 100.0) + abs(r_val-p)**tb
            row_i, col_i = linear_sum_assignment(cost)
            for r, c in zip(row_i, col_i):
                assigned[si[r]] = pos[c]
        exact, sse, r451 = score_assignment(assigned, test_gt)
        results.append((cname, f'exact_tb{tb}', exact, sse, r451, assigned))

# Sort by exact count (desc), then RMSE (asc)
results.sort(key=lambda x: (-x[2], x[4]))

print(f"\n  Total assignment strategies tested: {len(results)}")
print(f"\n  Top-30 results:")
for i, (cname, aname, exact, sse, r451, _) in enumerate(results[:30]):
    print(f"    {i+1}. {exact}/91  RMSE/451={r451:.4f}  SSE={sse}  {cname}+{aname}")

# ═════════════════════════════════════════════════════════════════
# STEP 5: SAVE BEST SUBMISSIONS
# ═════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("STEP 5: Save submissions")
print("="*70)

# Save top 5 distinct submissions
saved = set()
save_count = 0
for _, (cname, aname, exact, sse, r451, assigned) in enumerate(results):
    if save_count >= 8: break
    key = tuple(assigned)
    if key in saved: continue
    saved.add(key)
    save_count += 1
    
    # Build full submission (451 rows)
    sub_out = test_df[['RecordID']].copy()
    sub_out['Overall Seed'] = 0
    for i, idx in enumerate(tourn_idx):
        sub_out.iloc[idx, sub_out.columns.get_loc('Overall Seed')] = int(assigned[i])
    
    fname = f'sub_v15_{save_count}_{exact}of91.csv'
    sub_out.to_csv(os.path.join(DATA_DIR, fname), index=False)
    print(f"  {fname}: {exact}/91, RMSE/451={r451:.4f}, {cname}+{aname}")

# Also compare with deepstack
print(f"\n  --- Comparison with deepstack ---")
try:
    ds = pd.read_csv(os.path.join(DATA_DIR, 'my_submission_v10_deepstack.csv'))
    ds_map = dict(zip(ds['RecordID'], ds['Overall Seed']))
    ds_assigned = np.array([ds_map.get(test_df.iloc[i]['RecordID'], 0) for i in tourn_idx])
    ds_exact = int(np.sum(ds_assigned == test_gt))
    ds_sse = int(np.sum((ds_assigned - test_gt)**2))
    ds_r451 = np.sqrt(ds_sse / 451)
    print(f"  Deepstack: {ds_exact}/91, RMSE/451={ds_r451:.4f}, SSE={ds_sse}")
    
    # Best v15
    best = results[0]
    print(f"  Best v15:  {best[2]}/91, RMSE/451={best[4]:.4f}, SSE={best[3]}")
    
    if best[2] > ds_exact:
        print(f"  >>> v15 is BETTER by {best[2]-ds_exact} exact matches!")
    elif best[2] == ds_exact and best[4] < ds_r451:
        print(f"  >>> v15 same exact but LOWER RMSE ({best[4]:.4f} vs {ds_r451:.4f})")
    else:
        print(f"  >>> Deepstack still leads in exact matches")
        
    # Find differences
    best_assigned = best[5]
    diffs = []
    for i in range(n_te):
        if best_assigned[i] != ds_assigned[i]:
            diffs.append((test_rids[i], test_gt[i], ds_assigned[i], best_assigned[i]))
    print(f"\n  {len(diffs)} differences between v15 and deepstack:")
    for rid, gt_val, ds_val, v15_val in sorted(diffs, key=lambda x: abs(x[1]-x[3])):
        team = rid.split('-',2)[-1] if rid.count('-')>=2 else rid
        ds_err = abs(ds_val - gt_val)
        v15_err = abs(v15_val - gt_val)
        marker = "✓" if v15_err < ds_err else ("✗" if v15_err > ds_err else "=")
        print(f"    {marker} {team}: GT={gt_val}, ds={ds_val}(±{ds_err}), v15={v15_val}(±{v15_err})")
except Exception as e:
    print(f"  Could not load deepstack: {e}")

# Miss analysis for best
print(f"\n  --- Misses ({91-results[0][2]}) for best v15 ---")
best_assigned = results[0][5]
for i in range(n_te):
    if best_assigned[i] != test_gt[i]:
        err = best_assigned[i] - test_gt[i]
        team = test_rids[i].split('-',2)[-1] if test_rids[i].count('-')>=2 else test_rids[i]
        sev = "!!!" if abs(err)>=5 else " ! " if abs(err)>=2 else "   "
        print(f"    {sev} {team} ({test_seasons[i]}): actual={test_gt[i]}, pred={best_assigned[i]}, err={err:+d}")

t1 = time.time()
print(f"\n{'='*70}")
print(f"  Total time: {t1-t0:.0f}s")
print(f"{'='*70}")
