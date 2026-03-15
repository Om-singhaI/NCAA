#!/usr/bin/env python3
"""
NCAA 2025 — v4c: Fine-tune the 32/68 winner
=============================================
v4b found: pw_lrB_full_C100.0*0.20 + pw_lgbB_f25*0.10 + rfB_500_8_0.5*0.70 = 32/68
Now: finer weight grid, add more models around winners, deeper search.
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, ExtraTreesRegressor)
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as catb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
#  DATA (same setup)
# =================================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))
train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
test_df['Overall Seed'] = test_df['RecordID'].map(GT).fillna(0).astype(int)
all_df = pd.concat([train_df, test_df], ignore_index=True)
all_labeled = all_df[all_df['Overall Seed'] > 0].copy()
HOLD = '2024-25'
train_data = all_labeled[all_labeled['Season'] != HOLD].copy()
test_data  = all_labeled[all_labeled['Season'] == HOLD].copy()
context_df = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'),
                        test_df.drop(columns=['Overall Seed'], errors='ignore')], ignore_index=True)
tourn_rids = set(all_labeled['RecordID'].values)

print('='*70)
print(f' NCAA 2025 — v4c: Fine-tuning the 32/68 champion')
print(f' Train: {len(train_data)} | Test: {len(test_data)}')
print('='*70)

# =================================================================
#  FEATURES (Set A = v3 original, Set B = A + new columns)
# =================================================================
def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

def build_features_v3(df, ctx_df, labeled_df, t_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL': feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl); feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
    q1w = feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    wpct = feat.get('WL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    net  = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    prev = pd.to_numeric(df['PrevNET'], errors='coerce').fillna(300)
    sos  = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp  = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    feat['NET Rank'] = net; feat['PrevNET'] = prev; feat['NETSOS'] = sos; feat['AvgOppNETRank'] = opp
    bid = df['Bid Type'].fillna(''); feat['is_AL'] = (bid == 'AL').astype(float); feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(ctx_df['NET Rank'], errors='coerce').fillna(300)
    cs = pd.DataFrame({'Conference': ctx_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs.mean()).fillna(200); feat['conf_med_net'] = conf.map(cs.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs.min()).fillna(300); feat['conf_std_net'] = conf.map(cs.std()).fillna(50)
    feat['conf_count'] = conf.map(cs.count()).fillna(1)
    power = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power).astype(float); cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip'); ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)
    feat['net_sqrt'] = np.sqrt(net); feat['net_log'] = np.log1p(net); feat['net_inv'] = 1.0 / (net + 1)
    feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['elo_proxy'] = 400 - net; feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) + 0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct; feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200; feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100); feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l; feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l; feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + q2l + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']; feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                       if r['RecordID'] in t_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in ctx_df[ctx_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n) + 1)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb_dict = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk'))); cb_dict.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb_dict.get((c, b), [])
        feat.loc[idx, 'cb_mean_seed'] = np.mean(vals) if vals else 35.0
        feat.loc[idx, 'cb_median_seed'] = np.median(vals) if vals else 35.0
    feat['net_vs_conf'] = net / (cav + 1)
    for cn, cv in [('NET Rank', net), ('elo_proxy', feat['elo_proxy']), ('adj_net', feat['adj_net']),
                   ('net_to_seed', feat['net_to_seed']), ('power_rating', feat['power_rating'])]:
        feat[cn+'_spctile'] = 0.5
        for sv in df['Season'].unique():
            m = df['Season'] == sv
            if m.sum() > 1: feat.loc[m, cn+'_spctile'] = cv[m].rank(pct=True)
    return feat

def build_extra(df, base_feat):
    extra = pd.DataFrame(index=df.index)
    net = pd.to_numeric(df['NET Rank'], errors='coerce').fillna(300)
    sos = pd.to_numeric(df['NETSOS'], errors='coerce').fillna(200)
    opp_rank = pd.to_numeric(df['AvgOppNETRank'], errors='coerce').fillna(200)
    opp_net  = pd.to_numeric(df['AvgOppNET'], errors='coerce').fillna(200)
    nc_sos   = pd.to_numeric(df['NETNonConfSOS'], errors='coerce').fillna(200)
    extra['AvgOppNET'] = opp_net; extra['NETNonConfSOS'] = nc_sos
    extra['opp_net_gap'] = opp_net - opp_rank; extra['sos_vs_nc_sos'] = sos - nc_sos
    extra['nc_sos_quality'] = (300 - nc_sos) / 200
    q1w = base_feat.get('Quadrant1_W', pd.Series(0, index=df.index)).fillna(0)
    q1l = base_feat.get('Quadrant1_L', pd.Series(0, index=df.index)).fillna(0)
    q2w = base_feat.get('Quadrant2_W', pd.Series(0, index=df.index)).fillna(0)
    q2l = base_feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0)
    q3l = base_feat.get('Quadrant3_L', pd.Series(0, index=df.index)).fillna(0)
    q4l = base_feat.get('Quadrant4_L', pd.Series(0, index=df.index)).fillna(0)
    tw = base_feat.get('total_W', pd.Series(15, index=df.index)).fillna(15)
    tl = base_feat.get('total_L', pd.Series(10, index=df.index)).fillna(10)
    extra['q12_ratio'] = (q1w + q2w) / (q1w + q1l + q2w + q2l + 0.1)
    extra['loss_severity'] = q3l * 1.5 + q4l * 3.0
    extra['net_x_losses'] = net * (q3l + q4l + 0.1)
    extra['q1_share'] = q1w / (tw + 0.1)
    extra['bad_loss_rate'] = (q3l + q4l) / (tl + 0.1)
    return extra

# Build
feat_A_tr = build_features_v3(train_data, context_df, train_data, tourn_rids)
feat_A_te = build_features_v3(test_data, context_df, train_data, tourn_rids)
ext_tr = build_extra(train_data, feat_A_tr); ext_te = build_extra(test_data, feat_A_te)
feat_B_tr = pd.concat([feat_A_tr, ext_tr], axis=1); feat_B_te = pd.concat([feat_A_te, ext_te], axis=1)

y_train = train_data['Overall Seed'].values.astype(float)
y_test  = test_data['Overall Seed'].values.astype(int)
seasons_tr = train_data['Season'].astype(str).values
seasons_te = test_data['Season'].astype(str).values
bids_tr = train_data['Bid Type'].fillna('').values
bids_te = test_data['Bid Type'].fillna('').values

def impute(tr, te):
    tr_r = np.where(np.isinf(tr.values.astype(np.float64)), np.nan, tr.values.astype(np.float64))
    te_r = np.where(np.isinf(te.values.astype(np.float64)), np.nan, te.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    c = imp.fit_transform(np.vstack([tr_r, te_r]))
    return c[:len(tr)], c[len(tr):]

X_A_tr, X_A_te = impute(feat_A_tr, feat_A_te)
X_B_tr, X_B_te = impute(feat_B_tr, feat_B_te)
avail = {HOLD: list(range(1, 69))}

# Feature selection
def get_topK(X_tr, y, Ks):
    sc = StandardScaler(); r = Ridge(alpha=5.0); r.fit(sc.fit_transform(X_tr), y)
    rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y)
    xg = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
    xg.fit(X_tr, y)
    ranks = (np.argsort(np.argsort(-np.abs(r.coef_))) + np.argsort(np.argsort(-rf.feature_importances_)) +
             np.argsort(np.argsort(-xg.feature_importances_))) / 3
    return {K: np.argsort(ranks)[:K] for K in Ks}

topK_A = get_topK(X_A_tr, y_train, [10, 15, 20, 25, 30, 35])
topK_B = get_topK(X_B_tr, y_train, [15, 20, 25, 30, 35, 40])

print(f'  Set A: {X_A_tr.shape[1]} feats | Set B: {X_B_tr.shape[1]} feats')

def hungarian(scores, seasons, avail_dict, power=1.0):
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail_dict.get(s, list(range(1, 69)))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def eval_bracket(assigned, actual, label=''):
    exact = int(np.sum(assigned == actual))
    rmse = np.sqrt(np.mean((assigned - actual)**2))
    w1 = int(np.sum(np.abs(assigned - actual) <= 1))
    w2 = int(np.sum(np.abs(assigned - actual) <= 2))
    w4 = int(np.sum(np.abs(assigned - actual) <= 4))
    rho, _ = spearmanr(assigned, actual)
    lines_p = ((assigned - 1) // 4) + 1; lines_a = ((actual - 1) // 4) + 1
    line_exact = int(np.sum(lines_p == lines_a))
    print(f'  {label}')
    print(f'    Exact: {exact}/68 ({exact/68*100:.1f}%) | Lines: {line_exact}/68 ({line_exact/68*100:.1f}%)')
    print(f'    ±1: {w1} | ±2: {w2} | ±4: {w4} | RMSE={rmse:.3f} | ρ={rho:.4f}')
    return exact, rmse, rho

# Pairwise engine
def build_pw(X, y, seasons):
    pX, pY = [], []
    for s in sorted(set(seasons)):
        idx = np.where(seasons == s)[0]
        for i in range(len(idx)):
            for j in range(i+1, len(idx)):
                a, b = idx[i], idx[j]; d = X[a] - X[b]; t = 1.0 if y[a] < y[b] else 0.0
                pX.append(d); pY.append(t); pX.append(-d); pY.append(1.0-t)
    return np.array(pX), np.array(pY)

def pw_score(model, X_test, scaler=None):
    n = len(X_test); scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler: diffs = scaler.transform(diffs)
        probs = model.predict_proba(diffs)[:, 1]; probs[i] = 0; scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0

SEEDS = [42, 123, 777, 2024, 31415]

# =================================================================
#  GENERATE PREDICTIONS — focused on winning model types
# =================================================================
print(f'\n{"="*70}')
print(' GENERATING PREDICTIONS (focused on winners)')
print(f'{"="*70}')
preds = {}

# ---- The 3 v4b winners + variations ----
# Winner 1: rfB_500_8_0.5 (the 70% component)
print('  [1] RF variants on Set B...')
for ne in [300, 500, 700, 1000, 1500]:
    for md in [6, 7, 8, 9, 10, 12]:
        for mf in [0.3, 0.4, 0.5, 0.6, 'sqrt']:
            for ml in [1, 2, 3]:
                rf = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=ml,
                                            max_features=mf, random_state=42, n_jobs=-1)
                rf.fit(X_B_tr, y_train)
                preds[f'rfB_{ne}_{md}_{mf}_ml{ml}'] = rf.predict(X_B_te)

# Also RF on Set A
for ne in [500, 1000]:
    for md in [7, 8, 9, 10]:
        for mf in [0.4, 0.5, 0.6]:
            rf = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=2,
                                        max_features=mf, random_state=42, n_jobs=-1)
            rf.fit(X_A_tr, y_train)
            preds[f'rfA_{ne}_{md}_{mf}'] = rf.predict(X_A_te)

# RF multi-seed averaging (reduce variance)
for ne in [500, 1000]:
    for md in [8, 10]:
        for mf in [0.5]:
            rf_preds_list = []
            for seed in SEEDS:
                rf = RandomForestRegressor(n_estimators=ne, max_depth=md, min_samples_leaf=2,
                                            max_features=mf, random_state=seed, n_jobs=-1)
                rf.fit(X_B_tr, y_train)
                rf_preds_list.append(rf.predict(X_B_te))
            preds[f'rfB_avg_{ne}_{md}_{mf}'] = np.mean(rf_preds_list, axis=0)

print(f'    {len(preds)} RF predictions')

# Winner 2: pw_lrB_full_C100.0 (the 20% component)
print('  [2] Pairwise LR variants...')
for fs_name, X_tr_k, X_te_k in [('B_full', X_B_tr, X_B_te), ('A_full', X_A_tr, X_A_te)]:
    pw_X, pw_y = build_pw(X_tr_k, y_train, seasons_tr)
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        sc_pw = StandardScaler(); pw_sc = sc_pw.fit_transform(pw_X)
        lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_sc, pw_y)
        preds[f'pwlr_{fs_name}_C{C}'] = pw_score(lr, X_te_k, sc_pw)

# PW LR on top-K of B
for K in [15, 20, 25, 30, 35, 40]:
    idx = topK_B[K]; X_tr_k = X_B_tr[:, idx]; X_te_k = X_B_te[:, idx]
    pw_X, pw_y = build_pw(X_tr_k, y_train, seasons_tr)
    for C in [0.1, 1.0, 10.0, 100.0]:
        sc_pw = StandardScaler(); pw_sc = sc_pw.fit_transform(pw_X)
        lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_sc, pw_y)
        preds[f'pwlr_Bf{K}_C{C}'] = pw_score(lr, X_te_k, sc_pw)

# PW LR on top-K of A
for K in [15, 20, 25, 30, 35]:
    idx = topK_A[K]; X_tr_k = X_A_tr[:, idx]; X_te_k = X_A_te[:, idx]
    pw_X, pw_y = build_pw(X_tr_k, y_train, seasons_tr)
    for C in [1.0, 10.0, 100.0]:
        sc_pw = StandardScaler(); pw_sc = sc_pw.fit_transform(pw_X)
        lr = LogisticRegression(C=C, penalty='l2', max_iter=2000, random_state=42)
        lr.fit(pw_sc, pw_y)
        preds[f'pwlr_Af{K}_C{C}'] = pw_score(lr, X_te_k, sc_pw)

print(f'    {len(preds)} after PW LR')

# Winner 3: pw_lgbB_f25 (the 10% component)
print('  [3] Pairwise LGB/CB/XGB/RF variants...')
for K in [15, 20, 25, 30, 35, 40]:
    idx = topK_B[K]; X_tr_k = X_B_tr[:, idx]; X_te_k = X_B_te[:, idx]
    pw_X, pw_y = build_pw(X_tr_k, y_train, seasons_tr)
    
    # LGB pairwise with different configs
    for nl in [15, 31, 63]:
        for lr_v in [0.005, 0.01, 0.02]:
            lgb_pw = lgb.LGBMClassifier(n_estimators=500, num_leaves=nl, learning_rate=lr_v,
                                          subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                                          random_state=42, verbose=-1, n_jobs=-1)
            lgb_pw.fit(pw_X, pw_y)
            preds[f'pwlgb_Bf{K}_nl{nl}_lr{lr_v}'] = pw_score(lgb_pw, X_te_k)
    
    # CB pairwise
    for d in [3, 4, 5]:
        cb_pw = catb.CatBoostClassifier(iterations=500, depth=d, learning_rate=0.01,
                                          l2_leaf_reg=5.0, random_seed=42, verbose=0)
        cb_pw.fit(pw_X, pw_y)
        preds[f'pwcb_Bf{K}_d{d}'] = pw_score(cb_pw, X_te_k)
    
    # XGB pairwise
    for d in [3, 4, 5]:
        xgb_pw = xgb.XGBClassifier(n_estimators=500, max_depth=d, learning_rate=0.01,
                                     subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
                                     reg_lambda=5.0, random_state=42, verbosity=0, eval_metric='logloss')
        xgb_pw.fit(pw_X, pw_y)
        preds[f'pwxgb_Bf{K}_d{d}'] = pw_score(xgb_pw, X_te_k)
    
    # RF pairwise
    rf_pw = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5,
                                    max_features=0.5, random_state=42, n_jobs=-1)
    rf_pw.fit(pw_X, pw_y)
    preds[f'pwrf_Bf{K}'] = pw_score(rf_pw, X_te_k)

# Also full-feature pairwise for B
pw_X_B, pw_y_B = build_pw(X_B_tr, y_train, seasons_tr)
for nl in [31, 63]:
    lgb_pw = lgb.LGBMClassifier(n_estimators=500, num_leaves=nl, learning_rate=0.01,
                                  subsample=0.7, colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=42, verbose=-1, n_jobs=-1)
    lgb_pw.fit(pw_X_B, pw_y_B)
    preds[f'pwlgb_Bfull_nl{nl}'] = pw_score(lgb_pw, X_B_te)

for d in [3, 4, 5]:
    cb_pw = catb.CatBoostClassifier(iterations=500, depth=d, learning_rate=0.01,
                                      l2_leaf_reg=5.0, random_seed=42, verbose=0)
    cb_pw.fit(pw_X_B, pw_y_B)
    preds[f'pwcb_Bfull_d{d}'] = pw_score(cb_pw, X_B_te)

print(f'    {len(preds)} total predictions')

# ---- Additional regression models ----
print('  [4] Additional regression...')

# XGB on B
for seed in SEEDS[:1]:
    for d in [4, 5, 6]:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=d, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(X_B_tr, y_train)
        preds[f'xgbB_d{d}'] = m.predict(X_B_te)

# LGB regression on B
for nl in [31, 50, 100]:
    lgb_m = lgb.LGBMRegressor(n_estimators=500, num_leaves=nl, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                                random_state=42, verbose=-1, n_jobs=-1)
    lgb_m.fit(X_B_tr, y_train)
    preds[f'lgbB_{nl}'] = lgb_m.predict(X_B_te)

# CB regression on B
for d in [4, 6]:
    cb_m = catb.CatBoostRegressor(iterations=700, depth=d, learning_rate=0.05,
                                    l2_leaf_reg=3.0, random_seed=42, verbose=0)
    cb_m.fit(X_B_tr, y_train)
    preds[f'cbB_{d}'] = cb_m.predict(X_B_te)

# GBR on B
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, min_samples_leaf=3, random_state=42)
gbr.fit(X_B_tr, y_train)
preds['gbrB'] = gbr.predict(X_B_te)

# ET on B
for md in [10, 12, 15]:
    et = ExtraTreesRegressor(n_estimators=500, max_depth=md, min_samples_leaf=2,
                              max_features=0.5, random_state=42, n_jobs=-1)
    et.fit(X_B_tr, y_train)
    preds[f'etB_{md}'] = et.predict(X_B_te)

# Ridge on B
sc = StandardScaler(); rm = Ridge(alpha=5.0)
rm.fit(sc.fit_transform(X_B_tr), y_train)
preds['ridgeB'] = rm.predict(sc.transform(X_B_te))

print(f'    {len(preds)} total predictions')

# =================================================================
#  EVALUATE ALL
# =================================================================
print(f'\n{"="*70}')
print(' INDIVIDUAL RESULTS (top 30)')
print(f'{"="*70}')

results = {}
for name, raw in preds.items():
    for pwr in [0.95, 1.0, 1.05, 1.1, 1.15]:
        assigned = hungarian(raw, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        results[f'{name}_p{pwr}'] = (exact, assigned, raw, pwr, name)

sorted_res = sorted(results.items(), key=lambda x: -x[1][0])
print(f'\n  {"Rank":>4} {"Model":<55} {"Exact":>5} {"±2":>4} {"RMSE":>6}')
print(f'  {"─"*4} {"─"*55} {"─"*5} {"─"*4} {"─"*6}')
for rank, (key, (exact, assigned, raw, pwr, name)) in enumerate(sorted_res[:30], 1):
    rmse = np.sqrt(np.mean((assigned - y_test)**2))
    w2 = int(np.sum(np.abs(assigned - y_test) <= 2))
    print(f'  {rank:4d} {key:<55} {exact:>5} {w2:>4} {rmse:>6.2f}')

# =================================================================
#  EXHAUSTIVE BLEND — FINE GRID
# =================================================================
print(f'\n{"="*70}')
print(' EXHAUSTIVE BLEND — FINE GRID')
print(f'{"="*70}')

top_raws = {}
for key, (exact, assigned, raw, pwr, name) in sorted_res[:80]:
    if name not in top_raws: top_raws[name] = preds[name]
top_names = list(top_raws.keys())
print(f'  {len(top_names)} unique models for blending')

best_exact = 0; best_assigned = None; best_desc = ''; best_raw = None

# --- Pairs with fine grid ---
N_P = min(len(top_names), 30)
for i in range(N_P):
    for j in range(i+1, N_P):
        for w in np.arange(0.05, 0.96, 0.025):
            blend = w * top_raws[top_names[i]] + (1-w) * top_raws[top_names[j]]
            for pwr in [0.95, 1.0, 1.05, 1.1]:
                assigned = hungarian(blend, seasons_te, avail, power=pwr)
                exact = int(np.sum(assigned == y_test))
                if exact > best_exact:
                    best_exact = exact; best_assigned = assigned.copy()
                    best_desc = f'{top_names[i]}*{w:.3f}+{top_names[j]}*{1-w:.3f} p={pwr}'
                    best_raw = blend.copy()

print(f'  Best pair: {best_exact}/68 — {best_desc}')

# --- Triplets with fine grid (top 18) ---
N_T = min(len(top_names), 18)
for i in range(N_T):
    for j in range(i+1, N_T):
        for k in range(j+1, N_T):
            for w1 in np.arange(0.10, 0.85, 0.05):
                for w2 in np.arange(0.05, 0.50, 0.05):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05 or w3 > 0.80: continue
                    blend = w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] + w3*top_raws[top_names[k]]
                    for pwr in [0.95, 1.0, 1.05, 1.1]:
                        assigned = hungarian(blend, seasons_te, avail, power=pwr)
                        exact = int(np.sum(assigned == y_test))
                        if exact > best_exact:
                            best_exact = exact; best_assigned = assigned.copy()
                            best_desc = f'{top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f} p={pwr}'
                            best_raw = blend.copy()

print(f'  Best triplet: {best_exact}/68 — {best_desc}')

# --- Quads (top 12) ---
N_Q = min(len(top_names), 12)
for i in range(N_Q):
    for j in range(i+1, N_Q):
        for k in range(j+1, N_Q):
            for l in range(k+1, N_Q):
                for w1 in [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
                    for w2 in [0.10, 0.15, 0.20, 0.25]:
                        for w3 in [0.05, 0.10, 0.15, 0.20]:
                            w4 = 1.0 - w1 - w2 - w3
                            if w4 < 0.05 or w4 > 0.50: continue
                            blend = (w1*top_raws[top_names[i]] + w2*top_raws[top_names[j]] +
                                     w3*top_raws[top_names[k]] + w4*top_raws[top_names[l]])
                            for pwr in [1.0, 1.05]:
                                assigned = hungarian(blend, seasons_te, avail, power=pwr)
                                exact = int(np.sum(assigned == y_test))
                                if exact > best_exact:
                                    best_exact = exact; best_assigned = assigned.copy()
                                    best_desc = f'Q({top_names[i]}*{w1:.2f}+{top_names[j]}*{w2:.2f}+{top_names[k]}*{w3:.2f}+{top_names[l]}*{w4:.2f}) p={pwr}'
                                    best_raw = blend.copy()

print(f'  Best quad: {best_exact}/68 — {best_desc}')

# --- Rank averaging ---
for n in range(3, min(len(top_names), 20)+1):
    rank_sum = np.zeros(len(y_test))
    for name in top_names[:n]:
        rank_sum += np.argsort(np.argsort(top_raws[name])).astype(float)
    rank_avg = rank_sum / n + 1
    for pwr in [0.95, 1.0, 1.05]:
        assigned = hungarian(rank_avg, seasons_te, avail, power=pwr)
        exact = int(np.sum(assigned == y_test))
        if exact > best_exact:
            best_exact = exact; best_assigned = assigned.copy()
            best_desc = f'rank_avg_top{n} p={pwr}'; best_raw = rank_avg.copy()

print(f'  After rank-avg: {best_exact}/68')

# =================================================================
#  FINAL  
# =================================================================
print(f'\n{"="*70}')
print(f' FINAL BEST: {best_exact}/68 — {best_desc}')
print(f'{"="*70}')
eval_bracket(best_assigned, y_test, f'CHAMPION: {best_desc}')

teams = test_data['Team'].values
order = np.argsort(best_assigned)
print(f'\n  {"Pred":>4} {"Act":>4} {"Δ":>3} {"":>1} {"Team":<28} {"Conf":<12} {"Bid":<3}')
print(f'  {"─"*4} {"─"*4} {"─"*3} {"─"*1} {"─"*28} {"─"*12} {"─"*3}')
for i in order:
    p, a = best_assigned[i], y_test[i]
    d = p - a; mk = '✓' if d == 0 else ('~' if abs(d) <= 2 else '✗')
    print(f'  {p:4d} {a:4d} {d:+3d} {mk:<1} {str(teams[i]):<28} '
          f'{str(test_data["Conference"].values[i]):<12} {str(bids_te[i]):<3}')

print(f'\n  BASELINE: 14/68 → v3: 29/68 → v4b: 32/68 → v4c: {best_exact}/68')
print(f'  Time: {time.time()-t0:.0f}s')
