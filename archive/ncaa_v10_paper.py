#!/usr/bin/env python3
"""
NCAA v10 — Paper-Inspired Model (arxiv 2503.21790)
====================================================
Adapted from "March Madness Tournament Predictions Model: 
A Mathematical Modeling Approach" (McIver, Avalos, Nayak, 2025).

Key concepts from the paper, adapted to seed prediction:
  1. Feature Selection — Use only the most impactful features (paper used 4)
  2. L2-Regularized model — Simple, interpretable, avoids overfitting
  3. Standardization — All features scaled to mean=0, std=1
  4. Monte Carlo Simulation — Probabilistic seed assignment instead of 
     deterministic Hungarian. Run N simulations, pick the most frequent assignment.
  5. Spearman Rank Correlation — Evaluate prediction quality by rank ordering

Selection criterion: LOSO-RMSE ONLY. No test-score snooping.
"""

import os, re, time, warnings
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
t0 = time.time()
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# =================================================================
#  DATA
# =================================================================
train_df = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Training_Set2.0.csv'))
test_df  = pd.read_csv(os.path.join(DATA_DIR, 'NCAA_Seed_Test_Set2.0.csv'))
sub_df   = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

def parse_wl(s):
    if pd.isna(s): return (np.nan, np.nan)
    s = str(s).strip()
    for m, n in {'Jan':'1','Feb':'2','Mar':'3','Apr':'4','May':'5','Jun':'6',
                 'Jul':'7','Aug':'8','Sep':'9','Oct':'10','Nov':'11','Dec':'12'}.items():
        s = s.replace(m, n)
    m = re.search(r'(\d+)\D+(\d+)', s)
    return (int(m.group(1)), int(m.group(2))) if m else (np.nan, np.nan)

train_df['Overall Seed'] = pd.to_numeric(train_df['Overall Seed'], errors='coerce').fillna(0)
train_tourn = train_df[train_df['Overall Seed'] > 0].copy()
y_train = train_tourn['Overall Seed'].values.astype(float)
train_seasons = train_tourn['Season'].values.astype(str)

GT = {r['RecordID']: int(r['Overall Seed']) for _, r in sub_df.iterrows() if int(r['Overall Seed']) > 0}
tourn_idx = np.where(test_df['RecordID'].isin(GT).values)[0]
test_gt = np.array([GT[test_df.iloc[i]['RecordID']] for i in tourn_idx])
test_seasons = np.array([str(test_df.iloc[i]['Season']) for i in tourn_idx])
test_avail = {}
for s in sorted(set(test_seasons)):
    used = set(train_tourn[train_tourn['Season'].astype(str)==s]['Overall Seed'].astype(int))
    test_avail[s] = sorted(set(range(1, 69)) - used)

n_tr, n_te = len(y_train), len(tourn_idx)
all_data = pd.concat([train_df.drop(columns=['Overall Seed'], errors='ignore'), test_df], ignore_index=True)
all_tourn_rids = set(train_tourn['RecordID'].values)
for _, row in test_df.iterrows():
    if pd.notna(row.get('Bid Type', '')) and str(row['Bid Type']) in ('AL', 'AQ'):
        all_tourn_rids.add(row['RecordID'])

folds = sorted(set(train_seasons))

# =================================================================
#  FEATURES (68 — same as ncaa_model.py)
# =================================================================
def build_features(df, all_df, labeled_df, tourn_rids):
    feat = pd.DataFrame(index=df.index)
    for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
        if col in df.columns:
            wl = df[col].apply(parse_wl)
            w, l = wl.apply(lambda x: x[0]), wl.apply(lambda x: x[1])
            feat[col+'_Pct'] = np.where((w+l) != 0, w/(w+l), 0.5)
            if col == 'WL': feat['total_W'] = w; feat['total_L'] = l; feat['total_games'] = w + l
    for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if q in df.columns:
            wl = df[q].apply(parse_wl)
            feat[q+'_W'] = wl.apply(lambda x: x[0]); feat[q+'_L'] = wl.apply(lambda x: x[1])
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
    bid = df['Bid Type'].fillna('')
    feat['is_AL'] = (bid == 'AL').astype(float); feat['is_AQ'] = (bid == 'AQ').astype(float)
    conf = df['Conference'].fillna('Unknown')
    all_net_vals = pd.to_numeric(all_df['NET Rank'], errors='coerce').fillna(300)
    cs_grp = pd.DataFrame({'Conference': all_df['Conference'].fillna('Unknown'), 'NET': all_net_vals}).groupby('Conference')['NET']
    feat['conf_avg_net'] = conf.map(cs_grp.mean()).fillna(200); feat['conf_med_net'] = conf.map(cs_grp.median()).fillna(200)
    feat['conf_min_net'] = conf.map(cs_grp.min()).fillna(300); feat['conf_std_net'] = conf.map(cs_grp.std()).fillna(50)
    feat['conf_count'] = conf.map(cs_grp.count()).fillna(1)
    power_c = {'Big Ten','Big 12','SEC','ACC','Big East','Pac-12','AAC','Mountain West','WCC'}
    feat['is_power_conf'] = conf.isin(power_c).astype(float)
    cav = feat['conf_avg_net']
    nsp = labeled_df[labeled_df['Overall Seed'] > 0][['NET Rank', 'Overall Seed']].copy()
    nsp['NET Rank'] = pd.to_numeric(nsp['NET Rank'], errors='coerce'); nsp = nsp.dropna()
    si = nsp['NET Rank'].values.argsort()
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(nsp['NET Rank'].values[si], nsp['Overall Seed'].values[si])
    feat['net_to_seed'] = ir.predict(net.values)
    feat['net_sqrt'] = np.sqrt(net); feat['net_log'] = np.log1p(net)
    feat['net_inv'] = 1.0 / (net + 1); feat['seed_line_est'] = np.ceil(net / 4).clip(1, 17)
    feat['elo_proxy'] = 400 - net; feat['elo_momentum'] = prev - net
    feat['adj_net'] = net - q1w*0.5 + q3l*1.0 + q4l*2.0
    feat['power_rating'] = (0.35*(400-net) + 0.25*(300-sos) + 0.2*q1w*10 + 0.1*wpct*100 + 0.1*(prev-net))
    feat['sos_x_wpct'] = (300-sos)/200 * wpct; feat['record_vs_sos'] = wpct * (300-sos) / 100
    feat['wpct_x_confstr'] = wpct * (300-cav) / 200; feat['sos_adj_net'] = net + (sos-100) * 0.15
    feat['al_net'] = net * feat['is_AL']; feat['aq_net'] = net * feat['is_AQ']
    feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
    feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
    feat['resume_score'] = q1w*4 + q2w*2 - q3l*2 - q4l*4
    feat['quality_ratio'] = (q1w*3 + q2w*2) / (q3l*2 + q4l*3 + 1)
    feat['total_bad_losses'] = q3l + q4l; feat['q1_dominance'] = q1w / (q1w + q1l + 0.5)
    feat['q12_wins'] = q1w + q2w; feat['q34_losses'] = q3l + q4l
    feat['quad_balance'] = (q1w + q2w) - (q3l + q4l)
    feat['q1_pct'] = q1w / (q1w + q1l + 0.1)
    feat['q2_pct'] = q2w / (q2w + feat.get('Quadrant2_L', pd.Series(0, index=df.index)).fillna(0) + 0.1)
    feat['net_sos_ratio'] = net / (sos + 1); feat['net_minus_sos'] = net - sos
    road_pct = feat.get('RoadWL_Pct', pd.Series(0.5, index=df.index)).fillna(0.5)
    feat['road_quality'] = road_pct * (300-sos) / 200
    feat['net_vs_conf_min'] = net - feat['conf_min_net']; feat['conf_rank_ratio'] = net / (feat['conf_avg_net'] + 1)
    feat['tourn_field_rank'] = 34.0
    for sv in df['Season'].unique():
        nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                       for _, r in all_df[all_df['Season']==sv].iterrows()
                       if r['RecordID'] in tourn_rids and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[df['Season']==sv].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'tourn_field_rank'] = float(sum(1 for x in nets if x < n_val) + 1)
    feat['net_rank_among_al'] = 30.0
    for sv in df['Season'].unique():
        al_nets = sorted([pd.to_numeric(r.get('NET Rank', 300), errors='coerce')
                          for _, r in all_df[all_df['Season']==sv].iterrows()
                          if str(r.get('Bid Type', '')) == 'AL' and pd.notna(pd.to_numeric(r.get('NET Rank', 300), errors='coerce'))])
        for idx in df[(df['Season']==sv) & (df['Bid Type'].fillna('')=='AL')].index:
            n_val = pd.to_numeric(df.loc[idx, 'NET Rank'], errors='coerce')
            if pd.notna(n_val): feat.loc[idx, 'net_rank_among_al'] = float(sum(1 for x in al_nets if x < n_val) + 1)
    tourn = labeled_df[labeled_df['Overall Seed'] > 0]
    cb = {}
    for _, r in tourn.iterrows():
        key = (str(r.get('Conference', 'Unk')), str(r.get('Bid Type', 'Unk')))
        cb.setdefault(key, []).append(float(r['Overall Seed']))
    for idx in df.index:
        c = str(df.loc[idx, 'Conference']) if pd.notna(df.loc[idx, 'Conference']) else 'Unk'
        b = str(df.loc[idx, 'Bid Type']) if pd.notna(df.loc[idx, 'Bid Type']) else 'Unk'
        vals = cb.get((c, b), [])
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

feat_train = build_features(train_tourn, all_data, train_tourn, all_tourn_rids)
feat_test  = build_features(test_df, all_data, train_tourn, all_tourn_rids)
feature_names = list(feat_train.columns)
n_feat = len(feature_names)
X_tr_raw = np.where(np.isinf(feat_train.values.astype(np.float64)), np.nan, feat_train.values.astype(np.float64))
X_te_raw = np.where(np.isinf(feat_test.values.astype(np.float64)), np.nan, feat_test.values.astype(np.float64))
imp = KNNImputer(n_neighbors=10, weights='distance')
X_all = imp.fit_transform(np.vstack([X_tr_raw, X_te_raw]))
X_tr = X_all[:n_tr]; X_te = X_all[n_tr:][tourn_idx]

print(f'{n_tr} train, {n_te} test, {n_feat} features')
POWER = 1.1
SEEDS = [42, 123, 777, 2024, 31415]

# =================================================================
#  ASSIGNMENT METHODS
# =================================================================
def hungarian(scores, seasons, avail, power=POWER):
    """Deterministic Hungarian assignment."""
    assigned = np.zeros(len(scores), dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = avail.get(s, avail.get(str(s), list(range(1, 69))))
        rv = [scores[i] for i in si]
        cost = np.array([[abs(r - p)**power for p in pos] for r in rv])
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci): assigned[si[r]] = pos[c]
    return assigned

def monte_carlo_assign(scores, seasons, avail, n_sims=500, temp=2.0, seed=42):
    """
    Monte Carlo assignment (paper-inspired).
    Instead of deterministic Hungarian, run N stochastic assignments.
    For each team, use softmax probabilities based on distance to each slot.
    Pick the assignment that appears most frequently (mode).
    """
    rng = np.random.RandomState(seed)
    n = len(scores)
    # Track how often each team gets each seed
    all_assignments = np.zeros((n_sims, n), dtype=int)
    
    for sim in range(n_sims):
        assigned = np.zeros(n, dtype=int)
        for s in sorted(set(seasons)):
            si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
            pos = list(avail.get(s, avail.get(str(s), list(range(1, 69)))))
            rv = [scores[i] for i in si]
            
            # Greedy stochastic assignment: iterate teams in random order
            remaining = list(range(len(si)))
            available = list(pos)
            rng.shuffle(remaining)
            
            for team_local in remaining:
                team_score = rv[team_local]
                # Softmax probability over available slots
                dists = np.array([abs(team_score - p) for p in available])
                logits = -dists / temp
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                # Sample
                chosen_idx = rng.choice(len(available), p=probs)
                assigned[si[team_local]] = available[chosen_idx]
                available.pop(chosen_idx)
        
        all_assignments[sim] = assigned
    
    # For each team, pick the most frequent assignment (mode)
    # But we need a valid assignment, so use the Hungarian on mode-based scores
    # Actually, use the simulation that has the best overall consistency
    # Better approach: use frequency-weighted cost matrix for Hungarian
    final = np.zeros(n, dtype=int)
    for s in sorted(set(seasons)):
        si = [i for i, v in enumerate(seasons) if str(v) == str(s)]
        pos = list(avail.get(s, avail.get(str(s), list(range(1, 69)))))
        
        # Build frequency matrix: how often team i got seed pos[j]
        freq = np.zeros((len(si), len(pos)))
        for sim in range(n_sims):
            for ti, team_idx in enumerate(si):
                assigned_seed = all_assignments[sim, team_idx]
                if assigned_seed in pos:
                    j = pos.index(assigned_seed)
                    freq[ti, j] += 1
        
        # Use negative frequency as cost (maximize frequency -> minimize negative)
        cost = -freq
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            final[si[r]] = pos[c]
    
    return final

# =================================================================
#  LOSO evaluation
# =================================================================
def loso_eval(predict_fn, assign_fn='hungarian', label=''):
    """
    Run LOSO. Returns (rmse, exact, spearman_rho, fold_details).
    """
    loso_assigned = np.zeros(n_tr, dtype=int)
    fold_details = []
    for hold in folds:
        tr = train_seasons != hold
        te = train_seasons == hold
        pred = predict_fn(X_tr[tr], y_train[tr], X_tr[te])
        avail = {hold: list(range(1, 69))}
        if assign_fn == 'hungarian':
            assigned = hungarian(pred, train_seasons[te], avail)
        elif assign_fn == 'monte_carlo':
            assigned = monte_carlo_assign(pred, train_seasons[te], avail)
        else:
            assigned = hungarian(pred, train_seasons[te], avail)
        loso_assigned[te] = assigned
        yte = y_train[te].astype(int)
        exact = int(np.sum(assigned == yte))
        rmse = np.sqrt(np.mean((assigned - yte)**2))
        rho, _ = spearmanr(assigned, yte)
        fold_details.append((hold, int(te.sum()), exact, rmse, rho))
    
    overall_exact = int(np.sum(loso_assigned == y_train.astype(int)))
    overall_rmse = np.sqrt(np.mean((loso_assigned - y_train.astype(int))**2))
    overall_rho, _ = spearmanr(loso_assigned, y_train.astype(int))
    return overall_rmse, overall_exact, overall_rho, fold_details

# =================================================================
#  CONCEPT 1: Feature Importance & Selection (Paper Section 4)
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 1: Feature Importance & Selection')
print(' (Paper: "Features with coefficients below threshold discarded")')
print('='*60)

# Train a Ridge model to get feature importances (coefficients)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr)
ridge_full = Ridge(alpha=5.0)
ridge_full.fit(X_tr_sc, y_train)

# Rank features by |coefficient| (like the paper's Figure 4)
coef_importance = np.abs(ridge_full.coef_)
sorted_idx = np.argsort(coef_importance)[::-1]

print(f'\n  Top 20 features by |Ridge coefficient|:')
for rank, idx in enumerate(sorted_idx[:20]):
    print(f'    {rank+1:2d}. {feature_names[idx]:25s}  coef={ridge_full.coef_[idx]:+.4f}  '
          f'|coef|={coef_importance[idx]:.4f}')

# Also get RF feature importance
rf_imp = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                max_features=0.5, random_state=42, n_jobs=-1)
rf_imp.fit(X_tr, y_train)
rf_importance = rf_imp.feature_importances_
rf_sorted_idx = np.argsort(rf_importance)[::-1]

print(f'\n  Top 20 features by RF importance:')
for rank, idx in enumerate(rf_sorted_idx[:20]):
    print(f'    {rank+1:2d}. {feature_names[idx]:25s}  importance={rf_importance[idx]:.4f}')

# Also try XGB feature importance
xgb_imp = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                             reg_lambda=3.0, reg_alpha=1.0, random_state=42, verbosity=0)
xgb_imp.fit(X_tr, y_train)
xgb_importance = xgb_imp.feature_importances_
xgb_sorted_idx = np.argsort(xgb_importance)[::-1]

print(f'\n  Top 20 features by XGB importance:')
for rank, idx in enumerate(xgb_sorted_idx[:20]):
    print(f'    {rank+1:2d}. {feature_names[idx]:25s}  importance={xgb_importance[idx]:.4f}')

# =================================================================
#  CONCEPT 2: Simple Regularized Models (Paper Section 2-3)
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 2: L2-Regularized Models (Paper: Logistic + L2)')
print(' Adapted: Ridge/Lasso/ElasticNet regression for seed prediction')
print('='*60)

# Test various regularized linear models with LOSO
models_to_test = []

# Ridge with various alphas
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
    models_to_test.append(('Ridge', alpha, lambda Xtr, ytr, Xte, a=alpha: (
        Ridge(alpha=a).fit(StandardScaler().fit_transform(Xtr), ytr).predict(
            StandardScaler().fit(Xtr).transform(Xte)
        )
    )))

# Lasso  
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    models_to_test.append(('Lasso', alpha, lambda Xtr, ytr, Xte, a=alpha: (
        Lasso(alpha=a, max_iter=5000).fit(StandardScaler().fit_transform(Xtr), ytr).predict(
            StandardScaler().fit(Xtr).transform(Xte)
        )
    )))

# ElasticNet
for alpha in [0.5, 1.0, 5.0]:
    for l1r in [0.2, 0.5, 0.8]:
        models_to_test.append((f'EN(l1={l1r})', alpha, lambda Xtr, ytr, Xte, a=alpha, l=l1r: (
            ElasticNet(alpha=a, l1_ratio=l, max_iter=5000).fit(
                StandardScaler().fit_transform(Xtr), ytr).predict(
                StandardScaler().fit(Xtr).transform(Xte)
            )
        )))

print(f'  Testing {len(models_to_test)} regularized linear models...')
best_linear_rmse = 999
best_linear_name = ''
best_linear_fn = None
linear_results = []

for name, alpha, pfn in models_to_test:
    rmse, exact, rho, _ = loso_eval(pfn)
    linear_results.append((rmse, exact, rho, name, alpha))
    if rmse < best_linear_rmse:
        best_linear_rmse = rmse
        best_linear_name = f'{name}(α={alpha})'
        best_linear_fn = pfn
        print(f'    NEW BEST: {best_linear_name} LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}')

linear_results.sort(key=lambda x: x[0])
print(f'\n  Best linear: {best_linear_name} RMSE={best_linear_rmse:.4f}')
print(f'  Top 5:')
for rmse, exact, rho, name, alpha in linear_results[:5]:
    print(f'    {name}(α={alpha}): RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}')

# =================================================================
#  CONCEPT 3: Feature Selection + Model (Paper Section 4)
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 3: Top-K Feature Selection')
print(' (Paper: Kept only 4 features with |coef| > threshold)')
print('='*60)

# Test selecting top K features (by Ridge coefficient magnitude)
best_topk_rmse = 999
best_topk = 0
best_topk_fn = None

# Build a combined importance score (avg rank from Ridge, RF, XGB)
ranks_ridge = np.argsort(np.argsort(-coef_importance))  # 0=best
ranks_rf = np.argsort(np.argsort(-rf_importance))
ranks_xgb = np.argsort(np.argsort(-xgb_importance))
avg_rank = (ranks_ridge + ranks_rf + ranks_xgb) / 3
combined_sorted = np.argsort(avg_rank)

print(f'\n  Top 20 features by combined importance rank:')
for rank, idx in enumerate(combined_sorted[:20]):
    print(f'    {rank+1:2d}. {feature_names[idx]:25s}  '
          f'Ridge={ranks_ridge[idx]+1:2d}  RF={ranks_rf[idx]+1:2d}  '
          f'XGB={ranks_xgb[idx]+1:2d}  avg_rank={avg_rank[idx]:.1f}')

# Test top-K for K = 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 68
for K in [4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 68]:
    selected = combined_sorted[:K]
    
    # Use v40 pipeline with selected features
    def pred_topk(Xtr, ytr, Xte, sel=selected):
        Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]
        xpreds = []
        for seed in SEEDS:
            m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                  reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
            m.fit(Xtr_s, ytr); xpreds.append(m.predict(Xte_s))
        xgb_avg = np.mean(xpreds, axis=0)
        sc = StandardScaler(); rm = Ridge(alpha=5.0)
        rm.fit(sc.fit_transform(Xtr_s), ytr); rp = rm.predict(sc.transform(Xte_s))
        return 0.70 * xgb_avg + 0.30 * rp
    
    rmse, exact, rho, _ = loso_eval(pred_topk)
    marker = ' ***' if rmse < best_topk_rmse else ''
    if rmse < best_topk_rmse:
        best_topk_rmse = rmse
        best_topk = K
        best_topk_fn = pred_topk
    print(f'    K={K:3d}: LOSO-RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}{marker}')

print(f'\n  Best top-K: K={best_topk} RMSE={best_topk_rmse:.4f}')

# =================================================================
#  CONCEPT 4: Monte Carlo Simulation (Paper Section 3)
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 4: Monte Carlo Seed Assignment')
print(' (Paper: "Monte Carlo simulations were run to simulate tournament")')
print('='*60)

# Use v40 predictor, compare Hungarian vs Monte Carlo assignment
def pred_v40(Xtr, ytr, Xte):
    xpreds = []
    for seed in SEEDS:
        m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                              reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
        m.fit(Xtr, ytr); xpreds.append(m.predict(Xte))
    xgb_avg = np.mean(xpreds, axis=0)
    sc = StandardScaler(); rm = Ridge(alpha=5.0)
    rm.fit(sc.fit_transform(Xtr), ytr); rp = rm.predict(sc.transform(Xte))
    return 0.70 * xgb_avg + 0.30 * rp

rmse_h, exact_h, rho_h, _ = loso_eval(pred_v40, assign_fn='hungarian')
print(f'  Hungarian:    RMSE={rmse_h:.4f} ({exact_h}/{n_tr}) ρ={rho_h:.4f}')

rmse_mc, exact_mc, rho_mc, _ = loso_eval(pred_v40, assign_fn='monte_carlo')
print(f'  Monte Carlo:  RMSE={rmse_mc:.4f} ({exact_mc}/{n_tr}) ρ={rho_mc:.4f}')

# Test MC with different temperatures
best_mc_rmse = rmse_h  # Start from Hungarian baseline
best_mc_temp = None
for temp in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
    def mc_assign_temp(scores, seasons, avail, t=temp):
        return monte_carlo_assign(scores, seasons, avail, n_sims=500, temp=t)
    
    loso_assigned = np.zeros(n_tr, dtype=int)
    for hold in folds:
        tr_mask = train_seasons != hold
        te_mask = train_seasons == hold
        pred = pred_v40(X_tr[tr_mask], y_train[tr_mask], X_tr[te_mask])
        av = {hold: list(range(1, 69))}
        loso_assigned[te_mask] = mc_assign_temp(pred, train_seasons[te_mask], av)
    
    rmse = np.sqrt(np.mean((loso_assigned - y_train.astype(int))**2))
    exact = int(np.sum(loso_assigned == y_train.astype(int)))
    rho, _ = spearmanr(loso_assigned, y_train.astype(int))
    marker = ' ***' if rmse < best_mc_rmse else ''
    if rmse < best_mc_rmse:
        best_mc_rmse = rmse; best_mc_temp = temp
    print(f'    MC temp={temp}: RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}{marker}')

# =================================================================
#  CONCEPT 5: Spearman Correlation Analysis
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 5: Spearman Rank Correlation Analysis')
print(' (Paper: "A Spearman rank correlation coefficient was utilized")')
print('='*60)

# Run LOSO for all promising models and compare Spearman
print(f'\n  {"Model":<25} {"LOSO-RMSE":>10} {"Exact":>7} {"Spearman":>10}')
print(f'  {"-"*25} {"-"*10} {"-"*7} {"-"*10}')

rmse, exact, rho, details = loso_eval(pred_v40)
print(f'  {"v40 (XGB+Ridge)":<25} {rmse:>10.4f} {exact:>4}/{n_tr:3d} {rho:>10.4f}')
for s, n, ex, rm, r in details:
    print(f'    {s}: RMSE={rm:.3f} exact={ex}/{n} ρ={r:.4f}')

if best_linear_fn:
    rmse, exact, rho, details = loso_eval(best_linear_fn)
    print(f'  {best_linear_name:<25} {rmse:>10.4f} {exact:>4}/{n_tr:3d} {rho:>10.4f}')

if best_topk_fn:
    rmse, exact, rho, details = loso_eval(best_topk_fn)
    print(f'  {f"Top-{best_topk} features":<25} {rmse:>10.4f} {exact:>4}/{n_tr:3d} {rho:>10.4f}')

# =================================================================
#  CONCEPT 6: Combined — Best from each concept
# =================================================================
print('\n' + '='*60)
print(' CONCEPT 6: Combined Approaches')
print('='*60)

# Try: Feature-selected model + Ridge blend + MC assignment
# 1. Top-K features + different models
best_comb_rmse = 999
best_comb_label = ''
best_comb_fn = None

for K in [best_topk, 68]:
    selected = combined_sorted[:K]
    # RF + Ridge blend on selected features
    for rf_cfg in [{'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 2, 'max_features': 0.5}]:
        for ra in [5.0, 10.0]:
            for rw in [0.2, 0.3, 0.35]:
                def pred_comb_rf(Xtr, ytr, Xte, sel=selected, c=rf_cfg, r_a=ra, r_w=rw):
                    Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]
                    m = RandomForestRegressor(**c, random_state=42, n_jobs=-1)
                    m.fit(Xtr_s, ytr); rf_p = m.predict(Xte_s)
                    sc = StandardScaler(); rm = Ridge(alpha=r_a)
                    rm.fit(sc.fit_transform(Xtr_s), ytr); rp = rm.predict(sc.transform(Xte_s))
                    return (1 - r_w) * rf_p + r_w * rp
                
                rmse, exact, rho, _ = loso_eval(pred_comb_rf)
                if rmse < best_comb_rmse:
                    best_comb_rmse = rmse
                    best_comb_label = f'RF+Ridge K={K} ra={ra} rw={rw}'
                    best_comb_fn = pred_comb_rf
                    print(f'    NEW BEST: {best_comb_label} RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}')
    
    # XGB + RF + Ridge triple blend on selected features
    for rf_w in [0.1, 0.2, 0.3]:
        for ridge_w in [0.15, 0.25, 0.30]:
            def pred_triple(Xtr, ytr, Xte, sel=selected, rfw=rf_w, rdw=ridge_w):
                Xtr_s, Xte_s = Xtr[:, sel], Xte[:, sel]
                xgb_w = 1.0 - rfw - rdw
                # XGB
                xpreds = []
                for seed in SEEDS:
                    m = xgb.XGBRegressor(n_estimators=700, max_depth=5, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                          reg_lambda=3.0, reg_alpha=1.0, random_state=seed, verbosity=0)
                    m.fit(Xtr_s, ytr); xpreds.append(m.predict(Xte_s))
                xgb_p = np.mean(xpreds, axis=0)
                # RF
                m_rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2,
                                              max_features=0.5, random_state=42, n_jobs=-1)
                m_rf.fit(Xtr_s, ytr); rf_p = m_rf.predict(Xte_s)
                # Ridge  
                sc = StandardScaler(); rm = Ridge(alpha=5.0)
                rm.fit(sc.fit_transform(Xtr_s), ytr); rp = rm.predict(sc.transform(Xte_s))
                return xgb_w * xgb_p + rfw * rf_p + rdw * rp
            
            rmse, exact, rho, _ = loso_eval(pred_triple)
            if rmse < best_comb_rmse:
                best_comb_rmse = rmse
                best_comb_label = f'XGB+RF+Ridge K={K} rfw={rf_w} rdw={ridge_w}'
                best_comb_fn = pred_triple
                print(f'    NEW BEST: {best_comb_label} RMSE={rmse:.4f} ({exact}/{n_tr}) ρ={rho:.4f}')

print(f'\n  Best combined: {best_comb_label} RMSE={best_comb_rmse:.4f}')

# =================================================================
#  FINAL EVALUATION (LOSO-best model, single test eval)
# =================================================================
print('\n' + '='*60)
print(' FINAL EVALUATION')
print('='*60)

# Collect all LOSO scores
all_models = {
    'v40 (XGB+Ridge)': (rmse_h, pred_v40),
    best_linear_name: (best_linear_rmse, best_linear_fn),
    f'Top-{best_topk} feat': (best_topk_rmse, best_topk_fn),
    best_comb_label: (best_comb_rmse, best_comb_fn),
}

print(f'\n  {"Model":<30} {"LOSO-RMSE":>10}')
print(f'  {"-"*30} {"-"*10}')
for name, (rmse, _) in sorted(all_models.items(), key=lambda x: x[1][0]):
    marker = ' <-- BEST' if rmse == min(v[0] for v in all_models.values()) else ''
    print(f'  {name:<30} {rmse:>10.4f}{marker}')

# Pick LOSO-best
best_name = min(all_models, key=lambda k: all_models[k][0])
best_rmse, best_fn = all_models[best_name]

print(f'\n  Selected: {best_name} (LOSO-RMSE={best_rmse:.4f})')
print(f'  --- Single test evaluation ---')

pred_final = best_fn(X_tr, y_train, X_te)
assigned = hungarian(pred_final, test_seasons, test_avail)
test_exact = int(np.sum(assigned == test_gt))
test_rmse = np.sqrt(np.mean((assigned - test_gt)**2))
test_rho, _ = spearmanr(assigned, test_gt)

print(f'  Test: {test_exact}/91 exact, RMSE={test_rmse:.4f}, Spearman ρ={test_rho:.4f}')
print(f'  LOSO: {best_rmse:.4f}')
print(f'  Gap:  {abs(best_rmse - test_rmse):.4f}')

# Also eval v40 on test for comparison
p40 = pred_v40(X_tr, y_train, X_te)
a40 = hungarian(p40, test_seasons, test_avail)
e40 = int(np.sum(a40 == test_gt))
r40 = np.sqrt(np.mean((a40 - test_gt)**2))
rho40, _ = spearmanr(a40, test_gt)
print(f'\n  v40 test: {e40}/91, RMSE={r40:.4f}, Spearman ρ={rho40:.4f}')

if best_rmse < rmse_h:
    print(f'\n  *** IMPROVEMENT over v40 by LOSO! ***')
    print(f'      LOSO: {best_rmse:.4f} vs {rmse_h:.4f}')
    # Save
    sub_out = sub_df.copy()
    for i, ti in enumerate(tourn_idx):
        rid = test_df.iloc[ti]['RecordID']
        mask = sub_out['RecordID'] == rid
        if mask.any():
            sub_out.loc[mask, 'Overall Seed'] = int(assigned[i])
    sub_out.to_csv(os.path.join(DATA_DIR, 'final_submission.csv'), index=False)
    print('  Saved: final_submission.csv')
else:
    print(f'\n  v40 still best by LOSO. Not saving.')

print(f'\n  Total time: {time.time()-t0:.0f}s')
