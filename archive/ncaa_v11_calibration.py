"""
NCAA v11 — Calibration & Temperature Scaling
=============================================
Inspired by arxiv 2508.02725v1:
  "Forecasting NCAA Basketball Outcomes with Deep Learning"
  (Habib, Université Côte d'Azur, Aug 2025)

Key ideas from paper applied to our pairwise seed prediction:

1. PROBABILITY CALIBRATION (isotonic/Platt) on pairwise LR
   - Paper shows Brier-loss models produce better-calibrated probabilities
   - CalibratedClassifierCV with isotonic regression ≈ Brier optimization
   - Better calibration → more accurate pairwise scores → better Hungarian

2. TEMPERATURE SCALING of pairwise predictions
   - T < 1 → sharper/more confident probabilities
   - T > 1 → smoother/less confident probabilities
   - Fine-tune how confident the model is in pairwise comparisons

3. DUAL-LOSS ENSEMBLE (paper's main recommendation)
   - "LSTM+Brier for calibration + Transformer+BCE for ranking"
   - For us: blend standard LR (discriminative) with calibrated LR
   - Combine ranking accuracy with probability reliability

All tested on top of force-NET (best from v10c: maintains 56/91
Kaggle, improves LOSO from 3.678 to 3.584).
"""

import sys, os, time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import KNNImputer
import xgboost as xgb

from ncaa_2026_model import (
    load_data, build_features, build_pairwise_data,
    pairwise_score, hungarian, select_top_k_features
)

# ── Constants ──
KAGGLE_POWER = 0.15
USE_TOP_K = 25
GT_FILE = 'NCAA_Seed_Test_Set2.0.csv'


# ══════════════════════════════════════════════════════════
#  TEMPERATURE-SCALED PAIRWISE SCORING
# ══════════════════════════════════════════════════════════
def pairwise_score_temp(model, X_test, scaler, temperature):
    """Pairwise scoring with temperature-scaled probabilities.
    
    Temperature controls prediction confidence:
      T < 1.0 → sharper (more confident)
      T = 1.0 → standard (equivalent to normal pairwise_score)
      T > 1.0 → smoother (less confident, more uniform)
    """
    n = len(X_test)
    scores = np.zeros(n)
    for i in range(n):
        diffs = X_test[i] - X_test
        if scaler is not None:
            diffs = scaler.transform(diffs)
        
        # Get raw logits
        if hasattr(model, 'decision_function'):
            logits = model.decision_function(diffs)
        else:
            # For calibrated models / XGB: convert probabilities to logits
            probs_raw = model.predict_proba(diffs)[:, 1]
            eps = 1e-7
            probs_raw = np.clip(probs_raw, eps, 1 - eps)
            logits = np.log(probs_raw / (1 - probs_raw))
        
        # Temperature-scaled sigmoid
        probs = 1.0 / (1.0 + np.exp(-logits / temperature))
        probs[i] = 0
        scores[i] = probs.sum()
    return np.argsort(np.argsort(-scores)).astype(float) + 1.0


# ══════════════════════════════════════════════════════════
#  PREDICTION FUNCTION WITH CALIBRATION/TEMPERATURE OPTIONS
# ══════════════════════════════════════════════════════════
def predict_blend(X_tr, y_tr, X_te, s_tr, top_k_idx,
                  c1=5.0, c3=0.5, w1=0.64, w3=0.28, w4=0.08,
                  cal_method=None, cal_components=(),
                  temperature=None, temp_components=()):
    """
    v6-style blend with optional calibration and temperature scaling.
    
    cal_method: 'isotonic' or 'sigmoid' (or None)
    cal_components: tuple of component numbers (1, 3, 4) to calibrate
    temperature: float (or None for standard)
    temp_components: tuple of component numbers to temperature-scale
    """
    pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    
    # ── Component 1: PW-LogReg C=c1 on full features ──
    base_lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    if 1 in cal_components and cal_method:
        lr1 = CalibratedClassifierCV(base_lr1, cv=3, method=cal_method)
        lr1.fit(pw_X_sc, pw_y)
    else:
        lr1 = base_lr1
        lr1.fit(pw_X_sc, pw_y)
    
    if 1 in temp_components and temperature:
        s1 = pairwise_score_temp(lr1, X_te, sc, temperature)
    else:
        s1 = pairwise_score(lr1, X_te, sc)
    
    # ── Component 3: PW-LogReg C=c3 on top-K features ──
    X_tr_k = X_tr[:, top_k_idx]
    X_te_k = X_te[:, top_k_idx]
    pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
    sck = StandardScaler()
    pw_Xk_sc = sck.fit_transform(pw_Xk)
    
    base_lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    if 3 in cal_components and cal_method:
        lr3 = CalibratedClassifierCV(base_lr3, cv=3, method=cal_method)
        lr3.fit(pw_Xk_sc, pw_yk)
    else:
        lr3 = base_lr3
        lr3.fit(pw_Xk_sc, pw_yk)
    
    if 3 in temp_components and temperature:
        s3 = pairwise_score_temp(lr3, X_te_k, sck, temperature)
    else:
        s3 = pairwise_score(lr3, X_te_k, sck)
    
    # ── Component 4: PW-XGB on full features ──
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    if 4 in cal_components and cal_method:
        cal_xgb = CalibratedClassifierCV(xgb_clf, cv=3, method=cal_method)
        cal_xgb.fit(pw_X_sc, pw_y)
        xgb_model = cal_xgb
    else:
        xgb_clf.fit(pw_X_sc, pw_y)
        xgb_model = xgb_clf
    
    if 4 in temp_components and temperature:
        s4 = pairwise_score_temp(xgb_model, X_te, sc, temperature)
    else:
        s4 = pairwise_score(xgb_model, X_te, sc)
    
    return w1 * s1 + w3 * s3 + w4 * s4


# ══════════════════════════════════════════════════════════
#  DUAL-ENSEMBLE PREDICTION (paper's key recommendation)
# ══════════════════════════════════════════════════════════
def predict_dual_ensemble(X_tr, y_tr, X_te, s_tr, top_k_idx,
                          alpha=0.5, c1=5.0, c3=0.5,
                          w1=0.64, w3=0.28, w4=0.08):
    """
    Dual-loss ensemble inspired by paper's recommendation:
    "Combine discriminative model (BCE/ranking) with calibrated model (Brier)"
    
    For component 1 (dominant 64% weight):
      - Standard LR → discriminative ranking scores
      - Isotonic-calibrated LR → well-calibrated probability scores
      - Blend: alpha * standard + (1-alpha) * calibrated
    
    Components 3 and 4 remain standard.
    """
    pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    
    # Component 1: DUAL — standard + calibrated
    lr1_std = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1_std.fit(pw_X_sc, pw_y)
    s1_std = pairwise_score(lr1_std, X_te, sc)
    
    base_lr1_cal = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1_cal = CalibratedClassifierCV(base_lr1_cal, cv=3, method='isotonic')
    lr1_cal.fit(pw_X_sc, pw_y)
    s1_cal = pairwise_score(lr1_cal, X_te, sc)
    
    s1 = alpha * s1_std + (1 - alpha) * s1_cal
    
    # Component 3: standard top-K
    X_tr_k = X_tr[:, top_k_idx]
    X_te_k = X_te[:, top_k_idx]
    pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
    sck = StandardScaler()
    pw_Xk_sc = sck.fit_transform(pw_Xk)
    lr3 = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3.fit(pw_Xk_sc, pw_yk)
    s3 = pairwise_score(lr3, X_te_k, sck)
    
    # Component 4: standard XGB
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_sc, pw_y)
    s4 = pairwise_score(xgb_clf, X_te, sc)
    
    return w1 * s1 + w3 * s3 + w4 * s4


# ══════════════════════════════════════════════════════════
#  DUAL ENSEMBLE ON COMPONENT 3 (calibrated top-K)
# ══════════════════════════════════════════════════════════
def predict_dual_comp3(X_tr, y_tr, X_te, s_tr, top_k_idx,
                       alpha=0.5, c1=5.0, c3=0.5,
                       w1=0.64, w3=0.28, w4=0.08):
    """
    Dual-loss on component 3 instead of component 1.
    Component 3 is the regularized model (C=0.5, top-K features) — 
    this might benefit MORE from calibration since it's already smoothed.
    """
    pw_X, pw_y = build_pairwise_data(X_tr, y_tr, s_tr)
    sc = StandardScaler()
    pw_X_sc = sc.fit_transform(pw_X)
    
    # Component 1: standard
    lr1 = LogisticRegression(C=c1, penalty='l2', max_iter=2000, random_state=42)
    lr1.fit(pw_X_sc, pw_y)
    s1 = pairwise_score(lr1, X_te, sc)
    
    # Component 3: DUAL — standard + calibrated on top-K
    X_tr_k = X_tr[:, top_k_idx]
    X_te_k = X_te[:, top_k_idx]
    pw_Xk, pw_yk = build_pairwise_data(X_tr_k, y_tr, s_tr)
    sck = StandardScaler()
    pw_Xk_sc = sck.fit_transform(pw_Xk)
    
    lr3_std = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3_std.fit(pw_Xk_sc, pw_yk)
    s3_std = pairwise_score(lr3_std, X_te_k, sck)
    
    base_lr3_cal = LogisticRegression(C=c3, penalty='l2', max_iter=2000, random_state=42)
    lr3_cal = CalibratedClassifierCV(base_lr3_cal, cv=3, method='isotonic')
    lr3_cal.fit(pw_Xk_sc, pw_yk)
    s3_cal = pairwise_score(lr3_cal, X_te_k, sck)
    
    s3 = alpha * s3_std + (1 - alpha) * s3_cal
    
    # Component 4: standard XGB
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=3.0, reg_alpha=1.0, min_child_weight=5,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_clf.fit(pw_X_sc, pw_y)
    s4 = pairwise_score(xgb_clf, X_te, sc)
    
    return w1 * s1 + w3 * s3 + w4 * s4


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    
    print('=' * 60)
    print(' NCAA v11 — CALIBRATION & TEMPERATURE SCALING')
    print(' (Inspired by arxiv 2508.02725v1)')
    print('=' * 60)
    
    # ── Load data (same pattern as v10c) ──
    all_df, labeled, _, train_df, test_df, sub_df, GT = load_data()
    tourn_rids = set(labeled['RecordID'].values)
    context_df = pd.concat([
        train_df.drop(columns=['Overall Seed'], errors='ignore'),
        test_df.drop(columns=['Overall Seed'], errors='ignore')
    ], ignore_index=True)
    
    feat_df = build_features(labeled, context_df, labeled, tourn_rids)
    feature_names = list(feat_df.columns)
    
    y = labeled['Overall Seed'].values.astype(float)
    seasons = labeled['Season'].values.astype(str)
    record_ids = labeled['RecordID'].values.astype(str)
    folds = sorted(set(seasons))
    n = len(y)
    
    X_raw = np.where(np.isinf(feat_df.values.astype(np.float64)),
                     np.nan, feat_df.values.astype(np.float64))
    imp = KNNImputer(n_neighbors=10, weights='distance')
    X = imp.fit_transform(X_raw)
    
    # Feature index map for forced features
    fi = {f: i for i, f in enumerate(feature_names)}
    
    # Test mask for Kaggle evaluation
    test_rids = set(GT.keys())
    test_mask = np.array([rid in test_rids for rid in record_ids])
    
    # ── Evaluation functions (same pattern as v10c) ──
    results = []
    
    def kaggle_eval(pred_fn):
        """Evaluate against Kaggle GT teams."""
        test_assigned = np.zeros(n, dtype=int)
        for hold in folds:
            smask = (seasons == hold)
            test_in_season = test_mask & smask
            if test_in_season.sum() == 0:
                continue
            train_mask = ~test_in_season
            scores = pred_fn(X[train_mask], y[train_mask], X[smask], seasons[train_mask], feature_names)
            season_idx = np.where(smask)[0]
            for i, gi in enumerate(season_idx):
                if not test_mask[gi]:
                    scores[i] = y[gi]
            avail = {hold: list(range(1, 69))}
            assigned = hungarian(scores, seasons[smask], avail, power=KAGGLE_POWER)
            for i, gi in enumerate(season_idx):
                if test_mask[gi]:
                    test_assigned[gi] = assigned[i]
        gt_arr = y[test_mask].astype(int)
        pred_arr = test_assigned[test_mask]
        exact = int((pred_arr == gt_arr).sum())
        rmse = np.sqrt(np.mean((pred_arr - gt_arr) ** 2))
        total = int(test_mask.sum())
        return exact, total, rmse
    
    def loso_eval(pred_fn):
        """Leave-one-season-out RMSE evaluation."""
        fold_rmses = []
        for hold in folds:
            tr = seasons != hold
            te = seasons == hold
            scores = pred_fn(X[tr], y[tr], X[te], seasons[tr], feature_names)
            avail = {hold: list(range(1, 69))}
            assigned = hungarian(scores, seasons[te], avail, power=KAGGLE_POWER)
            rmse = np.sqrt(np.mean((assigned - y[te].astype(int)) ** 2))
            fold_rmses.append(rmse)
        return np.mean(fold_rmses) + 0.5 * np.std(fold_rmses)
    
    def evaluate(tag, pred_fn):
        """Run both Kaggle + LOSO evaluation."""
        exact, total, kaggle_rmse = kaggle_eval(pred_fn)
        loso_rmse = loso_eval(pred_fn)
        return exact, total, kaggle_rmse, loso_rmse
    
    def show(tag, exact, total, rmse, loso_rmse, ref_exact=56, ref_rmse=2.474, ref_loso=3.6776):
        kaggle_ok = exact >= ref_exact and rmse <= ref_rmse + 0.001
        loso_ok = loso_rmse < ref_loso
        
        star = ''
        if kaggle_ok and loso_ok:
            star = '★★'
        elif loso_ok and exact >= ref_exact:
            star = '=★'
        elif loso_ok:
            star = ' ★'
        
        print(f'  {tag:<45s} Kaggle={exact}/{total} RMSE={rmse:.4f} {"↑" if exact > ref_exact else "↓"}  '
              f'LOSO={loso_rmse:.4f} {"↑" if loso_rmse < ref_loso else "↓"} {star}')
        results.append((tag, exact, total, rmse, loso_rmse))
    
    # ══════════════════════════════════════════════════════
    #  Helper: build top-K with forced features
    # ══════════════════════════════════════════════════════
    def get_top_k_forced(X_tr, y_tr, feature_names, forced_feats, total_k=25):
        forced_idx = [fi[f] for f in forced_feats if f in fi]
        if forced_idx:
            auto_k = max(total_k - len(forced_idx), 5)
            top_k_auto = select_top_k_features(X_tr, y_tr, feature_names, k=auto_k)[0]
            combined = list(forced_idx)
            for idx in top_k_auto:
                if idx not in combined:
                    combined.append(idx)
            return combined[:total_k]
        else:
            return list(select_top_k_features(X_tr, y_tr, feature_names, k=total_k)[0])
    
    # ══════════════════════════════════════════════════════
    #  PART 1: BASELINES
    # ══════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 1: Baselines')
    print('─' * 60)
    
    # v6 baseline
    def v6_baseline(X_tr, y_tr, X_te, s_tr, fn):
        tk = get_top_k_forced(X_tr, y_tr, fn, [], 25)
        return predict_blend(X_tr, y_tr, X_te, s_tr, tk)
    
    ex, tot, rmse, loso = evaluate('v6-baseline', v6_baseline)
    show('v6-baseline', ex, tot, rmse, loso)
    
    # force-NET baseline (best from v10c)
    def force_net(X_tr, y_tr, X_te, s_tr, fn):
        tk = get_top_k_forced(X_tr, y_tr, fn, ['NET Rank'], 25)
        return predict_blend(X_tr, y_tr, X_te, s_tr, tk)
    
    ex, tot, rmse, loso = evaluate('force-NET', force_net)
    show('force-NET', ex, tot, rmse, loso)
    
    # ══════════════════════════════════════════════════════
    #  PART 2: ISOTONIC CALIBRATION (Brier-like optimization)
    # ══════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 2: Isotonic Calibration (Brier-like, per paper §5.3)')
    print('─' * 60)
    
    cal_configs = [
        ('force-NET+iso-LR1',       'isotonic', (1,)),
        ('force-NET+iso-LR3',       'isotonic', (3,)),
        ('force-NET+iso-LR1+LR3',   'isotonic', (1, 3)),
        ('force-NET+iso-all',       'isotonic', (1, 3, 4)),
        ('force-NET+sig-LR1',       'sigmoid',  (1,)),
        ('force-NET+sig-LR1+LR3',   'sigmoid',  (1, 3)),
        ('v6+iso-LR1',              'isotonic', (1,)),
        ('v6+iso-LR1+LR3',          'isotonic', (1, 3)),
    ]
    
    for tag, cal_method, cal_comps in cal_configs:
        forced = ['NET Rank'] if 'force-NET' in tag else []
        def make_fn(f=forced, cm=cal_method, cc=cal_comps):
            def fn(X_tr, y_tr, X_te, s_tr, fn_names):
                tk = get_top_k_forced(X_tr, y_tr, fn_names, f, 25)
                return predict_blend(X_tr, y_tr, X_te, s_tr, tk,
                                     cal_method=cm, cal_components=cc)
            return fn
        ex, tot, rmse, loso = evaluate(tag, make_fn())
        show(tag, ex, tot, rmse, loso)
    
    # ══════════════════════════════════════════════════════
    #  PART 3: TEMPERATURE SCALING
    # ══════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 3: Temperature Scaling (prediction confidence)')
    print('─' * 60)
    
    temp_configs = [
        ('force-NET+T=0.5',         0.5,  (1, 3)),
        ('force-NET+T=0.7',         0.7,  (1, 3)),
        ('force-NET+T=0.8',         0.8,  (1, 3)),
        ('force-NET+T=1.2',         1.2,  (1, 3)),
        ('force-NET+T=1.5',         1.5,  (1, 3)),
        ('force-NET+T=2.0',         2.0,  (1, 3)),
        ('force-NET+T=0.7-LR1only', 0.7,  (1,)),
        ('force-NET+T=0.7-LR3only', 0.7,  (3,)),
        ('v6+T=0.7',                0.7,  (1, 3)),
        ('v6+T=0.8',                0.8,  (1, 3)),
    ]
    
    for tag, temp, temp_comps in temp_configs:
        forced = ['NET Rank'] if 'force-NET' in tag else []
        def make_fn(f=forced, t=temp, tc=temp_comps):
            def fn(X_tr, y_tr, X_te, s_tr, fn_names):
                tk = get_top_k_forced(X_tr, y_tr, fn_names, f, 25)
                return predict_blend(X_tr, y_tr, X_te, s_tr, tk,
                                     temperature=t, temp_components=tc)
            return fn
        ex, tot, rmse, loso = evaluate(tag, make_fn())
        show(tag, ex, tot, rmse, loso)
    
    # ══════════════════════════════════════════════════════
    #  PART 4: DUAL-LOSS ENSEMBLE (paper's main recommendation)
    # ══════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 4: Dual-Loss Ensemble (discriminative + calibrated)')
    print('         Paper: "LSTM+Brier + Transformer+BCE"')
    print('─' * 60)
    
    # Dual ensemble on Component 1 (dominant 64% weight)
    dual_c1_configs = [
        ('force-NET+dual1-α=0.7',  0.7),   # 70% standard, 30% calibrated
        ('force-NET+dual1-α=0.5',  0.5),   # 50/50
        ('force-NET+dual1-α=0.3',  0.3),   # 30% standard, 70% calibrated
        ('force-NET+dual1-α=0.8',  0.8),
        ('v6+dual1-α=0.5',         0.5),
    ]
    
    for tag, alpha in dual_c1_configs:
        forced = ['NET Rank'] if 'force-NET' in tag else []
        def make_fn(f=forced, a=alpha):
            def fn(X_tr, y_tr, X_te, s_tr, fn_names):
                tk = get_top_k_forced(X_tr, y_tr, fn_names, f, 25)
                return predict_dual_ensemble(X_tr, y_tr, X_te, s_tr, tk, alpha=a)
            return fn
        ex, tot, rmse, loso = evaluate(tag, make_fn())
        show(tag, ex, tot, rmse, loso)
    
    # Dual ensemble on Component 3 (28% weight, regularized)
    dual_c3_configs = [
        ('force-NET+dual3-α=0.7',  0.7),
        ('force-NET+dual3-α=0.5',  0.5),
        ('force-NET+dual3-α=0.3',  0.3),
    ]
    
    for tag, alpha in dual_c3_configs:
        def make_fn(a=alpha):
            def fn(X_tr, y_tr, X_te, s_tr, fn_names):
                tk = get_top_k_forced(X_tr, y_tr, fn_names, ['NET Rank'], 25)
                return predict_dual_comp3(X_tr, y_tr, X_te, s_tr, tk, alpha=a)
            return fn
        ex, tot, rmse, loso = evaluate(tag, make_fn())
        show(tag, ex, tot, rmse, loso)
    
    # ══════════════════════════════════════════════════════
    #  PART 5: COMBINED — calibration + temperature
    # ══════════════════════════════════════════════════════
    print('\n' + '─' * 60)
    print(' PART 5: Combined Approaches')
    print('─' * 60)
    
    combo_configs = [
        # Calibrate component 3, temperature-scale component 1
        ('force-NET+iso3+T0.8-LR1',  'isotonic', (3,), 0.8, (1,)),
        ('force-NET+iso3+T0.7-LR1',  'isotonic', (3,), 0.7, (1,)),
        # Calibrate component 1, temperature-scale component 3
        ('force-NET+iso1+T0.8-LR3',  'isotonic', (1,), 0.8, (3,)),
        # Both calibrated + both temperature
        ('force-NET+iso-both+T0.8',   'isotonic', (1, 3), 0.8, (1, 3)),
    ]
    
    for tag, cm, cc, temp, tc in combo_configs:
        def make_fn(cm_=cm, cc_=cc, t_=temp, tc_=tc):
            def fn(X_tr, y_tr, X_te, s_tr, fn_names):
                tk = get_top_k_forced(X_tr, y_tr, fn_names, ['NET Rank'], 25)
                return predict_blend(X_tr, y_tr, X_te, s_tr, tk,
                                     cal_method=cm_, cal_components=cc_,
                                     temperature=t_, temp_components=tc_)
            return fn
        ex, tot, rmse, loso = evaluate(tag, make_fn())
        show(tag, ex, tot, rmse, loso)
    
    # ══════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY — ALL VARIANTS')
    print('=' * 60)
    
    print(f'\n  {"Approach":<48s} {"Kaggle":>10s} {"RMSE":>8s} {"LOSO":>8s}   St')
    print(f'  {"─"*48} {"─"*10} {"─"*8} {"─"*8} {"─"*4}')
    
    ref_exact, ref_rmse, ref_loso = 56, 2.474, 3.6776
    
    for tag, exact, total, rmse, loso_rmse in results:
        kaggle_ok = exact >= ref_exact and rmse <= ref_rmse + 0.001
        loso_ok = loso_rmse < ref_loso
        
        star = ''
        if kaggle_ok and loso_ok:
            star = '=★'
        elif exact > ref_exact and loso_ok:
            star = '★★'
        elif loso_ok and exact >= ref_exact:
            star = '=★'
        elif loso_ok:
            star = ' ★'
        
        marker = ' →' if tag == 'v6-baseline' else '  '
        print(f'{marker}{tag:<48s} {exact}/{total:>3d}   {rmse:7.4f}  {loso_rmse:7.4f}  {star}')
    
    # Show approaches that beat both baselines
    print(f'\n  ── Approaches with Kaggle ≥56/91, RMSE≤2.474, and LOSO < {ref_loso:.4f}: ──')
    winners = [(t, e, tot, r, l) for t, e, tot, r, l in results
               if e >= 56 and r <= 2.4741 and l < ref_loso]
    winners.sort(key=lambda x: x[4])  # sort by LOSO
    for t, e, tot, r, l in winners:
        print(f'    {t}: Kaggle={e}/{tot} RMSE={r:.4f} LOSO={l:.4f} (↑{ref_loso - l:.4f})')
    
    # Also show anything that beats 56 on Kaggle (even if RMSE slightly different)
    print(f'\n  ── Approaches with Kaggle > 56/91: ──')
    above = [(t, e, tot, r, l) for t, e, tot, r, l in results if e > 56]
    above.sort(key=lambda x: (-x[1], x[4]))
    for t, e, tot, r, l in above:
        print(f'    {t}: Kaggle={e}/{tot} RMSE={r:.4f} LOSO={l:.4f}')
    
    if not above:
        print('    (none)')
    
    elapsed = time.time() - t0
    print(f'\n  Time: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
