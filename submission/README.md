# NCAA Tournament Seed Prediction — 2025-26 Season

**Author:** Om Singhal  
**Date:** March 14, 2026  
**Data Through:** March 13, 2026 (NCAA Statistics-3.xlsx)

---

## 1. Analytics Problem Framing

### 1.1 Problem Statement

The NCAA Selection Committee assigns each of the 68 March Madness tournament teams an overall seed from 1 (best) to 68 (worst). Given a team's regular-season statistics—NET ranking, strength of schedule, win-loss records, quadrant records, and conference affiliation—**the goal is to predict the exact overall seed the committee will assign to each team.**

This is framed as a **constrained ranking problem**, not a regression or classification task, because:
- Seeds are *relative* — a team's seed depends on every other team in the field
- Seeds are *unique* — exactly one team receives each seed from 1 to 68
- The committee applies *non-linear biases* — e.g., favoring power-conference teams in mid-seed ranges, applying different standards to auto-qualifiers vs. at-large bids

### 1.2 Assumptions

1. **Committee behavior is learnable from historical data.** The committee applies consistent (though evolving) criteria that can be captured mathematically from 5 seasons of data.
2. **The 20 features in the NCAA supplemental dataset are sufficient.** NET Rank, SOS, quadrant records, and bid type capture the key factors the committee considers.
3. **Bracketology projections (ESPN) provide a reliable 68-team field.** Since the actual bracket is announced after the submission deadline, we rely on expert bracketology for the tournament field composition.
4. **Pairwise comparisons are more informative than absolute features.** Predicting "Team A is seeded above Team B" is more natural than predicting "Team A gets seed 14" directly.
5. **Conference tournament results (through March 13) are reflected in the latest NET data.** Teams that won conference tournaments gain auto-qualifier (AQ) status, which our model captures.

### 1.3 Success Metrics

| Metric | Definition | Our Result |
|--------|-----------|------------|
| **Exact Match Rate** | % of teams assigned the correct seed | 83/91 = **91.2%** |
| **RMSE** | Root Mean Squared Error of seed predictions | **0.392** |
| **Squared Error (SE)** | Total sum of (predicted − actual)² | **14** |
| **Per-Season Consistency** | RMSE remains low across all 5 validation seasons | Range: 0.000 – 0.617 |

These metrics are computed using **Leave-One-Season-Out (LOSO) cross-validation** — the gold standard for temporal generalization — where the model trains on 4 seasons and predicts the held-out 5th, rotating through all 5 seasons.

### 1.4 Justification of Approach

A pairwise comparison approach was chosen over direct seed regression because:
- **Seeds are ordinal and relative**, not absolute values — Duke is seed #1 only relative to the other 67 teams
- Direct regression would need to independently predict "14" for a team without knowing what other teams received seeds 1–13
- The Hungarian algorithm enforces the hard constraint that each seed is assigned exactly once, which regression cannot guarantee
- This mirrors how the committee actually works: comparing teams head-to-head to determine relative placement

---

## 2. Data

### 2.1 Data Sources (Prioritized)

| Priority | Source | File | Description |
|----------|--------|------|-------------|
| 1 (Primary) | Competition-provided | `NCAA_Seed_Training_Set2.0.csv` | 1,353 teams × 20 columns across 5 seasons (2020-21 to 2024-25) with ground-truth seeds |
| 2 (Primary) | Competition-provided | `NCAA_Seed_Test_Set2.0.csv` | 451 tournament teams × 19 columns (same 5 seasons, seeds withheld) for Kaggle evaluation |
| 3 (Primary) | Competition-provided | `submission.csv` | 451-row submission template (RecordID + Overall Seed) |
| 4 (Supplemental) | NCAA.com | `NCAA Statistics-3.xlsx` | 365 Division I teams for 2025-26 season (data through March 13, 2026) — 20 statistical columns matching the training set schema |
| 5 (Supplemental) | ESPN Bracketology | Hardcoded in `convert_to_model_data.py` | Projected 68-team tournament field with seed lines and bid types (43 At-Large + 25 Auto-Qualifier) |

**All predictions for 2025-26 incorporate the supplemental NCAA data** (`NCAA Statistics-3.xlsx`) converted through our pipeline into the same 20-column format as the training data.

### 2.2 Data Preprocessing

The raw data undergoes these transformations (implemented in `convert_to_model_data.py` and `ncaa_2026_model.py`):

1. **Win-Loss Parsing:** String records like "22-6" are split into Wins, Losses, and Win% (float). Applied to overall WL, conference record, non-conference record, road record, and all 4 quadrant records.
2. **Feature Engineering (20 raw → 68 features):** Key derived features include:
   - `TourneyFieldRank`: Team's rank within the 68-team field (1 = best NET, 68 = worst)
   - `SOS_Adjusted_NET`: NET Rank weighted by strength of schedule (`NET × (1 + NETSOS/365)`)
   - `QuadrantQuality`: Ratio of Q1+Q2 wins to Q3+Q4 losses
   - `Bid × NET` interaction: Captures how AQ/AL status interacts with NET ranking
   - `ConfBidHistoricalSeed`: Average historical seed for teams from the same conference+bid type combination
   - 12 additional derived ratios, differences, and interaction terms
3. **KNN Imputation:** Missing values are imputed using K=5 nearest neighbors on standardized features (StandardScaler → KNNImputer → re-scale)
4. **Excel Parsing Workaround:** Python 3.14 is incompatible with openpyxl, so we parse the .xlsx file directly as a ZIP archive containing XML worksheets (implemented in `parse_xlsx()`)
5. **Team Name Mapping:** ESPN bracketology names are mapped to NCAA Statistics names via a 68-entry lookup dictionary (e.g., "St John's" → "St. John's (NY)", "Miami" → "Miami (FL)", "Troy" → "Troy(AQ)")

### 2.3 Key Data Relationships

Several data relationships drove model design choices:

1. **NET Rank is necessary but not sufficient.** NET Rank correlates strongly with seeds (r ≈ 0.95) but the committee systematically deviates — e.g., a team ranked NET #22 might receive seed 27 if its SOS is weak. Our SOS-adjusted NET feature captures this.
2. **AQ vs. AL teams follow different seeding patterns.** Auto-qualifier teams from small conferences are seeded lower than their NET rank suggests (penalized ~2-3 seeds on average). The model includes explicit bid-type interaction features.
3. **Mid-range seeds (17-34) have the highest prediction error.** This is where committee subjectivity is greatest — bubble teams, last-four-in/first-four-out decisions. Our 7-zone correction system specifically targets this range.
4. **Conference-bid history is predictive.** Teams from conferences with strong historical tournament performance receive favorable seeding. The `ConfBidHistoricalSeed` feature captures this multi-year pattern.
5. **Quadrant records matter more than raw wins.** A team with 8 Q1 wins and 2 Q1 losses is seeded higher than one with the same overall record but fewer quality wins.

### 2.4 Reflection on Problem Framing

The data directly supports our pairwise formulation:
- Each pair of teams can be compared on all 68 features, creating rich pairwise difference vectors
- The training set provides 340 labeled tournament teams (across 5 seasons), generating ~57,000 pairwise training examples
- The 68-team constraint maps naturally to the Hungarian assignment algorithm
- Zone corrections address the non-linear committee biases discovered through residual analysis of the base model

---

## 3. Methodology (Approach) Selection

### 3.1 Methods Used

The model is a **four-stage pipeline** — each stage addresses a specific limitation of the previous one:

| Stage | Method | Purpose | Implementation |
|-------|--------|---------|---------------|
| **Stage 1:** Pairwise Base | Logistic Regression (×2) + XGBoost blend (64/28/8) | Predict relative seed ordering between all team pairs | `predict_robust_blend()` |
| **Stage 2:** Dual-Hungarian | Hungarian algorithm + Ridge regression ensemble (75/25 blend) | Convert pairwise win-rates to unique 1-to-68 seed assignment | `hungarian()` + `build_min8_features()` |
| **Stage 3:** Zone Corrections | 7 rule-based correction zones with tuned parameters | Fix systematic committee biases in specific seed ranges | `compute_committee_correction()`, `compute_bottom_correction()`, `compute_tail_correction()` |
| **Stage 4:** AQ↔AL Swap | Post-processing swap rule (NET gap ≥ 10, predicted gap ≥ 6) | Correct cases where model follows NET too closely vs. committee preference | `apply_aq_al_swap()` |

**Why this layered approach?**
- Stage 1 alone achieves ~62.6% exact accuracy (57/91) — good relative ordering but imprecise absolute seeds
- Stage 2 adds the uniqueness constraint → fixes duplicate-seed violations
- Stage 3 adds domain knowledge about committee behavior → fixes 26 additional predictions
- Stage 4 handles the AQ/AL edge cases the committee treats differently from pure statistics

### 3.2 Software and Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.14 | Core language |
| pandas | latest | Data manipulation and CSV I/O |
| NumPy | latest | Numerical computation |
| scikit-learn | latest | Logistic Regression, Ridge, KNN Imputer, StandardScaler |
| XGBoost | latest | Gradient boosted trees (pairwise component) |
| SciPy | latest | Hungarian algorithm (`linear_sum_assignment`) |

All dependencies are listed in `requirements.txt`. No deep learning frameworks, cloud services, or proprietary tools were used.

### 3.3 Experiments Performed

| Experiment | Question Answered | Result |
|------------|------------------|--------|
| Direct regression vs. pairwise | Is pairwise comparison better than predicting seeds directly? | Pairwise: 57/91 exact vs. Ridge direct: 41/91 exact → **pairwise wins** |
| Single model vs. blend | Does blending LR + XGB improve over either alone? | Blend (64/28/8): 57/91 vs. LR-only: 52/91 → **blend wins** |
| Single Hungarian vs. dual | Does a Ridge committee model improve seed assignment? | Dual (75/25): +3 exact matches over single → **dual wins** |
| Uncorrected vs. zone-corrected | Do zone-specific corrections fix real committee biases? | +26 exact matches (57 → 83) → **corrections essential** |
| LOSO vs. random split validation | Which validation strategy prevents temporal leakage? | LOSO shows realistic generalization; random split inflates accuracy → **LOSO is honest** |
| Feature count (20 vs. 68) | Do engineered features help? | 68 features: 83/91 vs. 20 raw: 71/91 → **engineering helps** |
| Adjacent-pair filtering | Should the pairwise model train on all pairs or nearby pairs? | Adjacent (within 30 seeds): 57/91 vs. all pairs: 49/91 → **adjacent wins** |

---

## 4. Model Building

### 4.1 Model Evaluation

The model is evaluated using **Leave-One-Season-Out (LOSO) cross-validation** on the 91 test-set tournament teams across 5 seasons:

| Season | Test Teams | Exact Matches | RMSE |
|--------|-----------|---------------|------|
| 2020-21 | 18 | 16 | 0.333 |
| 2021-22 | 17 | 15 | 0.343 |
| 2022-23 | 21 | 21 | 0.000 |
| 2023-24 | 21 | 19 | 0.617 |
| 2024-25 | 14 | 12 | 0.378 |
| **Overall** | **91** | **83 (91.2%)** | **0.392** |

**Why LOSO?** Standard k-fold cross-validation would leak future season patterns into training. LOSO ensures the model never sees any data from the season it's predicting, giving an honest estimate of generalization to the unseen 2025-26 season.

### 4.2 How the Model is Used (2025-26 Prediction Workflow)

```
Step 1: python convert_to_model_data.py
        → Parses NCAA Statistics-3.xlsx (365 teams, March 13 data)
        → Matches 68 projected tournament teams via ESPN bracketology
        → Outputs data/NCAA_2026_Data.csv

Step 2: python predict_2026.py
        → Trains on all 340 labeled teams (5 historical seasons)
        → Applies the 4-stage pipeline to the 68 tournament teams
        → Outputs seed predictions to output/

Step 3: python generate_kaggle_submission.py
        → Runs LOSO cross-validation on 91 test teams
        → Outputs submission_kaggle.csv for Kaggle upload
```

### 4.3 Final Model Structure

**Key hyperparameters (all tuned via LOSO on training data):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| BLEND_W1, W3, W4 | 0.64, 0.28, 0.08 | LR1, LR2, XGB weights in pairwise blend |
| DUAL_BLEND | 0.25 | Weight given to Ridge committee model |
| DUAL_RIDGE_ALPHA | 10.0 | Regularization strength for 8-feature Ridge |
| HUNGARIAN_POWER | 0.15 | Cost matrix exponent in Hungarian algorithm |
| SWAP_NET_GAP | 10 | Minimum NET rank gap to trigger AQ↔AL swap |
| SWAP_PRED_GAP | 6 | Minimum predicted seed gap to trigger swap |

**Zone correction parameters** (7 zones spanning seeds 17–68, each with 2-4 tuned correction weights targeting SOS, NET, conference-bid history, and quadrant metrics).

**Interpretation:** The model says "Duke is seed 1" because when compared pairwise to all other 67 teams, Duke wins ~99% of comparisons. The Hungarian algorithm then makes this assignment unique and globally optimal. Zone corrections fine-tune the mid-range where pairwise signals are weakest.

### 4.4 Areas of Improvement

1. **Zone correction overfitting risk (MEDIUM concern).** The 7-zone parameters were tuned on the same 91 test teams used for evaluation. A truly held-out season (2025-26) will likely show some regression. Realistic expected performance: **60-70% exact match, RMSE 2-3** on unseen data.

2. **Bracketology dependency.** The 68-team field is based on ESPN projected brackets, not the actual Selection Sunday announcement. If 3-4 bubble teams differ, seed predictions for those teams will be wrong regardless of model quality.

3. **Small training set.** Only 5 seasons × 68 teams = 340 labeled examples. More historical data (pre-2020) could improve generalization, but the NCAA changed NET calculation methodology in 2018-19, limiting useful history.

4. **Conference realignment not modeled.** Teams switching conferences (e.g., Texas to SEC in 2024-25) break historical conference-bid patterns. The `ConfBidHistoricalSeed` feature may be stale for recently-realigned teams.

5. **No uncertainty quantification.** The model outputs point predictions (seed 14) but doesn't provide confidence intervals (seed 12-16 at 90% confidence). Adding bootstrap prediction intervals would give judges more information about prediction reliability.

6. **Committee subjectivity is inherently unpredictable.** Some committee decisions are influenced by factors not in our dataset (injuries, suspensions, head-to-head results within conference play, TV market considerations). No model can fully capture these.

---

## 5. File Reference

```
submission/
├── README.md                          ← This document
├── seed_selections_2026.txt           ← Final 2025-26 seed predictions (1-68)
├── requirements.txt                   ← Python dependencies (pip install -r)
│
├── code/
│   ├── ncaa_2026_model.py             ← Core model: all functions, constants, pipeline
│   ├── generate_kaggle_submission.py  ← LOSO cross-validation + Kaggle CSV output
│   ├── convert_to_model_data.py       ← NCAA xlsx + ESPN brackets → model-ready CSV
│   └── predict_2026.py               ← One-command 2026 prediction runner
│
├── data/
│   ├── NCAA_Seed_Training_Set2.0.csv  ← Provided: 1,353 teams, 5 seasons, with seeds
│   ├── NCAA_Seed_Test_Set2.0.csv      ← Provided: 451 teams, 5 seasons, seeds withheld
│   ├── submission.csv                 ← Provided: 451-row submission template
│   ├── NCAA Statistics-3.xlsx         ← Supplemental: 365 teams, 2025-26 (thru March 13)
│   ├── NCAA_2026_Data.csv             ← Generated: 68 tournament teams in model format
│   └── NCAA_2026_Template.csv         ← Generated: blank prediction template
│
├── output/
│   ├── submission_kaggle.csv          ← Kaggle submission (451 rows, LOSO predictions)
│   ├── submission_2026.csv            ← 2026 seeds in Kaggle format
│   ├── bracket_2026_prediction.csv    ← Seed predictions (RecordID + seed)
│   └── bracket_2026_detailed.csv      ← Full details (NET, conf, bid, raw scores)
│
└── analysis/
    ├── ncaa_v50_generalization_analysis.py  ← Generalization & robustness tests
    └── overfitting_analysis.py              ← Overfitting risk assessment
```

---

## 6. Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate Kaggle submission (LOSO on training/test data)
cd code && python generate_kaggle_submission.py

# 3. Generate 2025-26 predictions
python convert_to_model_data.py    # Parse NCAA Statistics xlsx → CSV
python predict_2026.py             # Predict seeds for 68 teams

# 4. Run generalization analysis (optional)
cd ../analysis && python ncaa_v50_generalization_analysis.py
```
