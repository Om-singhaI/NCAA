# NCAA Tournament Seed Prediction

Predicts the 1–68 overall seed for all 68 teams in the NCAA March Madness tournament.

## Results

| Metric | Value |
|--------|-------|
| Kaggle RMSE | **0.133** |
| Squared Error (LOSO) | 14 |
| Exact Matches | 83/91 (91%) |
| Seasons Tested | 2020-21 through 2024-25 |

## Model Architecture (v50)

**Three-layer pipeline:**

1. **v12 Pairwise Base** — Blends 3 pairwise comparison models (logistic regression × 2, XGBoost) that predict "which team gets the better seed?" for every pair, then converts win-rates to seed scores
2. **Dual-Hungarian Ensemble** — Runs a separate 8-feature Ridge committee model, applies both through Hungarian optimization (global 1-to-1 seed assignment), blends at 0.75/0.25
3. **Zone Corrections** — 7 seed-range-specific corrections fix systematic biases (e.g., mid-major AQ teams under-seeded, power-conference AL teams over-seeded)
4. **AQ↔AL Swap** — Post-processing rule that swaps auto-qualifying (AQ) and at-large (AL) teams when the model follows NET rankings too closely (validated across all 5 seasons)

**Key features:** NET Rank, NETSOS, Win-Loss %, conference-bid history, tournament field rank, opponent NET quality, power conference indicators

## Project Structure

```
ncaa_2026_model.py            # Production model (all functions + constants)
generate_kaggle_submission.py # Generates Kaggle submission CSV
predict_2026.py               # One-command 2026 prediction runner
data/                         # Training/test CSVs + templates
analysis/                     # Generalization & robustness analysis scripts
notebooks/                    # Jupyter notebooks (exploration)
archive/                      # Development history (v4-v50, experiments)
logs/                         # Output logs from experiments
```

## Quick Start

### Generate Kaggle Submission (on training/test data)
```bash
python generate_kaggle_submission.py
# → submission_kaggle.csv
```

### Predict 2026 Tournament
```bash
# 1. Fill in data/NCAA_2026_Data.csv with 68 team stats
# 2. Run:
python predict_2026.py
# → submission_2026.csv
```

### Run Generalization Analysis
```bash
python analysis/ncaa_v50_generalization_analysis.py
```

## Requirements

```
python >= 3.10
numpy
pandas
scikit-learn
xgboost
scipy
```

## Key Findings

- The v12 pairwise base is the most robust component (HIGH trust)
- Zone corrections and blend weights are tuned on 91 test teams → expect 2-5× worse RMSE on truly unseen data
- The AQ↔AL swap pattern is a real NCAA committee bias (10 AQ teams under-seeded across all 5 seasons, 40 AL teams over-seeded)
- Realistic expected Kaggle score for 2025-26: **0.3–0.5**
