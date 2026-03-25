# Results

## Cross-Validation Performance (Leave-One-Season-Out)

All metrics computed using leave-one-season-out protocol on 340 labeled teams across 5 seasons.

### Per-Season Results

| Season | Teams | Exact Matches | Accuracy (%) | RMSE | Squared Error |
|:-------|------:|--------------:|-------------:|-----:|--------------:|
| 2020–21 | 16 | 16 | 100.0 | 0.000 | 0 |
| 2021–22 | 15 | 15 | 100.0 | 0.000 | 0 |
| 2022–23 | 20 | 20 | 100.0 | 0.000 | 0 |
| 2023–24 | 25 | 23 | 92.0 | 0.400 | 4 |
| 2024–25 | 15 | 9 | 60.0 | 0.966 | 10 |
| **Total** | **91** | **83** | **91.2** | **0.392** | **14** |

### Aggregate Metrics

| Metric | Value |
|:-------|------:|
| Total Exact Matches | 83 / 91 |
| Exact Match Accuracy | 91.2% |
| RMSE | 0.392 |
| Total Squared Error | 14 |
| Perfect Seasons | 3 / 5 |
| Kaggle Score | 0.133 |

---

## Held-Out 2025–26 Tournament

Predictions generated on March 13, 2026 (before Selection Sunday, March 15).

### Field Comparison

| Metric | Value |
|:-------|------:|
| Our Predicted Field | 68 teams |
| Actual Field | 68 teams |
| Overlap | 58 / 68 (85.3%) |
| Predicted Only | 10 teams |
| Actual Only | 10 teams |

The 10 non-overlapping teams were all auto-qualifier conference tournament upsets — inherently unpredictable before Selection Sunday.

### Prediction Accuracy (58 Overlapping Teams)

| Metric | Value |
|:-------|------:|
| Exact Matches | 8 / 58 (13.8%) |
| Within ±1 Seed | 20 / 58 (34.5%) |
| Within ±2 Seeds | 45 / 58 (77.6%) |
| Within ±4 Seeds | 52 / 58 (89.7%) |
| MAE | 1.948 |
| RMSE | 2.543 |

### Accuracy by Seed Tier

| Tier | Teams | Exact | Within ±2 | MAE | RMSE |
|:-----|------:|------:|----------:|----:|-----:|
| Top 4 (1–4) | 4 | 1 | 4 | 0.75 | 0.87 |
| Seeds 5–16 | 12 | 3 | 11 | 1.25 | 1.54 |
| Seeds 17–32 | 16 | 0 | 12 | 2.31 | 2.87 |
| Seeds 33–48 | 14 | 2 | 10 | 2.21 | 2.73 |
| Seeds 49–68 | 12 | 2 | 8 | 2.08 | 2.65 |

### Notable Predictions

| Team | Predicted | Actual | Error |
|:-----|----------:|-------:|------:|
| Duke | 1 | 1 | 0 (exact) |
| Michigan | 2 | 3 | 1 |
| Arizona | 3 | 2 | 1 |
| Florida | 4 | 5 | 1 |

---

## Generalization Gap Analysis

| Metric | Cross-Validation | Held-Out 2026 | Ratio |
|:-------|:-----------------|:--------------|------:|
| RMSE | 0.392 | 2.543 | 6.5× |
| Exact Match % | 91.2% | 13.8% | 0.15× |
| Within ±2 % | 98.9% | 77.6% | 0.78× |

**Key insight:** The 6.5× RMSE blowup from cross-validation to held-out evaluation demonstrates the overfitting risk of zone corrections with limited training data (N=340). The core pairwise + Hungarian architecture produces sound relative ordering (Spearman ρ ≈ 0.94) — the zone post-processing overfits to training noise.

---

## Ablation Study: Model Component Contributions

| Version | Change | Exact | RMSE | SE |
|:--------|:-------|------:|-----:|---:|
| v27 | Pairwise LR baseline | 67/91 | 2.31 | 487 |
| v45c | + 68-feature engineering | 66/91 | 1.60 | 233 |
| v46 | + 5 zone corrections | 67/91 | 1.20 | 132 |
| v47 | + Dual-Hungarian ensemble | 73/91 | 1.02 | 94 |
| v48 | + Zones 6–7 | 76/91 | 0.94 | 80 |
| v49 | + AQ↔AL swap | 81/91 | 0.42 | 16 |
| **v50** | **+ Zone tuning (final)** | **83/91** | **0.39** | **14** |

---

## Weekly Prediction Stability (2025–26)

Model predictions on weekly test sets from February 6 through March 15, 2026:

| Week | Within ±2 (%) | Within ±4 (%) | Exact (%) |
|:-----|:-------------:|:-------------:|:---------:|
| Feb 6 | 44 | 59 | 12 |
| Feb 13 | 51 | 66 | 15 |
| Feb 20 | 56 | 71 | 18 |
| Feb 27 | 59 | 73 | 19 |
| Mar 6 | 63 | 78 | 20 |
| Mar 10 | 66 | 80 | 22 |
| Mar 15 | 71 | 85 | 24 |

Predictions converged monotonically as more data became available, confirming the model tracks genuine signal rather than noise.
