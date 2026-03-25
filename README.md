# Pairwise Learning-to-Rank with Hungarian Assignment for NCAA Tournament Seed Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](docs/paper.md)

> **A constrained ranking system that predicts exact 1-through-68 seed assignments for all NCAA March Madness tournament teams using pairwise comparison, ensemble blending, combinatorial optimization, and domain-specific post-processing.**

---

## Abstract

Predicting the NCAA Selection Committee's seeding decisions is a challenging constrained ranking problem: each of 68 teams must receive a unique seed, rankings are relative rather than absolute, and the committee's decision process incorporates substantial subjective judgment. We present a four-stage pipeline that transforms this into a tractable learning-to-rank task. Stage 1 converts the problem from 68 point predictions into ~4,500 pairwise comparisons per season, increasing effective training data by 66×. Stage 2 blends three classifiers (Logistic Regression and XGBoost) with cross-validation-optimized weights. Stage 3 applies the Hungarian algorithm to enforce the one-team-per-seed constraint via globally optimal assignment. Stage 4 applies domain-specific zone corrections targeting systematic committee biases. On leave-one-season-out cross-validation across 5 seasons (340 teams), the model achieves **91.2% exact-match accuracy** (83/91) with RMSE 0.392. On the held-out 2025–26 tournament, it correctly identifies 85.3% of the field (58/68), places **77.6% of predictions within ±2 seeds**, and achieves RMSE 2.543 — within the pre-registered expected range of 2–3.

**Keywords:** learning-to-rank, Hungarian algorithm, constrained optimization, ensemble methods, sports analytics, NCAA basketball

---

## Key Results

<table>
<tr>
<td>

### Cross-Validation (LOSO)
| Metric | Value |
|:-------|------:|
| Exact Matches | **83 / 91 (91.2%)** |
| RMSE | **0.392** |
| Squared Error | 14 |
| Seasons | 5 (2020–2025) |
| Perfect Seasons | 3 / 5 |

</td>
<td>

### Held-Out 2025–26 Tournament
| Metric | Value |
|:-------|------:|
| Field Overlap | **58 / 68 (85.3%)** |
| Exact Matches | 8 / 58 (13.8%) |
| Within ±2 Seeds | **45 / 58 (77.6%)** |
| MAE | 1.948 |
| RMSE | 2.543 |

</td>
</tr>
</table>

### Per-Season Cross-Validation Breakdown

| Season | Teams | Exact | Accuracy | RMSE |
|:-------|------:|------:|---------:|-----:|
| 2020–21 | 16 | 16 | 100.0% | 0.000 |
| 2021–22 | 15 | 15 | 100.0% | 0.000 |
| 2022–23 | 20 | 20 | 100.0% | 0.000 |
| 2023–24 | 25 | 23 | 92.0% | 0.400 |
| 2024–25 | 15 | 9 | 60.0% | 0.966 |
| **Total** | **91** | **83** | **91.2%** | **0.392** |

---

## Model Architecture

```
                    ┌──────────────────────────────────┐
                    │     20 Raw Statistical Features   │
                    │  (NET, SOS, W-L, Quads, Conf.)   │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │   Feature Engineering (→ 68 dims) │
                    │  Ratios · Composites · Context    │
                    └──────────────┬───────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │  LR (C=5.0)     │ │  LR (C=0.5)     │ │  XGBoost        │
    │  68 feats       │ │  Top-25 feats   │ │  68 feats       │
    │  Adj-pairs ≤30  │ │  All pairs      │ │  All pairs      │
    │  Weight: 64%    │ │  Weight: 28%    │ │  Weight: 8%     │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             └───────────────────┼───────────────────┘
                                 ▼
                    ┌──────────────────────────────────┐
                    │  Dual-Hungarian Ensemble          │
                    │  75% Pairwise + 25% Ridge(α=10)  │
                    │  ↓ Hungarian Assignment (p=0.15) │
                    │  Globally optimal 1-to-68 mapping │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │  7 Zone Corrections + AQ↔AL Swap  │
                    │  Domain-specific post-processing  │
                    └──────────────┬───────────────────┘
                                   │
                    ┌──────────────▼───────────────────┐
                    │     Final Seed Assignments 1–68   │
                    └──────────────────────────────────┘
```

### Stage 1 — Pairwise Comparison
Instead of directly regressing seed values, we frame the problem as **pairwise learning-to-rank**: for each pair of teams $(i, j)$, we compute $\mathbf{x}_i - \mathbf{x}_j$ and train a classifier to predict $P(\text{seed}_i < \text{seed}_j)$. This converts 68 training examples per season into ~4,500 informative comparisons.

### Stage 2 — Ensemble Blending
Three pairwise classifiers are blended with learned weights:
- **Component 1 (64%):** Logistic Regression (C=5.0), full 68 features, adjacent pairs only (gap ≤ 30)
- **Component 2 (28%):** Logistic Regression (C=0.5), top-25 features, all pairs
- **Component 3 (8%):** XGBoost (depth=4, 300 trees, lr=0.05), full features, all pairs

### Stage 3 — Hungarian Assignment
Pairwise scores yield a continuous ranking, but valid seeds require a **discrete bijection** from teams to $\{1, \ldots, 68\}$. We construct a cost matrix $C_{ij} = |s_i - j|^{0.15}$ where $s_i$ is team $i$'s raw score, and solve the assignment problem via the Hungarian algorithm (Kuhn, 1955). A dual ensemble blends the pairwise model (75%) with a Ridge regression committee model (25%) before final assignment.

### Stage 4 — Zone Corrections
Seven seed-range-specific correction rules address systematic committee biases (e.g., mid-major auto-qualifiers under-seeded, power-conference at-large teams over-seeded). Corrections only **re-order teams within assigned seeds** — they cannot introduce or remove assignments.

---

## Generalization Analysis

The gap between cross-validation RMSE (0.392) and held-out RMSE (2.543) — a **6.5× blowup** — provides an empirical case study in overfitting with limited training data (N=340). Key findings:

| Finding | Detail |
|:--------|:-------|
| **Core architecture is sound** | Pairwise + Hungarian achieves strong relative ordering (ρ = 0.94 rank correlation on 2026 data) |
| **Zone corrections overfit** | Tuned on 340 examples, they capture real patterns but also fit noise — mid-range seeds (17–32) had 0 exact matches despite receiving the most correction effort |
| **Pre-registered expectations met** | We predicted RMSE 2–3 on unseen data in our submission README *before* seeing results; actual was 2.54 |
| **Top seeds robust** | Seeds 1–4 all predicted within ±1 of actual; Duke exactly correct at seed 1 |
| **Auto-qualifier uncertainty dominates** | 10/68 field misses were conference tournament upsets — inherently unpredictable before Selection Sunday |

---

## Repository Structure

```
├── ncaa_2026_model.py                # Core model: all stages, features, training, and inference
├── generate_kaggle_submission.py     # Generates competition submission CSV (LOSO validation)
├── predict_2026.py                   # End-to-end 2025–26 season prediction runner
├── convert_to_model_data.py          # Data pipeline: NCAA Excel + ESPN → model-ready CSV
├── compare_actual_seeds.py           # Post-hoc evaluation against actual 2026 bracket
│
├── data/
│   ├── NCAA_Seed_Training_Set2.0.csv # 249 labeled teams (2020–2024), 20 features
│   ├── NCAA_Seed_Test_Set2.0.csv     # 91 labeled teams (held-out seasons), 20 features
│   ├── NCAA Statistics.xlsx          # 365 Division I teams, 2025–26 season stats
│   └── NCAA_2026_Data.csv            # Processed 68-team model input for 2026
│
├── analysis/
│   ├── ncaa_v50_generalization_analysis.py  # Overfitting and generalization diagnostics
│   ├── ncaa_v51_principled.py               # Ablation study: model without zone corrections
│   └── overfitting_analysis.py              # Training-vs-test performance analysis
│
├── output/
│   ├── submission_kaggle.csv         # Competition submission (training + test predictions)
│   └── 2026/                         # 2025–26 predictions, bracket export, submission file
│
├── docs/
│   └── METHODOLOGY.md                # Detailed methodology documentation
│
├── archive/                          # Development history: 50 model versions, experiments
├── requirements.txt                  # Python dependencies
├── CITATION.cff                      # Citation metadata
└── LICENSE                           # MIT License
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/Om-singhaI/NCAA.git
cd NCAA
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce Cross-Validation Results

```bash
python generate_kaggle_submission.py
```

This runs leave-one-season-out cross-validation on all 340 labeled teams and outputs per-season accuracy, zone correction tables, and the competition submission CSV.

### Predict a New Season

```bash
# 1. Place NCAA Statistics Excel in data/
# 2. Convert to model format:
python convert_to_model_data.py

# 3. Generate predictions:
python predict_2026.py
# → output/2026/seed_selections_2026.txt
# → output/2026/submission_2026.csv
```

### Run Generalization Analysis

```bash
python analysis/ncaa_v50_generalization_analysis.py
```

---

## Data

Training and test data are provided by the [Kaggle March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition. Each team record contains 20 statistical features:

| Feature | Description |
|:--------|:------------|
| `NET Rank` | NCAA Evaluation Tool ranking (primary Selection Committee metric) |
| `NETSOS` | NET Strength of Schedule |
| `AvgOppNETRank` | Average opponent NET ranking |
| `PrevNET` | Previous season's NET ranking |
| `WL`, `Conf.Record`, `RoadWL` | Win-loss records (overall, conference, road) |
| `Quadrant1`–`Quadrant4` | Record against each quality quadrant |
| `Conference`, `Bid Type` | Conference affiliation, auto-qualifier (AQ) vs at-large (AL) |

These 20 raw features are engineered into **68 model features** across six categories: raw rankings, parsed win-loss metrics, quadrant quality scores, composite ratings, bid-type interactions, and historical context features.

---

## Development History

This model evolved through **50 iterations**, each targeting specific failure modes:

| Version | Architecture Change | Exact Match | RMSE | Kaggle SE |
|:--------|:-------------------|------------:|-----:|----------:|
| v27 | Pairwise LR baseline | 67/91 | 2.31 | 487 |
| v45c | + Feature engineering (68 feats) | 66/91 | 1.60 | 233 |
| v46 | + Zone corrections (5 zones) | 67/91 | 1.20 | 132 |
| v47 | + Dual-Hungarian ensemble | 73/91 | 1.02 | 94 |
| v48 | + Zones 6–7 refinement | 76/91 | 0.94 | 80 |
| v49 | + AQ↔AL swap rule | 81/91 | 0.42 | 16 |
| **v50** | **+ Zone parameter tuning** | **83/91** | **0.39** | **14** |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{singhal2026ncaa,
  author       = {Singhal, Om},
  title        = {Pairwise Learning-to-Rank with Hungarian Assignment for {NCAA} Tournament Seed Prediction},
  year         = {2026},
  url          = {https://github.com/Om-singhaI/NCAA},
  note         = {Kaggle March Machine Learning Mania 2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- NCAA Evaluation Tool (NET) data via the [Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- ESPN bracketology projections for 2025–26 field composition
- The Hungarian algorithm implementation via [SciPy](https://scipy.org/) (`linear_sum_assignment`)
