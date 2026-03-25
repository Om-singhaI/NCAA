# Methodology

## 1. Problem Formulation

The NCAA Selection Committee assigns a unique seed $s \in \{1, 2, \ldots, 68\}$ to each tournament team. This defines a **constrained ranking problem** with two key properties:

1. **Bijectivity constraint:** The mapping from teams to seeds is a bijection — each seed is assigned to exactly one team.
2. **Relative ordering:** A team's seed depends not on its absolute quality, but on its quality *relative to all other tournament teams*.

Standard regression approaches (e.g., predicting seed directly from features) violate the bijectivity constraint and ignore the relative nature of the problem. We address both issues through a four-stage pipeline.

### Notation

- $N = 68$: number of tournament teams per season
- $\mathbf{x}_i \in \mathbb{R}^d$: feature vector for team $i$ ($d = 68$)
- $s_i \in \{1, \ldots, N\}$: true seed for team $i$
- $\hat{s}_i$: predicted seed for team $i$
- $K = 5$: number of training seasons (2020–21 through 2024–25)
- $N_{\text{total}} = 340$: total labeled teams across all seasons

## 2. Stage 1 — Pairwise Learning-to-Rank

### 2.1 Motivation

Instead of learning $f: \mathbb{R}^d \to \{1, \ldots, 68\}$ directly, we learn a **pairwise preference function**:

$$g(\mathbf{x}_i, \mathbf{x}_j) = P(s_i < s_j)$$

This reframes the problem: given teams $i$ and $j$, which receives the better (lower) seed?

### 2.2 Data Construction

For each season with $n$ teams, we generate $\binom{n}{2}$ pairs. For each pair $(i, j)$:

- **Input:** $\mathbf{x}_i - \mathbf{x}_j$ (feature difference vector)
- **Target:** $y_{ij} = \mathbb{1}[s_i < s_j]$

Both orderings are included (i.e., both $(\mathbf{x}_i - \mathbf{x}_j, y_{ij})$ and $(\mathbf{x}_j - \mathbf{x}_i, 1 - y_{ij})$), yielding $n(n-1)$ examples per season. With $n = 68$, this produces **~4,556 training pairs per season** — a 66× increase over the 68 direct examples.

### 2.3 Adjacent-Pair Filtering

For the primary model component, we restrict training pairs to those with $|s_i - s_j| \leq 30$. This focuses learning on **informative comparisons** and excludes trivially separable pairs (e.g., seed 1 vs. seed 68) that contribute gradient noise without useful signal.

### 2.4 Scoring

At inference, each test team $i$ is scored against all other test teams:

$$\text{score}(i) = \sum_{j \neq i} g(\mathbf{x}_i - \mathbf{x}_j)$$

Teams are ranked by score to produce a continuous ranking, which is converted to seeds via the assignment step (Stage 3).

## 3. Stage 2 — Ensemble Blending

Three pairwise classifiers are trained and blended:

| Component | Classifier | Features | Pairs | Weight |
|:----------|:-----------|:---------|:------|-------:|
| C1 | Logistic Regression ($C = 5.0$) | All 68 | Adjacent ($\Delta s \leq 30$) | 0.64 |
| C2 | Logistic Regression ($C = 0.5$) | Top 25 | All | 0.28 |
| C3 | XGBoost (depth=4, 300 trees, lr=0.05) | All 68 | All | 0.08 |

The blend weights were selected via **leave-one-season-out (LOSO) cross-validation** to maximize exact-match accuracy.

**Design rationale:**
- C1 uses all features with moderate regularization and adjacent-pair filtering — the primary workhorse.
- C2 uses aggressive regularization ($C = 0.5$) on only the 25 most important features — a bias-toward-simplicity check.
- C3 captures nonlinear feature interactions that linear models miss, but receives low weight (8%) due to overfitting risk on small data.

### 3.1 Feature Selection

Top-$k$ features are selected by **combined importance ranking** across three methods:

1. Ridge regression coefficient magnitudes ($\alpha = 5.0$)
2. Random Forest feature importances (500 trees, max depth 10)
3. XGBoost feature importances (700 trees, depth 5)

Each method ranks all 68 features; the average rank determines the combined ordering. The feature `NET Rank` is force-included in the top-$k$ set.

## 4. Stage 3 — Hungarian Assignment

### 4.1 The Assignment Problem

The pairwise blend produces continuous scores $\hat{r}_i$, but valid seeds require a bijection $\sigma: \{1, \ldots, N\} \to \{1, \ldots, N\}$. We construct a cost matrix:

$$C_{ij} = |\hat{r}_i - j|^{p}, \quad p = 0.15$$

where $\hat{r}_i$ is team $i$'s raw score and $j \in \{1, \ldots, 68\}$ is a candidate seed. The exponent $p = 0.15$ compresses cost differences, making the assignment more robust to score noise (compared to $p = 1$ or $p = 2$).

The Hungarian algorithm (Kuhn, 1955) finds the globally optimal assignment:

$$\sigma^* = \arg\min_{\sigma \in S_N} \sum_{i=1}^{N} C_{i, \sigma(i)}$$

where $S_N$ is the set of all permutations of $\{1, \ldots, N\}$.

### 4.2 Dual-Hungarian Ensemble

Two models are run through Hungarian independently:

1. **Pairwise model** (75%): The three-component blend from Stage 2
2. **Committee model** (25%): Ridge regression ($\alpha = 10$) on 8 carefully selected features that capture committee-specific biases

The assignments are averaged and a **final Hungarian pass** produces valid seeds:

$$\hat{s}_i^\text{final} = \text{Hungarian}\left(0.75 \cdot \hat{s}_i^\text{pairwise} + 0.25 \cdot \hat{s}_i^\text{committee}\right)$$

## 5. Stage 4 — Zone Corrections

### 5.1 Motivation

Even after dual-Hungarian assignment, systematic biases persist in specific seed ranges. These arise from committee behaviors that the statistical model cannot fully capture (e.g., geographic considerations, strength-of-schedule adjustments for mid-major conferences).

### 5.2 Correction Mechanism

Each zone defines:
- A seed range $[lo, hi]$
- Correction signal weights for domain features (SOS gap, conference strength, bid-type history)
- A correction score computed as a weighted sum of these signals

Within each zone, teams are **re-ordered by correction score** using a local Hungarian assignment on the correction cost matrix. Critically, corrections only permute seeds within the zone — they never add or remove seeds.

### 5.3 Zone Definitions

| Zone | Range | Signal | Target Bias |
|:-----|:------|:-------|:------------|
| Z1 | 17–34 | SOS gap | Mid-range teams misordered by strength of schedule |
| Z2 | 34–44 | AQ/AL/SOS (reversed) | Upper-mid teams affected by conference strength bias |
| Z3 | 42–50 | SOSNET/conf/history | VCU, Drake-type auto-qualifier positioning |
| Z4 | 48–52 | NET-conf/history | Transition region between mid and bottom seeds |
| Z5 | 52–60 | SOSNET/conf/history | Bottom seed ordering by schedule quality |
| Z6 | 60–63 | Opponent rank | Tail ordering by opponent quality |
| Z7 | 63–68 | SOSNET/conf/history | Extreme tail auto-qualifier positioning |

### 5.4 AQ↔AL Swap Rule

A post-processing rule addresses a specific committee pattern: **auto-qualifier (AQ) teams with strong NET rankings tend to be under-seeded relative to at-large (AL) teams with weaker NET rankings** in the 30–45 seed range. When an AQ team's predicted-minus-actual NET gap exceeds 10 and a nearby AL team has NET worse than its predicted seed would suggest, the two are swapped.

## 6. Feature Engineering

The 20 raw input features are expanded to **68 model features** in six categories:

| Category | Count | Examples |
|:---------|------:|:--------|
| Raw rankings | 4 | NET Rank, NETSOS, AvgOppNETRank, PrevNET |
| Win-loss metrics | 8 | Overall W-L%, Conference W-L%, Road W-L%, total games |
| Quadrant scores | 14 | Q1-Q4 wins/losses, Q1 dominance, quad balance, resume score |
| Composite ratings | 12 | Power rating, Elo proxy, SOS-adjusted NET, quality ratio |
| Bid-type interactions | 4 | AL×NET, AQ×NET, AQ×SOS penalty, mid-major AQ indicator |
| Context features | 26 | Tournament field rank, conference stats, historical seed, season percentiles |

## 7. Evaluation Protocol

### 7.1 Leave-One-Season-Out (LOSO)

For each season $k \in \{1, \ldots, K\}$:
1. Train on all teams from seasons $\{1, \ldots, K\} \setminus \{k\}$
2. Predict seeds for season $k$
3. Compute metrics against ground truth

This protocol evaluates true temporal generalization — the model never sees the test season during training.

### 7.2 Metrics

| Metric | Definition |
|:-------|:-----------|
| Exact Match | $\frac{1}{N} \sum_{i} \mathbb{1}[\hat{s}_i = s_i]$ |
| RMSE | $\sqrt{\frac{1}{N} \sum_{i} (\hat{s}_i - s_i)^2}$ |
| MAE | $\frac{1}{N} \sum_{i} |\hat{s}_i - s_i|$ |
| Within ±$k$ | $\frac{1}{N} \sum_{i} \mathbb{1}[|\hat{s}_i - s_i| \leq k]$ |
| Squared Error (SE) | $\sum_{i} (\hat{s}_i - s_i)^2$ (competition metric) |

## 8. Limitations and Future Work

1. **Small training set (N=340):** Five seasons provide limited data for learning 7 zone correction rules. The 6.5× RMSE blowup from CV to held-out evaluation is a direct consequence.

2. **Auto-qualifier uncertainty:** Conference tournament upsets (10/68 field misses in 2026) are inherently unpredictable before Selection Sunday.

3. **Committee subjectivity:** The Selection Committee's decisions incorporate qualitative factors (eye test, game film review) that no statistical model can fully capture.

4. **Temporal non-stationarity:** Committee behavior may shift across seasons as new members join and evaluation criteria evolve.

**Future directions:**
- Nested cross-validation for zone parameter selection to reduce overfitting
- Incorporating real-time conference tournament results
- Expanding training data as more seasons become available
- Bayesian approaches to quantify prediction uncertainty per team

## References

1. Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2), 83–97.
2. Burges, C. J. (2010). From RankNet to LambdaRank to LambdaMART: An overview. *Microsoft Research Technical Report*.
3. NCAA (2026). NCAA Evaluation Tool (NET) Rankings. *National Collegiate Athletic Association*.
