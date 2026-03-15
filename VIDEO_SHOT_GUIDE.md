# VIDEO RECORDING GUIDE — Shot-by-Shot

Follow this sequence while screen recording. Each shot tells you exactly what to have on screen.
Import VIDEO_CAPTIONS.srt into your editor for the captions.

---

## HOW TO RECORD

1. **Screen record your entire desktop** (QuickTime → File → New Screen Recording, or OBS)
2. Follow the shots below — just switch between VS Code tabs and terminal as directed
3. Import the recording into your video editor (iMovie, CapCut, DaVinci Resolve)
4. Import `VIDEO_CAPTIONS.srt` as subtitles — they're pre-timed to 15 minutes
5. Read the captions and record your voiceover as a separate audio track
6. The captions appear on screen as subtitles for you to read from

---

## PRE-RECORDING SETUP

Open these in VS Code tabs (in this order):
1. `README.md`
2. `ncaa_2026_model.py`
3. `generate_kaggle_submission.py`
4. `convert_to_model_data.py`
5. `output/2026/seed_selections_2026.txt`

Have a terminal open at project root with venv activated:
```
cd /Users/omsinghal/Desktop/NCAA-1
source .venv/bin/activate
```

---

## SHOT SEQUENCE

### SHOT 1 — Title (0:00 – 0:05)
**SHOW:** README.md — the title "# NCAA Tournament Seed Prediction" visible
**CAPTION:** 1 — "NCAA Tournament Seed Prediction — Technical Design Video"

### SHOT 2 — Intro (0:05 – 0:16)
**SHOW:** README.md — Results table visible (RMSE, Exact Matches, Seasons)
**CAPTION:** 2-3

### SHOT 3 — Problem: 68 seeds (0:16 – 0:35)
**SHOW:** README.md — scroll slowly to show "Model Architecture (v50)"
**CAPTION:** 4-5-6

### SHOT 4 — Why it's hard (0:35 – 1:05)
**SHOW:** `ncaa_2026_model.py` — scroll to the top docstring (lines 1-70)
Show the architecture description text
**CAPTION:** 7-8-9-10-11

### SHOT 5 — The data (1:05 – 1:35)
**SHOW:** `ncaa_2026_model.py` — scroll to `load_data()` function (around line 235)
Slowly scroll through to show columns being loaded
**CAPTION:** 12-13-14

### SHOT 6 — Pairwise insight intro (1:35 – 2:06)
**SHOW:** `ncaa_2026_model.py` — scroll to the comment block before `build_pairwise_data` (around line 1050)
**CAPTION:** 15-16-17

### SHOT 7 — Why relative not absolute (2:06 – 2:28)
**SHOW:** Still near `build_pairwise_data` — show the function signature
**CAPTION:** 18-19-20

### SHOT 8 — Pairwise function (2:28 – 2:56)
**SHOW:** `build_pairwise_data` function (around line 1055-1070)
Highlight this code with cursor:
```python
diff = X[a] - X[b]
target = 1.0 if y[a] < y[b] else 0.0
pairs_X.append(diff); pairs_y.append(target)
```
**CAPTION:** 21-22-23-24-25

### SHOT 9 — Scoring mechanism (2:56 – 3:18)
**SHOW:** `pairwise_score` function (around line 1085-1095)
Point cursor at `scores[i] = probs.sum()` and `np.argsort(np.argsort(-scores))`
**CAPTION:** 26-27-28

### SHOT 10 — Adjacent pairs (3:18 – 3:48)
**SHOW:** `build_pairwise_data_adjacent` function (around line 1070-1085)
Point cursor at `if abs(y[a] - y[b]) > max_gap: continue`
**CAPTION:** 29-30-31-32

### SHOT 11 — Three-model blend intro (3:48 – 4:11)
**SHOW:** `predict_robust_blend` function (around line 1095)
Show the function docstring and signature
**CAPTION:** 33-34-35

### SHOT 12 — Component 1 (4:11 – 4:34)
**SHOW:** Inside `predict_robust_blend` — Component 1 code block
Point cursor at `lr1 = LogisticRegression(C=PW_C1` and `BLEND_W1 = 0.64`
**CAPTION:** 36-37

### SHOT 13 — Component 3 (4:34 – 4:56)
**SHOW:** Component 3 code block — top-25 features
Point cursor at `LogisticRegression(C=PW_C3`
**CAPTION:** 38-39-40

### SHOT 14 — Component 4: XGBoost (4:56 – 5:18)
**SHOW:** Component 4 code block — XGBoost section
Point cursor at `xgb.XGBClassifier(n_estimators=300, max_depth=4`
**CAPTION:** 41-42-43

### SHOT 15 — Blend weights (5:18 – 5:33)
**SHOW:** Scroll up to configuration section — show the blend weight constants
```python
BLEND_W1 = 0.64  # pw_lr_A_full_C5.0
BLEND_W3 = 0.28  # pw_lr_Ak_C0.5
BLEND_W4 = 0.08  # pw_xgb_A_full
```
**CAPTION:** 44-45

### SHOT 16 — Feature engineering intro (5:33 – 5:48)
**SHOW:** `build_features` function — show the function signature and first comment (around line 285)
**CAPTION:** 46-47

### SHOT 17 — Raw rankings (5:48 – 6:03)
**SHOW:** Scroll to core rankings section inside `build_features` (around line 320)
```python
net = pd.to_numeric(df['NET Rank'], errors='coerce')
prev = pd.to_numeric(df['PrevNET'], errors='coerce')
sos = pd.to_numeric(df['NETSOS'], errors='coerce')
```
**CAPTION:** 48-49

### SHOT 18 — Win-loss parsing (6:03 – 6:18)
**SHOW:** The W-L parsing loop at top of `build_features`
```python
for col in ['WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL']:
```
**CAPTION:** 50-51

### SHOT 19 — Quadrant records (6:18 – 6:40)
**SHOW:** Quadrant record section
```python
for q in ['Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
```
**CAPTION:** 52-53-54

### SHOT 20 — Composite features (6:40 – 7:03)
**SHOW:** Composite features section (around line 340-360)
Point cursor at `adj_net`, `power_rating`, `sos_adj_net`
**CAPTION:** 55-56-57

### SHOT 21 — Bid type interactions (7:03 – 7:18)
**SHOW:** Bid interaction features
```python
feat['aq_sos_penalty'] = feat['is_AQ'] * (sos / 100)
feat['midmajor_aq'] = feat['is_AQ'] * (1 - feat['is_power_conf'])
```
**CAPTION:** 58-59

### SHOT 22 — Tournament field rank (7:18 – 7:40)
**SHOW:** Tournament field rank computation (around line 380)
**CAPTION:** 60-61

### SHOT 23 — Conference-bid history (7:40 – 7:56)
**SHOW:** Conference-bid historical seed section (around line 400)
```python
feat['cb_mean_seed'] = np.mean(vals) if vals else 35.0
```
**CAPTION:** 62-63-64

### SHOT 24 — Hungarian function (7:56 – 8:40)
**SHOW:** `hungarian` function (around line 1165)
Point cursor at `linear_sum_assignment(cost)` and `abs(r - p)**power`
**CAPTION:** 65-66-67-68-69-70

### SHOT 25 — Hungarian power explanation (8:40 – 9:02)
**SHOW:** Still on `hungarian` function — point at `power=HUNGARIAN_POWER`
Then briefly show `HUNGARIAN_POWER = 0.15` in the constants section
**CAPTION:** 71-72-73-74

### SHOT 26 — Dual-Hungarian: min8 features (9:02 – 9:40)
**SHOW:** `build_min8_features` function (around line 563)
Show all 8 features listed in the docstring
**CAPTION:** 75-76-77-78

### SHOT 27 — Min8 rationale (9:40 – 10:02)
**SHOW:** Still on `build_min8_features` — point at docstring:
"tourn_field_rank (TFR) — primary signal (36.3% of Ridge weight)"
**CAPTION:** 79-80-81

### SHOT 28 — Dual blend (10:02 – 10:16)
**SHOW:** Show the dual constants:
```python
DUAL_RIDGE_ALPHA = 10.0
DUAL_BLEND = 0.25
```
**CAPTION:** 82-83-84

### SHOT 29 — Zone corrections title (10:16 – 10:39)
**SHOW:** Scroll to the top of the zone constant definitions (around line 140)
Show all 7 zone blocks visible — slow scroll down
**CAPTION:** 85-86-87-88

### SHOT 30 — Zone 1: Mid-range (10:39 – 11:17)
**SHOW:** Zone 1 constants:
```python
MIDRANGE_ZONE = (17, 34)
CORRECTION_SOS = 3
```
Then scroll to `apply_midrange_swap` function (around line 620)
**CAPTION:** 89-90-91-92

### SHOT 31 — Zone 2: Upper-mid (11:17 – 11:40)
**SHOW:** Zone 2 constants:
```python
UPPERMID_ZONE = (34, 44)
UPPERMID_AQ = -2; UPPERMID_AL = -3
```
**CAPTION:** 93-94-95

### SHOT 32 — Zones 3-7 overview (11:40 – 12:10)
**SHOW:** Slow scroll through zones 3-7 constant definitions
Show MIDBOT2_ZONE, MIDBOT_ZONE, BOTTOMZONE_ZONE, TAILZONE_ZONE, XTAIL_ZONE
**CAPTION:** 96-97-98

### SHOT 33 — Zone principle (12:10 – 12:32)
**SHOW:** Inside `apply_midrange_swap` — point at the core logic:
```python
mid_seeds = [pass1_assigned[i] for i in mid_test_indices]
# only re-ordering existing seeds — never changing the set
```
**CAPTION:** 99-100-101-102

### SHOT 34 — LIVE RUN: Kaggle submission (12:32 – 13:18)
**SHOW:** Switch to TERMINAL. Type and run:
```
python3 generate_kaggle_submission.py
```
Wait for output. Show the results scrolling. Point at:
- "83/91 exact (91.2%)"
- Per-season breakdown table
- RMSE line
**CAPTION:** 103-104-105-106-107

### SHOT 35 — Zone correction results (13:18 – 13:25)
**SHOW:** Still in terminal — scroll up to the "Teams changed by zone corrections" table
Point at the ★ markers indicating improvements
**CAPTION:** 108-109

### SHOT 36 — Honesty about overfitting (13:25 – 14:03)
**SHOW:** Stay on the terminal output. Then optionally switch to
`analysis/ncaa_v50_generalization_analysis.py` — show the docstring
or just stay on terminal results
**CAPTION:** 110-111-112-113-114

### SHOT 37 — LIVE RUN: 2026 pipeline (14:03 – 14:40)
**SHOW:** Switch to TERMINAL. Type and run:
```
python3 convert_to_model_data.py && python3 predict_2026.py
```
Wait for output. Show "68/68 teams matched" and the prediction table
**CAPTION:** 115-116-117-118-119-120

### SHOT 38 — 2026 predictions (14:40 – 15:02)
**SHOW:** Open `output/2026/seed_selections_2026.txt` tab in VS Code
Show the top seeds: Duke 1, Michigan 2, Arizona 3, Florida 4...
**CAPTION:** 121-122-123

### SHOT 39 — Summary (15:02 – 15:40)
**SHOW:** Switch back to README.md — show the Model Architecture section
as a visual reminder of the three key ideas
**CAPTION:** 124-125-126-127

---

## POST-RECORDING CHECKLIST

- [ ] Import screen recording into video editor
- [ ] Import VIDEO_CAPTIONS.srt as subtitle track
- [ ] Record voiceover while reading captions (or use text-to-speech)
- [ ] Verify timing aligns — adjust subtitle timing if needed
- [ ] Export as MP4, 1080p minimum
- [ ] Total length should be 15:00-15:40
- [ ] Watch once through to verify code on screen matches what captions describe
