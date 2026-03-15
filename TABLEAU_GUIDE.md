# Tableau Public — Step-by-Step Guide for NCAA Dashboard

**Time needed: ~30-45 minutes for a complete dashboard**

---

## STEP 0: Open Tableau Public

1. Open the **Tableau Public** app on your Mac
2. You'll see a blue start screen with "Connect" on the left

---

## STEP 1: Load your data (2 minutes)

1. On the left sidebar under **"Connect"**, click **"Text file"**
2. Navigate to your project folder → `tableau_exports/`
3. Select **`team_predictions.csv`** and click Open
4. You'll see your data in a grid (like a spreadsheet). This is the **Data Source** tab
5. At the bottom-left, click the orange **"Sheet 1"** tab to go to the worksheet view

> **You can add more CSV files later.** For now, `team_predictions.csv` has everything you need.

---

## STEP 2: The 5 Things You Need to Know

Tableau has 4 key areas in the worksheet view:

```
┌──────────────────────────────────────────────────┐
│  Toolbar (filters, marks, formatting)            │
├──────────┬───────────────────────────────────────┤
│          │  ┌─ Columns shelf ──────────────────┐ │
│  Data    │  ├─ Rows shelf ─────────────────────┤ │
│  Panel   │  │                                  │ │
│  (left)  │  │        Canvas                    │ │
│          │  │    (your chart appears here)      │ │
│  All     │  │                                  │ │
│  your    │  │                                  │ │
│  fields  │  │                                  │ │
│  are     │  └──────────────────────────────────┘ │
│  listed  │                                       │
│  here    │  Marks card (color, size, label, etc) │
└──────────┴───────────────────────────────────────┘
```

### The 5 actions:
1. **Drag** a field from the Data Panel to **Columns** → becomes the X-axis
2. **Drag** a field from the Data Panel to **Rows** → becomes the Y-axis
3. **Drag** a field to **Color** (in Marks card) → colors your chart
4. **Drag** a field to **Label** (in Marks card) → adds text labels
5. **Click "Show Me"** (top-right) → Tableau suggests chart types

That's it. Everything in Tableau is drag-and-drop.

---

## STEP 3: Build Chart 1 — Team Performance Scatter Plot (5 min)

This shows NET Rank vs Actual Seed, colored by conference.

1. **Drag** `NET Rank` to **Columns** (from the left panel)
2. **Drag** `Actual Seed` to **Rows**
3. In the **Marks** card (left of canvas), change the dropdown from "Automatic" to **"Circle"**
4. **Drag** `Conference` to **Color** (in Marks card)
5. **Drag** `Team` to **Label** (in Marks card)
6. **Drag** `Team` to **Detail** (in Marks card) — this makes each dot one team
7. Right-click the sheet tab at bottom → **Rename** → type "NET vs Seed"

**Make it better:**
- Click **Format** menu → **Font** → increase size
- Click a dot to see team details
- Add a filter: drag `Season` to the **Filters** shelf → check all seasons → click OK → right-click `Season` on Filters shelf → **Show Filter** (adds a dropdown for users)

---

## STEP 4: Build Chart 2 — Prediction Accuracy Bar Chart (5 min)

This shows how many predictions were exactly correct per season.

1. Click the **"New Worksheet"** tab at the bottom (the "+" icon next to Sheet 1)
2. **Drag** `Season` to **Columns**
3. **Drag** `Exactly Correct` to **Rows** (Tableau will automatically SUM it)
4. **Drag** `Season` to **Color**
5. In the **Marks** card, click **Label** → check **"Show mark labels"**
6. From the toolbar, click **"Show Me"** (top-right) → select the **bar chart** icon
7. Right-click the sheet tab → **Rename** → "Accuracy by Season"

**Make it better:**
- Right-click the Y-axis → **Edit Axis** → set range from 0 to 25
- Click **Format** → **Borders** to add cleaner grid lines

---

## STEP 5: Build Chart 3 — Prediction Error Distribution (5 min)

Shows where the model gets predictions wrong.

1. New Worksheet (+ tab at bottom)
2. First, filter to test teams only: **Drag** `Is Test Team` to **Filters** → check only **"True"** (or **1**) → OK
3. **Drag** `Prediction Error` to **Columns**
4. **Drag** `Number of Records` (or `RecordID`) to **Rows** — pick **COUNT**
5. Click **"Show Me"** → select **Histogram**
6. **Drag** `Seed Zone` to **Color**
7. Rename tab → "Error Distribution"

---

## STEP 6: Build Chart 4 — Conference Bubble Chart (5 min)

Shows which conferences are strongest.

1. New Worksheet
2. **Drag** `Conference` to **Detail** (in Marks card)
3. In **Marks** dropdown, change to **"Circle"**
4. **Drag** `Actual Seed` to **Size** (in Marks card) — it averages seed
5. **Drag** `NET Rank` to **Color** → click Color → **Edit Colors** → pick "Orange-Blue Diverging"
6. **Drag** `Conference` to **Label**
7. Click **"Show Me"** → select **Packed Bubbles**
8. Rename → "Conference Power"

---

## STEP 7: Build Chart 5 — Predicted vs Actual Heatmap (7 min)

This is the most impressive chart — shows model confusion matrix.

1. New Worksheet
2. First, load the heatmap data: go to **Data** menu → **New Data Source** → **Text file** → select `tableau_exports/model_accuracy.csv`
3. **Drag** `Actual Seed Bin` to **Columns**
4. **Drag** `Predicted Seed Bin` to **Rows**
5. **Drag** `Count` to **Color** (in Marks card)
6. Click **Color** → **Edit Colors** → choose **"Orange-Gold"** or **"Blue-Green Sequential"**
7. **Drag** `Count` to **Label**
8. In **Marks** dropdown, change to **"Square"**
9. Sort axes: right-click `Actual Seed Bin` on Columns → **Sort** → Manual → arrange as: 1-4, 5-8, 9-12, 13-16, 17-24, 25-34, 35-44, 45-52, 53-60, 61-68
10. Do the same for `Predicted Seed Bin` on Rows
11. Rename → "Accuracy Heatmap"

---

## STEP 8: Build Chart 6 — Feature Importance (5 min)

1. New Worksheet
2. **Data** menu → **New Data Source** → load `feature_importance.csv`
3. **Drag** `Feature` to **Rows**
4. **Drag** `Average Rank` to **Columns**
5. **Sort**: click the sort descending icon on the toolbar (or right-click → Sort → Ascending by `Average Rank`)
6. **Drag** `Category` to **Color**
7. Filter: drag `Combined Rank` to Filters → select "At most" → 25 (shows top 25 only)
8. Rename → "Top Features"

---

## STEP 9: Build the Dashboard (10 min)

This is where you combine all charts into one interactive page.

1. Click **"New Dashboard"** tab at the bottom (icon looks like a grid with a +)
2. On the left, you'll see all your sheets listed under **"Sheets"**
3. Set size: under **"Size"** on the left → click the dropdown → **"Automatic"** (or "Fixed: 1200 x 900")
4. **Drag** your sheets onto the dashboard canvas:

**Suggested layout:**
```
┌──────────────────────────────────────┐
│           TITLE (text box)           │
├──────────────┬───────────────────────┤
│  NET vs Seed │   Accuracy Heatmap   │
│  (scatter)   │   (heatmap)          │
├──────────────┼───────────────────────┤
│  Accuracy    │   Conference Power   │
│  by Season   │   (bubbles)          │
├──────────────┴───────────────────────┤
│  Top Features (horizontal bars)      │
└──────────────────────────────────────┘
```

5. **Add a title**: from the left panel under "Objects", drag **"Text"** to the top → type your title:
   ```
   NCAA March Madness — ML Seed Prediction Model
   v50 Pipeline: 91% Accuracy on 91 Test Teams
   ```
6. **Add interactivity**: click any chart on the dashboard → click the **funnel icon** (filter icon) on the chart border → this makes it a filter for the whole dashboard. Now clicking a season in one chart filters all others!

---

## STEP 10: Publish to Tableau Public (3 min)

1. **File** menu → **Save to Tableau Public As...**
2. Sign in with your Tableau Public account (create one at public.tableau.com if needed)
3. Name it: **"NCAA March Madness 2026 — ML Bracket Prediction"**
4. Click **Save**
5. It will open in your browser — **copy the URL**
6. Submit this URL to the Kaggle competition's Tableau Prize section

---

## TIPS FOR WINNING

- **Title matters** — make it clear and compelling
- **Interactivity wins** — add Season filters, tooltips, click-to-filter
- **Tell a story** — use Dashboard → New Story to create a narrative flow
- **Color consistently** — use the same color palette across charts
- **Keep it clean** — 4-6 charts max, don't overcrowd
- **Add annotations** — right-click any data point → Annotate → Point

---

## QUICK REFERENCE

| Want to...                    | Do this...                                         |
|-------------------------------|---------------------------------------------------|
| Change chart type             | Click "Show Me" (top-right)                        |
| Add a filter                  | Drag field to Filters shelf                        |
| Make filter visible to users  | Right-click filter → Show Filter                   |
| Color by category             | Drag field to Color (Marks card)                   |
| Add labels                    | Drag field to Label (Marks card)                   |
| Format numbers                | Right-click axis → Format                          |
| Add new data source           | Data menu → New Data Source                        |
| Undo anything                 | Cmd+Z                                              |
| Rename a sheet                | Right-click tab → Rename                           |
| Create dashboard              | Click grid+ icon at bottom                         |
| Publish                       | File → Save to Tableau Public                      |
