#!/usr/bin/env python3
"""
NCAA 2026 — One-Command Prediction Runner
==========================================

When the 2025-26 tournament field is announced (Selection Sunday),
fill in NCAA_2026_Data.csv and run:

    python3 predict_2026.py

This script will:
  1. Predict Overall Seeds (1-68) for all 68 tournament teams
  2. Run Monte Carlo bracket simulation (10,000 brackets)
  3. Generate submission files ready for Kaggle

OUTPUT FILES:
  • submission_2026.csv          — Kaggle submission (RecordID, Overall Seed)
  • bracket_2026_prediction.csv  — same format, backup
  • bracket_2026_full.csv        — full results with bracket probabilities
                                   (P_Rd32, P_Sweet16, P_Elite8, P_FF, P_Champ)
  • bracket_2026_detailed.csv    — detailed with NET rank, conference, etc.

DATA SOURCE:
  All stats come from the NCAA NET Rankings page:
    https://www.ncaa.com/rankings/basketball-men/d1/ncaa-mens-basketball-net-rankings
  and the team's schedule page for W-L records and quadrant records.
"""

import os
import sys
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, 'data', 'NCAA_2026_Data.csv')

def main():
    t0 = time.time()

    # ── Check data file exists ──
    if not os.path.exists(DATA_FILE):
        print('='*60)
        print(' ERROR: NCAA_2026_Data.csv not found!')
        print('='*60)
        print(f'\n  Expected at: {DATA_FILE}')
        print('\n  HOW TO CREATE IT:')
        print('  1. Copy data/NCAA_2026_Template.csv → data/NCAA_2026_Data.csv')
        print('  2. Delete the 2 example rows')
        print('  3. Fill in all 68 tournament teams (see below)')
        print()
        print_data_instructions()
        sys.exit(1)

    # ── Validate data ──
    import pandas as pd
    df = pd.read_csv(DATA_FILE)
    required_cols = [
        'RecordID', 'Season', 'Team', 'Conference', 'Bid Type',
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL',
        'NETSOS', 'NETNonConfSOS',
        'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f'ERROR: Missing columns: {missing}')
        print(f'Your columns: {list(df.columns)}')
        sys.exit(1)

    # Filter to tournament teams only
    tourn = df[df['Bid Type'].isin(['AL', 'AQ'])]
    n_tourn = len(tourn)
    if n_tourn == 0:
        print('ERROR: No tournament teams found (Bid Type must be AL or AQ)')
        sys.exit(1)

    n_al = (tourn['Bid Type'] == 'AL').sum()
    n_aq = (tourn['Bid Type'] == 'AQ').sum()
    print('='*60)
    print(' NCAA 2026 BRACKET PREDICTION')
    print('='*60)
    print(f'\n  Data file: {DATA_FILE}')
    print(f'  Total rows: {len(df)}')
    print(f'  Tournament teams: {n_tourn} ({n_al} At-Large + {n_aq} Auto-Qualifier)')

    if n_tourn != 68:
        print(f'\n  ⚠ WARNING: Expected 68 tournament teams, got {n_tourn}')
        if n_tourn < 64:
            print('    This will produce incomplete brackets.')
            resp = input('    Continue anyway? (y/N): ').strip().lower()
            if resp != 'y':
                sys.exit(0)

    # ── Step 1: Seed Prediction ──
    print('\n' + '─'*60)
    print(' STEP 1: PREDICTING OVERALL SEEDS (1-68)')
    print('─'*60)
    from ncaa_2026_model import run_prediction as predict_seeds
    predict_seeds(DATA_FILE)

    # ── Step 2: Bracket Simulation ──
    print('\n' + '─'*60)
    print(' STEP 2: BRACKET SIMULATION (Monte Carlo + Probabilities)')
    print('─'*60)
    from ncaa_bracket_predictor import run_prediction as predict_bracket
    predict_bracket(DATA_FILE)

    # ── Step 3: Generate Kaggle Submission ──
    print('\n' + '─'*60)
    print(' STEP 3: GENERATING KAGGLE SUBMISSION')
    print('─'*60)

    pred_path = os.path.join(DATA_DIR, 'bracket_2026_prediction.csv')
    if os.path.exists(pred_path):
        pred_df = pd.read_csv(pred_path)
        sub_path = os.path.join(DATA_DIR, 'submission_2026.csv')
        pred_df[['RecordID', 'Overall Seed']].to_csv(sub_path, index=False)
        print(f'  ✓ Kaggle submission saved: {sub_path}')
        print(f'    Format: RecordID, Overall Seed')
        print(f'    Rows: {len(pred_df)}')
    else:
        print('  ⚠ Prediction file not found — check for errors above')

    # ── Summary ──
    print('\n' + '='*60)
    print(' ALL OUTPUT FILES')
    print('='*60)
    outputs = [
        ('submission_2026.csv', 'Kaggle submission (RecordID, Overall Seed)'),
        ('bracket_2026_prediction.csv', 'Seed predictions'),
        ('bracket_2026_detailed.csv', 'Detailed with NET, conference, etc.'),
        ('bracket_2026_full.csv', 'Full bracket probabilities (all rounds)'),
    ]
    for fname, desc in outputs:
        fpath = os.path.join(DATA_DIR, fname)
        exists = '✓' if os.path.exists(fpath) else '✗'
        print(f'  {exists} {fname:<35} — {desc}')

    print(f'\n  Total time: {time.time()-t0:.0f}s')
    print('  Done!')


def print_data_instructions():
    """Print detailed instructions for filling in the data."""
    print('  ┌──────────────────────────────────────────────────────┐')
    print('  │          HOW TO FILL IN NCAA_2026_Data.csv           │')
    print('  └──────────────────────────────────────────────────────┘')
    print()
    print('  REQUIRED COLUMNS (20 total):')
    print('  ─────────────────────────────')
    print('  Column                Example Value        Where to Find')
    print('  ──────────────────    ─────────────────    ──────────────────')
    print('  RecordID              2025-26-Duke         Season-TeamName (no spaces)')
    print('  Season                2025-26              Always "2025-26"')
    print('  Team                  Duke                 Official team name')
    print('  Conference            ACC                  Team\'s conference')
    print('  Overall Seed          (leave blank)        We\'ll predict this!')
    print('  Bid Type              AL or AQ             AL=At-Large, AQ=Auto-Qualifier')
    print('  NET Rank              5                    NCAA NET Rankings page')
    print('  PrevNET               8                    NET rank from ~2 weeks prior')
    print('  AvgOppNETRank         20                   Avg opponent NET rank')
    print('  AvgOppNET             80                   Avg opponent NET rating')
    print('  WL                    28-3                 Overall W-L record')
    print('  Conf.Record           18-2                 Conference W-L record')
    print('  Non-ConferenceRecord  10-1                 Non-conf W-L record')
    print('  RoadWL                12-2                 Road W-L record')
    print('  NETSOS                10                   Strength of schedule (NET)')
    print('  NETNonConfSOS         120                  Non-conf SOS (NET)')
    print('  Quadrant1             12-3                 Q1 record (W-L)')
    print('  Quadrant2             6-0                  Q2 record (W-L)')
    print('  Quadrant3             5-0                  Q3 record (W-L)')
    print('  Quadrant4             5-0                  Q4 record (W-L)')
    print()
    print('  NOTES:')
    print('  • Include ALL 68 tournament teams (36 AQ + 32 AL)')
    print('  • You can also include non-tournament teams (no Bid Type)')
    print('    — they help with conference statistics')
    print('  • W-L records: use "W-L" format (e.g., "28-3")')
    print('  • RecordID format: "2025-26-TeamName" (remove spaces/dots)')
    print('  • Data source: NCAA.com NET Rankings + team schedule pages')
    print('  • PrevNET: if unavailable, use same as NET Rank')


if __name__ == '__main__':
    main()
