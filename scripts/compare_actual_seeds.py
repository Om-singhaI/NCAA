"""
Compare our predicted seeds vs actual 2026 NCAA tournament seeds.

Actual seeds sourced from Wikipedia / NCAA official release (March 15, 2026).
Our predictions from seed_selections_2026.txt (March 14, 2026).
"""
import numpy as np

# ── Our predicted seeds (from seed_selections_2026.txt) ──
OUR_PREDICTIONS = {
    'Duke': 1, 'Michigan': 2, 'Arizona': 3, 'Houston': 4, 'Florida': 5,
    'Iowa St.': 6, 'Purdue': 7, 'UConn': 8, 'Nebraska': 9, 'Illinois': 10,
    'Michigan St.': 11, 'Vanderbilt': 12, 'Virginia': 13, 'Arkansas': 14,
    'Alabama': 15, 'Kansas': 16, 'Texas Tech': 17, 'Gonzaga(AQ)': 18,
    'St. John\'s (NY)': 19, 'BYU': 20, 'Louisville': 21, 'Tennessee': 22,
    'Wisconsin': 23, 'North Carolina': 24, 'UCLA': 25, 'Kentucky': 26,
    'Ohio St.': 27, 'Saint Mary\'s (CA)': 28, 'Miami (FL)': 29, 'Clemson': 30,
    'Iowa': 31, 'Villanova': 32, 'Georgia': 33, 'TCU': 34, 'Utah St.': 35,
    'UCF': 36, 'Santa Clara': 37, 'NC State': 38, 'Auburn': 39,
    'Texas A&M': 40, 'Saint Louis': 41, 'SMU': 42, 'VCU': 43, 'Texas': 44,
    'Missouri': 45, 'South Fla.': 46, 'UNI(AQ)': 47, 'Yale': 48,
    'Miami (OH)': 49, 'Utah Valley': 50, 'SFA': 51, 'High Point(AQ)': 52,
    'Liberty': 53, 'Hofstra(AQ)': 54, 'UC Irvine': 55,
    'North Dakota St.(AQ)': 56, 'Troy(AQ)': 57, 'Wright St.(AQ)': 58,
    'Tennessee St.(AQ)': 59, 'Portland St.': 60, 'ETSU': 61,
    'Queens (NC)(AQ)': 62, 'Merrimack': 63, 'UMBC(AQ)': 64, 'LIU(AQ)': 65,
    'Howard': 66, 'Bethune-Cookman': 67, 'Lehigh(AQ)': 68,
}

# ── Actual seeds (from NCAA Selection Committee, March 15, 2026) ──
# Source: Wikipedia / SI.com official 1-68 ranking
ACTUAL_SEEDS = {
    'Duke': 1, 'Arizona': 2, 'Michigan': 3, 'Florida': 4, 'Houston': 5,
    'UConn': 6, 'Iowa State': 7, 'Purdue': 8, 'Michigan State': 9,
    'Illinois': 10, 'Gonzaga': 11, 'Virginia': 12, 'Nebraska': 13,
    'Alabama': 14, 'Kansas': 15, 'Arkansas': 16, 'Vanderbilt': 17,
    'St. John\'s': 18, 'Texas Tech': 19, 'Wisconsin': 20, 'Tennessee': 21,
    'North Carolina': 22, 'Louisville': 23, 'BYU': 24, 'Kentucky': 25,
    'Saint Mary\'s': 26, 'Miami (FL)': 27, 'UCLA': 28, 'Clemson': 29,
    'Villanova': 30, 'Ohio State': 31, 'Georgia': 32, 'Utah State': 33,
    'TCU': 34, 'Saint Louis': 35, 'Iowa': 36, 'Santa Clara': 37,
    'UCF': 38, 'Missouri': 39, 'Texas A&M': 40, 'NC State': 41,
    'Texas': 42, 'SMU': 43, 'Miami (OH)': 44, 'VCU': 45,
    'South Florida': 46, 'McNeese': 47, 'Akron': 48, 'Northern Iowa': 49,
    'High Point': 50, 'California Baptist': 51, 'Hofstra': 52, 'Troy': 53,
    'Hawai\'i': 54, 'North Dakota State': 55, 'Penn': 56, 'Wright State': 57,
    'Kennesaw State': 58, 'Tennessee State': 59, 'Idaho': 60, 'Furman': 61,
    'Queens': 62, 'Siena': 63, 'LIU': 64, 'Howard': 65,
    'UMBC': 66, 'Lehigh': 67, 'Prairie View A&M': 68,
}

# ── Name mapping: our names -> actual names ──
NAME_MAP = {
    'Iowa St.': 'Iowa State',
    'Michigan St.': 'Michigan State',
    'Gonzaga(AQ)': 'Gonzaga',
    'St. John\'s (NY)': 'St. John\'s',
    'Ohio St.': 'Ohio State',
    'Saint Mary\'s (CA)': 'Saint Mary\'s',
    'Utah St.': 'Utah State',
    'South Fla.': 'South Florida',
    'UNI(AQ)': 'Northern Iowa',
    'High Point(AQ)': 'High Point',
    'Hofstra(AQ)': 'Hofstra',
    'North Dakota St.(AQ)': 'North Dakota State',
    'Troy(AQ)': 'Troy',
    'Wright St.(AQ)': 'Wright State',
    'Tennessee St.(AQ)': 'Tennessee State',
    'Queens (NC)(AQ)': 'Queens',
    'UMBC(AQ)': 'UMBC',
    'LIU(AQ)': 'LIU',
    'Lehigh(AQ)': 'Lehigh',
}


def main():
    print('=' * 75)
    print(' NCAA 2026: PREDICTED vs ACTUAL SEEDS')
    print(' Predicted: March 14, 2026 | Actual: March 15, 2026 (Selection Sunday)')
    print('=' * 75)

    # ── 1. Field accuracy: which teams did we get right? ──
    our_teams = set()
    for name in OUR_PREDICTIONS:
        mapped = NAME_MAP.get(name, name)
        our_teams.add(mapped)

    actual_teams = set(ACTUAL_SEEDS.keys())

    correct_field = our_teams & actual_teams
    we_predicted_wrong = our_teams - actual_teams
    we_missed = actual_teams - our_teams

    print(f'\n  FIELD ACCURACY')
    print(f'  ──────────────')
    print(f'  Teams in both:    {len(correct_field)}/68 ({100*len(correct_field)/68:.1f}%)')
    print(f'  We predicted but NOT in actual ({len(we_predicted_wrong)}):')
    for t in sorted(we_predicted_wrong):
        # find our predicted seed
        for k, v in OUR_PREDICTIONS.items():
            mapped = NAME_MAP.get(k, k)
            if mapped == t:
                print(f'    Seed {v:2d}  {t}')
                break
    print(f'  In actual but we MISSED ({len(we_missed)}):')
    for t in sorted(we_missed):
        print(f'    Seed {ACTUAL_SEEDS[t]:2d}  {t}')

    # ── 2. Seed accuracy for overlapping teams ──
    print(f'\n  SEED-BY-SEED COMPARISON ({len(correct_field)} overlapping teams)')
    print(f'  {"Team":<25s} {"Pred":>5s} {"Actual":>6s} {"Diff":>5s} {"Seed Line":>10s}')
    print(f'  {"-"*25} {"-"*5} {"-"*6} {"-"*5} {"-"*10}')

    diffs = []
    abs_diffs = []
    seed_line_exact = 0
    exact_match = 0
    within_1 = 0
    within_2 = 0
    within_4 = 0

    comparisons = []
    for our_name, our_seed in sorted(OUR_PREDICTIONS.items(), key=lambda x: x[1]):
        actual_name = NAME_MAP.get(our_name, our_name)
        if actual_name not in ACTUAL_SEEDS:
            continue
        actual_seed = ACTUAL_SEEDS[actual_name]
        diff = our_seed - actual_seed
        comparisons.append((our_name, our_seed, actual_seed, diff))

    # Sort by actual seed for clean presentation
    comparisons.sort(key=lambda x: x[2])

    for our_name, our_seed, actual_seed, diff in comparisons:
        ad = abs(diff)
        diffs.append(diff)
        abs_diffs.append(ad)

        if ad == 0:
            exact_match += 1
        if ad <= 1:
            within_1 += 1
        if ad <= 2:
            within_2 += 1
        if ad <= 4:
            within_4 += 1

        our_line = ((our_seed - 1) // 4) + 1
        actual_line = ((actual_seed - 1) // 4) + 1
        if our_line == actual_line:
            seed_line_exact += 1
            line_str = f'{actual_line:2d} = {our_line:2d}  OK'
        else:
            line_str = f'{actual_line:2d} / {our_line:2d}  X'

        marker = ' <--' if ad > 5 else ('  *' if ad > 2 else '')
        sign = '+' if diff > 0 else (' ' if diff == 0 else '')
        display_name = our_name[:25]
        print(f'  {display_name:<25s} {our_seed:5d} {actual_seed:6d} {sign}{diff:>4d} {line_str}{marker}')

    n = len(diffs)
    diffs = np.array(diffs)
    abs_diffs = np.array(abs_diffs)

    # ── 3. Summary statistics ──
    print(f'\n  {"="*60}')
    print(f'  SUMMARY STATISTICS')
    print(f'  {"="*60}')
    print(f'  Overlapping teams:         {n}/68')
    print(f'  Exact match (diff=0):      {exact_match}/{n} ({100*exact_match/n:.1f}%)')
    print(f'  Within 1 seed:             {within_1}/{n} ({100*within_1/n:.1f}%)')
    print(f'  Within 2 seeds:            {within_2}/{n} ({100*within_2/n:.1f}%)')
    print(f'  Within 4 seeds:            {within_4}/{n} ({100*within_4/n:.1f}%)')
    print(f'  Seed LINE exact (1-17):    {seed_line_exact}/{n} ({100*seed_line_exact/n:.1f}%)')
    print()
    print(f'  Mean absolute error:       {abs_diffs.mean():.2f}')
    print(f'  Median absolute error:     {np.median(abs_diffs):.1f}')
    print(f'  RMSE:                      {np.sqrt(np.mean(diffs**2)):.3f}')
    print(f'  Max error:                 {abs_diffs.max()} seeds')
    print(f'  Mean signed error:         {diffs.mean():+.2f} (+ = we seeded higher/worse)')

    # ── 4. Biggest misses ──
    print(f'\n  BIGGEST MISSES (|diff| > 4):')
    misses = [(n, p, a, d) for n, p, a, d in comparisons if abs(d) > 4]
    misses.sort(key=lambda x: -abs(x[3]))
    if misses:
        for name, pred, actual, diff in misses:
            direction = 'too high (under-seeded)' if diff > 0 else 'too low (over-seeded)'
            print(f'    {name:<25s}  Pred={pred:2d}  Actual={actual:2d}  '
                  f'Off by {abs(diff):2d}  ({direction})')
    else:
        print(f'    None!')

    # ── 5. Analysis by seed tier ──
    print(f'\n  ACCURACY BY SEED TIER:')
    tiers = [
        ('Top seeds (1-4)', 1, 4),
        ('Seeds 5-16', 5, 16),
        ('Seeds 17-32', 17, 32),
        ('Seeds 33-48', 33, 48),
        ('Seeds 49-68', 49, 68),
    ]
    for label, lo, hi in tiers:
        tier_data = [(n, p, a, d) for n, p, a, d in comparisons if lo <= a <= hi]
        if tier_data:
            tier_abs = [abs(d) for _, _, _, d in tier_data]
            tier_exact = sum(1 for x in tier_abs if x == 0)
            tier_w2 = sum(1 for x in tier_abs if x <= 2)
            mae = np.mean(tier_abs)
            print(f'    {label:<22s}  n={len(tier_data):2d}  '
                  f'Exact={tier_exact}/{len(tier_data)}  '
                  f'Within2={tier_w2}/{len(tier_data)}  '
                  f'MAE={mae:.1f}')

    # ── 6. Kaggle-style evaluation (if we can) ──
    # Kaggle uses RMSE on overall seed for the 91 test teams
    # For the 2026 unseen data, we compare only overlapping teams
    print(f'\n  KAGGLE-STYLE METRICS (on {n} overlapping teams):')
    print(f'    RMSE:  {np.sqrt(np.mean(diffs**2)):.4f}')
    print(f'    MAE:   {abs_diffs.mean():.4f}')

    # ── 7. Auto-qualifier accuracy ──
    print(f'\n  AUTO-QUALIFIER ANALYSIS:')
    aq_teams = [n for n in OUR_PREDICTIONS if '(AQ)' in n or
                n in ['Howard', 'Bethune-Cookman', 'Liberty', 'Merrimack',
                       'Miami (OH)', 'Saint Louis', 'Utah Valley', 'Yale',
                       'UC Irvine', 'SFA', 'Portland St.', 'ETSU']]
    aq_correct = 0
    aq_wrong = 0
    for name in sorted(aq_teams):
        mapped = NAME_MAP.get(name, name)
        if mapped in ACTUAL_SEEDS:
            aq_correct += 1
        else:
            aq_wrong += 1
    # Count how many of 31 AQs we got right
    print(f'    AQ teams we predicted correctly in field: {aq_correct}')
    print(f'    AQ teams we predicted but NOT in actual:  {aq_wrong}')
    print(f'    (Total actual AQ teams: 31)')

    print(f'\n  Done!')


if __name__ == '__main__':
    main()
