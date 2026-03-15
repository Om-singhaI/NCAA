#!/usr/bin/env python3
"""
Convert NCAA Statistics.xlsx + ESPN Bracketology → model-ready NCAA_2026_Data.csv

ESPN Bracketology (Lunardi, updated 3/8/26):
  1-seeds: Duke, Florida, Michigan, Arizona
  
This script:
  1. Reads the 365-team Excel stats
  2. Tags the 68 projected tournament teams with Bid Type (AL/AQ)
  3. Renames columns to match model template
  4. Outputs data/NCAA_2026_Data.csv ready for predict_2026.py

Sources:
  - Stats: NCAA Statistics.xlsx (from NCAA NET rankings)
  - Bracket: ESPN Bracketology (Joe Lunardi, 3/8/2026)
"""

import zipfile
import xml.etree.ElementTree as ET
import csv
import os
import re

# ═══════════════════════════════════════════════════════════════
# ESPN BRACKETOLOGY (Lunardi, updated 3/8/2026)
# Format: (seed_line, team_name, bid_type)
# AQ = Auto-Qualifier (conference tournament champion)
# AL = At-Large
# ═══════════════════════════════════════════════════════════════

# Multi-bid conferences (from ESPN):
#   SEC: 11, Big Ten: 9, ACC: 8, Big 12: 8, Big East: 3, WCC: 3, A-10: 2
# Each multi-bid conf has 1 AQ (champion) + rest AL.
# Single-bid conferences: 1 AQ each.
#
# Conference tournament champions that are confirmed/projected AQ
# are marked. All other tournament teams from multi-bid conferences are AL.

BRACKET_TEAMS = [
    # ── EAST REGION ──
    # 1-seed line
    (1, "Duke", "AL"),           # ACC — likely AL (AQ may go to another ACC team)
    (16, "UMBC", "AQ"),          # America East champion
    (8, "Georgia", "AL"),        # SEC
    (9, "Ohio State", "AL"),     # Big Ten
    (5, "Arkansas", "AL"),       # SEC
    (12, "South Florida", "AQ"), # AAC champion
    (4, "Texas Tech", "AL"),     # Big 12
    (13, "SF Austin", "AQ"),     # WAC/Southland champion (Stephen F. Austin)
    (6, "Wisconsin", "AL"),      # Big Ten
    (11, "Santa Clara", "AL"),   # WCC — play-in (Last Four In)
    (11, "SMU", "AL"),           # ACC — play-in (Last Four In)
    (3, "Iowa State", "AL"),     # Big 12
    (14, "ETSU", "AQ"),          # SoCon champion
    (7, "Villanova", "AL"),      # Big East
    (10, "NC State", "AL"),      # ACC — Last Four Byes
    (15, "Tennessee St", "AQ"), # OVC champion
    (2, "Michigan St", "AL"),    # Big Ten (Michigan State)

    # ── SOUTH REGION ──
    (1, "Florida", "AL"),        # SEC
    (16, "B-CU", "AQ"),          # SWAC champion (Bethune-Cookman) — play-in
    (16, "Lehigh", "AQ"),        # Patriot League champion — play-in
    (8, "Utah State", "AQ"),     # MWC champion
    (9, "TCU", "AL"),            # Big 12
    (5, "St John's", "AL"),      # Big East (St. John's)
    (12, "Yale", "AQ"),          # Ivy League champion
    (4, "Virginia", "AL"),       # ACC
    (13, "Liberty", "AQ"),       # CUSA/ASUN champion
    (6, "Louisville", "AL"),     # ACC
    (11, "VCU", "AL"),           # A-10 — play-in (Last Four In)
    (11, "Auburn", "AL"),        # SEC — play-in (Last Four In)
    (3, "Nebraska", "AL"),       # Big Ten
    (14, "UC Irvine", "AQ"),     # Big West champion
    (7, "Saint Mary's", "AL"),   # WCC
    (10, "Missouri", "AL"),      # SEC — Last Four Byes
    (2, "Houston", "AL"),        # Big 12
    (15, "Portland St", "AQ"),   # Big Sky champion (Portland State)

    # ── MIDWEST REGION ──
    (1, "Michigan", "AL"),       # Big Ten
    (16, "Long Island", "AQ"),   # NEC champion (LIU) — play-in
    (16, "Howard", "AQ"),        # MEAC champion — play-in
    (8, "Clemson", "AL"),        # ACC
    (9, "Saint Louis", "AQ"),    # A-10 champion
    (5, "Vanderbilt", "AL"),     # SEC
    (12, "High Point", "AQ"),    # Big South champion
    (4, "Kansas", "AL"),         # Big 12
    (13, "N Dakota St", "AQ"),   # Summit League champion (North Dakota State)
    (6, "North Carolina", "AL"), # ACC
    (11, "Miami OH", "AQ"),      # MAC champion (Miami Ohio)
    (3, "Purdue", "AL"),         # Big Ten
    (14, "Hofstra", "AQ"),       # CAA champion
    (7, "Kentucky", "AL"),       # SEC
    (10, "Iowa", "AL"),          # Big Ten
    (2, "UConn", "AL"),          # Big East (Connecticut)
    (15, "Merrimack", "AQ"),     # NE Conference/Metro Atlantic champion

    # ── WEST REGION ──
    (1, "Arizona", "AL"),        # Big 12
    (16, "Queens", "AQ"),        # ASUN champion (Queens University)
    (8, "UCLA", "AL"),           # Big Ten
    (9, "Texas A&M", "AL"),      # SEC
    (5, "Tennessee", "AL"),      # SEC
    (12, "Northern Iowa", "AQ"), # MVC champion
    (4, "Gonzaga", "AL"),        # WCC
    (13, "Utah Valley", "AQ"),   # WAC champion
    (6, "BYU", "AL"),            # Big 12
    (11, "UCF", "AL"),           # Big 12 — Last Four Byes
    (3, "Alabama", "AL"),        # SEC
    (14, "Troy", "AQ"),          # Sun Belt champion
    (7, "Miami", "AL"),          # ACC
    (10, "Texas", "AL"),         # SEC — Last Four Byes
    (2, "Illinois", "AL"),       # Big Ten
    (15, "Wright St", "AQ"),     # Horizon League champion (Wright State)
]

# ═══════════════════════════════════════════════════════════════
# TEAM NAME MAPPING: ESPN name → Excel name
# The Excel uses names from NCAA NET rankings, ESPN uses short names.
# We need to match them.
# ═══════════════════════════════════════════════════════════════

ESPN_TO_EXCEL = {
    # Direct matches (most teams)
    "Duke": "Duke",
    "Michigan": "Michigan",
    "Arizona": "Arizona",
    "Florida": "Florida",
    "Georgia": "Georgia",
    "Ohio State": "Ohio St.",
    "Arkansas": "Arkansas",
    "Texas Tech": "Texas Tech",
    "Wisconsin": "Wisconsin",
    "Iowa State": "Iowa St.",
    "Villanova": "Villanova",
    "NC State": "NC State",
    "Michigan St": "Michigan St.",
    "TCU": "TCU",
    "St John's": "St. John's (NY)",
    "Virginia": "Virginia",
    "Louisville": "Louisville",
    "VCU": "VCU",
    "Auburn": "Auburn",
    "Nebraska": "Nebraska",
    "Saint Mary's": "Saint Mary's (CA)",
    "Missouri": "Missouri",
    "Houston": "Houston",
    "Clemson": "Clemson",
    "Saint Louis": "Saint Louis",
    "Vanderbilt": "Vanderbilt",
    "Kansas": "Kansas",
    "North Carolina": "North Carolina",
    "Purdue": "Purdue",
    "Kentucky": "Kentucky",
    "Iowa": "Iowa",
    "UConn": "UConn",
    "UCLA": "UCLA",
    "Texas A&M": "Texas A&M",
    "Tennessee": "Tennessee",
    "Gonzaga": "Gonzaga(AQ)",
    "BYU": "BYU",
    "UCF": "UCF",
    "Alabama": "Alabama",
    "Miami": "Miami (FL)",
    "Texas": "Texas",
    "Illinois": "Illinois",
    "Santa Clara": "Santa Clara",
    "SMU": "SMU",
    # AQ teams (smaller conferences)
    "UMBC": "UMBC(AQ)",
    "South Florida": "South Fla.",
    "SF Austin": "SFA",
    "ETSU": "ETSU",
    "B-CU": "Bethune-Cookman",
    "Lehigh": "Lehigh(AQ)",
    "Utah State": "Utah St.",
    "Yale": "Yale",
    "Liberty": "Liberty",
    "UC Irvine": "UC Irvine",
    "Portland St": "Portland St.",
    "Long Island": "LIU(AQ)",
    "Howard": "Howard",
    "High Point": "High Point(AQ)",
    "N Dakota St": "North Dakota St.(AQ)",
    "Miami OH": "Miami (OH)",
    "Hofstra": "Hofstra(AQ)",
    "Merrimack": "Merrimack",
    "Queens": "Queens (NC)(AQ)",
    "Northern Iowa": "UNI(AQ)",
    "Utah Valley": "Utah Valley",
    "Troy": "Troy(AQ)",
    "Wright St": "Wright St.(AQ)",
    "Tennessee St": "Tennessee St.(AQ)",
}


def parse_xlsx(filepath):
    """Parse NCAA Statistics.xlsx (inline strings, no shared strings)."""
    z = zipfile.ZipFile(filepath)
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
    rows = tree.findall('.//main:row', ns)

    all_data = []
    for row in rows:
        cells = row.findall('main:c', ns)
        row_data = {}
        for c in cells:
            ref = c.get('r', '')
            typ = c.get('t', '')
            if typ == 'inlineStr':
                is_elem = c.find('main:is', ns)
                if is_elem is not None:
                    t_elem = is_elem.find('main:t', ns)
                    val = t_elem.text if t_elem is not None else ''
                else:
                    val = ''
            else:
                v = c.find('main:v', ns)
                val = v.text if v is not None else ''
            col = ''.join(ch for ch in ref if ch.isalpha())
            row_data[col] = val
        all_data.append(row_data)

    # Get column letters in order
    all_cols = []
    for rd in all_data:
        for k in rd:
            if k not in all_cols:
                all_cols.append(k)

    # Convert to list of dicts with header names as keys
    headers = {col_letter: all_data[0].get(col_letter, col_letter) for col_letter in all_cols}
    result = []
    for row_data in all_data[1:]:
        row_dict = {}
        for col_letter in all_cols:
            header_name = headers[col_letter]
            row_dict[header_name] = row_data.get(col_letter, '')
        result.append(row_dict)

    return result


def make_record_id(team_name):
    """Generate RecordID from team name: 2025-26-TeamName (no spaces/dots)."""
    # Remove spaces, keep dots for consistency with training data
    clean = team_name.replace(' ', '').replace("'", "'")
    return f"2025-26-{clean}"


def main():
    xlsx_path = os.path.join(os.path.dirname(__file__), 'data', 'NCAA Statistics-3.xlsx')
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'NCAA_2026_Data.csv')

    print('=' * 60)
    print(' NCAA 2026 DATA CONVERTER')
    print(' Excel Stats + ESPN Bracketology -> Model-Ready CSV')
    print('=' * 60)

    # ── Step 1: Parse Excel ──
    print('\n  Parsing NCAA Statistics.xlsx...')
    excel_data = parse_xlsx(xlsx_path)
    print(f'  Found {len(excel_data)} teams in Excel')

    # Build lookup by team name
    excel_by_name = {}
    for row in excel_data:
        name = row.get('Team', '').strip()
        if name:
            excel_by_name[name] = row

    # ── Step 2: Match bracket teams to Excel data ──
    print(f'\n  Matching {len(BRACKET_TEAMS)} bracket teams to Excel stats...')

    matched = []
    unmatched = []
    for seed_line, espn_name, bid_type in BRACKET_TEAMS:
        excel_name = ESPN_TO_EXCEL.get(espn_name, espn_name)
        if excel_name in excel_by_name:
            row = excel_by_name[excel_name]
            matched.append((seed_line, espn_name, excel_name, bid_type, row))
        else:
            unmatched.append((seed_line, espn_name, excel_name, bid_type))

    print(f'  Matched: {len(matched)}/{len(BRACKET_TEAMS)}')
    if unmatched:
        print(f'  UNMATCHED ({len(unmatched)}):')
        for sl, en, xn, bt in unmatched:
            print(f'    Seed {sl}: "{en}" (tried "{xn}")')
            # Try fuzzy match
            candidates = [n for n in excel_by_name if en.lower() in n.lower() or n.lower() in en.lower()]
            if candidates:
                print(f'      Possible matches: {candidates}')
            else:
                # Try partial match
                parts = en.lower().split()
                for part in parts:
                    if len(part) > 3:
                        candidates = [n for n in excel_by_name if part in n.lower()]
                        if candidates:
                            print(f'      Partial matches for "{part}": {candidates[:5]}')

    # ── Step 3: Convert to model format ──
    print('\n  Converting to model format...')

    # Column mapping: Excel → Model template
    col_map = {
        'Team': 'Team',
        'Conference': 'Conference',
        'NET Rank': 'NET Rank',
        'PrevNET': 'PrevNET',
        'AvgOppNETRank': 'AvgOppNETRank',
        'AvgOppNET': 'AvgOppNET',
        'WL': 'WL',
        'ConfWL': 'Conf.Record',
        'NCWL': 'Non-ConferenceRecord',
        'RoadWL': 'RoadWL',
        'NETSOS': 'NETSOS',
        'NETNCSOS': 'NETNonConfSOS',
        'Q1': 'Quadrant1',
        'Q2': 'Quadrant2',
        'Q3': 'Quadrant3',
        'Q4': 'Quadrant4',
    }

    model_cols = [
        'RecordID', 'Season', 'Team', 'Conference', 'Overall Seed', 'Bid Type',
        'NET Rank', 'PrevNET', 'AvgOppNETRank', 'AvgOppNET',
        'WL', 'Conf.Record', 'Non-ConferenceRecord', 'RoadWL',
        'NETSOS', 'NETNonConfSOS', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4'
    ]

    output_rows = []

    # Add ALL teams from Excel (tournament + non-tournament)
    # Tournament teams get Bid Type; non-tournament teams get empty Bid Type
    tournament_excel_names = {excel_name for _, _, excel_name, _, _ in matched}

    for row in excel_data:
        team_name = row.get('Team', '').strip()
        if not team_name:
            continue

        out = {}
        out['RecordID'] = make_record_id(team_name)
        out['Season'] = '2025-26'
        out['Overall Seed'] = ''  # Model predicts this

        # Map columns
        for excel_col, model_col in col_map.items():
            out[model_col] = row.get(excel_col, '')

        # Set Bid Type for tournament teams
        if team_name in tournament_excel_names:
            # Find bid type
            for _, _, xn, bt, _ in matched:
                if xn == team_name:
                    out['Bid Type'] = bt
                    break
        else:
            out['Bid Type'] = ''

        output_rows.append(out)

    # ── Step 4: Write CSV ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=model_cols)
        writer.writeheader()
        writer.writerows(output_rows)

    # ── Summary ──
    n_tourn = sum(1 for r in output_rows if r['Bid Type'] in ('AL', 'AQ'))
    n_al = sum(1 for r in output_rows if r['Bid Type'] == 'AL')
    n_aq = sum(1 for r in output_rows if r['Bid Type'] == 'AQ')
    n_total = len(output_rows)

    print(f'\n  Output: {output_path}')
    print(f'  Total teams: {n_total}')
    print(f'  Tournament teams: {n_tourn} ({n_al} AL + {n_aq} AQ)')
    print(f'  Non-tournament: {n_total - n_tourn}')

    # Print tournament teams sorted by NET
    print(f'\n  TOURNAMENT FIELD ({n_tourn} teams):')
    print(f'  {"NET":>4} {"Team":<25} {"Conf":<12} {"Bid":<4} {"WL":<8}')
    print(f'  {"-"*4} {"-"*25} {"-"*12} {"-"*4} {"-"*8}')
    tourn_rows = [(r, int(r['NET Rank']) if r['NET Rank'] else 999)
                  for r in output_rows if r['Bid Type'] in ('AL', 'AQ')]
    tourn_rows.sort(key=lambda x: x[1])
    for r, net in tourn_rows:
        print(f'  {net:4d} {r["Team"]:<25} {r["Conference"]:<12} {r["Bid Type"]:<4} {r["WL"]:<8}')

    print(f'\n  Done! Next step: python predict_2026.py')


if __name__ == '__main__':
    main()
