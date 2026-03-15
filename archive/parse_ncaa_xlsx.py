#!/usr/bin/env python3
"""Parse NCAA Statistics.xlsx with inline string support and convert to CSV."""
import zipfile
import xml.etree.ElementTree as ET
import csv
import sys

z = zipfile.ZipFile('NCAA Statistics.xlsx')
ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
rows = tree.findall('.//main:row', ns)

all_data = []
for row in rows:
    cells = row.findall('main:c', ns)
    row_data = {}
    for c in cells:
        ref = c.get('r', '')  # e.g. "A1", "B2"
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
        
        # Extract column letter from ref
        col = ''.join(ch for ch in ref if ch.isalpha())
        row_data[col] = val
    all_data.append(row_data)

# Get all column letters in order
all_cols = []
for rd in all_data:
    for k in rd:
        if k not in all_cols:
            all_cols.append(k)

# Print header
headers = all_data[0] if all_data else {}
print("COLUMNS FOUND:")
for col_letter in all_cols:
    print(f"  {col_letter}: {headers.get(col_letter, '(no header)')}")

print(f"\nTotal rows (including header): {len(all_data)}")
print(f"Data rows: {len(all_data) - 1}")

# Print first 5 data rows
print("\nFIRST 5 TEAMS:")
header_vals = [headers.get(c, c) for c in all_cols]
print("  " + " | ".join(f"{h[:12]:>12}" for h in header_vals))
print("  " + "-" * (15 * len(all_cols)))
for row_data in all_data[1:6]:
    vals = [row_data.get(c, '') for c in all_cols]
    print("  " + " | ".join(f"{str(v)[:12]:>12}" for v in vals))

# Write to CSV
with open('ncaa_stats_raw.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([headers.get(c, c) for c in all_cols])
    for row_data in all_data[1:]:
        writer.writerow([row_data.get(c, '') for c in all_cols])

print(f"\nSaved to ncaa_stats_raw.csv")

# Count non-empty team rows
teams = [rd.get('A', '') for rd in all_data[1:] if rd.get('A', '').strip()]
print(f"Teams with names: {len(teams)}")

# Print last 5 teams
print("\nLAST 5 TEAMS:")
for row_data in all_data[-5:]:
    vals = [row_data.get(c, '') for c in all_cols]
    print("  " + " | ".join(f"{str(v)[:12]:>12}" for v in vals))
