#!/usr/bin/env python3
"""Quick check: compare old vs new Excel columns and row counts."""
import zipfile
import xml.etree.ElementTree as ET

def get_headers(filepath):
    z = zipfile.ZipFile(filepath)
    ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
    rows = tree.findall('.//main:row', ns)
    
    # Get header row
    headers = []
    cells = rows[0].findall('main:c', ns) if rows else []
    for c in cells:
        if c.get('t', '') == 'inlineStr':
            is_elem = c.find('main:is', ns)
            if is_elem is not None:
                t_elem = is_elem.find('main:t', ns)
                headers.append(t_elem.text if t_elem is not None else '')
    return headers, len(rows) - 1  # subtract header

old_h, old_n = get_headers('data/NCAA Statistics.xlsx')
new_h, new_n = get_headers('data/NCAA Statistics 3_10.xlsx')

print(f"OLD file: {old_n} teams, {len(old_h)} columns")
print(f"  Columns: {old_h}")
print(f"\nNEW file: {new_n} teams, {len(new_h)} columns")
print(f"  Columns: {new_h}")

if old_h != new_h:
    print("\n  COLUMN DIFFERENCES:")
    old_set, new_set = set(old_h), set(new_h)
    added = new_set - old_set
    removed = old_set - new_set
    if added: print(f"    Added: {added}")
    if removed: print(f"    Removed: {removed}")
else:
    print("\n  Columns are IDENTICAL")
