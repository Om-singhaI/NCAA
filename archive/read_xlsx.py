#!/usr/bin/env python3
"""Read NCAA Statistics.xlsx and print contents."""
import zipfile
import xml.etree.ElementTree as ET

z = zipfile.ZipFile('NCAA Statistics.xlsx')
ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

# Parse shared strings
ss = []
try:
    sst = ET.parse(z.open('xl/sharedStrings.xml'))
    for si in sst.findall('.//main:si', ns):
        t = si.find('.//main:t', ns)
        ss.append(t.text if t is not None else '')
except:
    pass

# Parse sheet1
tree = ET.parse(z.open('xl/worksheets/sheet1.xml'))
rows = tree.findall('.//main:row', ns)

print(f'Total rows: {len(rows)}')
print(f'Shared strings: {len(ss)}')

# Print first 10 rows
for i, row in enumerate(rows[:10]):
    cells = row.findall('main:c', ns)
    vals = []
    for c in cells:
        typ = c.get('t', '')
        v = c.find('main:v', ns)
        val = v.text if v is not None else ''
        if typ == 's' and val.isdigit():
            idx = int(val)
            val = ss[idx] if idx < len(ss) else val
        vals.append(val)
    print(f'Row {i}: {vals}')
