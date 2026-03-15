#!/usr/bin/env python3
"""Read NCAA Statistics.xlsx - full debug."""
import zipfile
import xml.etree.ElementTree as ET

z = zipfile.ZipFile('NCAA Statistics.xlsx')
ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

# List all files
print("Files in xlsx:")
for f in z.namelist():
    print(f"  {f}")

# Try to find shared strings
try:
    raw = z.read('xl/sharedStrings.xml').decode('utf-8')
    print(f"\nShared strings XML (first 2000 chars):\n{raw[:2000]}")
except KeyError:
    print("\nNo sharedStrings.xml found")

# Parse sheet1 - show raw XML for first row to understand structure
raw_sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
print(f"\nSheet1 XML (first 3000 chars):\n{raw_sheet[:3000]}")
