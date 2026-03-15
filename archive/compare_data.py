"""Compare old vs new NCAA Statistics Excel files."""
import zipfile
import xml.etree.ElementTree as ET

def parse_xlsx(path):
    with zipfile.ZipFile(path) as z:
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        
        # shared strings (may not exist)
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            ss_xml = z.read('xl/sharedStrings.xml')
            ss_root = ET.fromstring(ss_xml)
            for si in ss_root.findall('.//s:si', ns):
                parts = []
                for t in si.findall('.//s:t', ns):
                    parts.append(t.text or '')
                strings.append(''.join(parts))
        
        sheet_xml = z.read('xl/worksheets/sheet1.xml')
        sheet_root = ET.fromstring(sheet_xml)
        rows_data = {}
        for row in sheet_root.findall('.//s:sheetData/s:row', ns):
            row_num = int(row.get('r'))
            cells = {}
            for cell in row.findall('s:c', ns):
                ref = cell.get('r')
                col = ''.join(c for c in ref if c.isalpha())
                t = cell.get('t', '')
                val_el = cell.find('s:v', ns)
                # Also check for inline string
                is_el = cell.find('s:is', ns)
                if t == 'inlineStr' and is_el is not None:
                    t_el = is_el.find('s:t', ns)
                    val = t_el.text if t_el is not None and t_el.text else ''
                elif val_el is not None and val_el.text is not None:
                    if t == 's' and strings:
                        val = strings[int(val_el.text)]
                    else:
                        val = val_el.text
                else:
                    val = ''
                cells[col] = val
            rows_data[row_num] = cells
        return rows_data

old = parse_xlsx('data/NCAA Statistics.xlsx')
new = parse_xlsx('data/NCAA Statistics 3_10.xlsx')

headers_old = old.get(1, {})
headers_new = new.get(1, {})
cols = sorted(set(list(headers_old.keys()) + list(headers_new.keys())))

def build_team_map(data):
    team_map = {}
    for row_num, cells in data.items():
        if row_num == 1:
            continue
        team = cells.get('A', '')
        if team:
            team_map[team] = cells
    return team_map

old_teams = build_team_map(old)
new_teams = build_team_map(new)

old_set = set(old_teams.keys())
new_set = set(new_teams.keys())
added = new_set - old_set
removed = old_set - new_set

if added:
    print(f"Teams ADDED in new file: {added}")
if removed:
    print(f"Teams REMOVED in new file: {removed}")
if not added and not removed:
    print("Same 365 teams in both files.")

changed_count = 0
changed_teams = {}
col_headers = {col: headers_new.get(col, col) for col in cols}

for team in sorted(old_set & new_set):
    old_cells = old_teams[team]
    new_cells = new_teams[team]
    diffs = []
    for col in cols:
        ov = old_cells.get(col, '')
        nv = new_cells.get(col, '')
        if ov != nv:
            header = col_headers.get(col, col)
            diffs.append((header, ov, nv))
    if diffs:
        changed_count += 1
        changed_teams[team] = diffs

print(f"\nTotal teams compared: {len(old_set & new_set)}")
print(f"Teams with changes: {changed_count}")
print(f"Teams unchanged: {len(old_set & new_set) - changed_count}")
print()

# Which columns changed most
col_change_count = {}
for team, diffs in changed_teams.items():
    for header, ov, nv in diffs:
        col_change_count[header] = col_change_count.get(header, 0) + 1

print("Columns that changed (# of teams affected):")
for col, cnt in sorted(col_change_count.items(), key=lambda x: -x[1]):
    print(f"  {col}: {cnt} teams")

# Show a few example teams with biggest changes
print("\n--- Sample changes (first 10 teams) ---")
count = 0
for team, diffs in sorted(changed_teams.items()):
    if count >= 10:
        break
    print(f"\n{team}:")
    for header, ov, nv in diffs:
        print(f"  {header}: {ov} -> {nv}")
    count += 1
