#!/usr/bin/env python3
"""
Fill the Ours raw values of the two-sheet benchmark workbook (ISPD98 / Titan23;
hMETIS / KaHyPar columns and all formulas untouched) from a bench_best.py CSV and
relabel the system id in the note row.

    python3 patch_xlsx.py TEMPLATE.xlsx OURS.csv NEW_ID NEW.xlsx OLD_ID
"""

import csv
import sys

import openpyxl

OLD, CSV, NEW_ID, NEW, OLD_ID = sys.argv[1:6]

data = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        cut = int(r["cut"]) if r["cut"] else None
        data[(r["instance"], int(r["k"]), int(r["run"]))] = (cut, float(r["seconds"]))

wb = openpyxl.load_workbook(OLD)
for ws in wb:
    # The raw "Ours" block header: a cell whose value is exactly "Ours" on a row
    # that also holds "hMETIS" (the raw group header row of each k block).
    ours_col = None
    header_rows = []
    for row in ws.iter_rows():
        vals = {c.value for c in row}
        if "Ours" in vals and "hMETIS" in vals:
            header_rows.append(row[0].row)
            ours_col = next(c.column for c in row if c.value == "Ours")
    assert ours_col and len(header_rows) == 3, (ws.title, ours_col, header_rows)

    # k of each block, from the "k = N" title cell just above the header row.
    replaced = 0
    for hdr in header_rows:
        k = int(str(ws.cell(hdr - 1, 1).value).split("=")[1])
        r = hdr + 2
        while True:
            inst = ws.cell(r, 1).value
            if not inst or str(inst).startswith("Geomean"):
                break
            for seed in (1, 2, 3):
                cut, secs = data[(inst, k, seed)]
                ws.cell(r, ours_col + (seed - 1), cut)           # blank if infeasible
                ws.cell(r, ours_col + 3 + (seed - 1), round(secs, 2))
                replaced += 1
            r += 1
    print(f"{ws.title}: Ours raw col {ours_col}, {replaced} seed-cells rewritten")

    # Note row: swap the system id.
    for row in ws.iter_rows():
        for c in row:
            if isinstance(c.value, str) and OLD_ID in c.value:
                c.value = c.value.replace(OLD_ID, NEW_ID)

wb.save(NEW)
print("wrote", NEW)
