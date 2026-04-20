"""
patch_reference_values.py
=========================
contour_detection03 schreibt OCR-Werte auf die Measure-Zellen (is_color_cell=False).
`measure_warped` in main_bounding_orb.py erwartet die Werte aber auf den Farb-
Swatches (is_color_cell=True) pro Zeile/Gruppe.

Dieses Skript propagiert pro (row_idx, group_idx) den gefundenen Messwert auf
alle Farb-Swatches derselben Zeile derselben Gruppe.
"""
import json

ref  = json.load(open('reference.json', encoding='utf-8'))

# (row_idx, group_idx) -> value, aus den bereits befüllten Measure-Zellen
row_group_value = {}
for c in ref['cells']:
    if not c['is_color_cell'] and c.get('value') is not None \
       and c.get('row_idx') is not None and c.get('group_idx') is not None:
        row_group_value[(c['row_idx'], c['group_idx'])] = c['value']

print(f"Gefundene (row, group) Werte: {len(row_group_value)}")

# Auf Farb-Swatches übertragen
patched = 0
for c in ref['cells']:
    if c['is_color_cell']:
        key = (c.get('row_idx'), c.get('group_idx'))
        v   = row_group_value.get(key)
        if v is not None:
            c['value'] = v
            patched += 1

json.dump(ref, open('reference.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)

cc = [c for c in ref['cells'] if c['is_color_cell']]
with_v = sum(1 for c in cc if c.get('value') is not None)
print(f"Farbzellen kalibriert: {with_v}/{len(cc)}  (gepatched: {patched})")
