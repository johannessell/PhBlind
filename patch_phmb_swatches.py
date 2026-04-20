"""
patch_phmb_swatches.py
======================
contour_detection03 klassifiziert beide PHMB-Spalten (x=513 + x=620) als
'measure', weil die Sättigung kolumnen-global unter der Schwelle (60.0) liegt.
Optisch ist aber die linke PHMB-Spalte (x=513) der Farb-Swatch (Reihen 2-4
haben S=87..118), während die rechte (x=620) die Label-/Messspalte ist.

Dieses Skript
  1. setzt alle PHMB-Zellen mit x=513 auf is_color_cell=True
  2. setzt measure_col der PHMB-Parameter-Definition auf 6 (rechts)
  3. ruft danach patch_reference_values zum Wertpropagieren auf.
"""
import json
import subprocess
import sys

ref = json.load(open('reference.json', encoding='utf-8'))

flipped = 0
for c in ref['cells']:
    if c['parameter'] == 'PHMB' and c['x'] == 513:
        if not c['is_color_cell']:
            c['is_color_cell'] = True
            flipped += 1

for p in ref['parameters']:
    if p['name'] == 'PHMB':
        p['measure_col'] = 6

json.dump(ref, open('reference.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print(f"PHMB Zellen auf is_color_cell=True gesetzt: {flipped}")
print("PHMB measure_col -> 6")

# Werte jetzt auf die neuen Farb-Swatches uebertragen
subprocess.run([sys.executable, 'patch_reference_values.py'], check=True)
