"""
compute_best_channels.py
========================
Pro Parameter: bestimmt den besten LAB-Kanal, berechnet die
Polynomial-Koeffizienten und den Median-Kanalwert auf dem REFERENZBILD.

Schreibt in reference.json unter parameters[i]:
  best_channel       'L'|'A'|'B'
  best_r             Korrelation im Ref
  poly_coeffs        Poly-Fit (value = f(ch)) auf den Ref-Swatches
  ref_channel_median Median des Kanalwerts aller Swatches im Ref
  val_min, val_max   Fit-Range

Runtime (measure_warped):
  Pro Target-Bild den Kanalwert-Offset berechnen
  (Median der Ziel-Swatches - ref_channel_median)
  und vom Sample-Kanalwert abziehen → in ref-Koordinaten rechnen.
Das eliminiert Beleuchtungs-Drift ohne den Poly-Fit pro Bild neu
zu rechnen (was bei komprimierter Swatch-Dynamik instabil war).
"""
import cv2, json
import numpy as np

ref     = json.load(open('reference.json', encoding='utf-8'))
ref_img = cv2.imread(ref['image_file'])
lab     = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)

color_cells = [c for c in ref['cells'] if c['is_color_cell'] and c['value'] is not None]
names = {0: 'L', 1: 'A', 2: 'B'}

def roi_median(cell, ch):
    x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
    roi = lab[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    return float(np.median(roi[:, :, ch]))

# LAB-Median je Zelle speichern (fuer target->ref Transformation zur Laufzeit)
for c in ref['cells']:
    c['lab_median'] = [roi_median(c, ch) for ch in range(3)]

for p in ref['parameters']:
    pc = [c for c in color_cells if c['parameter'] == p['name']]
    if len(pc) < 3:
        print(f"{p['name']:5s}  --  nur {len(pc)} Farbzellen, kein Fit")
        for k in ('best_channel','best_r','poly_coeffs','val_min','val_max'):
            p[k] = None
        continue

    labs = np.array([[roi_median(c, ch) for ch in range(3)] for c in pc])
    vals = np.array([c['value'] for c in pc], dtype=np.float64)

    best_r, best_ch = 0.0, 1
    for ch in range(3):
        x_arr = labs[:, ch]
        if x_arr.std() < 1e-6: continue
        r = float(np.corrcoef(x_arr, vals)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_ch = r, ch

    x_arr  = labs[:, best_ch]
    degree = min(2, len(pc) - 1)
    coeffs = np.polyfit(x_arr, vals, degree).tolist()

    p['best_channel'] = names[best_ch]
    p['best_r']       = round(best_r, 3)
    p['poly_coeffs']  = [float(c) for c in coeffs]
    p['val_min']      = float(vals.min())
    p['val_max']      = float(vals.max())

    print(f"{p['name']:5s}  ch={names[best_ch]}  r={best_r:+.3f}  "
          f"refRange=[{x_arr.min():.0f}..{x_arr.max():.0f}]  "
          f"range={p['val_min']}..{p['val_max']}  n={len(pc)}")

json.dump(ref, open('reference.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
