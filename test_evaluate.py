"""
test_evaluate.py
================
Batch-Pipeline: tracker.find → warp → measure_warped für alle raw_*.jpg.
Gibt Mess-Werte pro Parameter aus.
"""
import cv2, json, glob
from tracker import IndicatorTracker
from main_bounding_orb import measure_warped

ref      = json.load(open('reference.json', encoding='utf-8'))
ref_img  = cv2.imread(ref['image_file'])
template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker  = IndicatorTracker(template)

# Kalibrationsstand prüfen: measure_warped braucht `value` an Farbfeldern
cc      = [c for c in ref['cells'] if c['is_color_cell']]
with_v  = [c for c in cc if c.get('value') is not None]
print(f"Referenz: {len(with_v)}/{len(cc)} Farbzellen kalibriert")
print(f"Parameter: {[p['name'] for p in ref['parameters']]}\n")

images = sorted(glob.glob('raw_*.jpg'))
for p in images:
    frame = cv2.imread(p)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    quad, method = tracker.find(gray)
    if quad is None:
        print(f"  --  {p}  nicht erkannt")
        continue

    warped  = tracker.warp(frame, quad)
    out     = p.replace('raw_', 'warped_')
    cv2.imwrite(out, warped)
    results = measure_warped(warped, ref)

    if not results:
        print(f"  ??  {p}  [{method}] -> {out} ({warped.shape[1]}x{warped.shape[0]}) -- keine Messwerte (Kalibration fehlt)")
        continue

    parts = [f"{k}={v['value']} (ch={v['channel']} r={v['r']})" for k, v in sorted(results.items())]
    print(f"  OK  {p}  [{method}]  " + "  ".join(parts))
