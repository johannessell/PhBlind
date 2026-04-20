"""
test_diagnose_ph.py
===================
Pro raw_*.jpg und Parameter:
  - Sample-LAB (L, A, B) aus der Measure-Zelle
  - min/max Kanal-Werte der Farb-Swatches
  - gewaehlter Kanal + r + Wert

Ziel: sehen, ob pH-Varianz von LAB-Drift (Beleuchtung) oder
      Warp-Region (Alignment) kommt.
"""
import cv2, json, glob
import numpy as np
from tracker import IndicatorTracker

ref      = json.load(open('reference.json', encoding='utf-8'))
ref_img  = cv2.imread(ref['image_file'])
template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker  = IndicatorTracker(template)

color_cells   = [c for c in ref['cells'] if     c['is_color_cell'] and c['value'] is not None]
measure_cells = [c for c in ref['cells'] if not c['is_color_cell']]
parameters    = sorted({p['name'] for p in ref['parameters']})

def cell_lab(lab_img, cell):
    x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
    roi = lab_img[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    return (float(np.median(roi[:,:,0])),
            float(np.median(roi[:,:,1])),
            float(np.median(roi[:,:,2])))

def best_channel(vals, labs):
    y = np.array(vals, dtype=np.float64)
    best_r, best_ch = 0.0, 1
    for ch in range(3):
        x = np.array([l[ch] for l in labs])
        if x.std() < 1e-6: continue
        r = float(np.corrcoef(x, y)[0,1])
        if abs(r) > abs(best_r):
            best_r, best_ch = r, ch
    return best_ch, best_r

names = {0:'L',1:'A',2:'B'}

images = sorted(glob.glob('raw_*.jpg'))
for p in images:
    frame = cv2.imread(p)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    quad, method = tracker.find(gray)
    if quad is None:
        print(f"-- {p}  nicht erkannt"); continue
    warped = tracker.warp(frame, quad)
    lab    = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

    print(f"\n=== {p} [{method}] {warped.shape[1]}x{warped.shape[0]} ===")
    for param in parameters:
        pc = [c for c in color_cells   if c['parameter'] == param]
        pm = [c for c in measure_cells if c['parameter'] == param]
        if not pc or not pm:
            print(f"  {param:5s}  --  skip (color={len(pc)} measure={len(pm)})")
            continue

        labs, vals = [], []
        for c in pc:
            l = cell_lab(lab, c)
            if l is None: continue
            labs.append(l); vals.append(c['value'])
        if len(labs) < 3:
            print(f"  {param:5s}  --  nur {len(labs)} Farbzellen")
            continue

        ch_idx, r = best_channel(vals, labs)
        ch_name   = names[ch_idx]
        ch_vals   = np.array([l[ch_idx] for l in labs])
        y_vals    = np.array(vals)
        coeffs    = np.polyfit(ch_vals, y_vals, min(2, len(labs)-1))

        # Sample
        sample_labs = [cell_lab(lab, c) for c in pm]
        sample_labs = [l for l in sample_labs if l is not None]
        L_m = np.mean([s[0] for s in sample_labs])
        A_m = np.mean([s[1] for s in sample_labs])
        B_m = np.mean([s[2] for s in sample_labs])
        sample_ch = [L_m, A_m, B_m][ch_idx]
        value     = float(np.polyval(coeffs, sample_ch))
        clipped   = float(np.clip(value, min(vals), max(vals)))

        print(f"  {param:5s}  sample L={L_m:5.1f} A={A_m:5.1f} B={B_m:5.1f}  "
              f"ch={ch_name}(r={r:+.2f}) sample_{ch_name}={sample_ch:5.1f}  "
              f"swatch_{ch_name}=[{ch_vals.min():.0f}..{ch_vals.max():.0f}]  "
              f"raw={value:6.2f} clip={clipped:5.2f}")
