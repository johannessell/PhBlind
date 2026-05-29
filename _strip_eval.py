"""For each pH-strip frame, evaluate pH with EVERY projection method and
report which method is most consistent across frames."""
import cv2, sys, os, glob
import numpy as np
sys.path.insert(0, 'android/PoolWaterTester/app/src/main/python')
import measurement as M
import reference_builder as RB
M.init('')

PARAM = 'pH'
frames = sorted(glob.glob('reference_input/training/frame_1780073*.jpg'))

# collect per-method values across frames
by_method = {}

print(f"{'frame':>16s}  grid  " +
      "  ".join(f"{m:>8s}" for m in
               ['L', 'A', 'B', 'hue_lch', 'pc1_lab', 'pc1_ab']))

for fn in frames:
    base = os.path.basename(fn)[6:16]
    bgr = cv2.imread(fn)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = RB._detect_reference_quad(gray, bgr)
    if quad is None:
        print(f'{base:>16s}  NOQUAD'); continue
    warped = M._TRACKER.warp(bgr, quad)
    grid, (rows, cols), _ = M._detect_warped_grid(warped)
    lab_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

    color_cells = [c for c in M._REF['cells']
                   if c['is_color_cell'] and c['parameter'] == PARAM
                   and c['value'] is not None]
    measure_cells = [c for c in M._REF['cells']
                     if not c['is_color_cell'] and c['parameter'] == PARAM]

    def slot_for(c):
        if grid is not None:
            return grid.get((c['row_idx'], c['col_idx']))
        return (c['x'] + c['w'] / 2.0, c['y'] + c['h'] / 2.0, c['w'], c['h'])

    labs, vals = [], []
    for c in color_cells:
        s = slot_for(c)
        if s is None:
            continue
        lab = M._sample_lab_at(lab_warped, s)
        if lab is None:
            continue
        labs.append(lab); vals.append(c['value'])
    if len(labs) < 3:
        print(f'{base:>16s}  fewcolor'); continue
    labs_arr = np.array(labs, float); vals_arr = np.array(vals, float)
    cand = M._build_projections(labs_arr)

    probe_labs = []
    for c in measure_cells:
        s = slot_for(c)
        if s is None:
            continue
        lab = M._sample_lab_at(lab_warped, s)
        if lab is not None:
            probe_labs.append(lab)

    row = {}
    for name, (proj, desc) in cand.items():
        mask = ~np.isnan(proj)
        if int(mask.sum()) < 3:
            continue
        x = proj[mask]; y = vals_arr[mask]
        if x.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        sign = 1.0 if r >= 0 else -1.0
        xs = x * sign
        coeffs = np.polyfit(xs, y, min(2, len(xs) - 1))
        pv = []
        for lab in probe_labs:
            v = M._project_probe(lab, desc, sign)
            if v is not None and not np.isnan(v):
                pv.append(v)
        if not pv:
            continue
        val = float(np.polyval(coeffs, float(np.mean(pv))))
        val = float(np.clip(val, min(vals), max(vals)))
        row[name] = val
        by_method.setdefault(name, []).append(val)

    gs = f'{rows}x{cols}'
    cells = "  ".join(f"{row.get(m, float('nan')):8.2f}" for m in
                      ['L', 'A', 'B', 'hue_lch', 'pc1_lab', 'pc1_ab'])
    print(f'{base:>16s}  {gs:>5s}  {cells}')

print()
print(f"{'method':>10s}  {'n':>3s}  {'mean':>6s}  {'std':>6s}  {'min':>6s}  {'max':>6s}")
for m in ['L', 'A', 'B', 'hue_lch', 'pc1_lab', 'pc1_ab']:
    vals = by_method.get(m, [])
    if not vals:
        continue
    a = np.array(vals)
    print(f'{m:>10s}  {len(a):3d}  {a.mean():6.2f}  {a.std():6.2f}  '
          f'{a.min():6.2f}  {a.max():6.2f}')
