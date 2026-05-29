"""Step-by-step measurement trace for ONE frame and ONE parameter.

Shows:
  - runtime grid positions (row, col) -> (cx, cy, w, h)
  - per color-cell: (row, col, printed_value, sampled LAB, projected scalar)
  - polynomial fit + chosen projection
  - per measure-cell: (row, col, sampled LAB, projected scalar)
  - probe mean -> polynomial value -> clipped value
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402
import reference_builder as RB  # noqa: E402

FRAME = os.path.join(REPO, 'reference_input', 'training', 'frame_1780070551005.jpg')
PARAM = 'pH'

M.init('')

bgr = cv2.imread(FRAME)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
quad = RB._detect_reference_quad(gray, bgr)
if quad is None:
    raise SystemExit('quad detection failed')
warped = M._TRACKER.warp(bgr, quad)
print(f'warped shape: {warped.shape}')

runtime_grid, (gr_rows, gr_cols), _ = M._detect_warped_grid(warped)
print(f'runtime grid: {gr_rows} rows x {gr_cols} cols  '
      f'(expected {M._EXPECTED_ROWS}x{M._EXPECTED_COLS})')

# Snapshot what cells.json knows about pH
color_cells = [c for c in M._REF['cells']
               if c['is_color_cell'] and c['parameter'] == PARAM
               and c['value'] is not None]
measure_cells = [c for c in M._REF['cells']
                 if not c['is_color_cell'] and c['parameter'] == PARAM]
print(f'\nfrom reference.json for parameter {PARAM!r}:')
print(f'  color_cells: {len(color_cells)}')
print(f'  measure_cells: {len(measure_cells)}')

lab_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)


def slot_for(c):
    if runtime_grid is not None:
        return runtime_grid.get((c['row_idx'], c['col_idx']))
    return (c['x'] + c['w'] / 2.0, c['y'] + c['h'] / 2.0,
            c['w'], c['h'])


# ----------- Step 1: sample colour-cell LABs from current image ----------
print(f'\n--- step 1: sample LAB at each printed-value swatch ---')
labs, vals = [], []
print(f'{"row,col":>8s}  {"val":>5s}  {"slot (cx,cy,w,h)":>26s}  '
      f'{"L":>6s} {"A":>6s} {"B":>6s}')
for c in sorted(color_cells, key=lambda c: c['value']):
    slot = slot_for(c)
    if slot is None:
        print(f'  ({c["row_idx"]},{c["col_idx"]})  val={c["value"]}  '
              f'NO SLOT')
        continue
    lab = M._sample_lab_at(lab_warped, slot)
    if lab is None:
        print(f'  ({c["row_idx"]},{c["col_idx"]})  val={c["value"]}  '
              f'EMPTY ROI')
        continue
    cx, cy, w, h = slot[0], slot[1], slot[2], slot[3]
    print(f'  ({c["row_idx"]:2d},{c["col_idx"]:2d})  {c["value"]:5.2f}  '
          f'({cx:6.1f},{cy:6.1f},{w:5.1f},{h:5.1f})  '
          f'{lab[0]:6.1f} {lab[1]:6.1f} {lab[2]:6.1f}')
    labs.append(lab)
    vals.append(c['value'])

labs_arr = np.array(labs, dtype=np.float64)
vals_arr = np.array(vals, dtype=np.float64)

# ----------- Step 2: build projections ----------
print(f'\n--- step 2: candidate projections and their |r| against values ---')
candidates = M._build_projections(labs_arr)
print(f'{"projection":>12s}  {"r":>7s}  values')
evaluated = {}
for name, (proj, desc) in candidates.items():
    mask = ~np.isnan(proj)
    if int(mask.sum()) < 3:
        print(f'  {name:>12s}: too few non-nan')
        continue
    x = proj[mask].astype(np.float64)
    y = vals_arr[mask].astype(np.float64)
    if x.std() < 1e-6:
        print(f'  {name:>12s}: degenerate (zero std)')
        continue
    r = float(np.corrcoef(x, y)[0, 1])
    evaluated[name] = (r, x, y, mask, desc)
    print(f'  {name:>12s}:  r={r:+.3f}   '
          f'[{", ".join(f"{v:+.2f}" for v in x)}]')

# ----------- Step 3: pick projection ----------
print(f'\n--- step 3: pick projection by preference / global best ---')
prefs = M.DEFAULT_PROJECTION_PREFS.get(PARAM, ())
print(f'  preference order: {prefs}')
print(f'  MIN_R_PREFERRED: {M.MIN_R_PREFERRED}')
chosen = None
for pref in prefs:
    ev = evaluated.get(pref)
    if ev is not None and abs(ev[0]) >= M.MIN_R_PREFERRED:
        chosen = pref
        print(f'  -> picked PREFERRED {pref} (|r|={abs(ev[0]):.3f})')
        break
if chosen is None:
    chosen = max(evaluated, key=lambda n: abs(evaluated[n][0]))
    print(f'  -> picked GLOBAL BEST {chosen} (|r|='
          f'{abs(evaluated[chosen][0]):.3f})')

best_r, x_fit, y_fit, _msk, best_desc = evaluated[chosen]
sign = 1.0 if best_r >= 0 else -1.0
print(f'  sign = {sign:+.0f}')

# ----------- Step 4: polynomial fit ----------
x_fit_signed = x_fit * sign
coeffs = np.polyfit(x_fit_signed, y_fit, min(2, len(x_fit_signed) - 1))
rmse = float(np.sqrt(np.mean(
    (np.polyval(coeffs, x_fit_signed) - y_fit) ** 2)))
print(f'\n--- step 4: polynomial value = poly(projected) ---')
print(f'  coeffs (high->low): {coeffs}')
print(f'  rmse: {rmse:.3f}')
print(f'  printed values range: [{vals_arr.min()}, {vals_arr.max()}]')
print(f'  fit table:')
print(f'    {"x_signed":>10s}  {"printed":>8s}  {"predicted":>10s}')
for xi, yi in zip(x_fit_signed, y_fit):
    pred = float(np.polyval(coeffs, xi))
    print(f'    {xi:+10.3f}  {yi:8.2f}  {pred:10.3f}')

# ----------- Step 5: probe (measure cells) ----------
print(f'\n--- step 5: probe (measure) cells ---')
print(f'{"row,col":>8s}  {"slot":>26s}  '
      f'{"L":>6s} {"A":>6s} {"B":>6s}  {"proj_signed":>11s}')
probe_projs = []
for c in measure_cells:
    slot = slot_for(c)
    if slot is None:
        print(f'  ({c["row_idx"]},{c["col_idx"]})  NO SLOT')
        continue
    lab = M._sample_lab_at(lab_warped, slot)
    if lab is None:
        print(f'  ({c["row_idx"]},{c["col_idx"]})  EMPTY')
        continue
    v = M._project_probe(lab, best_desc, sign)
    cx, cy, w, h = slot[0], slot[1], slot[2], slot[3]
    print(f'  ({c["row_idx"]:2d},{c["col_idx"]:2d})  '
          f'({cx:6.1f},{cy:6.1f},{w:5.1f},{h:5.1f})  '
          f'{lab[0]:6.1f} {lab[1]:6.1f} {lab[2]:6.1f}  '
          f'{v:+11.3f}')
    if v is not None and not np.isnan(v):
        probe_projs.append(v)

# ----------- Step 6: final value ----------
print(f'\n--- step 6: final ---')
print(f'  probe projections: {probe_projs}')
print(f'  mean projected:   {np.mean(probe_projs):+.3f}')
value = float(np.polyval(coeffs, float(np.mean(probe_projs))))
print(f'  poly value:       {value:.3f}')
ref_vals_sorted = sorted(vals)
clipped = float(np.clip(value, ref_vals_sorted[0], ref_vals_sorted[-1]))
print(f'  clipped to [{ref_vals_sorted[0]}, {ref_vals_sorted[-1]}]: {clipped}')
