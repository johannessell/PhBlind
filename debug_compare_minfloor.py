"""
Visualise the runtime-grid difference for frame_1778005034108 with and
without the 2000 px^2 floor applied to _detect_cell_rects in the warped
image. Saves side-by-side overlays so we can SEE which columns got
aliased onto each other.
"""
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402
import reference_builder as RB  # noqa: E402

M.init('')
FRAME = os.path.join(REPO, 'test_input', 'frame_1778005034108.jpg')
OUT = os.path.join(REPO, 'compare_minfloor')
os.makedirs(OUT, exist_ok=True)

bgr = cv2.imread(FRAME)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
quad = RB._detect_reference_quad(gray, bgr)
warped = M._TRACKER.warp(bgr, quad)
cv2.imwrite(os.path.join(OUT, '00_warped.jpg'), warped)


def draw_cells(warped, rects, label):
    vis = warped.copy()
    for x, y, w, h in rects:
        cv2.rectangle(vis, (int(x), int(y)),
                      (int(x + w), int(y + h)), (0, 220, 0), 2)
    cv2.putText(vis, f'{label}  n={len(rects)}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return vis


# Run _detect_cells in warped image (the function actually used to
# build the runtime grid) under the two regimes.
gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
edges_w = RB._compute_edges(gray_w)

raw_low = RB._detect_cell_rects(edges_w, min_cell_floor=0.0)
raw_high = RB._detect_cell_rects(edges_w, min_cell_floor=2000.0)

# Convert (cx, cy, lo, sh, area, ang) -> approx axis-aligned (x, y, w, h)
def to_axis(rects):
    out = []
    for cx, cy, lo, sh, _, _ in rects:
        x = int(round(cx - lo / 2)); y = int(round(cy - sh / 2))
        out.append((x, y, int(lo), int(sh)))
    return out

cv2.imwrite(os.path.join(OUT, '01_warped_cells_low_floor.jpg'),
            draw_cells(warped, to_axis(raw_low),
                       'WARPED detect (floor=0) -- previous behaviour'))
cv2.imwrite(os.path.join(OUT, '02_warped_cells_high_floor.jpg'),
            draw_cells(warped, to_axis(raw_high),
                       'WARPED detect (floor=2000) -- broken: wood-grain rule applied here'))

# Overlay the runtime grid slot positions
runtime_grid, (rows, cols), _ = M._detect_warped_grid(warped)
slot_vis = warped.copy()
for (r, c), s in sorted(runtime_grid.items()):
    cx, cy, w, h = s[0], s[1], s[2], s[3]
    x = int(round(cx - w / 2)); y = int(round(cy - h / 2))
    est = len(s) > 6 and s[6] == 'est'
    color = (0, 140, 255) if est else (0, 220, 220)
    cv2.rectangle(slot_vis, (x, y), (x + int(w), y + int(h)), color, 2)
    cv2.putText(slot_vis, f'{r},{c}', (x + 2, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
cv2.putText(slot_vis, 'runtime grid (yellow=detected, orange=estimated)',
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)
cv2.imwrite(os.path.join(OUT, '03_runtime_grid.jpg'), slot_vis)
print(f'Wrote 4 images to {OUT}')
print(f'  warped cells (floor=0):    {len(raw_low)}')
print(f'  warped cells (floor=2000): {len(raw_high)}')
print(f'  runtime grid: {rows}x{cols}')
