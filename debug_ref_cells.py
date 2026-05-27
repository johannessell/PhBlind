"""Diagnostic: find the cell grid by projecting dark pixels onto each axis.
   Grid lines (horizontal + vertical) form sharp projection peaks; cells
   sit between them. Works regardless of cell content (labels OR colors).
"""
import glob
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import reference_builder as RB  # noqa: E402
from tracker import IndicatorTracker  # noqa: E402

OUT = os.path.join(REPO, 'ref_cells_debug')
os.makedirs(OUT, exist_ok=True)

py_dir = os.path.join(REPO, 'android', 'PoolWaterTester',
                      'app', 'src', 'main', 'python')
tpl = cv2.imread(os.path.join(py_dir, 'template02.jpg'))
tracker = IndicatorTracker(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))


def _find_grid_lines(projection, min_separation):
    """Return positions of local maxima in the 1-D projection, requiring
    at least `min_separation` apart and above a relative threshold."""
    if projection.size < 3:
        return []
    threshold = float(projection.max()) * 0.30
    smoothed = cv2.GaussianBlur(projection.astype(np.float32).reshape(-1, 1),
                                (1, 5), 0).ravel()
    candidates = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] < threshold:
            continue
        if smoothed[i] >= smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            candidates.append((i, smoothed[i]))
    # Greedy non-max suppression: keep strongest peaks, enforce min separation
    candidates.sort(key=lambda x: -x[1])
    kept = []
    for pos, _ in candidates:
        if all(abs(pos - k) >= min_separation for k in kept):
            kept.append(pos)
    kept.sort()
    return kept


def detect_cells_projection(warped_bgr):
    """Find horizontal + vertical grid lines via dark-pixel projections.
    Returns list of (x, y, w, h) axis-aligned cell rects."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # "darkness" map: how dark each pixel is. Grid lines + text are dark.
    dark = (255 - gray)

    # Project: row sums for HORIZONTAL grid lines, col sums for VERTICAL.
    row_dark = dark.sum(axis=1)
    col_dark = dark.sum(axis=0)

    # Cells are ~h/8 tall and ~w/7 wide. Grid lines must be at least
    # half a cell apart.
    min_h_sep = int(h / 16)
    min_v_sep = int(w / 14)

    h_lines = _find_grid_lines(row_dark, min_h_sep)   # y positions
    v_lines = _find_grid_lines(col_dark, min_v_sep)   # x positions

    rects = []
    for ri in range(len(h_lines) - 1):
        y0, y1 = h_lines[ri], h_lines[ri + 1]
        if y1 - y0 < 6:
            continue
        for ci in range(len(v_lines) - 1):
            x0, x1 = v_lines[ci], v_lines[ci + 1]
            if x1 - x0 < 6:
                continue
            rects.append((int(x0), int(y0),
                          int(x1 - x0), int(y1 - y0)))
    return rects, h_lines, v_lines


for path in sorted(glob.glob(os.path.join(REPO, 'test_input',
                                          'frame_*.jpg')))[:4]:
    base = os.path.splitext(os.path.basename(path))[0]
    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = RB._detect_reference_quad(gray, bgr)
    if quad is None:
        print(f'{base}: no quad'); continue
    warped = tracker.warp(bgr, quad)

    rects_old = RB._detect_cells(warped)
    vis_old = warped.copy()
    for x, y, w, h in rects_old:
        cv2.rectangle(vis_old, (x, y), (x + w, y + h), (0, 220, 0), 2)
    cv2.imwrite(os.path.join(OUT, f'{base}_old.jpg'), vis_old)

    rects_new, hl, vl = detect_cells_projection(warped)
    vis_new = warped.copy()
    for y in hl:
        cv2.line(vis_new, (0, y), (warped.shape[1] - 1, y), (0, 0, 255), 1)
    for x in vl:
        cv2.line(vis_new, (x, 0), (x, warped.shape[0] - 1), (0, 0, 255), 1)
    for x, y, w, h in rects_new:
        cv2.rectangle(vis_new, (x + 2, y + 2),
                      (x + w - 2, y + h - 2), (0, 220, 0), 1)
    cv2.imwrite(os.path.join(OUT, f'{base}_new.jpg'), vis_new)

    print(f'{base}: old={len(rects_old)} new={len(rects_new)} '
          f'(h={len(hl)} v={len(vl)})')
