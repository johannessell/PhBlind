"""
debug_reference_pipeline.py
===========================
Step-by-step diagnostic for the current reference_builder pipeline,
running against the close-up reference photos in reference_input/training/.

For every frame, dumps numbered overlay images for every pipeline stage
and prints a one-line summary with the count produced by each step.
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import reference_builder as RB  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'pipeline_debug')
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
os.makedirs(OUT_ROOT, exist_ok=True)

CELL_COLOR = {
    'all':       (0, 200, 255),   # yellow-orange
    'size':      (0, 165, 255),   # orange
    'dense':     (0, 220,   0),   # green
    'kept':      (0, 220,   0),
    'dropped':   (0,   0, 220),   # red
}


def _draw_oriented_rects(img_bgr, cells, color, label=None):
    """Draw rotated bounding rects for (cx, cy, long_s, short_s, area, ang)."""
    vis = img_bgr.copy()
    for c in cells:
        cx, cy, cw, ch, _area, ang = c
        rect = ((float(cx), float(cy)),
                (float(cw), float(ch)),
                float(ang))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(vis, [box], True, color, 2)
    if label:
        cv2.putText(vis, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis


def _draw_axis_rects(img_bgr, rects, color, label=None):
    """Draw axis-aligned (x, y, w, h) rects."""
    vis = img_bgr.copy()
    for x, y, w, h in rects:
        cv2.rectangle(vis, (int(x), int(y)),
                      (int(x + w), int(y + h)), color, 2)
    if label:
        cv2.putText(vis, label, (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return vis


def _unique_axis_groups(rects, axis, tol):
    """Mirror reference_builder._group_by_axis but only return count."""
    if not rects:
        return 0
    rs = sorted(rects, key=lambda r: r[axis])
    groups = [[rs[0]]]
    for r in rs[1:]:
        if abs(r[axis] - groups[-1][0][axis]) < tol:
            groups[-1].append(r)
        else:
            groups.append([r])
    return len(groups)


def process(path):
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(path)
    if bgr is None:
        return f'{base}: read failed'
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    summary = {'frame': base}

    # ---------- 01: Canny on full-res ----------
    edges = RB._compute_edges(gray)
    cv2.imwrite(os.path.join(out_dir, '01_canny.jpg'), edges)

    # ---------- 02: all cell rects ----------
    cells = RB._detect_cell_rects(edges)
    summary['raw_rects'] = len(cells)
    cv2.imwrite(os.path.join(out_dir, '02_all_cells.jpg'),
                _draw_oriented_rects(bgr, cells, CELL_COLOR['all'],
                                     f'raw rects = {len(cells)}'))

    # ---------- 03: size-filtered ----------
    sized = RB._filter_cells_by_size(cells)
    summary['size_filtered'] = len(sized)
    cv2.imwrite(os.path.join(out_dir, '03_size_filtered.jpg'),
                _draw_oriented_rects(bgr, sized, CELL_COLOR['size'],
                                     f'size-filtered = {len(sized)}'))

    # ---------- 04: dense filter ----------
    dense = RB._filter_dense_cells(sized, k_neighbors=4)
    summary['dense'] = len(dense)
    cv2.imwrite(os.path.join(out_dir, '04_dense.jpg'),
                _draw_oriented_rects(bgr, dense, CELL_COLOR['dense'],
                                     f'dense = {len(dense)}'))

    # ---------- 05: drop boundary outliers ----------
    kept, dropped = RB._drop_boundary_outliers(dense)
    summary['boundary_filtered'] = len(kept)
    vis = bgr.copy()
    for c in kept:
        cx, cy, cw, ch, _, ang = c
        box = cv2.boxPoints(((float(cx), float(cy)),
                             (float(cw), float(ch)),
                             float(ang))).astype(np.int32)
        cv2.polylines(vis, [box], True, CELL_COLOR['kept'], 2)
    for c in dropped:
        cx, cy, cw, ch, _, ang = c
        box = cv2.boxPoints(((float(cx), float(cy)),
                             (float(cw), float(ch)),
                             float(ang))).astype(np.int32)
        cv2.polylines(vis, [box], True, CELL_COLOR['dropped'], 2)
    cv2.putText(vis, f'kept={len(kept)}  dropped={len(dropped)}',
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                CELL_COLOR['kept'], 2)
    cv2.imwrite(os.path.join(out_dir, '05_grid_filtered.jpg'), vis)

    # ---------- 06: card quad ----------
    quad = RB._detect_reference_quad(gray, bgr)
    summary['quad'] = quad is not None
    if quad is not None:
        vis = bgr.copy()
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 220, 0), 4)
        for i, p in enumerate(quad.astype(int)):
            cv2.putText(vis, ['TL', 'TR', 'BR', 'BL'][i], tuple(p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
        cv2.imwrite(os.path.join(out_dir, '06_quad.jpg'), vis)
    else:
        return summary | {'failed_at': 'quad_detection'}

    # ---------- 07: warped canonical ----------
    warped_bgr, (cw, ch) = RB._warp_to_canonical(bgr, quad)
    summary['canonical_size'] = (cw, ch)
    cv2.imwrite(os.path.join(out_dir, '07_warped.jpg'), warped_bgr)

    # ---------- 08: cells inside warped image ----------
    warped_rects = RB._detect_cells(warped_bgr)
    summary['warped_cells'] = len(warped_rects)
    # Show with their detected x positions colour-coded so we can spot
    # if any column is missing entirely
    vis = warped_bgr.copy()
    for x, y, w, h in warped_rects:
        cv2.rectangle(vis, (int(x), int(y)),
                      (int(x + w), int(y + h)), (0, 220, 0), 2)
    cv2.putText(vis, f'warped_cells = {len(warped_rects)}',
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 220, 0), 2)
    cv2.imwrite(os.path.join(out_dir, '08_warped_cells.jpg'), vis)

    if len(warped_rects) < 6:
        return summary | {'failed_at': 'warped_cells'}

    # Count unique x-groups and y-groups (mirror reference_builder
    # tolerance: max(10, canonical_width * 0.03)).
    tol = max(10, int(cw * 0.03))
    summary['unique_x_groups'] = _unique_axis_groups(warped_rects, 0, tol)
    summary['unique_y_groups'] = _unique_axis_groups(warped_rects, 1, tol)

    # ---------- 09: build_grid output (the matrix fill) ----------
    grid, sorted_cols, col_positions, col_widths, \
        row_positions, row_heights = RB._build_grid(warped_rects, cw)
    rows = max(c['row_idx'] for c in grid) + 1
    cols = max(c['col_idx'] for c in grid) + 1
    summary['built_grid'] = f'{rows}x{cols}'
    summary['cell_count'] = len(grid)
    vis = warped_bgr.copy()
    for c in grid:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(vis, f"{c['row_idx']},{c['col_idx']}",
                    (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
    cv2.putText(vis, f'grid {rows}x{cols} ({len(grid)} cells)',
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 200, 0), 2)
    cv2.imwrite(os.path.join(out_dir, '09_built_grid.jpg'), vis)

    # ---------- 10: column classification ----------
    col_types, col_stats = RB._classify_columns(warped_bgr, sorted_cols)
    summary['col_types'] = dict(col_types)
    vis = warped_bgr.copy()
    for c in grid:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        t = col_types.get(c['col_idx'], '?')
        color = (0, 220, 0) if t == 'color' else (220, 120, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if c['row_idx'] == 0:
            cv2.putText(vis, t, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(os.path.join(out_dir, '10_col_types.jpg'), vis)

    n_color = sum(1 for v in col_types.values() if v == 'color')
    n_meas = sum(1 for v in col_types.values() if v == 'measure')
    summary['n_color_cols'] = n_color
    summary['n_measure_cols'] = n_meas
    return summary


def main():
    sources = sorted(glob.glob(os.path.join(REPO, 'reference_input',
                                            'training',
                                            'frame_17796*.jpg')))
    for src in sources:
        result = process(src)
        # Pretty one-line summary
        print('---')
        for k, v in result.items():
            print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
