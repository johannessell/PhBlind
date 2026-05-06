"""
debug_hough_compare.py
======================
Hough comparison: raw output (top) + per-segment-accepted (bottom) for each
parameter variant, stacked into a 2x2 grid per frame.

For each test_input/frame_*.jpg:
  - Set up crop + edges + card-frame info (same as _refine_quad_via_hough).
  - Run HoughLinesP for two parameter variants.
  - Top row: every returned segment drawn green.
  - Bottom row: only segments passing the angle + inner-bbox + side-position
    filter, drawn per side (top/bottom green, left/right yellow).
  - Saved as test_debug_out/<frame>/26b_raw_compare.jpg.

Run:  python debug_hough_compare.py
"""

from __future__ import annotations

import glob
import math
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import reference_builder  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'test_debug_out')

VARIANTS = [
    ('A: minLineLength=25, thr//2',
     dict(threshold_div=2, min_line_floor=25, min_line_force=True,
          max_line_gap=15)),
    ('B: thr//3, minLineLength=max(half,25)',
     dict(threshold_div=3, min_line_floor=25, min_line_force=False,
          max_line_gap=15)),
]


def _setup(gray, bgr):
    """Reproduce the crop + edges + card-frame info _refine_quad_via_hough has.

    Returns dict with all the data needed to replicate per-segment acceptance.
    """
    edges_full = reference_builder._compute_edges(gray)
    cells = reference_builder._detect_cell_rects(edges_full)
    if len(cells) < 6:
        return None
    sized = reference_builder._filter_cells_by_size(cells)
    if len(sized) < 6:
        return None
    cluster = reference_builder._filter_dense_cells(sized, k_neighbors=4)
    if len(cluster) < 6:
        return None

    sizes = np.array([max(c[2], c[3]) for c in cluster], dtype=np.float32)
    half_cell = int(np.median(sizes) // 2)
    cell_short = float(np.median([min(c[2], c[3]) for c in cluster]))

    cell_angles = np.array([c[5] for c in cluster], dtype=np.float32)
    angle_deg = float(np.median(cell_angles))

    h_img, w_img = gray.shape[:2]
    src_short = min(h_img, w_img)
    margin = max(100, src_short // 8)
    xs = [c[0] for c in cluster]
    ys = [c[1] for c in cluster]
    x0 = max(0, min(xs) - half_cell - margin)
    y0 = max(0, min(ys) - half_cell - margin)
    x1 = min(w_img, max(xs) + half_cell + margin)
    y1 = min(h_img, max(ys) + half_cell + margin)

    crop = gray[y0:y1, x0:x1]
    edges = edges_full[y0:y1, x0:x1]

    card_angle_rad = np.deg2rad(angle_deg)
    cos_t, sin_t = float(np.cos(card_angle_rad)), float(np.sin(card_angle_rad))
    cell_pts_crop = np.array([(c[0] - x0, c[1] - y0) for c in cluster],
                             dtype=np.float32)
    cell_centroid = cell_pts_crop.mean(axis=0)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    rotated_cells = (cell_pts_crop - cell_centroid) @ R.T
    cmin = rotated_cells.min(axis=0)
    cmax = rotated_cells.max(axis=0)

    return dict(
        crop=crop, edges=edges, cell_short=cell_short,
        cos_t=cos_t, sin_t=sin_t, cell_centroid=cell_centroid,
        cmin=cmin, cmax=cmax, card_angle_rad=card_angle_rad,
    )


def _categorize(lines, info):
    """Mirror of _refine_quad_via_hough's per-segment acceptance.

    Returns dict {side: [(x1,y1,x2,y2), ...]} per side."""
    cell_short = info['cell_short']
    cos_t, sin_t = info['cos_t'], info['sin_t']
    centroid = info['cell_centroid']
    cmin, cmax = info['cmin'], info['cmax']

    pad = 0.5 * cell_short
    inner_xmin = cmin[0] - pad
    inner_xmax = cmax[0] + pad
    inner_ymin = cmin[1] - pad
    inner_ymax = cmax[1] + pad
    margin_lo = 0.3 * cell_short
    margin_hi = 2.5 * cell_short
    h_target = info['card_angle_rad'] % np.pi
    v_target = (info['card_angle_rad'] + np.pi / 2) % np.pi
    angle_tol = np.deg2rad(15)

    def _to_card(x, y):
        dx, dy = x - centroid[0], y - centroid[1]
        return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t

    def _ang_dist(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    out = {'top': [], 'bottom': [], 'left': [], 'right': []}
    if lines is None:
        return out
    for line in lines:
        x1, y1, x2, y2 = (float(v) for v in line[0])
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        rx, ry = _to_card(mx, my)
        if inner_xmin <= rx <= inner_xmax and inner_ymin <= ry <= inner_ymax:
            continue
        seg_angle = float(math.atan2(y2 - y1, x2 - x1)) % np.pi
        seg_deg = np.degrees(seg_angle)
        is_h = seg_deg <= 15.0 or seg_deg >= 165.0
        is_v = 75.0 <= seg_deg <= 105.0
        if is_h and _ang_dist(seg_angle, h_target) < angle_tol:
            if cmin[1] - margin_hi <= ry <= cmin[1] - margin_lo:
                out['top'].append((x1, y1, x2, y2))
            elif cmax[1] + margin_lo <= ry <= cmax[1] + margin_hi:
                out['bottom'].append((x1, y1, x2, y2))
        elif is_v and _ang_dist(seg_angle, v_target) < angle_tol:
            if cmin[0] - margin_hi <= rx <= cmin[0] - margin_lo:
                out['left'].append((x1, y1, x2, y2))
            elif cmax[0] + margin_lo <= rx <= cmax[0] + margin_hi:
                out['right'].append((x1, y1, x2, y2))
    return out


def _draw_lines(crop, lines, color):
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if lines is None:
        return vis
    for line in lines:
        x1, y1, x2, y2 = (int(v) for v in line[0])
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
    return vis


def _label(img, text):
    cv2.rectangle(img, (0, 0), (min(420, img.shape[1]), 32), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)


def _resolve_params(cell_short, kw):
    threshold_norm = max(int(cell_short * 1.0), 30)
    min_line_norm = max(int(cell_short * 2.5), 50)
    relaxed_threshold = max(threshold_norm // kw['threshold_div'], 10)
    if kw['min_line_force']:
        relaxed_min_line = kw['min_line_floor']
    else:
        relaxed_min_line = max(min_line_norm // 2, kw['min_line_floor'])
    return relaxed_threshold, relaxed_min_line, kw['max_line_gap']


SIDE_COLOR = {'top': (0, 220, 0), 'bottom': (0, 220, 0),
              'left': (0, 200, 255), 'right': (0, 200, 255)}


def _draw_accepted(crop, accepted_by_side):
    vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for side, segs in accepted_by_side.items():
        c = SIDE_COLOR[side]
        for x1, y1, x2, y2 in segs:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
    return vis


def process(path):
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    info = _setup(gray, bgr)
    if info is None:
        return f'{base}: no cells'
    crop = info['crop']
    edges = info['edges']
    cell_short = info['cell_short']

    columns = []
    summary = [base]
    for label, kw in VARIANTS:
        thr, mll, mlg = _resolve_params(cell_short, kw)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                threshold=thr, minLineLength=mll,
                                maxLineGap=mlg)
        n_raw = 0 if lines is None else len(lines)
        accepted = _categorize(lines, info)
        n_top, n_bot = len(accepted['top']), len(accepted['bottom'])
        n_left, n_right = len(accepted['left']), len(accepted['right'])
        n_acc = n_top + n_bot + n_left + n_right

        raw_vis = _draw_lines(crop, lines, (0, 220, 0))
        _label(raw_vis, f'{label}  thr={thr} mll={mll}  raw n={n_raw}')

        acc_vis = _draw_accepted(crop, accepted)
        _label(acc_vis,
               f'accepted  T={n_top} B={n_bot} L={n_left} R={n_right} '
               f'(total {n_acc})')

        # stack raw above accepted as one column
        col = np.vstack([raw_vis, acc_vis])
        columns.append(col)
        summary.append(f'{label.split(":")[0]}: raw={n_raw} acc T{n_top}/B{n_bot}/L{n_left}/R{n_right}')

    h = max(c.shape[0] for c in columns)
    w = max(c.shape[1] for c in columns)
    def pad(img):
        out = np.zeros((h, w, 3), np.uint8)
        out[:img.shape[0], :img.shape[1]] = img
        return out
    columns = [pad(c) for c in columns]
    gap = np.zeros((h, 8, 3), np.uint8) + 60
    side_by_side = columns[0]
    for c in columns[1:]:
        side_by_side = np.hstack([side_by_side, gap, c])
    cv2.imwrite(os.path.join(out_dir, '26b_raw_compare.jpg'), side_by_side)
    return ' | '.join(summary)


def main():
    files = sorted(glob.glob(os.path.join(REPO, 'test_input', 'frame_*.jpg')))
    for path in files:
        print(process(path))


if __name__ == '__main__':
    main()
