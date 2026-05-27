"""
debug_segmentation.py
=====================
Phase 1 diagnostic — threshold-based cell detection.

Otsu threshold makes each cell a clean bright connected component.
Erode slightly so cells don't bleed into each other through thin
grid-line gaps, then connected-components + per-component filtering.

For every test_input/frame_*.jpg dumps:
  00_input.jpg                480-wide gray
  01_otsu_mask.jpg            Otsu binary mask
  02_eroded.jpg               after erosion
  03_cell_candidates.jpg      bright dots = component centres surviving filter
  04_picked_quad.jpg          input + winning cluster's rotated bbox

Run:  python debug_segmentation.py
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'seg_debug_out')

ERODE_PX = 2
MIN_SHORT, MAX_SHORT = 6, 50         # px short side at 480 wide
MAX_AR = 4.5                          # aspect ratio cap
MIN_RECTANGULARITY = 0.55             # area / (minAreaRect area)
MIN_CLUSTER = 6                       # cells needed to form a card cluster


def _downscale(gray):
    h, w = gray.shape
    if w > M._LIVE_TRACK_W * 1.25:
        s = M._LIVE_TRACK_W / float(w)
        return cv2.resize(gray, (int(round(w * s)), int(round(h * s))),
                          interpolation=cv2.INTER_AREA)
    return gray


def _detect_cells_from_otsu(gray):
    """Returns list of (cx, cy, long, short, area, angle_deg)."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, mask = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if ERODE_PX > 0:
        mask_e = cv2.erode(mask, np.ones((ERODE_PX, ERODE_PX), np.uint8),
                           iterations=1)
    else:
        mask_e = mask
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        mask_e, connectivity=8)
    cells = []
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < MIN_SHORT * MIN_SHORT:
            continue
        ys, xs = np.where(labels == k)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        if min(rw, rh) < MIN_SHORT or max(rw, rh) > MAX_SHORT * 1.5:
            continue
        long_s = max(rw, rh)
        short_s = min(rw, rh)
        if short_s < MIN_SHORT or short_s > MAX_SHORT:
            continue
        if long_s / max(short_s, 1e-3) > MAX_AR:
            continue
        if area / max(rw * rh, 1) < MIN_RECTANGULARITY:
            continue
        ang = rect[2] + (90.0 if rw < rh else 0.0)
        while ang > 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        cx, cy = rect[0]
        cells.append((int(cx), int(cy), int(long_s), int(short_s),
                      float(area), float(ang)))
    return cells, mask, mask_e


def _cluster_by_proximity(cells, radius_factor=2.5):
    n = len(cells)
    if n == 0:
        return []
    pts = np.array([(c[0], c[1]) for c in cells], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cells], dtype=np.float32)
    radius = max(float(np.median(sizes)) * radius_factor, 30.0)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.where(d < radius)[0]:
            if j > i:
                union(i, int(j))
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _pick_best_cluster(cells, img_shape):
    clusters = _cluster_by_proximity(cells)
    sh, sw = img_shape[:2]
    img_center = np.array([sw * 0.5, sh * 0.5], dtype=np.float32)
    img_diag = float(np.hypot(sw, sh))
    best = None
    best_score = -1e9
    best_cells = []
    for idxs in clusters:
        if len(idxs) < MIN_CLUSTER:
            continue
        cs = [cells[i] for i in idxs]
        pts = np.array([(c[0], c[1]) for c in cs], dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        if min(rect[1]) < 5:
            continue
        cd = float(np.linalg.norm(
            np.array(rect[0], dtype=np.float32) - img_center)) / img_diag
        score = len(idxs) - 5.0 * cd
        if score > best_score:
            best_score = score
            best = rect
            best_cells = cs
    return best, best_cells


def process(path):
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(path)
    if bgr is None:
        return f'{base}: read failed'
    gray = _downscale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(out_dir, '00_input.jpg'), gray)

    cells, mask, mask_e = _detect_cells_from_otsu(gray)
    cv2.imwrite(os.path.join(out_dir, '01_otsu_mask.jpg'), mask)
    cv2.imwrite(os.path.join(out_dir, '02_eroded.jpg'), mask_e)

    vis_cells = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in cells:
        cv2.circle(vis_cells, (c[0], c[1]), 3, (0, 220, 220), -1)
    cv2.imwrite(os.path.join(out_dir, '03_cell_candidates.jpg'), vis_cells)

    rect, picked_cells = _pick_best_cluster(cells, gray.shape)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for c in picked_cells:
        cv2.circle(vis, (c[0], c[1]), 3, (0, 220, 0), -1)
    if rect is not None:
        margin = max(int(np.median([min(c[2], c[3]) for c in picked_cells])
                         * 0.6), 4)
        (cx, cy), (rw, rh), ang = rect
        rw += 2 * margin
        rh += 2 * margin
        box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
        cv2.polylines(vis, [box], True, (0, 220, 0), 2)
    cv2.imwrite(os.path.join(out_dir, '04_picked_quad.jpg'), vis)

    return f'{base}: cells={len(cells):3d} picked={len(picked_cells):3d}'


def main():
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPO, 'test_input',
                                          'frame_*.jpg')))
    for path in files:
        print(process(path))


if __name__ == '__main__':
    main()
