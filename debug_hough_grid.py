"""
debug_hough_grid.py
===================
Prototype: find the card in the UNWARPED image using a color-aware
cell mask + connected-component cluster.

Cells are EITHER bright (label cells with dark text on white background)
OR saturated (color swatch cells). Their union is a reliable card
indicator regardless of strip-on-card or not.
"""

import glob
import os
import shutil
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, 'hough_debug')
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)

WORK_W = 960


def _to_work(bgr):
    h, w = bgr.shape[:2]
    if w > WORK_W * 1.25:
        s = WORK_W / float(w)
        small = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
        return small, s
    return bgr.copy(), 1.0


def _detect_card_quad(bgr):
    work, scale = _to_work(bgr)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    # Otsu on luminance: cell interior backgrounds
    _t, m_bright = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Saturation threshold (relative): color swatches
    sat = hsv[:, :, 1]
    sat_thresh = max(40, int(np.median(sat) * 1.4))
    m_sat = (sat > sat_thresh).astype(np.uint8) * 255
    mask = cv2.bitwise_or(m_bright, m_sat)
    # Erode 2 so cells stay disjoint along dark grid lines
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = float(h * w)
    cells = []
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < frame_area * 0.0005 or area > frame_area * 0.05:
            continue
        bx = int(stats[k, cv2.CC_STAT_LEFT])
        by = int(stats[k, cv2.CC_STAT_TOP])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        if min(bw, bh) < 8 or max(bw, bh) > min(h, w) * 0.25:
            continue
        ys, xs = np.where(labels[by:by + bh, bx:bx + bw] == k)
        pts = np.column_stack((xs + bx, ys + by)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        if min(rw, rh) < 6:
            continue
        ar = max(rw, rh) / max(min(rw, rh), 1e-3)
        if ar > 4.5:
            continue
        if area / max(rw * rh, 1) < 0.45:
            continue
        cells.append((float(rect[0][0]), float(rect[0][1]),
                      float(max(rw, rh)), float(min(rw, rh))))
    if len(cells) < 12:
        return None, work, mask, cells, None

    # Spatial union-find clustering
    pts = np.array([(c[0], c[1]) for c in cells], dtype=np.float32)
    sizes = np.array([c[2] for c in cells], dtype=np.float32)
    radius = float(np.median(sizes)) * 2.5
    n = len(cells)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.where(d < radius)[0]:
            if j > i:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    biggest_idx = max(groups.values(), key=len)
    if len(biggest_idx) < 12:
        return None, work, mask, cells, None
    cluster = pts[biggest_idx]

    # Rotated bbox via minAreaRect, expand by ~half cell short side.
    rect = cv2.minAreaRect(cluster)
    cell_short = float(np.median([c[3] for c in cells]))
    margin = cell_short * 0.6
    (cx, cy), (rw, rh), ang = rect
    rw += 2 * margin
    rh += 2 * margin
    box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)
    quad_full = (box / scale).astype(np.float32)
    return quad_full, work, mask, cells, cluster


def main():
    sources = (sorted(glob.glob(os.path.join(REPO, 'reference_input',
                                             'training', 'frame_17796*.jpg')))
               + sorted(glob.glob(os.path.join(REPO, 'test_input',
                                               'frame_*.jpg'))))
    for src in sources:
        base = os.path.splitext(os.path.basename(src))[0]
        out_dir = os.path.join(OUT, base)
        os.makedirs(out_dir, exist_ok=True)
        bgr = cv2.imread(src)
        quad, work, mask, cells, cluster = _detect_card_quad(bgr)
        cv2.imwrite(os.path.join(out_dir, '01_mask.jpg'), mask)
        cell_vis = work.copy()
        for cx, cy, lo, sh in cells:
            cv2.circle(cell_vis, (int(cx), int(cy)), 3, (0, 220, 220), -1)
        cv2.imwrite(os.path.join(out_dir, '02_cells.jpg'), cell_vis)
        if cluster is not None:
            clu_vis = work.copy()
            for p in cluster:
                cv2.circle(clu_vis, (int(p[0]), int(p[1])), 3, (0, 220, 0), -1)
            cv2.imwrite(os.path.join(out_dir, '03_cluster.jpg'), clu_vis)
        result = bgr.copy()
        if quad is not None:
            cv2.polylines(result, [quad.astype(np.int32)], True,
                          (0, 220, 0), 4)
        cv2.imwrite(os.path.join(out_dir, '04_quad.jpg'), result)
        print(f"{base}: cells={len(cells)} cluster="
              f"{0 if cluster is None else len(cluster)} "
              f"quad={'YES' if quad is not None else 'NO'}")


if __name__ == '__main__':
    main()
