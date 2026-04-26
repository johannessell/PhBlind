# HYBRID PIPELINE (FINAL): CELLS + CONSTRAINED RANSAC
# - Cells define global card geometry (stable)
# - RANSAC refines each edge locally (robust)
# - No Canny, no Hough

import cv2
import numpy as np
from typing import Optional, Tuple, List

CANONICAL_WIDTH = 720

# -----------------------------
# Utils
# -----------------------------

def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.empty((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return np.array([x, y], dtype=np.float32)


# -----------------------------
# CELL DETECTION
# -----------------------------

def detect_cells(gray: np.ndarray):
    gray = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)

    gx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    frame_area = h * w

    cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (frame_area * 0.0001 < area < frame_area * 0.02):
            continue

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if min(rw, rh) < 5:
            continue

        ar = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ar > 4:
            continue

        cx, cy = rect[0]
        cells.append((cx, cy, max(rw, rh), min(rw, rh)))

    return cells


# -----------------------------
# DENSE FILTER
# -----------------------------

def filter_dense_cells(cells, k=4):
    if len(cells) < k:
        return []

    pts = np.array([(c[0], c[1]) for c in cells])
    sizes = np.array([c[2] for c in cells])
    radius = np.median(sizes) * 2.5

    keep = []
    for i, p in enumerate(pts):
        d = np.linalg.norm(pts - p, axis=1)
        if np.sum(d < radius) >= k:
            keep.append(cells[i])

    return keep


# -----------------------------
# ROI EDGE POINTS
# -----------------------------

def extract_roi_points(gray, cells):
    pts = np.array([(c[0], c[1]) for c in cells])
    sizes = np.array([c[2] for c in cells])

    half = int(np.median(sizes))

    x0 = int(max(0, pts[:, 0].min() - half))
    y0 = int(max(0, pts[:, 1].min() - half))
    x1 = int(min(gray.shape[1], pts[:, 0].max() + half))
    y1 = int(min(gray.shape[0], pts[:, 1].max() + half))

    roi = gray[y0:y1, x0:x1]

    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(mag, 40, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(mask > 0)
    points = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)

    return points


# -----------------------------
# RANSAC (FAST VERSION)
# -----------------------------

def fit_line_ransac(points, iterations=1000, threshold=3.0):
    best_line = None
    best_count = 0

    n = len(points)
    if n < 2:
        return None

    for _ in range(iterations):
        i1, i2 = np.random.choice(n, 2, replace=False)
        p1, p2 = points[i1], points[i2]

        dx, dy = p2 - p1
        norm = np.hypot(dx, dy)
        if norm < 1e-6:
            continue

        a, b = dy, -dx
        c = dx * p1[1] - dy * p1[0]

        d = np.abs(a * points[:, 0] + b * points[:, 1] + c) / norm
        count = np.sum(d < threshold)

        if count > best_count:
            best_count = count
            best_line = (a, b, c)

    return best_line


# -----------------------------
# CONSTRAINED RANSAC PER EDGE
# -----------------------------

def points_near_line(points, p1, p2, max_dist):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = np.hypot(dx, dy)

    a = dy
    b = -dx
    c = dx * p1[1] - dy * p1[0]

    d = np.abs(a * points[:, 0] + b * points[:, 1] + c) / norm
    return points[d < max_dist]


# -----------------------------
# MAIN DETECTOR
# -----------------------------

def detect_card_quad(gray: np.ndarray) -> Optional[np.ndarray]:
    cells = detect_cells(gray)
    if len(cells) < 6:
        return None

    cluster = filter_dense_cells(cells)
    if len(cluster) < 6:
        return None

    pts = np.array([(c[0], c[1]) for c in cluster], dtype=np.float32)

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    cell_quad = order_quad_corners(box)

    points = extract_roi_points(gray, cluster)
    if len(points) < 200:
        return cell_quad

    refined_lines = []

    for i in range(4):
        p1 = cell_quad[i]
        p2 = cell_quad[(i + 1) % 4]

        side_pts = points_near_line(points, p1, p2, max_dist=30)

        if len(side_pts) < 50:
            return cell_quad

        line = fit_line_ransac(side_pts)
        if line is None:
            return cell_quad

        refined_lines.append(line)

    tl = intersect(refined_lines[0], refined_lines[3])
    tr = intersect(refined_lines[1], refined_lines[0])
    br = intersect(refined_lines[2], refined_lines[1])
    bl = intersect(refined_lines[3], refined_lines[2])

    if any(p is None for p in [tl, tr, br, bl]):
        return cell_quad

    quad = np.array([tl, tr, br, bl], dtype=np.float32)
    return order_quad_corners(quad)


# -----------------------------
# WARP
# -----------------------------

def warp_to_canonical(bgr: np.ndarray, quad: np.ndarray):
    pts = quad.astype(np.float32)

    w = int(max(np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[3] - pts[0]), np.linalg.norm(pts[2] - pts[1])))

    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(pts, dst)

    return cv2.warpPerspective(bgr, M, (w, h))


# -----------------------------
# PROCESS / MAIN
# -----------------------------

import os
import sys
import glob


def process(image_path: str, out_root: str = 'test_detect_out') -> bool:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f'[{base}] read failed')
        return False

    src_h, src_w = bgr.shape[:2]
    cv2.imwrite(os.path.join(out_dir, '00_input.jpg'), bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, '01_gray.jpg'), gray)

    quad = detect_card_quad(gray)
    if quad is None:
        print(f'[{base}] FAILED: no quad')
        return False

    vis = bgr.copy()
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 6)
    for i, p in enumerate(quad.astype(int)):
        cv2.putText(vis, ['TL', 'TR', 'BR', 'BL'][i], tuple(p),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.imwrite(os.path.join(out_dir, '30_quad.jpg'), vis)

    warped = warp_to_canonical(bgr, quad)
    cv2.imwrite(os.path.join(out_dir, '40_warped.jpg'), warped)

    print(f'[{base}] OK  src={src_w}x{src_h}  quad-area={int(cv2.contourArea(quad))}')
    return True


def main(argv):
    if len(argv) > 1:
        targets = argv[1:]
    else:
        os.makedirs('test_input', exist_ok=True)
        all_files = sorted(glob.glob('test_input/*.jpg') + glob.glob('test_input/*.png'))

        import re
        skip_re = re.compile(r'^(0[1-9]|[12]\d|30|40)[a-z]?_')
        targets = [p for p in all_files if not skip_re.match(os.path.basename(p))]

        if not targets:
            print('No capture images in test_input/.')
            return

    out_root = 'test_detect_out'
    os.makedirs(out_root, exist_ok=True)

    results = []
    for path in targets:
        base = os.path.splitext(os.path.basename(path))[0]
        sub = os.path.join(out_root, base)
        if os.path.isdir(sub):
            import shutil
            shutil.rmtree(sub)

        ok = process(path, out_root=out_root)
        results.append((os.path.basename(path), ok))

    n_ok = sum(1 for _, ok in results if ok)
    print(f'\n=== Summary: {n_ok}/{len(results)} OK ===')
    for name, ok in results:
        flag = 'OK ' if ok else 'XX '
        print(f'  {flag} {name}')


if __name__ == '__main__':
    main(sys.argv)
