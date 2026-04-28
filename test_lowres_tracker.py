"""
test_lowres_tracker.py
======================
Prototype a minimal live-tracker pipeline at very low resolution.

Goal: find the indicator card with a "biggest 4-vertex polygon" approach,
no cell-cluster logic. The live overlay only needs an approximate quad —
the precise warp happens via reference_builder on the full-res capture.

Strategy variants we try here:
  v1  Canny -> findContours -> approxPolyDP -> largest convex 4-vertex
  v2  Canny + MORPH_CLOSE   -> same
  v3  Otsu  -> findContours -> same

Run:
    python test_lowres_tracker.py
Outputs go to test_lowres_out/<basename>/ — _v1.jpg, _v2.jpg, _v3.jpg.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import time
from typing import Optional, Tuple

import cv2
import numpy as np


TARGET_W = 480


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


def downscale(gray: np.ndarray, target_w: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    if w <= target_w * 1.25:
        return gray, 1.0
    s = target_w / w
    return cv2.resize(gray, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA), s


def _largest_quad_from_contours(contours, frame_area: float) -> Optional[np.ndarray]:
    """Pick the largest 4-vertex convex polygon among the contours."""
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.05 or area > frame_area * 0.95:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1:
            continue
        for eps_frac in (0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if area > best_area:
                    best_area = area
                    best = approx
                break
    if best is None:
        return None
    return order_quad_corners(best.reshape(4, 2).astype(np.float32))


def detect_v1_canny(gray: np.ndarray) -> Optional[np.ndarray]:
    """Plain Canny -> findContours -> largest 4-vertex polygon."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    return _largest_quad_from_contours(contours, h * w)


def detect_v4_pad_canny_close(gray: np.ndarray) -> Optional[np.ndarray]:
    """Pad with black border (recovers contours that touch the frame edge),
    Canny, small MORPH_CLOSE to bridge 1-pixel gaps, then largest 4-vertex.
    Designed to work both for far-away (card occupies <50%) and close-up
    (card fills the frame) shots.
    """
    pad = 10
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=0)
    blurred = cv2.GaussianBlur(padded, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = padded.shape
    quad = _largest_quad_from_contours(contours, h * w)
    if quad is None:
        return None
    quad[:, 0] -= pad
    quad[:, 1] -= pad
    return quad


def detect_v2_canny_close(gray: np.ndarray) -> Optional[np.ndarray]:
    """Canny + MORPH_CLOSE to bridge gaps in the card outline."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    return _largest_quad_from_contours(contours, h * w)


def detect_v3_otsu(gray: np.ndarray) -> Optional[np.ndarray]:
    """Otsu threshold (card brighter than bg) -> contours -> largest 4-vertex."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fk = max(15, min(gray.shape) // 20)
    if fk % 2 == 0:
        fk += 1
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                              np.ones((fk, fk), np.uint8), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    quad = _largest_quad_from_contours(contours, h * w)
    if quad is not None:
        return quad
    # invert (dark card on bright bg)
    closed_inv = cv2.morphologyEx(cv2.bitwise_not(m), cv2.MORPH_CLOSE,
                                  np.ones((fk, fk), np.uint8), iterations=1)
    contours, _ = cv2.findContours(closed_inv, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    return _largest_quad_from_contours(contours, h * w)


def detect_v5_v1_then_v4(gray: np.ndarray) -> Optional[np.ndarray]:
    """Try clean Canny first; fall back to pad+close only if it fails.
    Far-away (clean edges) hits the v1 path: precise quad, no merging with
    background. Close-up (edges touch frame) falls through to v4: padding
    closes the loop, small morph close bridges 1-px gaps in the outline.
    """
    quad = detect_v1_canny(gray)
    if quad is not None:
        return quad
    return detect_v4_pad_canny_close(gray)


VARIANTS = [
    ('v1_canny', detect_v1_canny),
    ('v4_pad_canny_close', detect_v4_pad_canny_close),
    ('v5_v1_then_v4', detect_v5_v1_then_v4),
]


def _ground_truth_quad(bgr: np.ndarray) -> Optional[np.ndarray]:
    """Use reference_builder's full-res detector as ground truth."""
    import sys
    sys.path.insert(0,
        r'c:/Users/johan/PycharmProjects/PhBlind/android/PoolWaterTester/app/src/main/python')
    import reference_builder  # noqa: E402
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return reference_builder._detect_reference_quad(gray, bgr)


def _quad_iou(q1: np.ndarray, q2: np.ndarray, w: int, h: int) -> float:
    """Approximate IoU via rasterization — handles rotated quads."""
    a = np.zeros((h, w), np.uint8)
    b = np.zeros((h, w), np.uint8)
    cv2.fillPoly(a, [q1.astype(np.int32)], 255)
    cv2.fillPoly(b, [q2.astype(np.int32)], 255)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return inter / union if union > 0 else 0.0


def run_one(path: str, out_dir: str):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray, scale = downscale(gray_full, TARGET_W)

    cv2.imwrite(os.path.join(out_dir, '00_input_small.jpg'), gray)

    # Ground truth (full-res reference_builder quad), scaled into the small frame
    gt_full = _ground_truth_quad(bgr)
    gt_small = (gt_full * scale).astype(np.float32) if gt_full is not None else None
    if gt_small is not None:
        vis_gt = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis_gt, [gt_small.astype(np.int32)], True, (255, 200, 0), 2)
        cv2.putText(vis_gt, 'ground truth (reference_builder, full-res)',
                    (10, vis_gt.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, '_ground_truth.jpg'), vis_gt)

    results = {}
    for name, fn in VARIANTS:
        # warm-up + 5 timed runs
        fn(gray)
        t0 = time.perf_counter()
        for _ in range(5):
            quad = fn(gray)
        dt_ms = (time.perf_counter() - t0) * 1000.0 / 5

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if gt_small is not None:
            cv2.polylines(vis, [gt_small.astype(np.int32)], True, (255, 200, 0), 1)
        iou = float('nan')
        if quad is not None:
            cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 2)
            for p, lab in zip(quad.astype(int), ['TL', 'TR', 'BR', 'BL']):
                cv2.putText(vis, lab, tuple(p), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
            if gt_small is not None:
                iou = _quad_iou(quad, gt_small, gray.shape[1], gray.shape[0])
        else:
            cv2.putText(vis, 'NO QUAD', (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
        label = f'{name} {dt_ms:.1f}ms IoU={iou:.2f}' if not np.isnan(iou) \
            else f'{name} {dt_ms:.1f}ms'
        cv2.putText(vis, label, (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f'{name}.jpg'), vis)

        results[name] = {'found': quad is not None, 'ms': dt_ms, 'iou': iou}
    return results


def main():
    files = sorted(glob.glob('test_input/*.jpg') + glob.glob('test_input/*.png'))
    skip_re = re.compile(r'^(0[1-9]|[12]\d|30|40)[a-z]?_')
    files = [p for p in files
             if not skip_re.match(os.path.basename(p))
             or os.path.basename(p).startswith('00_input')]

    if not files:
        print('No images in test_input/. Drop captures there first.')
        return

    out_root = 'test_lowres_out'
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    os.makedirs(out_root, exist_ok=True)

    print(f'TARGET_W = {TARGET_W} px')
    print(f'{"image":35s} ' + ' '.join(f'{n:>22s}' for n, _ in VARIANTS))
    summary = {n: {'hits': 0, 'ms': 0.0, 'iou_sum': 0.0, 'iou_n': 0}
               for n, _ in VARIANTS}
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        sub = os.path.join(out_root, base)
        os.makedirs(sub, exist_ok=True)
        res = run_one(path, sub)
        if res is None:
            continue
        cells = []
        for n, _ in VARIANTS:
            r = res[n]
            iou_str = f'IoU{r["iou"]:.2f}' if not np.isnan(r['iou']) else '       '
            cells.append(f'{("OK" if r["found"] else "--"):>2s} {r["ms"]:5.1f}ms {iou_str}')
            summary[n]['hits'] += int(r['found'])
            summary[n]['ms'] += r['ms']
            if not np.isnan(r['iou']):
                summary[n]['iou_sum'] += r['iou']
                summary[n]['iou_n'] += 1
        print(f'{base:35s} ' + ' '.join(cells))

    n_total = len(files)
    print()
    for n, _ in VARIANTS:
        s = summary[n]
        avg_iou = s['iou_sum'] / s['iou_n'] if s['iou_n'] else float('nan')
        print(f'  {n:20s}  hits={s["hits"]}/{n_total}  '
              f'avg={s["ms"] / n_total:.1f}ms  avg_IoU={avg_iou:.3f}')


if __name__ == '__main__':
    main()
