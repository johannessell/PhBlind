"""
On-device measurement pipeline.

Lifecycle:
  - Kotlin calls init(data_dir) after Python.start(). Data dir is the
    app's private filesDir/reference. If it contains reference.json +
    template.jpg, those are used; otherwise the bundled files next to
    this module are used as fallback.
  - Kotlin then calls find_quad_y / measure_rgba per frame.
  - After the user saves a new reference, Kotlin calls init(data_dir)
    again to reload.
"""

import json
import os

import cv2
import numpy as np

from tracker import IndicatorTracker, QuadStabilityChecker, order_quad_corners

_HERE = os.path.dirname(__file__)
_STABILITY = QuadStabilityChecker(required_frames=5, max_drift=20.0)

# Live-tracker target width. The live overlay only needs an approximate quad
# (capture-time precision comes from _TRACKER.find at full-res in measure_rgba).
# At ~480 px the simple "Canny -> largest 4-vertex polygon" detector runs in
# ~2 ms desktop / ~10 ms phone with IoU ~0.82 against the full-res quad —
# more than tight enough for a "you're aimed at the card" overlay.
_LIVE_TRACK_W = 480


def _largest_quad_from_contours(contours, frame_area: float, ar_lo: float,
                                ar_hi: float):
    """Return the largest convex 4-vertex polygon among contours, or None.
    Filters by area fraction (5%-99%) and aspect-ratio against the template."""
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.05 or area > frame_area * 0.99:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1:
            continue
        for eps_frac in (0.02, 0.03, 0.04, 0.06, 0.08):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(approx)
                rw, rh = rect[1]
                if min(rw, rh) < 1:
                    break
                ar = max(rw, rh) / max(min(rw, rh), 1e-6)
                if not (ar_lo <= ar <= ar_hi):
                    break
                if area > best_area:
                    best_area = area
                    best = approx
                break
    return best


def _find_quad_lowres(gray: np.ndarray):
    """Live-preview detector: downscale + Canny + largest 4-vertex contour.

    Two-pass:
      Pass 1 — plain Canny on the downscaled gray. Fast and precise when the
      card has a clear margin around it.
      Pass 2 (fallback) — pad the downscaled gray with a black border, Canny,
      then a small MORPH_CLOSE to bridge 1-px gaps in the outline. Recovers
      the contour when the card edges touch the frame border (close-up).

    Returns the quad in the original input-frame coordinates.
    """
    h, w = gray.shape[:2]
    if w > _LIVE_TRACK_W * 1.25:
        scale = _LIVE_TRACK_W / float(w)
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = gray

    tpl_ar = _TRACKER.aspect_ratio if _TRACKER is not None else 1.4
    ar_lo = tpl_ar * (1.0 - 0.35)
    ar_hi = tpl_ar * (1.0 + 0.35)

    # Pass 1: clean Canny (no padding, no morph close).
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    sh, sw = small.shape
    best = _largest_quad_from_contours(contours, sh * sw, ar_lo, ar_hi)

    pad = 0
    if best is None:
        # Pass 2: pad to recover edges that touch the frame, small close to
        # bridge 1-pixel gaps in the outline.
        pad = 10
        padded = cv2.copyMakeBorder(small, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=0)
        blurred = cv2.GaussianBlur(padded, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 90)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        ph, pw = padded.shape
        best = _largest_quad_from_contours(contours, ph * pw, ar_lo, ar_hi)

    if best is None:
        return None
    quad = order_quad_corners(best.reshape(4, 2).astype(np.float32))
    if pad:
        quad[:, 0] -= pad
        quad[:, 1] -= pad
    if scale != 1.0:
        quad = (quad / scale).astype(np.float32)
    return quad

_REF: dict = {}
_TEMPLATE = None
_TEMPLATE_GRAY = None
_TRACKER: IndicatorTracker = None  # type: ignore[assignment]
_LOADED_FROM: str = ''


def _resolve_paths(data_dir: str):
    """Prefer filesDir assets, fall back to bundled files in _HERE."""
    if data_dir:
        cand_ref = os.path.join(data_dir, 'reference.json')
        cand_tpl = os.path.join(data_dir, 'template.jpg')
        if os.path.exists(cand_ref) and os.path.exists(cand_tpl):
            return cand_ref, cand_tpl, 'filesDir'
    return (os.path.join(_HERE, 'reference.json'),
            os.path.join(_HERE, 'template02.jpg'),
            'bundled')


def init(data_dir: str = '') -> dict:
    """
    (Re)load reference.json + template from data_dir (filesDir) or bundled.
    Safe to call multiple times; resets the stability buffer.
    """
    global _REF, _TEMPLATE, _TEMPLATE_GRAY, _TRACKER, _LOADED_FROM

    ref_path, tpl_path, source = _resolve_paths(data_dir)
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    img = cv2.imread(tpl_path)
    if img is None:
        raise RuntimeError(f"Template nicht gefunden: {tpl_path}")

    _REF = ref
    _TEMPLATE = img
    _TEMPLATE_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _TRACKER = IndicatorTracker(_TEMPLATE_GRAY)
    _STABILITY.reset()
    _LOADED_FROM = source

    return {
        'source': source,
        'ref_path': ref_path,
        'tpl_path': tpl_path,
        'width': int(_TEMPLATE.shape[1]),
        'height': int(_TEMPLATE.shape[0]),
        'n_cells': len(_REF.get('cells', [])),
        'parameters': [p.get('name') for p in _REF.get('parameters', [])],
    }


# Load bundled defaults eagerly so the module is usable even if Kotlin
# forgets to call init().
init('')


def _measure_warped(warped: np.ndarray) -> dict:
    lab_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    color_cells = [c for c in _REF['cells'] if c['is_color_cell'] and c['value'] is not None]
    measure_cells = [c for c in _REF['cells'] if not c['is_color_cell']]
    param_meta = {p['name']: p for p in _REF['parameters']}
    name_to_ch = {'L': 0, 'A': 1, 'B': 2}
    out: dict = {}

    for param, meta in param_meta.items():
        p_colors = [c for c in color_cells if c['parameter'] == param]
        p_measure = [c for c in measure_cells if c['parameter'] == param]
        if not p_colors or not p_measure:
            continue

        labs, vals = [], []
        for cell in p_colors:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            labs.append([
                float(np.median(roi[:, :, 0])),
                float(np.median(roi[:, :, 1])),
                float(np.median(roi[:, :, 2])),
            ])
            vals.append(cell['value'])

        if len(labs) < 3:
            continue

        fixed_ch = meta.get('best_channel')
        ref_coeffs = meta.get('poly_coeffs')

        if fixed_ch in name_to_ch and ref_coeffs is not None:
            ch_idx = name_to_ch[fixed_ch]
            ch_name = fixed_ch

            ref_ch, tgt_ch, y_arr = [], [], []
            for cell, tgt_lab in zip(p_colors, labs):
                ref_lab = cell.get('lab_median')
                if ref_lab is None or ref_lab[ch_idx] is None:
                    continue
                ref_ch.append(ref_lab[ch_idx])
                tgt_ch.append(tgt_lab[ch_idx])
                y_arr.append(cell['value'])
            ref_ch = np.array(ref_ch, dtype=np.float64)
            tgt_ch = np.array(tgt_ch, dtype=np.float64)
            y_arr = np.array(y_arr, dtype=np.float64)

            if len(ref_ch) < 3 or tgt_ch.std() < 1e-6:
                continue

            r = float(np.corrcoef(tgt_ch, y_arr)[0, 1])
            ref_r = meta.get('best_r') or 0.0
            MIN_ABS_R = 0.70
            if ref_r != 0 and (np.sign(r) != np.sign(ref_r) or abs(r) < MIN_ABS_R):
                continue

            t2r = np.polyfit(tgt_ch, ref_ch, 1)
            coeffs = np.array(ref_coeffs, dtype=np.float64)
            pred_ref = np.polyval(t2r, tgt_ch)
            rmse = float(np.sqrt(np.mean(
                (np.polyval(coeffs, pred_ref) - y_arr) ** 2)))
        else:
            y_f = np.array(vals, dtype=np.float64)
            best_r, best_ch = 0.0, 1
            for ch in range(3):
                x = np.array([lab[ch] for lab in labs], dtype=np.float64)
                if x.std() < 1e-6:
                    continue
                rr = float(np.corrcoef(x, y_f)[0, 1])
                if abs(rr) > abs(best_r):
                    best_r, best_ch = rr, ch
            ch_idx = best_ch
            ch_name = {0: 'L', 1: 'A', 2: 'B'}[best_ch]
            r = best_r
            x_fit = np.array([lbl[ch_idx] for lbl in labs], dtype=np.float64)
            coeffs = np.polyfit(x_fit, y_f, min(2, len(x_fit) - 1))
            rmse = float(np.sqrt(np.mean(
                (np.polyval(coeffs, x_fit) - y_f) ** 2)))
            t2r = None

        probe_vals = []
        for cell in p_measure:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            probe_vals.append(float(np.median(roi[:, :, ch_idx])))
        if not probe_vals:
            continue

        probe_ch = float(np.mean(probe_vals))
        if t2r is not None:
            probe_ch = float(np.polyval(t2r, probe_ch))
        value = float(np.polyval(coeffs, probe_ch))
        ref_vals = sorted(vals)
        value = float(np.clip(value, ref_vals[0], ref_vals[-1]))

        out[param] = {
            'value': round(value, 2),
            'channel': ch_name,
            'r': round(r, 3),
            'rmse': round(rmse, 3),
        }

    return out


def find_quad_y(y_bytes: bytes, width: int, height: int,
                row_stride: int, rotation_deg: int) -> dict:
    """
    Lightweight per-frame tracker. Reads only the Y plane, rotates to
    display orientation, runs the low-res Canny detector + stability check.
    Returns coords in the upright (post-rotation) frame.

    Detection runs via _find_quad_lowres at ~480 px wide — the live overlay
    only needs an approximate quad (~IoU 0.82 against the precise full-res
    quad on test images). The capture path measure_rgba uses the full-res
    cell-cluster pipeline for precision.

    Open follow-ups:
      - Tune QuadStabilityChecker(required, max_drift) once on-device fps
        is measured. Currently required=5, max_drift=20.
      - Add an overlay guide rectangle in OverlayView so the user knows
        where to aim the phone — collapses detection to "is the card inside
        the guide?" and drops the search-the-frame problem entirely.
    """
    arr = np.frombuffer(y_bytes, dtype=np.uint8)
    if row_stride == width:
        gray = arr.reshape(height, width)
    else:
        gray = arr.reshape(height, row_stride)[:, :width]
    k = (-rotation_deg // 90) % 4
    if k:
        gray = np.rot90(gray, k=k)
    gray = np.ascontiguousarray(gray)

    quad = _find_quad_lowres(gray)
    stable = _STABILITY.update(quad)
    return {
        'found': quad is not None,
        'quad': quad.tolist() if quad is not None else None,
        'method': 'canny_lowres' if quad is not None else 'none',
        'stable': bool(stable),
        'progress': int(_STABILITY.progress()),
        'required': int(_STABILITY.required),
        'width': int(gray.shape[1]),
        'height': int(gray.shape[0]),
    }


def reset_stability() -> None:
    _STABILITY.reset()


def measure_rgba(rgba_bytes: bytes, width: int, height: int) -> dict:
    """
    rgba_bytes: tightly packed RGBA, len = width*height*4.
    Returns: {'found': bool, 'results': {param: {...}}, 'quad': [[x,y]*4] or None}
    """
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad, method = _TRACKER.find(gray)
    if quad is None:
        return {'found': False, 'results': {}, 'quad': None, 'method': method}
    warped = _TRACKER.warp(bgr, quad)
    results = _measure_warped(warped)
    return {
        'found': True,
        'results': results,
        'quad': quad.tolist(),
        'method': method,
    }
