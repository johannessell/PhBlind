"""
On-device measurement pipeline.

Lifecycle:
  - Module import loads reference.json + template02.jpg from this dir.
  - Kotlin calls measure_rgba(rgba_bytes, width, height) on a captured frame.
  - Returns {found, results, quad} as a plain dict (Chaquopy converts to Java).
"""

import json
import os

import cv2
import numpy as np

from tracker import IndicatorTracker, QuadStabilityChecker

_HERE = os.path.dirname(__file__)
_STABILITY = QuadStabilityChecker(required_frames=5, max_drift=20.0)

with open(os.path.join(_HERE, 'reference.json'), 'r', encoding='utf-8') as _f:
    _REF = json.load(_f)

_TEMPLATE = cv2.imread(os.path.join(_HERE, 'template02.jpg'))
if _TEMPLATE is None:
    raise RuntimeError("template02.jpg konnte nicht geladen werden")
_TEMPLATE_GRAY = cv2.cvtColor(_TEMPLATE, cv2.COLOR_BGR2GRAY)
_TRACKER = IndicatorTracker(_TEMPLATE_GRAY)


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
    display orientation, runs IndicatorTracker + stability check.
    Returns coords in the upright (post-rotation) frame.

    TODO(stability+lag): on-device tracking is both less stable AND laggy
    compared to the desktop CLI. The two symptoms share a root cause
    (1080p Y plane into pixel-hardcoded tracker), so one fix (1a) helps
    both. Likely causes, in order of impact:

      1. Resolution mismatch. CLI ran with --scale 0.5 (~540p). Here we
         feed 1080p. Several constants in tracker.detect_card_by_cell_cluster
         are hardcoded in pixels, not frame-fraction:
             - radius = 80.0   (neighbor search for dense-cell filter)
             - min_cells = 6
             - _verify_quad: warp to 128x90, HoughLinesP thresholds
               (threshold=15, minLineLength=12, h>=20, v>=4)
         At 2x resolution the raster spacing grows, so 80 px no longer
         covers "2-4 cell widths". Quad flickers because the dense-cell
         set jitters frame-to-frame.

         Fix options:
           a) Downscale gray to ~720 px wide before tracker.find(), then
              scale quad back up by the same factor. Minimal code change.
           b) Make radius (and the other px constants in tracker.py)
              proportional to min(h, w) — cleaner long-term.

      2. Stability gate tuned for CLI fps. QuadStabilityChecker(required=5,
         max_drift=15) at ~30 fps = ~170 ms. On-device analysis runs slower
         (tracker per frame is heavy at 1080p), so 5 frames takes longer
         and user-hand drift accumulates. We already bumped max_drift=20;
         consider required=3 once (1) is fixed.

      3. Tracking runs on every analyzer frame. If CPU-bound, drop to
         every 2nd/3rd frame — gives tracker a bigger time budget and
         reduces jitter from partial results.

      4. Measure actual fps via a frame counter in Kotlin; only then tune
         required_frames. Guessing without numbers will churn.
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

    quad, method = _TRACKER.find(gray)
    stable = _STABILITY.update(quad)
    return {
        'found': quad is not None,
        'quad': quad.tolist() if quad is not None else None,
        'method': method,
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
