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

from tracker import IndicatorTracker, QuadStabilityChecker

_HERE = os.path.dirname(__file__)
_STABILITY = QuadStabilityChecker(required_frames=5, max_drift=20.0)

# Detection runs at this width — tracker.py's px-hardcoded thresholds
# (radius=80, _verify_quad min-line-length etc.) were tuned at ~720p.
# Phone preview is typically 1080p+ so we scale down before find(), then
# scale the quad back up so the returned coordinates match the caller's
# original frame (used for the preview overlay).
_DETECT_TARGET_W = 720


def _detect_at_reduced_scale(gray: np.ndarray):
    """Run _TRACKER.find on a downscaled copy; return (quad_full, method).

    Quad is rescaled to the input frame's coordinate system. If the input
    is already <= target width we skip the resize.
    """
    h, w = gray.shape[:2]
    if w > _DETECT_TARGET_W * 1.25:
        scale = _DETECT_TARGET_W / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = gray

    quad_small, method = _TRACKER.find(small)
    if quad_small is None:
        return None, method
    if scale == 1.0:
        return quad_small, method
    quad_full = (quad_small / scale).astype(np.float32)
    return quad_full, method

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
    display orientation, runs IndicatorTracker + stability check.
    Returns coords in the upright (post-rotation) frame.

    Detection is run via _detect_at_reduced_scale (downscale to ~720 px wide)
    so tracker.py's px-hardcoded thresholds match the resolution they were
    tuned at. The returned quad is in the original (post-rotation) frame.

    Open follow-ups (in order of likely impact):
      - Tune QuadStabilityChecker(required, max_drift) once on-device fps
        is measured. Currently required=5, max_drift=20.
      - If still CPU-bound, skip every other analyzer frame.
      - Long-term: make tracker.detect_card_by_cell_cluster scale-invariant
        (radius proportional to median cell size, like reference_builder
        does) so the explicit downscale isn't needed.
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

    quad, method = _detect_at_reduced_scale(gray)
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
    quad, method = _detect_at_reduced_scale(gray)
    if quad is None:
        return {'found': False, 'results': {}, 'quad': None, 'method': method}
    # Warp the full-res BGR with the upscaled quad — keep original color
    # resolution for measurement; only detection ran at reduced scale.
    warped = _TRACKER.warp(bgr, quad)
    results = _measure_warped(warped)
    return {
        'found': True,
        'results': results,
        'quad': quad.tolist(),
        'method': method,
    }
