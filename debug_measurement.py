"""
debug_measurement.py
====================
One-shot diagnostic for the on-device measurement pipeline.

For each frame in test_input/frame_*.jpg:
  1. Run reference_builder._detect_reference_quad with set_debug_dir() so we
     keep the existing 20-27 stage images.
  2. Warp the BGR with the detected quad.
  3. Save extra debug visuals:
       30_warped_with_cells.jpg   warped + every cell ROI (color=green, measure=blue)
       31_cell_swatches.jpg       runtime swatch (top) vs reference swatch (bottom)
       32_lab_per_cell.txt        per-cell L/A/B medians, runtime vs reference
       33_per_param.txt           per-parameter regression diagnostics
  4. Print a single-line summary per frame.

No edits to the Android pipeline — this is diagnostic only.

Run:  python debug_measurement.py
"""

from __future__ import annotations

import glob
import os
import shutil
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import reference_builder  # noqa: E402
from tracker import IndicatorTracker  # noqa: E402

OUT_ROOT = os.path.join(_HERE, 'test_debug_out')


def _load_reference():
    """Mirror measurement.init() but return tracker + ref + template."""
    py_dir = os.path.join(_HERE, 'android', 'PoolWaterTester',
                          'app', 'src', 'main', 'python')
    import json
    with open(os.path.join(py_dir, 'reference.json'), 'r', encoding='utf-8') as f:
        ref = json.load(f)
    tpl = cv2.imread(os.path.join(py_dir, 'template02.jpg'))
    tracker = IndicatorTracker(cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY))
    return ref, tpl, tracker


def _draw_cells(warped: np.ndarray, ref: dict) -> np.ndarray:
    """Overlay every cell ROI on the warped image. Color cells green,
    measure cells blue, parameter name + cell_idx labelled."""
    vis = warped.copy()
    for c in ref['cells']:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        is_color = c.get('is_color_cell')
        color = (0, 220, 0) if is_color else (255, 80, 80)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
        label = f"{c.get('parameter') or '?'}#{c['cell_idx']}"
        cv2.putText(vis, label, (x + 2, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return vis


def _swatch_strip(warped_bgr: np.ndarray, tpl_bgr: np.ndarray,
                  ref: dict) -> np.ndarray:
    """For every color cell, build a 2-row strip:
         top = runtime swatch (sampled from warped, median color)
         bottom = reference swatch (sampled from template, median color)
       Width of each tile = 60 px. Useful to spot color-shift mismatches by eye.
    """
    color_cells = [c for c in ref['cells']
                   if c.get('is_color_cell') and c.get('value') is not None]
    if not color_cells:
        return np.zeros((40, 100, 3), dtype=np.uint8)
    # group by parameter for legibility
    by_param: dict = {}
    for c in color_cells:
        by_param.setdefault(c.get('parameter') or '?', []).append(c)

    tile_w, tile_h, gap = 60, 40, 4
    rows = []
    for param, cells in by_param.items():
        cells = sorted(cells, key=lambda c: (c.get('value') or 0))
        n = len(cells)
        row = np.zeros((tile_h * 2 + 22, n * (tile_w + gap) + gap, 3),
                       dtype=np.uint8)
        cv2.putText(row, param, (gap, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        for i, c in enumerate(cells):
            x, y, w, h = c['x'], c['y'], c['w'], c['h']
            warp_roi = warped_bgr[y:y + h, x:x + w]
            tpl_roi = tpl_bgr[y:y + h, x:x + w]
            if warp_roi.size == 0 or tpl_roi.size == 0:
                continue
            warp_med = np.median(warp_roi.reshape(-1, 3), axis=0).astype(np.uint8)
            tpl_med = np.median(tpl_roi.reshape(-1, 3), axis=0).astype(np.uint8)
            x0 = gap + i * (tile_w + gap)
            row[20:20 + tile_h, x0:x0 + tile_w] = warp_med
            row[20 + tile_h:20 + 2 * tile_h, x0:x0 + tile_w] = tpl_med
            cv2.putText(row, f"{c.get('value')}", (x0 + 4, 20 + tile_h + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        rows.append(row)
    # stack rows vertically
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
    return np.vstack(padded)


def _per_cell_lab_table(warped_bgr: np.ndarray, ref: dict) -> str:
    """Text table of per-cell L/A/B medians, runtime vs reference."""
    lab_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    lines = [f"{'param':6s} {'cell':4s} {'val':>6s}  "
             f"{'L_run':>6s} {'A_run':>6s} {'B_run':>6s}  "
             f"{'L_ref':>6s} {'A_ref':>6s} {'B_ref':>6s}  "
             f"{'dL':>5s} {'dA':>5s} {'dB':>5s}"]
    for c in ref['cells']:
        if not c.get('is_color_cell'):
            continue
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        roi = lab_warped[y:y + h, x:x + w]
        if roi.size == 0:
            continue
        run = [float(np.median(roi[:, :, k])) for k in range(3)]
        rl = c.get('lab_median') or [None, None, None]
        rl_str = [f'{v:6.1f}' if v is not None else '   --' for v in rl]
        d = [run[k] - (rl[k] or run[k]) for k in range(3)]
        lines.append(
            f"{c.get('parameter') or '?':6s} "
            f"{c['cell_idx']:4d} {c.get('value') if c.get('value') is not None else '--':>6} "
            f" {run[0]:6.1f} {run[1]:6.1f} {run[2]:6.1f}  "
            f" {rl_str[0]} {rl_str[1]} {rl_str[2]}  "
            f"{d[0]:5.1f} {d[1]:5.1f} {d[2]:5.1f}"
        )
    return '\n'.join(lines)


def _measure_with_diag(warped_bgr: np.ndarray, ref: dict):
    """Re-run _measure_warped's logic with extra diagnostic output.

    Returns (results_dict, per_param_diag_text, summary_str).
    """
    lab_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    color_cells = [c for c in ref['cells']
                   if c.get('is_color_cell') and c.get('value') is not None]
    measure_cells = [c for c in ref['cells'] if not c.get('is_color_cell')]
    param_meta = {p['name']: p for p in ref['parameters']}
    name_to_ch = {'L': 0, 'A': 1, 'B': 2}
    out: dict = {}
    diag_lines = []
    summary_parts = []

    for param, meta in param_meta.items():
        diag_lines.append(f"\n=== {param} ===")
        p_colors = [c for c in color_cells if c.get('parameter') == param]
        p_measure = [c for c in measure_cells if c.get('parameter') == param]
        diag_lines.append(f"  color_cells={len(p_colors)} measure_cells={len(p_measure)}")
        if not p_colors or not p_measure:
            diag_lines.append("  SKIP: missing color or measure cells")
            summary_parts.append(f"{param}=cells?")
            continue

        labs, vals = [], []
        for cell in p_colors:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            labs.append([float(np.median(roi[:, :, k])) for k in range(3)])
            vals.append(cell['value'])

        if len(labs) < 3:
            diag_lines.append(f"  SKIP: <3 valid swatches ({len(labs)})")
            summary_parts.append(f"{param}=<3sw")
            continue

        fixed_ch = meta.get('best_channel')
        ref_coeffs = meta.get('poly_coeffs')
        diag_lines.append(f"  ref: best_channel={fixed_ch} ref_r={meta.get('best_r')}")

        if fixed_ch in name_to_ch and ref_coeffs is not None:
            ch_idx = name_to_ch[fixed_ch]
            ref_ch, tgt_ch, y_arr = [], [], []
            for cell, tgt_lab in zip(p_colors, labs):
                rl = cell.get('lab_median')
                if rl is None or rl[ch_idx] is None:
                    continue
                ref_ch.append(rl[ch_idx])
                tgt_ch.append(tgt_lab[ch_idx])
                y_arr.append(cell['value'])
            ref_ch = np.array(ref_ch, dtype=np.float64)
            tgt_ch = np.array(tgt_ch, dtype=np.float64)
            y_arr = np.array(y_arr, dtype=np.float64)

            if len(ref_ch) < 3 or tgt_ch.std() < 1e-6:
                diag_lines.append(
                    f"  SKIP: ref_ch={len(ref_ch)} tgt_std={tgt_ch.std():.2f}")
                summary_parts.append(f"{param}=stdev")
                continue

            r = float(np.corrcoef(tgt_ch, y_arr)[0, 1])
            ref_r = meta.get('best_r') or 0.0
            diag_lines.append(f"  runtime r (tgt_ch vs y) = {r:.3f}  (sign "
                              f"{'OK' if np.sign(r) == np.sign(ref_r) else 'FLIP'})")
            if ref_r != 0 and np.sign(r) != np.sign(ref_r):
                diag_lines.append("  REJECTED: sign mismatch")
                summary_parts.append(f"{param}=sign")
                continue

            t2r = np.polyfit(tgt_ch, ref_ch, 1)
            coeffs = np.array(ref_coeffs, dtype=np.float64)
            pred_ref = np.polyval(t2r, tgt_ch)
            rmse = float(np.sqrt(np.mean(
                (np.polyval(coeffs, pred_ref) - y_arr) ** 2)))
            diag_lines.append(f"  t2r linear: slope={t2r[0]:.3f} intercept={t2r[1]:.3f}")
            diag_lines.append(f"  swatch rmse vs printed: {rmse:.2f}")
            ch_name = fixed_ch
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
            x_fit = np.array([lbl[ch_idx] for lbl in labs], dtype=np.float64)
            coeffs = np.polyfit(x_fit, y_f, min(2, len(x_fit) - 1))
            rmse = float(np.sqrt(np.mean(
                (np.polyval(coeffs, x_fit) - y_f) ** 2)))
            r = best_r
            t2r = None
            diag_lines.append(f"  fallback: chose channel {ch_name} r={best_r:.3f} rmse={rmse:.2f}")

        # measure cells
        probe_vals = []
        for cell in p_measure:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            probe_vals.append(float(np.median(roi[:, :, ch_idx])))
        if not probe_vals:
            diag_lines.append("  SKIP: no probe samples")
            summary_parts.append(f"{param}=noprobe")
            continue
        probe_ch = float(np.mean(probe_vals))
        diag_lines.append(f"  probe channel mean = {probe_ch:.2f}")
        if t2r is not None:
            probe_ch_corrected = float(np.polyval(t2r, probe_ch))
            diag_lines.append(f"  probe -> reference frame = {probe_ch_corrected:.2f}")
        else:
            probe_ch_corrected = probe_ch
        raw = float(np.polyval(np.array(meta.get('poly_coeffs') or coeffs),
                               probe_ch_corrected)) if t2r is not None else \
            float(np.polyval(coeffs, probe_ch_corrected))
        ref_vals_sorted = sorted(vals)
        clipped = float(np.clip(raw, ref_vals_sorted[0], ref_vals_sorted[-1]))
        clip_flag = '' if abs(clipped - raw) < 1e-3 else ' (clipped)'
        diag_lines.append(f"  raw={raw:.2f}  clipped={clipped:.2f}{clip_flag}  "
                          f"range=[{ref_vals_sorted[0]}, {ref_vals_sorted[-1]}]")
        out[param] = {'value': round(clipped, 2), 'channel': ch_name,
                      'r': round(r, 3), 'rmse': round(rmse, 3)}
        summary_parts.append(f"{param}={clipped:.1f}")

    return out, '\n'.join(diag_lines), ' | '.join(summary_parts)


def process(path: str, ref: dict, tpl: np.ndarray,
            tracker: IndicatorTracker) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(path)
    if bgr is None:
        return f'{base}: read failed'
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    reference_builder.set_debug_dir(out_dir)
    quad = reference_builder._detect_reference_quad(gray, bgr)
    reference_builder.set_debug_dir('')
    if quad is None:
        return f'{base}: NO QUAD'

    warped = tracker.warp(bgr, quad)
    cv2.imwrite(os.path.join(out_dir, '30_warped.jpg'), warped)
    cv2.imwrite(os.path.join(out_dir, '30_warped_with_cells.jpg'),
                _draw_cells(warped, ref))
    cv2.imwrite(os.path.join(out_dir, '31_cell_swatches.jpg'),
                _swatch_strip(warped, tpl, ref))
    with open(os.path.join(out_dir, '32_lab_per_cell.txt'), 'w',
              encoding='utf-8') as f:
        f.write(_per_cell_lab_table(warped, ref))

    results, diag, summary = _measure_with_diag(warped, ref)
    with open(os.path.join(out_dir, '33_per_param.txt'), 'w',
              encoding='utf-8') as f:
        f.write(diag)

    n = len(results)
    flag = 'OK' if n == 3 else (f'{n}/3' if n > 0 else 'NONE')
    return f'{base}: [{flag}] {summary}'


def main():
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    ref, tpl, tracker = _load_reference()
    files = sorted(glob.glob(os.path.join(_HERE, 'test_input', 'frame_*.jpg')))
    if not files:
        print('No test_input/frame_*.jpg files. Pull captures first.')
        return

    for path in files:
        print(process(path, ref, tpl, tracker))


if __name__ == '__main__':
    main()
