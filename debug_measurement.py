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


def _draw_cells(warped: np.ndarray, ref: dict,
                runtime_grid: dict = None) -> np.ndarray:
    """Overlay reference.json positions (red) + runtime-detected positions
    (yellow). When the new pipeline works, yellow boxes hug the actual
    swatches even though red boxes don't.
    """
    vis = warped.copy()
    # Reference positions in red — stored x/y/w/h as-is
    for c in ref['cells']:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        cv2.rectangle(vis, (x, y), (x + w, y + h), (60, 60, 200), 1)

    # Runtime-detected positions in yellow (with row/col label).
    # Estimated (filled-in from row/col medians) drawn in orange.
    if runtime_grid:
        for (ri, ci), slot in runtime_grid.items():
            cx, cy, w, h = slot[0], slot[1], slot[2], slot[3]
            x = int(round(cx - w / 2))
            y = int(round(cy - h / 2))
            estimated = len(slot) > 6 and slot[6] == 'est'
            color = (0, 140, 255) if estimated else (0, 220, 220)
            cv2.rectangle(vis, (x, y), (x + int(w), y + int(h)),
                          color, 2)
            cv2.putText(vis, f'{ri},{ci}', (x + 2, y + 12),
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


def _measure_with_diag(warped_bgr: np.ndarray, ref: dict,
                       runtime_grid: dict, grid_status: str):
    """Wrap the new measurement pipeline and emit per-frame diagnostics.

    Returns (results_dict, per_param_diag_text, summary_str).
    """
    import measurement as M
    diag_lines = [f"=== grid: {grid_status} ===\n"]

    results = M._measure_warped(warped_bgr, runtime_grid=runtime_grid)
    param_meta = {p['name']: p for p in ref['parameters']}

    summary_parts = []
    for param in param_meta:
        diag_lines.append(f"=== {param} ===")
        if param not in results:
            diag_lines.append('  no result (cells missing or all-projections rejected)')
            summary_parts.append(f'{param}=miss')
            continue
        r = results[param]
        diag_lines.append(
            f"  picked projection: {r['channel']}  r={r['r']}  rmse={r['rmse']}")
        diag_lines.append(f"  value: {r['value']}")
        summary_parts.append(f"{param}={r['value']}({r['channel']})")

    return results, '\n'.join(diag_lines), ' | '.join(summary_parts)


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

    # Re-detect grid in the warped image (same logic as measurement.measure_rgba)
    import measurement as M
    runtime_grid, (gr_rows, gr_cols), _detected_cells = M._detect_warped_grid(warped)
    grid_status = (f'{gr_rows}x{gr_cols}'
                   f' ({"OK" if runtime_grid is not None else "mismatch"}'
                   f' / expected {M._EXPECTED_ROWS}x{M._EXPECTED_COLS})')

    cv2.imwrite(os.path.join(out_dir, '30_warped_with_cells.jpg'),
                _draw_cells(warped, ref, runtime_grid=runtime_grid))
    cv2.imwrite(os.path.join(out_dir, '31_cell_swatches.jpg'),
                _swatch_strip(warped, tpl, ref))

    results, diag, summary = _measure_with_diag(
        warped, ref, runtime_grid=runtime_grid, grid_status=grid_status)
    with open(os.path.join(out_dir, '33_per_param.txt'), 'w',
              encoding='utf-8') as f:
        f.write(diag)

    n = len(results)
    flag = 'OK' if n == 3 else (f'{n}/3' if n > 0 else 'NONE')
    return f'{base}: [{flag}] grid={gr_rows}x{gr_cols} | {summary}'


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
