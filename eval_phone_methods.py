"""
eval_phone_methods.py
=====================

Offline method-spread evaluation for frames pulled off the phone.

The app saves every measurement attempt to
``/sdcard/Android/data/com.example.poolwatertester/files/measurements/<ts>/``
containing ``input.jpg`` (the rotated frame that was sent to
``measure_rgba``), ``overlay.jpg`` (the runtime grid render) and
``result.txt`` (the chosen-method result). Pull a corpus with::

    adb pull \\
      /sdcard/Android/data/com.example.poolwatertester/files/measurements \\
      ./phone_measurements

This script re-runs the full Python pipeline on every ``input.jpg`` and
emits **all** projection methods' values per parameter — so you can see
which projection actually disagrees with your eye when the chosen one is
off by ~0.2.

Output:
  - ``phone_method_eval.csv`` — one row per ``(frame, parameter)`` with
    columns: ts, param, gt (from ground_truth.csv if present), L, A, B,
    hue_lch, pc1_lab, pc1_ab, chosen_method, chosen_value.
  - A stdout aggregate: per-method std across frames and (if gt present)
    per-method mean signed error against gt.

Optional ground truth file ``ground_truth.csv`` (or the path passed via
``--gt``) with header ``ts,param,value`` — fill in your human estimate
per frame×parameter. Missing rows are OK.

No app rebuild needed.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402
import reference_builder as RB  # noqa: E402

# Method column order in the CSV / aggregate. Must match `_build_projections`
# keys (`L`, `A`, `B`, `hue_lch`, `pc1_lab`, `pc1_ab`).
METHODS = ('L', 'A', 'B', 'hue_lch', 'pc1_lab', 'pc1_ab')


def load_ground_truth(path: str) -> dict[tuple[int, str], float]:
    gt: dict[tuple[int, str], float] = {}
    if not path or not os.path.isfile(path):
        return gt
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = int(row['ts'])
                param = row['param'].strip()
                v = float(row['value'])
            except (KeyError, ValueError):
                continue
            gt[(ts, param)] = v
    return gt


def per_method_values(warped_bgr: np.ndarray, runtime_grid: dict) -> dict:
    """For every parameter in the reference, evaluate every candidate
    projection on its color cells, fit the polynomial against printed
    values, project the measure probe(s), and return the predicted value
    per method. Returns {param: {method: value or NaN}}.
    """
    lab_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    color_cells = [c for c in M._REF['cells']
                   if c['is_color_cell'] and c['value'] is not None]
    measure_cells = [c for c in M._REF['cells'] if not c['is_color_cell']]
    param_meta = {p['name']: p for p in M._REF['parameters']}

    def slot_for(c):
        return runtime_grid.get((c['row_idx'], c['col_idx']))

    out: dict[str, dict[str, float]] = {}
    for param in param_meta:
        p_colors = [c for c in color_cells if c['parameter'] == param]
        p_measure = [c for c in measure_cells if c['parameter'] == param]
        if not p_colors or not p_measure:
            continue

        labs, vals = [], []
        for cell in p_colors:
            s = slot_for(cell)
            if s is None:
                continue
            lab = M._sample_lab_at(lab_warped, s)
            if lab is None:
                continue
            labs.append(lab)
            vals.append(cell['value'])
        if len(labs) < 3:
            continue

        labs_arr = np.array(labs, dtype=np.float64)
        vals_arr = np.array(vals, dtype=np.float64)
        cand = M._build_projections(labs_arr)

        # Pre-sample the probe LABs once — shared across methods.
        probe_labs = []
        for cell in p_measure:
            s = slot_for(cell)
            if s is None:
                continue
            lab = M._sample_lab_at(lab_warped, s)
            if lab is not None:
                probe_labs.append(lab)

        method_values: dict[str, float] = {m: float('nan') for m in METHODS}
        for name, (proj, desc) in cand.items():
            if name not in method_values:
                continue
            mask = ~np.isnan(proj)
            if int(mask.sum()) < 3:
                continue
            x = proj[mask]; y = vals_arr[mask]
            if x.std() < 1e-6:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            sign = 1.0 if r >= 0 else -1.0
            xs = x * sign
            coeffs = np.polyfit(xs, y, min(2, len(xs) - 1))
            pv = []
            for lab in probe_labs:
                v = M._project_probe(lab, desc, sign)
                if v is not None and not np.isnan(v):
                    pv.append(v)
            if not pv:
                continue
            val = float(np.polyval(coeffs, float(np.mean(pv))))
            val = float(np.clip(val, min(vals), max(vals)))
            method_values[name] = val

        out[param] = method_values
    return out


def chosen_for_frame(bgr: np.ndarray) -> dict[str, tuple[str, float]]:
    """Run the app's `measure_rgba` end-to-end so we can record the
    method the picker actually chose plus the resulting value. Returns
    {param: (method_name, value)}; absent params are not included.
    """
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    height, width = rgba.shape[:2]
    res = M.measure_rgba(rgba.tobytes(), width, height)
    out: dict[str, tuple[str, float]] = {}
    for name, info in (res.get('results') or {}).items():
        out[name] = (str(info.get('channel', '')),
                     float(info.get('value', float('nan'))))
    return out


def process_frame(path: str):
    ts_str = os.path.basename(os.path.dirname(path))
    try:
        ts = int(ts_str)
    except ValueError:
        ts = 0
    bgr = cv2.imread(path)
    if bgr is None:
        return ts, {}, {}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = RB._detect_reference_quad(gray, bgr)
    if quad is None:
        return ts, {}, {}
    warped = M._TRACKER.warp(bgr, quad)
    grid, _, _ = M._detect_warped_grid(warped)
    if grid is None:
        return ts, {}, {}
    methods = per_method_values(warped, grid)
    chosen = chosen_for_frame(bgr)
    return ts, methods, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', default='phone_measurements',
                        help='Folder with <ts>/input.jpg subfolders '
                             '(default: phone_measurements)')
    parser.add_argument('--gt', default='ground_truth.csv',
                        help='Optional ground-truth CSV path')
    parser.add_argument('--out', default='phone_method_eval.csv',
                        help='Per-frame×param output CSV path')
    args = parser.parse_args()

    M.init('')  # load reference from android/.../python/reference.json copy

    inputs = sorted(glob.glob(os.path.join(args.frames, '*', 'input.jpg')))
    if not inputs:
        print(f'No input.jpg under {args.frames}/<ts>/.', file=sys.stderr)
        sys.exit(2)

    gt = load_ground_truth(args.gt)
    if gt:
        print(f'Loaded {len(gt)} ground-truth row(s) from {args.gt}.')
    else:
        print(f'(no ground truth at {args.gt}; per-method error skipped)')

    rows = []
    by_method: dict[str, list[float]] = defaultdict(list)
    err_by_method: dict[str, list[float]] = defaultdict(list)

    for fn in inputs:
        ts, methods, chosen = process_frame(fn)
        if not methods:
            print(f'  {ts}  no-grid')
            continue
        for param, mv in methods.items():
            ch_name, ch_val = chosen.get(param, ('', float('nan')))
            gt_v = gt.get((ts, param), float('nan'))
            row = {
                'ts': ts, 'param': param,
                'gt': '' if np.isnan(gt_v) else f'{gt_v:.3f}',
                'chosen_method': ch_name,
                'chosen_value': '' if np.isnan(ch_val) else f'{ch_val:.3f}',
            }
            for m in METHODS:
                v = mv.get(m, float('nan'))
                row[m] = '' if np.isnan(v) else f'{v:.3f}'
                if not np.isnan(v):
                    by_method[(param, m)].append(v)
                    if not np.isnan(gt_v):
                        err_by_method[(param, m)].append(v - gt_v)
            rows.append(row)
        cells = '  '.join(f'{m}={methods[list(methods)[0]].get(m, float("nan")):>6.2f}'
                          for m in METHODS)
        print(f'  {ts}  params={list(methods)}  ({cells})')

    if not rows:
        print('No frames produced a runtime grid.', file=sys.stderr)
        sys.exit(3)

    fields = ['ts', 'param', 'gt', *METHODS, 'chosen_method', 'chosen_value']
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nWrote {len(rows)} row(s) to {args.out}.')

    # Aggregate per (param, method).
    params = sorted({k[0] for k in by_method})
    print()
    print(f'{"param":>6}  {"method":>9}  {"n":>3}  {"mean":>7}  '
          f'{"std":>6}  {"min":>7}  {"max":>7}  {"err_mean":>9}  {"err_std":>8}')
    for p in params:
        for m in METHODS:
            vals = by_method.get((p, m), [])
            if not vals:
                continue
            a = np.array(vals)
            err = err_by_method.get((p, m), [])
            err_mean = f'{np.mean(err):+9.3f}' if err else '       —'
            err_std  = f'{np.std(err):8.3f}'   if err else '       —'
            print(f'{p:>6}  {m:>9}  {len(a):3d}  {a.mean():7.3f}  '
                  f'{a.std():6.3f}  {a.min():7.3f}  {a.max():7.3f}  '
                  f'{err_mean}  {err_std}')


if __name__ == '__main__':
    main()
