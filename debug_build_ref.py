"""Run build_reference end-to-end on every close-up reference photo."""
import glob
import os
import shutil
import sys

import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import reference_builder as RB  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'build_ref_debug')
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
os.makedirs(OUT_ROOT, exist_ok=True)

# the 4 fresh close-ups have timestamps starting with 17796
sources = sorted(
    p for p in glob.glob(os.path.join(REPO, 'reference_input', 'training',
                                      'frame_*.jpg'))
    if os.path.basename(p) >= 'frame_17796'
)

for src in sources:
    base = os.path.splitext(os.path.basename(src))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)
    RB.set_debug_dir(out_dir)

    bgr = cv2.imread(src)
    try:
        ref = RB.build_reference(bgr)
    except Exception as e:
        print(f'{base}: FAILED -> {e}')
        continue

    rows = max(c['row_idx'] for c in ref['cells']) + 1
    cols = max(c['col_idx'] for c in ref['cells']) + 1
    n_color = sum(1 for c in ref['cells'] if c['is_color_cell'])
    print(f"{base}: {len(ref['cells'])} cells ({rows}x{cols}), "
          f"{n_color} color cells, params={len(ref['parameters'])}, "
          f"col_types={ref['col_types']}")

    warped = cv2.imread(os.path.join(out_dir, '03_warped.jpg'))
    if warped is not None:
        for c in ref['cells']:
            x, y, ww, hh = c['x'], c['y'], c['w'], c['h']
            color = (0, 220, 0) if c['is_color_cell'] else (220, 120, 0)
            cv2.rectangle(warped, (x, y), (x + ww, y + hh), color, 2)
            cv2.putText(warped, f"{c['row_idx']},{c['col_idx']}",
                        (x + 2, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imwrite(os.path.join(out_dir, '99_built_cells.jpg'), warped)
