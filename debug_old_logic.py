"""Run the OLD reference_core.find_rects + build_grid against the 4 close-ups."""
import glob
import os
import shutil
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, '_old_logic'))
import reference_core as RC  # noqa: E402

OUT = os.path.join(REPO, 'old_logic_debug')
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT, exist_ok=True)


def process(path):
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT, base)
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(path)
    gray_enh = RC.preprocess(img)
    cv2.imwrite(os.path.join(out_dir, '00_gray_enhanced.jpg'), gray_enh)
    rects = RC.find_rects(gray_enh)

    vis = img.copy()
    for x, y, w, h in rects:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 2)
    cv2.imwrite(os.path.join(out_dir, '01_rects.jpg'), vis)

    n_rects = len(rects)
    grid_info = ''
    if rects:
        try:
            layout = RC.build_grid(rects)
            grid_info = f'grid={layout.n_rows}x{layout.n_cols} cells={len(layout.cells)}'
            grid_vis = RC.draw_grid(img, layout)
            cv2.imwrite(os.path.join(out_dir, '02_grid.jpg'), grid_vis)
        except Exception as e:
            grid_info = f'build_grid FAILED: {e}'

    return f'{base}: {n_rects} rects  {grid_info}'


for p in (sorted(glob.glob(os.path.join(REPO, 'reference_input',
                                        'training', 'frame_17796*.jpg')))):
    print(process(p))
