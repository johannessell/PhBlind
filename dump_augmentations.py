"""
dump_augmentations.py
=====================
Save the RAW augmented images the live tracker sees, so we can eyeball
whether the perturbations applied by benchmark_live_tracker.py are
realistic for actual hand-held use.

Pipeline mirrors the analyzer:
  1. Load the source JPEG full-res.
  2. Downscale to _LIVE_TRACK_W = 480 px wide (same as live mode).
  3. Apply each augmentation in benchmark_live_tracker.all_variants().
  4. Save the gray result (no overlays) into aug_samples/.

Run:  python dump_augmentations.py
"""

from __future__ import annotations

import os
import shutil
import sys

import cv2

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402
import benchmark_live_tracker as LB  # noqa: E402

OUT_DIR = os.path.join(REPO, 'aug_samples')
SOURCE = os.path.join(REPO, 'test_input', 'frame_1778005012521.jpg')


def main():
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    bgr = cv2.imread(SOURCE)
    if bgr is None:
        raise SystemExit(f'cannot read {SOURCE}')
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > M._LIVE_TRACK_W * 1.25:
        s = M._LIVE_TRACK_W / float(w)
        small = cv2.resize(gray, (int(round(w * s)), int(round(h * s))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray

    cv2.imwrite(os.path.join(OUT_DIR, '00_baseline_480w.jpg'), small)

    for cat, aug_name, spec in LB.all_variants():
        aug, _tf = LB.apply_aug(small, spec)
        cv2.imwrite(os.path.join(OUT_DIR, f'{cat}__{aug_name}.jpg'), aug)

    print(f'wrote {len(LB.all_variants()) + 1} files to {OUT_DIR}')


if __name__ == '__main__':
    main()
