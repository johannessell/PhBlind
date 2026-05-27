"""
benchmark_live_tracker.py
=========================
Stress-test ``measurement._find_quad_lowres`` (the only detector used in live
preview) on real test_input/frame_*.jpg captures, perturbed with:

  - Translation       (dx, dy in {-12, -6, 0, 6, 12} px) -- 25 variants
  - Motion blur       (7-px line kernel at 0/45/90 deg) -- 3 variants
  - In-plane rotation (+-5, +-10 deg)                   -- 4 variants
  - Perspective tilt  (shrink one side 10% / 20%)       -- 8 variants

For each frame we:
  1. Compute baseline quad on the unperturbed frame.
  2. For every augmentation, transform the gray image AND the baseline quad
     by the same map, run the detector on the augmented image, and compare
     against the transformed baseline quad via mask-IoU.

Per-frame line + aggregate report. Failures and lowest-IoU successes per
category written to bench_out/<frame>/<aug>.jpg with the detected quad drawn.

No production code changes.

Run:  python benchmark_live_tracker.py
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'android', 'PoolWaterTester',
                                'app', 'src', 'main', 'python'))

import measurement as M  # noqa: E402

OUT_ROOT = os.path.join(REPO, 'bench_out')


# ---------------------------------------------------------------- augmentations

def _translation_variants():
    out = []
    for dy in (-12, -6, 0, 6, 12):
        for dx in (-12, -6, 0, 6, 12):
            out.append(('transl', f'tx{dx:+d}_ty{dy:+d}',
                        ('translate', dx, dy)))
    return out


def _rotation_variants():
    out = []
    for deg in (-10, -5, 5, 10):
        out.append(('rot', f'rot{deg:+d}', ('rotate', deg)))
    return out


def _tilt_variants():
    """Shrink one side by 10% -- simulates a mild card tilt toward/away
    from the camera on that side."""
    out = []
    for side in ('top', 'bottom', 'left', 'right'):
        out.append(('tilt', f'tilt_{side}_10',
                    ('tilt', side, 0.10)))
    return out


def all_variants():
    return (_translation_variants() + _rotation_variants()
            + _tilt_variants())


def _line_kernel(size: int, deg: float) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    c = (size - 1) / 2.0
    rad = np.deg2rad(deg)
    dx, dy = np.cos(rad), np.sin(rad)
    for t in np.linspace(-c, c, size * 4):
        x = int(round(c + dx * t))
        y = int(round(c + dy * t))
        if 0 <= x < size and 0 <= y < size:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def apply_aug(gray: np.ndarray, spec):
    """Return (augmented_gray, transform_fn, affine_or_homog_matrix_or_None).

    transform_fn(quad_4x2_float32) -> transformed 4x2 float32 quad.
    """
    h, w = gray.shape[:2]
    kind = spec[0]
    if kind == 'translate':
        dx, dy = spec[1], spec[2]
        M_a = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        out = cv2.warpAffine(gray, M_a, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        def tf(q):
            return q + np.array([[dx, dy]], dtype=np.float32)
        return out, tf
    if kind == 'blur':
        size, deg = spec[1], spec[2]
        ker = _line_kernel(size, deg)
        out = cv2.filter2D(gray, -1, ker, borderType=cv2.BORDER_REPLICATE)

        def tf(q):
            return q.copy()
        return out, tf
    if kind == 'rotate':
        deg = spec[1]
        c = (w / 2.0, h / 2.0)
        M_a = cv2.getRotationMatrix2D(c, deg, 1.0)
        out = cv2.warpAffine(gray, M_a, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

        def tf(q):
            return cv2.transform(q.reshape(1, -1, 2),
                                 M_a).reshape(-1, 2).astype(np.float32)
        return out, tf
    if kind == 'tilt':
        side, frac = spec[1], spec[2]
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float32)
        dst = src.copy()
        # Shrink the chosen side toward its midpoint by `frac`.
        if side == 'top':
            dst[0, 0] += w * frac / 2.0
            dst[1, 0] -= w * frac / 2.0
        elif side == 'bottom':
            dst[3, 0] += w * frac / 2.0
            dst[2, 0] -= w * frac / 2.0
        elif side == 'left':
            dst[0, 1] += h * frac / 2.0
            dst[3, 1] -= h * frac / 2.0
        elif side == 'right':
            dst[1, 1] += h * frac / 2.0
            dst[2, 1] -= h * frac / 2.0
        H = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(gray, H, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        def tf(q):
            return cv2.perspectiveTransform(
                q.reshape(1, -1, 2).astype(np.float32),
                H).reshape(-1, 2).astype(np.float32)
        return out, tf
    raise ValueError(f'unknown aug {kind}')


# -------------------------------------------------------------- geometry helpers

def quad_iou(a: np.ndarray, b: np.ndarray, canvas_shape) -> float:
    """Mask-based IoU on a half-resolution canvas to keep cost low while
    handling arbitrary quads."""
    h, w = canvas_shape[:2]
    scale = 0.5
    sw, sh = int(w * scale), int(h * scale)
    ma = np.zeros((sh, sw), dtype=np.uint8)
    mb = np.zeros((sh, sw), dtype=np.uint8)
    cv2.fillConvexPoly(ma, (a * scale).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, (b * scale).astype(np.int32), 1)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union > 0 else 0.0


# -------------------------------------------------------------- run / reporting

def _draw_quad(img_gray: np.ndarray, quad, color=(0, 220, 0),
               quad_ref=None) -> np.ndarray:
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if quad_ref is not None:
        cv2.polylines(vis, [quad_ref.astype(np.int32)], True,
                      (0, 0, 220), 2)
    if quad is not None:
        cv2.polylines(vis, [quad.astype(np.int32)], True, color, 2)
    return vis


def process_frame(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    base_quad = M._find_quad_lowres(gray)
    base_ms = (time.perf_counter() - t0) * 1000.0
    if base_quad is None:
        return {'name': os.path.basename(path), 'baseline_ms': base_ms,
                'no_baseline': True}

    per_cat = defaultdict(list)  # cat -> list of dicts {aug, detected, iou, ms}
    canvas = gray.shape
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(OUT_ROOT, base)
    os.makedirs(out_dir, exist_ok=True)

    # Save baseline
    cv2.imwrite(os.path.join(out_dir, '00_baseline.jpg'),
                _draw_quad(gray, base_quad))

    for cat, aug_name, spec in all_variants():
        aug_gray, tf = apply_aug(gray, spec)
        expected = tf(base_quad)
        t0 = time.perf_counter()
        q = M._find_quad_lowres(aug_gray)
        ms = (time.perf_counter() - t0) * 1000.0
        if q is None:
            per_cat[cat].append({'aug': aug_name, 'detected': False,
                                 'iou': 0.0, 'ms': ms,
                                 'aug_gray': aug_gray, 'expected': expected,
                                 'quad': None})
        else:
            iou = quad_iou(q, expected, canvas)
            per_cat[cat].append({'aug': aug_name, 'detected': True,
                                 'iou': iou, 'ms': ms,
                                 'aug_gray': aug_gray, 'expected': expected,
                                 'quad': q})

    # Dump overlays: every failure + worst-IoU success per category
    for cat, entries in per_cat.items():
        for e in entries:
            if not e['detected']:
                cv2.imwrite(
                    os.path.join(out_dir, f'FAIL_{cat}_{e["aug"]}.jpg'),
                    _draw_quad(e['aug_gray'], None, quad_ref=e['expected']))
        successes = [e for e in entries if e['detected']]
        if successes:
            worst = min(successes, key=lambda e: e['iou'])
            cv2.imwrite(
                os.path.join(out_dir, f'WORST_{cat}_{worst["aug"]}_iou{worst["iou"]:.2f}.jpg'),
                _draw_quad(worst['aug_gray'], worst['quad'],
                           quad_ref=worst['expected']))

    # Strip heavy fields before returning
    for entries in per_cat.values():
        for e in entries:
            e.pop('aug_gray', None)
            e.pop('expected', None)
            e.pop('quad', None)

    return {'name': os.path.basename(path), 'baseline_ms': base_ms,
            'per_cat': per_cat}


def _cat_summary(entries):
    n = len(entries)
    hits = sum(1 for e in entries if e['detected'])
    ious = [e['iou'] for e in entries if e['detected']]
    mean_iou = float(np.mean(ious)) if ious else 0.0
    worst = None
    if entries:
        worst_e = min(entries, key=lambda e: (e['detected'], e['iou']))
        worst = worst_e['aug']
    return hits, n, mean_iou, worst


def main():
    if os.path.isdir(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    M.init('')  # uses bundled reference + template -> sets _TRACKER

    files = sorted(glob.glob(os.path.join(REPO, 'test_input',
                                          'frame_*.jpg')))
    if not files:
        print('No test_input/frame_*.jpg files.')
        return

    agg = defaultdict(list)  # cat -> list of all entries (across frames)
    all_ms = []
    no_baseline = []

    print('per-frame results (T=transl, R=rot, P=tilt; hits/N, '
          'mean IoU on hits):')
    for path in files:
        r = process_frame(path)
        if r is None:
            continue
        if r.get('no_baseline'):
            no_baseline.append(r['name'])
            print(f'{r["name"]}: BASELINE FAILED')
            continue
        parts = [r['name']]
        for cat in ('transl', 'rot', 'tilt'):
            entries = r['per_cat'].get(cat, [])
            agg[cat].extend(entries)
            hits, n, mean_iou, worst = _cat_summary(entries)
            parts.append(f'{cat[:1].upper()} {hits}/{n} IoU={mean_iou:.2f}')
        # collect ms
        for entries in r['per_cat'].values():
            for e in entries:
                if e['detected']:
                    all_ms.append(e['ms'])
        # worst category label
        worst_label = None
        worst_score = (1, 1.0)  # (detected, iou) -- minimize
        for cat, entries in r['per_cat'].items():
            for e in entries:
                score = (1 if e['detected'] else 0, e['iou'])
                if score < worst_score:
                    worst_score = score
                    worst_label = f'{cat}:{e["aug"]}'
        if worst_label is not None:
            parts.append(f'worst: {worst_label}')
        print('  ' + '   '.join(parts))

    print()
    print('===== aggregate =====')
    for cat in ('transl', 'rot', 'tilt'):
        entries = agg[cat]
        hits = sum(1 for e in entries if e['detected'])
        n = len(entries)
        ious = [e['iou'] for e in entries if e['detected']]
        if ious:
            mean_iou = float(np.mean(ious))
            p10 = float(np.percentile(ious, 10))
        else:
            mean_iou = p10 = 0.0
        rate = (hits / n * 100.0) if n else 0.0
        print(f'  {cat:6s}: hits {hits:4d}/{n:<4d} ({rate:5.1f}%)   '
              f'mean IoU {mean_iou:.3f}   p10 IoU {p10:.3f}')

    if all_ms:
        print(f'\n  detector ms: median {np.median(all_ms):.2f}  '
              f'p90 {np.percentile(all_ms, 90):.2f}  '
              f'max {max(all_ms):.2f}  (n={len(all_ms)})')

    failures = []
    for cat, entries in agg.items():
        for e in entries:
            if not e['detected']:
                failures.append((cat, e['aug']))
    if failures:
        print(f'\n  {len(failures)} (cat, aug) failures '
              '(see bench_out/<frame>/FAIL_*.jpg):')
        # show first 20
        for cat, aug in failures[:20]:
            print(f'    {cat:6s} {aug}')
        if len(failures) > 20:
            print(f'    ... and {len(failures) - 20} more')

    if no_baseline:
        print(f'\n  frames with no baseline quad: {no_baseline}')


if __name__ == '__main__':
    main()
