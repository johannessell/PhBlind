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

import reference_builder
from tracker import IndicatorTracker, QuadStabilityChecker, order_quad_corners

_HERE = os.path.dirname(__file__)
_STABILITY = QuadStabilityChecker(required_frames=5, max_drift=20.0)

# Live-tracker target width. Detection runs on the downscaled frame; the
# quad is scaled back to the original frame for warp at full color resolution.
_LIVE_TRACK_W = 480


_MIN_CELLS_PER_CLUSTER = 6
_CELL_MIN_SHORT_PX = 6      # at 480 wide; below this is noise
_CELL_MAX_SHORT_PX = 50     # above this isn't a single cell
_CELL_MAX_AR = 4.5
_CELL_MIN_RECTANGULARITY = 0.55


def _detect_cells_otsu(gray: np.ndarray):
    """Otsu-threshold the gray, erode 2 px so adjacent cells stay disjoint,
    then take each bright connected component as a cell candidate.

    Returns list of (cx, cy, long_side, short_side, area, angle_deg) —
    same tuple format as reference_builder._detect_cell_rects, so the
    downstream cluster/score logic is unchanged.

    Otsu gives clean, filled cell shapes (much better than Canny edges,
    which only outline cells and need closure to be useful).
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _t, mask = cv2.threshold(blurred, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

    num, labels, stats, _cent = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    cells = []
    min_area = _CELL_MIN_SHORT_PX * _CELL_MIN_SHORT_PX
    max_area = _CELL_MAX_SHORT_PX * _CELL_MAX_SHORT_PX * 1.5
    for k in range(1, num):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        bx = int(stats[k, cv2.CC_STAT_LEFT])
        by = int(stats[k, cv2.CC_STAT_TOP])
        bw = int(stats[k, cv2.CC_STAT_WIDTH])
        bh = int(stats[k, cv2.CC_STAT_HEIGHT])
        # Cheap pre-filter on the stats bbox before paying for minAreaRect.
        if (min(bw, bh) < _CELL_MIN_SHORT_PX
                or max(bw, bh) > _CELL_MAX_SHORT_PX * 1.5):
            continue
        # Crop to component bbox so np.where is bounded.
        sub = labels[by:by + bh, bx:bx + bw] == k
        ys, xs = np.where(sub)
        pts = np.column_stack((xs + bx, ys + by)).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        long_s = max(rw, rh)
        short_s = min(rw, rh)
        if (short_s < _CELL_MIN_SHORT_PX or short_s > _CELL_MAX_SHORT_PX
                or long_s > _CELL_MAX_SHORT_PX * 1.5):
            continue
        if long_s / max(short_s, 1e-3) > _CELL_MAX_AR:
            continue
        if area / max(rw * rh, 1) < _CELL_MIN_RECTANGULARITY:
            continue
        ang = rect[2] + (90.0 if rw < rh else 0.0)
        while ang > 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        cx, cy = rect[0]
        cells.append((int(cx), int(cy), int(long_s), int(short_s),
                      float(area), float(ang)))
    return cells


def _cluster_cells_by_proximity(cells, radius_factor: float = 2.5):
    """Single-link cluster cells by center proximity. Returns list of lists
    of cell indices. Union-find over an n*n distance test; n is small."""
    n = len(cells)
    if n == 0:
        return []
    pts = np.array([(c[0], c[1]) for c in cells], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cells], dtype=np.float32)
    radius = max(float(np.median(sizes)) * radius_factor, 30.0)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        d = np.linalg.norm(pts - pts[i], axis=1)
        for j in np.where(d < radius)[0]:
            if j > i:
                union(i, int(j))

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _find_quad_lowres(gray: np.ndarray):
    """Live-preview detector.

    Otsu-threshold the gray frame, take each bright connected component as
    a cell candidate, cluster the candidates by spatial proximity, and
    return the rotated bbox of the largest near-centre cluster, expanded
    by ~half a cell on each side.

    The card outline itself is unreliable (transparent plastic on table)
    so we anchor on the cells. Otsu cleanly separates the light cell
    interiors from the dark grid borders even when clutter (glasses,
    magazines, monitors) sits behind the card.

    Returns None if no cluster reaches _MIN_CELLS_PER_CLUSTER cells.

    Used by find_quad_y per analyzer frame; ~5 ms on desktop at 480 wide.
    """
    h, w = gray.shape[:2]
    if w > _LIVE_TRACK_W * 1.25:
        scale = _LIVE_TRACK_W / float(w)
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = gray

    cells = _detect_cells_otsu(small)
    if len(cells) < _MIN_CELLS_PER_CLUSTER:
        return None

    clusters = _cluster_cells_by_proximity(cells)

    sh, sw = small.shape
    img_center = np.array([sw * 0.5, sh * 0.5], dtype=np.float32)
    img_diag = float(np.hypot(sw, sh))

    best_rect = None
    best_score = -1e9
    best_cell_short = 0.0
    for indices in clusters:
        if len(indices) < _MIN_CELLS_PER_CLUSTER:
            continue
        cluster_cells = [cells[i] for i in indices]
        pts = np.array([(c[0], c[1]) for c in cluster_cells],
                       dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        if min(rect[1]) < 1:
            continue
        cdist = float(np.linalg.norm(
            np.array(rect[0], dtype=np.float32) - img_center)) / img_diag
        score = len(indices) - 5.0 * cdist
        if score > best_score:
            best_score = score
            best_rect = rect
            best_cell_short = float(
                np.median([min(c[2], c[3]) for c in cluster_cells]))

    if best_rect is None:
        return None

    # Margin chosen to land the overlay on the card's outer border, not
    # just the inner grid. The card extends roughly one cell short-side
    # past the cell grid on each side.
    margin = max(best_cell_short * 1.0, 6.0)
    (cx, cy), (rw, rh), ang = best_rect
    rw += 2.0 * margin
    rh += 2.0 * margin
    box = cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)

    quad = order_quad_corners(box)
    if scale != 1.0:
        quad = (quad / scale).astype(np.float32)
    return quad

_REF: dict = {}
_TEMPLATE = None
_TEMPLATE_GRAY = None
_TRACKER: IndicatorTracker = None  # type: ignore[assignment]
_LOADED_FROM: str = ''
_EXPECTED_ROWS: int = 0
_EXPECTED_COLS: int = 0


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
    global _EXPECTED_ROWS, _EXPECTED_COLS

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

    cells = ref.get('cells', [])
    # Older references don't store col_idx (it's None) — derive from x
    # positions within each row, in left-to-right order.
    if cells and all(c.get('col_idx') is None for c in cells):
        by_row: dict = {}
        for c in cells:
            by_row.setdefault(c['row_idx'], []).append(c)
        for row_cells in by_row.values():
            row_cells.sort(key=lambda c: c['x'])
            for ci, c in enumerate(row_cells):
                c['col_idx'] = ci
    _EXPECTED_ROWS = (max(c['row_idx'] for c in cells) + 1) if cells else 0
    _EXPECTED_COLS = (max(c.get('col_idx') or 0 for c in cells) + 1) if cells else 0

    return {
        'source': source,
        'ref_path': ref_path,
        'tpl_path': tpl_path,
        'width': int(_TEMPLATE.shape[1]),
        'height': int(_TEMPLATE.shape[0]),
        'n_cells': len(_REF.get('cells', [])),
        'parameters': [p.get('name') for p in _REF.get('parameters', [])],
        'expected_rows': _EXPECTED_ROWS,
        'expected_cols': _EXPECTED_COLS,
    }


# Load bundled defaults eagerly so the module is usable even if Kotlin
# forgets to call init().
init('')


def _detect_warped_grid(warped_bgr: np.ndarray):
    """Re-detect the cell grid in the warped image and assign each detected
    cell a (row_idx, col_idx). Returns:
      - dict[(row_idx, col_idx)] -> (cx, cy, w, h)
      - (rows, cols) tuple of detected counts
      - list of all detected cells (for debug overlay)
    Or (None, (rows, cols), all_cells) if the detected shape doesn't match
    (expected_rows, expected_cols).

    Reuses the existing reference_builder cell-detection helpers so we get
    identical filtering as the original calibration. The point is to sample
    at the runtime-detected positions instead of trusting reference.json's
    pixel coords.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    edges = reference_builder._compute_edges(gray)
    cells = reference_builder._detect_cell_rects(edges)
    if len(cells) < 6:
        return None, (0, 0), cells
    sized = reference_builder._filter_cells_by_size(cells)
    if len(sized) < 6:
        return None, (0, 0), sized
    dense = reference_builder._filter_dense_cells(sized, k_neighbors=4)
    if len(dense) < 6:
        return None, (0, 0), dense
    clean, _ = reference_builder._drop_boundary_outliers(dense)
    if len(clean) < 6:
        return None, (0, 0), clean

    # 1-D cluster by image-frame y (rows top→bottom) and x (cols left→right).
    # We KNOW the expected count, so split at the (expected - 1) largest
    # gaps between consecutive sorted positions — gives exactly the right
    # number of clusters whenever the cells span the full grid. Reject only
    # if any resulting cluster is empty (cells didn't span the full grid).
    def _cluster_to_n(positions, expected):
        order = np.argsort(positions)
        if expected <= 1 or len(order) < expected:
            return [list(map(int, order))]
        sorted_vals = positions[order]
        gaps = np.diff(sorted_vals)
        # indices of the (expected - 1) largest gaps, ascending order
        boundaries = sorted(np.argsort(gaps)[-(expected - 1):].tolist())
        clusters = []
        start = 0
        for b in boundaries:
            clusters.append([int(order[i]) for i in range(start, b + 1)])
            start = b + 1
        clusters.append([int(order[i]) for i in range(start, len(order))])
        return clusters

    cy_arr = np.array([c[1] for c in clean], dtype=np.float32)
    cx_arr = np.array([c[0] for c in clean], dtype=np.float32)
    row_clusters = _cluster_to_n(cy_arr, _EXPECTED_ROWS)
    col_clusters = _cluster_to_n(cx_arr, _EXPECTED_COLS)
    rows, cols = len(row_clusters), len(col_clusters)

    # Reject if the targeted split couldn't produce the expected counts
    # (too few cells overall) or any cluster ended up empty.
    if (rows, cols) != (_EXPECTED_ROWS, _EXPECTED_COLS):
        return None, (rows, cols), clean
    if any(len(c) == 0 for c in row_clusters) \
            or any(len(c) == 0 for c in col_clusters):
        return None, (rows, cols), clean

    # Build cell_idx -> (row_idx, col_idx)
    row_of = {}
    for ri, members in enumerate(row_clusters):
        for m in members:
            row_of[m] = ri
    col_of = {}
    for ci, members in enumerate(col_clusters):
        for m in members:
            col_of[m] = ci

    # Per (row, col) slot, keep cell whose centroid is closest to that
    # row's median cy and that column's median cx (handles a rare case
    # where two cells fall in the same slot).
    row_med = [float(np.median(cy_arr[m])) for m in row_clusters]
    col_med = [float(np.median(cx_arr[m])) for m in col_clusters]

    runtime_grid = {}
    for idx, c in enumerate(clean):
        ri = row_of[idx]
        ci = col_of[idx]
        d = (c[0] - col_med[ci]) ** 2 + (c[1] - row_med[ri]) ** 2
        prev = runtime_grid.get((ri, ci))
        if prev is None or d < prev[1]:
            runtime_grid[(ri, ci)] = (c, d)

    runtime_grid = {k: v[0] for k, v in runtime_grid.items()}

    # Grid completion: fill any missing (row, col) slots by intersecting the
    # row's median cy with the column's median cx. Use the column's own
    # median width (labels are wider than color swatches, so a global
    # median picks the wrong size) and the row's own median height,
    # falling back to the global median when a row/col has no detected
    # cells. The slot tuple gets a synthetic flag at index 6 so debug
    # code can color it differently.
    global_w = float(np.median([c[2] for c in clean]))
    global_h = float(np.median([c[3] for c in clean]))
    col_w = {
        ci: (float(np.median([clean[m][2] for m in members]))
             if members else global_w)
        for ci, members in enumerate(col_clusters)
    }
    row_h = {
        ri: (float(np.median([clean[m][3] for m in members]))
             if members else global_h)
        for ri, members in enumerate(row_clusters)
    }
    for ri in range(_EXPECTED_ROWS):
        for ci in range(_EXPECTED_COLS):
            if (ri, ci) in runtime_grid:
                continue
            cx = col_med[ci]
            cy = row_med[ri]
            runtime_grid[(ri, ci)] = (
                int(round(cx)), int(round(cy)),
                int(round(col_w[ci])), int(round(row_h[ri])),
                0.0, 0.0, 'est',
            )

    return runtime_grid, (rows, cols), clean


def _sample_lab_at(lab_warped, slot):
    """slot is (cx, cy, w, h, ...) from runtime grid (or (x,y,w,h) tuple).
    Returns the median (L, A, B) inside that slot, or None if empty."""
    if len(slot) >= 4:
        cx, cy, w, h = slot[0], slot[1], slot[2], slot[3]
    else:
        return None
    x = int(round(cx - w / 2))
    y = int(round(cy - h / 2))
    H, W = lab_warped.shape[:2]
    x = max(0, x); y = max(0, y)
    x2 = min(W, x + int(w)); y2 = min(H, y + int(h))
    if x2 <= x or y2 <= y:
        return None
    roi = lab_warped[y:y2, x:x2]
    if roi.size == 0:
        return None
    return (float(np.median(roi[:, :, 0])),
            float(np.median(roi[:, :, 1])),
            float(np.median(roi[:, :, 2])))


def _build_projections(labs_arr: np.ndarray):
    """Given labs (N, 3) in OpenCV LAB units (L: 0-255, a/b centered at 128),
    return a dict {name: (proj_values_(N,), descriptor_for_probe_replay)}.

    The descriptor lets us project a single probe sample onto the same axis
    later (used in measure cells). Format is name-specific:
      - 'L' / 'A' / 'B': descriptor is the channel index (0/1/2)
      - 'hue_lch':       descriptor is (median_hue_for_unwrap_reference,
                                        chroma_threshold_for_drop)
      - 'pc1_lab':       descriptor is (mean_lab_(3,), axis_lab_(3,))
      - 'pc1_ab':        descriptor is (mean_ab_(2,), axis_ab_(2,))
    """
    L = labs_arr[:, 0]
    A = labs_arr[:, 1] - 128.0
    B = labs_arr[:, 2] - 128.0
    out = {}

    out['L'] = (L.copy(), ('chan', 0))
    out['A'] = (A.copy(), ('chan_centered', 1))
    out['B'] = (B.copy(), ('chan_centered', 2))

    chroma = np.sqrt(A * A + B * B)
    hue = np.degrees(np.arctan2(B, A)) % 360.0
    # Drop low-saturation swatches from hue selection — hue is unreliable
    # there. Keep enough to fit (>= 3) or skip the candidate.
    keep_hue = chroma > 10.0
    if int(keep_hue.sum()) >= 3:
        hue_kept = hue[keep_hue]
        # Unwrap so values cluster around their median (handle 360°/0° wrap)
        med = float(np.median(hue_kept))
        hue_unwrapped = ((hue - med + 180.0) % 360.0) - 180.0  # (-180, 180]
        # Mark dropped slots as nan so the regression skips them
        hue_proj = np.where(keep_hue, hue_unwrapped, np.nan)
        out['hue_lch'] = (hue_proj, ('hue_lch', med))

    # PC1 of full LAB
    if len(labs_arr) >= 3:
        ab3 = labs_arr.copy().astype(np.float64)
        ab3[:, 1] -= 128.0
        ab3[:, 2] -= 128.0
        mean3 = ab3.mean(axis=0)
        centered3 = ab3 - mean3
        try:
            _, _, Vt = np.linalg.svd(centered3, full_matrices=False)
            axis3 = Vt[0]
            proj3 = centered3 @ axis3
            if float(proj3.std()) > 1e-6:
                out['pc1_lab'] = (proj3, ('pc1_lab', mean3, axis3))
        except np.linalg.LinAlgError:
            pass

    # PC1 of (a, b) only
    if len(labs_arr) >= 3:
        ab2 = np.stack([A, B], axis=1)
        mean2 = ab2.mean(axis=0)
        centered2 = ab2 - mean2
        try:
            _, _, Vt = np.linalg.svd(centered2, full_matrices=False)
            axis2 = Vt[0]
            proj2 = centered2 @ axis2
            if float(proj2.std()) > 1e-6:
                out['pc1_ab'] = (proj2, ('pc1_ab', mean2, axis2))
        except np.linalg.LinAlgError:
            pass

    return out


def _project_probe(lab_triple, descriptor, sign):
    """Given a single probe LAB sample and the descriptor returned by
    _build_projections, return the scalar projection on the SAME axis.
    `sign` is +1 or -1 to match the orientation chosen at fit time."""
    L, a8, b8 = lab_triple
    A = a8 - 128.0
    B = b8 - 128.0
    kind = descriptor[0]
    if kind == 'chan':
        idx = descriptor[1]
        v = (L, a8, b8)[idx]
    elif kind == 'chan_centered':
        idx = descriptor[1]
        v = (L, A, B)[idx]
    elif kind == 'hue_lch':
        med = descriptor[1]
        hue = np.degrees(np.arctan2(B, A)) % 360.0
        v = ((hue - med + 180.0) % 360.0) - 180.0
    elif kind == 'pc1_lab':
        _, mean3, axis3 = descriptor
        x = np.array([L, A, B], dtype=np.float64) - mean3
        v = float(x @ axis3)
    elif kind == 'pc1_ab':
        _, mean2, axis2 = descriptor
        x = np.array([A, B], dtype=np.float64) - mean2
        v = float(x @ axis2)
    else:
        return None
    return float(v) * sign


# Per-parameter preferred projection order. The picker tries each in order
# and uses the FIRST one whose runtime |r| clears MIN_R_PREFERRED. Falls
# back to "best |r|" search if no preferred candidate qualifies. Reasoning:
#   pH walks along a hue arc → hue_lch is the natural axis.
#   H2O2 walks across both lightness and chroma → pc1_lab.
#   PHMB has small pure-chroma variation → pc1_ab is most stable; B as
#     backup since the chroma path is roughly along the b axis.
# Per-parameter override goes in reference.json:
#   parameters: [{name: 'pH', ..., preferred_projection: ['hue_lch', 'A']}]
DEFAULT_PROJECTION_PREFS = {
    'pH':   ('hue_lch', 'A'),
    'H2O2': ('pc1_lab', 'pc1_ab', 'B'),
    'PHMB': ('pc1_ab', 'B', 'hue_lch'),
}
MIN_R_PREFERRED = 0.9


def _measure_warped(warped: np.ndarray, runtime_grid: dict = None) -> dict:
    """Measure all parameters using runtime-only color regression.

    If `runtime_grid` is supplied (the (row, col) -> detected-cell map from
    _detect_warped_grid), sample at runtime-detected positions. Otherwise
    fall back to reference.json's stored (x, y, w, h) — used when grid
    re-detection has failed.
    """
    lab_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    color_cells = [c for c in _REF['cells']
                   if c['is_color_cell'] and c['value'] is not None]
    measure_cells = [c for c in _REF['cells'] if not c['is_color_cell']]
    param_meta = {p['name']: p for p in _REF['parameters']}
    out: dict = {}

    def _slot_for(c):
        if runtime_grid is not None:
            return runtime_grid.get((c['row_idx'], c['col_idx']))
        return (c['x'] + c['w'] / 2.0, c['y'] + c['h'] / 2.0,
                c['w'], c['h'])

    for param in param_meta:
        p_colors = [c for c in color_cells if c['parameter'] == param]
        p_measure = [c for c in measure_cells if c['parameter'] == param]
        if not p_colors or not p_measure:
            continue

        labs, vals = [], []
        for cell in p_colors:
            slot = _slot_for(cell)
            if slot is None:
                continue
            lab = _sample_lab_at(lab_warped, slot)
            if lab is None:
                continue
            labs.append(lab)
            vals.append(cell['value'])

        if len(labs) < 3:
            continue

        labs_arr = np.array(labs, dtype=np.float64)
        vals_arr = np.array(vals, dtype=np.float64)

        # Build all candidate projections + their |r| against printed values.
        # Each candidate carries a descriptor so probe cells can be projected
        # later via the SAME axis.
        candidates = _build_projections(labs_arr)
        evaluated = {}
        for name, (proj, desc) in candidates.items():
            mask = ~np.isnan(proj)
            if int(mask.sum()) < 3:
                continue
            x = proj[mask].astype(np.float64)
            y = vals_arr[mask].astype(np.float64)
            if x.std() < 1e-6:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            evaluated[name] = (r, x, y, mask, desc)

        if not evaluated:
            continue

        # Pick by per-parameter preferred order if any preferred candidate
        # has |r| >= MIN_R_PREFERRED. Otherwise fall back to global best.
        prefs = (param_meta[param].get('preferred_projection')
                 or DEFAULT_PROJECTION_PREFS.get(param, ()))
        chosen = None
        for pref in prefs:
            ev = evaluated.get(pref)
            if ev is not None and abs(ev[0]) >= MIN_R_PREFERRED:
                chosen = pref
                break
        if chosen is None:
            chosen = max(evaluated, key=lambda n: abs(evaluated[n][0]))

        best_r, x_fit, y_fit, _msk, best_desc = evaluated[chosen]
        best_name = chosen
        best_proj = (x_fit, y_fit, _msk)

        x_fit, y_fit, _ = best_proj
        sign = 1.0 if best_r >= 0 else -1.0
        # Flip sign so values increase monotonically with the projection
        x_fit_signed = x_fit * sign
        coeffs = np.polyfit(x_fit_signed, y_fit,
                            min(2, len(x_fit_signed) - 1))
        rmse = float(np.sqrt(np.mean(
            (np.polyval(coeffs, x_fit_signed) - y_fit) ** 2)))

        # Sample probes at runtime grid positions, project each onto the
        # same axis, average projected scalars, evaluate polynomial.
        probe_projs = []
        for cell in p_measure:
            slot = _slot_for(cell)
            if slot is None:
                continue
            lab = _sample_lab_at(lab_warped, slot)
            if lab is None:
                continue
            v = _project_probe(lab, best_desc, sign)
            if v is None or np.isnan(v):
                continue
            probe_projs.append(v)
        if not probe_projs:
            continue

        probe_x = float(np.mean(probe_projs))
        value = float(np.polyval(coeffs, probe_x))
        ref_vals_sorted = sorted(vals)
        value = float(np.clip(value, ref_vals_sorted[0], ref_vals_sorted[-1]))

        out[param] = {
            'value': round(value, 2),
            'channel': best_name,
            'r': round(best_r, 3),
            'rmse': round(rmse, 3),
        }

    return out


def find_quad_y(y_bytes: bytes, width: int, height: int,
                row_stride: int, rotation_deg: int) -> dict:
    """
    Lightweight per-frame tracker. Reads only the Y plane, rotates to
    display orientation, runs the cell-cluster detector at downscaled
    resolution + stability check. Returns coords in the upright frame.

    No BGR is supplied (Y-plane only), so reference_builder skips its
    grabcut path and uses the gray-only fallbacks (Otsu / Canny / etc.) —
    those still hit 100% on real on-device frames in test_input/.
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
        'method': 'cell_cluster' if quad is not None else 'none',
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

    Two-stage:
      1. Card detection (cell-cluster + Hough refinement) → quad → warp.
      2. Re-detect the cell GRID inside the warped image and sample at the
         runtime-detected positions instead of trusting reference.json's
         (x, y, w, h). The runtime-only color regression then runs against
         those samples.

    If the runtime grid shape doesn't match expected (rows × cols), we fall
    back to reference.json positions — better to attempt with possibly-off
    samples than refuse to measure.
    """
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    quad = reference_builder._detect_reference_quad(gray, bgr)
    if quad is None:
        return {'found': False, 'results': {}, 'quad': None, 'method': 'none'}
    warped = _TRACKER.warp(bgr, quad)

    runtime_grid, (gr_rows, gr_cols), _ = _detect_warped_grid(warped)
    grid_status = (f'{gr_rows}x{gr_cols}'
                   f' ({"OK" if runtime_grid is not None else "mismatch"}'
                   f' / expected {_EXPECTED_ROWS}x{_EXPECTED_COLS})')

    results = _measure_warped(warped, runtime_grid=runtime_grid)
    return {
        'found': True,
        'results': results,
        'quad': quad.tolist(),
        'method': 'cell_cluster',
        'grid_status': grid_status,
    }
