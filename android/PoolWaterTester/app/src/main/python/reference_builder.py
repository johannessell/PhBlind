"""
reference_builder.py
====================
Baut eine kanonische Referenz aus einem rohen Aufnahmebild:

  1. Quad des Messindikators im Bild finden
  2. Perspektivkorrektur auf kanonische Größe (preserves Aspect)
     -> saved als template.jpg; reference.json enthaelt Zellkoords
        im KANONISCHEN Frame. Kamera-Laufzeit warpt ebenfalls in
        diesen Frame -> Zellen passen 1:1.
  3. Zell-Rects im gewarpten Frame detektieren (Canny + approxPolyDP)
  4. Raster aus Zeilen/Spalten ableiten
  5. Spalten als color/measure klassifizieren, raeumlich gruppieren

Returns ein Dict, das die ReferenceActivity + ML Kit weiterverarbeiten:
  parameter-Namen und swatch-Values werden dort ergaenzt, danach
  compute_best_channels aufrufen.
"""

from typing import Optional

import cv2
import numpy as np

from tracker import order_quad_corners

CANONICAL_WIDTH = 720
COLOR_SAT_MAX = 70.0
GROUP_MAX_DISTANCE_FRAC = 0.22  # relative to canonical_width


# ----------------------------------------------------------
# Quad-Erkennung auf der Referenz
# ----------------------------------------------------------

def _touches_border(cnt, w, h, margin=3):
    pts = cnt.reshape(-1, 2)
    return bool(np.any(pts[:, 0] <= margin) or np.any(pts[:, 0] >= w - 1 - margin)
                or np.any(pts[:, 1] <= margin) or np.any(pts[:, 1] >= h - 1 - margin))


def _score_quad(cnt, frame_area, frame_w, frame_h):
    """Score a contour. Returns (score, reason). score<=0 means rejected."""
    area = cv2.contourArea(cnt)
    area_frac = area / frame_area
    if area_frac < 0.05:
        return -1.0, f'small({area_frac:.2f})'
    if area_frac > 0.95:
        return -1.0, f'huge({area_frac:.2f})'
    if _touches_border(cnt, frame_w, frame_h):
        return -1.0, 'border'
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return -1.0, 'thin'
    rect_area = rw * rh
    rectangularity = area / rect_area
    if rectangularity < 0.50:
        return -1.0, f'rect({rectangularity:.2f})'
    return (rectangularity ** 3) * area_frac, f'ok({rectangularity:.2f})'


def _grabcut_mask(bgr: np.ndarray) -> np.ndarray:
    """GrabCut mit zentralem Rechteck als Vordergrund-Hint.

    Nutzt sowohl Farbe als auch Position — funktioniert auch wenn die Karte
    teiltransparent ist und intensitaetsmaessig mit dem Hintergrund verschmilzt.
    """
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    margin_x = w // 6
    margin_y = h // 6
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return out.astype(np.uint8)


def _detect_reference_quad(gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Findet den Tester als groesste rechteckige Region im Bild.

    Primaerstrategie: GrabCut mit zentralem Hint -> Vordergrundsegment ist
    die Karte. Fallbacks: Otsu / Canny / Sobel mit Morph-Close.
    """
    h, w = gray.shape[:2]
    frame_area = float(h * w)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kernelgroesse fuer "fill inner cells" skaliert mit dem Bild —
    # gross genug um Innenzellen+Luecken (~50-60px bei 720) zu schliessen.
    fill_k = max(31, min(w, h) // 12)
    if fill_k % 2 == 0:
        fill_k += 1

    masks = []

    # 1) GrabCut mit Zentrum-Hint — Hauptpfad, robust bei transparenter Karte.
    if bgr is not None:
        gc = _grabcut_mask(bgr)
        # Leichtes Schliessen, falls innere Zellen als BG markiert wurden
        gc_close = cv2.morphologyEx(gc, cv2.MORPH_CLOSE,
                                    np.ones((fill_k, fill_k), np.uint8),
                                    iterations=1)
        masks.append(('grabcut', gc_close))

    # Otsu (helle Karte = Vordergrund). MORPH_CLOSE schliesst die dunklen
    # Innenzellen, damit das Kartenkorpus als ein einzelner Blob erscheint.
    _, m_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fk = np.ones((fill_k, fill_k), np.uint8)
    masks.append(('otsu_filled',
                  cv2.morphologyEx(m_otsu, cv2.MORPH_CLOSE, fk, iterations=1)))
    masks.append(('otsu_inv_filled',
                  cv2.morphologyEx(cv2.bitwise_not(m_otsu), cv2.MORPH_CLOSE,
                                   fk, iterations=1)))

    # Canny + Close: Kartenrand + Innenzellen werden zu einem dichten Blob.
    edges = cv2.Canny(blurred, 30, 120)
    masks.append(('canny_filled',
                  cv2.morphologyEx(edges, cv2.MORPH_CLOSE, fk, iterations=2)))

    # Adaptive Threshold + Fill — robust unter wechselnder Beleuchtung.
    adapt = cv2.adaptiveThreshold(blurred, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 51, 5)
    masks.append(('adapt_filled',
                  cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, fk, iterations=2)))

    # Sobel-Magnitude + Fill — faengt schwache Kontraste, die Canny verpasst.
    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sobel_bin = cv2.threshold(mag_norm, 40, 255, cv2.THRESH_BINARY)
    masks.append(('sobel_filled',
                  cv2.morphologyEx(sobel_bin, cv2.MORPH_CLOSE, fk, iterations=2)))

    if _DEBUG_DIR:
        for name, mask in masks:
            _debug_save(f'10_mask_{name}.jpg', mask)

    candidates = []
    for name, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if _DEBUG_DIR:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # Halbtransparente Maske einblenden
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            vis = cv2.addWeighted(vis, 0.6, mask_bgr, 0.4, 0)
        for cnt in contours:
            score, reason = _score_quad(cnt, frame_area, w, h)
            if score > 0:
                candidates.append((score, cnt, name))
            if _DEBUG_DIR:
                color = (0, 255, 0) if score > 0 else (0, 0, 255)
                cv2.drawContours(vis, [cnt], -1, color, 2)
                x, y, _, _ = cv2.boundingRect(cnt)
                txt = f'{reason}'
                if score > 0:
                    txt = f'{reason} s={score:.2f}'
                cv2.putText(vis, txt, (x, max(y - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if _DEBUG_DIR:
            _debug_save(f'12_contours_{name}.jpg', vis)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    if _DEBUG_DIR:
        # Top-Kandidaten visualisieren
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, (score, cnt, name) in enumerate(candidates[:5]):
            color = (0, 255, 0) if i == 0 else (0, 165, 255)
            cv2.drawContours(vis, [cnt], -1, color, 2)
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.putText(vis, f'#{i} {name} {score:.2f}',
                        (x, max(y - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        _debug_save('11_candidates.jpg', vis)

    # Versuche fuer die top-N Kandidaten approxPolyDP -> Quad
    for _, cnt, _name in candidates[:10]:
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1:
            continue
        for eps_frac in (0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10):
            approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = approx.reshape(4, 2).astype(np.float32)
                return order_quad_corners(quad)

    # Fallback: minAreaRect des besten Kandidaten
    box = cv2.boxPoints(cv2.minAreaRect(candidates[0][1])).astype(np.float32)
    return order_quad_corners(box)


def _warp_to_canonical(bgr: np.ndarray, quad: np.ndarray,
                       canonical_width: int = CANONICAL_WIDTH):
    """Perspektivkorrektur: Quad -> achsenausgerichtetes Rechteck."""
    pts = quad.astype(np.float32)
    w1 = float(np.linalg.norm(pts[1] - pts[0]))
    w2 = float(np.linalg.norm(pts[2] - pts[3]))
    h1 = float(np.linalg.norm(pts[3] - pts[0]))
    h2 = float(np.linalg.norm(pts[2] - pts[1]))
    w_px = max((w1 + w2) * 0.5, 1.0)
    h_px = max((h1 + h2) * 0.5, 1.0)

    cw = int(canonical_width)
    ch = int(round(canonical_width * h_px / w_px))
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(
        bgr, M, (cw, ch),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, (cw, ch)


# ----------------------------------------------------------
# Zell-Rects im gewarpten Frame
# ----------------------------------------------------------

def _detect_cells(warped_bgr: np.ndarray):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    canny = cv2.Canny(enhanced, 20, 80)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_frame, w_frame = gray.shape
    frame_area = h_frame * w_frame

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < frame_area * 0.001:
            continue
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if w < 6 or h < 6:
                continue
            rects.append((int(x), int(y), int(w), int(h)))
    return rects


# ----------------------------------------------------------
# Gitter-Aufbau
# ----------------------------------------------------------

def _group_by_axis(rects, axis, tol):
    rects_sorted = sorted(rects, key=lambda r: r[axis])
    groups = []
    for r in rects_sorted:
        placed = False
        for g in groups:
            if abs(g[0][axis] - r[axis]) < tol:
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])
    return groups


def _build_grid(rects, canonical_width):
    row_tol = max(10, int(canonical_width * 0.03))
    col_tol = max(10, int(canonical_width * 0.03))

    rows = _group_by_axis(rects, axis=1, tol=row_tol)
    cols = _group_by_axis(rects, axis=0, tol=col_tol)

    row_positions = [min(r[1] for r in row) for row in rows]
    row_heights = [max(r[3] for r in row) for row in rows]
    col_positions = [min(r[0] for r in col) for col in cols]
    col_widths = [max(r[2] for r in col) for col in cols]

    row_order = sorted(range(len(rows)), key=lambda i: row_positions[i])
    col_order = sorted(range(len(cols)), key=lambda i: col_positions[i])

    sorted_cols = [cols[ci] for ci in col_order]
    sorted_col_positions = [col_positions[ci] for ci in col_order]
    sorted_col_widths = [col_widths[ci] for ci in col_order]
    sorted_row_positions = [row_positions[ri] for ri in row_order]
    sorted_row_heights = [row_heights[ri] for ri in row_order]

    grid = []
    for ri, ry in enumerate(sorted_row_positions):
        for ci, cx in enumerate(sorted_col_positions):
            grid.append({
                'row_idx': ri,
                'col_idx': ci,
                'x': cx,
                'y': ry,
                'w': sorted_col_widths[ci],
                'h': sorted_row_heights[ri],
            })
    return (grid, sorted_cols, sorted_col_positions,
            sorted_col_widths, sorted_row_positions, sorted_row_heights)


# ----------------------------------------------------------
# Spalten-Klassifikation + Gruppierung
# ----------------------------------------------------------

def _classify_columns(warped_bgr, cols):
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    col_types = {}
    col_stats = {}
    for i, col_rects in enumerate(cols):
        cell_sat_medians = []
        for (x, y, w, h) in col_rects:
            roi = hsv[y:y + h, x:x + w]
            if roi.size > 0:
                cell_sat_medians.append(float(np.median(roi[:, :, 1])))
        max_sat = max(cell_sat_medians) if cell_sat_medians else 0.0
        col_stats[i] = {'cell_sat_max': max_sat}
        col_types[i] = 'color' if max_sat >= COLOR_SAT_MAX else 'measure'
    return col_types, col_stats


def _group_columns_spatially(col_types, col_positions, canonical_width):
    max_distance = canonical_width * GROUP_MAX_DISTANCE_FRAC
    measure_cols = [i for i, t in col_types.items() if t == 'measure']
    color_cols = [i for i, t in col_types.items() if t == 'color']

    groups = []
    used = set()

    for mi in sorted(measure_cols):
        if mi in used:
            continue
        grp_cols = [mi]
        used.add(mi)
        mx = col_positions[mi]
        for ci in color_cols:
            if ci in used:
                continue
            if abs(col_positions[ci] - mx) <= max_distance:
                grp_cols.append(ci)
                used.add(ci)
        has_color = any(col_types[c] == 'color' for c in grp_cols)
        if has_color:
            grp_cols.sort(key=lambda i: col_positions[i])
            groups.append({'cols': grp_cols, 'measure_col': mi})
        else:
            used.discard(mi)

    for i in col_types:
        if i in used:
            continue
        used.add(i)
        groups.append({
            'cols': [i],
            'measure_col': i if col_types[i] == 'measure' else None,
        })

    groups.sort(key=lambda g: col_positions[g['cols'][0]])
    return groups


# ----------------------------------------------------------
# Haupt-Entrypoints
# ----------------------------------------------------------

_DEBUG_DIR = ''


def set_debug_dir(path: str) -> None:
    """Setzt das Verzeichnis fuer Debug-Bilder (input, masks, detected quad)."""
    global _DEBUG_DIR
    _DEBUG_DIR = path or ''
    if _DEBUG_DIR:
        import os
        os.makedirs(_DEBUG_DIR, exist_ok=True)


def _debug_save(name: str, img: np.ndarray) -> None:
    if not _DEBUG_DIR:
        return
    import os
    cv2.imwrite(os.path.join(_DEBUG_DIR, name), img)


def build_reference(bgr: np.ndarray) -> dict:
    """
    Raw BGR reference photo -> canonical warped template + cells + groups.
    parameter-Namen und Values bleiben None (werden vom ML-Kit / User gefuellt).
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("Leeres Eingangsbild")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _debug_save('00_input.jpg', bgr)
    _debug_save('01_gray.jpg', gray)

    quad = _detect_reference_quad(gray, bgr)
    if quad is None:
        raise RuntimeError("Messindikator nicht im Referenzbild erkannt")

    if _DEBUG_DIR:
        vis = bgr.copy()
        cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 3)
        for i, p in enumerate(quad.astype(int)):
            cv2.putText(vis, ['TL', 'TR', 'BR', 'BL'][i],
                        tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        _debug_save('02_quad.jpg', vis)

    warped_bgr, (cw, ch) = _warp_to_canonical(bgr, quad)
    _debug_save('03_warped.jpg', warped_bgr)
    rects = _detect_cells(warped_bgr)
    if len(rects) < 6:
        raise RuntimeError(f"Zu wenige Zellen erkannt: {len(rects)}")

    (grid, sorted_cols, col_positions, col_widths,
     row_positions, row_heights) = _build_grid(rects, cw)
    col_types, col_stats = _classify_columns(warped_bgr, sorted_cols)
    col_groups = _group_columns_spatially(col_types, col_positions, cw)

    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)

    cells = []
    for i, g in enumerate(grid):
        ci = g['col_idx']
        is_color = col_types.get(ci) == 'color'
        g_idx = next(
            (gi for gi, grp in enumerate(col_groups) if ci in grp['cols']),
            None,
        )
        x, y, w, h = g['x'], g['y'], g['w'], g['h']
        roi_hsv = hsv[y:y + h, x:x + w]
        roi_lab = lab[y:y + h, x:x + w]
        if roi_hsv.size > 0:
            hsv_med = [
                float(np.median(roi_hsv[:, :, 0])),
                float(np.median(roi_hsv[:, :, 1])),
                float(np.median(roi_hsv[:, :, 2])),
            ]
            lab_med = [
                float(np.median(roi_lab[:, :, 0])),
                float(np.median(roi_lab[:, :, 1])),
                float(np.median(roi_lab[:, :, 2])),
            ]
        else:
            hsv_med = [None, None, None]
            lab_med = [None, None, None]

        cells.append({
            'cell_idx': i,
            'row_idx': g['row_idx'],
            'col_idx': ci,
            'group_idx': g_idx,
            'parameter': None,
            'is_color_cell': is_color,
            'x': x, 'y': y, 'w': w, 'h': h,
            'hsv_median': hsv_med,
            'lab_median': lab_med,
            'value': None,
        })

    parameters = []
    for gi, grp in enumerate(col_groups):
        parameters.append({
            'name': None,
            'group_idx': gi,
            'cols': grp['cols'],
            'measure_col': grp['measure_col'],
            'best_channel': None,
            'best_r': None,
            'poly_coeffs': None,
            'val_min': None,
            'val_max': None,
        })

    ok, buf = cv2.imencode('.jpg', warped_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    warped_jpg = bytes(buf) if ok else b''

    return {
        'warped_jpg': warped_jpg,
        'width': cw,
        'height': ch,
        'cells': cells,
        'parameters': parameters,
        'col_positions': [int(p) for p in col_positions],
        'col_widths': [int(w) for w in col_widths],
        'col_types': {int(k): v for k, v in col_types.items()},
    }


def build_reference_from_rgba(rgba_bytes: bytes, width: int, height: int) -> dict:
    """Reference build aus voller Aufloesung — kein Downscaling.

    Detection-Code muss skalierungs-invariant arbeiten (Schwellen als
    Anteil von frame_area, Radius als Vielfaches der Median-Zellgroesse).
    """
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return build_reference(bgr)


# ----------------------------------------------------------
# Java <-> Python Coercion (Chaquopy passes java.util.List/Map
# which are NOT Python iterables/dicts — must coerce first).
# ----------------------------------------------------------

try:
    from java.util import Map as _JMap, List as _JList
except ImportError:  # not running on Chaquopy (e.g. unit tests)
    _JMap = _JList = None


def _to_py(obj):
    """Recursively convert Java List/Map to Python list/dict."""
    if obj is None:
        return None
    if _JMap is not None and isinstance(obj, _JMap):
        out = {}
        for entry in obj.entrySet().toArray():
            out[_to_py(entry.getKey())] = _to_py(entry.getValue())
        return out
    if _JList is not None and isinstance(obj, _JList):
        return [_to_py(obj.get(i)) for i in range(obj.size())]
    return obj


# ----------------------------------------------------------
# OCR-Zuordnung (Blocks von ML Kit)
# ----------------------------------------------------------

_OCR_CORRECTIONS = {
    'HO2': 'H2O2', 'H02': 'H2O2', 'H0²': 'H2O2', 'HO²': 'H2O2',
    'Ho2': 'H2O2', 'h2o2': 'H2O2', 'h2o': 'H2O',
    'Ph': 'pH', 'PH': 'pH', 'ph': 'pH',
    'PhMS': 'PHMB', 'PHMS': 'PHMB', 'Phmb': 'PHMB',
}


def _correct_param(text: str) -> str:
    return _OCR_CORRECTIONS.get(text.strip(), text.strip())


def _parse_number(text: str):
    t = text.strip().replace(' ', '').replace(',', '.')
    try:
        return float(t)
    except ValueError:
        return None


def _block_center(b: dict):
    return (b['x'] + b['w'] * 0.5, b['y'] + b['h'] * 0.5)


def _cell_at(cells: list, cx: float, cy: float):
    """Return the cell whose rect contains (cx, cy), or None."""
    for c in cells:
        if c['x'] <= cx <= c['x'] + c['w'] and c['y'] <= cy <= c['y'] + c['h']:
            return c
    return None


def _group_bounds(cells: list, parameters: list):
    """Per group: (x_min, x_max, y_min_of_first_row)."""
    out = {}
    for p in parameters:
        gi = p['group_idx']
        cells_in_g = [c for c in cells if c.get('group_idx') == gi]
        if not cells_in_g:
            continue
        xs = [c['x'] for c in cells_in_g]
        xe = [c['x'] + c['w'] for c in cells_in_g]
        ys = [c['y'] for c in cells_in_g]
        out[gi] = (min(xs), max(xe), min(ys))
    return out


def assign_ocr_to_cells(ref: dict, ocr_blocks: list) -> dict:
    """
    ref: output of build_reference() (mutable).
    ocr_blocks: list of {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float}
                in canonical warped-template coordinates (ML Kit on the warped bitmap).

    Strategy:
      - Split blocks into "header" (above all cells) and "body" (overlapping cells).
      - Header blocks -> parameter names per group (best score wins).
      - Body blocks -> parsed as number, written into the cell they fall inside
        (measure cells only — color swatches don't carry printed numbers).

    Returns the same ref dict, mutated with parameter names + cell values prefilled.
    """
    ocr_blocks = _to_py(ocr_blocks) or []
    cells = ref.get('cells', [])
    parameters = ref.get('parameters', [])
    if not cells:
        return ref

    grid_top = min(c['y'] for c in cells)
    grid_bottom = max(c['y'] + c['h'] for c in cells)
    bounds = _group_bounds(cells, parameters)

    header_blocks = [b for b in ocr_blocks if _block_center(b)[1] < grid_top]
    body_blocks = [b for b in ocr_blocks if grid_top <= _block_center(b)[1] <= grid_bottom]

    # Header -> parameter names (one per group, highest score wins)
    best_per_group: dict = {}
    for b in header_blocks:
        cx, _ = _block_center(b)
        for gi, (x_min, x_max, _) in bounds.items():
            if x_min <= cx <= x_max:
                prev = best_per_group.get(gi)
                if prev is None or b.get('score', 0.0) > prev.get('score', 0.0):
                    best_per_group[gi] = b
                break

    for p in parameters:
        blk = best_per_group.get(p['group_idx'])
        if blk and blk.get('text'):
            p['name'] = _correct_param(blk['text'])

    # Propagate parameter name onto cells
    param_by_group = {p['group_idx']: p.get('name') for p in parameters}
    for c in cells:
        gi = c.get('group_idx')
        if gi is not None:
            c['parameter'] = param_by_group.get(gi)

    # Body -> numeric values per cell (measure cells only)
    for b in body_blocks:
        val = _parse_number(b.get('text', ''))
        if val is None:
            continue
        cx, cy = _block_center(b)
        cell = _cell_at(cells, cx, cy)
        if cell is None or cell.get('is_color_cell'):
            continue
        # For measure cells, OCR gives the row's VALUE which applies to all
        # color swatches in the same row+group. Store on the measure cell
        # for now; ReferenceActivity will distribute to the matching color
        # cells on save.
        cell['value'] = val

    return ref


def spread_row_values(ref: dict) -> dict:
    """
    After user confirmation in the editor, copy the measure-column value
    of each (row, group) onto its color swatches. Swatches carry the
    y-value that the poly-fit needs.
    """
    cells = ref.get('cells', [])
    by_row_group = {}
    for c in cells:
        if c.get('is_color_cell'):
            continue
        v = c.get('value')
        if v is None:
            continue
        key = (c.get('row_idx'), c.get('group_idx'))
        by_row_group[key] = v
    for c in cells:
        if not c.get('is_color_cell'):
            continue
        key = (c.get('row_idx'), c.get('group_idx'))
        v = by_row_group.get(key)
        if v is not None:
            c['value'] = v
    return ref


# ----------------------------------------------------------
# Poly-Fit fuer die finale reference.json
# ----------------------------------------------------------

def compute_best_channels(warped_bgr: np.ndarray,
                          cells: list,
                          parameters: list) -> tuple:
    """
    Fuer jeden Parameter mit zugewiesenen color-Zellen + Werten:
    besten LAB-Kanal, Poly-Fit (value = f(channel)), val_min, val_max.
    Mutates cells (lab_median refresh) and parameters in place.
    """
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    for c in cells:
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        roi = lab[y:y + h, x:x + w]
        if roi.size > 0:
            c['lab_median'] = [
                float(np.median(roi[:, :, 0])),
                float(np.median(roi[:, :, 1])),
                float(np.median(roi[:, :, 2])),
            ]

    names = {0: 'L', 1: 'A', 2: 'B'}
    for p in parameters:
        pc = [c for c in cells
              if c.get('parameter') == p.get('name')
              and c.get('is_color_cell')
              and c.get('value') is not None]
        if len(pc) < 3:
            p['best_channel'] = None
            p['best_r'] = None
            p['poly_coeffs'] = None
            p['val_min'] = None
            p['val_max'] = None
            continue
        labs_arr = np.array([c['lab_median'] for c in pc], dtype=np.float64)
        vals = np.array([c['value'] for c in pc], dtype=np.float64)

        best_r, best_ch = 0.0, 1
        for ch in range(3):
            x_arr = labs_arr[:, ch]
            if x_arr.std() < 1e-6:
                continue
            r = float(np.corrcoef(x_arr, vals)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_ch = r, ch
        x_arr = labs_arr[:, best_ch]
        degree = min(2, len(pc) - 1)
        coeffs = np.polyfit(x_arr, vals, degree).tolist()

        p['best_channel'] = names[best_ch]
        p['best_r'] = round(best_r, 3)
        p['poly_coeffs'] = [float(c) for c in coeffs]
        p['val_min'] = float(vals.min())
        p['val_max'] = float(vals.max())

    return cells, parameters


# ----------------------------------------------------------
# Android-Bridge: Edits anwenden + Finalisierung + Serialisierung
# ----------------------------------------------------------

def apply_edits_and_finalize(ref: dict,
                             edited_values,
                             edited_names) -> dict:
    """
    edited_values: map cell_idx (int or str) -> str (user-typed, possibly '')
    edited_names:  map group_idx (int or str) -> str (user-typed parameter name)

    Overwrites ref['cells'][*]['value'] and ref['parameters'][*]['name'],
    then propagates names onto cells and spreads measure-cell values onto
    color swatches in the same (row, group).
    """
    def _to_int(k):
        try:
            return int(k)
        except (TypeError, ValueError):
            return None

    edited_values = _to_py(edited_values) or {}
    edited_names = _to_py(edited_names) or {}

    values_by_idx = {}
    for k, v in edited_values.items():
        ki = _to_int(k)
        if ki is None:
            continue
        s = (v or '').strip() if isinstance(v, str) else v
        values_by_idx[ki] = _parse_number(s) if s else None

    names_by_group = {}
    for k, v in edited_names.items():
        ki = _to_int(k)
        if ki is None:
            continue
        s = (v or '').strip() if isinstance(v, str) else v
        names_by_group[ki] = s if s else None

    for c in ref.get('cells', []):
        if c.get('is_color_cell'):
            continue
        if c['cell_idx'] in values_by_idx:
            c['value'] = values_by_idx[c['cell_idx']]

    for p in ref.get('parameters', []):
        gi = p.get('group_idx')
        if gi in names_by_group:
            p['name'] = names_by_group[gi]

    name_by_group = {p.get('group_idx'): p.get('name')
                     for p in ref.get('parameters', [])}
    for c in ref.get('cells', []):
        gi = c.get('group_idx')
        if gi is not None:
            c['parameter'] = name_by_group.get(gi)

    spread_row_values(ref)
    return ref


def compute_best_channels_rgba(ref: dict,
                               rgba_bytes: bytes,
                               width: int,
                               height: int) -> dict:
    """Convenience wrapper for Android: decode RGBA -> BGR, run poly-fit."""
    arr = np.frombuffer(rgba_bytes, dtype=np.uint8).reshape(height, width, 4)
    warped_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    compute_best_channels(warped_bgr, ref.get('cells', []),
                          ref.get('parameters', []))
    return ref


def ref_to_json_str(ref: dict) -> str:
    """
    Serialize ref to a JSON string suitable for reference.json.
    Drops the raw warped_jpg bytes (template is saved separately as template.jpg).
    """
    import json
    out = {k: v for k, v in ref.items() if k != 'warped_jpg'}
    return json.dumps(out, ensure_ascii=False, indent=2)
