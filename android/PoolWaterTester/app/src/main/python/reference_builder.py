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

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tracker import order_quad_corners

CANONICAL_WIDTH = 720
COLOR_SAT_MAX = 70.0
GROUP_MAX_DISTANCE_FRAC = 0.22  # relative to canonical_width


# ----------------------------------------------------------
# Quad-Erkennung auf der Referenz
# ----------------------------------------------------------

def _dedupe_cells(cells, center_tol_factor: float = 0.3, size_tol: float = 0.25):
    """Verwerfe Duplikate: zwei Zellen am gleichen Zentrum mit aehnlicher Groesse.

    findContours liefert oft sowohl die Aussen- als auch Innenkante eines
    Zellrahmens — beide ergeben fast deckungsgleiche minAreaRect-Boxen.
    Wir behalten pro Cluster den groesseren Eintrag (=Aussenkontur).
    """
    if len(cells) < 2:
        return cells
    sorted_cells = sorted(cells, key=lambda c: c[2] * c[3], reverse=True)
    keep = []
    for c in sorted_cells:
        cx, cy, w, h, _, _ = c
        center_tol = max(w, h) * center_tol_factor
        is_dup = False
        for k in keep:
            kx, ky, kw, kh, _, _ = k
            if (abs(kx - cx) <= center_tol and abs(ky - cy) <= center_tol
                    and abs(kw - w) <= max(kw, w) * size_tol
                    and abs(kh - h) <= max(kh, h) * size_tol):
                is_dup = True
                break
        if not is_dup:
            keep.append(c)
    return keep


def _drop_nested(cells):
    """Entferne Zellen, die komplett in einer groesseren Zelle liegen.

    Innenstrukturen wie Zahlen/Text in einer Zelle erzeugen eigene Konturen;
    deren minAreaRect liegt mittig in der echten Zelle.
    """
    if len(cells) < 2:
        return cells
    sorted_cells = sorted(cells, key=lambda c: c[2] * c[3], reverse=True)
    keep = []
    for c in sorted_cells:
        cx, cy, w, h, _, _ = c
        nested = False
        for k in keep:
            kx, ky, kw, kh, _, _ = k
            if (kw * kh > w * h * 1.2
                    and kx - kw / 2 <= cx <= kx + kw / 2
                    and ky - kh / 2 <= cy <= ky + kh / 2):
                nested = True
                break
        if not nested:
            keep.append(c)
    return keep


def _compute_edges(gray: np.ndarray) -> np.ndarray:
    """Einmalige Edge-Berechnung — wird sowohl von Zell-Detection als auch
    von der Hough-Verfeinerung wiederverwendet.
    """
    filtered = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    edges = cv2.Canny(enhanced, 30, 90)
    _debug_save('20_edges.jpg', edges)
    return edges


def _detect_cell_rects(edges: np.ndarray) -> List[Tuple[int, int, int, int, float, float]]:
    """Finde alle quadratischen/rechteckigen kleinen Konturen.

    Rueckgabe: Liste von (cx, cy, w, h, area, angle_deg).
    """
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8), iterations=1)
    _debug_save('20b_edges_closed.jpg', edges_closed)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = edges.shape
    frame_area = float(h * w)
    min_cell = frame_area * 0.0001
    max_cell = frame_area * 0.02

    cells: List[Tuple[int, int, int, int, float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_cell or area > max_cell:
            continue
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]
        if min(rw, rh) < 4:
            continue
        ar = max(rw, rh) / (min(rw, rh) + 1e-6)
        if ar > 4.5:
            continue
        rect_area = rw * rh
        if rect_area < 1:
            continue
        rectangularity = area / rect_area
        if rectangularity < 0.6:
            continue
        ang = rect[2]
        if rw < rh:
            ang += 90.0
        while ang > 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        cx, cy = rect[0]
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        cells.append((int(cx), int(cy), int(long_side), int(short_side),
                      float(area), float(ang)))

    cells = _dedupe_cells(cells)
    cells = _drop_nested(cells)
    return cells


def _filter_dense_cells(cells, k_neighbors: int = 6, radius_factor: float = 2.5):
    """Behalte nur Zellen mit >= k Nachbarn im Radius (skaliert mit Median-Zellgroesse)."""
    if len(cells) < k_neighbors:
        return []
    pts = np.array([(c[0], c[1]) for c in cells], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cells], dtype=np.float32)
    median_size = float(np.median(sizes))
    radius = max(median_size * radius_factor, 30.0)

    keep_idx = []
    for i, p in enumerate(pts):
        dists = np.linalg.norm(pts - p, axis=1)
        if int(np.sum(dists < radius)) >= k_neighbors:
            keep_idx.append(i)
    return [cells[i] for i in keep_idx]


def _drop_boundary_outliers(cells, count_floor_frac: float = 0.5,
                            gap_factor: float = 1.5):
    """Drop sparse boundary columns/rows from the dense cluster.

    A real card grid has uniform spacing — every column has roughly the same
    number of cells, every row likewise. A stray rectangle from clutter that
    survived the density filter typically shows up as a 1-cell-only column or
    row at the EDGE of the cell-bbox. Dropping it pulls the bbox back to the
    actual card.

    Algorithm:
      1. Project cells into the card frame (rotate by median per-cell angle).
      2. 1-D-cluster rx values into columns and ry into rows. A gap larger
         than `gap_factor * median(cell_short)` between adjacent values
         starts a new cluster.
      3. Boundary clusters (first and last in sort order) whose member count
         is < count_floor_frac * median(counts) are flagged.
      4. Drop the cells in those flagged boundary clusters.

    Returns (kept_cells, dropped_cells) — dropped is for debug visualization.
    """
    if len(cells) < 6:
        return cells, []
    pts = np.array([(c[0], c[1]) for c in cells], dtype=np.float32)
    cell_angles = np.array([c[5] for c in cells], dtype=np.float32)
    angle_deg = float(np.median(cell_angles))
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    centroid = pts.mean(axis=0)
    rotated = (pts - centroid) @ R.T
    rx_arr = rotated[:, 0]
    ry_arr = rotated[:, 1]

    cell_short = float(np.median([min(c[2], c[3]) for c in cells]))
    gap = cell_short * gap_factor

    def _cluster_1d(values):
        order = np.argsort(values)
        clusters = [[int(order[0])]]
        for i in range(1, len(order)):
            if values[order[i]] - values[order[i - 1]] > gap:
                clusters.append([])
            clusters[-1].append(int(order[i]))
        return clusters

    rx_clusters = _cluster_1d(rx_arr)
    ry_clusters = _cluster_1d(ry_arr)

    drop_idx = set()

    def _flag_boundaries(clusters):
        if len(clusters) < 3:
            return  # Need >=3 columns/rows so removing a boundary keeps a grid
        counts = [len(c) for c in clusters]
        median_count = float(np.median(counts))
        threshold = median_count * count_floor_frac
        if counts[0] < threshold:
            drop_idx.update(clusters[0])
        if counts[-1] < threshold:
            drop_idx.update(clusters[-1])

    _flag_boundaries(rx_clusters)
    _flag_boundaries(ry_clusters)

    kept = [c for i, c in enumerate(cells) if i not in drop_idx]
    dropped = [c for i, c in enumerate(cells) if i in drop_idx]
    return kept, dropped


def _filter_cells_by_size(cells, tolerance: float = 0.40, n_bins: int = 20):
    """Behalte nur Zellen, deren kurze Seite zum Histogramm-Modus passt.

    Karten-Zellen sind alle gleich hoch; Streu-Rechtecke (Stiftehalter,
    Monitorrahmen, Tastenkappen) haben i.d.R. andere Groessen.
    """
    if len(cells) < 4:
        return cells
    heights = np.array([c[3] for c in cells], dtype=np.float32)
    hist, edges = np.histogram(heights, bins=n_bins)
    mode_idx = int(np.argmax(hist))
    mode_h = float(0.5 * (edges[mode_idx] + edges[mode_idx + 1]))
    lo = mode_h * (1.0 - tolerance)
    hi = mode_h * (1.0 + tolerance)
    return [c for c in cells if lo <= c[3] <= hi]


def _refine_quad_via_hough(gray: np.ndarray, edges_full: np.ndarray, cells,
                           card_angle_deg: float, margin: int = 100) -> Optional[np.ndarray]:
    """ROI um Zellen + HoughLines -> 4 dominante Linien -> Schnittpunkte = Eckpunkte."""
    if not cells:
        return None
    h_img, w_img = gray.shape[:2]
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    sizes = [max(c[2], c[3]) for c in cells]
    half_cell = int(np.median(sizes) // 2)

    x0 = max(0, min(xs) - half_cell - margin)
    y0 = max(0, min(ys) - half_cell - margin)
    x1 = min(w_img, max(xs) + half_cell + margin)
    y1 = min(h_img, max(ys) + half_cell + margin)

    crop = gray[y0:y1, x0:x1]
    edges = edges_full[y0:y1, x0:x1]
    if crop.size == 0 or edges.size == 0:
        return None

    if _DEBUG_DIR:
        vis_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_roi, (x0, y0), (x1, y1), (255, 200, 0), 4)
        _debug_save('24_roi.jpg', vis_roi)
        _debug_save('25_crop.jpg', crop)
        _debug_save('26_crop_edges.jpg', edges)

    cell_short = float(np.median([min(c[2], c[3]) for c in cells]))
    threshold = max(int(cell_short * 1.0), 30)
    min_line_len = max(int(cell_short * 2.5), 50)
    lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                              threshold=threshold,
                              minLineLength=min_line_len,
                              maxLineGap=10)
    if lines_p is None:
        lines_p = []

    card_angle_rad = np.deg2rad(card_angle_deg)
    cos_t, sin_t = float(np.cos(card_angle_rad)), float(np.sin(card_angle_rad))
    cell_pts_crop = np.array([(c[0] - x0, c[1] - y0) for c in cells], dtype=np.float32)
    cell_centroid = cell_pts_crop.mean(axis=0)
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    rotated_cells = (cell_pts_crop - cell_centroid) @ R.T
    cmin = rotated_cells.min(axis=0)
    cmax = rotated_cells.max(axis=0)
    pad = 0.5 * cell_short
    inner_xmin = cmin[0] - pad
    inner_xmax = cmax[0] + pad
    inner_ymin = cmin[1] - pad
    inner_ymax = cmax[1] + pad

    def _to_card_frame(x, y):
        dx = x - cell_centroid[0]
        dy = y - cell_centroid[1]
        return dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t

    def _is_inside_inner(x, y):
        rx, ry = _to_card_frame(x, y)
        return inner_xmin <= rx <= inner_xmax and inner_ymin <= ry <= inner_ymax

    margin_lo = 0.3 * cell_short
    margin_hi = 2.5 * cell_short

    h_target = card_angle_rad % np.pi
    v_target = (card_angle_rad + np.pi / 2) % np.pi
    angle_tol = np.deg2rad(15)

    def _ang_dist(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    def _is_horizontal(seg_deg):
        return seg_deg <= 15.0 or seg_deg >= 165.0

    def _is_vertical(seg_deg):
        return 75.0 <= seg_deg <= 105.0

    def _categorize(lines):
        top_, bottom_, left_, right_, acc_ = [], [], [], [], []
        for line in lines:
            x1, y1, x2, y2 = (float(v) for v in line[0])
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5
            if _is_inside_inner(mx, my):
                continue
            seg_angle = float(math.atan2(y2 - y1, x2 - x1)) % np.pi
            seg_deg = np.degrees(seg_angle)
            rx, ry = _to_card_frame(mx, my)
            if _is_horizontal(seg_deg) and _ang_dist(seg_angle, h_target) < angle_tol:
                if cmin[1] - margin_hi <= ry <= cmin[1] - margin_lo:
                    top_.append((x1, y1, x2, y2, seg_angle, ry))
                    acc_.append(('top', x1, y1, x2, y2))
                elif cmax[1] + margin_lo <= ry <= cmax[1] + margin_hi:
                    bottom_.append((x1, y1, x2, y2, seg_angle, ry))
                    acc_.append(('bottom', x1, y1, x2, y2))
            elif _is_vertical(seg_deg) and _ang_dist(seg_angle, v_target) < angle_tol:
                if cmin[0] - margin_hi <= rx <= cmin[0] - margin_lo:
                    left_.append((x1, y1, x2, y2, seg_angle, rx))
                    acc_.append(('left', x1, y1, x2, y2))
                elif cmax[0] + margin_lo <= rx <= cmax[0] + margin_hi:
                    right_.append((x1, y1, x2, y2, seg_angle, rx))
                    acc_.append(('right', x1, y1, x2, y2))
        return top_, bottom_, left_, right_, acc_

    horizontal_top, horizontal_bottom, vertical_left, vertical_right, accepted = \
        _categorize(lines_p)

    # Per-side relaxed retry: if a side has zero candidates after the normal
    # pass, run HoughLinesP again on the same edges with halved threshold +
    # minLineLength. Only the empty sides absorb the new candidates — sides
    # that already had hits keep their stricter (higher-quality) results.
    if not (horizontal_top and horizontal_bottom
            and vertical_left and vertical_right):
        relaxed_lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=max(threshold // 3, 10),
            minLineLength=max(min_line_len // 2, 25),
            maxLineGap=15,
        )
        if relaxed_lines is not None:
            r_top, r_bot, r_left, r_right, r_acc = _categorize(relaxed_lines)
            if not horizontal_top:
                horizontal_top = r_top
                accepted.extend(s for s in r_acc if s[0] == 'top')
            if not horizontal_bottom:
                horizontal_bottom = r_bot
                accepted.extend(s for s in r_acc if s[0] == 'bottom')
            if not vertical_left:
                vertical_left = r_left
                accepted.extend(s for s in r_acc if s[0] == 'left')
            if not vertical_right:
                vertical_right = r_right
                accepted.extend(s for s in r_acc if s[0] == 'right')

    # Per-side selection: pick the candidate line whose position is closest
    # to the MEDIAN of the candidates on that side.
    def _median_pick(items, key_idx):
        if not items:
            return None
        vals = sorted(it[key_idx] for it in items)
        median = vals[len(vals) // 2]
        return min(items, key=lambda it: abs(it[key_idx] - median))

    top = _median_pick(horizontal_top, 5)
    bottom = _median_pick(horizontal_bottom, 5)
    left = _median_pick(vertical_left, 5)
    right = _median_pick(vertical_right, 5)

    # Synthesize a line from cell-bbox + half_cell margin for any side that
    # is STILL empty. Card-frame to image-frame conversion: a card-frame
    # point (rx, ry) maps to image as cell_centroid + (rx, ry) @ R.
    def _card_to_img(rx_, ry_):
        return (rx_ * cos_t - ry_ * sin_t + cell_centroid[0],
                rx_ * sin_t + ry_ * cos_t + cell_centroid[1])

    half = 0.5 * cell_short
    synth_sides = set()
    if top is None:
        ry0 = cmin[1] - half
        p1 = _card_to_img(-1000.0, ry0)
        p2 = _card_to_img(+1000.0, ry0)
        top = (p1[0], p1[1], p2[0], p2[1], h_target, ry0)
        synth_sides.add('top')
    if bottom is None:
        ry0 = cmax[1] + half
        p1 = _card_to_img(-1000.0, ry0)
        p2 = _card_to_img(+1000.0, ry0)
        bottom = (p1[0], p1[1], p2[0], p2[1], h_target, ry0)
        synth_sides.add('bottom')
    if left is None:
        rx0 = cmin[0] - half
        p1 = _card_to_img(rx0, -1000.0)
        p2 = _card_to_img(rx0, +1000.0)
        left = (p1[0], p1[1], p2[0], p2[1], v_target, rx0)
        synth_sides.add('left')
    if right is None:
        rx0 = cmax[0] + half
        p1 = _card_to_img(rx0, -1000.0)
        p2 = _card_to_img(rx0, +1000.0)
        right = (p1[0], p1[1], p2[0], p2[1], v_target, rx0)
        synth_sides.add('right')

    if _DEBUG_DIR:
        vis_lines = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for line in lines_p:
            lx1, ly1, lx2, ly2 = (int(v) for v in line[0])
            cv2.line(vis_lines, (lx1, ly1), (lx2, ly2), (60, 60, 60), 1)
        side_color = {'top': (0, 255, 0), 'bottom': (0, 255, 0),
                      'left': (0, 200, 255), 'right': (0, 200, 255)}
        for side, ax1, ay1, ax2, ay2 in accepted:
            cv2.line(vis_lines, (int(ax1), int(ay1)), (int(ax2), int(ay2)),
                     side_color[side], 2)
        chosen_lines = [('top', top, (0, 0, 255)), ('bottom', bottom, (0, 0, 255)),
                        ('left', left, (255, 0, 255)), ('right', right, (255, 0, 255))]
        for side_name, chosen, color in chosen_lines:
            if chosen is None:
                continue
            cx1, cy1, cx2, cy2 = chosen[0], chosen[1], chosen[2], chosen[3]
            if side_name in synth_sides:
                # Dashed white = synthesized from cell-bbox + half_cell margin
                pts = np.linspace([cx1, cy1], [cx2, cy2], num=40)
                for i in range(0, len(pts) - 1, 2):
                    cv2.line(vis_lines,
                             (int(pts[i][0]), int(pts[i][1])),
                             (int(pts[i + 1][0]), int(pts[i + 1][1])),
                             (255, 255, 255), 2)
            else:
                cv2.line(vis_lines, (int(cx1), int(cy1)),
                         (int(cx2), int(cy2)), color, 3)
        _debug_save('26b_hough.jpg', vis_lines)

    def _line_intersect(l1, l2):
        x1, y1, x2, y2 = l1[0], l1[1], l1[2], l1[3]
        x3, y3, x4, y4 = l2[0], l2[1], l2[2], l2[3]
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)

    tl = _line_intersect(top, left)
    tr = _line_intersect(top, right)
    br = _line_intersect(bottom, right)
    bl = _line_intersect(bottom, left)
    if any(p is None for p in (tl, tr, br, bl)):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    rect = cv2.minAreaRect(quad)
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return None
    quad_ar = max(rw, rh) / max(min(rw, rh), 1e-6)
    cells_w = float(cmax[0] - cmin[0])
    cells_h = float(cmax[1] - cmin[1])
    cell_ar = max(cells_w, cells_h) / max(min(cells_w, cells_h), 1e-6)
    if quad_ar > cell_ar * 1.5 or cell_ar > quad_ar * 1.5:
        return None

    quad[:, 0] += x0
    quad[:, 1] += y0
    return order_quad_corners(quad)


def _detect_reference_quad(gray: np.ndarray, bgr: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Findet die Karte ueber das Zellraster.

    Strategie (aus test_detect.py):
      1. Canny -> alle 4-Eck-Kandidaten (= Zellen).
      2. Filter: Groesse-Histogramm (Modus = Karten-Zellen).
      3. Density-Filter: nur Zellen mit Nachbarn behalten -> Karten-Cluster.
      4. minAreaRect ueber Zentroide + halbe Zellbreite Rand -> Initial-Quad.
      5. Hough-Verfeinerung: 4 dominante Aussenlinien um die Zellen.

    `bgr` wird nicht benoetigt (Detection laeuft auf gray); Parameter bleibt
    aus Kompatibilitaetsgruenden erhalten.
    """
    edges = _compute_edges(gray)
    cells = _detect_cell_rects(edges)

    if _DEBUG_DIR:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, cw, ch, _, _ in cells:
            cv2.rectangle(vis, (cx - cw // 2, cy - ch // 2),
                          (cx + cw // 2, cy + ch // 2), (0, 200, 255), 1)
        _debug_save('21_all_cells.jpg', vis)

    if len(cells) < 6:
        return None

    sized = _filter_cells_by_size(cells)
    if _DEBUG_DIR:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, cw, ch, _, _ in sized:
            cv2.rectangle(vis, (cx - cw // 2, cy - ch // 2),
                          (cx + cw // 2, cy + ch // 2), (0, 255, 255), 2)
        _debug_save('22_size_filtered.jpg', vis)

    if len(sized) < 6:
        return None

    cluster = _filter_dense_cells(sized, k_neighbors=4)
    if _DEBUG_DIR:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, cw, ch, _, _ in cluster:
            cv2.rectangle(vis, (cx - cw // 2, cy - ch // 2),
                          (cx + cw // 2, cy + ch // 2), (0, 200, 0), 2)
        _debug_save('23_dense.jpg', vis)
    if len(cluster) < 6:
        return None

    # Drop sparse boundary rows/columns — kills outliers like a stray
    # background rectangle that passed density. Strict subset of cluster.
    cluster, dropped = _drop_boundary_outliers(cluster)
    if _DEBUG_DIR:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, cw, ch, _, _ in cluster:
            cv2.rectangle(vis, (cx - cw // 2, cy - ch // 2),
                          (cx + cw // 2, cy + ch // 2), (0, 200, 0), 2)
        for cx, cy, cw, ch, _, _ in dropped:
            cv2.rectangle(vis, (cx - cw // 2, cy - ch // 2),
                          (cx + cw // 2, cy + ch // 2), (0, 0, 255), 2)
        _debug_save('23b_grid_filtered.jpg', vis)
    if len(cluster) < 6:
        return None

    pts = np.array([(c[0], c[1]) for c in cluster], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cluster], dtype=np.float32)
    cell_angles = np.array([c[5] for c in cluster], dtype=np.float32)
    half_cell = float(np.median(sizes)) * 0.5

    angle_deg = float(np.median(cell_angles))

    theta = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    centroid = pts.mean(axis=0)
    rotated = (pts - centroid) @ R.T
    rx0, ry0 = rotated.min(axis=0)
    rx1, ry1 = rotated.max(axis=0)
    rw = float(rx1 - rx0)
    rh = float(ry1 - ry0)
    if min(rw, rh) < 20:
        return None

    rw += 2 * half_cell
    rh += 2 * half_cell
    cx_r = (rx0 + rx1) * 0.5
    cy_r = (ry0 + ry1) * 0.5

    corners_r = np.array([
        [cx_r - rw / 2, cy_r - rh / 2],
        [cx_r + rw / 2, cy_r - rh / 2],
        [cx_r + rw / 2, cy_r + rh / 2],
        [cx_r - rw / 2, cy_r + rh / 2],
    ], dtype=np.float32)
    R_inv = R.T
    quad = corners_r @ R_inv.T + centroid

    h_img, w_img = gray.shape
    quad[:, 0] = np.clip(quad[:, 0], 0, w_img - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h_img - 1)
    cell_quad = order_quad_corners(quad)

    # Hough-Verfeinerung. Margin skaliert mit der Bildgroesse — ein 4K-Foto
    # braucht mehr Pixel Spielraum als ein 720p-Bild.
    src_short = min(gray.shape[:2])
    roi_margin = max(100, src_short // 8)
    refined = _refine_quad_via_hough(gray, edges, cluster,
                                     card_angle_deg=angle_deg,
                                     margin=roi_margin)
    if refined is not None:
        if _DEBUG_DIR:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [refined.astype(np.int32)], True, (0, 255, 0), 4)
            _debug_save('27_refined_quad.jpg', vis)
        return refined
    return cell_quad


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
    from java.util import Map as _JMap, List as _JList  # pyright: ignore[reportMissingImports]
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
