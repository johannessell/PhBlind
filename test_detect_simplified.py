"""
test_detect.py
==============
Lokaler Iterations-Sandbox fuer die Referenz-Karten-Erkennung.

Strategie (aus contour_detection03 abgeleitet):
  1. Canny + approxPolyDP -> alle 4-Eck-Kandidaten (= Zellen).
  2. Filter: Groesse, Aspect-Ratio, Konvexitaet.
  3. Dichtefilter: nur Zellen mit Nachbarn behalten -> Karten-Cluster.
  4. minAreaRect auf den dichten Zellzentren -> rotiertes Kartenquad.
  5. Erweitern um halbe Zellgroesse (Rand der Karte liegt jenseits der
     Zellen) und perspektivisch entzerren auf 720 px breit.

Aufruf:
    python test_detect.py <bildpfad>
        oder ohne Argument -> alle reference*/raw_*.jpg im aktuellen Ordner.

Debug-Bilder landen in ./test_detect_out/<basename>/.
"""

from __future__ import annotations

import os
import sys
import glob
from typing import List, Optional, Tuple
import math

import cv2
import numpy as np

CANONICAL_WIDTH = 720


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """TL -> TR -> BR -> BL via Summe/Differenz der Koordinaten."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.empty((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]   # TL
    out[2] = pts[np.argmax(s)]   # BR
    out[1] = pts[np.argmin(d)]   # TR
    out[3] = pts[np.argmax(d)]   # BL
    return out


def downscale_for_detection(bgr: np.ndarray, target_w: int = 720) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if w <= target_w * 1.25:
        return bgr.copy(), 1.0
    scale = target_w / w
    new = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return new, scale


# ---------------------------------------------------------------------------
# Zell-Detektion
# ---------------------------------------------------------------------------

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
    deren minAreaRect liegt mittig in der echten Zelle. Wir verwerfen jede
    Zelle, deren Zentrum innerhalb einer >=1.2x groesseren Zelle liegt.
    """
    if len(cells) < 2:
        return cells
    # nach Flaeche absteigend sortieren, um spaeter aussen->innen zu pruefen
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


def detect_cell_rects(gray: np.ndarray, debug_dir: Optional[str] = None) -> List[Tuple[int, int, int, int, float, float]]:
    """Finde alle quadratischen/rechteckigen kleinen Konturen.

    Rueckgabe: Liste von (cx, cy, w, h, area, angle_deg).
    angle_deg ist der Winkel der laengeren Kante zur x-Achse (-45..45).
    """
    gray = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 30, 90)
    # Schliesse kleine Luecken in Zellumrandungen (z.B. wenn Text die
    # Zellrahmenlinie unterbricht), damit findContours die Zelle als
    # geschlossenes Polygon liefert.
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8), iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, '20_edges.jpg'), edges)
        cv2.imwrite(os.path.join(debug_dir, '20b_edges_closed.jpg'), edges_closed)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    frame_area = float(h * w)
    min_cell = frame_area * 0.0001
    max_cell = frame_area * 0.02

    cells: List[Tuple[int, int, int, int, float, float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_cell or area > max_cell:
            continue
        # Convex Hull glaettet rauhe Konturkanten von Canny. minAreaRect
        # darauf liefert das umschliessende rotierte Rechteck. Statt einer
        # strengen "exakt 4 Ecken"-Pruefung filtern wir per
        # Rectangularity (= Konturflaeche / Rect-Flaeche) — Zellen liegen
        # bei ~0.9, Kreise bei ~0.78, irregulaere Formen darunter.
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
        # Winkel der laengeren Kante normalisiert auf [-45, 45].
        ang = rect[2]
        if rw < rh:
            ang += 90.0
        while ang > 45.0:
            ang -= 90.0
        while ang < -45.0:
            ang += 90.0
        cx, cy = rect[0]
        # Normalisiere w/h so dass w >= h (laengere Seite zuerst). Sonst
        # gibt minAreaRect bei "Innen-" und "Aussen"-Konturen derselben
        # Zelle Werte mit getauschtem (w, h) zurueck — die Visualisierung
        # zeigt dann einmal liegend, einmal stehend an gleicher Stelle.
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        cells.append((int(cx), int(cy), int(long_side), int(short_side),
                      float(area), float(ang)))

    cells = _dedupe_cells(cells)
    cells = _drop_nested(cells)
    return cells, edges


def filter_dense_cells(cells, k_neighbors: int = 6, radius_factor: float = 2.5):
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


def filter_by_size(cells, tolerance: float = 0.40, n_bins: int = 20):
    """Behalte nur Zellen, deren kurze Seite (Hoehe) zum Histogramm-Modus passt.

    Karten-Zellen sind alle gleich hoch; Streu-Rechtecke (Stiftehalter,
    Monitorrahmen, Tastenkappen) haben i.d.R. andere Groessen.
    Histogramm-Modus statt Median macht das robust gegen mehrere
    konkurrierende Gruppen — die Karte hat normalerweise die meisten
    gleichgrossen Zellen.
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


def refine_quad_via_hough(edges: np.ndarray, cells, card_angle_deg: float,
                          margin: int = 100,
                          debug_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """ROI um Zellen + HoughLines -> 4 dominante Linien -> Schnittpunkte = Eckpunkte.

    Cannys Kantenring um die Karte ist optisch geschlossen, aber numerisch
    luckenhaft -> findContours findet keine geschlossene Kontur. HoughLines
    mittelt ueber die Lueeken hinweg und liefert die 4 echten Geraden, deren
    Schnittpunkte die Kartenecken sind.
    """
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

    edges_crop = edges[y0:y1, x0:x1]

    if debug_dir:
        vis_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_roi, (x0, y0), (x1, y1), (255, 200, 0), 4)
        cv2.imwrite(os.path.join(debug_dir, '24_roi.jpg'), vis_roi)
        cv2.imwrite(os.path.join(debug_dir, '25_crop.jpg'), crop)
        cv2.imwrite(os.path.join(debug_dir, '26_crop_edges.jpg'), edges)

    cell_short = float(np.median([min(c[2], c[3]) for c in cells]))

    # HoughLinesP: Liniensegmente. minLineLength etwas kleiner als die kurze
    # Kartenkante, damit Aussenkanten auch in zwei kurze Stuecke zerfallen
    # nicht verloren gehen.
    threshold = max(int(cell_short * 1.0), 30)
    min_line_len = max(int(cell_short * 2.5), 50)
    lines_p = cv2.HoughLinesP(edges_crop, rho=1, theta=np.pi / 180,
                              threshold=threshold,
                              minLineLength=min_line_len,
                              maxLineGap=10)
    if lines_p is None or len(lines_p) < 4:
        return None

    # Inner-Box (rotiertes Rechteck der Zellen + halbe Zellbreite Rand) in
    # Crop-Koordinaten. Linien deren Mittelpunkt INNERHALB liegen, sind
    # Gitterlinien und keine Kartenaussenkanten -> verwerfen.
    card_angle_rad = np.deg2rad(card_angle_deg)
    cos_t, sin_t = float(np.cos(card_angle_rad)), float(np.sin(card_angle_rad))
    cell_pts_crop = np.array([(c[0] - x0, c[1] - y0) for c in cells], dtype=np.float32)
    cell_centroid = cell_pts_crop.mean(axis=0)
    # Forward rotation in card frame: R^T * (p - centroid). Card frame: x along
    # card width, y along card height.
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

    # Erwartete Bereiche fuer die 4 Kartenseiten (im Karten-Frame, y-Achse).
    margin_lo = 0.3 * cell_short
    margin_hi = 2.5 * cell_short

    h_target = card_angle_rad % np.pi   # Linien parallel zur Karten-x (Zell-Kante)
    v_target = (card_angle_rad + np.pi / 2) % np.pi  # Linien parallel zur Karten-y
    angle_tol = np.deg2rad(15)

    def _ang_dist(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    def _is_horizontal(seg_deg):
        # Segment-Winkel in [0,180) — horizontale Linie nahe 0 oder 180.
        return seg_deg <= 15.0 or seg_deg >= 165.0

    def _is_vertical(seg_deg):
        # Vertikale Linie: 75..105 (mit User-Vorgabe ~80..110, leicht asymmetrisch).
        return 75.0 <= seg_deg <= 105.0

    horizontal_top, horizontal_bottom = [], []  # parallel to card width
    vertical_left, vertical_right = [], []      # parallel to card height
    accepted = []  # for debug overlay

    for line in lines_p:
        x1, y1, x2, y2 = (float(v) for v in line[0])
        mx = (x1 + x2) * 0.5
        my = (y1 + y2) * 0.5
        if _is_inside_inner(mx, my):
            continue  # Gitterlinie

        # Liniensegment-Winkel (in [0, pi))
        seg_angle = float(math.atan2(y2 - y1, x2 - x1)) % np.pi
        seg_deg = np.degrees(seg_angle)
        # Mittelpunkt im Karten-Frame -> direkte y/x-Position
        rx, ry = _to_card_frame(mx, my)

        # Top/Bottom: muss horizontal sein (Bildframe) UND parallel zur Karten-x.
        if _is_horizontal(seg_deg) and _ang_dist(seg_angle, h_target) < angle_tol:
            if cmin[1] - margin_hi <= ry <= cmin[1] - margin_lo:
                horizontal_top.append((x1, y1, x2, y2, seg_angle, ry))
                accepted.append(('top', x1, y1, x2, y2))
            elif cmax[1] + margin_lo <= ry <= cmax[1] + margin_hi:
                horizontal_bottom.append((x1, y1, x2, y2, seg_angle, ry))
                accepted.append(('bottom', x1, y1, x2, y2))
        # Left/Right: muss vertikal sein (Bildframe ~80-110°) UND parallel zur Karten-y.
        elif _is_vertical(seg_deg) and _ang_dist(seg_angle, v_target) < angle_tol:
            if cmin[0] - margin_hi <= rx <= cmin[0] - margin_lo:
                vertical_left.append((x1, y1, x2, y2, seg_angle, rx))
                accepted.append(('left', x1, y1, x2, y2))
            elif cmax[0] + margin_lo <= rx <= cmax[0] + margin_hi:
                vertical_right.append((x1, y1, x2, y2, seg_angle, rx))
                accepted.append(('right', x1, y1, x2, y2))

    if not (horizontal_top and horizontal_bottom and vertical_left and vertical_right):
        return None

    # Aus jeder Seite die aeusserste Linie waehlen.
    top = min(horizontal_top, key=lambda l: l[5])
    bottom = max(horizontal_bottom, key=lambda l: l[5])
    left = min(vertical_left, key=lambda l: l[5])
    right = max(vertical_right, key=lambda l: l[5])

    if debug_dir:
        vis_lines = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for line in lines_p:
            x1, y1, x2, y2 = (int(v) for v in line[0])
            cv2.line(vis_lines, (x1, y1), (x2, y2), (60, 60, 60), 1)
        side_color = {'top': (0, 255, 0), 'bottom': (0, 255, 0),
                      'left': (0, 200, 255), 'right': (0, 200, 255)}
        for side, x1, y1, x2, y2 in accepted:
            cv2.line(vis_lines, (int(x1), int(y1)), (int(x2), int(y2)),
                     side_color[side], 2)
        for chosen, color in [(top, (0, 0, 255)), (bottom, (0, 0, 255)),
                              (left, (255, 0, 255)), (right, (255, 0, 255))]:
            x1, y1, x2, y2 = chosen[0], chosen[1], chosen[2], chosen[3]
            cv2.line(vis_lines, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.imwrite(os.path.join(debug_dir, '26b_hough.jpg'), vis_lines)

    def _line_intersect(l1, l2):
        # l = (x1, y1, x2, y2, ...). Parametrisch: P = A + t*(B-A); Q = C + u*(D-C)
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

    # Sanity check: Quad-Aspekt sollte zum Zell-Bounding-Aspekt passen,
    # sonst ist eine Linie kaputt -> fallback auf cell_quad.
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


def detect_card_quad(gray: np.ndarray, debug_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Hauptfunktion: gray -> 4 Eckpunkte (TL,TR,BR,BL)."""
    cells, edges = detect_cell_rects(gray, debug_dir)
    if debug_dir:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, w, h, _, _ in cells:
            cv2.rectangle(vis, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                          (0, 200, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, '21_all_cells.jpg'), vis)

    if len(cells) < 6:
        return None

    # Stage 22: Groessenfilter zuerst — verwirft Streu-Rechtecke abweichender
    # Groesse, BEVOR der Density-Filter den Median-Radius berechnet.
    sized = filter_by_size(cells)
    if debug_dir:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, w, h, _, _ in sized:
            cv2.rectangle(vis, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                          (0, 255, 255), 2)
        cv2.imwrite(os.path.join(debug_dir, '22_size_filtered.jpg'), vis)

    if len(sized) < 6:
        return None

    # Stage 23: Density-Filter auf bereits groessen-gefilterten Zellen.
    # Radius ist nun nicht mehr von Junk-Groessen verfaelscht.
    cluster = filter_dense_cells(sized, k_neighbors=4)
    if debug_dir:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for cx, cy, w, h, _, _ in cluster:
            cv2.rectangle(vis, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                          (0, 200, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, '23_dense.jpg'), vis)
    if len(cluster) < 6:
        return None

    pts = np.array([(c[0], c[1]) for c in cluster], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cluster], dtype=np.float32)
    cell_angles = np.array([c[5] for c in cluster], dtype=np.float32)
    half_cell = float(np.median(sizes)) * 0.5

    # Median-Winkel der einzelnen Zellen-Rects ist die robusteste
    # Schaetzung der Kartenrotation, auch wenn Zellen unvollstaendig
    # erkannt werden. minAreaRect der Zentroide kippt bei luckenhaften
    # Daten, der Median-Winkel der Zellen nicht.
    angle_deg = float(np.median(cell_angles))

    # Drehe die Punkte in Karten-Koordinaten zurueck, baue dort die
    # achsenausgerichtete BoundingBox, dann zurueck ins Bildkoordinatensystem.
    theta = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    centroid = pts.mean(axis=0)
    rotated = (pts - centroid) @ R.T
    x0, y0 = rotated.min(axis=0)
    x1, y1 = rotated.max(axis=0)
    rw = float(x1 - x0)
    rh = float(y1 - y0)
    if min(rw, rh) < 20:
        return None

    # Erweitern um halbe Zelle (Zellzentren liegen nicht am Kartenrand).
    rw += 2 * half_cell
    rh += 2 * half_cell
    cx_r = (x0 + x1) * 0.5
    cy_r = (y0 + y1) * 0.5

    # 4 Ecken im rotierten Frame, dann zurueckdrehen
    corners_r = np.array([
        [cx_r - rw / 2, cy_r - rh / 2],
        [cx_r + rw / 2, cy_r - rh / 2],
        [cx_r + rw / 2, cy_r + rh / 2],
        [cx_r - rw / 2, cy_r + rh / 2],
    ], dtype=np.float32)
    R_inv = R.T
    quad = corners_r @ R_inv.T + centroid

    h, w = gray.shape
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    cell_quad = order_quad_corners(quad)

    # Stage 24-27: Verfeinerung — innerhalb einer ROI um die Zellen
    # die 4 Kartengeraden per HoughLines bestimmen, mit Linien-Richtungen
    # gefiltert auf die Zell-Orientierung (Karten-Kanten parallel zu Zellen).
    refined = refine_quad_via_hough(edges, cluster, card_angle_deg=angle_deg,
                                    margin=100, debug_dir=debug_dir)
    if refined is not None:
        if debug_dir:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [refined.astype(np.int32)], True, (0, 255, 0), 4)
            cv2.imwrite(os.path.join(debug_dir, '27_refined_quad.jpg'), vis)
        return refined
    return cell_quad


# ---------------------------------------------------------------------------
# Perspektivkorrektur
# ---------------------------------------------------------------------------

def warp_to_canonical(bgr: np.ndarray, quad: np.ndarray,
                      canonical_width: int = CANONICAL_WIDTH) -> Tuple[np.ndarray, Tuple[int, int]]:
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
    warped = cv2.warpPerspective(bgr, M, (cw, ch),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, (cw, ch)


# ---------------------------------------------------------------------------
# Pipeline pro Bild
# ---------------------------------------------------------------------------

def process(image_path: str, out_root: str = 'test_detect_out') -> bool:
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(out_root, base)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f'[{base}] read failed')
        return False

    src_h, src_w = bgr.shape[:2]
    cv2.imwrite(os.path.join(out_dir, '00_input.jpg'), bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(out_dir, '01_gray.jpg'), gray)

    quad = detect_card_quad(gray, debug_dir=out_dir)
    if quad is None:
        print(f'[{base}] FAILED: no quad')
        return False

    vis = bgr.copy()
    cv2.polylines(vis, [quad.astype(np.int32)], True, (0, 255, 0), 6)
    for i, p in enumerate(quad.astype(int)):
        cv2.putText(vis, ['TL', 'TR', 'BR', 'BL'][i], tuple(p),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.imwrite(os.path.join(out_dir, '30_quad.jpg'), vis)

    # Reference: warp at source resolution (no downscaling).
    warped, (cw, ch) = warp_to_canonical(bgr, quad, canonical_width=src_w)
    cv2.imwrite(os.path.join(out_dir, '40_warped.jpg'), warped)
    print(f'[{base}] OK  src={src_w}x{src_h}  quad-area={int(cv2.contourArea(quad))}  warp={cw}x{ch}')
    return True


def main(argv):
    if len(argv) > 1:
        targets = argv[1:]
    else:
        os.makedirs('test_input', exist_ok=True)
        # Volle Aufloesung — Phone-Captures vom Android-Debug-Export.
        # Drop captures aus Download/PoolWaterTester/ in test_input/.
        all_files = sorted(glob.glob('test_input/*.jpg') + glob.glob('test_input/*.png'))
        # Phone debug exports use 00_input*.jpg for the raw capture; the other
        # numbered files (01_gray, 02_quad, 10_mask_*, 12_contours_*, 03_warped,
        # 20_*, 21_*, 22_*, 23_*, 30_*, 40_*) are derived debug outputs — skip those.
        import re
        skip_re = re.compile(r'^(0[1-9]|[12]\d|30|40)[a-z]?_')
        targets = [p for p in all_files if not skip_re.match(os.path.basename(p))]
        if not targets:
            print('No capture images in test_input/. Pull captures from the phone '
                  '(Download/PoolWaterTester/) and drop them there.')
            return

    out_root = 'test_detect_out'
    os.makedirs(out_root, exist_ok=True)

    results = []
    for path in targets:
        # Pro Input ein dediziertes Subfolder; existierende Inhalte werden
        # ueberschrieben (frische Debug-Bilder pro Run, alte bleiben nicht stehen).
        base = os.path.splitext(os.path.basename(path))[0]
        sub = os.path.join(out_root, base)
        if os.path.isdir(sub):
            import shutil
            shutil.rmtree(sub)
        ok = process(path, out_root=out_root)
        results.append((os.path.basename(path), ok))

    # Zusammenfassung
    n_ok = sum(1 for _, ok in results if ok)
    print(f'\n=== Summary: {n_ok}/{len(results)} OK ===')
    for name, ok in results:
        flag = 'OK ' if ok else 'XX '
        print(f'  {flag} {name}')


if __name__ == '__main__':
    main(sys.argv)
