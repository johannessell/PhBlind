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
    edges = cv2.Canny(enhanced, 30, 120)
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
    return cells


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


def refine_quad_via_hough(gray: np.ndarray, cells, card_angle_deg: float,
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

    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    if debug_dir:
        vis_roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_roi, (x0, y0), (x1, y1), (255, 200, 0), 4)
        cv2.imwrite(os.path.join(debug_dir, '24_roi.jpg'), vis_roi)
        cv2.imwrite(os.path.join(debug_dir, '25_crop.jpg'), crop)
        cv2.imwrite(os.path.join(debug_dir, '26_crop_edges.jpg'), edges)

    cell_short = float(np.median([min(c[2], c[3]) for c in cells]))

    # Niedrigere Schwelle, damit auch schwache Karten-Aussenkanten als Linien
    # auftauchen — die Auswahl filtert dann ueber Abstand zur Zellgruppe.
    threshold = max(int(cell_short * 1.5), 40)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=threshold)
    if lines is None or len(lines) < 4:
        return None

    cell_cx_crop = float(np.mean([c[0] for c in cells])) - x0
    cell_cy_crop = float(np.mean([c[1] for c in cells])) - y0

    card_angle_rad = np.deg2rad(card_angle_deg)
    h_target = (np.pi / 2 + card_angle_rad) % np.pi
    v_target = card_angle_rad % np.pi
    angle_tol = np.deg2rad(15)

    def _ang_dist(a, b):
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    def _signed(rho, theta):
        return rho - (cell_cx_crop * np.cos(theta) + cell_cy_crop * np.sin(theta))

    # Cell-Projektionen: extreme Zellgrenzen in beiden Karten-Richtungen,
    # sodass wir Linien filtern koennen, die nur knapp ausserhalb liegen.
    def _cell_projections(theta):
        # signed perpendicular distance vom Zell-Cluster-Mittel zu jeder Zelle
        ds = []
        for c in cells:
            cx = c[0] - x0
            cy = c[1] - y0
            d = (cx - cell_cx_crop) * np.cos(theta) + (cy - cell_cy_crop) * np.sin(theta)
            ds.append(d)
        return min(ds), max(ds)

    h_min, h_max = _cell_projections(h_target)
    v_min, v_max = _cell_projections(v_target)

    # Erwarteter Abstand der Kartenkante: 0.3..2.5 Zell-Hoehen jenseits
    # der aeusseren Zelle. Inneren Gitter-Linien liegen INNERHALB der
    # Zellgrenzen und werden so verworfen.
    margin_lo = 0.3 * cell_short
    margin_hi = 2.5 * cell_short

    horizontal_top = []     # signed dist <= h_min - margin_lo
    horizontal_bottom = []  # signed dist >= h_max + margin_lo
    vertical_left = []
    vertical_right = []

    for line in lines:
        rho, theta = line[0]
        sd = _signed(float(rho), float(theta))
        if _ang_dist(theta, h_target) < angle_tol:
            if h_min - margin_hi <= sd <= h_min - margin_lo:
                horizontal_top.append((float(rho), float(theta), sd))
            elif h_max + margin_lo <= sd <= h_max + margin_hi:
                horizontal_bottom.append((float(rho), float(theta), sd))
        elif _ang_dist(theta, v_target) < angle_tol:
            if v_min - margin_hi <= sd <= v_min - margin_lo:
                vertical_left.append((float(rho), float(theta), sd))
            elif v_max + margin_lo <= sd <= v_max + margin_hi:
                vertical_right.append((float(rho), float(theta), sd))

    if not (horizontal_top and horizontal_bottom and vertical_left and vertical_right):
        return None

    # Aus jeder Seite die aeusserste Linie waehlen (= weiteste vom Zentrum).
    top = min(horizontal_top, key=lambda l: l[2])[:2]
    bottom = max(horizontal_bottom, key=lambda l: l[2])[:2]
    left = min(vertical_left, key=lambda l: l[2])[:2]
    right = max(vertical_right, key=lambda l: l[2])[:2]

    if debug_dir:
        vis_lines = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for line in lines:
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x_, y_ = a * rho, b * rho
            p1 = (int(x_ + 2000 * (-b)), int(y_ + 2000 * a))
            p2 = (int(x_ - 2000 * (-b)), int(y_ - 2000 * a))
            cv2.line(vis_lines, p1, p2, (80, 80, 80), 1)
        for (rho, theta), color in [(top, (0, 255, 0)), (bottom, (0, 255, 0)),
                                    (left, (0, 200, 255)), (right, (0, 200, 255))]:
            a, b = np.cos(theta), np.sin(theta)
            x_, y_ = a * rho, b * rho
            p1 = (int(x_ + 2000 * (-b)), int(y_ + 2000 * a))
            p2 = (int(x_ - 2000 * (-b)), int(y_ - 2000 * a))
            cv2.line(vis_lines, p1, p2, color, 3)
        cv2.imwrite(os.path.join(debug_dir, '26b_hough.jpg'), vis_lines)

    def _intersect(l1, l2):
        rho1, theta1 = l1
        rho2, theta2 = l2
        A = np.array([[np.cos(theta1), np.sin(theta1)],
                      [np.cos(theta2), np.sin(theta2)]], dtype=np.float64)
        b = np.array([rho1, rho2], dtype=np.float64)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-6:
            return None
        return np.linalg.solve(A, b)

    tl = _intersect(top, left)
    tr = _intersect(top, right)
    br = _intersect(bottom, right)
    bl = _intersect(bottom, left)
    if any(p is None for p in (tl, tr, br, bl)):
        return None

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Sanity check: Quad muss zur Karte passen. Aspekt-Ratio sollte zur
    # Zell-Bounding-Box passen, sonst ist eine Linie kaputt.
    rect = cv2.minAreaRect(quad)
    rw, rh = rect[1]
    if min(rw, rh) < 1:
        return None
    quad_ar = max(rw, rh) / max(min(rw, rh), 1e-6)
    cell_w_extent = float(v_max - v_min)
    cell_h_extent = float(h_max - h_min)
    cell_ar = max(cell_w_extent, cell_h_extent) / max(min(cell_w_extent, cell_h_extent), 1e-6)
    # Quad-Aspekt darf hoechstens 1.5x von Zell-Bounding-Aspekt abweichen.
    if quad_ar > cell_ar * 1.5 or cell_ar > quad_ar * 1.5:
        return None

    quad[:, 0] += x0
    quad[:, 1] += y0
    return order_quad_corners(quad)


def detect_card_quad(gray: np.ndarray, debug_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """Hauptfunktion: gray -> 4 Eckpunkte (TL,TR,BR,BL)."""
    cells = detect_cell_rects(gray, debug_dir)
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
    refined = refine_quad_via_hough(gray, cluster, card_angle_deg=angle_deg,
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
                                 flags=cv2.INTER_LINEAR,
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
