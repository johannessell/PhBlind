"""
tracker.py
==========
Messindikator-Tracker: Findet die Messkarte im Kamerabild über ihre Gitterzellen.

Erkennungsstrategie:
  Cell-Cluster: Canny → alle Konturen → kleine rechteckige (= Zellen) sammeln
                → Dichtefilter (Zellen liegen nahe beieinander)
                → minAreaRect der dichten Zellzentren als rotiertes Quad
                → _verify_quad bestätigt Gitterstruktur (HoughLinesP)

Liefert 4 geordnete Eckpunkte (TL, TR, BR, BL) im Frame-Koordinatensystem.
warp() wendet getPerspectiveTransform an und liefert das Kartenbild in exakt
der Template-Größe (inklusive Rotations- und Perspektivkorrektur).

Verwendung:
    tracker   = IndicatorTracker(template_gray)
    stability = QuadStabilityChecker(required_frames=5)

    quad, method = tracker.find(scene_gray)
    if stability.update(quad):
        warped = tracker.warp(frame_bgr, stability.mean_quad())
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════

def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sortiert 4 Eckpunkte konsistent: TL → TR → BR → BL.
    TL = kleinste Summe x+y, BR = größte, TR = kleinste Differenz x-y, BL = größte.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    out = np.empty((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]   # TL
    out[2] = pts[np.argmax(s)]   # BR
    out[1] = pts[np.argmin(d)]   # TR
    out[3] = pts[np.argmax(d)]   # BL
    return out


# ══════════════════════════════════════════════════════════
# Zell-Cluster-Detektion
# ══════════════════════════════════════════════════════════

def _dedupe_cells(cells, center_tol_factor: float = 0.3, size_tol: float = 0.25):
    """findContours returns inner+outer edges of each cell frame as separate
    contours; drop the smaller of any near-duplicate pair (keep outer)."""
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
    """Drop cells whose center sits inside a >=1.2x larger cell (= text/digit
    contour inside a real cell)."""
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


def _detect_cell_rects(edges: np.ndarray):
    """Find rectangle-ish small contours; return (cx, cy, long, short, area, ang_deg)."""
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                    np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = edges.shape
    frame_area = float(h * w)
    min_cell = frame_area * 0.0001
    max_cell = frame_area * 0.02

    cells = []
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


def _filter_cells_by_size(cells, tolerance: float = 0.40, n_bins: int = 20):
    """Keep cells whose short side is near the histogram mode. Card cells are
    uniform; stray rectangles (background clutter) typically aren't."""
    if len(cells) < 4:
        return cells
    heights = np.array([c[3] for c in cells], dtype=np.float32)
    hist, edges = np.histogram(heights, bins=n_bins)
    mode_idx = int(np.argmax(hist))
    mode_h = float(0.5 * (edges[mode_idx] + edges[mode_idx + 1]))
    lo = mode_h * (1.0 - tolerance)
    hi = mode_h * (1.0 + tolerance)
    return [c for c in cells if lo <= c[3] <= hi]


def _filter_dense_cells(cells, k_neighbors: int = 4, radius_factor: float = 2.5):
    """Keep cells with >= k neighbors within radius scaled by median cell size.
    Scale-invariant: the radius adapts to the resolution."""
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


def detect_card_by_cell_cluster(
    gray: np.ndarray,
    aspect_ratio: float,
    ratio_tol: float = 0.35,
    min_cells: int   = 6,
) -> Optional[np.ndarray]:
    """
    Scale-invariant cell-cluster detection.

    Pipeline:
      1. Canny on bilateral+CLAHE-enhanced gray.
      2. Find rectangle-ish contours (rectangularity >= 0.6 instead of strict
         4-vertex approxPolyDP — survives noisy edges).
      3. Dedup near-coincident contours, drop nested (text inside a cell).
      4. Size-histogram filter: keep cells whose short side matches the mode.
         Card cells are uniform so they dominate the histogram; stray
         rectangles end up in other bins and get dropped before they bias
         the density radius.
      5. Density filter: radius proportional to median cell size (NOT a fixed
         80 px), so it works at any resolution.
      6. Card rotation = median of per-cell angles (robust to outlier corners).
      7. Axis-aligned bbox in card frame, expanded by half a cell, rotated back
         into image coordinates.

    Best run at the camera's native resolution: the size-histogram filter
    needs enough pixels to separate card-cell sizes from background noise.
    Hough refinement (which the reference build does) is intentionally
    skipped here — the live overlay just needs the cell-cluster quad; the
    stabilized capture goes through reference_builder for the precise warp.
    """
    h_img, w_img = gray.shape[:2]

    filtered = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    edges = cv2.Canny(enhanced, 30, 90)

    cells = _detect_cell_rects(edges)
    if len(cells) < min_cells:
        return None

    sized = _filter_cells_by_size(cells)
    if len(sized) < min_cells:
        return None

    cluster = _filter_dense_cells(sized, k_neighbors=4)
    if len(cluster) < min_cells:
        return None

    pts = np.array([(c[0], c[1]) for c in cluster], dtype=np.float32)
    sizes = np.array([max(c[2], c[3]) for c in cluster], dtype=np.float32)
    cell_angles = np.array([c[5] for c in cluster], dtype=np.float32)
    half_cell = float(np.median(sizes)) * 0.5

    # Median angle of individual cell rects = card rotation. minAreaRect over
    # the cluster centroids tilts when the dense set is asymmetric; the
    # median per-cell angle does not.
    angle_deg = float(np.median(cell_angles))

    theta = np.deg2rad(-angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    centroid = pts.mean(axis=0)
    rotated = (pts - centroid) @ R.T
    rx0, ry0 = rotated.min(axis=0)
    rx1, ry1 = rotated.max(axis=0)
    rw = float(rx1 - rx0) + 2 * half_cell
    rh = float(ry1 - ry0) + 2 * half_cell
    if min(rw, rh) < 20:
        return None

    ar = max(rw, rh) / (min(rw, rh) + 1e-6)
    ar_lo = aspect_ratio * (1.0 - ratio_tol)
    ar_hi = aspect_ratio * (1.0 + ratio_tol)
    if not (ar_lo <= ar <= ar_hi):
        return None

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

    quad[:, 0] = np.clip(quad[:, 0], 0, w_img - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h_img - 1)
    return order_quad_corners(quad)


# ══════════════════════════════════════════════════════════
# Stabilitätsprüfung
# ══════════════════════════════════════════════════════════

class QuadStabilityChecker:
    """
    Bewertet ob das erkannte Viereck über N aufeinanderfolgende Frames stabil ist.
    Stabilität = max. Eckpunktdrift über die letzten N Frames < max_drift Pixel.
    """

    def __init__(self, required_frames: int = 5, max_drift: float = 15.0):
        self.required  = required_frames
        self.max_drift = max_drift
        self._history: list = []

    def update(self, quad: Optional[np.ndarray]) -> bool:
        if quad is None:
            self._history.clear()
            return False
        self._history.append(quad.copy())
        if len(self._history) < self.required:
            return False
        self._history = self._history[-self.required:]
        stacked = np.stack(self._history)                      # (N, 4, 2)
        drift   = float(np.max(stacked.max(axis=0) - stacked.min(axis=0)))
        return drift < self.max_drift

    def reset(self):
        self._history.clear()

    def progress(self) -> int:
        return len(self._history)

    def mean_quad(self) -> Optional[np.ndarray]:
        if not self._history:
            return None
        return np.mean(np.stack(self._history), axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════
# Haupt-Tracker
# ══════════════════════════════════════════════════════════

class IndicatorTracker:
    """
    Findet die Messkarte über Gitterzell-Clustering.

    find() gibt (quad, method) zurück:
        quad   – (4,2) float32 TL→TR→BR→BL im Frame, oder None
        method – 'cell' | 'none'
    """

    def __init__(self, template_gray: np.ndarray):
        self.template      = template_gray
        h, w               = template_gray.shape[:2]
        self.template_size = (w, h)
        self.aspect_ratio  = max(w, h) / min(w, h)

    # ──────────────────────────────────────────────────────

    def find(self, scene_gray: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Sucht den Indikator im Graustufen-Frame.
        Fallback-Pfade (Rect-Kontur, ORB/SIFT) sind deaktiviert — sie waren
        langsam und lieferten falsche Overlays.
        """
        quad = detect_card_by_cell_cluster(scene_gray, self.aspect_ratio)
        if quad is not None and self._verify_quad(scene_gray, quad):
            return quad, 'cell'
        return None, 'none'

    # ──────────────────────────────────────────────────────

    def _verify_quad(self, scene_gray: np.ndarray, quad: np.ndarray) -> bool:
        """
        Prüft ob die erkannte Region ein Gitter enthält (Falsch-Positiv-Schutz).
        Warpt auf 128x90 und zählt H/V-Linien mit HoughLinesP.
        """
        vw, vh      = 128, 90
        dst_corners = np.float32([[0, 0], [vw, 0], [vw, vh], [0, vh]])
        M      = cv2.getPerspectiveTransform(quad, dst_corners)
        warped = cv2.warpPerspective(scene_gray, M, (vw, vh))

        edges = cv2.Canny(warped, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=15, minLineLength=12, maxLineGap=4)
        if lines is None:
            return False

        h_lines = v_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if angle < 20 or angle > 160:
                h_lines += 1
            elif 70 < angle < 110:
                v_lines += 1

        return h_lines >= 20 and v_lines >= 4

    # ──────────────────────────────────────────────────────

    def warp(self, frame_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """
        Perspektivkorrektur: Kartenbereich → exakte Template-Größe (BGR).

        getPerspectiveTransform auf 4 korrespondierende Eckpunkte liefert
        eine 3x3-Homographie, die Rotation + perspektivische Verzerrung
        vollständig entzerrt. Das Ergebnis ist achsenausgerichtet in
        Template-Größe — direkt geeignet für zellbasierte Farbmessung.
        """
        w_t, h_t    = self.template_size
        dst_corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]])
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_corners)
        return cv2.warpPerspective(
            frame_bgr, M, (w_t, h_t),
            flags       = cv2.INTER_LINEAR,
            borderMode  = cv2.BORDER_REPLICATE,
        )

    # ──────────────────────────────────────────────────────

    def draw_quad(
        self,
        frame: np.ndarray,
        quad:  np.ndarray,
        color: tuple = (0, 220, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Zeichnet das erkannte Viereck mit TL/TR/BR/BL-Markierungen."""
        vis    = frame.copy()
        pts    = quad.astype(np.int32).reshape((-1, 1, 2))
        labels = ['TL', 'TR', 'BR', 'BL']
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
        for pt, label in zip(quad.astype(int), labels):
            cv2.circle(vis, tuple(pt), 5, color, -1)
            cv2.putText(vis, label, (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return vis
