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

def detect_card_by_cell_cluster(
    gray: np.ndarray,
    aspect_ratio: float,
    ratio_tol: float = 0.35,
    min_cells: int   = 6,
) -> Optional[np.ndarray]:
    """
    Findet die Karte über ihre Gitterzellen (statt der Außenkontur).

    Die Außenkontur verschmilzt häufig mit dem Hintergrund (Tischkante etc.).
    Die internen Zell-Rechtecke bleiben aber sichtbar und dicht gruppiert.

    Ablauf:
      1. Canny (ohne Dilatation, um Verschmelzung zu vermeiden)
      2. RETR_LIST: alle Konturen, nicht nur äußerste
      3. Kleine rechteckige Konturen = Zellkandidaten
      4. Dichtefilter: nur Zellen mit ≥3 Nachbarn innerhalb 80px behalten
         (Karten­raster = dicht; Hintergrundrechtecke = verteilt)
      5. minAreaRect auf den dichten Zellen → rotiertes Viereck
      6. Erweiterung um ~halbe Zellgröße (Zellzentren liegen nicht am Kartenrand)

    Returns:
        quad (4,2 float32) TL→TR→BR→BL, oder None
    """
    h, w       = gray.shape[:2]
    frame_area = h * w

    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges    = cv2.Canny(enhanced, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_cell = frame_area * 0.0001
    max_cell = frame_area * 0.02

    centers:    List[Tuple[float, float]] = []
    cell_sizes: List[float]               = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_cell or area > max_cell:
            continue

        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1:
            continue
        approx = cv2.approxPolyDP(hull, 0.08 * peri, True)
        if len(approx) != 4:
            continue

        rect_box   = cv2.minAreaRect(approx)
        rw_c, rh_c = rect_box[1]
        if min(rw_c, rh_c) < 4:
            continue
        cell_ar = max(rw_c, rh_c) / (min(rw_c, rh_c) + 1e-6)
        if cell_ar > 4.5:
            continue

        cx, cy = rect_box[0]
        centers.append((cx, cy))
        cell_sizes.append(max(rw_c, rh_c))

    if len(centers) < min_cells:
        return None

    pts = np.array(centers, dtype=np.float32)

    # Dichte Zellen: ≥3 Nachbarn in 80px (ca. 2-4 Zellbreiten bei typischer Entfernung)
    radius = 80.0
    keep   = []
    for i, p in enumerate(pts):
        dists = np.linalg.norm(pts - p, axis=1)
        if int(np.sum(dists < radius)) >= 4:            # self + 3 Nachbarn
            keep.append(i)
    if len(keep) < min_cells:
        return None
    dense       = pts[keep]
    dense_sizes = np.array([cell_sizes[i] for i in keep])

    # Rotiertes Rechteck (ermöglicht korrekte Entzerrung bei gekippter Karte)
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(dense)
    if min(rw, rh) < 20:
        return None

    ar    = max(rw, rh) / (min(rw, rh) + 1e-6)
    ar_lo = aspect_ratio * (1.0 - ratio_tol)
    ar_hi = aspect_ratio * (1.0 + ratio_tol)
    if not (ar_lo <= ar <= ar_hi):
        return None

    # Zellzentren liegen nicht am Kartenrand — um halbe Zellgröße nach außen erweitern
    half_cell = float(np.median(dense_sizes)) * 0.5
    expanded  = ((cx, cy), (rw + 2 * half_cell, rh + 2 * half_cell), angle)

    quad = cv2.boxPoints(expanded).astype(np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)

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
