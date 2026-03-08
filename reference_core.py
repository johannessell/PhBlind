"""
reference_core.py
=================
Kermodul für die Erkennung und Kalibrierung von Schwimmbad-Messindikatoren.

Verantwortlichkeiten:
  - Bild laden & vorverarbeiten
  - Grid erkennen (robust, layout-unabhängig)
  - Farbspalten automatisch vorschlagen (via Sättigungs-Analyse)
  - HSV-Median pro Zelle extrahieren
  - Referenz-Datenstruktur aufbauen & als JSON speichern/laden
  - Messung: Farbe eines neuen Bildes gegen Referenz vergleichen

Keine UI-Logik hier – nur pure Datenverarbeitung.
"""

import cv2
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path


# ══════════════════════════════════════════════════════════
# Datenstrukturen
# ══════════════════════════════════════════════════════════

@dataclass
class GridCell:
    """Eine einzelne Zelle im erkannten Grid."""
    col_idx: int
    row_idx: int
    x: int
    y: int
    w: int
    h: int
    hsv_median: list[float]        # [H, S, V]  OpenCV-Skala: H=0-180, S=0-255, V=0-255
    is_color_cell: bool = False    # True = Farbfeld, False = Beschriftung/leer
    parameter: Optional[str] = None   # z.B. "pH", "Chlor_frei", "PHMB"
    value: Optional[float] = None     # z.B. 7.2


@dataclass
class GridLayout:
    """Beschreibt das erkannte Grid."""
    n_cols: int
    n_rows: int
    cells: list[GridCell] = field(default_factory=list)

    def get_cell(self, col: int, row: int) -> Optional[GridCell]:
        for c in self.cells:
            if c.col_idx == col and c.row_idx == row:
                return c
        return None

    def color_columns(self) -> list[int]:
        """Gibt Indizes aller Spalten zurück, die als Farbspalten markiert sind."""
        return sorted({c.col_idx for c in self.cells if c.is_color_cell})

    def parameters(self) -> list[str]:
        """Gibt alle eindeutigen Parameter zurück."""
        return sorted({c.parameter for c in self.cells
                       if c.parameter and c.parameter != "leer"})


@dataclass
class ReferenceDB:
    """Vollständige Referenz-Datenbank für einen Indikator-Typ."""
    name: str                          # z.B. "Lovibond_3in1"
    image_file: str
    layout: Optional[GridLayout] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image_file": self.image_file,
            "n_cols": self.layout.n_cols,
            "n_rows": self.layout.n_rows,
            "cells": [asdict(c) for c in self.layout.cells]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReferenceDB":
        cells = [GridCell(**c) for c in data["cells"]]
        layout = GridLayout(
            n_cols=data["n_cols"],
            n_rows=data["n_rows"],
            cells=cells
        )
        return cls(name=data["name"], image_file=data["image_file"], layout=layout)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ReferenceDB":
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))


# ══════════════════════════════════════════════════════════
# Bildvorverarbeitung
# ══════════════════════════════════════════════════════════

def preprocess(image: np.ndarray) -> np.ndarray:
    """Graustufen + Rauschreduzierung + CLAHE."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 200, 200)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")
    return img


# ══════════════════════════════════════════════════════════
# Kontur- & Rechteck-Erkennung
# ══════════════════════════════════════════════════════════

def find_rects(gray_enhanced: np.ndarray,
               min_area: int = 500,
               canny_low: int = 30,
               canny_high: int = 120) -> list[tuple]:
    """
    Erkennt alle viereckigen Konturen.
    Gibt Liste von (x, y, w, h) zurück.
    """
    canny = cv2.Canny(gray_enhanced, canny_low, canny_high)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if 0.1 < w / float(h) < 10:
                rects.append((x, y, w, h))
    return rects


def crop_to_rects(image: np.ndarray,
                  rects: list[tuple],
                  margin: int = 50) -> tuple[np.ndarray, tuple]:
    """
    Schneidet das Bild auf den Bereich aller erkannten Rechtecke zu.
    Gibt (cropped_image, (offset_x, offset_y)) zurück.
    """
    if not rects:
        raise ValueError("Keine Rechtecke für Crop vorhanden.")
    x_min = max(min(r[0] for r in rects) - margin, 0)
    y_min = max(min(r[1] for r in rects) - margin, 0)
    x_max = min(max(r[0] + r[2] for r in rects) + margin, image.shape[1])
    y_max = min(max(r[1] + r[3] for r in rects) + margin, image.shape[0])
    return image[y_min:y_max, x_min:x_max].copy(), (x_min, y_min)


# ══════════════════════════════════════════════════════════
# Grid-Aufbau
# ══════════════════════════════════════════════════════════

def _group(rects: list[tuple], axis: int, tol: int) -> list[list[tuple]]:
    """Gruppiert Rechtecke nach x- oder y-Position mit Toleranz."""
    rects_sorted = sorted(rects, key=lambda r: r[axis])
    groups: list[list[tuple]] = []
    for r in rects_sorted:
        placed = False
        for g in groups:
            rep = g[0][axis]
            if abs(rep - r[axis]) < tol:
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])
    return groups


def build_grid(rects: list[tuple], tol: int = 20) -> GridLayout:
    """
    Baut ein reguläres Grid aus erkannten Rechtecken.
    Jede Kombination (col, row) wird als GridCell angelegt –
    auch wenn kein Rechteck exakt dort liegt.
    """
    if not rects:
        raise ValueError("Keine Rechtecke für Grid vorhanden.")

    rows    = _group(rects, axis=1, tol=tol)
    cols    = _group(rects, axis=0, tol=tol)

    # Repräsentative Position & Größe je Spalte/Zeile
    col_data = sorted(
        [(min(r[0] for r in c), max(r[2] for r in c)) for c in cols],
        key=lambda t: t[0]
    )  # [(x_pos, width), ...]
    row_data = sorted(
        [(min(r[1] for r in rw), max(r[3] for r in rw)) for rw in rows],
        key=lambda t: t[0]
    )  # [(y_pos, height), ...]

    n_cols = len(col_data)
    n_rows = len(row_data)

    cells = []
    for row_idx, (y_pos, rh) in enumerate(row_data):
        for col_idx, (x_pos, cw) in enumerate(col_data):
            cells.append(GridCell(
                col_idx=col_idx, row_idx=row_idx,
                x=int(x_pos), y=int(y_pos),
                w=int(cw),    h=int(rh),
                hsv_median=[0.0, 0.0, 0.0]
            ))

    return GridLayout(n_cols=n_cols, n_rows=n_rows, cells=cells)


# ══════════════════════════════════════════════════════════
# HSV-Analyse
# ══════════════════════════════════════════════════════════

def extract_hsv(image: np.ndarray, layout: GridLayout) -> GridLayout:
    """Füllt HSV-Median für alle Zellen im Layout."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for cell in layout.cells:
        roi = hsv[cell.y:cell.y+cell.h, cell.x:cell.x+cell.w]
        if roi.size > 0:
            cell.hsv_median = [
                float(np.median(roi[:, :, 0])),
                float(np.median(roi[:, :, 1])),
                float(np.median(roi[:, :, 2]))
            ]
    return layout


def suggest_color_columns(layout: GridLayout,
                           saturation_threshold: float = 30.0) -> list[int]:
    """
    Auto-Vorschlag: Spalten deren mittlere Sättigung über dem Threshold liegt
    werden als Farbspalten vorgeschlagen.
    Gibt Liste der Spalten-Indizes zurück.
    """
    col_sat: dict[int, list[float]] = {}
    for cell in layout.cells:
        col_sat.setdefault(cell.col_idx, []).append(cell.hsv_median[1])

    color_cols = []
    for col_idx, sats in col_sat.items():
        if np.mean(sats) >= saturation_threshold:
            color_cols.append(col_idx)

    return sorted(color_cols)


# ══════════════════════════════════════════════════════════
# Zuweisung: Parameter & Werte
# ══════════════════════════════════════════════════════════

def assign_columns(layout: GridLayout,
                   col_params: dict[int, str],
                   color_col_indices: list[int]) -> GridLayout:
    """
    Weist jeder Zelle Parameter und is_color_cell-Flag zu.
    col_params: {col_idx: "pH" | "Chlor_frei" | "leer" | ...}
    color_col_indices: welche Spalten als Farbfelder gelten
    """
    for cell in layout.cells:
        cell.parameter    = col_params.get(cell.col_idx, "leer")
        cell.is_color_cell = cell.col_idx in color_col_indices
    return layout


def assign_row_values(layout: GridLayout,
                      row_values: dict[int, dict[str, float]]) -> GridLayout:
    """
    Weist jeder Zelle den Messwert für ihren Parameter zu.
    row_values: {row_idx: {"pH": 7.2, "Chlor_frei": 0.5, ...}}
    """
    for cell in layout.cells:
        if not cell.is_color_cell or not cell.parameter:
            continue
        values_for_row = row_values.get(cell.row_idx, {})
        cell.value = values_for_row.get(cell.parameter)
    return layout


# ══════════════════════════════════════════════════════════
# Messung: Farbvergleich gegen Referenz
# ══════════════════════════════════════════════════════════

def hsv_distance(hsv_a: list[float], hsv_b: list[float]) -> float:
    """
    Berechnet den gewichteten Abstand zwischen zwei HSV-Farben.
    H (Farbton) wird zirkulär behandelt (0-180 in OpenCV).
    Gewichtung: H stärker als S und V, da Farbton das wichtigste Merkmal ist.
    """
    h_diff = abs(hsv_a[0] - hsv_b[0])
    h_diff = min(h_diff, 180 - h_diff)   # zirkulärer Abstand
    s_diff = abs(hsv_a[1] - hsv_b[1]) / 255.0 * 90   # normiert auf H-Skala
    v_diff = abs(hsv_a[2] - hsv_b[2]) / 255.0 * 60
    return h_diff * 2.0 + s_diff * 1.0 + v_diff * 0.5


def find_closest_value(query_hsv: list[float],
                        ref_db: ReferenceDB,
                        parameter: str) -> dict:
    """
    Findet den nächsten Referenzwert für eine gemessene Farbe.

    Returns:
        {
          "parameter": "pH",
          "value": 7.2,
          "distance": 12.3,
          "confidence": "high" | "medium" | "low",
          "candidates": [{"value": 7.2, "distance": 12.3}, ...]
        }
    """
    candidates = []
    for cell in ref_db.layout.cells:
        if cell.parameter != parameter or cell.value is None:
            continue
        dist = hsv_distance(query_hsv, cell.hsv_median)
        candidates.append({"value": cell.value, "distance": round(dist, 2)})

    if not candidates:
        return {"parameter": parameter, "value": None,
                "distance": None, "confidence": "none", "candidates": []}

    candidates.sort(key=lambda c: c["distance"])
    best = candidates[0]

    # Konfidenz basierend auf Abstand
    if best["distance"] < 15:
        confidence = "high"
    elif best["distance"] < 35:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "parameter": parameter,
        "value": best["value"],
        "distance": best["distance"],
        "confidence": confidence,
        "candidates": candidates[:3]   # Top-3 zur Anzeige
    }


def measure_image(measure_image_path: str,
                  ref_db: ReferenceDB,
                  min_area: int = 500) -> list[dict]:
    """
    Hauptfunktion für die Messung:
    Lädt ein neues Bild, erkennt Zellen, vergleicht mit Referenz.

    Returns: Liste von Messergebnissen pro Parameter.
    """
    image = load_image(measure_image_path)
    gray  = preprocess(image)
    rects = find_rects(gray, min_area=min_area)

    if not rects:
        raise ValueError("Keine Zellen im Messbild erkannt.")

    cropped, _ = crop_to_rects(image, rects)
    gray_crop  = preprocess(cropped)
    rects_crop = find_rects(gray_crop, min_area=min_area)
    layout     = build_grid(rects_crop)
    layout     = extract_hsv(cropped, layout)

    results = []
    for param in ref_db.layout.parameters():
        # Alle Farbzellen dieses Parameters im Messbild mitteln
        query_cells = [c for c in layout.cells if c.is_color_cell]
        if not query_cells:
            # Fallback: alle Zellen mit hoher Sättigung
            query_cells = [c for c in layout.cells if c.hsv_median[1] > 30]

        if query_cells:
            avg_hsv = [
                float(np.mean([c.hsv_median[i] for c in query_cells]))
                for i in range(3)
            ]
            result = find_closest_value(avg_hsv, ref_db, param)
            results.append(result)

    return results


# ══════════════════════════════════════════════════════════
# Hilfsfunktion: Debug-Visualisierung
# ══════════════════════════════════════════════════════════

def draw_grid(image: np.ndarray, layout: GridLayout) -> np.ndarray:
    """
    Zeichnet das Grid auf das Bild (für Debugging & UI-Preview).
    Farbspalten = grün, Beschriftung = grau, kein Parameter = dunkelgrau.
    """
    vis = image.copy()
    for cell in layout.cells:
        if cell.is_color_cell:
            color = (0, 220, 0)
        elif cell.parameter and cell.parameter != "leer":
            color = (0, 180, 220)
        else:
            color = (80, 80, 80)

        cv2.rectangle(vis, (cell.x, cell.y),
                      (cell.x + cell.w, cell.y + cell.h), color, 2)

        # Parameter-Label
        label = cell.parameter[:5] if cell.parameter else ""
        if label:
            cv2.putText(vis, label, (cell.x + 3, cell.y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Wert
        if cell.value is not None:
            cv2.putText(vis, str(cell.value),
                        (cell.x + 3, cell.y + cell.h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return vis
