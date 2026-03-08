"""
ocr_values.py
=============
Erkennt Messwerte automatisch aus den Beschriftungsspalten des Indikator-Bildes.

Strategie:
  1. Nicht-Farbspalten (Beschriftungen) aus dem Grid ausschneiden
  2. Jede Zeilen-ROI per OCR verarbeiten
  3. Zahlen aus dem OCR-Text extrahieren und Zeilen zuordnen
  4. Ergebnis: {row_idx: {parameter: float}} – kompatibel mit reference_core

Unterstützte OCR-Backends (in Prioritätsreihenfolge):
  - openocr-python  (bereits im Projekt, bevorzugt)
  - easyocr         (gutes Fallback, kein Setup nötig)
  - tesseract       (via pytesseract, klassisch)

Verwendung:
  from ocr_values import extract_row_values
  row_values = extract_row_values(cropped_image, layout, col_params)
"""

import cv2
import numpy as np
import re
from typing import Optional
from reference_core import GridLayout


# ══════════════════════════════════════════════════════════
# OCR-Backend laden (mit automatischem Fallback)
# ══════════════════════════════════════════════════════════

class OCREngine:
    """Wrapper der verschiedene OCR-Backends abstrahiert."""

    def __init__(self, preferred: str = "openocr"):
        self.backend_name = None
        self._engine = None
        self._init(preferred)

    def _init(self, preferred: str):
        order = [preferred] + [b for b in ["openocr", "easyocr", "tesseract"]
                               if b != preferred]
        for backend in order:
            try:
                if backend == "openocr":
                    from openocr import OpenOCR
                    self._engine = OpenOCR(backend='onnx', device='cpu')
                    self.backend_name = "openocr"
                    print(f"✅ OCR-Backend: openocr")
                    return
                elif backend == "easyocr":
                    import easyocr
                    self._engine = easyocr.Reader(['en', 'de'], gpu=False)
                    self.backend_name = "easyocr"
                    print(f"✅ OCR-Backend: easyocr")
                    return
                elif backend == "tesseract":
                    import pytesseract
                    pytesseract.get_tesseract_version()
                    self._engine = pytesseract
                    self.backend_name = "tesseract"
                    print(f"✅ OCR-Backend: tesseract")
                    return
            except (ImportError, Exception):
                continue

        print("⚠️  Kein OCR-Backend gefunden – manuelle Eingabe erforderlich.")
        self.backend_name = None

    def is_available(self) -> bool:
        return self.backend_name is not None

    def read_image(self, image: np.ndarray) -> str:
        """
        Liest Text aus einem numpy-Array (BGR).
        Gibt erkannten Text als String zurück.
        """
        if not self.is_available():
            return ""

        try:
            if self.backend_name == "openocr":
                # openocr erwartet Dateipfad oder numpy-Array (grau oder BGR)
                import tempfile, os
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    cv2.imwrite(tmp.name, gray)
                    result, _ = self._engine(tmp.name)
                os.unlink(tmp.name)
                return result if isinstance(result, str) else str(result)

            elif self.backend_name == "easyocr":
                results = self._engine.readtext(image, detail=0)
                return " ".join(results)

            elif self.backend_name == "tesseract":
                import pytesseract
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Digits-only Modus für Zahlen
                config = '--psm 7 -c tessedit_char_whitelist=0123456789.,'
                return pytesseract.image_to_string(gray, config=config)

        except Exception as e:
            print(f"⚠️  OCR-Fehler ({self.backend_name}): {e}")
            return ""


# Singleton – einmal laden, mehrfach nutzen
_ocr_engine: Optional[OCREngine] = None

def get_ocr_engine(preferred: str = "openocr") -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(preferred)
    return _ocr_engine


# ══════════════════════════════════════════════════════════
# Bild-Vorverarbeitung für OCR
# ══════════════════════════════════════════════════════════

def preprocess_for_ocr(roi: np.ndarray, scale: float = 3.0) -> np.ndarray:
    """
    Bereitet ein ROI-Bild für OCR vor:
    - Hochskalieren (OCR mag größere Bilder)
    - Graustufen + Schwellwert für klaren Kontrast
    - Leichtes Denoising
    """
    # Hochskalieren
    h, w = roi.shape[:2]
    enlarged = cv2.resize(roi, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # Graustufen
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) \
           if len(enlarged.shape) == 3 else enlarged

    # Kontrast erhöhen
    gray = cv2.equalizeHist(gray)

    # Adaptiver Schwellwert – robust bei unterschiedlicher Beleuchtung
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # Leichtes Denoise
    binary = cv2.medianBlur(binary, 3)

    # Zurück zu BGR für OCR-Backends die BGR erwarten
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════════
# Zahlen aus OCR-Text extrahieren
# ══════════════════════════════════════════════════════════

def extract_numbers(text: str) -> list[float]:
    """
    Extrahiert alle Zahlen aus einem OCR-Text.
    Behandelt ',' als Dezimaltrennzeichen (deutsches Format).

    Beispiele:
      "7.2"      → [7.2]
      "7,2"      → [7.2]
      "mg/l 0.5" → [0.5]
      "6.8\n7.0" → [6.8, 7.0]
      "100"      → [100.0]
    """
    # Komma → Punkt
    text = text.replace(',', '.')
    # Alle Zahlen (inkl. Dezimalzahlen) finden
    matches = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return [float(m) for m in matches]


def best_number(numbers: list[float],
                expected_range: Optional[tuple[float, float]] = None) -> Optional[float]:
    """
    Wählt die plausibelste Zahl aus einer Liste.
    Falls expected_range angegeben, wird die Zahl im Bereich bevorzugt.
    """
    if not numbers:
        return None
    if expected_range is not None:
        lo, hi = expected_range
        in_range = [n for n in numbers if lo <= n <= hi]
        if in_range:
            return in_range[0]
    return numbers[0]


# ══════════════════════════════════════════════════════════
# ROI für Beschriftungsspalten bestimmen
# ══════════════════════════════════════════════════════════

def get_label_rois(cropped_image: np.ndarray,
                   layout: GridLayout,
                   col_params: dict[int, str]) -> dict[int, np.ndarray]:
    """
    Schneidet für jede Zeile den Bereich der Beschriftungsspalten aus.

    Strategie: Alle Nicht-Farbspalten ("leer") links des ersten Farbfeldes
    werden als Beschriftungsspalten behandelt.

    Returns: {row_idx: roi_image}
    """
    # Beschriftungsspalten = links gelegene "leer"-Spalten
    label_cols = [col for col, param in col_params.items()
                  if param == "leer"]

    if not label_cols:
        # Fallback: erste Spalte
        label_cols = [0]

    rois = {}
    for row_idx in range(layout.n_rows):
        row_cells = [c for c in layout.cells
                     if c.row_idx == row_idx and c.col_idx in label_cols]
        if not row_cells:
            continue

        # Bounding Box über alle Label-Zellen der Zeile
        x_min = min(c.x for c in row_cells)
        y_min = min(c.y for c in row_cells)
        x_max = max(c.x + c.w for c in row_cells)
        y_max = max(c.y + c.h for c in row_cells)

        # Kleinen Rand abschneiden
        pad = 4
        x_min = max(x_min + pad, 0)
        y_min = max(y_min + pad, 0)
        x_max = min(x_max - pad, cropped_image.shape[1])
        y_max = min(y_max - pad, cropped_image.shape[0])

        roi = cropped_image[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            rois[row_idx] = roi

    return rois


# ══════════════════════════════════════════════════════════
# Hauptfunktion: Zeilenwerte per OCR bestimmen
# ══════════════════════════════════════════════════════════

def extract_row_values(
    cropped_image: np.ndarray,
    layout: GridLayout,
    col_params: dict[int, str],
    ocr_backend: str = "openocr",
    value_ranges: Optional[dict[str, tuple[float, float]]] = None,
    fallback_manual: bool = True,
    debug: bool = False
) -> dict[int, dict[str, float]]:
    """
    Erkennt Messwerte automatisch aus den Beschriftungsspalten.

    Args:
        cropped_image:  Das auf das Grid zugeschnittene Bild
        layout:         GridLayout aus reference_core
        col_params:     {col_idx: parameter_name} z.B. {0: "leer", 1: "pH", ...}
        ocr_backend:    "openocr" | "easyocr" | "tesseract"
        value_ranges:   Plausibilitätsgrenzen je Parameter,
                        z.B. {"pH": (6.0, 9.0), "Chlor_frei": (0.0, 10.0)}
        fallback_manual: Bei OCR-Fehler manuelle Eingabe abfragen
        debug:          Debug-Bilder anzeigen

    Returns:
        {row_idx: {"pH": 7.2, "Chlor_frei": 0.5, ...}}
        Kompatibel mit reference_core.assign_row_values()
    """
    engine = get_ocr_engine(ocr_backend)
    active_params = [p for p in col_params.values() if p != "leer"]

    # Bekannte Wertebereiche (Standard-Defaults falls nicht angegeben)
    default_ranges = {
        "pH":            (6.0, 9.0),
        "Chlor_frei":    (0.0, 10.0),
        "Chlor_gesamt":  (0.0, 10.0),
        "PHMB":          (0.0, 200.0),
        "Alkalinität":   (0.0, 500.0),
    }
    if value_ranges:
        default_ranges.update(value_ranges)

    # ROIs für Beschriftungsspalten
    label_rois = get_label_rois(cropped_image, layout, col_params)

    row_values: dict[int, dict[str, float]] = {}
    ocr_results_raw: dict[int, str] = {}

    print(f"\n🔍 OCR-Erkennung ({engine.backend_name or 'nicht verfügbar'})...")
    print(f"   {len(label_rois)} Zeilen werden verarbeitet")

    for row_idx, roi in sorted(label_rois.items()):
        # Vorverarbeitung
        roi_processed = preprocess_for_ocr(roi)

        # Debug-Ausgabe
        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 1))
            plt.title(f"Zeile {row_idx} OCR-Input")
            plt.imshow(cv2.cvtColor(roi_processed, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

        # OCR
        raw_text = engine.read_image(roi_processed) if engine.is_available() else ""
        ocr_results_raw[row_idx] = raw_text.strip()

        numbers = extract_numbers(raw_text)

        # Zahlen den Parametern zuordnen
        # Annahme: Beschriftungsspalte enthält eine Zahl je Parameter
        # (bei Lovibond: eine Zahl für alle Parameter der Zeile)
        row_dict: dict[str, float] = {}

        if len(active_params) == 1:
            # Einfachster Fall: eine Zahl → ein Parameter
            param = active_params[0]
            val = best_number(numbers, default_ranges.get(param))
            if val is not None:
                row_dict[param] = val

        elif len(numbers) >= len(active_params):
            # Eine Zahl pro Parameter in Reihenfolge
            for i, param in enumerate(active_params):
                val = best_number([numbers[i]], default_ranges.get(param))
                if val is not None:
                    row_dict[param] = val

        elif len(numbers) == 1:
            # Nur eine Zahl erkannt → gilt für alle Parameter (Lovibond-Stil)
            # Jeder Parameter bekommt dieselbe Zahl (unterschiedliche Skalen!)
            # → Besser: Zahl dem ersten Parameter zuordnen, Rest manuell
            param = active_params[0]
            val = best_number(numbers, default_ranges.get(param))
            if val is not None:
                row_dict[param] = val

        row_values[row_idx] = row_dict
        status = "✅" if row_dict else "❌"
        print(f"   Zeile {row_idx:2d}: {status} OCR='{raw_text.strip()[:30]}' "
              f"→ Zahlen={numbers} → {row_dict}")

    # ── Fallback: Manuelle Eingabe für fehlgeschlagene Zeilen ──
    if fallback_manual:
        failed_rows = [r for r in range(layout.n_rows)
                       if not row_values.get(r)]
        if failed_rows:
            print(f"\n⚠️  {len(failed_rows)} Zeilen ohne erkannten Wert → manuelle Eingabe:")
            for row_idx in failed_rows:
                raw = ocr_results_raw.get(row_idx, "")
                hint = f" (OCR erkannte: '{raw}')" if raw else ""
                print(f"   Parameter: {active_params}")
                val_input = input(f"   Zeile {row_idx}{hint} [{' / '.join(active_params)}]: ").strip()
                parts = val_input.replace(',', '.').split()
                row_dict = {}
                for i, param in enumerate(active_params):
                    if i < len(parts):
                        try:
                            row_dict[param] = float(parts[i])
                        except ValueError:
                            pass
                row_values[row_idx] = row_dict

    return row_values


# ══════════════════════════════════════════════════════════
# Validierung & Zusammenfassung
# ══════════════════════════════════════════════════════════

def validate_row_values(
    row_values: dict[int, dict[str, float]],
    n_rows: int,
    active_params: list[str]
) -> dict:
    """
    Prüft die erkannten Werte auf Vollständigkeit und Plausibilität.

    Returns:
        {
          "complete": bool,
          "coverage": float,        # Anteil erkannter Werte 0.0-1.0
          "missing": [(row, param)],
          "summary": str
        }
    """
    missing = []
    total = n_rows * len(active_params)
    found = 0

    for row_idx in range(n_rows):
        for param in active_params:
            val = row_values.get(row_idx, {}).get(param)
            if val is not None:
                found += 1
            else:
                missing.append((row_idx, param))

    coverage = found / total if total > 0 else 0.0
    complete = len(missing) == 0

    summary = (f"{'✅ Vollständig' if complete else '⚠️  Unvollständig'}: "
               f"{found}/{total} Werte erkannt ({coverage*100:.0f}%)")

    return {
        "complete": complete,
        "coverage": coverage,
        "missing": missing,
        "summary": summary
    }


def print_ocr_summary(row_values: dict[int, dict[str, float]],
                       n_rows: int,
                       active_params: list[str]):
    """Gibt eine übersichtliche Tabelle der erkannten Werte aus."""
    validation = validate_row_values(row_values, n_rows, active_params)
    print(f"\n{'═'*55}")
    print("OCR-ERGEBNIS")
    print(f"{'═'*55}")
    print(f"{'Zeile':>6}  " + "  ".join(f"{p:>12}" for p in active_params))
    print(f"{'─'*55}")
    for row_idx in range(n_rows):
        vals = row_values.get(row_idx, {})
        row_str = f"{row_idx:>6}  "
        for param in active_params:
            v = vals.get(param)
            row_str += f"{str(v) if v is not None else '—':>12}  "
        print(row_str)
    print(f"{'─'*55}")
    print(validation["summary"])
