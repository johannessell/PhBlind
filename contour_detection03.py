import cv2
import numpy as np
from matplotlib import pyplot as plt
from openocr import OpenOCR

# -------------------------------
# 1. Bild laden
# -------------------------------
image = cv2.imread('reference2.jpg')

def capture_reference_photo(cap, scale):
    """
    Zeigt Kamerabild an und wartet auf Leertaste zum Aufnehmen.
    Gibt das aufgenommene Bild in ORIGINAL-Auflösung zurück.
    """
    print("Kein Referenzbild vorhanden.")
    print("  → Messindikator vor die Kamera halten und LEERTASTE drücken.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Anzeige skaliert, aber Original behalten
        display = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame.copy()
        cv2.rectangle(display, (10, 10), (500, 50), (0, 0, 0), -1)
        cv2.putText(display, "Referenz aufnehmen: LEERTASTE druecken",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow('Pool Check', display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            print(f"  Foto aufgenommen: {frame.shape[1]}x{frame.shape[0]}px (Original)")
            return frame  # Original-Auflösung für OCR
        if key == ord('q'):
            return None

# cap = cv2.VideoCapture(1)

# image = capture_reference_photo(cap, 1)

if image is None:
    raise Exception("Bild konnte nicht geladen werden!")

# Graustufen für erste Konturerkennung
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 5, 200, 200)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# Canny
canny = cv2.Canny(gray_enhanced, 30, 120)
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Bilder für Konturen
contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
original_with_contours = image.copy()

# -------------------------------
# 2. Rechtecke sammeln
# -------------------------------
rects = []
for cnt in contours:
    if cv2.contourArea(cnt) < 1000:
        continue
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(original_with_contours, [cnt], -1, (0,0,255), 2)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 10:
            cv2.drawContours(contour_image, [approx], -1, (0,255,0), 2)
            cv2.drawContours(original_with_contours, [approx], -1, (0,255,0), 2)
            rects.append((x, y, w, h))

print(f"Gefundene Rechtecke: {len(rects)}")
for i, r in enumerate(rects):
    print(f"  Rechteck {i}: x={r[0]}, y={r[1]}, w={r[2]}, h={r[3]}")

# -------------------------------
# 3. Gemeinsames blaues Rechteck
# -------------------------------
margin = 50
x_min = max(min(r[0] for r in rects) - margin, 0)
y_min = max(min(r[1] for r in rects) - margin, 0)
x_max = min(max(r[0]+r[2] for r in rects) + margin, image.shape[1])
y_max = min(max(r[1]+r[3] for r in rects) + margin, image.shape[0])

cv2.rectangle(contour_image, (x_min,y_min), (x_max,y_max), (255,0,0), 5)

# -------------------------------
# 4. Cropping
# -------------------------------
cropped_image = image[y_min:y_max, x_min:x_max].copy()

# -------------------------------
# 5. Kantenerkennung im Cropped-Image
# -------------------------------
gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
gray_cropped = cv2.bilateralFilter(gray_cropped, 5, 200, 200)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_cropped_enhanced = clahe.apply(gray_cropped)
canny_cropped = cv2.Canny(gray_cropped_enhanced, 20, 80)  # Niedrigere Schwellen für mehr Konturen
contours_cropped, _ = cv2.findContours(canny_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------------
# 6. Gefüllte Konturen in wrapped eintragen
# -------------------------------
wrapped = np.zeros_like(cropped_image)  # schwarzes Image
rects_cropped = []
for cnt in contours_cropped:
    if cv2.contourArea(cnt) < 300:  # Reduzierte Schwelle für mehr Konturen
        continue
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        rects_cropped.append((x, y, w, h))
        cv2.drawContours(wrapped, [approx], -1, (0,255,0), thickness=-1)


# -------------------------------
# 7. Hilfsfunktionen für Grid
# -------------------------------
def group_rects_by_axis(rects, axis=1, tol=20):
    # axis=0 -> sortiere nach x, axis=1 -> sortiere nach y
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

def group_columns(rects, tol=3):  # Noch niedrigere Toleranz
    rects_sorted = sorted(rects, key=lambda r: r[0])  # sortiere nach x
    columns = []
    for r in rects_sorted:
        placed = False
        for col in columns:
            if abs(col[0][0] - r[0]) < tol:
                col.append(r)
                placed = True
                break
        if not placed:
            columns.append([r])
    return columns

def classify_and_group_columns(columns, hsv_image, max_group_distance=150):
    """
    Neue Gruppierungslogik basierend auf räumlicher Nähe:
      - Spalten werden zu Gruppen zusammengefasst, wenn sie benachbart sind
      - Jede Gruppe enthält eine Measure-Spalte und ihre benachbarten Color-Spalten
      - Hue-Ähnlichkeit wird als Validierung verwendet

    Returns:
        col_types : dict {col_idx: 'color'|'measure'}
        col_stats : dict {col_idx: {'hue_median': float, 'sat_median': float, 'hue_iqr': float}}
        groups    : list of {'cols': [col_idx,...], 'measure_col': col_idx}
    """
    col_stats = {}
    for i, col_rects in enumerate(columns):
        hue_vals, sat_vals = [], []
        cell_sat_medians = []
        for (x, y, w, h) in col_rects:
            roi = hsv_image[y:y+h, x:x+w]
            if roi.size > 0:
                hue_vals.extend(roi[:, :, 0].flatten().tolist())
                sat_vals.extend(roi[:, :, 1].flatten().tolist())
                cell_sat_medians.append(float(np.median(roi[:, :, 1])))
        col_stats[i] = {
            'hue_median': float(np.median(hue_vals)) if hue_vals else 0.0,
            'sat_median': float(np.median(sat_vals)) if sat_vals else 0.0,
            'hue_iqr': float(np.percentile(hue_vals, 75) - np.percentile(hue_vals, 25))
                       if len(hue_vals) >= 4 else 0.0,
            # Maximale Zellen-Median-Saettigung: robust gegen Spalten mit
            # einigen blassen Reihen (z.B. Label- und Randreihen), in denen
            # sat_median zu niedrig ausfaellt, obwohl mittlere Reihen echte
            # Farb-Swatches enthalten.
            'cell_sat_max': max(cell_sat_medians) if cell_sat_medians else 0.0,
            'cell_sat_p75': float(np.percentile(cell_sat_medians, 75))
                            if len(cell_sat_medians) >= 4 else
                            (max(cell_sat_medians) if cell_sat_medians else 0.0),
        }

    # Schritt 1: Measure-Spalten identifizieren.
    # Regel: Enthaelt die Spalte mindestens eine Zelle mit hoher Median-
    # Saettigung (>= 70), ist es eine Farb-Spalte. Measure-/Label-Spalten
    # zeigen ueber ALLE Reihen blasse Sat-Mediane; pooled sat_median ist
    # unzuverlaessig, weil einige Label- und Randreihen innerhalb einer
    # Farbspalte die Mediansaettigung unter 60 druecken koennen
    # (z.B. PHMB-Swatch-Spalte mit blasser oberster und unterster Zeile).
    col_types = {}
    measure_cols = []
    color_cols = []

    COLOR_SAT_MAX = 70.0
    for i, s in col_stats.items():
        if s['cell_sat_max'] >= COLOR_SAT_MAX:
            col_types[i] = 'color'
            color_cols.append(i)
        else:
            col_types[i] = 'measure'
            measure_cols.append(i)

    # Schritt 2: Gruppen bilden basierend auf räumlicher Nähe
    # Jede Gruppe muss mindestens eine measure- und eine color-Spalte haben
    groups = []
    used_cols = set()

    # Berechne x-Positionen der Spalten
    col_positions = []
    for i, col_rects in enumerate(columns):
        x_pos = min(r[0] for r in col_rects)
        col_positions.append(x_pos)

    for measure_idx in sorted(measure_cols):
        if measure_idx in used_cols:
            continue

        group_cols = [measure_idx]
        used_cols.add(measure_idx)
        measure_x = col_positions[measure_idx]

        # Sammle benachbarte Spalten (links und rechts)
        nearby_color_cols = []
        for col_idx in range(len(columns)):
            if col_idx in used_cols:
                continue

            col_x = col_positions[col_idx]
            distance = abs(col_x - measure_x)
            if distance <= max_group_distance and col_types[col_idx] == 'color':
                nearby_color_cols.append(col_idx)
                group_cols.append(col_idx)
                used_cols.add(col_idx)

        # Gruppe nur bilden, wenn mindestens eine color-Spalte vorhanden ist
        if nearby_color_cols:
            # Sortiere die Gruppe nach x-Position
            group_cols.sort(key=lambda idx: col_positions[idx])

            groups.append({
                'cols': group_cols,
                'measure_col': measure_idx
            })
        else:
            # Wenn keine color-Spalten gefunden, measure-Spalte wieder freigeben
            used_cols.remove(measure_idx)

    # Schritt 3: Verbleibende Spalten ohne vollständige Gruppe werden als separate Gruppen behandelt,
    # auch wenn sie keine color-Spalten haben (für den Fall, dass alle Spalten measure sind)
    for col_idx in range(len(columns)):
        if col_idx not in used_cols:
            # Finde benachbarte measure-Spalten
            group_cols = [col_idx]
            used_cols.add(col_idx)
            col_x = col_positions[col_idx]
            for other_idx in range(len(columns)):
                if other_idx not in used_cols and col_types[other_idx] == 'measure':
                    other_x = col_positions[other_idx]
                    if abs(other_x - col_x) <= max_group_distance:
                        group_cols.append(other_idx)
                        used_cols.add(other_idx)
            group_cols.sort(key=lambda idx: col_positions[idx])
            groups.append({
                'cols': group_cols,
                'measure_col': min(group_cols) if col_types[min(group_cols)] == 'measure' else None
            })

    # Nach x-Position sortieren
    groups.sort(key=lambda g: col_positions[g['cols'][0]])

    return col_types, col_stats, groups


# -------------------------------
# 8. Grid auf dem Cropped-Image zeichnen
# -------------------------------
cropped_with_grid = wrapped.copy()
rects_grid = []
if rects_cropped:
    rows = group_rects_by_axis(rects_cropped, axis=1, tol=20)
    for row in rows:
        row.sort(key=lambda r: r[0])
    columns = group_columns(rects_cropped, tol=20)
    col_positions = [min(r[0] for r in col) for col in columns]
    col_widths = [max(r[2] for r in col) for col in columns]  # Breite = w
    row_positions = [min(r[1] for r in row) for row in rows]
    row_heights = [max(r[3] for r in row) for row in rows]  # Höhe = h

    for row_idx, rh in enumerate(row_heights):
        y_pos = row_positions[row_idx]
        for col_idx, cw in enumerate(col_widths):
            x_pos = col_positions[col_idx]
            rects_grid.append((x_pos, y_pos, cw, rh))
            cv2.rectangle(
                cropped_with_grid,
                (x_pos, y_pos),
                (x_pos + cw, y_pos + rh),
                (0,255,255), 2
            )

# -------------------------------
# 8b. Spalten klassifizieren und gruppieren
# -------------------------------
hsv_for_classify = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
col_types, col_stats, col_groups = classify_and_group_columns(
    columns, hsv_for_classify,
    max_group_distance=130,  # Passende Distanz für die erwarteten Gruppen
)

# Ausgabe
print("\n--- Spalten-Positionen ---")
col_positions = [min(r[0] for r in col) for col in columns]
for i, pos in enumerate(col_positions):
    print(f"  Spalte {i}: x={pos}")
print("\n--- Spalten-Klassifikation ---")
for i in range(len(columns)):
    s = col_stats[i]
    print(f"  Spalte {i:2d}: {col_types[i]:8s}  hue={s['hue_median']:.1f}  sat={s['sat_median']:.1f}  hue_iqr={s['hue_iqr']:.1f}")
print("\n--- Spaltengruppen ---")
for g in col_groups:
    print(f"  cols={g['cols']}  measure_col={g['measure_col']}")

# -------------------------------
# 8c. Crosscheck: color-Spalten einer Gruppe per Hue-Ähnlichkeit validieren
# -------------------------------
def crosscheck_color_columns(group, columns, hsv_image, rows, hue_diff_threshold=25.0):
    """
    Prüft ob die color-Spalten einer Gruppe in jeder Zeile ähnliche Hue-Werte haben.
    Berechnet pro Zeile den Hue-Median jeder color-Spalte und vergleicht die Differenz.
    Gibt Warnungen aus wenn color-Spalten sich stark unterscheiden.

    Returns: list of (row_idx, hue_per_col, max_diff, ok)
    """
    color_cols = [c for c in group['cols'] if col_types[c] == 'color']
    if len(color_cols) < 2:
        return []  # nichts zu vergleichen

    results = []
    for row_idx, row_rects in enumerate(rows):
        row_y = min(r[1] for r in row_rects)
        row_h = max(r[3] for r in row_rects)

        hues = {}
        for col_idx in color_cols:
            col_x = min(r[0] for r in columns[col_idx])
            col_w = max(r[2] for r in columns[col_idx])
            roi = hsv_image[row_y:row_y+row_h, col_x:col_x+col_w, 0]
            hues[col_idx] = float(np.median(roi)) if roi.size > 0 else -1.0

        hue_vals = list(hues.values())
        # zirkuläre Differenz (H in 0-180)
        diffs = []
        for a in hue_vals:
            for b in hue_vals:
                d = abs(a - b)
                diffs.append(min(d, 180.0 - d))
        max_diff = max(diffs) if diffs else 0.0
        ok = max_diff <= hue_diff_threshold
        results.append((row_idx, hues, round(max_diff, 1), ok))

    return results

print("\n--- Crosscheck: Hue-Ähnlichkeit color-Spalten pro Gruppe ---")
rows = group_rects_by_axis(rects_cropped, axis=1, tol=20)
for g_idx, g in enumerate(col_groups):
    color_cols = [c for c in g['cols'] if col_types[c] == 'color']
    if len(color_cols) < 2:
        continue
    checks = crosscheck_color_columns(g, columns, hsv_for_classify, rows, hue_diff_threshold=30.0)
    warnings = [c for c in checks if not c[3]]
    status = "OK" if not warnings else f"{len(warnings)} Zeilen abweichend"
    print(f"  Gruppe {g_idx} cols={g['cols']}  -> {status}")
    for (row_idx, hues, max_diff, ok) in checks:
        marker = "  " if ok else "⚠️"
        hue_str = "  ".join(f"Sp{c}:{v:.0f}" for c, v in hues.items())
        print(f"    {marker} Zeile {row_idx:2d}: {hue_str}  max_diff={max_diff}")

# Visualisierung: Gruppen farbig einzeichnen
GROUP_COLORS = [
    (0,255,255),(255,0,255),(0,165,255),
    (0,255,0),  (255,128,0),(128,0,255),
]

cropped_with_groups = cropped_image.copy()
for g_idx, g in enumerate(col_groups):
    grp_color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
    for col_idx in g['cols']:
        t = col_types[col_idx]
        # rects_grid verwenden damit alle Zellen (auch interpolierte) sichtbar sind
        for (x, y, w, h) in [(r[0], r[1], r[2], r[3])
                              for r in rects_grid
                              if col_positions[col_idx] == r[0]]:
            cv2.rectangle(cropped_with_groups, (x, y), (x+w, y+h), grp_color, 3)
            cv2.putText(cropped_with_groups, 'C' if t == 'color' else 'M',
                        (x+3, y+h-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, grp_color, 1, cv2.LINE_AA)

# -------------------------------
# 9. Overlay auf Cropped-Image
# -------------------------------
alpha = 0.5
overlayed = cv2.addWeighted(cropped_image, 1-alpha, cropped_with_grid, alpha, 0)

# -------------------------------
# Vollständigen HSV-Mittelwert pro Grid-Zelle berechnen
# -------------------------------
hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
overlay_hsv = overlayed.copy()
grid_hsv_stats = []

for idx, (x, y, w, h) in enumerate(rects_grid):

    roi = hsv[y:y+h, x:x+w]

    if roi.size > 0:
        H = float(np.median(roi[:,:,0]))
        S = float(np.median(roi[:,:,1]))
        V = float(np.median(roi[:,:,2]))
    else:
        H = S = V = None

    grid_hsv_stats.append((idx, x, y, w, h, H, S, V))

    # Text im Bild darstellen, wenn Werte vorhanden
    if H is not None:
        cv2.putText(
            overlay_hsv,
            f"H:{H:.0f}",
            (x + 3, y + int(h * 0.45)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255,255,255),
            2,
            cv2.LINE_AA
        )
        # cv2.putText(
        #     overlay_hsv,
        #     f"S:{S:.0f} V:{V:.0f}",
        #     (x + 3, y + int(h * 0.8)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.50,
        #     (255,255,255),
        #     2,
        #     cv2.LINE_AA
        # )

print("\n--- HSV pro Grid-Zelle ---")
for cell in grid_hsv_stats:
    idx, x, y, w, h, H, S, V = cell
    print(f"Zelle {idx:02d}: H={H:.1f}, S={S:.1f}, V={V:.1f}  (x={x}, y={y}, w={w}, h={h})")

# -------------------------------
# Zellen farblich markieren, falls S < 10
# -------------------------------
overlay_marked = overlay_hsv.copy()

for (idx, x, y, w, h, H, S, V) in grid_hsv_stats:
    if S is not None and S < 20:
        # sehr geringe Sättigung -> schwach / weiß / unmarkiert
        color = (255, 0, 0)   # ROTER Rahmen für "zu blass"
    else:
        color = (0, 255, 0)   # GRÜNER Rahmen für "ok"

    cv2.rectangle(
        overlay_marked,
        (x, y),
        (x + w, y + h),
        color,
        3
    )

## OCR
# Parameter für bessere Erkennung:
# drop_score: Schwellenwert für Erkennungskonfidenz (0.0-1.0, niedriger = mehr Treffer)
# mode: 'mobile' für schnell, 'server' für genau
import os
model_dir = os.path.expanduser('~/.cache/openocr/')

# Ergebnis parsen: result[0] = "dateiname\t[{...JSON...}]"
import json, re

def parse_ocr_result(result):
    """
    Parst die OpenOCR-Ausgabe.
    Gibt Liste von {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'score': float} zurück.
    """
    blocks = []
    for item in result:
        # Format: "dateiname\t[{JSON}]"  oder nur "[{JSON}]"
        json_str = item.split('\t')[-1].strip()
        try:
            detections = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        for det in detections:
            text  = det.get('transcription', '').strip()
            score = det.get('score', 0.0)
            pts   = det.get('points', [])
            if not text or not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y
            blocks.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h, 'score': score})
    return blocks

# Mehrere OCR-Läufe mit verschiedenen Vorverarbeitungen
def run_multi_ocr(image):
    """Führt OCR mit mehreren Vorverarbeitungsvarianten aus"""
    all_blocks = []

    # Variante 1: Original (einfach)
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('tmp1.jpg', gray1)
    ocr1 = OpenOCR(backend='onnx', device='cpu', drop_score=0.5, mode='mobile',
                   onnx_det_model_path=os.path.join(model_dir, 'openocr_det_model.onnx'),
                   onnx_rec_model_path=os.path.join(model_dir, 'openocr_rec_model.onnx'))
    result1, _ = ocr1('tmp1.jpg')
    all_blocks.extend(parse_ocr_result(result1))

    # Variante 2: Mit CLAHE
    gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray2)
    cv2.imwrite('tmp2.jpg', enhanced)
    ocr2 = OpenOCR(backend='onnx', device='cpu', drop_score=0.4, mode='server',
                   onnx_det_model_path=os.path.join(model_dir, 'openocr_det_model.onnx'),
                   onnx_rec_model_path=os.path.join(model_dir, 'openocr_rec_model.onnx'))
    result2, _ = ocr2('tmp2.jpg')
    all_blocks.extend(parse_ocr_result(result2))

    # Entferne Duplikate (gleiche Position ±5px)
    unique_blocks = []
    for block in all_blocks:
        is_duplicate = False
        for existing in unique_blocks:
            if (abs(block['x'] - existing['x']) < 5 and
                abs(block['y'] - existing['y']) < 5 and
                abs(block['w'] - existing['w']) < 5):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_blocks.append(block)

    return unique_blocks

ocr_blocks = run_multi_ocr(cropped_image)

# Grenze zwischen Kopfbereich (Parameter-Namen) und Grid (Messwerte)
# = y-Position der obersten Grid-Zeile
grid_top = min(r[1] for r in rects_grid) if rects_grid else 0

# Blöcke zuordnen
ocr_header = [b for b in ocr_blocks if b['y'] + b['h'] // 2 < grid_top]  # Parameter-Namen
ocr_values = [b for b in ocr_blocks if b['y'] + b['h'] // 2 >= grid_top] # Messwerte

# Messwert-Blöcke den Zeilen zuordnen (per y-Überlappung)
def assign_to_row(block, row_positions, row_heights, tol=10):
    cy = block['y'] + block['h'] // 2
    for row_idx, (ry, rh) in enumerate(zip(row_positions, row_heights)):
        if ry - tol <= cy <= ry + rh + tol:
            return row_idx
    return None

# Parameter-Blöcke den Gruppen zuordnen (per x-Überlappung)
def assign_to_group(block, col_groups, col_positions, col_widths):
    cx = block['x'] + block['w'] // 2
    for g_idx, g in enumerate(col_groups):
        x_min = min(col_positions[c] for c in g['cols'])
        x_max = max(col_positions[c] + col_widths[c] for c in g['cols'])
        if x_min <= cx <= x_max:
            return g_idx
    return None

print("\n--- OCR Ergebnisse ---")
print(f"Erkannte Blöcke gesamt: {len(ocr_blocks)}")

# Bekannte OCR-Korrekturen für chemische Bezeichnungen
OCR_CORRECTIONS = {
    'HO2': 'H2O2', 'H02': 'H2O2', 'H0²': 'H2O2', 'HO²': 'H2O2',
    'Ho2': 'H2O2', 'h2o2': 'H2O2', 'h2o': 'H2O',
    'Ph':  'pH',   'PH':  'pH',   'ph':  'pH',
    'PhMS': 'PHMB', 'PHMS': 'PHMB', 'Phmb': 'PHMB',
}

def correct_param(text):
    return OCR_CORRECTIONS.get(text, text)

# -------------------------------
# Parameter-Namen den Gruppen zuordnen
# -------------------------------
group_params = {}  # {g_idx: text}
for b in ocr_header:
    g_idx = assign_to_group(b, col_groups, col_positions, col_widths)
    if g_idx is not None:
        if g_idx not in group_params or b['score'] > group_params[g_idx]['score']:
            group_params[g_idx] = b

print("\nParameter-Namen:")
for g_idx, b in sorted(group_params.items()):
    corrected = correct_param(b['text'])
    print(f"  Gruppe {g_idx}: '{b['text']}' → '{corrected}'  score={b['score']:.2f}")

# -------------------------------
# Messwerte den Zeilen UND Gruppen zuordnen
# -------------------------------
row_group_values = {}  # {row_idx: {g_idx: float}}

for b in ocr_values:
    is_number = bool(re.match(r'^\d+([.,]\d+)?$', b['text'].replace(' ', '')))
    if not is_number:
        continue
    row_idx = assign_to_row(b, row_positions, row_heights)
    g_idx   = assign_to_group(b, col_groups, col_positions, col_widths)
    if row_idx is None or g_idx is None:
        continue
    value = float(b['text'].replace(',', '.'))
    row_group_values.setdefault(row_idx, {})[g_idx] = value

# Zusammenfassung ausgeben
print("\nMesswerte (Zeile × Parameter):")
param_names = {g_idx: correct_param(b['text']) for g_idx, b in group_params.items()}
g_indices = sorted({g for row in row_group_values.values() for g in row})
header = "  Zeile  " + "  ".join(f"{param_names.get(g, f'Gr{g}'):>8}" for g in g_indices)
print(header)
for row_idx in sorted(row_group_values):
    vals = row_group_values[row_idx]
    row_str = f"  {row_idx:5d}  " + "  ".join(
        f"{str(vals.get(g, '—')):>8}" for g in g_indices
    )
    print(row_str)


# -------------------------------
# JSON Referenzdatenbank speichern
# -------------------------------
reference = {
    "image_file": 'template02.jpg',
    "parameters": [],
    "cells": []
}

# Parameter-Namen + Gruppen
for g_idx, g in enumerate(col_groups):
    param = param_names.get(g_idx)
    if param:
        reference["parameters"].append({
            "name": param,
            "group_idx": g_idx,
            "cols": g['cols'],
            "measure_col": g['measure_col']
        })

# Zellen: HSV + Messwert + Parameter
# col_positions und row_positions sind bereits sortiert →
# Spalten-/Zeilen-Index direkt aus rects_grid-Position ableiten
for list_idx, (idx, x, y, w, h, H, S, V) in enumerate(grid_hsv_stats):
    if H is None:
        continue

    # Spalten-Index: welche col_position passt zu x?
    cell_col_idx = None
    for ci, cx in enumerate(col_positions):
        if abs(cx - x) < col_widths[ci]:
            cell_col_idx = ci
            break

    # Zeilen-Index: welche row_position passt zu y?
    row_idx = None
    for ri, (ry, rh) in enumerate(zip(row_positions, row_heights)):
        if abs(ry - y) < rh:
            row_idx = ri
            break

    # Gruppen-Index
    g_idx = None
    for gi, g in enumerate(col_groups):
        if cell_col_idx in g['cols']:
            g_idx = gi
            break

    # Messwert aus OCR – nur für measure-Zellen
    is_color = col_types.get(cell_col_idx) == 'color' if cell_col_idx is not None else False
    value = None
    if not is_color and row_idx is not None and g_idx is not None:
        value = row_group_values.get(row_idx, {}).get(g_idx)

    param_name = param_names.get(g_idx) if g_idx is not None else None

    reference["cells"].append({
        "cell_idx": idx,
        "row_idx": row_idx,
        "group_idx": g_idx,
        "parameter": param_name,
        "is_color_cell": is_color,
        "x": x, "y": y, "w": w, "h": h,
        "hsv_median": [round(H, 1), round(S, 1), round(V, 1)],
        "value": value
    })

ref_path = 'reference.json'
with open(ref_path, 'w', encoding='utf-8') as f:
    json.dump(reference, f, indent=2, ensure_ascii=False)

n_with_value = sum(1 for c in reference["cells"] if c["value"] is not None)
print(f"\n✅ Referenz gespeichert: {ref_path}")
print(f"   Parameter: {[p['name'] for p in reference['parameters']]}")
print(f"   Zellen gesamt: {len(reference['cells'])}  |  mit Messwert: {n_with_value}")


plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title("Original + Konturen")
plt.imshow(cv2.cvtColor(original_with_contours, cv2.COLOR_BGR2RGB))
plt.savefig('debug_original.png')

plt.subplot(1,3,2)
plt.title("Cropped + Gefüllte Rechtecke")
plt.imshow(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB))
plt.savefig('debug_cropped.png')

plt.subplot(1,3,3)
plt.title("Overlay + Grid + Hue")
plt.imshow(cv2.cvtColor(overlay_marked, cv2.COLOR_BGR2RGB))
plt.savefig('debug_overlay.png')
# plt.show()

# Spaltengruppen separat anzeigen
plt.figure(figsize=(10, 6))
plt.title("Spaltengruppen  (C=Color  M=Measure  |  gleiche Farbe = eine Gruppe)")
plt.imshow(cv2.cvtColor(cropped_with_groups, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.tight_layout()
plt.savefig('debug_groups.png')
plt.show()

# OpenCV Fenster anzeigen
cv2.imshow('Original + Konturen', original_with_contours)
cv2.imshow('Cropped + Gefüllte Rechtecke', wrapped)
cv2.imshow('Overlay + Grid + Hue', overlay_marked)
cv2.imshow('Spaltengruppen', cropped_with_groups)
cv2.waitKey(0)
cv2.destroyAllWindows()