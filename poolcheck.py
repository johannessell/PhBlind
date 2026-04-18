"""
poolcheck.py
============
Kombinierter Workflow:
  1. Kamera öffnen
  2. Falls keine Referenz vorhanden: Foto aufnehmen → Grid erkennen → OCR → reference.json
  3. Live-Erkennung: ORB+SIFT → Homographie → LAB-Polyfit → Messwerte anzeigen

Verwendung:
  python poolcheck.py [--reference reference.json] [--camera 0]
"""

import cv2
import numpy as np
import json
import re
import os
import time
import argparse
from openocr import OpenOCR


# ══════════════════════════════════════════════════════════
# PHASE 1: Referenz erstellen (aus contour_detection03)
# ══════════════════════════════════════════════════════════

def group_rects_by_axis(rects, axis=1, tol=20):
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


def group_columns(rects, tol=20):
    rects_sorted = sorted(rects, key=lambda r: r[0])
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


def classify_and_group_columns(columns, hsv_image, hue_iqr_threshold=5.0):
    col_stats = {}
    for i, col_rects in enumerate(columns):
        hue_vals, sat_vals = [], []
        for (x, y, w, h) in col_rects:
            roi = hsv_image[y:y+h, x:x+w]
            if roi.size > 0:
                hue_vals.extend(roi[:, :, 0].flatten().tolist())
                sat_vals.extend(roi[:, :, 1].flatten().tolist())
        col_stats[i] = {
            'sat':     float(np.median(sat_vals)) if sat_vals else 0.0,
            'hue_iqr': float(np.percentile(hue_vals, 75) - np.percentile(hue_vals, 25))
                       if len(hue_vals) >= 4 else 0.0,
        }

    col_types = {}
    for i, s in col_stats.items():
        if s['sat'] < 30.0:
            col_types[i] = 'measure'
        elif s['hue_iqr'] >= hue_iqr_threshold:
            col_types[i] = 'color'
        else:
            col_types[i] = 'measure'

    n = len(columns)
    used = set()
    groups = []
    for i in range(n):
        if col_types[i] != 'measure':
            continue
        grp = []
        if i > 0 and col_types.get(i-1) == 'color' and (i-1) not in used:
            grp.append(i-1); used.add(i-1)
        grp.append(i); used.add(i)
        if i < n-1 and col_types.get(i+1) == 'color' and (i+1) not in used:
            grp.append(i+1); used.add(i+1)
        groups.append({'cols': sorted(grp), 'measure_col': i})
    for i in range(n):
        if i not in used and col_types[i] == 'color':
            groups.append({'cols': [i], 'measure_col': None})
            used.add(i)
    groups.sort(key=lambda g: g['cols'][0])
    return col_types, col_stats, groups


OCR_CORRECTIONS = {
    'HO2': 'H2O2', 'H02': 'H2O2', 'H0²': 'H2O2', 'HO²': 'H2O2',
    'Ho2': 'H2O2', 'h2o2': 'H2O2', 'h2o': 'H2O',
    'Ph':  'pH',   'PH':  'pH',   'ph':  'pH',
    'PhMS': 'PHMB', 'PHMS': 'PHMB', 'Phmb': 'PHMB',
}

def correct_param(text):
    return OCR_CORRECTIONS.get(text, text)


def parse_ocr_result(result):
    blocks = []
    if result is None:
        return blocks
    for item in result:
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


def ocr_roi(ocr_engine, roi, tmp_path='tmp_roi.jpg', scale=3.0):
    h, w = roi.shape[:2]
    enlarged = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray   = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY) if len(enlarged.shape) == 3 else enlarged
    gray   = cv2.equalizeHist(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 4)
    cv2.imwrite(tmp_path, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    result, _ = ocr_engine(tmp_path)
    return parse_ocr_result(result)


def assign_to_row(block, row_positions, row_heights, tol=10):
    cy = block['y'] + block['h'] // 2
    for row_idx, (ry, rh) in enumerate(zip(row_positions, row_heights)):
        if ry - tol <= cy <= ry + rh + tol:
            return row_idx
    return None


def assign_to_group(block, col_groups, col_positions, col_widths):
    cx = block['x'] + block['w'] // 2
    for g_idx, g in enumerate(col_groups):
        x_min = min(col_positions[c] for c in g['cols'])
        x_max = max(col_positions[c] + col_widths[c] for c in g['cols'])
        if x_min <= cx <= x_max:
            return g_idx
    return None


def build_reference(image: np.ndarray, ref_path: str) -> dict:
    """
    Verarbeitet ein Foto des Messindikators:
    Grid erkennen → OCR → reference.json speichern.
    Gibt das reference-Dict zurück.
    """
    print("Erstelle Referenz...")

    # Konturen + Crop
    gray = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5, 200, 200)
    gray_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    canny = cv2.Canny(gray_enh, 30, 120)
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            if 0.1 < w / float(h) < 10:
                rects.append((x, y, w, h))

    if not rects:
        raise RuntimeError("Keine Rechtecke gefunden – Foto zu unscharf oder falsch?")

    margin = 50
    x_min = max(min(r[0] for r in rects) - margin, 0)
    y_min = max(min(r[1] for r in rects) - margin, 0)
    x_max = min(max(r[0]+r[2] for r in rects) + margin, image.shape[1])
    y_max = min(max(r[1]+r[3] for r in rects) + margin, image.shape[0])
    cropped = image[y_min:y_max, x_min:x_max].copy()

    # Grid im Crop – per Hough-Linien robuster als Konturerkennung
    gray_c     = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray_c_enh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray_c)
    canny_c    = cv2.Canny(gray_c_enh, 20, 80)

    # Horizontale + vertikale Linien per HoughLinesP
    lines = cv2.HoughLinesP(canny_c, 1, np.pi/180, threshold=60,
                            minLineLength=cropped.shape[1]//6,
                            maxLineGap=20)

    h_lines, v_lines = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 15:          # horizontal
                h_lines.append((y1+y2)//2)
            elif angle > 75:        # vertikal
                v_lines.append((x1+x2)//2)

    def cluster(vals, tol=15):
        """Gruppiert ähnliche Werte → Median pro Gruppe."""
        if not vals:
            return []
        vals = sorted(vals)
        groups, cur = [], [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] < tol:
                cur.append(v)
            else:
                groups.append(cur)
                cur = [v]
        groups.append(cur)
        return [int(np.median(g)) for g in groups]

    col_positions = cluster(v_lines, tol=40)
    row_positions = cluster(h_lines, tol=30)

    print(f"  Hough-Linien: {len(h_lines)} horizontal, {len(v_lines)} vertikal")
    print(f"  Spalten-Positionen: {col_positions}")
    print(f"  Zeilen-Positionen:  {row_positions}")

    # Fallback auf Konturerkennung wenn Hough zu wenig findet
    if len(col_positions) < 3 or len(row_positions) < 3:
        print("  Hough unzureichend → Fallback auf Konturen")
        contours_c, _ = cv2.findContours(canny_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects_cropped = []
        for cnt in contours_c:
            if cv2.contourArea(cnt) < 200:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                if 0.3 < w/float(h) < 5.0:
                    rects_cropped.append((x, y, w, h))
        rows    = group_rects_by_axis(rects_cropped, axis=1, tol=20)
        columns = group_columns(rects_cropped, tol=20)
        col_positions = [min(r[0] for r in col) for col in columns]
        row_positions = [min(r[1] for r in row) for row in rows]
        col_widths    = [max(r[2] for r in col) for col in columns]
        row_heights   = [max(r[3] for r in row) for row in rows]
    else:
        # Zellengrößen aus Abständen ableiten
        col_positions = sorted(col_positions)
        row_positions = sorted(row_positions)
        col_widths  = [col_positions[i+1] - col_positions[i] - 2
                       for i in range(len(col_positions)-1)] + [60]
        row_heights = [row_positions[i+1] - row_positions[i] - 2
                       for i in range(len(row_positions)-1)] + [40]

    print(f"  Crop-Größe: {cropped.shape[1]}x{cropped.shape[0]}px")
    print(f"  Spalten: {len(col_positions)}  Zeilen: {len(row_positions)}")

    # Dummy columns-Liste für classify_and_group_columns (braucht Rect-Listen pro Spalte)
    columns = [[(cx, row_positions[0], cw, row_heights[0])] for cx, cw in zip(col_positions, col_widths)]

    rects_grid = []
    for ry, rh in zip(row_positions, row_heights):
        for cx, cw in zip(col_positions, col_widths):
            rects_grid.append((cx, ry, cw, rh))

    # grid_top = y-Position der ersten Datenzeile (nicht der Kopfzeile)
    # Kopfbereich = alles oberhalb der zweiten Zeile (erste Zeile = "Phenol Red / mg/l")
    grid_top = row_positions[1] if len(row_positions) > 1 else row_positions[0]
    print(f"  grid_top: {grid_top}px")

    # Spaltenklassifikation
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    col_types, col_stats, col_groups = classify_and_group_columns(columns, hsv)

    # HSV-Stats pro Zelle
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

    # OCR
    print("  OCR läuft...")
    ocr = OpenOCR(backend='onnx', device='cpu')
    grid_top = min(r[1] for r in rects_grid) if rects_grid else 0

    # Kopfbereich pro Gruppe
    group_params = {}
    for g_idx, g in enumerate(col_groups):
        x_min_g = min(col_positions[c] for c in g['cols'])
        x_max_g = max(col_positions[c] + col_widths[c] for c in g['cols'])
        roi = cropped[0:grid_top, x_min_g:x_max_g]
        if roi.size == 0:
            continue
        blocks = ocr_roi(ocr, roi)
        if blocks:
            best = max(blocks, key=lambda b: b['score'])
            group_params[g_idx] = best

    # Messwerte pro Zeile + measure-Spalte
    measure_col_indices = [g['measure_col'] for g in col_groups if g['measure_col'] is not None]
    row_group_values = {}
    for row_idx, (ry, rh) in enumerate(zip(row_positions, row_heights)):
        for mc in measure_col_indices:
            x = col_positions[mc]
            w = col_widths[mc]
            roi = cropped[ry:ry+rh, x:x+w]
            if roi.size == 0:
                continue
            blocks = ocr_roi(ocr, roi)
            # g_idx dieser measure-Spalte
            g_idx = next((gi for gi, g in enumerate(col_groups) if g['measure_col'] == mc), None)
            for b in blocks:
                if re.match(r'^\d+([.,]\d+)?$', b['text'].replace(' ', '')):
                    value = float(b['text'].replace(',', '.'))
                    row_group_values.setdefault(row_idx, {})[g_idx] = value

    param_names = {g_idx: correct_param(b['text']) for g_idx, b in group_params.items()}

    # Referenz aufbauen
    img_path = ref_path.replace('.json', '.jpg')
    cv2.imwrite(img_path, cropped)

    reference = {
        "image_file": img_path,
        "parameters": [],
        "cells": []
    }

    for g_idx, g in enumerate(col_groups):
        param = param_names.get(g_idx)
        if param:
            reference["parameters"].append({
                "name": param,
                "group_idx": g_idx,
                "cols": g['cols'],
                "measure_col": g['measure_col']
            })

    for list_idx, (idx, x, y, w, h, H, S, V) in enumerate(grid_hsv_stats):
        if H is None:
            continue
        cell_col_idx = next((ci for ci, cx in enumerate(col_positions) if abs(cx - x) < col_widths[ci]), None)
        row_idx      = next((ri for ri, (ry, rh) in enumerate(zip(row_positions, row_heights)) if abs(ry - y) < rh), None)
        g_idx        = next((gi for gi, g in enumerate(col_groups) if cell_col_idx in g['cols']), None)
        is_color     = col_types.get(cell_col_idx) == 'color' if cell_col_idx is not None else False
        value        = None
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

    with open(ref_path, 'w', encoding='utf-8') as f:
        json.dump(reference, f, indent=2, ensure_ascii=False)

    n_vals = sum(1 for c in reference["cells"] if c["value"] is not None)
    print(f"  ✅ Referenz gespeichert: {ref_path}")
    print(f"     Parameter: {[p['name'] for p in reference['parameters']]}")
    print(f"     Zellen: {len(reference['cells'])}  |  mit Messwert: {n_vals}")
    return reference


# ══════════════════════════════════════════════════════════
# PHASE 2: Live-Messung (aus main_bounding_orb)
# ══════════════════════════════════════════════════════════

def best_lab_channel(values, labs):
    y = np.array(values, dtype=np.float64)
    best_r, best_ch = 0.0, 1
    for ch in range(3):
        x = np.array([lab[ch] for lab in labs], dtype=np.float64)
        if x.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_ch = r, ch
    return best_ch, best_r, {0: 'L', 1: 'A', 2: 'B'}[best_ch]


def measure_warped(warped: np.ndarray, ref: dict, degree: int = 2) -> dict:
    lab_warped    = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    color_cells   = [c for c in ref['cells'] if     c['is_color_cell'] and c['value'] is not None]
    measure_cells = [c for c in ref['cells'] if not c['is_color_cell']]
    parameters    = {p['name'] for p in ref['parameters']}
    results       = {}

    for param in parameters:
        param_colors  = [c for c in color_cells  if c['parameter'] == param]
        param_measure = [c for c in measure_cells if c['parameter'] == param]
        if not param_colors or not param_measure:
            continue

        labs, vals = [], []
        for cell in param_colors:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            labs.append([float(np.median(roi[:,:,i])) for i in range(3)])
            vals.append(cell['value'])

        if len(labs) < 3:
            continue

        ch_idx, r, ch_name = best_lab_channel(vals, labs)
        x_fit  = np.array([lab[ch_idx] for lab in labs], dtype=np.float64)
        y_fit  = np.array(vals, dtype=np.float64)
        coeffs = np.polyfit(x_fit, y_fit, min(degree, len(x_fit) - 1))
        rmse   = float(np.sqrt(np.mean((np.polyval(coeffs, x_fit) - y_fit) ** 2)))

        probe_ch_vals = []
        for cell in param_measure:
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            roi = lab_warped[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            probe_ch_vals.append(float(np.median(roi[:, :, ch_idx])))

        if not probe_ch_vals:
            continue

        probe_ch = float(np.mean(probe_ch_vals))
        value    = float(np.clip(np.polyval(coeffs, probe_ch),
                                 min(vals), max(vals)))

        results[param] = {
            'value':   round(value, 2),
            'channel': ch_name,
            'r':       round(r, 3),
            'rmse':    round(rmse, 3),
        }

    return results


class StabilityChecker:
    def __init__(self, required_frames=5, max_drift=10.0):
        self.required  = required_frames
        self.max_drift = max_drift
        self.history   = []

    def update(self, M) -> bool:
        if M is None:
            self.history.clear()
            return False
        tx, ty = M[0, 2], M[1, 2]
        self.history.append((tx, ty))
        if len(self.history) < self.required:
            return False
        self.history = self.history[-self.required:]
        txs = [h[0] for h in self.history]
        tys = [h[1] for h in self.history]
        return max(max(txs)-min(txs), max(tys)-min(tys)) < self.max_drift

    def reset(self):
        self.history.clear()

    def progress(self):
        return len(self.history)


def draw_results(frame, results, status=''):
    vis   = frame.copy()
    box_h = 40 + len(results) * 35
    cv2.rectangle(vis, (10, 10), (430, box_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (10, 10), (430, box_h), (80, 80, 80), 1)
    cv2.putText(vis, status, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i, (param, res) in enumerate(sorted(results.items())):
        r     = abs(res.get('r', 0))
        color = (0, 200, 0) if r > 0.95 else (0, 165, 255) if r > 0.85 else (0, 0, 255)
        text  = f"{param}: {res['value']}  [ch={res['channel']} r={res['r']} rmse={res['rmse']}]"
        cv2.putText(vis, text, (18, 55 + i*32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def draw_searching(frame, n_matches, progress, required, roi_rect=None, status=''):
    vis = frame.copy()
    if roi_rect:
        x, y, w, h = roi_rect
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 255), 2)
    cv2.rectangle(vis, (10, 10), (400, 65), (0, 0, 0), -1)
    if n_matches > 0:
        bar_w = int(200 * progress / max(required, 1))
        cv2.rectangle(vis, (18, 44), (18+bar_w, 57), (0, 200, 0), -1)
        cv2.rectangle(vis, (18, 44), (218, 57), (80, 80, 80), 1)
        cv2.putText(vis, f"Erkannt ({n_matches} Matches) – stabilisiere...",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        msg = status if status else "Suche Messindikator..."
        cv2.putText(vis, msg, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)
    return vis


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


def live_measure(cap, ref, scale, stable_frames):
    """Live-Messung mit ORB+SIFT."""
    ref_img  = cv2.imread(ref['image_file'])
    if ref_img is None:
        raise FileNotFoundError(f"Referenzbild nicht gefunden: {ref['image_file']}")
    template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    orb  = cv2.ORB_create(nfeatures=2000)
    sift = cv2.SIFT_create(nfeatures=500)
    kp_orb,  des_orb  = orb.detectAndCompute(template, None)
    kp_sift, des_sift = sift.detectAndCompute(template, None)
    print(f"ORB: {len(kp_orb)} kp   SIFT: {len(kp_sift)} kp")
    print("Kamera aktiv.  [LEERTASTE]=neu messen  [q]=Beenden")

    stability      = StabilityChecker(required_frames=stable_frames)
    frozen_results = None
    frozen_frame   = None
    last_matches   = 0
    last_roi       = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Eingefroren
        if frozen_results is not None:
            vis = draw_results(frozen_frame, frozen_results,
                               status='Messung abgeschlossen  [LEERTASTE]=neu')
            cv2.imshow('Pool Check', vis)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                frozen_results = None
                frozen_frame   = None
                stability.reset()
            continue

        M = None

        # ORB
        kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
        if des_scene is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(bf.match(des_orb, des_scene), key=lambda x: x.distance)
            good    = [m for m in matches if m.distance < 40]

            if len(good) >= 4:
                pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                x, y, w, h = cv2.boundingRect(pts)
                pad = 20
                fh, fw = scene_gray.shape[:2]
                x = max(0, x-pad); y = max(0, y-pad)
                w = min(fw-x, w+2*pad); h = min(fh-y, h+2*pad)
                last_roi     = (x, y, w, h)
                last_matches = len(good)

                roi_gray = scene_gray[y:y+h, x:x+w]
                kp_roi, des_roi = sift.detectAndCompute(roi_gray, None)
                if des_roi is not None and len(kp_roi) >= 4:
                    ms = cv2.BFMatcher().knnMatch(des_sift, des_roi, k=2)
                    gs = [m for m, n in ms if m.distance < 0.75 * n.distance]
                    if len(gs) >= 4:
                        src = np.float32([kp_sift[m.queryIdx].pt for m in gs]).reshape(-1,1,2)
                        dst = np.float32([kp_roi[m.trainIdx].pt  for m in gs]).reshape(-1,1,2)
                        M, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
                        last_matches = len(gs)

        stable = stability.update(M)

        if stable and M is not None:
            x, y, w, h = last_roi
            h_t, w_t   = template.shape
            warped      = cv2.warpPerspective(frame[y:y+h, x:x+w], M, (w_t, h_t))
            results     = measure_warped(warped, ref)
            frozen_results = results
            frozen_frame   = frame.copy()
            print("\n--- Messergebnis ---")
            for param, res in sorted(results.items()):
                print(f"  {param}: {res['value']}  [ch={res['channel']} r={res['r']}]")
        else:
            vis = draw_searching(frame, last_matches,
                                 stability.progress(), stable_frames, last_roi)
            cv2.imshow('Pool Check', vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            stability.reset()
            last_matches = 0
            last_roi     = None


# ══════════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser()
parser.add_argument('--reference', default='reference.json')
parser.add_argument('--camera',    type=int,   default=0)
parser.add_argument('--scale',     type=float, default=0.5)
parser.add_argument('--stable',    type=int,   default=5)
parser.add_argument('--image',     default=None, help='Bild statt Kamera für Referenz verwenden (z.B. reference2.jpg)')
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Kamera {args.camera} nicht verfügbar.")

# Referenz laden oder erstellen
if os.path.exists(args.reference):
    print(f"Referenz gefunden: {args.reference}")
    with open(args.reference, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    print(f"Parameter: {[p['name'] for p in ref['parameters']]}")
elif args.image is not None:
    # Bild direkt aus Datei laden
    photo = cv2.imread(args.image)
    if photo is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {args.image}")
    print(f"Verwende Bild: {args.image} ({photo.shape[1]}x{photo.shape[0]}px)")
    ref = build_reference(photo, args.reference)
else:
    photo = capture_reference_photo(cap, args.scale)
    if photo is None:
        cap.release()
        cv2.destroyAllWindows()
        exit(0)
    ref = build_reference(photo, args.reference)

# Live-Messung
live_measure(cap, ref, args.scale, args.stable)

cap.release()
cv2.destroyAllWindows()