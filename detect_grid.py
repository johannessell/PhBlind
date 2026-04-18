"""
detect_grid.py
==============
Visualisiert die Grid-Erkennung und erlaubt Korrektur der OCR-Werte.

Verwendung:
  python detect_grid.py --image reference2.jpg
"""

import cv2
import numpy as np
import argparse
import json
import re
from openocr import OpenOCR


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='reference2.jpg')
args = parser.parse_args()

# ──────────────────────────────────────────
# 1. Bild laden
# ──────────────────────────────────────────
image = cv2.imread(args.image)
if image is None:
    raise FileNotFoundError(f"Bild nicht gefunden: {args.image}")
print(f"Bild: {args.image}  ({image.shape[1]}x{image.shape[0]}px)")

# ──────────────────────────────────────────
# 2. Großes Rechteck (Bounding Box) erkennen
# ──────────────────────────────────────────
gray     = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5, 200, 200)
gray_enh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
canny    = cv2.Canny(gray_enh, 30, 120)
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

margin = 50
x_min = max(min(r[0] for r in rects) - margin, 0)
y_min = max(min(r[1] for r in rects) - margin, 0)
x_max = min(max(r[0]+r[2] for r in rects) + margin, image.shape[1])
y_max = min(max(r[1]+r[3] for r in rects) + margin, image.shape[0])

# Großes Rechteck anzeigen
vis_bbox = image.copy()
cv2.rectangle(vis_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
cv2.putText(vis_bbox, f"Bounding Box: {x_max-x_min}x{y_max-y_min}px",
            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# cv2.imshow('1. Bounding Box', vis_bbox)
# cv2.waitKey(0)

# ──────────────────────────────────────────
# 3. Crop + Canny-Grid
# ──────────────────────────────────────────
cropped  = image[y_min:y_max, x_min:x_max].copy()
gray_c     = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
gray_c_enh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray_c)
canny_c    = cv2.Canny(gray_c_enh, 50, 120)

cv2.imshow("Canny", canny_c)
cv2.waitKey(0)

kernel = np.ones((5,5), np.uint8)
canny = cv2.dilate(canny_c, kernel, iterations=1)
canny = cv2.erode(canny, kernel, iterations=1)


cv2.imshow("Canny closing 5", canny)
cv2.waitKey(0)

contours_c, _ = cv2.findContours(canny_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_image = np.zeros_like(cropped)

cv2.drawContours(contour_image, contours_c, -1, (0, 255, 0), 2)
cv2.imshow('canny ergebnis', contour_image)


rects_cropped = []

cv2.destroyAllWindows()
for cnt in contours_c:
    x, y, w, h = cv2.boundingRect(cnt)
    ratio = w / float(h)
    area = w*h
    if 0.3 <ratio < 5.0 and area > 500:
       rects_cropped.append((x, y, w, h))

print(f"Canny: {len(rects_cropped)} Rechtecke gefunden")

contour_image = np.zeros_like(cropped)
for (x, y, w, h) in rects_cropped:
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('cropped ergebnis', contour_image)



# Zeilen und Spalten gruppieren
def group_by_axis(rects, axis, tol=20):
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

rows    = group_by_axis(rects_cropped, axis=1, tol=20)
columns = group_by_axis(rects_cropped, axis=0, tol=20)

col_pos     = sorted([min(r[0] for r in col) for col in columns])
col_widths  = [max(r[2] for r in col) for col in sorted(columns, key=lambda c: min(r[0] for r in c))]
row_pos     = sorted([min(r[1] for r in row) for row in rows])
row_heights = [max(r[3] for r in row) for row in sorted(rows, key=lambda rw: min(r[1] for r in rw))]

print(f"Spalten ({len(col_pos)}): {col_pos}")
print(f"Zeilen  ({len(row_pos)}): {row_pos}")

# Grid visualisieren
vis_grid = cropped.copy()
for lx in col_pos:
    cv2.line(vis_grid, (lx, 0), (lx, cropped.shape[0]), (0, 255, 255), 1)
for ly in row_pos:
    cv2.line(vis_grid, (0, ly), (cropped.shape[1], ly), (0, 255, 255), 1)

# Kleine Rechtecke einzeichnen
rects_grid = []
for ri, (ry, rh) in enumerate(zip(row_pos, row_heights)):
    for ci, (cx, cw) in enumerate(zip(col_pos, col_widths)):
        rects_grid.append((ci, ri, cx, ry, cw, rh))
        cv2.rectangle(vis_grid, (cx, ry), (cx+cw, ry+rh), (0, 200, 0), 1)
        cv2.putText(vis_grid, f"{ci},{ri}", (cx+3, ry+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

cv2.imshow('2. Erkannte Zellen', vis_grid)
cv2.waitKey(0)

# ──────────────────────────────────────────
# 4. OCR auf measure-Zellen
# ──────────────────────────────────────────
print("\nOCR läuft...")
ocr = OpenOCR(backend='onnx', device='cpu')

def ocr_roi(roi, tmp_path='tmp_roi.jpg', scale=3.0):
    h, w = roi.shape[:2]
    if h < 5 or w < 5:
        return []
    enlarged = cv2.resize(roi, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    gray_r   = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray_r   = cv2.equalizeHist(gray_r)
    binary   = cv2.adaptiveThreshold(gray_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 4)
    cv2.imwrite(tmp_path, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
    result, _ = ocr(tmp_path)
    if result is None:
        return []
    blocks = []
    for item in result:
        json_str = item.split('\t')[-1].strip()
        try:
            for det in json.loads(json_str):
                text = det.get('transcription', '').strip()
                if text:
                    blocks.append({'text': text, 'score': det.get('score', 0)})
        except:
            pass
    return blocks

# Kopfzeile (oberhalb erster Zeile)
grid_top = row_pos[1] if len(row_pos) > 1 else row_pos[0]
header_roi = cropped[0:grid_top, :]
header_blocks = ocr_roi(header_roi)
print(f"\nKopfzeile: {[b['text'] for b in header_blocks]}")

# Jede Zelle einzeln
cell_texts = {}  # {(col, row): text}
for ci, (cx, cw) in enumerate(zip(col_pos, col_widths)):
    for ri, (ry, rh) in enumerate(zip(row_pos, row_heights)):
        roi = cropped[ry:ry+rh, cx:cx+cw]
        blocks = ocr_roi(roi)
        if blocks:
            best = max(blocks, key=lambda b: b['score'])
            text = best['text']
            if re.match(r'^\d+([.,]\d+)?$', text.replace(' ', '')):
                cell_texts[(ci, ri)] = text
                print(f"  Zelle ({ci},{ri}): '{text}'  score={best['score']:.2f}")

# ──────────────────────────────────────────
# 5. Erkannte Zahlen visualisieren
# ──────────────────────────────────────────
vis_ocr = cropped.copy()
for (ci, ri), text in cell_texts.items():
    cx, cw = col_pos[ci], col_widths[ci]
    ry, rh = row_pos[ri], row_heights[ri]
    cv2.rectangle(vis_ocr, (cx, ry), (cx+cw, ry+rh), (0, 255, 0), 2)
    cv2.putText(vis_ocr, text, (cx+3, ry+rh-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

cv2.imshow('3. Erkannte Zahlen', vis_ocr)
cv2.waitKey(0)

# ──────────────────────────────────────────
# 6. Manuelle Korrektur
# ──────────────────────────────────────────
print("\n--- Manuelle Korrektur ---")
print("Format: col,row,wert  (z.B. '0,3,7.6')")
print("Leer lassen und Enter drücken zum Beenden.")

while True:
    inp = input("Korrektur: ").strip()
    if not inp:
        break
    parts = inp.split(',')
    if len(parts) != 3:
        print("  Format: col,row,wert")
        continue
    try:
        ci, ri, val = int(parts[0]), int(parts[1]), parts[2].strip()
        cell_texts[(ci, ri)] = val
        print(f"  ✅ ({ci},{ri}) = '{val}'")
    except ValueError:
        print("  Ungültige Eingabe.")

print("\nFinal erkannte Werte:")
for (ci, ri), text in sorted(cell_texts.items()):
    print(f"  ({ci},{ri}): {text}")

cv2.destroyAllWindows()