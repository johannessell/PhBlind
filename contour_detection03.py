import cv2
import numpy as np
from matplotlib import pyplot as plt
from openocr import OpenOCR
import time

# -------------------------------
# 1. Bild laden
# -------------------------------
image = cv2.imread('template03.jpg')
image = cv2.imread('WIN_20250907_10_33_19_Pro.jpg')
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
canny_cropped = cv2.Canny(gray_cropped_enhanced, 30, 120)
contours_cropped, _ = cv2.findContours(canny_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------------
# 6. Gefüllte Konturen in wrapped eintragen
# -------------------------------
wrapped = np.zeros_like(cropped_image)  # schwarzes Image
rects_cropped = []
for cnt in contours_cropped:
    if cv2.contourArea(cnt) < 500:
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

def group_columns(rects, tol=20):
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
# 9. Overlay auf Cropped-Image
# -------------------------------
alpha = 0.5
overlayed = cv2.addWeighted(cropped_image, 1-alpha, cropped_with_grid, alpha, 0)

# -------------------------------
# 9b. Median HSB-B-Kanal je Spalte
# -------------------------------
# Cropped-Image in HSV umwandeln
hsv_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
B_channel = hsv_cropped[:,:,1]  # "Saturateion"-Kanal (Value)

column_stats = []

if rects_cropped:
    # Gruppiere Rechtecke nach Spalten
    columns = group_columns(rects_cropped, tol=20)

    for col_idx, col in enumerate(columns):
        rect_medians = []
        for (x, y, w, h) in col:
            roi = B_channel[y:y+h, x:x+w]  # B-Kanal im Rechteck
            if roi.size > 0:
                rect_median = np.median(roi)
                rect_medians.append(rect_median)

        if rect_medians:
            min_median = float(np.min(rect_medians))
            max_median = float(np.max(rect_medians))
        else:
            min_median = max_median = None

        column_stats.append((col_idx, min_median, max_median))

# Ausgabe der Ergebnisse pro Spalte
for col_idx, min_median, max_median in column_stats:
    print(f"Spalte {col_idx}: kleinster Median={min_median}, größter Median={max_median}")

## OCR
# # --- Stelle sicher, dass cropped_image schon existiert
ocr = OpenOCR(backend='onnx', device='cpu')  # oder "torch" falls du PyTorch nutzt
ocr_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('OCR', ocr_image)
cv2.imwrite('tmp.jpg', ocr_image)

#
# start_time = time.time()
result, elapse = ocr('tmp.jpg')
# elapsed = time.time() - start_time
#
# print("Erkannter Text im cropped_image:")
# print(result.strip())
# print(f"(Verarbeitungszeit: {elapsed:.2f} s)")
# print("Erkannter Text im cropped_image:")
# print(result.strip())


# -------------------------------
# 10. Ergebnisse anzeigen
# -------------------------------
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title("Original + Konturen")
plt.imshow(cv2.cvtColor(original_with_contours, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,2)
plt.title("Cropped + Gefüllte Rechtecke")
plt.imshow(cv2.cvtColor(wrapped, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,3)
plt.title("Overlay + Grid")
plt.imshow(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))
plt.show()
