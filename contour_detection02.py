import cv2
import numpy as np
from matplotlib import pyplot as plt

# -------------------------------
# 1. Bild laden und vorbereiten
# -------------------------------
# image = cv2.imread('WIN_20250907_10_33_19_Pro.jpg')
image = cv2.imread('template02.jpg')
# image = cv2.imread('template03.jpg')

if image is None:
    raise Exception("Bild konnte nicht geladen werden!")

# Graustufen für erste Konturerkennung (roter Kanal)
# gray = image[:, :, 2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#gray = gray[:,:,0]

gray = cv2.bilateralFilter(gray, 5, 200, 200)

# -------------------------------
# 2. CLAHE
# -------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# -------------------------------
# 3. Canny-Kanten
# -------------------------------
canny = cv2.Canny(gray_enhanced, 30, 120)

# -------------------------------
# 4. Konturen finden
# -------------------------------
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------------
# 5. Bilder für Konturen
# -------------------------------
contour_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
original_with_contours = image.copy()

# -------------------------------
# 6. Rechtecke sammeln
# -------------------------------
rects = []
for cnt in contours:
    if cv2.contourArea(cnt) < 1000:
        continue
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # Alle Konturen ins Originalbild
    cv2.drawContours(original_with_contours, [cnt], -1, (0, 0, 255), 2)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 10:
            cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)
            cv2.drawContours(original_with_contours, [approx], -1, (0, 255, 0), 2)
            rects.append((x, y, w, h))

# -------------------------------
# 7. Gemeinsames blaues Rechteck
# -------------------------------
margin = 50
x_min = max(min(r[0] for r in rects) - margin, 0)
y_min = max(min(r[1] for r in rects) - margin, 0)
x_max = min(max(r[0]+r[2] for r in rects) + margin, image.shape[1])
y_max = min(max(r[1]+r[3] for r in rects) + margin, image.shape[0])

cv2.rectangle(
    contour_image,
    (x_min, y_min),
    (x_max, y_max),
    (255, 0, 0), 5
)

# -------------------------------
# 8. Bild auf blaue Box zuschneiden
# -------------------------------
cropped_image = image[y_min:y_max, x_min:x_max].copy()

# -------------------------------
# 9. Erneute Kantenerkennung im zugeschnittenen Bild
# -------------------------------
gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
gray_cropped = cv2.bilateralFilter(gray_cropped, 5, 200, 200)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_cropped_enhanced = clahe.apply(gray_cropped)
canny_cropped = cv2.Canny(gray_cropped_enhanced, 30, 120)
contours_cropped, _ = cv2.findContours(canny_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Rechtecke innerhalb des zugeschnittenen Bildes sammeln
rects_cropped = []
cropped_with_contours = cropped_image.copy()
for cnt in contours_cropped:
    if cv2.contourArea(cnt) < 500:
        continue
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        rects_cropped.append((x, y, w, h))
        cv2.drawContours(cropped_with_contours, approx, -1, (125, 125, 0),  thickness=-1)

# -------------------------------
# 10. Hilfsfunktionen für Grid
# -------------------------------
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

# -------------------------------
# 11. Grid im zugeschnittenen Bild zeichnen
# -------------------------------
if rects_cropped:
    rows = group_rects_by_axis(rects_cropped, axis=1, tol=20)
    for row in rows:
        row.sort(key=lambda r: r[0])

    columns = group_columns(rects_cropped, tol=20)
    col_positions = [min(r[0] for r in col) for col in columns]
    col_widths = [max(r[2] for r in col) for col in columns]
    row_positions = [min(r[1] for r in row) for row in rows]
    row_heights = [max(r[3] for r in row) for row in rows]

    for row_idx, rh in enumerate(row_heights):
        y_pos = row_positions[row_idx]
        for col_idx, cw in enumerate(col_widths):
            x_pos = col_positions[col_idx]
            cv2.rectangle(
                cropped_with_contours,
                (x_pos, y_pos),
                (x_pos + cw, y_pos + rh),
                (0, 255, 255), 2
            )

# -------------------------------
# 12. Ergebnisse anzeigen
# -------------------------------
plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
plt.title("Enhanced Gray Original")
plt.imshow(gray_enhanced, cmap='gray')

plt.subplot(1,3,2)
plt.title("Konturen + Blaue Box")
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))

plt.subplot(1,3,3)
plt.title("Cropped Contours + Grid")
plt.imshow(cv2.cvtColor(cropped_with_contours, cv2.COLOR_BGR2RGB))

# # -------------------------------
# # 13. LAB- und RGB-Kanäle anzeigen
# # -------------------------------
# lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# L, A, B = cv2.split(lab)
# b, g, r = cv2.split(image)
#
# plt.figure(figsize=(15,10))
# plt.subplot(2,3,1)
# plt.title("L (Lightness)")
# plt.imshow(L, cmap='gray')
# plt.subplot(2,3,2)
# plt.title("A (Green-Red)")
# plt.imshow(A, cmap='gray')
# plt.subplot(2,3,3)
# plt.title("B (Blue-Yellow)")
# plt.imshow(B, cmap='gray')
# plt.subplot(2,3,4)
# plt.title("R-Kanal")
# plt.imshow(r, cmap='gray')
# plt.subplot(2,3,5)
# plt.title("G-Kanal")
# plt.imshow(g, cmap='gray')
# plt.subplot(2,3,6)
# plt.title("B-Kanal")
# plt.imshow(b, cmap='gray')
# plt.tight_layout()
plt.show()
