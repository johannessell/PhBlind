"""
test_detection_size.py
======================
Hilfsskript: Erkennungsgrösse diagnostizieren und minimalen Abstand bestimmen.

Zeigt live:
  - Alle Rect-Kandidaten (grau) mit Fläche und Seitenverhältnis
  - Verifizierter Kandidat (grün) mit ORB-Trefferanzahl
  - Fallback Feature-Match (blau)

Bedienung:
  LEERTASTE  – aktuelles Bild speichern als detection_sample.jpg
  q          – beenden
"""

import cv2
import numpy as np
import json

from tracker import IndicatorTracker

# ── Referenz laden ────────────────────────────────────────
ref      = json.load(open('reference.json', encoding='utf-8'))
ref_img  = cv2.imread(ref['image_file'])
template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker  = IndicatorTracker(template)

th, tw = template.shape[:2]
print(f"Template: {tw}x{th}px  Aspect-Ratio: {tracker.aspect_ratio:.2f}")

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Kamera 1 nicht verfügbar.")

print("Messindikator in verschiedenen Abstaenden halten.")
print("  LEERTASTE = Bild speichern   q = Beenden")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape[:2]
    frame_area = fh * fw

    vis = frame.copy()

    # ── Finale Erkennung (Zell-Cluster) ─────────────────
    quad, method = tracker.find(gray)

    if quad is not None:
        card_area = cv2.contourArea(quad.astype(np.int32))
        area_pct  = 100.0 * card_area / frame_area
        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        min_side = min(bw, bh)

        color = (255, 180, 0)
        vis = tracker.draw_quad(vis, quad, color=color)

        info = (f"ERKANNT [{method}]  {bw}x{bh}px  "
                f"Flaeche: {area_pct:.1f}%  Kuerzeste Seite: {min_side}px")
    else:
        info = "NICHT ERKANNT"

    # ── Statuszeile ───────────────────────────────────────
    text_color = (0, 220, 0) if quad is not None else (0, 60, 220)
    cv2.rectangle(vis, (8, 8), (fw - 8, 38), (0, 0, 0), -1)
    cv2.putText(vis, info, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    border_color = (0, 220, 0) if quad is not None else (0, 60, 200)
    cv2.rectangle(vis, (0, 0), (fw - 1, fh - 1), border_color, 3)

    cv2.imshow('Detection-Size Test', vis)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord(' '):
        import glob as _glob, re as _re
        existing = _glob.glob('raw_*.jpg')
        nums = [int(_re.search(r'raw_(\d+)', f).group(1)) for f in existing if _re.search(r'raw_(\d+)', f)]
        n   = (max(nums) + 1) if nums else 1
        out = f'raw_{n:03d}.jpg'        # sauber (für Batch-Tests)
        ann = f'ann_{n:03d}.jpg'        # annotiert (für visuelle Kontrolle)
        cv2.imwrite(out, frame)
        cv2.imwrite(ann, vis)
        print(f"Gespeichert: {out} + {ann}  --  {info}")

cap.release()
cv2.destroyAllWindows()
