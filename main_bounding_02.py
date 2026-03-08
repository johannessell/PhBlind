"""
main_bounding_orb.py
====================
Live-Kamera Mess-Modul für Schwimmbad-Messindikator.

Ablauf:
  1. Kamera öffnen, Frames live anzeigen
  2. ORB: grobes Matching → ROI bestimmen
  3. SIFT: präzises Matching auf ROI → Homographie
  4. Stabilitätsprüfung: Indikator über N Frames stabil?
  5. Einfrieren → HSV pro Zelle → gegen reference.json vergleichen
  6. Messwerte anzeigen

Verwendung:
  python main_bounding_orb.py --reference reference.json --camera 0
"""

import cv2
import numpy as np
import os
import time
import json
import argparse

print("Current working directory:", os.getcwd())


# ══════════════════════════════════════════════════════════
# Referenz laden
# ══════════════════════════════════════════════════════════

def load_reference(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_reference_image(ref: dict) -> np.ndarray:
    img = cv2.imread(ref['image_file'])
    if img is None:
        raise FileNotFoundError(f"Referenzbild nicht gefunden: {ref['image_file']}")
    return img


# ══════════════════════════════════════════════════════════
# HSV-Farbvergleich gegen Referenz
# ══════════════════════════════════════════════════════════

def hsv_distance(hsv_a: list, hsv_b: list) -> float:
    """Gewichteter HSV-Abstand, H zirkulär (0-180)."""
    h_diff = abs(hsv_a[0] - hsv_b[0])
    h_diff = min(h_diff, 180.0 - h_diff)
    s_diff = abs(hsv_a[1] - hsv_b[1]) / 255.0 * 90
    v_diff = abs(hsv_a[2] - hsv_b[2]) / 255.0 * 60
    return h_diff * 2.0 + s_diff + v_diff * 0.5

def measure_warped(warped: np.ndarray, ref: dict) -> dict:
    """
    Vergleicht HSV-Werte der color-Zellen im gewarpten Bild
    gegen die Referenz → nächsten Messwert bestimmen.

    Returns:
        {parameter: {'value': float, 'confidence': str, 'distance': float}}
    """
    hsv_warped  = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    color_cells = [c for c in ref['cells'] if c['is_color_cell'] and c['value'] is not None]
    parameters  = {p['name'] for p in ref['parameters']}
    results     = {}

    for param in parameters:
        param_colors = [c for c in color_cells if c['parameter'] == param]
        if not param_colors:
            continue

        candidates = []
        for ref_cell in param_colors:
            x, y, w, h = ref_cell['x'], ref_cell['y'], ref_cell['w'], ref_cell['h']
            roi = hsv_warped[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            measured_hsv = [
                float(np.median(roi[:, :, 0])),
                float(np.median(roi[:, :, 1])),
                float(np.median(roi[:, :, 2]))
            ]
            dist = hsv_distance(measured_hsv, ref_cell['hsv_median'])
            candidates.append({'value': ref_cell['value'], 'distance': dist})

        if not candidates:
            continue

        candidates.sort(key=lambda c: c['distance'])
        best = candidates[0]
        confidence = 'high'   if best['distance'] < 15 else \
                     'medium' if best['distance'] < 35 else 'low'
        results[param] = {
            'value':      best['value'],
            'distance':   round(best['distance'], 1),
            'confidence': confidence
        }

    return results


# ══════════════════════════════════════════════════════════
# Stabilitätsprüfung
# ══════════════════════════════════════════════════════════

class StabilityChecker:
    """Prüft ob der Indikator über N Frames stabil erkannt wird."""
    def __init__(self, required_frames: int = 5, max_drift: float = 10.0):
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
        txs   = [h[0] for h in self.history]
        tys   = [h[1] for h in self.history]
        drift = max(max(txs) - min(txs), max(tys) - min(tys))
        return drift < self.max_drift

    def reset(self):
        self.history.clear()

    def progress(self) -> int:
        return len(self.history)


# ══════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════

CONFIDENCE_COLORS = {
    'high':   (0, 200, 0),
    'medium': (0, 165, 255),
    'low':    (0, 0, 255),
}

def draw_results(frame: np.ndarray, results: dict, status: str = '') -> np.ndarray:
    vis   = frame.copy()
    box_h = 40 + len(results) * 35
    cv2.rectangle(vis, (10, 10), (380, box_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (10, 10), (380, box_h), (80, 80, 80), 1)
    cv2.putText(vis, status, (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i, (param, res) in enumerate(sorted(results.items())):
        color = CONFIDENCE_COLORS.get(res['confidence'], (255, 255, 255))
        text  = f"{param}: {res['value']}  [{res['confidence']}  d={res['distance']}]"
        cv2.putText(vis, text, (18, 55 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return vis

def draw_searching(frame: np.ndarray, n_matches: int,
                   progress: int, required: int,
                   roi_rect=None) -> np.ndarray:
    vis = frame.copy()
    if roi_rect:
        x, y, w, h = roi_rect
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 255), 2)
    cv2.rectangle(vis, (10, 10), (380, 65), (0, 0, 0), -1)
    if n_matches > 0:
        bar_w = int(200 * progress / max(required, 1))
        cv2.rectangle(vis, (18, 44), (18 + bar_w, 57), (0, 200, 0), -1)
        cv2.rectangle(vis, (18, 44), (218, 57), (80, 80, 80), 1)
        cv2.putText(vis, f"Erkannt ({n_matches} Matches) - stabilisiere...",
                    (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv2.putText(vis, "Suche Messindikator...",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)
    return vis


# ══════════════════════════════════════════════════════════
# Hauptprogramm
# ══════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Live-Kamera Messung')
parser.add_argument('--reference', default='reference.json')
parser.add_argument('--camera',    type=int,   default=0)
parser.add_argument('--scale',     type=float, default=0.5)
parser.add_argument('--stable',    type=int,   default=5)
args = parser.parse_args()

# Referenz laden
print(f"Lade Referenz: {args.reference}")
ref      = load_reference(args.reference)
ref_img  = load_reference_image(ref)
template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
print(f"Parameter: {[p['name'] for p in ref['parameters']]}")

# Detektoren einmalig initialisieren
orb  = cv2.ORB_create(nfeatures=2000)
sift = cv2.SIFT_create(nfeatures=500)
kp_template_orb,  des_template_orb  = orb.detectAndCompute(template, None)
kp_template_sift, des_template_sift = sift.detectAndCompute(template, None)
print(f"ORB-Keypoints: {len(kp_template_orb)}  SIFT-Keypoints: {len(kp_template_sift)}")

# Kamera öffnen
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Kamera {args.camera} nicht verfügbar.")
print("Kamera geöffnet.  [q]=Beenden  [r]=Zurücksetzen")

stability      = StabilityChecker(required_frames=args.stable, max_drift=10.0)
frozen_results = None
frozen_frame   = None
last_matches   = 0
last_roi       = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if args.scale != 1.0:
        frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

    scene_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Eingefroren: Ergebnis weiter anzeigen ──
    if frozen_results is not None:
        vis = draw_results(frozen_frame, frozen_results,
                           status='Messung abgeschlossen  [r]=neu messen')
        cv2.imshow('Schwimmbad Messung', vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            frozen_results = None
            frozen_frame   = None
            stability.reset()
        continue

    start_time = time.perf_counter()
    M = None

    # --- Schritt 1: ORB grobes Matching ---
    kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)
    if des_scene is not None:
        bf          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches     = bf.match(des_template_orb, des_scene)
        matches     = sorted(matches, key=lambda x: x.distance)
        good_matches = [m for m in matches if m.distance < 40]

        if len(good_matches) >= 4:
            # --- Schritt 2: ROI aus ORB Matches ---
            pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            pad = 20
            fh, fw = scene_gray.shape[:2]
            x = max(0, x - pad);  y = max(0, y - pad)
            w = min(fw - x, w + 2*pad);  h = min(fh - y, h + 2*pad)
            last_roi   = (x, y, w, h)
            last_matches = len(good_matches)

            roi_scene = scene_gray[y:y+h, x:x+w]

            # --- Schritt 3: Präzises Matching mit SIFT ---
            kp_roi, des_roi = sift.detectAndCompute(roi_scene, None)
            if des_roi is not None and len(kp_roi) >= 4:
                bf_sift      = cv2.BFMatcher()
                matches_sift = bf_sift.knnMatch(des_template_sift, des_roi, k=2)
                good_sift    = [m for m, n in matches_sift if m.distance < 0.75 * n.distance]

                if len(good_sift) >= 4:
                    # --- Schritt 4: Homography + RANSAC ---
                    src_pts = np.float32([kp_template_sift[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_roi[m.trainIdx].pt           for m in good_sift]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    last_matches = len(good_sift)

    end_time = time.perf_counter()

    stable = stability.update(M)

    if stable and M is not None:
        # --- Schritt 5: Warp auf Template-Größe ---
        x, y, w, h   = last_roi
        roi_color     = frame[y:y+h, x:x+w]
        h_t, w_t      = template.shape
        warped        = cv2.warpPerspective(roi_color, M, (w_t, h_t))

        results = measure_warped(warped, ref)

        frozen_results = results
        frozen_frame   = frame.copy()

        print(f"\n--- Messergebnis ({end_time - start_time:.3f}s) ---")
        for param, res in sorted(results.items()):
            print(f"  {param}: {res['value']}  [{res['confidence']}  d={res['distance']}]")

    else:
        vis = draw_searching(frame, last_matches,
                             stability.progress(), args.stable,
                             last_roi)
        cv2.imshow('Schwimmbad Messung', vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        stability.reset()
        last_matches = 0
        last_roi     = None

cap.release()
cv2.destroyAllWindows()