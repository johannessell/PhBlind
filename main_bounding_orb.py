"""
main_bounding_orb.py
====================
Live-Kamera Mess-Modul für Schwimmbad-Messindikator.

Ablauf:
  1. Kamera öffnen, Frames live anzeigen
  2. tracker.find():  Rechteck-Detektion (primär) oder ORB+SIFT (Fallback)
                      → 4 Eckpunkte des Messindikators im Frame
  3. QuadStabilityChecker: Indikator über N Frames stabil genug?
  4. Einfrieren → Perspektivwarp → LAB-Farbmessung pro Zelle
  5. Polynomial-Fit → Messwerte ausgeben

Verwendung:
  python main_bounding_orb.py --reference reference.json --camera 0
"""

import cv2
import numpy as np
import os
import json
import argparse

from tracker import IndicatorTracker, QuadStabilityChecker

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
# Farbmessung gegen Referenz
# ══════════════════════════════════════════════════════════

def best_lab_channel(values: list, labs: list) -> tuple:
    """
    Bestimmt welcher LAB-Kanal (L=0, A=1, B=2) am stärksten
    mit den Messwerten korreliert.

    Returns:
        (channel_idx, correlation, channel_name)
    """
    y = np.array(values, dtype=np.float64)
    best_r, best_ch = 0.0, 1
    for ch in range(3):
        x = np.array([lab[ch] for lab in labs], dtype=np.float64)
        if x.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > abs(best_r):
            best_r, best_ch = r, ch
    names = {0: 'L', 1: 'A', 2: 'B'}
    return best_ch, best_r, names[best_ch]


def fit_poly(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    return np.polyfit(x, y, min(degree, len(x) - 1))


def measure_warped(warped: np.ndarray, ref: dict, degree: int = 2) -> dict:
    """
    Pro Parameter:
      1. LAB jeder color-Zelle aus dem gewarpten Bild lesen
      2. Besten LAB-Kanal per Korrelation bestimmen
      3. Polynomialer Fit: Wert = f(Kanal) über alle color-Zellen
      4. LAB der Messprobe aus measure-Zellen → Wert aus Fit

    Returns:
        {parameter: {'value': float, 'channel': str, 'r': float, 'rmse': float}}
    """
    lab_warped    = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    color_cells   = [c for c in ref['cells'] if     c['is_color_cell'] and c['value'] is not None]
    measure_cells = [c for c in ref['cells'] if not c['is_color_cell']]
    param_meta    = {p['name']: p for p in ref['parameters']}
    name_to_ch    = {'L': 0, 'A': 1, 'B': 2}
    results       = {}

    for param, meta in param_meta.items():
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
            labs.append([
                float(np.median(roi[:, :, 0])),
                float(np.median(roi[:, :, 1])),
                float(np.median(roi[:, :, 2]))
            ])
            vals.append(cell['value'])

        if len(labs) < 3:
            continue

        fixed_ch   = meta.get('best_channel')
        ref_coeffs = meta.get('poly_coeffs')

        if fixed_ch in name_to_ch and ref_coeffs is not None:
            # Stabiler Pfad:
            #  1) Polynom (value=f(ch)) stammt aus dem Referenzbild.
            #  2) Pro Target-Bild: lineare Transformation Target->Ref auf
            #     dem Kanal fitten (paired swatches).
            #  3) Sample in Ref-Koordinaten transformieren, dann Ref-Poly
            #     auswerten.
            ch_idx  = name_to_ch[fixed_ch]
            ch_name = fixed_ch

            ref_ch = []
            tgt_ch = []
            y_arr  = []
            for cell, tgt_lab in zip(param_colors, labs):
                ref_lab = cell.get('lab_median')
                if ref_lab is None or ref_lab[ch_idx] is None:
                    continue
                ref_ch.append(ref_lab[ch_idx])
                tgt_ch.append(tgt_lab[ch_idx])
                y_arr.append(cell['value'])
            ref_ch = np.array(ref_ch, dtype=np.float64)
            tgt_ch = np.array(tgt_ch, dtype=np.float64)
            y_arr  = np.array(y_arr,  dtype=np.float64)

            if len(ref_ch) < 3 or tgt_ch.std() < 1e-6:
                continue

            # r auf dem Target (Qualitaetsmass pro Bild)
            r = float(np.corrcoef(tgt_ch, y_arr)[0, 1])

            # Warp-Qualitaet: Zielkorrelation muss Vorzeichen und genug
            # Staerke gegen die Referenz zeigen, sonst ist die Zell-
            # registrierung zu stark verzerrt und die Messung nicht
            # vertrauenswuerdig -> ueberspringen.
            ref_r = meta.get('best_r') or 0.0
            MIN_ABS_R = 0.70
            if ref_r != 0 and (np.sign(r) != np.sign(ref_r) or abs(r) < MIN_ABS_R):
                continue

            # Linearer Fit Target->Ref
            t2r = np.polyfit(tgt_ch, ref_ch, 1)
            coeffs = np.array(ref_coeffs, dtype=np.float64)

            pred_ref = np.polyval(t2r, tgt_ch)
            rmse = float(np.sqrt(np.mean(
                (np.polyval(coeffs, pred_ref) - y_arr) ** 2)))
        else:
            ch_idx, r, ch_name = best_lab_channel(vals, labs)
            x_fit  = np.array([l[ch_idx] for l in labs], dtype=np.float64)
            y_fit  = np.array(vals,         dtype=np.float64)
            coeffs = fit_poly(x_fit, y_fit, degree)
            rmse   = float(np.sqrt(np.mean((np.polyval(coeffs, x_fit) - y_fit) ** 2)))
            t2r    = None

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
        if t2r is not None:
            probe_ch = float(np.polyval(t2r, probe_ch))
        value    = float(np.polyval(coeffs, probe_ch))
        ref_vals = sorted(vals)
        value    = float(np.clip(value, ref_vals[0], ref_vals[-1]))

        results[param] = {
            'value':   round(value, 2),
            'channel': ch_name,
            'r':       round(r, 3),
            'rmse':    round(rmse, 3),
        }

    return results


# ══════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════

def draw_results(frame: np.ndarray, results: dict, status: str = '') -> np.ndarray:
    vis   = frame.copy()
    box_h = 40 + len(results) * 35
    cv2.rectangle(vis, (10, 10), (460, box_h), (0, 0, 0), -1)
    cv2.rectangle(vis, (10, 10), (460, box_h), (80, 80, 80), 1)
    cv2.putText(vis, status, (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    for i, (param, res) in enumerate(sorted(results.items())):
        r     = abs(res.get('r', 0))
        color = (0, 200, 0) if r > 0.95 else (0, 165, 255) if r > 0.85 else (0, 0, 255)
        text  = f"{param}: {res['value']}  [ch={res['channel']} r={res['r']} rmse={res['rmse']}]"
        cv2.putText(vis, text, (18, 55 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def draw_searching(
    frame: np.ndarray,
    method: str,
    progress: int,
    required: int,
) -> np.ndarray:
    vis = frame.copy()
    cv2.rectangle(vis, (10, 10), (400, 72), (0, 0, 0), -1)
    if method != 'none':
        bar_w = int(200 * progress / max(required, 1))
        cv2.rectangle(vis, (18, 48), (18 + bar_w, 62), (0, 200, 0), -1)
        cv2.rectangle(vis, (18, 48), (218, 62), (80, 80, 80), 1)
        label = 'Rechteck' if method == 'rect' else 'Features'
        cv2.putText(vis, f"Erkannt ({label}) \u2014 stabilisiere...",
                    (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv2.putText(vis, "Suche Messindikator...",
                    (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 180, 255), 1)
    return vis


# ══════════════════════════════════════════════════════════
# Hauptprogramm
# ══════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live-Kamera Messung')
    parser.add_argument('--reference', default='reference.json')
    parser.add_argument('--camera',    type=int,   default=0)
    parser.add_argument('--scale',     type=float, default=0.5)
    parser.add_argument('--stable',    type=int,   default=5,
                        help='Anzahl stabiler Frames vor Messung')
    args = parser.parse_args()

    print(f"Lade Referenz: {args.reference}")
    ref      = load_reference(args.reference)
    ref_img  = load_reference_image(ref)
    template = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    print(f"Parameter: {[p['name'] for p in ref['parameters']]}")

    tracker   = IndicatorTracker(template)
    stability = QuadStabilityChecker(required_frames=args.stable, max_drift=15.0)
    print(f"Aspect-Ratio: {tracker.aspect_ratio:.2f}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {args.camera} nicht verfügbar.")
    print("Kamera geöffnet.  [q]=Beenden  [r]=Zurücksetzen")

    frozen_results = None
    frozen_frame   = None
    last_quad      = None
    last_method    = 'none'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

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
                last_quad   = None
                last_method = 'none'
            continue

        scene_gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_quad, last_method = tracker.find(scene_gray)
        stable              = stability.update(last_quad)

        if stable:
            mean_quad = stability.mean_quad()
            warped    = tracker.warp(frame, mean_quad)
            results   = measure_warped(warped, ref)

            frozen_frame   = tracker.draw_quad(frame, mean_quad)
            frozen_results = results

            print(f"\n--- Messergebnis [{last_method}] ---")
            for param, res in sorted(results.items()):
                print(f"  {param}: {res['value']}  "
                      f"[ch={res['channel']} r={res['r']} rmse={res['rmse']}]")

        else:
            vis = draw_searching(frame, last_method, stability.progress(), args.stable)
            if last_quad is not None:
                color = (0, 220, 0) if last_method == 'rect' else (0, 165, 255)
                vis   = tracker.draw_quad(vis, last_quad, color=color)
            cv2.imshow('Schwimmbad Messung', vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            stability.reset()
            last_quad   = None
            last_method = 'none'

    cap.release()
    cv2.destroyAllWindows()
