"""
calibrate_cli.py
================
Kommandozeilen-Tool zum Kalibrieren eines neuen Indikator-Typs.
Nutzt reference_core.py als Backend.

Verwendung:
  python calibrate_cli.py --image template03.jpg --name "Lovibond_3in1" --output reference.json
"""

import argparse
import sys
import cv2
from matplotlib import pyplot as plt
from reference_core import (
    load_image, preprocess, find_rects, crop_to_rects,
    build_grid, extract_hsv, suggest_color_columns,
    assign_columns, assign_row_values,
    ReferenceDB, draw_grid
)
from ocr_values import (
    extract_row_values, print_ocr_summary, validate_row_values
)


def parse_float(s: str) -> float | None:
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return None


def run_calibration(image_path: str, name: str, output: str,
                    tol: int = 20, min_area: int = 500,
                    ocr_backend: str = "openocr", debug_ocr: bool = False):

    print(f"\n📷  Lade Bild: {image_path}")
    image = load_image(image_path)
    gray  = preprocess(image)
    rects = find_rects(gray, min_area=min_area)
    print(f"    {len(rects)} Rechtecke im Originalbild")

    cropped, offset = crop_to_rects(image, rects)
    gray_crop       = preprocess(cropped)
    rects_crop      = find_rects(gray_crop, min_area=min_area)
    print(f"    {len(rects_crop)} Rechtecke im Crop")

    layout = build_grid(rects_crop, tol=tol)
    layout = extract_hsv(cropped, layout)
    print(f"    Grid: {layout.n_cols} Spalten × {layout.n_rows} Zeilen")

    # ── Auto-Vorschlag Farbspalten ──
    suggested = suggest_color_columns(layout, saturation_threshold=30.0)
    print(f"\n🎨  Auto-Vorschlag Farbspalten: {suggested}")

    # ── Vorschau anzeigen ──
    preview = draw_grid(cropped, layout)
    plt.figure(figsize=(10, 7))
    plt.title(f"Grid-Vorschau: {layout.n_cols} Spalten × {layout.n_rows} Zeilen")
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)

    # ══════════════════════════════════════════════════════
    # Schritt 1: Farbspalten bestätigen / korrigieren
    # ══════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print("SCHRITT 1: Farbspalten definieren")
    print("═"*55)
    print(f"Spalten 0 bis {layout.n_cols - 1} vorhanden.")
    print(f"Auto-Vorschlag: {suggested}")
    user_input = input("Farbspalten übernehmen? [Enter=Ja / oder eigene eingeben, z.B. '1 2 4 5']: ").strip()

    if user_input:
        try:
            color_cols = [int(x) for x in user_input.split()]
        except ValueError:
            print("⚠️  Ungültige Eingabe – Auto-Vorschlag wird verwendet.")
            color_cols = suggested
    else:
        color_cols = suggested

    print(f"✅  Farbspalten: {color_cols}")

    # ══════════════════════════════════════════════════════
    # Schritt 2: Parameter pro Farbspalte
    # ══════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print("SCHRITT 2: Parameter pro Farbspalte")
    print("═"*55)
    print("Beispiele: pH, Chlor_frei, Chlor_gesamt, PHMB, Alkalinität")
    print("Nicht-Farbspalten werden automatisch als 'leer' gesetzt.\n")

    col_params: dict[int, str] = {}
    for col in range(layout.n_cols):
        if col in color_cols:
            name_input = input(f"  Parameter für Spalte {col}: ").strip()
            col_params[col] = name_input if name_input else f"Parameter_{col}"
        else:
            col_params[col] = "leer"

    print(f"\n✅  Spalten-Zuweisung: {col_params}")

    # ══════════════════════════════════════════════════════
    # Schritt 3: Messwerte per OCR automatisch erkennen
    # ══════════════════════════════════════════════════════
    print("\n" + "═"*55)
    print("SCHRITT 3: Messwerte automatisch per OCR erkennen")
    print("═"*55)
    params_active = [v for v in col_params.values() if v != "leer"]

    row_values = extract_row_values(
        cropped_image=cropped,
        layout=layout,
        col_params=col_params,
        ocr_backend=ocr_backend,
        fallback_manual=True,
        debug=debug_ocr
    )

    print_ocr_summary(row_values, layout.n_rows, params_active)

    # Nutzer kann Korrekturen vornehmen
    validation = validate_row_values(row_values, layout.n_rows, params_active)
    if not validation["complete"]:
        print(f"\n⚠️  {len(validation['missing'])} Werte fehlen noch.")
        fix = input("Weitere Korrekturen vornehmen? [j/N]: ").strip().lower()
        if fix == 'j':
            for (row_idx, param) in validation["missing"]:
                val_input = input(f"  Zeile {row_idx}, {param}: ").strip()
                v = parse_float(val_input)
                if v is not None:
                    row_values.setdefault(row_idx, {})[param] = v

    # ══════════════════════════════════════════════════════
    # Zuweisung & Speichern
    # ══════════════════════════════════════════════════════
    layout = assign_columns(layout, col_params, color_cols)
    layout = assign_row_values(layout, row_values)

    # Finale Vorschau
    preview_final = draw_grid(cropped, layout)
    plt.figure(figsize=(10, 7))
    plt.title("Finale Referenz-Vorschau")
    plt.imshow(cv2.cvtColor(preview_final, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("reference_preview_final.png", dpi=150)
    plt.show()

    ref_db = ReferenceDB(name=name, image_file=image_path, layout=layout)
    ref_db.save(output)

    print(f"\n✅  Referenz gespeichert: {output}")
    print(f"    Parameter: {layout.parameters()}")
    n_valued = sum(1 for c in layout.cells if c.value is not None)
    print(f"    Zellen mit Wert: {n_valued} / {len(layout.cells)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Indikator-Kalibrierung")
    parser.add_argument('--image',    required=True,              help='Pfad zum Referenzbild')
    parser.add_argument('--name',     default='Referenz_01',      help='Name des Indikator-Typs')
    parser.add_argument('--output',   default='reference.json',   help='Ausgabe JSON')
    parser.add_argument('--tol',        type=int, default=20,        help='Grid-Toleranz (px)')
    parser.add_argument('--min-area',   type=int, default=500,       help='Min. Kontur-Fläche (px²)')
    parser.add_argument('--ocr-backend',default='openocr',
                        choices=['openocr','easyocr','tesseract'],   help='OCR-Backend')
    parser.add_argument('--debug-ocr',  action='store_true',         help='OCR-ROIs anzeigen')
    args = parser.parse_args()

    run_calibration(
        image_path=args.image,
        name=args.name,
        output=args.output,
        tol=args.tol,
        min_area=args.min_area,
        ocr_backend=args.ocr_backend,
        debug_ocr=args.debug_ocr
    )
