"""Debug cell-cluster detection: show how many cells and AR for each image."""
import cv2, json, glob, numpy as np
from tracker import IndicatorTracker, detect_card_by_cell_cluster, order_quad_corners

ref     = json.load(open('reference.json', encoding='utf-8'))
ref_img = cv2.imread(ref['image_file'])
tmpl    = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker = IndicatorTracker(tmpl)

def count_cells(gray, aspect_ratio):
    h, w       = gray.shape[:2]
    frame_area = h * w
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges    = cv2.Canny(enhanced, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_cell = frame_area * 0.0001
    max_cell = frame_area * 0.02
    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_cell or area > max_cell:
            continue
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        if peri < 1: continue
        approx = cv2.approxPolyDP(hull, 0.08 * peri, True)
        if len(approx) != 4: continue
        rbox = cv2.minAreaRect(approx)
        rw_c, rh_c = rbox[1]
        if min(rw_c, rh_c) < 4: continue
        cell_ar = max(rw_c, rh_c) / (min(rw_c, rh_c) + 1e-6)
        if cell_ar > 4.5: continue
        cx, cy = rbox[0]
        centers.append((cx, cy))

    ar_str = "n/a"
    if len(centers) >= 2:
        pts = np.array(centers, dtype=np.float32)
        x, y, cw, ch = cv2.boundingRect(pts.reshape(-1,1,2).astype(np.int32))
        ar = max(cw, ch) / (min(cw, ch) + 1e-6)
        ar_lo = aspect_ratio * 0.65
        ar_hi = aspect_ratio * 1.35
        ok = ar_lo <= ar <= ar_hi
        ar_str = f"AR={ar:.2f} {'OK' if ok else f'(need {ar_lo:.2f}-{ar_hi:.2f})'}"
    return len(centers), ar_str

images = sorted(glob.glob('raw_*.jpg'))
for p in images:
    frame = cv2.imread(p)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    n, ar = count_cells(gray, tracker.aspect_ratio)
    quad, method = tracker.find(gray)
    status = f"[{method}]" if quad is not None else "MISS"
    print(f"{p}  cells={n:3d}  {ar}  {status}")
