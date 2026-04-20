"""Debug raw_007 specifically."""
import cv2, json, numpy as np
from tracker import IndicatorTracker, detect_card_by_cell_cluster

ref     = json.load(open('reference.json', encoding='utf-8'))
ref_img = cv2.imread(ref['image_file'])
tmpl    = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker = IndicatorTracker(tmpl)

frame = cv2.imread('raw_007.jpg')
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fh, fw = gray.shape[:2]
print(f"Frame: {fw}x{fh}")

# Check cell cluster (before verify)
blurred  = cv2.GaussianBlur(gray, (5, 5), 0)
clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)
edges    = cv2.Canny(enhanced, 30, 120)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
frame_area = fw*fh
min_cell = frame_area * 0.0001
max_cell = frame_area * 0.02
centers = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_cell or area > max_cell: continue
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

print(f"\nAll cell centers: {len(centers)}")

if centers:
    pts = np.array(centers, dtype=np.float32)
    radius = 80.0
    keep = []
    for i, p in enumerate(pts):
        dists = np.linalg.norm(pts - p, axis=1)
        if int(np.sum(dists < radius)) >= 4:
            keep.append(i)
    print(f"Dense cells (>=3 neighbors in 80px): {len(keep)}")
    if keep:
        dense = pts[keep]
        x, y, cw, ch = cv2.boundingRect(dense.reshape(-1,1,2).astype(np.int32))
        ar = max(cw,ch)/(min(cw,ch)+1e-6)
        print(f"Dense cluster bbox: {cw}x{ch}  AR={ar:.2f}  (need {tracker.aspect_ratio*0.65:.2f}-{tracker.aspect_ratio*1.35:.2f})")
        # Check verify
        quad = np.float32([[x,y],[x+cw,y],[x+cw,y+ch],[x,y+ch]])
        ok = tracker._verify_quad(gray, quad)
        print(f"_verify_quad: {ok}")
