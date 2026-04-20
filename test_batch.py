"""Batch test: run tracker.find() on all raw_*.jpg and report results."""
import cv2, json, glob
from tracker import IndicatorTracker, detect_card_by_cell_cluster

ref     = json.load(open('reference.json', encoding='utf-8'))
ref_img = cv2.imread(ref['image_file'])
tmpl    = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
tracker = IndicatorTracker(tmpl)

images = sorted(glob.glob('raw_*.jpg'))
found  = 0
for p in images:
    frame = cv2.imread(p)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape[:2]

    quad, method = tracker.find(gray)
    if quad is not None:
        found += 1
        area = cv2.contourArea(quad.astype('int32'))
        pct  = 100.0 * area / (fw * fh)
        x, y, bw, bh = cv2.boundingRect(quad.astype('int32'))
        print(f"  OK  {p}  [{method}]  {bw}x{bh}px  {pct:.1f}%")
    else:
        print(f"  --  {p}  not found")

print(f"\n{found}/{len(images)} detected")
