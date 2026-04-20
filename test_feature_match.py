"""
test_feature_match.py
=====================
Batch-Test: ORB und SIFT Feature-Matching auf allen raw_*.jpg Bildern.
Zeigt wie viele Matches und ob eine Homographie gefunden wird.
"""

import cv2
import numpy as np
import json
import glob

ref     = json.load(open('reference.json', encoding='utf-8'))
ref_img = cv2.imread(ref['image_file'])
tmpl    = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
th, tw  = tmpl.shape[:2]
print(f"Template: {tw}x{th}  AR={max(tw,th)/min(tw,th):.2f}")

orb  = cv2.ORB_create(nfeatures=2000)
sift = cv2.SIFT_create(nfeatures=500)

kp_orb,  des_orb  = orb.detectAndCompute(tmpl, None)
kp_sift, des_sift = sift.detectAndCompute(tmpl, None)
print(f"Template keypoints:  ORB={len(kp_orb)}  SIFT={len(kp_sift)}")
print()

tmpl_corners = np.float32([[0,0],[tw,0],[tw,th],[0,th]])

images = sorted(glob.glob('raw_*.jpg'))

for img_path in images:
    frame = cv2.imread(img_path)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape[:2]

    # ── ORB matches ─────────────────────────────────────────
    kp_s, des_s = orb.detectAndCompute(gray, None)
    orb_matches_all  = 0
    orb_matches_good = 0
    if des_s is not None:
        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des_orb, des_s), key=lambda x: x.distance)
        orb_matches_all  = len(matches)
        orb_matches_good = sum(1 for m in matches if m.distance < 40)

    # ── SIFT matches (on full image) ─────────────────────────
    kp_sf, des_sf = sift.detectAndCompute(gray, None)
    sift_good = 0
    sift_homo = False
    quad_ok   = False
    if des_sf is not None and des_sift is not None:
        bf2      = cv2.BFMatcher()
        matches2 = bf2.knnMatch(des_sift, des_sf, k=2)
        good     = [m for m, n in matches2 if m.distance < 0.75 * n.distance]
        sift_good = len(good)
        if len(good) >= 4:
            tp = np.float32([kp_sift[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            sp = np.float32([kp_sf[m.trainIdx].pt   for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(tp, sp, cv2.RANSAC, 5.0)
            if H is not None:
                sift_homo = True
                corners = cv2.perspectiveTransform(tmpl_corners.reshape(1,-1,2), H).reshape(4,2)
                rbox = cv2.minAreaRect(corners.astype(np.float32))
                rw, rh = rbox[1]
                area_frac = (rw * rh) / (fw * fh) * 100
                quad_ok = min(rw,rh) >= 20
                print(f"{img_path}  ORB_all={orb_matches_all:4d} good<40={orb_matches_good:3d}"
                      f"  SIFT_good={sift_good:3d}  homo={'YES' if sift_homo else 'NO '}"
                      f"  size={rw:.0f}x{rh:.0f}px ({area_frac:.1f}%)  quad={'OK' if quad_ok else 'BAD'}")
                continue

    print(f"{img_path}  ORB_all={orb_matches_all:4d} good<40={orb_matches_good:3d}"
          f"  SIFT_good={sift_good:3d}  homo={'YES' if sift_homo else 'NO '}")
