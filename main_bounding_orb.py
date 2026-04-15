import cv2
import numpy as np
import os
import time

print("Current working directory:", os.getcwd())

# Template laden
template = cv2.imread('img/sample.jpg', cv2.IMREAD_GRAYSCALE)

# Liste der Szenenbilder
filelist = [
    'PXL_20250808_175147650.MP.jpg',
    'PXL_20250808_175154582.MP.jpg',
    'PXL_20250808_175202897.MP.jpg',
    'PXL_20250808_175205192.MP.jpg',
    'PXL_20250808_175207641.MP.jpg',
    'PXL_20250808_175233391.MP.jpg'
]

for idx, fname in enumerate(filelist):
    print(f"\nRound {idx+1}: {fname}")

    # Szene laden & skalieren
    scene = cv2.imread('img/' + fname)
    scene = cv2.resize(scene, None, fx=0.5, fy=0.5)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    start_time = time.perf_counter()

    # --- Schritt 1: ORB grobes Matching ---
    orb = cv2.ORB_create(nfeatures=2000)
    kp_template, des_template = orb.detectAndCompute(template, None)
    kp_scene, des_scene = orb.detectAndCompute(scene_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des_scene)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 40]

    if len(good_matches) < 4:
        print(f"Nicht genug ORB-Matches ({len(good_matches)}/4), überspringe Bild")
        continue

    # --- Schritt 2: ROI aus ORB Matches ---
    pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    x, y, w, h = cv2.boundingRect(pts)
    roi_scene = scene_gray[y:y+h, x:x+w]

    # --- Schritt 3: Präzises Matching mit SIFT ---
    sift = cv2.SIFT_create(nfeatures=500)
    kp_roi, des_roi = sift.detectAndCompute(roi_scene, None)
    kp_temp_sift, des_temp_sift = sift.detectAndCompute(template, None)

    bf_sift = cv2.BFMatcher()
    matches_sift = bf_sift.knnMatch(des_temp_sift, des_roi, k=2)

    # Lowe's ratio test
    good_sift = []
    for m, n in matches_sift:
        if m.distance < 0.75 * n.distance:
            good_sift.append(m)

    if len(good_sift) < 4:
        print(f"Nicht genug SIFT-Matches ({len(good_sift)}/4), überspringe Bild")
        continue

    # --- Schritt 4: Homography + RANSAC ---
    src_pts = np.float32([kp_temp_sift[m.queryIdx].pt for m in good_sift]).reshape(-1,1,2)
    dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good_sift]).reshape(-1,1,2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None:
        print("Homography konnte nicht berechnet werden")
        continue

    # --- Schritt 5: Warp auf Template-Größe ---
    h_t, w_t = template.shape
    warped = cv2.warpPerspective(roi_scene, M, (w_t, h_t))

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.4f} sec")

    # --- Ergebnis anzeigen ---
    cv2.imshow(f"Warped Hybrid ORB+SIFT {idx+1}", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()