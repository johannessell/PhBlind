import cv2
import numpy as np
import os
import time

print("Current working directory:", os.getcwd())



image1 = cv2.imread('img/sample.jpg')
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# image2 = cv2.imread('img/PXL_20250808_175205192.MP.jpg')



filelist = [
'PXL_20250808_175147650.MP.jpg',
'PXL_20250808_175154582.MP.jpg',
'PXL_20250808_175202897.MP.jpg',
'PXL_20250808_175205192.MP.jpg',
'PXL_20250808_175207641.MP.jpg',
'PXL_20250808_175233391.MP.jpg']

for idx in range(len(filelist)):

    print(f"round {idx+1}")
    image2 = cv2.imread('img/' + filelist[idx])
    image2 = cv2.resize(image2, None, fx=.5, fy=.5)
    # cv2.imshow('image1', image1)
    # cv2.imshow('image2', image2)
    # cv2.waitKey(0)

    # Start timing
    start = time.perf_counter()
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # init SIFT detector

    sift = cv2.SIFT_create()
    # --- ORB ---
    # sift = cv2.ORB_create()

    # detect keypoints and compute descriptors
    keypoints1, descriptor1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(image2_gray, None)

    # create a brute force mathcer
    matcher = cv2.BFMatcher()

    # match descriptors
    matches = matcher.knnMatch(descriptor1, descriptor2, k=2)


    # get boundary / boxes

    good_matches = []
    for m, n in matches:
        if m.distance < .75 * n.distance:
            good_matches.append(m)

    # draw matches

    if len(good_matches) > 4:  # need at least 4 for homography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        M, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, 5.0)

        # Get the bounding box corners of the template
        h, w = image1.shape
        # pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        warped = cv2.warpPerspective(image2, M, (w, h))
        # End timing
        end = time.perf_counter()
        print(f"Execution time: {end - start:.4f} seconds")


        cv2.imshow(f"Warped Region {idx}", warped)
        cv2.imwrite(f'img/{filelist[idx]}_warp', warped)



warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)

# b-Kanal extrahieren
b_channel = lab[:,:,2]

# Canny-Kantenerkennung auf b-Kanal anwenden

# warped_b = cv2.equalizeHist(b_channel)
blurred_b= cv2.GaussianBlur(b_channel, (1, 1), 0)

cv2.imshow("Blurred B", blurred_b)

# Kanten finden (Canny funktioniert meist sehr gut für Farbfelder)
edges = cv2.Canny(blurred_b,5, 30)
cv2.imshow('Edges', edges)

# Konturen finden
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # Filter nach Größe/Füllung, damit kleine "Krümel" ignoriert werden
    area = cv2.contourArea(contour)
    if area > 300:  # Schwelle ggf. anpassen
        x, y, w, h = cv2.boundingRect(contour)
        # Rechtecke optional vergrößern/verkleinern z.B. durch Multiplikation von w/h
        cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow(f"Warped Region bounding {idx}", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()

# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# #display the matched images
# cv2.namedWindow('Matched Image',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Matched Image', matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

