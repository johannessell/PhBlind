import cv2
import numpy as np
import os
print("Current working directory:", os.getcwd())

image1 = cv2.imread('img/sample.jpg')
# image2 = cv2.imread('img/PXL_20250808_175205192.MP.jpg')



filelist = [
'PXL_20250808_175147650.MP.jpg',
'PXL_20250808_175154582.MP.jpg',
'PXL_20250808_175202897.MP.jpg',
'PXL_20250808_175205192.MP.jpg',
'PXL_20250808_175207641.MP.jpg',
'PXL_20250808_175233391.MP.jpg']

idx = 5
image2 = cv2.imread('img/' + filelist[idx])

# image2 = cv2.resize(image2, None, fx=.5, fy=.5)
# cv2.imshow('image1', image1)
# cv2.imshow('image2', image2)
# cv2.waitKey(0)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# init SIFT detector

sift = cv2.SIFT_create()

# detect keypoints and compute descriptors
keypoints1, descriptor1 = sift.detectAndCompute(image1, None)
keypoints2, descriptor2 = sift.detectAndCompute(image2, None)

# create a brute force mathcer
matcher = cv2.BFMatcher()

# match descriptors
matches = matcher.knnMatch(descriptor1, descriptor2, k=2)


# get boundary / boxes

good_matches = []
for m, n in matches:
    if m.distance < .7 * n.distance:
        good_matches.append(m)

# draw matches

if len(good_matches) > 4:  # need at least 4 for homography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Get the bounding box corners of the template
    h, w = image1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

    # Project corners into the scene
    dst = cv2.perspectiveTransform(pts, M)

    # Draw bounding box on scene image
    img2_color = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

    img2_color = cv2.resize(img2_color, None, fx=.25, fy=.25)
    cv2.imshow("Detected", img2_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
# #display the matched images
# cv2.namedWindow('Matched Image',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('Matched Image', matched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

