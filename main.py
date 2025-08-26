import cv2
import numpy as np
import os
print("Current working directory:", os.getcwd())

image1 = cv2.imread('img/sample.jpg')
image2 = cv2.imread('img/PXL_20250808_175205192.MP.jpg')



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
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# draw matches
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#display the matched images
cv2.namedWindow('Matched Image',cv2.WINDOW_AUTOSIZE)
cv2.imshow('Matched Image', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

