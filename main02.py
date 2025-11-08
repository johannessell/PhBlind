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

template_height, template_width = image1.shape[:2]

result = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

top_left = max_loc

bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

cv2.rectangle(image2, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow('image2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()