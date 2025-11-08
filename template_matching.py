import cv2
import numpy as np

# Read the images
large_img = cv2.imread('WIN_20250906_21_46_58_Pro.jpg')
template = cv2.imread('template02.jpg')

# Convert images to grayscale
large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Template matching
result = cv2.matchTemplate(large_gray, template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle
top_left = max_loc
h, w = template_gray.shape
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(large_img, top_left, bottom_right, (0,255,0), 2)

# Save or display the image
cv2.imwrite('result.jpg', large_img)