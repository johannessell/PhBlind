import cv2
import matplotlib.pyplot as plt

filter_type = "bilateral"   # options: "gaussian", "median", "bilateral", "none"
# Read image
img = cv2.imread("template02.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("WIN_20250906_21_46_58_Pro.jpg", cv2.IMREAD_GRAYSCALE)

# --- Apply filter based on choice ---
if filter_type == "gaussian":
    filtered = cv2.GaussianBlur(img, (5, 5), 0)
elif filter_type == "median":
    filtered = cv2.medianBlur(img, 5)
elif filter_type == "bilateral":
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
else:  # no filtering
    filtered = img

# Initialize ORB detector
orb = cv2.SIFT_create(10000)

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)

# Draw keypoints on the image
img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

# Show the result with matplotlib
plt.figure(figsize=(8,6))
plt.imshow(img_keypoints)
plt.axis("off")
plt.show()