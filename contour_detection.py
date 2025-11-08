import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('WIN_20250907_10_33_19_Pro.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bilateralFilter(image,5,200,200)
image = cv2.resize(image,None, fx=.5, fy=.5)
# image = cv2.equalizeHist(image)

canny = cv2.Canny(image,50,150)

_, thresh = cv2.threshold(image, 200,255,cv2.THRESH_BINARY)

contours,_ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_image = image.copy()

cv2.drawContours(contour_image,contours, -1, (0, 255, 0), 3)

# image = cv2.dilate(image, None, iterations=10)
print(image.shape, image.dtype)
if image is None:
    raise Exception("Bild konnte nicht geladen werden!")
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
cv2.imshow('canny', canny)
cv2.imshow('contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()