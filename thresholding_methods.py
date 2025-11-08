import os

import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

foldername = 'img/zoom/'

filelist = os.listdir(foldername)

for filename in filelist:
    print(filename)
    image = cv2.imread(os.path.join(foldername, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = abs(image[:, :, 1])#  + abs(image[:,:,2])# Try 0 for L channel too

    image = cv2.bilateralFilter(image, 15, 200, 200)

    # Adaptive threshold
    thresh_adaptive = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 0
    )



    # Otsu threshold
    _, threshold_otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Gradient
    gradient = cv2.morphologyEx(
        image, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
    )

    kernel = np.ones((5, 5), np.uint8)
    thresh_adaptive = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)

    # Contours
    contours_thr_ada, _ = cv2.findContours(thresh_adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_thr_otsu, contour_hierachy = cv2.findContours(threshold_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_grad, _ = cv2.findContours(gradient, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on copies of the original image
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ada_contours_img = img_color.copy()
    otsu_contours_img = img_color.copy()
    grad_contours_img = img_color.copy()

    cv2.drawContours(ada_contours_img, contours_thr_ada, -1, (0, 255, 0), 1)
    cv2.drawContours(otsu_contours_img, contours_thr_otsu, -1, (255, 0, 0), 1)
    cv2.drawContours(grad_contours_img, contours_grad, -1, (0, 0, 255), 1)

    # Show results
    plt.figure(figsize=(12, 8))

    plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('original')
    plt.subplot(232), plt.imshow(thresh_adaptive, cmap='gray'), plt.title('adaptive threshold')
    plt.subplot(233), plt.imshow(threshold_otsu, cmap='gray'), plt.title('otsu threshold')
    plt.subplot(234), plt.imshow(gradient, cmap='gray'), plt.title('gradient')
    plt.subplot(235), plt.imshow(cv2.cvtColor(grad_contours_img, cv2.COLOR_BGR2RGB)), plt.title('Contours (contours_grad)')
    plt.subplot(236), plt.imshow(cv2.cvtColor(otsu_contours_img, cv2.COLOR_BGR2RGB)), plt.title('Contours (Otsu)')

    plt.tight_layout()
    plt.show()

    # break  # remove if you want to process all images
