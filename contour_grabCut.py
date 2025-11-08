import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('img/PXL_20250808_175205192.MP.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.bilateralFilter(image,5,200,200)

image = cv2.resize(image,None, fx=.5, fy=.5)
# image = cv2.equalizeHist(image)

mask = np.ones(image.shape[:2], np.uint8)

image_size = image.shape[:2]

rect = (500,500,image_size[0]-500,image_size[1]-500)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

result = image * mask2[:,:,np.newaxis]

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

plt.subplot(121),
plt.imshow(image)

plt.subplot(122)
plt.imshow(result)

plt.tight_layout()
plt.show()


# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()