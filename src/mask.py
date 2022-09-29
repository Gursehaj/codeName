import cv2
import skimage.exposure

import numpy as np

mask = cv2.imread("../edgeMask.png")

# blur threshold image
blur = cv2.GaussianBlur(mask, (0,0), sigmaX=4, sigmaY=4, borderType = cv2.BORDER_DEFAULT)
print(np.unique(blur))
# stretch so that 255 -> 255 and 127.5 -> 0
# C = A*X+B
# 255 = A*255+B
# 0 = A*127.5+B
# Thus A=2 and B=-127.5
#aa = a*2.0-255.0 does not work correctly, so use skimage
# result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
cv2.imshow("OG", mask)
cv2.imshow("result", blur)
cv2.waitKey(0)