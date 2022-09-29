from inspect import ismethoddescriptor
import cv2
import numpy as np
import skimage.exposure


backgroundVideo = cv2.VideoCapture("../backgroundVideos/2.mp4")
camera =  cv2.imread("../face.jpg")
mask = cv2.imread("../mask.jpg", cv2.COLOR_BGR2RGB)

# cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

dim = (640, 480)
camera = cv2.resize(camera, dim)
mask = cv2. resize(mask, dim)

# blur threshold image
blur = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)

# stretch so that 255 -> 255 and 127.5 -> 0
# C = A*X+B
# 255 = A*255+B
# 0 = A*127.5+B
# Thus A=2 and B=-127.5
#aa = a*2.0-255.0 does not work correctly, so use skimage
result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))




cv2.imshow("mask", result)
cv2.imshow("cutout", mask)
cv2.imshow("camera", camera)
# print(mask.shape)
_, frame = backgroundVideo.read()
h, w, c = frame.shape
# h = int(h/10)
# w = int(w/10)
# cv2.resizeWindow("video", w, h)
# print(h, w)

while True:
    
    _, frame = backgroundVideo.read()
    if _:
        # cv2.imshow("video",frame)
        cutout = frame.copy()
        camera[mask<=1] = 0
        # camera[mask<=230] = 0
        # cutout[mask==0] = 0

        # camera = cv2.resize(camera, (w, h))
        # mask = np.repeat(cv2.resize(mask, (w,h))[:, :, np.newaxis], 3, axis = 2)
        # mask = cv2.resize(mask, (w,h))
        cv2.imshow("cutout", camera)
        # print(mask.shape)


        # print("camera shape: " + str(cv2.resize(camera, (w, h)).shape))
        # print("mask shape: " + str(cv2.resize(mask, (w, h)).shape))

        # n_frame[mask > 0] = camera[mask] 
        # n_frame[mask==0] = 0
        # cv2.imshow("output", n_frame)
    else:
        backgroundVideo.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
# When everything done, release
# the video capture object
backgroundVideo.release()
 
# Closes all the frames
cv2.destroyAllWindows()