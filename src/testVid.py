# import the necessary packages
import numpy as np
import cv2
import time

# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture('../backgroundVideos/2.mp4')

fps = stream.get(cv2.CV_CAP_PROP_FPS)
delay = 1000 / fps

secs = time.time()

while True:
    # loop over frames from the video file stream
    if time.time() - secs < delay:
    	grabbed, frame = stream.read()

        if not grabbed:
    		break
    	cv2.imshow("Frame", frame)
    	cv2.waitKey(1/30)
    else:
        secs = time.time()


    

# do a bit of cleanup
# def stop(self):
stream.release()
cv2.destroyAllWindows()

# Clock.schedule_interval(playVid, 1/30)