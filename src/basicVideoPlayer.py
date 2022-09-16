# importing libraries
import cv2
import numpy as np
 
WINDOW_NAME = "Player"

class videoplayer:

    def init(self):
        # Create a VideoCapture object and read from input file
        self.cap = cv2.VideoCapture('../backgroundVideos/1.mp4')
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)/4)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/4)

        print('height is: ' + str(self.h) + '\nwidth is: ' + str(self.w))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow("Player", self.w, self.h)

    def playvideo(self):

        # Check if camera opened successfully
        if (self.cap.isOpened()== False):
            print("Error opening video file")

        # Read until video is completed
        while(self.cap.isOpened()):

        # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
            # Display the resulting frame
                # frame = cv2.resize(frame, (w, h))
                cv2.imshow(WINDOW_NAME, frame)

            # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    self.stopplayer()
                
        # Break the loop
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def stopplayer(self):  
        # When everything done, release
        # the video capture object
        self.cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()