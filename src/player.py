# importing required libraries
import cv2  # OpenCV library
from threading import Thread # library for multi-threading

# defining a helper class for implementing multi-threading
class WebcamStream :

    # initialization method
    def __init__(self, video):
        self.video = video # default is 0 for main camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture("../backgroundVideos/"+video)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # self.vcap.set(cv2.CAP_PROP_FPS, fps_input_stream)

        # reading a single frame from vcap stream for initializing
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False
        self.stopped = True
        # thread instantiation
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background

    # method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return self.grabbed, self.frame

    # method to stop reading frames
    def stop(self):
        self.stopped = True