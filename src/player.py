# # importing required libraries
# import cv2  # OpenCV library
# from threading import Thread # library for multi-threading

# # defining a helper class for implementing multi-threading
# class WebcamStream :

#     # initialization method
#     def __init__(self, video):
#         self.video = video # default is 0 for main camera

#         # opening video capture stream
#         self.vcap = cv2.VideoCapture("../backgroundVideos/"+video)
#         if self.vcap.isOpened() is False :
#             print("[Exiting]: Error accessing webcam stream.")
#             exit(0)
#         fps_input_stream = int(self.vcap.get(5)) # hardware fps
#         print("FPS of input stream: {}".format(fps_input_stream))

#         # self.vcap.set(cv2.CAP_PROP_FPS, fps_input_stream)

#         # reading a single frame from vcap stream for initializing
#         self.grabbed , self.frame = self.vcap.read()
#         if self.grabbed is False :
#             print('[Exiting] No more frames to read')
#             exit(0)
#         # self.stopped is initialized to False
#         self.stopped = True
#         # thread instantiation
#         self.t = Thread(target=self.update, args=())
#         self.t.daemon = True # daemon threads run in background

#     # method to start thread
#     def start(self):
#         self.stopped = False
#         self.t.start()

#     # method passed to thread to read next available frame
#     def update(self):
#         while True :
#             if self.stopped is True :
#                 break
#             self.grabbed , self.frame = self.vcap.read()
#             if self.grabbed is False :
#                 print('[Exiting] No more frames to read')
#                 self.stopped = True
#                 break
#         self.vcap.release()

#     # method to return latest read frame
#     def read(self):
#         return self.grabbed, self.frame

#     # method to stop reading frames
#     def stop(self):
#         self.stopped = True



from kivy.clock import Clock
from imutils.video import FPS
import cv2

videoFR = 30.0

class DemoCapture:
    def build(self):
        # layout = MDBoxLayout(orientation= 'vertical')
        # self.image = Image()
        # layout.add_widget(self.image)
        self.capture = cv2.VideoCapture('../backgroundVideos/2.mp4')
        Clock.schedule_interval(self.load_frame, 1.0/videoFR)
    def load_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.image_frame = frame
            cv2.imshow("Frame", self.image_frame)
            print('hello')
        else:
            self.capture(cv2.CAP_PROP_POS_FRAMES, 0)
        # buffer = cv2.flip(frame,0).tostring()
        # texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        # texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        # self.image.texture = texture

if __name__ == '__main__':
    dc = DemoCapture()
    dc.build()