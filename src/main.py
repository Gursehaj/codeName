from multiprocessing import Queue
from subprocess import CREATE_NEW_CONSOLE
from torch.multiprocessing import Pool, Process, set_start_method
import cv2
import time
from multiprocessing import Process

# import numpy as np
# import PIL.Image
# import matplotlib.pyplot as plt
from utils import *

## section to setup deeplab
#=============================================
# Load the DeepLabv3 model to memory
# model = utils.load_model()

# Define two axes for showing the mask and the true video in realtime
# And set the ticks to none for both the axes
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
# ax1.set_title("Background Changed Video")
# ax2.set_title("Mask")

# ax1.set_xticks([])
# ax1.set_yticks([])
# ax2.set_xticks([])
# ax2.set_yticks([])

# Create two image objects to picture on top of the axes defined above
# im1 = ax1.imshow(utils.grab_frame(video_session))
# im2 = ax2.imshow(utils.grab_frame(video_session))

# Switch on the interactive mode in matplotlib
# plt.ion()
# plt.show()
#=============================================

# try:
#      set_start_method('spawn', force=True)
# except RuntimeError:
#     pass


def instanceSegmentor():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    # FPS = 1/X, X = desired FPS
    FPS = 1/120
    FPS_MS = int(FPS * 1000) #in milli seconds (use if required)

    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
                # Using cv2.flip() method
                # Use Flip code 0 to flip vertically
                frame = cv2.flip(frame, 1)
                cv2.imshow('frame', frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FPS)

    capture.release()
    cv2.destroyAllWindows()

def runVideos(video,name):
    cap = cv2.VideoCapture(video)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    # FPS = 1/X, X = desired FPS
    FPS = 1/120
    FPS_MS = int(FPS * 1000)
    while True:
        if cap.isOpened():
            ret, img = cap.read()
            if ret:    
                cv2.imshow(name, img)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FPS)

    cap.release()

if __name__ == '__main__':

    # try:
    #     set_start_method('spawn', force=True)
    # except RuntimeError:
    #     pass

    #Setting up shared memory (Queue) for processes to shared background frame
    backgroundImageQueue = Queue()

    videos = ['../backgroundVideos/1.mp4', '../backgroundVideos/2.mp4']

    backgroundProcess = Process(target=runVideos, args=(videos[1], str(videos[1])))
    backgroundProcess.start()

    detectionProcess = Process(target=instanceSegmentor, args=())
    detectionProcess.start() 
