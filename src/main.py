from multiprocessing import Queue
# from subprocess import CREATE_NEW_CONSOLE
# from torch.multiprocessing import Pool, Process, set_start_method
import cv2
import time
from multiprocessing import Process, Pipe

import numpy as np
# from PIL import Image as im

# import PIL.Image
import matplotlib.pyplot as plt
from utils import *

def instanceSegmentor():
    # Load the DeepLabv3 model to memory
    model = utils.load_model()
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)

    print("Waiting for camera to load")
    time.sleep(2)
    
    # FPS = 1/X, X = desired FPS
    # FPS = 1/120
    # FPS_MS = int(FPS * 1000) #in milli seconds (use if required)

    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            
            # Ensure valid frame
            if status:
                # Using cv2.flip() method
                # Use Flip code 0 to flip vertically
                frame = cv2.flip(frame, 1)
                width, height, channels = frame.shape
                labels = utils.get_pred(frame, model)
                mask = labels == 15
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 
                mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
                cv2.imshow("raw", frame)
                cv2.imshow("mask", mask * 1.0)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # time.sleep(FPS)
    
    # Empty the cache and switch off the interactive mode
    torch.cuda.empty_cache()
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

    #Setting up shared memory (Queue) for processes to shared background frame
    backgroundImageQueue = Queue()

    videos = ['../backgroundVideos/1.mp4', '../backgroundVideos/2.mp4']

    detectionProcess = Process(target=instanceSegmentor, args=())
    detectionProcess.start() 

    backgroundProcess = Process(target=runVideos, args=(videos[1], str(videos[1])))
    backgroundProcess.start()


