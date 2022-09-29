from multiprocessing import Queue
from queue import Empty
# from subprocess import CREATE_NEW_CONSOLE
# from torch.multiprocessing import Pool, Process, set_start_method
import cv2
import time
from multiprocessing import Process, Pipe
import skimage.exposure

import numpy as np
# from PIL import Image as im

# import PIL.Image
import matplotlib.pyplot as plt
from utils import *

ogDim = (1280, 720)
predDim = (640, 360)

def instanceSegmentor(queue):
    global ogDim
    global predDim
    
    # Load the DeepLabv3 model to memory
    model = utils.load_model()
    
    # Load webcam
    print("Waiting for camera to load")
    capture = cv2.VideoCapture(0) 
    print("Camera loaded!")
    print("Setting camera properties")
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, ogDim[1])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, ogDim[0])
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    print("Properties Set! Starting Webcam!")

    # cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("raw", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("cutout", cv2.WINDOW_AUTOSIZE)

    # time.sleep(2)
    
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
                ogFrame = frame.copy()

                # width, height, channels = frame.shape
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.cvtColor(cv2.resize(frame, (640,360), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
                labels = utils.get_pred(cv2.resize(frame, (predDim[0], predDim[1]), interpolation=cv2.INTER_NEAREST), model)
                
                mask = labels == 15
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 

                mask = mask.astype(np.uint8)
                
                mask[mask!=1] = 0
                mask[mask==1] = 255
                mask = cv2.GaussianBlur(mask, (0,0), sigmaX=4, sigmaY=4, borderType = cv2.BORDER_DEFAULT)
                mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,255))

                # print(np.unique(labels))

                try:
                    queue.put_nowait(mask)
                    # print("data Sent!")
                except:
                    print("Could not send mask data!")

                n_frame = cv2.resize(frame.copy(), (predDim[0], predDim[1]), interpolation=cv2.INTER_NEAREST)
                n_frame[mask==0] = 0
                # mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
                # mask = labels.astype(np.uint8) #* 1.0
                # cv2.imshow("raw", frame)
                cv2.imshow("cutout", n_frame)
                # cv2.imshow("mask", mask)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # time.sleep(FPS)
    
    # Empty the cache and switch off the interactive mode
    torch.cuda.empty_cache()
    capture.release()
    cv2.destroyAllWindows()

def runVideos(queue, video, name):
    global ogDim
    global predDim
    cap = cv2.VideoCapture(video)
    # cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("blurmasked", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("masked", cv2.WINDOW_AUTOSIZE)

    # FPS = 1/X, X = desired FPS
    FPS = 1/120
    FPS_MS = int(FPS * 1000)
    while True:
        if cap.isOpened():
            ret, img = cap.read()
            if ret:    
                # cv2.imshow(name, img)
                n_frame = img.copy()
                # try:
                if not queue.empty():
                    # print("got data!")
                    mask = queue.get_nowait()
                    cv2.imshow("masked", mask)
                    # blur threshold image
                    blur = cv2.GaussianBlur(mask, (0,0), sigmaX=4, sigmaY=4, borderType = cv2.BORDER_DEFAULT)
                    # stretch so that 255 -> 255 and 127.5 -> 0
                    # C = A*X+B
                    # 255 = A*255+B
                    # 0 = A*127.5+B
                    # Thus A=2 and B=-127.5
                    #aa = a*2.0-255.0 does not work correctly, so use skimage
                    # result = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255))
                    cv2.imshow("blurmasked", blur)
                
                # except Exception as e:
                #     print("could not get mask data!\n" + str(e))
                # mask = cv2.resize(pipeData[0], (1920,1080), interpolation=cv2.INTER_NEAREST)
                # ogImg = pipeData[1]
                # print("Background Shape is: " + str(n_frame.shape))
                # print("Original Frame Shape is: " + str(ogImg.shape))
                # print("Mask Shape is: " + str(mask.shape)) 
                # n_frame[mask!=0] = ogImg           

                # cv2.imshow("masked", n_frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FPS)
    cap.release()

if __name__ == '__main__':

    q = Queue()

    videos = ['../backgroundVideos/1.mp4', '../backgroundVideos/2.mp4']

    detectionProcess = Process(target=instanceSegmentor, args=(q,))
    detectionProcess.start() 

    backgroundProcess = Process(target=runVideos, args=(q, videos[1], str(videos[1])))
    backgroundProcess.start()