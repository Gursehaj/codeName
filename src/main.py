from itertools import count
from turtle import back
import cv2
import time
import skimage.exposure
import argparse
import serial
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Queue, Value, Process
from pynput import keyboard 
from utils import *

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-c", "--Control", help = "keyboard/arduino", required=True)
parser.add_argument("-b", "--Baudrate", help = "define baud rate if using Arduino to control")
parser.add_argument("-d", "--Debug", help="use this to view masked data")

# Read arguments from command line
args = vars(parser.parse_args())
arduino = serial.Serial()
baudrate = 9600
isKeyboard = False
isDebuging = False

#defining background frame size
ogDim = (1920, 1080)
predDim = (640, 360)
vidLen = 0

if args["Control"] == 'arduino' and args["Baudrate"] != None:
    try:
        baudrate = args["Baudrate"]
        arduino = serial.Serial(port='COM4', baudrate=baudrate, timeout=.1)
    except Exception as e: 
        print(e)
        exit()        
elif args["Control"] == 'keyboard':
    isKeyboard = True
else:    
    print("Baurd rate not passed!")
    exit()

if args["Debug"] == "true":
    isDebuging = True
else:
    isDebuging = False


def on_press(key):
    try:
        if (key == key.up):
            changeBackgroundVideo()
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def changeBackgroundVideo():
    global sharedPos
    global vidLen
    sharedPos.value += 1
    if sharedPos.value >= vidLen:
        sharedPos.value = 0
    # print(sharedPos.value)

def instanceSegmentor(frameQueue, maskQueue):
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

    if(isDebuging):
        cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("cutout", cv2.WINDOW_AUTOSIZE)    
    # FPS = 1/X, X = desired FPS
    FPS = 1/30
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
                originalFrame = frame.copy()

                # width, height, channels = frame.shape
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.cvtColor(cv2.resize(frame, (640,360), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)

                #reducing the captured frame image size for quick prediction
                labels = utils.get_pred(cv2.resize(frame, (predDim[0], predDim[1]), interpolation=cv2.INTER_NEAREST), model)
                
                mask = labels == 15
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 

                mask = mask.astype(np.uint8)
                
                mask[mask!=1] = 0
                mask[mask==1] = 255
                
                # mask = cv2.resize(mask, ogDim)
                mask = cv2.GaussianBlur(cv2.resize(mask, ogDim), (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
                mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(255, 0))
                
                originalFrame[mask==255] = 255

                if(isDebuging):
                    cv2.imshow("mask", mask)
                    cv2.imshow("cutout", originalFrame) 

                try:
                    # maskQueue.put_nowait(mask)
                    frameQueue.put_nowait(originalFrame)
                    # print("data Sent!")
                except:
                    print("Could not send frame and mask data!")

            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # time.sleep(FPS)
    
    # Empty the cache and switch off the interactive mode
    torch.cuda.empty_cache()
    capture.release()
    cv2.destroyAllWindows()

def runVideos(frameQueue, maskQueue, videos, name, sharedPos):
    global ogDim
    global predDim

    caps = []

    for i in range(len(videos)):
        caps.append(cv2.VideoCapture(videos[i]))
    cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("masked", cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty("masked", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # FPS = 1/X, X = desired FPS
    FPS = 1/60
    FPS_MS = int(FPS * 1000)
    while True:
        cap = caps[sharedPos.value]
        if cap.isOpened():
            ret, backgroundImg = cap.read()
            if ret:    
                if not frameQueue.empty():
                    # print("got data!")
                    backgroundImg = cv2.resize(backgroundImg, ogDim)
                    maskImage = frameQueue.get_nowait()
                    
                    backgroundImg[maskImage != 255] = maskImage[maskImage != 255]

                    cv2.imshow("masked", backgroundImg)

            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(FPS)
    cap.release()

if __name__ == '__main__':

    frameQ = Queue()
    maskQ = Queue() 
    # create a integer value
    sharedPos = Value('i', 0)
    
    videos = ['../backgroundVideos/1.mp4', '../backgroundVideos/2.mp4', '../backgroundVideos/3.mp4']

    vidLen = len(videos)

    detectionProcess = Process(target=instanceSegmentor, args=(frameQ, maskQ,), name="Detection Process")
    detectionProcess.start() 

    backgroundProcess = Process(target=runVideos, args=(frameQ, maskQ, videos, "background", sharedPos), name="Background Video Process")
    backgroundProcess.start()
    
    print ("Image Segmentation PID is: " + str(detectionProcess.pid))
    print ("Background Video Process is: " + str(backgroundProcess.pid))

    if isKeyboard:
        #keyboard listening thread
        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()