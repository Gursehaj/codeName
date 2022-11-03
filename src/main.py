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
from torch import true_divide 
from utils import *
from drawing import *

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


# def on_press(key):
#     global experience
#     try:
#         if (key == key.enter):
#             experience = True
#     except AttributeError:
#         print('special key {0} pressed'.format(
#             key))

# def on_press(key):
#     try:
#         if (key == key.enter):
#             changeBackgroundVideo()
#     except AttributeError:
#         print('special key {0} pressed'.format(
#             key))

# def changeBackgroundVideo():
#     global sharedPos
#     global vidLen
#     sharedPos.value += 1
#     if sharedPos.value >= vidLen:
#         sharedPos.value = 0
#     # print(sharedPos.value)

def instanceSegmentor(frameQueue, maskQueue, waitForFrame):
    global ogDim
    global predDim
    
    # Load the DeepLabv3 model to memory
    model = utils.load_model()
    
    # Load webcam
    print("Waiting for camera to load")
    capture = cv2.VideoCapture(0) 
    print("Camera loaded!")
    print("Setting camera properties")
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, predDim[1])
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, predDim[0])
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    print("Properties Set! Starting Webcam!") 

    if(isDebuging):
        cv2.namedWindow("mask", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("cutout", cv2.WINDOW_AUTOSIZE)    

    _, f = capture.read()
    print(f.shape)
    # FPS = 1/X, X = desired FPS
    FPS = 1/60
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
                #frame = cv2.resize(frame, (predDim[0], predDim[1]), interpolation=cv2.INTER_NEAREST)
                labels = utils.get_pred(frame, model)
                
                mask = labels == 15
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 

                mask = mask.astype(np.uint8)
                
                mask[mask!=1] = 0
                mask[mask==1] = 255
                
                # mask = cv2.resize(mask, ogDim)
                mask = cv2.GaussianBlur(mask, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
                mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(255, 0))
                
                originalFrame[mask==255] = 255

                # originalFrame = scaleFrame.scaleDown(originalFrame, 50)

                if(isDebuging):
                    cv2.imshow("mask", mask)
                    cv2.imshow("cutout", originalFrame) 

                try:
                    # maskQueue.put_nowait(mask)
                    if (waitForFrame.value):
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

def runVideos(frameQueue, maskQueue, videos, name, sharedPos, waitForFrame):
    global ogDim
    global predDim
    experience = False
    left = 600
    top = 400
    caps = []

    for i in range(len(videos)):
        caps.append(cv2.VideoCapture(videos[i]))
    cv2.namedWindow("masked", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("masked", cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty("masked", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # FPS = 1/X, X = desired FPS
    FPS = 1/60
    FPS_MS = int(FPS * 1000)
    welcomeImg = cv2.imread('../Jhatka Board.png')
    lakeOld = cv2.imread("../1.png")
    cityOld = cv2.imread("../2.png")
    lakeOldCheck = False
    lakeNewCheck = False
    cityNewCheck = False
    cityOldCheck = False
    while True:            
        if (not experience):
            cv2.imshow("masked", welcomeImg)
            if cv2.waitKey(1) & 0xFF == ord('\r'):
                experience = True
                lakeOldCheck = True
        else:
            if (lakeOldCheck):
                cv2.imshow("masked",lakeOld)
                cv2.waitKey(3000)
                lakeOldCheck = False
                lakeNewCheck = True

            if (cityOldCheck):
                cv2.imshow("masked",cityOld)
                cv2.waitKey(3000)
                cityOldCheck = False
                cityNewCheck = True

            if(lakeNewCheck or cityNewCheck):
                waitForFrame.value = True
                if(lakeNewCheck):
                    cap = caps[0]
                elif(cityNewCheck):
                    cap = caps[1]
                if cap.isOpened():
                    ret, backgroundImg = cap.read()
                    if ret:    
                        if not frameQueue.empty():
                            # print("got data!")
                            # backgroundImg = cv2.resize(backgroundImg, ogDim)
                            maskImage = frameQueue.get_nowait()

                            # backgroundImg[maskImage != 255] = maskImage[maskImage != 255]

                            # backgroundImg[top:maskImage.shape[0]+top,left:maskImage.shape[1]+left] = maskImage
                            # 
                            back = backgroundImg[top:maskImage.shape[0]+top,left:maskImage.shape[1]+left]

                            back[maskImage != 255] = maskImage[maskImage != 255]
                            backgroundImg[top:maskImage.shape[0]+top,left:maskImage.shape[1]+left]= back
                            # backgroundImg[backgroundImg == 255] = backgroundImg[backgroundImg != 255]

                            cv2.imshow("masked", backgroundImg)

                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if(lakeNewCheck):
                            cityOldCheck = True
                            lakeNewCheck = False
                            lakeOldCheck = False
                            waitForFrame.value = False
                        elif (cityNewCheck):
                            lakeOldCheck = cityOldCheck = lakeNewCheck = cityNewCheck = experience = waitForFrame.value = False
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    time.sleep(FPS)
    cap.release()

if __name__ == '__main__':

    frameQ = Queue()
    maskQ = Queue() 
    # create a integer value
    sharedPos = Value('i', 0)
    waitForFrame = Value('i', False)
    videos = ['../backgroundVideos/1.MOV', '../backgroundVideos/2.MOV']

    vidLen = len(videos)

    detectionProcess = Process(target=instanceSegmentor, args=(frameQ, maskQ, waitForFrame,), name="Detection Process")
    detectionProcess.start() 

    backgroundProcess = Process(target=runVideos, args=(frameQ, maskQ, videos, "background", sharedPos, waitForFrame,), name="Background Video Process")
    backgroundProcess.start()
    
    print ("Image Segmentation PID is: " + str(detectionProcess.pid))
    print ("Background Video Process is: " + str(backgroundProcess.pid))

    # if isKeyboard:
    #     #keyboard listening thread
    #     listener = keyboard.Listener(
    #         on_press=on_press)
    #     listener.start()