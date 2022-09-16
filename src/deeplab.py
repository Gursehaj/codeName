import cv2
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from utils import *

class predictor:
    def setupDeeplab(self):
        # Load the DeepLabv3 model to memory
        self.model = utils.load_model()

        # Start a video cam session
        self.video_session = cv2.VideoCapture(0)
    
    def stopDeeplab(self):
        torch.cuda.empty_cache()

    def startPredictor(self):
        # Read frames from the video, make realtime predictions and display the same
        while self.video_session.isOpened():

            frame = utils.grab_frame(self.video_session)

            # Ensure there's something in the image (not completely blacnk)
            if np.any(frame):

                # Read the frame's width, height, channels and get the labels' predictions from utilities
                width, height, channels = frame.shape
                labels = utils.get_pred(frame, self.model)
        
                # The PASCAL VOC dataset has 20 categories of which Person is the 16th category
                # Hence wherever person is predicted, the label returned will be 15
                # Subsequently repeat the mask across RGB channels 
                self.mask = labels == 15
                self.mask = np.repeat(self.mask[:, :, np.newaxis], 3, axis = 2)
                
                bg[mask] = frame[mask]
                frame = bg
        
            else:
                self.stopDeeplab()
                break
    
    def getMask(self):
        return self.mask