import cv2
import numpy as np

class scaleFrame:
    @staticmethod
    def scaleDown(frame, scaleVal):
        scale_percent = scaleVal # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        return resized

    # @staticmethod
    # def addPadding(frame, x, y):
