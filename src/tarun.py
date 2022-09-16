from distutils.command.build import build
from multiprocessing import Process
import time
from arka import DemoCapture, runVideoPlayer
import cv2
import numpy as np
import PIL.Image
from utils import *
from deeplab import *

def foo():
  runVideoPlayer()

def bar():
  print('hibar')

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


if __name__ == '__main__':
  # freeze_support()
  # d = predictor()
  # d.setupDeeplab()
  tic = time.time()
  runInParallel(bar, foo)
  print(time.time() - tic)
  print('hello')