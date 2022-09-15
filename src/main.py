# import time # time library
# import cv2
# from player import WebcamStream

# # initializing and starting multi-threaded webcam input stream 
# webcam_stream = WebcamStream("2.mp4") # 0 id for main camera
# webcam_stream.start()
# # processing frames in input stream
# num_frames_processed = 0 
# start = time.time()
# while True :
#     if webcam_stream.stopped is True :
#         break
#     else :
#         _, frame = webcam_stream.read()
#     # adding a delay for simulating video processing time 
#     # delay = 0.03 # delay value in seconds
#     # time.sleep(delay) 
#     if _:
#         num_frames_processed += 1
#         # displaying frame 
#         cv2.imshow('frame' , frame)
#         key = cv2.waitKey(1)
#         if key == ord('q'):
#             break
# end = time.time()
# webcam_stream.stop() # stop the webcam stream

# # printing time elapsed and fps 
# elapsed = end-start
# fps = num_frames_processed/elapsed 
# print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))
# # closing all windows 
# cv2.destroyAllWindows()

from arka import DemoCapture
import time

if __name__ == '__main__':
    DemoCapture().run()
    print('hello')
    time.sleep(5.0)
    print('chala')
    DemoCapture.changeBackground()