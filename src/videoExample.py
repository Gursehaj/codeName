from torch.multiprocessing import Pool, Process, set_start_method
import cv2
import time
from multiprocessing import Process

# try:
#      set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

def runVideos(video,name):
    cap = cv2.VideoCapture(video)
    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    while cap.isOpened():
        pTime = time.time()
        ret, img = cap.read()
        if ret:
            cTime = time.time()
            fps = str(int(1 / (cTime - pTime)))

            img = cv2.resize(img, (700, 700))
            cv2.imshow(name, img)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # ret, img = cap.read()
            # img = cv2.resize(img, (700, 700))
            # cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    # freeze_support()
    videos = ['videos/video1.mp4', 'videos/video2.mp4']

    process1 = Process(target=runVideos, args=(videos[0], str(videos[0])))
    process1.start()

    process2 = Process(target=runVideos, args=(videos[1], str(videos[1])))
    process2.start() 

    # for i in videos:
    #     process = Process(target=runVideos, args=(i, str(i)))
    #     process.start()