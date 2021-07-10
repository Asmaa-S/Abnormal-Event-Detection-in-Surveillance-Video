import glob
import numpy as np
import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
from time import sleep
import pickle as pkl
import os

def show_vid(VID, mean_frame= None, play=False):

    if np.all(mean_frame != None):
        VID = VID + np.repeat(mean_frame[np.newaxis,:, :], VID.shape[0], axis=0)

    if os.path.isfile('vid.mp4'):
        os.remove("vid.mp4")
    if os.path.isfile('vid.avi'):
        os.remove("vid.avi")

    if VID.shape[0] <= 20:
        FPS = 4
        s= 0.5
    else:
        FPS = 11
        s = 0.1
    (width, height) = VID.shape[1:3]
    #print(width, height)
    fourcc =  VideoWriter_fourcc(*'MP42')
    video = VideoWriter('./vid.avi', fourcc, float(FPS), (height, width), 0)
    for i in range(VID.shape[0]):
        frame = VID[i,:,:]
        if frame.max() <= 1:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #print(frame.shape)
        video.write(frame)
        if play:
            sleep(s)
            try: 
                cv2.imshow('video snippit',frame)
            except:
                from google.colab.patches import cv2_imshow
                cv2_imshow(frame)

            cv2.waitKey(1)
            cv2.destroyAllWindows()


    video.release()
    os.system('ffmpeg -i vid.avi vid.mp4')


