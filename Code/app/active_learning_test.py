import sys

from cv2 import VideoWriter, VideoWriter_fourcc

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
#from show_vid import *
from IPython.core.display import display, HTML
from base64 import b64encode
from time import sleep
import pickle as pkl
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
import os
import numpy as np
from tensorflow.keras.models import load_model
import glob 
import cv2
from tensorflow.keras.layers import Dense, Activation, Lambda, Input, Concatenate,Flatten
import tensorflow.keras.backend as K 
from tensorflow.keras.utils import to_categorical
import streamlit as st


def show_vid(VID, mean_frame= None, play=False):

    if np.all(mean_frame != None):
        VID = VID + np.repeat(mean_frame[np.newaxis,:, :], VID.shape[0], axis=0)

    if os.path.isfile('Data/AL/vid.mp4'):
        os. remove("Data/AL/vid.mp4")
    if os.path.isfile('Data/AL/vid.avi'):
        os. remove("Data/AL/vid.avi")
        

    if VID.shape[0] <= 20:
        FPS = 4
        s= 0.5
    else:
        FPS = 11
        s = 0.1
    (width, height) = VID.shape[1:3]
    fourcc =  VideoWriter_fourcc(*'MP42')
    video = VideoWriter('Data/AL/vid.avi', fourcc, float(FPS), (height, width), 0)
    for i in range(VID.shape[0]):
        frame = VID[i,:,:]
        if frame.max() <= 1:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        video.write(frame)
    video.release()
    os.system('ffmpeg -i {0} -vcodec libx264 {1}'.format('Data/AL/vid.avi','Data/AL/vid.mp4'))



def Active_session(data_path, mean_frame,start = 0, budget=5):
    abnormal_samples = pkl.load(open(data_path, 'rb'))
    handlabled_data ={'X':[], 'y':[]}
    
    st.subheader(f'starting at sample index = {start}')    
    label = st.empty()
    for i, samp in  enumerate( abnormal_samples[start:start+budget]):    
        
        col1, col2 = st.beta_columns(2)

        #show volume
        sample = np.expand_dims(samp, axis =0)
        show_vid(sample[0].squeeze(), mean_frame)
        video_bytes = open('Data/AL/vid.mp4','rb').read()
        col1.video(video_bytes)
        sleep(0.1)
        
        #feedback
        col2.markdown('sample number {0}'.format(i))
        label.text_input('For sample {0},label with a single number- either 0 for normal sequence or 1 for abnormal: '.format(i), '')
        sleep(0.1)
        if label:
            handlabled_data['X'].append(samp)
            #handlabled_data['y'].append(int(label))
        else: st.warning("Please fill out so required fields")

        #Free
        if os.path.isfile('Data/AL/vid.mp4'):
            os. remove("Data/AL/vid.mp4")
        if os.path.isfile('Data/AL/vid.avi'):
            os. remove("Data/AL/vid.avi")
        col2 = st.empty()
        col1 =  st.empty()
        label =  st.empty()
        sleep(0.1)
    st.write('Retraining Model ......')
    return handlabled_data

