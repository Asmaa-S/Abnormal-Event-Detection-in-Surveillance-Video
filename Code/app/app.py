import pandas as pd
import numpy as np
import streamlit as st
from os.path import isfile, join
from PIL import Image
import time 
import os 
import re
import cv2 
import test_app
import fpstimer
import math 
from evaluate_PSNR import *
import pickle as pkl
from active_learning_test import *

timer = fpstimer.FPSTimer(25) # Make a timer that is set for 25 fps

st.title('Abnormal Event Detection in Surviellance Videos :movie_camera:')
parag = st.header("This is a demo application for our project. Our System is composed of two networks; a detection network,"+ 
    "and an active learning layer to improve the performance. To test our system, please choose an option from the sidebar on the left ^^")

options = ['', 'Test the System', 'Go to Documentation']
selected_option = st.sidebar.selectbox("What do you want to do?", options,format_func=lambda x: ' ' if x == '' else x)
if not selected_option:
    st.warning('What do you want to do?')

#################### Selecting Test the System ###########################
if selected_option == 'Test the System':

    tests = ['', 'Regularity Score', 'PSNR Regularity Score' , 'Live Abnormality', 'Active Learning Module']
    selected_test = st.sidebar.selectbox("You can choose whether to view live abnormality or view " 
    +"the regularity scores:", tests, format_func=lambda x: ' ' if x == '' else x)

    datasets = ['', 'UCSDped1', 'UCSDped2']
    selected_dataset = st.sidebar.selectbox("As our system highly depends on the dataset; please choose a dataset to test:",
     datasets, format_func=lambda x: ' ' if x == '' else x)
    
    if selected_test == 'Regularity Score':
        parag.write('_**Regularity Score**_ is a score calculated based on the reconstruction error between the original frame and the reconstructed frame. Normal frames have higher regularity than abnormal frames.')    
        if selected_dataset:
            test_videos = [f for f in (os.listdir("./Data/{}/Test_videos".format(selected_dataset))) 
            if join("./Data/{}/Test_videos".format(selected_dataset), f)]
            test_videos.insert(0,'')
            selected_video = st.sidebar.selectbox('Pick a test video', test_videos, format_func=lambda x: ' ' if x == '' else x)
            if selected_video:
                col1, col2 = st.beta_columns(2)
                video_file = open(os.path.join("./Data/{0}/Test_videos/{1}".format(selected_dataset, selected_video)), 'rb')
                video_bytes = video_file.read()
                col1.video(video_bytes)
                #progress_bar = st.sidebar.progress(0)
                #status_text = st.sidebar.empty()
                if col1.button('Plot Regularity Score'):
                    regularity_score = test_app.test(selected_video,selected_dataset)
                    print('Done')
                    col2.line_chart(pd.DataFrame(regularity_score, columns=['Regularity Score']))
                    #for i,score in enumerate(regularity_score):
                    #    status_text.text("%i%% Complete" %(i//(len(regularity_score)/100)))
                    #    chart.add_rows(pd.DataFrame([[score]], columns=['Regularity Score']))
                    #    progress_bar.progress(int(i//(len(regularity_score)/100)))
                    #    timer.sleep()
                    #status_text.text("%i%% Complete" %100)
                    #progress_bar.empty()
                    col2.write('Having a high score indicates a normal event while having a low score indicates an abnormal event.')


    elif selected_test == 'PSNR Regularity Score':
        parag.write('_**PSNR(Peak Signal to Noise Ratio):**_  It is used as a metric between a predicted frame and a ground truth frame. Having a high PSNR indicates a normal frame.')    
        if selected_dataset:
            test_videos = [f for f in (os.listdir("./Data/{}/Test_videos".format(selected_dataset))) 
                    if join("./Data/{}/Test_videos".format(selected_dataset), f)]
            test_videos.insert(0,'')
            selected_video = st.sidebar.selectbox('Pick a test video', test_videos, format_func=lambda x: ' ' if x == '' else x)
            if selected_video:
                col1, col2 = st.beta_columns(2)
                video_file = open(os.path.join("./Data/{0}/Test_videos/{1}".format(selected_dataset, selected_video)), 'rb')
                video_bytes = video_file.read()
                col1.video(video_bytes)
                if col1.button('Plot PSNR Regularity Score'):
                    PSNR_file = os.path.join("./Data/{0}/PSNR/{0}".format(selected_dataset, selected_video))
                    psnrs = np.array(load_psnr(PSNR_file))
                    index = int(re.findall(r'\d+', selected_video)[0].replace('00',''))
                    psnr = (psnrs[index] - np.min(psnrs[index]))/np.max(psnrs[index])
                    col2.line_chart(pd.DataFrame(psnr, columns=['PSNR']))
                    col2.write('Having a high PSNR indicates a normal event while having a low score indicates an abnormal event.')
    
    elif selected_test == 'Live Abnormality':
        parag.write('In this test, we show the input video, the reconstructed video, the difference between the two, and point the abnormality detected by the system. This is all base on a threshold.')    
        if selected_dataset:
            test_videos = [f for f in (os.listdir("./Data/{}/Test_videos".format(selected_dataset))) 
                    if join("./Data/{}/Test_videos".format(selected_dataset), f)]
            test_videos.insert(0,'')
            selected_video = st.sidebar.selectbox('Pick a test video', test_videos, format_func=lambda x: ' ' if x == '' else x)
            if selected_video:
                try:
                    video_file = open(os.path.join("./Data/{0}/live_abnormality/{1}".format(selected_dataset, selected_video)), 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                except:
                    pass
    
    elif selected_test == 'Active Learning Module':
        parag.write('In this part, you are shown a video volume and you are required to answer whether it is a normal or abnormal video. This will help the model retrain and improves its performance.')
        if selected_dataset:    
            t = 10
            mean_frame = np.load(os.path.join('Data/{0}/mean_frame_224.npy'.format(selected_dataset)))
            data_path = os.path.join('Data/{0}/data.pkl'.format(selected_dataset))
            Active_session(data_path, mean_frame,start = 0, budget=5)

                    
############# Selecting Go to Documentation #####################
elif selected_option == 'Go to Documentation':
        text = "Abnormal event detection (AED), also termed abnormality detection, is a relatively popular, but very challenging research problem in the field of computer vision. In simple terms, the goal is to develop a computer algorithm that goes through security footage to recognize and flag abnormal events. The definition of “abnormal” is heavily context dependent, however, the general consensus is that an abnormal event should be a fairly rare occurrence, that usually falls into the category of illegal acts or accidents."
        st.write(text)
        st.write('Two models are tested in this web interface:')
        st.write('1. Spatio-temporal Auto encoder:')
        st.image('Code/app/autoencoder.png')
        st.write('2. Future Frame Prediction:')
        st.image('Code/app/future.png')
