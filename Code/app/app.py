import pandas as pd
import numpy as np
import streamlit as st
from os.path import isfile, join
from PIL import Image
import time 
import os 
import cv2 
import test_app
import fpstimer
import math 

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

    tests = ['', 'Regularity Score', 'Live Abnormality']
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
                regularity_score = test_app.test(selected_video,selected_dataset)
                #regularity_score =  list(range(0,200))
                video_bytes = video_file.read()
                col1.video(video_bytes)

                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                chart = col2.line_chart()
                if st.button('Plot Regularity Score'):
                    col2.write('Having a high score indicates a normal event while having a low score indicates an abnormal event.')
                    for i,score in enumerate(regularity_score):
                        status_text.text("%i%% Complete" %(i//(len(regularity_score)/100)))
                        chart.add_rows(pd.DataFrame([[score]], columns=['Regularity Score']))
                        progress_bar.progress(int(i//(len(regularity_score)/100)))
                        timer.sleep()
                    status_text.text("%i%% Complete" %100)
                    progress_bar.empty()

                
############# Selecting Go to Documentation #####################
elif selected_option == 'Go to Documentation':
    text = "Abnormal event detection (AED), also termed abnormality detection, is a relatively popular, but very challenging research problem in the field of computer vision. In simple terms, the goal is to develop a computer algorithm that goes through security footage to recognize and flag abnormal events. The definition of “abnormal” is heavily context dependent, however, the general consensus is that an abnormal event should be a fairly rare occurrence, that usually falls into the category of illegal acts or accidents. In the context of highway surveillance, for example, an abnormal event that we might want to look out for is car accidents. In most contexts, violent acts like robbery and murder are also abnormal events. In specialized environments such as museums and art galleries; touching, damaging, or stealing exhibits is an event we need to monitor for. This high context dependency is one of many factors that complicate the problem of anomaly detection. Another confounding factor is that individuals committing illegal acts usually go through a lot of trouble to not draw attention or look suspicious which can often deceive even human observers. The biggest challenge relating to AED is the lack of clear definitions and boundaries for normal. Most AED algorithms rely on one class classification where the normal baseline is learned by the model and anything that deviates from the baseline is considered normal. The problem with this approach is that any unusual event that doesn't happen often would be flagged as abnormal. In the context of surveillance, we need additional functionalities such as human activity recognition to discern illegal acts from unusual, but perfectly legal acts."
    st.write(text)