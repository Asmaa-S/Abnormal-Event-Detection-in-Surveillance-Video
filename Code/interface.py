import pandas as pd
import numpy as np
import streamlit as st
from os import listdir
from os.path import isfile, join
from PIL import Image
import training
import test
from keras.models import load_model
from keras import backend as K

showpred = 0
try:
	model_path = './models/model.h5'
	model_weights_path = './models/weights.h5'
except: 
	print("Need to train model")
test_path = 'Data/Test'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)
st.sidebar.title("About")

st.sidebar.info()

onlyfiles = [f for f in listdir("Data/Test") if isfile(join("Data/Test", f))]

st.sidebar.title("Train Neural Network")
if st.sidebar.button('Train CNN'):
	training.train()

st.sidebar.title("Predict New Images")
imageselect = st.sidebar.selectbox("Pick an image.", onlyfiles)
if st.sidebar.button('Predict Animal'):
    showpred = 1
    prediction = test.test((model),"Data/Test/" + imageselect)


st.title('Animal Identification')
st.write("Pick an image from the left. You'll be able to view the image.")
st.write("When you're ready, submit a prediction on the left.")

st.write("")
image = Image.open("Data/Test/" + imageselect)
st.image(image, caption="Let's predict the animal!", use_column_width=True)

if showpred == 1:
    if prediction == 0:
        st.write("This is a **horse!**")
    if prediction == 1:
        st.write("This is an **elephant!**")
    if prediction == 2:
        st.write("This is a **cat!**")