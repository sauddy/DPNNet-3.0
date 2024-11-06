import streamlit as st
import pandas as pd
from tensorflow import keras as k
import tensorflow as tf
import numpy as np
import os
import base64
import cv2
from PIL import Image
import time


st.sidebar.markdown('<h1 style="color:white;">DPNNet-2.1</h1>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 style="color:gray;">Multi-planet classification and regression framework:</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h3 style="color:gray;">Given an imput PPD image in png/jpg/jpeg format, the model predicts the mass and number of planets.</h3>', unsafe_allow_html=True)
with st.sidebar.expander("Working tips:"):
    st.write('<h3 style="color:gray;"> Remove an uploaded image before uploading the next one. The current model assumes that the image uploaded is a PPD image, so uploading random images might give you an ouput too. An additional classifier to classify if the uploaded image is a PPD is under development.</h3>', unsafe_allow_html=True)



# background image to streamlit
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('unseen_data/1.jpg')

current_directory = os.getcwd()
path = current_directory + '/'

st.subheader("Please upload PPD image!")
upload= st.file_uploader('abc',type=['png','jpg'], label_visibility="collapsed")

X_res = Y_res = inp_res = 128
CNN_R = tf.keras.models.load_model(path+'data_for_subhrat_20M23/'+'TL_ResNet50_128_modelR')
CNN_C = tf.keras.models.load_model(path+'data_for_subhrat_20M23/'+'TL_ResNet50_128_modelC')


if upload is not None:
    c1, c2= st.columns(2)
    image = Image.open(upload)
    image = image.save("img.png")
    image = cv2.imread("img.png")
    
    left_right_list = []
    top_bottom_list = []
    for i in range(image.shape[1]):
        if image[int(image.shape[0]/2),i,2] > 25:
            left_right_list.append(i)
    for i in range(image.shape[0]):
        if image[i,int(image.shape[1]/2),2] > 25:
            top_bottom_list.append(i)
                        
    top = top_bottom_list[0]
    left = left_right_list[0]
    bottom = top_bottom_list[-1]
    right = left_right_list[-1]
    image = np.squeeze(image) 
    crop_image = image[top:bottom, left:right]
    crop_image = cv2.resize(crop_image, (X_res, Y_res)) 
    crop_image = k.preprocessing.image.img_to_array(crop_image)[:,:,::-1]/255. 
    datagen = k.preprocessing.image.ImageDataGenerator(featurewise_center=True, zca_whitening=True)
    crop_image = datagen.standardize(crop_image)
    c1.header('Input Image')
    c1.image(crop_image)


# prediction on model
if upload is not None:
    if st.button('Make Prediction!'):
        c_preds = CNN_C.predict(np.array([crop_image]))
        r_preds = CNN_R.predict(np.array([crop_image]))
        c2.header('Output')
        c2.subheader('Predicted system type :')
        if c_preds < 4.5 and c_preds > 3.5:
            c_pred = "Given PPD image corresponds to a three planet system!"
        elif c_preds < 3.5 and c_preds > 1.5:
            c_pred = "Given PPD image corresponds to a two planet system!"
        elif c_preds < 1.5 and c_preds > 0.5:
            c_pred = "Given PPD image corresponds to a one planet system!"
        c2.write(c_pred)
        c2.subheader('Predicted Masses :')
        if c_preds < 4.5 and c_preds > 3.5:
            r_pred = "There exists a {pm1} earth mass planet at 24-36 AU from the central star, a {pm2} earth mass planet at 54-66 AU from the central star, and a {pm3} earth mass planet 87-105 AU from the central star.".format(pm1=round(r_preds[0][0],),pm2=round(r_preds[0][1],),pm3=round(r_preds[0][2],))
        elif c_preds < 3.5 and c_preds > 2.5:
            r_pred = "There exists a {pm1} earth mass planet at 24-36 AU from the central star, and a {pm3} earth mass planet 87-105 AU from the central star".format(pm1=round(r_preds[0][0],),pm3=round(r_preds[0][2],))
        elif c_preds < 2.5 and c_preds > 1.5:
            r_pred = "There exists a {pm1} earth mass planet at 24-36 AU from the central star, and a {pm2} earth mass planet 54-66 AU from the central star".format(pm1=round(r_preds[0][0],),pm2=round(r_preds[0][1],))
        elif c_preds < 1.5 and c_preds > 0.5:
            r_pred = "There exists a {pm1} earth mass planet at 24-36 AU from the central star!".format(round(r_preds[0][0],))
        c2.write(r_pred)