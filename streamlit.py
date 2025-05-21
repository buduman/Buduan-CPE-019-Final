import streamlit as st
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('weathercl_model.h5')
    return model

model = load_model()

st.write("""
# Buduan - CPE 019 - Final Project""")
file = st.file_uploader("Upload weather image", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (28, 28)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype(np.float32) / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
  st.text("Please upload an image file")
else:
  image = Image.open(file)
  st.image(image, use_container_width=True)
  prediction = import_and_predict(image, model)
  class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
  string = "The output is: " + class_names[np.argmax(prediction)]
  st.success(string)
  confidence = np.max(prediction) * 100
  st.info(f"Model confidence: {confidence:.2f}%")
