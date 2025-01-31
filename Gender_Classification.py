import cv2
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Gender classification by Prem.h5')

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to match model input size
    image = np.array(image).astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("👨‍💻 Gender Classification App")
st.write("Upload an image or capture from the camera to classify gender.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Capture image from webcam
camera_image = st.camera_input("Capture an image")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    gender = "Female" if prediction < 0.5 else "Male"
    
    st.success(f"Predicted Gender: *{gender}*")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    gender = "Female" if prediction < 0.5 else "Male"
    
    st.success(f"Predicted Gender: *{gender}*")
