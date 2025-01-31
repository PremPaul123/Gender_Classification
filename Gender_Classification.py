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
    try:
        image = image.convert('RGB')  # Ensure image is in RGB mode
        image = image.resize((64, 64))  # Resize to match model input size
        image = np.array(image).astype('float32') / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("👨‍💻 Gender Classification App")
st.write("Upload an image to classify gender.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    if processed_image is not None:
        with st.spinner("Classifying..."):
            prediction = model.predict(processed_image)[0][0]  # Extract the prediction value
        
        # Determine gender and confidence
        gender = "Female" if prediction < 0.5 else "Male"
        confidence = 1 - prediction if gender == "Female" else prediction
        
        st.success(f"Predicted Gender: *{gender}*")
        st.progress(float(confidence))  # Show confidence as a progress bar
        st.write(f"Confidence Score: *{confidence:.2%}*")