import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Debug: show files in folder (optional, can remove later)
st.write("Files in folder:", os.listdir())

# Load model
model = load_model("mnist_model.h5")

st.title("🧠 Handwritten Digit Recognizer")

st.write("Upload an image of a digit (0–9)")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to array
    img_array = np.array(img)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Reshape for model
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    # Show result
    st.image(uploaded_file, caption="Uploaded Image", width=150)
    st.success(f"Predicted Digit: {predicted_digit}")