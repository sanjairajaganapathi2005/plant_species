import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

model = load_model("dcnn_model_ps.h5")

index = [
    "Alstonia Scholaris diseased",
    "Alstonia Scholaris healthy",
    "Arjun diseased",
    "Arjun healthy",
    "Bael diseased",
    "Basil healthy",
    "Chinar diseased",
    "Chinar healthy",
    "Guava diseased",
    "Guava healthy",
    "Jamun diseased",
    "Jamun healthy",
    "Jatropha diseased",
    "Jatropha healthy",
    "Lemon diseased",
    "Lemon healthy",
    "Mango diseased",
    "Mango healthy",
    "Pomegranate diseased",
    "Pomegranate healthy",
    "Pongamia Pinnata diseased",
    "Pongamia Pinnata healthy"
]

st.title("Plant Disease Classifier")

# Use markdown to make the text h5
st.markdown("<h5>Upload an image of a plant leaf to classify its health status.</h5>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    y = model.predict(x)
    preds = np.argmax(y, axis=1)

    result_text = "The classified plant species is: " + str(index[preds[0]])
    st.markdown(f"<h3 style='text-align: center; color: black;'>{result_text}</h3>", unsafe_allow_html=True)
