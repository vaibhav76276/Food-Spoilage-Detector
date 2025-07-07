import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

@st.cache_resource
def load_model():
    model_path = "C:/Users/Vaibh/OneDrive/Desktop/Shivang/xyz.h5"
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

labels = ['Fresh', 'Spoiled']

def prepare_image(img, target_size=(224, 224)):
    img = img.convert('RGB')   # Ensure image is RGB
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_spoilage(img):
    processed_img = prepare_image(img)
    preds = model.predict(processed_img)
    class_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][class_idx]
    return labels[class_idx], confidence

st.title("🍎 Food Spoilage Detection")
st.write("Upload an image of food to check if it is fresh or spoiled.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    if st.button("Detect Spoilage"):
        label, confidence = predict_spoilage(img)
        st.write(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{confidence*100:.2f}%**")
