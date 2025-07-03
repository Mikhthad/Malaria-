import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image,ImageOps


model = tf.keras.models.load_model('Malaria_detection.h5')

im_size = model.input_shape[1] 
class_names = ['Parasitized', 'Uninfected']

st.title("🦠 Malaria Cell Detection App")
st.write("Upload a cell image, and the model will predict whether it's Parasitized or Uninfected.")


uploaded_file = st.file_uploader("Choose a cell image...",type = ['jpg','jpeg','png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Cell Image", use_container_width=True)

    img = image.resize((im_size,im_size))
    img_array = np.array(img)/255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array,axis = 0)

    prediction = model.predict(img_array)[0][0]
    predicted_class = class_names[int(prediction > 0.5)]
    confidence = prediction if prediction > 0.5 else 1 - prediction


    st.subheader("Prediction:")
    st.success(f"**{predicted_class}** with {confidence*100:.2f}% confidence")