# Import required libraries
import PIL

import streamlit as st
from ultralytics import YOLO

# Replace the relative path to your weight file
model_path = r'Path your model YOLO'

st.title("Brain Tumor Detection!")

image = st.file_uploader("Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# confidence = float(st.slider("Select Model Confidence", 25, 100, 40)) / 100

confidence = st.number_input(label="Confidence Threshold (%)",value=40,max_value=100)

# Creating main page heading
st.header("Result!")

col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if image:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(image)
        # Adding the uploaded image to the page with a caption
        st.image(image,
                 caption="Uploaded Image",
                 use_column_width=True
                 )
        
try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.button(label='Analyze'):
    res = model.predict(uploaded_image,
                            conf=confidence/100
                            )
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
            caption='Detected Image',
            use_column_width=True
            )
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
        except Exception as ex:
            st.write("No image is uploaded yet!")
