import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained deepfake detection model
model = load_model('deepfake_detection_model.h5')

# Function to preprocess input image
def preprocess_image(image):
    image = cv2.resize(image, (96, 96))           # Resize image to match model input
    image = img_to_array(image)                   # Convert to array
    image = np.expand_dims(image, axis=0)         # Expand dimensions for batch
    image = image / 255.0                         # Normalize pixel values
    return image

# Function to make prediction (Fake/Real)
def predict_image(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "Fake" if predicted_class == 0 else "Real"

# Streamlit UI
st.markdown(
    "<h1 style='text-align: center; color: grey;'>DEEP FAKE DETECTION IN SOCIAL MEDIA CONTENT</h1>", 
    unsafe_allow_html=True
)

st.image("coverpage.png")

# Introductory explanation
st.header("Understanding Deepfakes")
st.write("""
Deepfakes are AI-generated synthetic media where one person's likeness is replaced with another's in a video or image. 
While useful in certain fields like entertainment, they pose serious threats in terms of misinformation and digital trust. 
Detection systems like this use AI to recognize subtle artifacts and inconsistencies that distinguish real from fake media.
""")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image for analysis...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read and decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Show the uploaded image
    st.image(image, channels="BGR")

    # Make prediction
    result = predict_image(image)

    # Define result styling and explanation
    if result == "Fake":
        color = "red"
        description = (
            "The model has detected this image as **fake**. Deepfake images often contain artifacts such as mismatched facial features, "
            "unnatural shadows, or inconsistent lighting that reveal tampering. This model was trained to identify such signs with high precision."
        )
    else:
        color = "green"
        description = (
            "The model has detected this image as **real**. Real images usually lack the subtle inconsistencies found in fakes. "
            "Our model's decision is based on learned patterns from a large dataset of authentic and synthetic examples."
        )

    # Show result with color-coded heading and details
    st.markdown(f"<h1 style='color:{color};'>The image is {result}</h1>", unsafe_allow_html=True)
    st.write(description)

# Display model training metrics
st.title("Model Training Performance")
st.markdown("### Accuracy: 95%")
st.image("Figure_2.png")

st.markdown("### Training Loss")
st.image("Figure_1.png")
