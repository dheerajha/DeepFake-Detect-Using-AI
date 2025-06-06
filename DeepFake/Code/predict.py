# ===============================
# 📦 Imports
# ===============================
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# 🔍 Load Trained Model
# ===============================
model = load_model('deepfake_detection_model.h5')

# ===============================
# 🧼 Image Preprocessing Function
# ===============================
def preprocess_image(image_path):
    """
    Reads and preprocesses an image for model prediction.
    - Resizes to 96x96
    - Converts to array
    - Normalizes pixel values
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize to [0,1]
    return image

# ===============================
# 🧠 Prediction Function
# ===============================
def predict_image(image_path):
    """
    Predicts whether the given image is real or fake.
    Returns: 'Real' or 'Fake'
    """
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "Fake" if predicted_class == 0 else "Real"

# ===============================
# ✅ Example Usage
# ===============================
image_path = "real_and_fake_face_detection/real_and_fake_face/training_real/real_00001.jpg"
result = predict_image(image_path)
print(f"The image is {result}")
