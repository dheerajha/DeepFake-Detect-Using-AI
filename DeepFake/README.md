# 🧠 DeepFake Detection in Social Media Content

## 📌 Overview

This project leverages deep learning to **detect deepfake images** — AI-generated media where a person’s likeness is synthetically altered. With increasing use of deepfakes in misinformation, this tool aims to provide a reliable method to distinguish real images from manipulated ones using a Convolutional Neural Network (CNN) built on top of **MobileNetV2**.

## 🚀 Features

- ✅ Upload and analyze any face image using a clean **Streamlit web app**
- 🧠 Predict whether the uploaded image is **Real** or **Fake**
- 📊 View training metrics such as **accuracy** and **loss** with visual plots
- 🔐 Powered by a custom-trained MobileNetV2 deep learning model
- 📦 Lightweight interface using **TensorFlow**, **OpenCV**, and **Streamlit**


## 2️⃣ Install Dependencies
Make sure you have Python 3.7+ and install required libraries:

- pip install -r requirements.txt
## Sample requirements.txt:

- nginx
- tensorflow
- numpy
- opencv-python
- streamlit
- matplotlib

## 3️⃣ Run the Streamlit App

- streamlit run app.py
- Then open your browser and navigate to: http://localhost:8501

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)

## Custom Layers:

- GlobalAveragePooling2D

- Dense(512, relu) → BatchNormalization → Dropout(0.3)

- Dense(128, relu) → Dropout(0.1)

- Dense(2, softmax)

## Loss Function:
- SparseCategoricalCrossentropy

## Optimizer:
- Adam with learning rate scheduler

## 📈 Model Performance
- Validation Accuracy: 95%