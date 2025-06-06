# ===============================
# 📦 Imports and Setup
# ===============================
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook as tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# ===============================
# 📂 Data Loading and Visualization
# ===============================

# Define paths for real and fake face images
real_dir = "real_and_fake_face_detection/real_and_fake_face/training_real/"
fake_dir = "real_and_fake_face_detection/real_and_fake_face/training_fake/"

# Get filenames from both directories
real_images = os.listdir(real_dir)
fake_images = os.listdir(fake_dir)

# Utility function to load and resize image
def load_img(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    return image[..., ::-1]  # Convert BGR to RGB

# Show real face samples
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(real_dir + real_images[i]))
    plt.axis('off')
plt.suptitle("Real Faces", fontsize=20)
plt.show()

# Show fake face samples
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(load_img(fake_dir + fake_images[i]))
    plt.title(fake_images[i][:4])
    plt.axis('off')
plt.suptitle("Fake Faces", fontsize=20)
plt.show()

# ===============================
# 🔄 Data Augmentation
# ===============================

dataset_root = "real_and_fake_face"

# ImageDataGenerator with augmentation and rescaling
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2
)

# Create train and validation generators
train_gen = data_gen.flow_from_directory(
    dataset_root,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = data_gen.flow_from_directory(
    dataset_root,
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ===============================
# 🧠 Model Building
# ===============================

# Load MobileNetV2 as base model (without top layer)
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(96, 96, 3))
base_model.trainable = False  # Freeze base model layers

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(2, activation='softmax')  # 2 classes: Real and Fake
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Show model summary
model.summary()

# ===============================
# 🔁 Training with Learning Rate Scheduler
# ===============================

# Define a custom learning rate scheduler
def lr_schedule(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_scheduler = LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[lr_scheduler]
)

# ===============================
# 💾 Save the Model
# ===============================
model.save('deepfake_detection_model.h5')

# ===============================
# 📊 Visualize Accuracy and Loss
# ===============================

# Extract history
epochs = 20
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(epochs)

# Plot Loss
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, train_loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')
plt.grid(True)
plt.legend()
plt.style.use('classic')

# Plot Accuracy
plt.figure(figsize=(7, 5))
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')
plt.grid(True)
plt.legend()
plt.style.use('classic')

plt.show()
