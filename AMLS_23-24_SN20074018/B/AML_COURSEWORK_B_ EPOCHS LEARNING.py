#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt

# Load the npz file
npz_file_path = r"C:\Users\An\AMLS_23-24_SN20074018\Datasets\pathMNIST.npz"
data = np.load(npz_file_path, allow_pickle=True)

# Extract arrays from the npz file
train_images = data['train_images'] / 255.0
val_images = data['val_images'] / 255.0
test_images = data['test_images'] / 255.0
train_labels = data['train_labels']
val_labels = data['val_labels']
test_labels = data['test_labels']

# One-hot encode labels
train_labels_one_hot = to_categorical(train_labels, num_classes=9)
val_labels_one_hot = to_categorical(val_labels, num_classes=9)
test_labels_one_hot = to_categorical(test_labels, num_classes=9)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax')  # 9 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model and save the history
history = model.fit(train_images, train_labels_one_hot, epochs=10, validation_data=(val_images, val_labels_one_hot))

# Plotting the learning curves
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()


# In[ ]:




