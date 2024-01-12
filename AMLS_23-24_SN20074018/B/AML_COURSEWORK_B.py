#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy

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
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 3)),
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

# Train the model
model.fit(train_images, train_labels_one_hot, epochs=10, validation_data=(val_images, val_labels_one_hot))

# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Print shapes of the datasets
print("Shape of training images:", train_images.shape)
print("Shape of validation images:", val_images.shape)
print("Shape of test images:", test_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of validation labels:", val_labels.shape)
print("Shape of test labels:", test_labels.shape)


# In[ ]:




