#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_npz_file(file_path):
    try:
        # Use NumPy's load function to load npz file
        data = np.load(file_path)

        # Output the names of all arrays in the file
        print("Arrays in the file:", list(data.keys()))

        # Use specific keys to access images and labels
        images = data['train_images']
        labels = data['train_labels']

        # Return the loaded data
        return images, labels
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Replace with your npz file path using two backslashes or a raw string
npz_file_path = r"C:\Users\An\AMLS_23-24_SN20074018\Datasets\pneumoniamnist.npz"

# Call the function to load the npz file
images, labels = load_npz_file(npz_file_path)

# Assuming 'normal' and 'pneumonia' are the keys in your npz file
X = images
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_acc}")

# Make predictions on new data
# new_data = ... # Replace with your new data
# predictions = model.predict(new_data)


# In[ ]:




