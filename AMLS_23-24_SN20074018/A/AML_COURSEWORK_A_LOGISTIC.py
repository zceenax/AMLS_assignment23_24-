#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Function to load npz file
def load_npz_file(file_path):
    try:
        data = np.load(file_path)
        print("Arrays in the file:", list(data.keys()))
        for array_name in data:
            print(f"{array_name}:")
            print(data[array_name])
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Replace with your npz file path
npz_file_path = r"C:\Users\An\AMLS_23-24_SN20074018\Datasets\pneumoniamnist.npz"

# Load npz file
loaded_data = load_npz_file(npz_file_path)

# Extract arrays from loaded data
train_images, val_images, test_images = loaded_data['train_images'], loaded_data['val_images'], loaded_data['test_images']
train_labels, val_labels, test_labels = loaded_data['train_labels'], loaded_data['val_labels'], loaded_data['test_labels']

# Flatten images
def flatten_images(images):
    return images.reshape(images.shape[0], -1)

X_train, X_val, X_test = flatten_images(train_images), flatten_images(val_images), flatten_images(test_images)
y_train, y_val, y_test = train_labels.flatten(), val_labels.flatten(), test_labels.flatten()

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Suppress the convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize logistic regression model
model = LogisticRegression(C=0.25, penalty='l1', solver='liblinear', max_iter=500)  # L1 regularization
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate accuracy on the training set
train_predictions = model.predict(X_train_scaled)
accuracy_train = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", accuracy_train)

# Make predictions and evaluate accuracy on the validation set
val_predictions = model.predict(X_val_scaled)
accuracy_val = accuracy_score(y_val, val_predictions)
print("Validation Accuracy:", accuracy_val)

# Make predictions and evaluate accuracy on the test set
test_predictions = model.predict(X_test_scaled)
accuracy_test = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", accuracy_test)

# Print confusion matrix for the test set
conf_matrix_test = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix (Test Set):")
print(conf_matrix_test)


# In[ ]:




