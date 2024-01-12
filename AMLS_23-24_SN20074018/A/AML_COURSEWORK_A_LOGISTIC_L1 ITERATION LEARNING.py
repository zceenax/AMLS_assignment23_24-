#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
npz_file_path = r"C:\Users\An\AMLS_23-24_SN20074018\Datasets\pneumoniamnist.npz"  # Update this path

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

# Set different max iteration values
max_iter_values = [10, 50, 100, 500, 1000, 2000]
train_accuracies = []
val_accuracies = []

# Train models with different max_iter values and record accuracies
for max_iter in max_iter_values:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train_scaled, y_train)
    
    # Training accuracy
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_accuracies.append(train_accuracy)
    
    # Validation accuracy
    val_predictions = model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_accuracies.append(val_accuracy)

# Plot the learning curve
plt.plot(max_iter_values, train_accuracies, label='Training Accuracy')
plt.plot(max_iter_values, val_accuracies, label='Validation Accuracy')
plt.xlabel('Max Iterations')
plt.ylabel('Accuracy')
plt.title('Learning Curve for Logistic Regression')
plt.legend()
plt.show()


# In[ ]:




