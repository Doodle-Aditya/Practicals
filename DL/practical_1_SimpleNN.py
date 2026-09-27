"""
Practical No: 01
BSc Data Science (Sem 5) - Deep Learning
AIM: Build and train a simple neural network for classification tasks.
Description: This practical demonstrates the implementation of a simple Artificial Neural
Network (ANN) using TensorFlow and Keras to classify Fashion MNIST images. The
dataset is preprocessed, the model is trained, and its performance is evaluated using
test data.
"""

# ==========================================
# Fashion MNIST Classification using CSV Files
# Latest Keras Syntax
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense

# Load Dataset
# Make sure these CSV files are in the same folder as this Python file.
# Otherwise, provide the full path.
train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

# Separate Features and Labels
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Normalize Pixel Values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape Images to 28 x 28
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Build Neural Network (Latest Keras Syntax)
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate the Model
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print("\n===================================")
print("Test Loss :", loss)
print("Test Accuracy :", accuracy)
print("===================================\n")

# Make Predictions
predictions = model.predict(
    X_test,
    verbose=0
)

# Convert probabilities into class labels
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted Classes:")
print(predicted_classes[:10])

print("\nActual Classes:")
print(y_test[:10])

# Class Names
class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

print("\n========== First 10 Predictions ==========\n")
for i in range(10):
    print(f"Image {i+1}")
    print("Actual    :", class_names[y_test[i]])
    print("Predicted :", class_names[predicted_classes[i]])
    print("------------------------------------------")
