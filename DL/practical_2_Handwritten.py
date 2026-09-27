"""
Practical No: 02
BSc Data Science (Sem 5) - Deep Learning
Aim: To Implement a DNN For handwritten digit classification using the MNIST dataset
with Dropout, Batch Normalization, and Batch Processing techniques in TensorFlow
"""

# Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.datasets import mnist

# Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training Images :", X_train.shape)
print("Testing Images :", X_test.shape)

# Display Sample Image
plt.imshow(X_train[0], cmap='gray')
plt.title("Sample Image\nLabel = " + str(y_train[0]))
plt.axis('off')
plt.show()

# Normalize Images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build Deep Learning Model
model = Sequential([
    Input(shape=(28, 28)),
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary
print("\nModel Summary")
model.summary()

# Train Model
history = model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict First 5 Test Images
predictions = model.predict(X_test[:5])
print("\nPrediction Results")

plt.figure(figsize=(12, 3))
for i in range(5):
    predicted = np.argmax(predictions[i])
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Predicted: {predicted}\nActual: {y_test[i]}")
    plt.axis('off')
    print("--------------")
    print("Image", i + 1)
    print("Predicted:", predicted)
    print("Actual:", y_test[i])
    print("--------------")
plt.show()

# Accuracy Graph
plt.figure(figsize=(12, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
