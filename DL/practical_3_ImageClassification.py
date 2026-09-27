"""
Practical No: 03
BSc Data Science (Sem 5) - Deep Learning
Aim: To Implement a CNN for image classification using the CIFAR-10 dataset
"""

# Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize images
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Class labels
classes = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Display one image
plt.figure(figsize=(4, 4))
plt.imshow(x_train[0], interpolation="nearest")
plt.title(classes[y_train[0][0]])
plt.axis("off")
plt.show()

# Build CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot accuracy
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# Prediction
prediction = model.predict(x_test[:1], verbose=0)
predicted = prediction.argmax()
print("Predicted:", classes[predicted])
print("Actual:", classes[y_test[0][0]])

plt.figure(figsize=(4, 4))
plt.imshow(x_test[0], interpolation="nearest")
plt.title("Prediction: " + classes[predicted])
plt.axis("off")
plt.show()
