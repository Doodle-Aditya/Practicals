"""
Practical 5
TY BSC Data Science - Deep Learning
AIM: Construct an autoencoder for image denoising applications
Description: This practical demonstrates a Denoising Autoencoder using TensorFlow and the Fashion MNIST 
dataset. The model is trained to remove noise from images and reconstruct clean images using an encoder-decoder 
neural network.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load Fashion MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten images
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# Add random noise
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

# Keep pixel values between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# --------- Autoencoder ---------
input_img = Input(shape=(784,))

# Encoder
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)

# Decoder
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Build Model
autoencoder = Model(input_img, decoded)

# Compile
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Train
autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=0
)

# Predict
decoded_imgs = autoencoder.predict(x_test_noisy)

# Display Results
plt.figure(figsize=(9, 4))
for i in range(3):
    # Noisy Image
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    # Denoised Image
    plt.subplot(2, 3, i + 4)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Denoised")
    plt.axis("off")

plt.show()
