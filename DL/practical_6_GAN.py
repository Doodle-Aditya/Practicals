"""
Practical 6
TY BSC Data Science - Deep Learning
AIM: Develop a GAN to generate synthetic images.
Description: This practical implements a Generative Adversarial Network (GAN) using the MNIST dataset. The 
GAN consists of a Generator that creates synthetic handwritten digits and a Discriminator that identifies 
whether an image is real or generated. Both models are trained together so that the generator gradually 
produces more realistic images.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 1. Load and preprocess MNIST
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, -1)

# Parameters
noise_dim = 100
batch_size = 128
epochs = 30

# 2. Generator
generator = tf.keras.Sequential([
    layers.Input(shape=(noise_dim,)),
    layers.Dense(7 * 7 * 128),
    layers.LeakyReLU(alpha=0.2),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(1, 7, padding="same", activation="tanh")
])

# 3. Discriminator
discriminator = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.3),
    layers.Conv2D(128, 4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
])

# 4. Loss and optimizers
loss = tf.keras.losses.BinaryCrossentropy()
g_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)
d_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

# 5. Dataset
data = tf.data.Dataset.from_tensor_slices(x_train)
data = data.shuffle(60000).batch(batch_size)

# 6. Training function
@tf.function
def train_step(real):
    noise = tf.random.normal(
        [tf.shape(real)[0], noise_dim]
    )

    # Train Discriminator
    with tf.GradientTape() as tape:
        fake = generator(noise, training=True)
        real_output = discriminator(
            real, training=True
        )
        fake_output = discriminator(
            fake, training=True
        )
        real_loss = loss(
            tf.ones_like(real_output),
            real_output
        )
        fake_loss = loss(
            tf.zeros_like(fake_output),
            fake_output
        )
        d_loss = real_loss + fake_loss

    gradients = tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )
    d_optimizer.apply_gradients(
        zip(
            gradients,
            discriminator.trainable_variables
        )
    )

    # Train Generator
    with tf.GradientTape() as tape:
        fake = generator(
            noise, training=True
        )
        fake_output = discriminator(
            fake, training=True
        )
        g_loss = loss(
            tf.ones_like(fake_output),
            fake_output
        )

    gradients = tape.gradient(
        g_loss,
        generator.trainable_variables
    )
    g_optimizer.apply_gradients(
        zip(
            gradients,
            generator.trainable_variables
        )
    )

    return g_loss, d_loss

# 7. Train GAN
for epoch in range(1, epochs + 1):
    for batch in data:
        g_loss, d_loss = train_step(batch)

    print(
        "Epoch:", epoch,
        "| Generator Loss:", round(float(g_loss), 4),
        "| Discriminator Loss:", round(float(d_loss), 4)
    )

    # Show images every 5 epochs
    if epoch % 5 == 0:
        noise = tf.random.normal(
            [16, noise_dim]
        )
        images = generator(
            noise, training=False
        )
        images = (images + 1) / 2
        plt.figure(figsize=(6, 6))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(
                images[i, :, :, 0],
                cmap="gray"
            )
            plt.axis("off")
        plt.suptitle(
            "Generated Images - Epoch " + str(epoch)
        )
        plt.tight_layout()
        plt.show()

# 8. Final images
noise = tf.random.normal([16, noise_dim])
images = generator(
    noise, training=False
)
images = (images + 1) / 2
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(
        images[i, :, :, 0],
        cmap="gray"
    )
    plt.axis("off")
plt.suptitle("Final Synthetic Images")
plt.tight_layout()
plt.show()
