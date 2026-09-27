"""
Practical 8
TY BSC Data Science - Deep Learning
AIM: Train a simple Neural Machine Translation (NTM) model using Tensorflow.
Description: In this practical, different Transformer model configurations are tested by changing 
hyperparameters such as embedding dimension, number of attention heads, learning rate, and number of 
epochs. The accuracy of each configuration is compared, and the model with the best accuracy is selected as 
the best model.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# DATASET
english = [
    "hello",
    "good morning",
    "thankyou",
    "how are you",
    "i am a student",
    "good night"
]

french = [
    "bonjour",
    "bonjour",
    "merci",
    "comment allez vous",
    "je suis etudiant",
    "bonne nuit"
]

# Tokenization
eng = layers.TextVectorization(output_sequence_length=6)
fra = layers.TextVectorization(output_sequence_length=6)

eng.adapt(english)
fra.adapt(french)

X = eng(english)
Y = fra(french)

vocab = len(fra.get_vocabulary())

# Function to create Transformer
def create_model(embedding_dim, heads, learning_rate):
    inputs = layers.Input(shape=(6,))
    x = layers.Embedding(
        len(eng.get_vocabulary()),
        embedding_dim
    )(inputs)

    attention = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=embedding_dim
    )(x, x)

    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    output = layers.Dense(
        vocab,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Different hyperparameter combinations
settings = [
    (16, 1, 0.01, 100),
    (32, 2, 0.001, 200),
    (64, 2, 0.001, 200),
    (32, 4, 0.0005, 300)
]

# Store best model
best_accuracy = 0
best_model = None
best_settings = None

print("HYPERPARAMETER EXPERIMENT\n")

# Experiment
for embedding_dim, heads, lr, epochs in settings:
    model = create_model(
        embedding_dim,
        heads,
        lr
    )

    history = model.fit(
        X,
        Y,
        epochs=epochs,
        verbose=0
    )

    accuracy = history.history["accuracy"][-1]

    print(
        "Embedding:", embedding_dim,
        "| Heads:", heads,
        "| Learning Rate:", lr,
        "| Epochs:", epochs,
        "| Accuracy:", round(accuracy, 4)
    )

    # Find best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_settings = (
            embedding_dim,
            heads,
            lr,
            epochs
        )

# Display best result
print("\n==============================")
print("BEST MODEL")
print("==============================")
print(
    "Embedding Dimension:",
    best_settings[0]
)
print(
    "Attention Heads:",
    best_settings[1]
)
print(
    "Learning Rate:",
    best_settings[2]
)
print(
    "Epochs:",
    best_settings[3]
)
print(
    "Best Accuracy:",
    round(best_accuracy, 4)
)

# Prediction using best model
sentence = "hello"
test_input = eng([sentence])
prediction = best_model.predict(
    test_input,
    verbose=0
)

# Get predicted word
word_id = np.argmax(
    prediction[0, 0]
)

predicted_word = fra.get_vocabulary()[word_id]

print("\n============================================")
print("PREDICTION")
print("\n============================================")
print("English:", sentence)
print("French:", predicted_word)
