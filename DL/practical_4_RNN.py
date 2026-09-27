"""
Practical No: 04
BSc Data Science (Sem 5) - Deep Learning
AIM: Implement a Recurrent Neural Network (RNN) for next-word prediction using TensorFlow.
Description: This practical demonstrates a SimpleRNN model for learning word relationships 
from a small set of training sentences. Text is tokenized into numerical values, the model is 
trained using input-output word pairs, and the trained RNN predicts the next word for a given 
input word.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

# Fix random results
np.random.seed(42)
tf.random.set_seed(42)

# Training sentences
sentences = [
    "machine learning",
    "deep learning",
    "data science",
    "artificial intelligence",
    "computer vision",
    "natural language",
    "big data",
    "image processing",
    "data mining"
]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

x = []
y = []

for sentence in sentences:
    words = sentence.split()
    x.append(tokenizer.word_index[words[0]])
    y.append(tokenizer.word_index[words[1]])

X = np.array(x)
y = np.array(y)

# Build RNN model
model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=8,
        input_length=1
    ),
    SimpleRNN(units=64),
    Dense(
        units=len(tokenizer.word_index) + 1,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(X, y, epochs=1000, verbose=0)

# Prediction
word = input("Enter first word: ").lower()

if word not in tokenizer.word_index:
    print("Word not found in training data")
else:
    test = np.array([[tokenizer.word_index[word]]])
    prediction = model.predict(test, verbose=0)
    predicted_index = np.argmax(prediction[0])
    for key, value in tokenizer.word_index.items():
        if value == predicted_index:
            print("Next word is:", key)
            break
