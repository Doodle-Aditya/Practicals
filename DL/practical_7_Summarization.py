"""
Practical 7
TY BSC Data Science - Deep Learning
AIM: Implement extractive and abstractive summarization using NLP models.
Description: In this practical, a Transformer-based model is implemented for language translation using an 
encoder-decoder architecture. The model uses embedding layers, multi-head attention, self-attention, cross-attention, 
and feed-forward layers to learn the relationship between English and French sentences and generate translated output.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


english = ['hello', 'thanks', 'yes', 'no', 'water', 'bread', 'morning', 'night']
french = ['banjour', 'merci', 'oui', 'non', 'eau', 'pain', 'matin', 'nuit']


eng = layers.TextVectorization(output_sequence_length=1)
fra = layers.TextVectorization(output_sequence_length=1)

eng.adapt(english)
fra.adapt(french)

X = eng(english)
y = fra(french)

eng_vocab = len(eng.get_vocabulary())
fra_vocab = len(fra.get_vocabulary())

inp = layers.Input(shape=(1,))
h = layers.Enbedding(eng_vocab, 32)(inp)
h = layers.flatten()(h)
h = layers.Dense(32, activation='relu')(h)
output = layers.Dense(fra_vocab, activation='softmax')(h)

model = tf.keras.Model(inp, output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])


model.fit(X, y, epochs=300, verbose=0)
print("Model trained successfully!")

## Translation Function
def translate(word):
    encoded = eng([word])
    prediction = model.predict(encoded, verbose=0)
    word_id = np.argmax(prediction[0])
    return fra.get_vocabulary()[word_id]

for word in english:
    print("English: ", word)
    print("French: ", translate(word))
    print()