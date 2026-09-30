"""
Q6. Word Embeddings

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python Word_Embeddings.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_lg  (python -m spacy download en_core_web_lg, ~400 MB)
"""

import numpy as np
import pandas as pd
import spacy

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

# ======================================================================
# Q6. WORD EMBEDDINGS
# ======================================================================
nlp_lg = spacy.load("en_core_web_lg")

# (a) Inspect a vector
king = nlp_lg.vocab["king"]
print("(a) Vector dimension of 'king' :", king.vector.shape)
print("    First 5 values :", np.round(king.vector[:5], 3))
for w in ["king", "mumbai", "xyzqwe"]:
    lex = nlp_lg.vocab[w]
    print(f"    {w:<8} has_vector={lex.has_vector} is_oov={lex.is_oov}")

# (b) Similarity matrix
q6_words = ["king", "queen", "man", "woman", "apple", "mango", "car", "bus"]
tokens = [nlp_lg(w)[0] for w in q6_words]
sim_matrix = pd.DataFrame(
    [[round(t1.similarity(t2), 2) for t2 in tokens] for t1 in tokens],
    index=q6_words, columns=q6_words)
print("\n(b) Similarity matrix\n", sim_matrix)

# (c) Nearest neighbours (helper functions rank words by cosine similarity)
candidates = ["doctor", "nurse", "hospital", "patient", "medicine", "surgeon",
              "cricket", "football", "batsman", "stadium", "tennis", "match",
              "python", "java", "code", "programming", "software", "snake",
              "king", "queen", "prince", "princess", "man", "woman", "boy",
              "girl", "apple", "mango", "banana", "car", "bus", "train"]


def cosine(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def most_similar(vector, n=5, exclude=()):
    scores = [(w, round(cosine(vector, nlp_lg.vocab[w].vector), 3))
              for w in candidates if w not in exclude]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:n]


for w in ["doctor", "cricket", "python"]:
    print(f"\n(c) Most similar to '{w}' :",
          most_similar(nlp_lg.vocab[w].vector, exclude=[w]))

# (d) Word analogy: king - man + woman ~ queen
analogy = (nlp_lg.vocab["king"].vector - nlp_lg.vocab["man"].vector
           + nlp_lg.vocab["woman"].vector)
print("\n(d) king - man + woman ~",
      most_similar(analogy, n=3, exclude=["king", "man", "woman"]))

# (e) Sentence similarity and how a doc vector is built
s1 = nlp_lg("I love eating spicy Indian food.")
s2 = nlp_lg("Delicious curry and biryani are my favourite meals.")
s3 = nlp_lg("The stock market crashed badly today.")
print(f"\n(e) s1 vs s2 : {s1.similarity(s2):.3f}")
print(f"    s1 vs s3 : {s1.similarity(s3):.3f}")
manual_avg = np.mean([t.vector for t in s1], axis=0)
print("    Doc vector == average of token vectors ? ",
      np.allclose(manual_avg, s1.vector))
