"""
Q7. Bag of Words & Bag of N-grams

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python Bag_of_Words.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_sm
"""

import numpy as np
import pandas as pd
import spacy

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

nlp = spacy.load("en_core_web_sm")
print("spaCy version :", spacy.__version__)
print("Model loaded  :", nlp.meta["name"])

# ======================================================================
# Q7. BAG OF WORDS & BAG OF N-GRAMS
# ======================================================================
from sklearn.feature_extraction.text import CountVectorizer

q7_corpus = [
    "The food was good and the service was good.",
    "The food was bad and the service was slow.",
    "Service was quick but the food was cold.",
    "Good food, good service, good price.",
]


# (a) Bag of Words from scratch
def simple_tokens(text):
    return [t.text.lower() for t in nlp(text) if t.is_alpha]


tokenised = [simple_tokens(d) for d in q7_corpus]
vocab = sorted(set(w for doc_tokens in tokenised for w in doc_tokens))
bow_manual = [[doc_tokens.count(w) for w in vocab] for doc_tokens in tokenised]
bow_manual_df = pd.DataFrame(bow_manual, columns=vocab,
                             index=[f"D{i+1}" for i in range(len(q7_corpus))])
print("(a) Manual Bag of Words\n", bow_manual_df)

# (b) Same with CountVectorizer and verify
cv = CountVectorizer()
X = cv.fit_transform(q7_corpus)
bow_sklearn_df = pd.DataFrame(X.toarray(), columns=cv.get_feature_names_out(),
                              index=bow_manual_df.index)
print("\n(b) CountVectorizer BoW\n", bow_sklearn_df)
print("    Manual == sklearn ? ", bow_manual_df.equals(bow_sklearn_df))
print(f"    Matrix shape {X.shape}, sparsity = "
      f"{100 * (1 - X.nnz / np.prod(X.shape)):.1f}% zeros")

# (c) Binary BoW (presence / absence)
cv_bin = CountVectorizer(binary=True)
print("\n(c) Binary BoW\n",
      pd.DataFrame(cv_bin.fit_transform(q7_corpus).toarray(),
                   columns=cv_bin.get_feature_names_out(),
                   index=bow_manual_df.index))

# (d) Bag of N-grams and vocabulary growth
for ngram in [(1, 1), (2, 2), (1, 2), (1, 3)]:
    cv_n = CountVectorizer(ngram_range=ngram)
    cv_n.fit(q7_corpus)
    print(f"\n(d) ngram_range={ngram} -> vocabulary size "
          f"{len(cv_n.vocabulary_)}")

cv_bi = CountVectorizer(ngram_range=(2, 2))
bi_df = pd.DataFrame(cv_bi.fit_transform(q7_corpus).toarray(),
                     columns=cv_bi.get_feature_names_out(),
                     index=bow_manual_df.index)
print("    Bigram matrix\n", bi_df)

# (e) Why n-grams matter: word order
pair = ["The food was good, not bad.", "The food was bad, not good."]
uni = CountVectorizer().fit(pair)
bi = CountVectorizer(ngram_range=(1, 2)).fit(pair)
u = uni.transform(pair).toarray()
b = bi.transform(pair).toarray()
print("\n(e) Unigram vectors identical?  ", np.array_equal(u[0], u[1]))
print("    Uni+bigram vectors identical?", np.array_equal(b[0], b[1]))
print("    Bag of Words ignores word order, so opposite sentences look the "
      "same. Bigrams like 'not bad' vs 'not good' capture the difference.")

# (f) Unseen words in a new sentence
new_doc = ["The pizza was good but the delivery was late."]
vec = cv.transform(new_doc).toarray()[0]
known = {w: int(c) for w, c in zip(cv.get_feature_names_out(), vec) if c > 0}
unknown = [w for w in simple_tokens(new_doc[0]) if w not in cv.vocabulary_]
print("\n(f) Counted words :", known)
print("    Ignored (OOV) :", unknown)
