"""
Q8. TF-IDF and Document Search

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python TF_IDF_Document_Search.py

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
# Q8. TF-IDF AND DOCUMENT SEARCH
# ======================================================================
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

q8_corpus = [
    "India won the cricket match against Australia by six wickets.",
    "Virat Kohli scored a brilliant century in the cricket test match.",
    "The new smartphone has a powerful processor and a great camera.",
    "Apple launched a new laptop with a faster processor and longer battery.",
    "Paneer butter masala and garlic naan are popular Indian dishes.",
    "This restaurant serves delicious Indian food like biryani and dosa.",
]
doc_ids = [f"D{i+1}" for i in range(len(q8_corpus))]


# (a) Preprocess with spaCy: lowercase lemma, no stop words / punctuation
def preprocess(text):
    return " ".join(t.lemma_.lower() for t in nlp(text)
                    if t.is_alpha and not t.is_stop)


clean_corpus = [preprocess(d) for d in q8_corpus]
print("(a) Preprocessed corpus")
for i, c in zip(doc_ids, clean_corpus):
    print(f"    {i}: {c}")

# (b) TF-IDF from scratch (same formula as sklearn's default)
#     tf    = raw count of term in document
#     idf   = ln((1 + N) / (1 + df)) + 1
#     tfidf = tf * idf, then each row is L2-normalised
cv = CountVectorizer()
tf = cv.fit_transform(clean_corpus).toarray()
terms = cv.get_feature_names_out()
N = tf.shape[0]
df = (tf > 0).sum(axis=0)
idf = np.log((1 + N) / (1 + df)) + 1
tfidf_manual = tf * idf
tfidf_manual = tfidf_manual / np.linalg.norm(tfidf_manual, axis=1, keepdims=True)

tfidf_vec = TfidfVectorizer()
tfidf_sklearn = tfidf_vec.fit_transform(clean_corpus).toarray()
print("\n(b) Manual TF-IDF == sklearn TF-IDF ? ",
      np.allclose(tfidf_manual, tfidf_sklearn))

idf_df = pd.DataFrame({"term": terms, "df": df, "idf": np.round(idf, 3)})
print("    Lowest IDF (common) terms\n",
      idf_df.sort_values("idf").head(5).to_string(index=False))

tfidf_df = pd.DataFrame(np.round(tfidf_sklearn, 3),
                        columns=tfidf_vec.get_feature_names_out(),
                        index=doc_ids)
print("\n    TF-IDF matrix\n", tfidf_df)

# (c) Top 3 keywords per document
print("\n(c) Top 3 keywords per document")
for i in doc_ids:
    top = tfidf_df.loc[i].sort_values(ascending=False).head(3)
    print(f"    {i}: {list(top.index)}")

# (d) Document-to-document cosine similarity
sim = pd.DataFrame(np.round(cosine_similarity(tfidf_sklearn), 2),
                   index=doc_ids, columns=doc_ids)
print("\n(d) Cosine similarity between documents\n", sim)
sim_values = sim.to_numpy(copy=True)
np.fill_diagonal(sim_values, 0)          # ignore self-similarity
r, c = np.unravel_index(sim_values.argmax(), sim_values.shape)
print(f"    Most similar pair : {doc_ids[r]} & {doc_ids[c]}")


# (e) Simple search engine
def search(query, top_k=3):
    q_vec = tfidf_vec.transform([preprocess(query)])
    scores = cosine_similarity(q_vec, tfidf_sklearn)[0]
    ranked = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[r], round(float(scores[r]), 3), q8_corpus[r])
            for r in ranked]


for query in ["cricket match score", "laptop with good battery",
              "tasty Indian dishes"]:
    print(f"\n(e) Query: '{query}'")
    for did, score, text in search(query):
        print(f"    {did} score={score:<6} {text}")

# (f) Compare with plain Bag of Words for the same query
bow_scores = cosine_similarity(cv.transform([preprocess("Indian food")]),
                               tf)[0]
tfidf_scores = cosine_similarity(
    tfidf_vec.transform([preprocess("Indian food")]), tfidf_sklearn)[0]
print("\n(f) Query 'Indian food' BoW vs TF-IDF scores")
print(pd.DataFrame({"BoW": np.round(bow_scores, 3),
                    "TF-IDF": np.round(tfidf_scores, 3)}, index=doc_ids))
i5, i6 = doc_ids.index("D5"), doc_ids.index("D6")
print(f"    D6 / D5 score ratio -> BoW: {bow_scores[i6] / bow_scores[i5]:.2f}x, "
      f"TF-IDF: {tfidf_scores[i6] / tfidf_scores[i5]:.2f}x")
print("    'food' (in 1 document) is rarer than 'indian' (in 2 documents), so "
      "TF-IDF gives it more weight and D6 pulls further ahead of D5.")
print("\n================ END OF PRACTICAL ================")
