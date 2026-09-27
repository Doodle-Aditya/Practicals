"""
Topic: Text Representation - Bag of Words (BOW) for Spam Classification

Uses scikit-learn's CountVectorizer to convert text messages into a numeric
Bag-of-Words representation, then trains a Multinomial Naive Bayes classifier
to detect spam vs. ham (non-spam) messages.

Requirements:
    pip install pandas numpy scikit-learn

Expects a CSV file named "spam.csv" with columns: Category, Message
(e.g. the classic UCI SMS Spam Collection dataset).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Load and inspect the data
# ---------------------------------------------------------------------------
df = pd.read_csv("spam.csv")
print(df.head())

print("\nClass distribution:")
print(df.Category.value_counts())  # data is imbalanced (more ham than spam)

# Convert the categorical label into a numeric target: spam -> 1, ham -> 0
df["spam"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)
print("\n", df.head())

# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df.Message, df.spam, test_size=0.2, random_state=42
)

print("\nX_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

# ---------------------------------------------------------------------------
# 3. Bag-of-Words representation with CountVectorizer
# ---------------------------------------------------------------------------
v = CountVectorizer()

X_train_cv = v.fit_transform(X_train.values)
print("\nBOW matrix shape (train):", X_train_cv.shape)

vocab = v.get_feature_names_out()
print("Vocabulary size:", vocab.shape[0])
print("Sample vocabulary words:", vocab[1000:1010])

# Vector representation of a single message
X_train_np = X_train_cv.toarray()
print("\nFirst message vector (non-zero word indexes):")
print(np.where(X_train_np[0] != 0))

# ---------------------------------------------------------------------------
# 4. Train a Multinomial Naive Bayes model
# ---------------------------------------------------------------------------
model = MultinomialNB()
model.fit(X_train_cv, y_train)

# Transform the test set using the SAME fitted vectorizer (don't fit again)
X_test_cv = v.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Evaluate performance
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_cv)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# 6. Try the model on new, unseen messages
# ---------------------------------------------------------------------------
emails = [
    "Hey mohan, can we get together to watch football game tomorrow?",
    "Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!",
]

emails_count = v.transform(emails)
print("\nPredictions for new messages (0 = ham, 1 = spam):")
print(model.predict(emails_count))

# ---------------------------------------------------------------------------
# 7. Same workflow using an sklearn Pipeline (cleaner, fewer lines of code)
# ---------------------------------------------------------------------------
clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("nb", MultinomialNB()),
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification report (pipeline version):")
print(classification_report(y_test, y_pred))
