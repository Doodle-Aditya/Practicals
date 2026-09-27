"""
Topic: TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF is a text representation technique that measures how important a word
is within a document relative to a collection of documents (corpus).

A word is considered important if it appears frequently in one document but
does NOT appear frequently across all documents in the corpus.

TF (Term Frequency):
    TF(t, d) = (Number of times term t appears in document d) / (Total number of words in d)
"""

from collections import Counter

# ---------------------------------------------------------------------------
# 1. Term Frequency on a single block of text
# ---------------------------------------------------------------------------
document = "Thor eating pizza, Loki is eating pizza, Ironman ate pizza already"

words = document.split()
print("Tokens:", words)

word_counts = Counter(words)
print("\nWord counts:", word_counts)

print("\nTerm Frequency (TF) for each word:")
for word, count in word_counts.items():
    tf = count / len(words)
    print(f"{word:10} : {tf}")

# ---------------------------------------------------------------------------
# 2. Term Frequency of a specific word across multiple documents
# ---------------------------------------------------------------------------
documents = [
    "the phone has a good camera",
    "the phone has a good battery",
    "The camera quality of this phone is excellent",
]

target_word = "phone"

print(f"\nTerm Frequency of '{target_word}' across documents:")
for i, doc in enumerate(documents, start=1):
    doc_words = doc.split()
    word_count = doc_words.count(target_word)
    tf = word_count / len(doc_words)

    print(f"\nDocument {i}")
    print("Total words :", len(doc_words))
    print("Word count  :", word_count)
    print("TF          :", tf)
