"""
Q1. Tokenization using spaCy

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python Tokenization.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_sm
"""

import pandas as pd
import spacy

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

nlp = spacy.load("en_core_web_sm")
print("spaCy version :", spacy.__version__)
print("Model loaded  :", nlp.meta["name"])

# ======================================================================
# Q1. TOKENIZATION USING SPACY
# ======================================================================
q1_text = ("Dr. Sharma paid Rs. 1,250.50 for 3 books at Crossword, Mumbai on "
           "15-Aug-2024! Email her at sharma.r@gmail.com or visit "
           "https://www.example.com for details. Isn't it amazing? "
           "She didn't expect a 20% discount.")
doc = nlp(q1_text)

# (a) Print every token with its index
print("(a) Tokens")
for token in doc:
    print(f"{token.i:>3} | {token.text}")

# (b) Token attribute table
rows = []
for token in doc:
    rows.append({
        "token": token.text,
        "char_idx": token.idx,
        "is_alpha": token.is_alpha,
        "is_punct": token.is_punct,
        "like_num": token.like_num,
        "like_email": token.like_email,
        "like_url": token.like_url,
        "is_currency": token.is_currency,
        "shape": token.shape_,
    })
q1_df = pd.DataFrame(rows)
print("\n(b) Token attributes\n", q1_df)

# (c) Counts
total_tokens = len(doc)
word_tokens = sum(1 for t in doc if t.is_alpha)
punct_tokens = sum(1 for t in doc if t.is_punct)
num_tokens = sum(1 for t in doc if t.like_num)
print(f"\n(c) Total tokens : {total_tokens}")
print(f"    Word tokens  : {word_tokens}")
print(f"    Punctuation  : {punct_tokens}")
print(f"    Numbers      : {num_tokens}")

# (d) Sentence segmentation
print("\n(d) Sentences")
for i, sent in enumerate(doc.sents, start=1):
    print(f"    S{i}: {sent.text}")

# (e) spaCy tokenizer vs Python split()
split_tokens = q1_text.split()
spacy_tokens = [t.text for t in doc]
print(f"\n(e) str.split() tokens : {len(split_tokens)}")
print(f"    spaCy tokens       : {len(spacy_tokens)}")
print("    split() words that spaCy breaks apart :",
      [w for w in split_tokens if w not in spacy_tokens])
print("    Observation: spaCy separates punctuation and splits contractions "
      "(Isn't -> Is + n't), but keeps emails, URLs and 'Dr.' as single tokens.")
