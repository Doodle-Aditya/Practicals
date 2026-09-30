"""
Q2. The spaCy Pipeline

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python spaCy_Pipeline.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_sm
"""

import time
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

nlp = spacy.load("en_core_web_sm")
print("spaCy version :", spacy.__version__)
print("Model loaded  :", nlp.meta["name"])

# ======================================================================
# Q2. THE SPACY PIPELINE
# ======================================================================
q2_reviews = [
    "The delivery was quick and the packaging was excellent.",
    "Worst customer service I have ever experienced.",
    "Battery life of this phone is amazing, lasts two full days.",
    "The laptop heats up quickly and the fan is very noisy.",
    "Good value for money, I would recommend it to my friends.",
]

# (a) Inspect the pipeline
print("(a) Pipeline components :", nlp.pipe_names)
for name, component in nlp.pipeline:
    print(f"    {name:<12} -> {type(component).__name__}")
print("    Note: the tokenizer always runs first and is not listed in pipe_names.")

# (b) Speed comparison: full pipeline vs only what is needed
big_corpus = q2_reviews * 200            # 1000 short documents

list(nlp.pipe(q2_reviews))               # warm-up run (loads caches, not timed)

start = time.perf_counter()
full_docs = list(nlp.pipe(big_corpus))
full_time = time.perf_counter() - start

with nlp.select_pipes(disable=["parser", "ner"]):
    start = time.perf_counter()
    light_docs = list(nlp.pipe(big_corpus))
    light_time = time.perf_counter() - start

print(f"\n(b) Full pipeline          : {full_time:.3f} s")
print(f"    Without parser and ner : {light_time:.3f} s")
print(f"    Speed-up               : {full_time / light_time:.2f}x")
print("    Entities available after disabling ner? ",
      light_docs[0].has_annotation("ENT_IOB"))

# (c) Custom component: word count and estimated reading time
if not Doc.has_extension("word_count"):
    Doc.set_extension("word_count", default=0)
if not Doc.has_extension("reading_time_sec"):
    Doc.set_extension("reading_time_sec", default=0.0)


@Language.component("text_stats")
def text_stats(doc):
    words = [t for t in doc if t.is_alpha]
    doc._.word_count = len(words)
    doc._.reading_time_sec = round(len(words) / 200 * 60, 2)   # 200 words/min
    return doc


if "text_stats" not in nlp.pipe_names:
    nlp.add_pipe("text_stats", last=True)
print("\n(c) Pipeline after adding custom component :", nlp.pipe_names)

# (d) Batch processing with nlp.pipe()
print("\n(d) Batch results")
for review_doc in nlp.pipe(q2_reviews):
    print(f"    words={review_doc._.word_count:>2} | "
          f"read={review_doc._.reading_time_sec:>4}s | {review_doc.text}")

# Clean up so later questions use the standard pipeline
nlp.remove_pipe("text_stats")
