"""
Q4. Part-of-Speech (POS) Tagging

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python POS_Tagging.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_sm
"""

from collections import Counter
import pandas as pd
import spacy

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

nlp = spacy.load("en_core_web_sm")
print("spaCy version :", spacy.__version__)
print("Model loaded  :", nlp.meta["name"])

# ======================================================================
# Q4. PART-OF-SPEECH (POS) TAGGING
# ======================================================================
q4_text = ("The young engineer quickly wrote clean Python code for the new "
           "banking app. Her manager was very happy and gave her a small "
           "bonus. The team will release the beautiful dashboard next week.")
doc = nlp(q4_text)

# (a) POS table with explanation
rows = [{"token": t.text,
         "pos": t.pos_,
         "tag": t.tag_,
         "explanation": spacy.explain(t.tag_)} for t in doc]
print("(a) POS tags\n", pd.DataFrame(rows))

# (b) POS frequency
pos_counts = Counter(t.pos_ for t in doc if not t.is_punct)
print("\n(b) POS frequency :", pos_counts.most_common())

# (c) Nouns, verbs (as lemmas) and adjectives
nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
adjectives = [t.text for t in doc if t.pos_ == "ADJ"]
print("\n(c) Nouns      :", nouns)
print("    Verbs      :", verbs)
print("    Adjectives :", adjectives)

# (d) Adjective -> noun pairs using the dependency tree
pairs = [(t.text, t.head.text) for t in doc
         if t.pos_ == "ADJ" and t.dep_ == "amod"]
print("\n(d) Adjective-noun pairs :", pairs)

# (e) Noun chunks
print("\n(e) Noun chunks :", [chunk.text for chunk in doc.noun_chunks])

# (f) Same word, different POS depending on context
print("\n(f) Ambiguity")
for s in ["I will book a flight to Delhi.", "I am reading a good book."]:
    for t in nlp(s):
        if t.text == "book":
            print(f"    '{s}' -> book = {t.pos_} ({spacy.explain(t.pos_)})")
