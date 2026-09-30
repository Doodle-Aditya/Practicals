"""
Q5. Named Entity Recognition (NER)

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python Named_Entity_Recognition.py

Requirements:
    pip install spacy nltk scikit-learn pandas numpy
    python -m spacy download en_core_web_sm
"""

import pandas as pd
import spacy
from spacy import displacy

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

nlp = spacy.load("en_core_web_sm")
print("spaCy version :", spacy.__version__)
print("Model loaded  :", nlp.meta["name"])

# ======================================================================
# Q5. NAMED ENTITY RECOGNITION (NER)
# ======================================================================
q5_text = ("Sundar Pichai, the CEO of Google, visited Bengaluru on Monday to "
           "meet Mukesh Ambani of Reliance Industries. Google announced an "
           "investment of $10 billion in India over the next 5 years. "
           "The meeting took place at the Taj West End hotel and was "
           "attended by over 200 people from Infosys, Wipro and TCS.")
doc = nlp(q5_text)

# (a) Entities with label and explanation
rows = [{"entity": ent.text,
         "label": ent.label_,
         "meaning": spacy.explain(ent.label_),
         "start_char": ent.start_char,
         "end_char": ent.end_char} for ent in doc.ents]
q5_df = pd.DataFrame(rows)
print("(a) Entities\n", q5_df)

# (b) Count of entities per label
print("\n(b) Entities per label\n", q5_df["label"].value_counts())

# (c) Group entity texts by type for PERSON, ORG, GPE
grouped = {}
for ent in doc.ents:
    if ent.label_ in ("PERSON", "ORG", "GPE"):
        grouped.setdefault(ent.label_, set()).add(ent.text)
print("\n(c) Grouped entities :", grouped)

# (d) Custom entities with EntityRuler (on a separate copy of the model)
nlp_ruler = spacy.load("en_core_web_sm")
ruler = nlp_ruler.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "TECH", "pattern": "Python"},
    {"label": "TECH", "pattern": "spaCy"},
    {"label": "TECH", "pattern": [{"LOWER": "spyder"}, {"LOWER": "ide"}]},
    {"label": "COURSE", "pattern": [{"LOWER": "natural"},
                                    {"LOWER": "language"},
                                    {"LOWER": "processing"}]},
])
q5_custom = ("Students of MIT-WPU Pune use Python, spaCy and Spyder IDE in the "
             "Natural Language Processing course taught by Prof. Kulkarni.")
print("\n(d) Default model :",
      [(e.text, e.label_) for e in nlp(q5_custom).ents])
print("    With ruler    :",
      [(e.text, e.label_) for e in nlp_ruler(q5_custom).ents])

# (e) Visualise with displaCy and save to an HTML file
html = displacy.render(doc, style="ent", page=True, jupyter=False)
with open("q5_entities.html", "w", encoding="utf-8") as f:
    f.write(html)
print("\n(e) Saved 'q5_entities.html' in the working directory. "
      "Open it in a browser.")
