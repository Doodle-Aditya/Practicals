"""
Topic: Named Entity Recognition (NER) using spaCy

Named Entity Recognition identifies real-world objects/entities in text such
as people, organizations, locations, money, dates, products, etc.

Requirements:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import spacy
from spacy import displacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
print("Pipeline components:", nlp.pipe_names)

# ---------------------------------------------------------------------------
# 1. Basic NER
# ---------------------------------------------------------------------------
doc = nlp("Tesla Inc. is going to acquire Twitter Inc. for $45 billion")
print("\nBasic NER:")
for ent in doc.ents:
    print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))

# Visualize entities (renders an HTML visualization; useful in Jupyter/Colab)
# displacy.render(doc, style="ent")

# ---------------------------------------------------------------------------
# 2. List all entity labels spaCy supports
# ---------------------------------------------------------------------------
print("\nAll supported entity labels:", nlp.pipe_labels["ner"])
print("Total number of entity labels:", len(nlp.pipe_labels["ner"]))

# ---------------------------------------------------------------------------
# 3. NER depends on how the model was trained -- it can make mistakes
# ---------------------------------------------------------------------------
doc = nlp("Michael Bloomberg founded Bloomberg in 1982")
print("\nModel may misclassify entities depending on training data:")
for ent in doc.ents:
    print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))

# ---------------------------------------------------------------------------
# 4. Entity character offsets
# ---------------------------------------------------------------------------
doc = nlp(
    "Tesla Inc. is going to acquire Twitter Inc. for $45 billion. "
    "Tesla Inc. is the unicorn of Silicon Valley"
)
print("\nEntities with character offsets:")
for ent in doc.ents:
    print(ent.text, "|", ent.label_, "|", ent.start_char, "|", ent.end_char)

# ---------------------------------------------------------------------------
# 5. Setting custom entities manually using Span
# ---------------------------------------------------------------------------
doc = nlp("Tesla is going to acquire twitter for $45 billion")
print("\nBefore setting custom entities:")
for ent in doc.ents:
    print(ent.text, "|", ent.label_)  # Tesla/Twitter not recognized as ORG

s1 = Span(doc, 0, 1, label="ORG")  # "Tesla"
s2 = Span(doc, 5, 6, label="ORG")  # "twitter"
doc.set_ents([s1, s2], default="unmodified")

print("\nAfter setting custom entities:")
for ent in doc.ents:
    print(ent.text, "|", ent.label_)

# ---------------------------------------------------------------------------
# 6. Entity type examples
# ---------------------------------------------------------------------------
examples = {
    "PERSON, ORGANIZATION, LOCATION": "Sundar Pichai is CEO of google and lives in California",
    "MONEY": "Apple earned $120 Billion in revenue last year",
    "DATES": "The meeting is scheduled on 15th August 2026 at 10 AM",
    "COUNTRIES": "Amazon has offices in India, Canada, Germany and Japan",
    "PRODUCTS": "I bought Iphone 17 Pro Max from apple store",
    "SPORTS": "Rohit Sharma scored a century at Lords against England",
    "PERCENTAGE": "India's GDP grew by 8.2% in 2024",
    "QUANTITIES": "The package weighs 25 kilograms and cost $120",
    "FAC (facilities)": "The flight landed at International Airport",
    "CARDINAL": "There are 150 students in the class",
}

for category, text in examples.items():
    print(f"\n{category}:")
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, "|", ent.label_, "|", spacy.explain(ent.label_))
