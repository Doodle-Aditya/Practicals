"""
Topic: Stemming, Lemmatization, Part-of-Speech Tagging and Stop Words (spaCy + NLTK)

Requirements:
    pip install spacy nltk
    python -m spacy download en_core_web_sm
"""

import spacy
from collections import Counter
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

# ---------------------------------------------------------------------------
# 1. Blank vs trained spaCy pipeline
# ---------------------------------------------------------------------------
nlp_blank = spacy.blank("en")
print("Blank pipeline components:", nlp_blank.pipe_names)  # empty, no components

nlp = spacy.load("en_core_web_sm")
print("Trained pipeline components:", nlp.pipe_names)
print("Full pipeline:", nlp.pipeline)

# ---------------------------------------------------------------------------
# 2. Part-of-Speech tagging + lemmatization
# ---------------------------------------------------------------------------
doc = nlp("Captain america ate 100$ of french fries. Then he said i could do this all day")

print("\nToken | POS explanation | Lemma")
for token in doc:
    print(token.text, "|", spacy.explain(token.pos_), "|", token.lemma_)

unique_pos = set(token.pos_ for token in doc)
print("\nUnique POS tags:", unique_pos)

pos_counts = Counter(token.pos_ for token in doc)
print("POS tag counts:", pos_counts)

# ---------------------------------------------------------------------------
# 3. Stemming with NLTK's PorterStemmer
# ---------------------------------------------------------------------------
stemmer = PorterStemmer()

words = ["eating", "eats", "eat", "ate", "adjustable", "rafting", "ability", "meeting"]

print("\nStemming (NLTK PorterStemmer):")
for word in words:
    print(word, "|", stemmer.stem(word))

more_words = [
    "running", "runner", "runs", "ran",
    "studies", "studying", "studied",
    "flying", "flies", "fly",
    "connection", "connected", "connecting",
    "beautiful", "beauty",
    "organization", "organize",
    "happiness", "happy",
    "better", "best",
]

print("\nMore stemming examples:")
for word in more_words:
    print(f"{word:15} --> {stemmer.stem(word)}")

# ---------------------------------------------------------------------------
# 4. Lemmatization with spaCy
# ---------------------------------------------------------------------------
doc = nlp("eating eats eat ate adjustable rafting ability meeting better")
print("\nLemmatization (spaCy):")
for token in doc:
    print(token.text, "|", token.lemma_, "|", token.lemma)

doc = nlp("Mando talked for 3 hours although talking isn't his thing")
print("\nLemmatization example sentence:")
for token in doc:
    print(token.text, "|", token.lemma_)

# Stemming vs Lemmatization comparison on the same word list
print("\nLemmatization (spaCy) on the extended word list:")
doc = nlp(" ".join(more_words))
for token in doc:
    if token.is_alpha:
        print(f"{token.text:15} --> {token.lemma_}")

# ---------------------------------------------------------------------------
# 5. Customizing the lemmatizer via the attribute_ruler
# ---------------------------------------------------------------------------
ar = nlp.get_pipe("attribute_ruler")

# Slang / abbreviation -> custom lemma
ar.add([[{"TEXT": "Bro"}], [{"TEXT": "Bruh"}]], {"LEMMA": "Brother"})
ar.add([[{"TEXT": "DS"}]], {"LEMMA": "Data Scientist"})
ar.add([[{"TEXT": "ML"}]], {"LEMMA": "Machine Learning"})
ar.add([[{"TEXT": "AI"}]], {"LEMMA": "Artificial Intelligence"})
ar.add([[{"TEXT": "ChatGPT"}], [{"TEXT": "GPT"}]], {"LEMMA": "OpenAI Model"})
ar.add([[{"TEXT": "LOL"}], [{"TEXT": "lol"}]], {"LEMMA": "Laugh"})

print("\nCustom lemmatizer examples:")
for text in [
    "Bro do you wanna go? Come on bruh let's go to meet my brother",
    "DS and ML are part of AI",
    "ChatGPT is based on GPT",
    "LOL this movie is funny lol",
]:
    doc = nlp(text)
    print(f"\n'{text}'")
    for token in doc:
        print(token.text, "-->", token.lemma_)

# ---------------------------------------------------------------------------
# 6. Stop words
# ---------------------------------------------------------------------------
print("\nNumber of spaCy stop words:", len(STOP_WORDS))

doc = nlp("She is reading a book in the library")
print("\nStop words found in the sentence:")
for token in doc:
    if token.is_stop:
        print(token.text)


def preprocess(text: str) -> str:
    """Remove stop words (and keep punctuation) from the given text."""
    doc = nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop]
    return " ".join(no_stop_words)


def preprocess_no_punct(text: str) -> list:
    """Remove stop words AND punctuation, returning a list of tokens."""
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop and not token.is_punct]


print("\npreprocess() examples:")
print(preprocess("We just opened our wings, the flying part is coming soon"))
print(preprocess("The other is not other but your divine brother"))
print(preprocess("Musk wants time to prepare for a trial over his"))

print("\npreprocess_no_punct() examples:")
print(preprocess_no_punct("We just opened our wings, the flying part is coming soon"))
print(preprocess_no_punct("The other is not other but your divine brother"))

# Example use-case: cleaning text for a chatbot / Q&A system
print("\nChatbot preprocessing example:")
print(preprocess("I don't find yoga mat on your websites. Can you help?"))

# ---------------------------------------------------------------------------
# 7. Applying preprocessing to a pandas DataFrame (optional, needs a data file)
# ---------------------------------------------------------------------------
# import pandas as pd
# df = pd.read_json("doj_press.json", lines=True)
# df = df[df["topics"].str.len() != 0]
# df["contents_new"] = df["contents"].apply(preprocess)
