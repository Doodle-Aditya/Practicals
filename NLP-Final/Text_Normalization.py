"""
Q3. Text Normalization : Stop Words, Stemming & Lemmatization

Run in Spyder (Ctrl + Enter on the whole file, or F5) or from a terminal:
    python Text_Normalization.py

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
# Q3. TEXT NORMALIZATION : STOP WORDS, STEMMING & LEMMATIZATION
# ======================================================================
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer

q3_words = ["running", "runs", "ran", "runner", "easily", "fairly", "studies",
            "studying", "better", "wolves", "geese", "happiness",
            "organization", "organizing", "generously"]
q3_text = ("The movie was not good at all. The actors were trying very hard but "
           "the story was boring and the songs were too long. I would not "
           "watch this movie again, although the music was quite nice.")

# (A) Stop words: count, remove, and customise
stop_words = nlp.Defaults.stop_words
print("(A1) Number of default stop words :", len(stop_words))
print("     First 20 (sorted) :", sorted(stop_words)[:20])

doc = nlp(q3_text)
tokens_all = [t.text.lower() for t in doc if not t.is_punct]
tokens_clean = [t.text.lower() for t in doc if not t.is_stop and not t.is_punct]
reduction = 100 * (1 - len(tokens_clean) / len(tokens_all))
print(f"\n(A2) Tokens before : {len(tokens_all)}")
print(f"     Tokens after  : {len(tokens_clean)} ({reduction:.1f}% reduction)")
print("     Clean tokens  :", tokens_clean)
print("\n(A3) Top 5 words before :", Counter(tokens_all).most_common(5))
print("     Top 5 words after  :", Counter(tokens_clean).most_common(5))

# Customising stop words: keep 'not' (important for sentiment), add 'movie'
nlp.Defaults.stop_words.discard("not")
nlp.vocab["not"].is_stop = False
nlp.Defaults.stop_words.add("movie")
nlp.vocab["movie"].is_stop = True

doc_custom = nlp(q3_text)
tokens_custom = [t.text.lower() for t in doc_custom
                 if not t.is_stop and not t.is_punct]
print("\n(A4) With custom stop words :", tokens_custom)
print("     'not' kept? ", "not" in tokens_custom,
      "| 'movie' removed? ", "movie" not in tokens_custom)

# Restore defaults so later questions are not affected
nlp.Defaults.stop_words.add("not")
nlp.vocab["not"].is_stop = True
nlp.Defaults.stop_words.discard("movie")
nlp.vocab["movie"].is_stop = False

# (B) Stemming vs Lemmatization
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

rows = []
for w in q3_words:
    rows.append({
        "word": w,
        "porter": porter.stem(w),
        "snowball": snowball.stem(w),
        "lancaster": lancaster.stem(w),
        "spacy_lemma": nlp(w)[0].lemma_,
    })
q3_df = pd.DataFrame(rows)
print("\n(B1) Stemmers vs spaCy lemmatizer\n", q3_df)

# Stems are often not real words ('easili', 'happi'); lemmas are dictionary
# words. Note: spaCy's rule-based lemmatizer works best inside a sentence, so
# isolated irregular words ('geese', 'easily') may come back unchanged.
differs = q3_df[q3_df["porter"] != q3_df["spacy_lemma"]]["word"].tolist()
print("\n(B2) Words where Porter stem differs from the lemma :", differs)

# Lemmatization depends on context (POS)
sentence = "The striped bats were hanging on their feet and ate the best fishes."
print("\n(B3) Context-aware lemmatization")
for t in nlp(sentence):
    if not t.is_punct:
        print(f"     {t.text:<8} POS={t.pos_:<5} lemma={t.lemma_:<8} "
              f"porter={porter.stem(t.text)}")
print("     'best' -> 'good', 'were' -> 'be' and 'feet' -> 'foot' work because "
      "the lemmatizer uses the POS tag. Stemmers just chop suffixes.")
