# GROUP: ARTHUR NEUMANN SALERNO, HENRIQUE ALVES SEMMER, VINICIUS TEIDER

import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
import unicodedata
import re
import numpy as np

download("stopwords")
download("punkt_tab")

START_LINE = 116  # Beginning of the actual book
END_LINE = 8698  # End of the book
WORD_BATCH_SIZE = 250
FILE_PATH = "guarani.txt"

nlp_pt = spacy.load("pt_core_news_md")  # Load the Portuguese model
snow_st = SnowballStemmer("portuguese")  # Create a stemmer for Portuguese
pt_stop_words = set(
    stopwords.words("portuguese")
)  # Create a set of stop words for Portuguese

# docs = []  # List to store tuples of (original_text, nlp_document)


def preprocess_text(text):
    global snow_st
    tokens = []

    # 1. All words to lowercase
    text = text.lower()

    # 2. Unicode normalization
    text = unicodedata.normalize("NFD", text)

    # 3. Replace punctuations for spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 4. Standardize multiple spaces and tabs
    text = re.sub(r"\s+", " ", text).strip()

    # tokens = word_tokenize(text, language="portuguese")
    # TODO: See if the word_tokenize function removes empty lines

    # 5. Tokenize and remove empty tokens
    # tokens = [token for token in text.split(" ") if token != ""]
    tokens = word_tokenize(text, language="portuguese")

    # 6. Apply stemming and remove stopwords
    steamed_tokens = [
        snow_st.stem(token) for token in tokens if token not in pt_stop_words
    ]

    # 5. Tokenize & Removing stopwords & Steeming
    return steamed_tokens


def process_file(file_path):
    docs = []
    file = open(file_path, "r", encoding="utf-8")

    # Skip to the start line
    for _ in range(START_LINE):
        file.readline()

    # Read and process in batches
    current_line = START_LINE
    words = []
    full_text = []

    for line in file:
        if current_line >= END_LINE:
            break

        if len(words) >= WORD_BATCH_SIZE:
            data = (" ".join(full_text), nlp_pt(" ".join(words)))
            docs.append(data)
            half = WORD_BATCH_SIZE // 2
            words = words[half:]
            # maintain 50% for superposition
            half = len(full_text) // 2
            full_text = full_text[half:]

        words.extend(preprocess_text(line))
        full_text.append(line.strip())
        current_line += 1

    # Process leftover words that didn't sum up to batch size
    if len(words) > 0:
        docs.append((" ".join(full_text), nlp_pt(" ".join(words))))

    file.close()

    return docs


# ======== MAIN =========

docs = process_file(FILE_PATH)


query = "Como o autor descreve os costumes dos colonizadores portugueses?"

query_tokens = preprocess_text(query)
query_doc = nlp_pt(" ".join(query_tokens))


# Find the most similar vector
greatest_similarity = 0
most_similar_vector = None
most_similar_text = None

for text, doc in docs:
    similarity = query_doc.similarity(doc)
    if similarity > greatest_similarity:
        greatest_similarity = similarity
        most_similar_vector = doc.vector
        most_similar_text = text

print("\n\n")
print("Query text:")
print(query)
print("\n\n")
print("Most similar text:")
print(most_similar_text)
print("\n\n")
print("Similarity score:", greatest_similarity)
