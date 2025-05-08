# GROUP: ARTHUR NEUMANN SALERNO, HENRIQUE ALVES SEMMER, VINICIUS TEIDER

import spacy
import unicodedata
import re
import numpy as np

START_LINE = 116
END_LINE = 9058
WORD_BATCH_SIZE = 250
FILE_PATH = "guarani.txt"

# Load spaCy model
nlp = spacy.load("pt_core_news_md")
nlp_tokenizer = spacy.load("pt_core_news_md", disable=["ner", "parser", "textcat"])
# Structure to store the vectors and the associated text chunk
vector_to_chunk = {}
docs = []
chunks = []


def similarity_calc(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def preprocess_text(text):
    global nlp_tokenizer
    tokens = []

    # 1. All words to lowercase
    text = text.lower()

    # 2. Unicode normalization
    text = unicodedata.normalize("NFD", text)

    # 3. Standardize multiple spaces and tabs
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Replace punctuations for spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # 5. Tokenize & Removing stopwords & Steeming
    docs = nlp_tokenizer(text)
    tokens = [t.lemma_ for t in docs if not t.is_stop]

    return tokens


def vectorize_text(lines):
    global nlp
    text = " ".join(lines)
    processed_text = preprocess_text(text)
    doc = nlp(processed_text)
    data = {"vector": doc.vector, "text": processed_text, "doc": doc}
    return data


def add_to_docs(data):
    global docs
    global vector_to_chunk
    vector_to_chunk[hash(data["text"])] = data
    docs.append(data["doc"])


# ======== MAIN =========


with open(FILE_PATH, "r", encoding="utf-8") as file:
    # Skip to the start line
    for _ in range(START_LINE):
        file.readline()

    # Read and process in batches
    current_line = START_LINE
    words = []

    for line in file:
        if current_line >= END_LINE:
            break

        if len(words) >= WORD_BATCH_SIZE:
            # chunks.append(" ".join(words))
            docs.append(nlp(" ".join(words)))
            half = WORD_BATCH_SIZE // 2
            words = words[half:]  # maintain 50% for superposition

        words.extend(preprocess_text(line))
        current_line += 1

    # Process leftover words that didn't sum up to batch size
    if len(words) > 0:
        # chunks.append(" ".join(words))
        docs.append(nlp(" ".join(words)))


query = "Como é descrita a natureza brasileira ao redor da casa de Dom Antônio?"
query_tokens = preprocess_text(query)
query_doc = nlp(" ".join(query_tokens))


# Find the most similar vector
greatest_similarity = 0
most_similar_vector = None
most_similar_text = None

for doc in docs:
    similarity = query_doc.similarity(doc)
    if similarity > greatest_similarity:
        greatest_similarity = similarity
        most_similar_vector = doc.vector
        most_similar_text = doc.text

print("\n\n")
print("Query text:")
print(query)
print("\n\n")
print("Most similar text:")
print(most_similar_text)
print("\n\n")
print("Similarity score:", greatest_similarity)
