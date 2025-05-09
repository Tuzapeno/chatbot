# GROUP: ARTHUR NEUMANN SALERNO, HENRIQUE ALVES SEMMER, VINICIUS TEIDER

import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
import unicodedata
import re
import time
import os


# ======== SETUP =========

try:
    stopwords.words("portuguese")
except LookupError:
    download("stopwords")

try:
    word_tokenize("Olá, tudo bem?", language="portuguese")
except LookupError:
    download("punkt_tab")


# ======== CONSTANTS =========


START_LINE = 116  # Beginning of book story content
END_LINE = 8698  # End of the book story content
WORD_BATCH_SIZE = 250  # Chunk size for processing
FILE_PATH = "guarani.txt"

nlp_pt = spacy.load("pt_core_news_md")  # Load the Portuguese model
snow_st = SnowballStemmer("portuguese")  # Create a stemmer for Portuguese
pt_stop_words = set(
    stopwords.words("portuguese")
)  # Create a set of stop words for Portuguese


# ======== FUNCTIONS =========


# Adds a typewriter effect for improved experience in terminal mode
def typewriter_effect(string, speed=0.01):
    for char in string:
        print(char, end="", flush=True)
        time.sleep(speed)
    print()


# Returns the chunk with most similarity score to the user's query
def answer_query(query, docs):
    _, query_doc = process_string(query)

    most_similar_text, greatest_similarity = max(
        ((text, doc.similarity(query_doc)) for text, doc in docs),
        key=lambda x: x[1],
        default=(None, 0),
    )

    return most_similar_text, greatest_similarity


# Applies a series of text preprocessing steps to the input text.
# In order to prepare it for the nlp model.
def preprocess_text(text):
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


# Applies the whole vectorization process to a string.
# Returns the original string and the spacy document.
def process_string(string):
    tokens = preprocess_text(string)
    doc = nlp_pt(" ".join(tokens))
    return (string, doc)


# Applies the whole vectorization process to a file.
# Returns a list of tuples, each containing the
# original chunk and the corresponding spacy document.
def process_file(file_path):
    docs = []

    with open(file_path, "r", encoding="utf-8") as file:
        # Skip to the start line using a more efficient approach
        for _ in range(START_LINE):
            file.readline()

        current_line = START_LINE
        words = []
        full_text = []

        for line in file:
            if current_line >= END_LINE:
                break

            if len(words) >= WORD_BATCH_SIZE:
                # Process batch with clear variable naming
                text_batch = " ".join(full_text)
                embedding_batch = nlp_pt(" ".join(words))

                docs.append((text_batch, embedding_batch))

                # Maintain 50% of content for superposition
                words = words[len(words) // 2 :]
                full_text = full_text[len(full_text) // 2 :]

            words.extend(preprocess_text(line))
            full_text.append(line.strip())
            current_line += 1

    # Process any remaining content
    if words:
        docs.append((" ".join(full_text), nlp_pt(" ".join(words))))

    return docs


# ======== MAIN =========

docs = process_file(FILE_PATH)
user_input = ""

typewriter_effect("Boa tarde! Sou Embrikenemotron, especialista no Livro Guarani.")
typewriter_effect("Por favor, sinta-se à vontade para fazer suas perguntas.")
typewriter_effect("Para encerrar a sessão, digite 'sair'.")

while True:
    user_input = input(">>  ")

    if user_input == "clear":
        os.system("clear")
        continue

    if user_input == "sair":
        typewriter_effect("Bom, até logo!")
        break

    answer, similarity = answer_query(user_input, docs)

    if not answer or similarity < 0.3:
        typewriter_effect(
            "Desculpe, com base no meu conhecimento nao consegui formular uma resposta."
        )
        continue

    typewriter_effect(answer, 0.00005)
    typewriter_effect(f"Similaridade: {similarity:.2f}")
