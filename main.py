# GROUP: ARTHUR NEUMANN SALERNO, HENRIQUE ALVES SEMMER, VINICIUS TEIDER

import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from nltk.stem import SnowballStemmer
from unidecode import unidecode
import re
import time


# ======== SETUP =========

try:
    stopwords.words("portuguese")
except LookupError:
    download("stopwords")

try:
    word_tokenize("Teste de tokenização.", language="portuguese")
except LookupError:
    download("punkt")

# ======== CONSTANTS =========


START_LINE = 47  # Beginning of book story content
END_LINE = 9501  # End of book story content
FILE_PATH = "guarani2.txt"
CLEAR_COMMAND = "\033[H\033[J"
SIM_CUTOFF = 0.3

nlp_pt = spacy.load("pt_core_news_md")  # Load the Portuguese model
snow_st = SnowballStemmer("portuguese")  # Create a stemmer for Portuguese
pt_stop_words = set(
    stopwords.words("portuguese")
)  # Create a set of stop words for Portuguese

# ======= CLASSES ===========


# Story point object contains the original paragraph
# associated with it and the spacy document
class StoryPoint:
    def __init__(self, text, doc):
        self.text = text
        self.doc = doc

    def similarity(self, other_doc):
        return self.doc.similarity(other_doc)


# Simple class to hold two values for an answer
class Answer:
    def __init__(self, text, similarity):
        self.text = text
        self.similarity = similarity

    def __getitem__(self, index):
        return (self.text, self.similarity)[index]


# ======== FUNCTIONS =========


# Adds a typewriter effect for improved experience in terminal mode
def type_effect(string, speed=0.01):
    for char in string:
        print(char, end="", flush=True)
        time.sleep(speed)


# Computes cossine similarity between a query_doc and a
# story_points list of docs, returns an ordered list
# of Answers
def compute_similarity(query_doc, story_points):
    answers = sorted(
        [Answer(sp.text, sp.similarity(query_doc)) for sp in story_points],
        key=lambda x: x[1],
        reverse=True,
    )
    return answers


# TODO: Smart answer is supposed to search only for
# story points that contains query identified entities.
# def smart_answer(query_doc, entity_map, entities):
#     possible_sps = []
#     for ent in entities:
#         if ent in entity_map:
#             possible_sps.extend(entity_map[ent])
#     best_text, best_similarity = compute_similarity(query_doc, possible_sps)
#     return best_text, best_similarity


def generic_answer(query_doc, story_points):
    candidates = compute_similarity(query_doc, story_points)
    return candidates[:3]


# Applies a series of text preprocessing steps to the input text.
# In order to prepare it for the nlp model.
def preprocess_text(text):
    text = text.lower()  # 1. All words to lowercase
    text = unidecode(text)  # 2. Unicode normalization
    # 3. Replace punctuations for spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # 4. Standardize multiple spaces and tabs
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Stems the tokens reducing to the radical
def stem_tokens(tokens):
    return [snow_st.stem(token) for token in tokens]


# Filters tokens removing stop words
def remove_stop_words(tokens):
    return [token for token in tokens if token not in pt_stop_words]


# Applies all processing and filtering necessary to feed
# the text to a nlp
def clean_pipeline(text):
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stop_words(tokens)
    cleaned_tokens = stem_tokens(tokens)
    cleaned_text = " ".join(cleaned_tokens)
    return cleaned_text


# Processes chunks of data (paragraphs) to
# create story point objects from those chunks
def process_story_points(chunks):
    data = []
    for chunk in chunks:
        cleaned_text = clean_pipeline(chunk)
        data.append(StoryPoint(chunk, nlp_pt(cleaned_text)))
    return data


# Applies the whole vectorization process to a string.
# Returns the document and the cleaned text
def process_query(string):
    cleaned_text = clean_pipeline(string)
    doc = nlp_pt(cleaned_text)
    return doc


# Processes a file into paragraph chunks
def process_file(file_path, start_line, end_line):
    paragraphs = []
    current_line = start_line
    lines = []
    line_count = 0

    with open(file_path, "r", encoding="utf-8") as file:
        # Skip to the start line
        for _ in range(start_line):
            file.readline()

        for line in file:
            # Reach end of book content
            if current_line > end_line:
                break

            current_line += 1

            # Reach end of paragraph
            if line_count < 7:
                # Skip empty lines
                if not line.strip():
                    continue
                lines.append(line)
                line_count += 1
                continue

            # Do not add empty lines
            if lines:
                paragraphs.append("".join(lines))
                # 50% superposition
                lines = lines[len(lines) // 2 :]
                line_count = 0
                continue

        # Process any remaining content
        if lines:
            paragraphs.append("".join(lines))

    return paragraphs


# Extract all entities from a spacy document
def extract_entities(doc):
    return [ent.text for ent in doc.ents]


# Creates a table with the key being the entity's name
# and the value being a list of all story points that
# the entity was detected
def create_entity_map(story_points):
    entity_map = {}
    for sp in story_points:
        for ent in extract_entities(sp.doc):
            if ent not in entity_map:
                entity_map[ent] = []
            entity_map[ent].append(sp)
    return entity_map


# Chatbot default fail response
def say_response_fail():
    type_effect("Desculpe, com base no meu conhecimento ")
    type_effect("não consegui encontrar uma resposta.\n")


# Chatbot default success response
def say_response_success(answer, similarity):
    type_effect(answer, 0.00005)
    type_effect(f"[Similaridade: {similarity:.2f}]\n\n")


# Chatbot default greetings
def say_greetings():
    type_effect("Olá! sou Embrikenemotron")
    type_effect(", especialista no livro Guarani.\n")
    type_effect("Sinta-se à vontade para fazer suas perguntas.\n")
    type_effect("Para encerrar a sessão, digite 'sair'.\n")


# ======== MAIN =========

if __name__ == "__main__":
    paragraphs = process_file(FILE_PATH, START_LINE, END_LINE)
    story_points = process_story_points(paragraphs)

    user_input = ""
    say_greetings()

    while True:
        user_input = input(">> ").lower()
        print()

        if user_input == "clear":
            print(CLEAR_COMMAND, end="")
            continue

        if user_input == "sair":
            type_effect("Bom, até logo!")
            break

        query = clean_pipeline(user_input)
        query_doc = nlp_pt(query)
        best_candidates = generic_answer(query_doc, story_points)

        if not best_candidates:
            say_response_fail()
            continue

        if not best_candidates:
            say_response_fail()
            continue

        found_good_answer = False
        for candidate in best_candidates:
            if candidate.text and candidate.similarity >= SIM_CUTOFF:
                say_response_success(candidate.text, candidate.similarity)
                found_good_answer = True

        if not found_good_answer:
            say_response_fail()
