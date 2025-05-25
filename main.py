# GROUP: ARTHUR NEUMANN SALERNO | HENRIQUE ALVES SEMMER | VINICIUS TEIDER

import spacy
from unidecode import unidecode
import re
import time
import numpy as np

# ======== CONSTANTS =========

NLP_MODEL = "pt_core_news_md"
FILE_PATH = "guarani.txt"
CLEAR_COMMAND = "\033[H\033[J"
SIMILARITY_TRESHOLD = 0.3
ROMAN_REGEX = r"\b[IVX]+\b\s+\b[A-Z_]+\b(?:\s+\b[A-Z_]+\b)*"

nlp_pt = spacy.load(NLP_MODEL)

# ======= CLASSES ===========


# Story point class to hold the text, vector, characters, and locations
class StoryPoint:
    def __init__(self, text, vector, characters, locations):
        self.text = text
        self.vector = vector
        self.characters = characters
        self.locations = locations
        self.similarity = None


# ======== FUNCTIONS =========


# Adds a typewriter effect for improved experience in terminal mode
def type_effect(string, speed=0.01):
    for char in string:
        print(char, end="", flush=True)
        time.sleep(speed)


# Finds the top 3 most similar story points
# to the query based on cosine similarity.
def generic_answer(query_sp, story_points):
    candidates = []
    for sp in story_points:
        similarity = cossine_similarity(query_sp.vector, sp.vector)
        if similarity >= SIMILARITY_TRESHOLD:
            sp.similarity = similarity
            candidates.append(sp)
    candidates.sort(key=lambda x: x.similarity, reverse=True)
    return candidates[:3]


# Finds the top 3 most similar story points to the query,
# prioritizing matches with shared characters or locations.
def smart_answer(query_sp, story_points):
    filtered_candidates = []
    target_characters = query_sp.characters
    target_locations = query_sp.locations

    for sp in story_points:
        similarity = cossine_similarity(query_sp.vector, sp.vector)
        if similarity >= SIMILARITY_TRESHOLD and (
            any(character in sp.characters for character in target_characters)
            or any(location in sp.locations for location in target_locations)
        ):
            sp.similarity = similarity
            filtered_candidates.append(sp)

    filtered_candidates.sort(key=lambda x: x.similarity, reverse=True)
    return filtered_candidates[:3]


# Computes the cosine similarity between two vectors.
def cossine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


# Applies a series of text preprocessing steps to the input text.
# In order to prepare it for the nlp model.
def preprocess_string(text):
    text = text.lower()  # 1. All words to lowercase
    text = unidecode(text)  # 2. Unicode normalization
    # 3. Replace punctuations for spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # 4. Standardize multiple spaces and tabs
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Applies all processing and filtering necessary to feed
# the text to a nlp
def clean_pipeline(text):
    text = preprocess_string(text)
    doc = nlp_pt(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    cleaned_text = " ".join(tokens)
    return cleaned_text


# Processes chunks of data (paragraphs) to
# create story point objects from those chunks
def process_story_points(chunks):
    data = []
    for chunk in chunks:
        characters, locations = retrieve_entities(chunk)
        cleaned_text = clean_pipeline(chunk)
        word2vec = nlp_pt(cleaned_text).vector
        sp = StoryPoint(chunk, word2vec, characters, locations)
        data.append(sp)
    return data


# Retrieves entities from a chunk of text using the nlp model.
def retrieve_entities(chunk):
    doc = nlp_pt(chunk)
    characters = set()
    locations = set()
    for ent in doc.ents:
        if ent.label_ == "PER":
            characters.add(ent.text)
        elif ent.label_ == "LOC":
            locations.add(ent.text)
    return list(characters), list(locations)


# Applies the whole vectorization process to a string.
# Returns the document and the cleaned text
def process_query(string):
    cleaned_text = clean_pipeline(string)
    doc = nlp_pt(cleaned_text)
    return doc


# Processes a file into paragraph chunks
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().replace("\n\n", " ").replace("\n", " ")
        chapters = re.split(ROMAN_REGEX, content)
    return chapters


# Chatbot default fail response
def say_response_fail():
    type_effect(
        "Desculpe, com base no meu conhecimento \
            não consegui encontrar uma resposta.\n"
    )


# Chatbot default success response
def say_response_success(answer, similarity):
    type_effect(f"{answer}[Similaridade: {similarity:.2f}]\n\n", 0.00005)


# Chatbot default greetings
def say_greetings():
    message = "Olá! sou Embrikenemotron, especialista no livro Guarani.\n \
        Sinta-se à vontade para fazer suas perguntas.\n \
        Para encerrar a sessão, digite 'SAIR'.\n"
    type_effect(message)


# Splits chapters into overlapping chunks of paragraphs for better processing.
def process_chapters(chapters):
    paragraphs = []
    for chptr in chapters:
        words = chptr.split()
        chunk_size = 250
        overlap = 125  # 50% overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) > 0:
                paragraph = " ".join(chunk_words)
                paragraphs.append(paragraph)
            if i + chunk_size >= len(words):
                break
    return paragraphs


# ======== MAIN =========

if __name__ == "__main__":
    chapters = process_file(FILE_PATH)
    paragraphs = process_chapters(chapters)
    story_points = process_story_points(paragraphs)

    user_input = ""
    answer_type = ""
    say_greetings()

    while True:
        user_input = input(">> ")
        print()

        if user_input == "clear":
            print(CLEAR_COMMAND, end="")
            continue

        if user_input == "sair":
            type_effect("Bom, até logo!")
            break

        query_sp = process_story_points([user_input])

        if query_sp:
            print(f"DEBUG: Query Text: {query_sp[0].text}")
            print(f"DEBUG: Characters Extracted: {query_sp[0].characters}")
            print(f"DEBUG: Locations Extracted: {query_sp[0].locations}")
        else:
            print("DEBUG: query_sp is empty after processing user input.")

        # First we try a smart answer
        answers = smart_answer(query_sp[0], story_points)
        answer_type = "smart"
        if not answers:
            # If no smart answer, we try a generic answer
            answers = generic_answer(query_sp[0], story_points)
            answer_type = "generic"
        if not answers:
            # If no answers at all, we say we failed
            say_response_fail()
            continue

        print(f"DEBUG: Answer type: {answer_type}\n")
        for ans in answers:
            say_response_success(ans.text, ans.similarity)
