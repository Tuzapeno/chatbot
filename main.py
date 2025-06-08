# GROUP: ARTHUR NEUMANN SALERNO | HENRIQUE ALVES SEMMER | VINICIUS TEIDER

import spacy
from unidecode import unidecode
import re
import numpy as np


# ======== CONSTANTS =========

NLP_MODEL = "pt_core_news_lg"
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
        self.weight = 0


# ======== FUNCTIONS =========


# Template for greeting messages
def greeting_template():
    replies = [
        "Olá! Como posso ajudar você hoje?",
        "Oi! Estou aqui para responder suas perguntas sobre Guarani.",
    ]
    return np.random.choice(replies)


# Template for character information
def character_info_template(characters):
    templates = {
        # One character identified
        "one_character": [
            "Ah, então você quer saber sobre {}. Vamos dar uma olhada...",
            "Claro! {} é um personagem fascinante. Aqui estão alguns detalhes...",
        ],
        # Multiple characters identified
        "multiple_characters": [
            "Você mencionou vários personagens: {}. Vamos falar sobre eles...",
            "Interessante! Vamos explorar os personagens: {}.",
        ],
        # No characters identified
        "no_characters": [
            "Vamos explorar um pouco sobre os personagens do livro Guarani.",
            "Vejo que você está curioso sobre os personagens. Vamos lá!",
        ],
    }
    if len(characters) == 1:
        return np.random.choice(templates["one_character"]).format(characters[0])
    elif len(characters) > 1:
        return np.random.choice(templates["multiple_characters"]).format(
            ", ".join(characters)
        )
    else:
        return np.random.choice(templates["no_characters"])


# Template for location information
def location_info_template(locations):
    templates = {
        # One location identified
        "one_location": [
            "Ah, você está interessado em um local chamado {}. Vamos explorar esse lugar...",
            "Claro! {} é um local importante na história. Aqui estão alguns detalhes...",
        ],
        # Multiple locations identified
        "multiple_locations": [
            "Você mencionou vários locais: {}. Vamos falar sobre eles...",
            "Interessante! Vamos explorar os locais: {}.",
        ],
        # No locations identified
        "no_locations": [
            "Vamos explorar um pouco sobre os locais do livro Guarani.",
            "Vejo que você está curioso sobre os locais. Vamos lá!",
        ],
    }
    if len(locations) == 1:
        return np.random.choice(templates["one_location"]).format(locations[0])
    elif len(locations) > 1:
        return np.random.choice(templates["multiple_locations"]).format(
            ", ".join(locations)
        )
    else:
        return np.random.choice(templates["no_locations"])


# Template for plot information, which is more generic
def plot_info_template():
    templates = [
        "A história se desenrola com diversos eventos e personagens interessantes.",
        "O enredo contém algumas reviravoltas e momentos envolventes.",
        "O livro apresenta uma narrativa com conflitos e resoluções.",
    ]
    return np.random.choice(templates)


# Generic template for when no specific intent is matched
def generic_template():
    templates = [
        "Vamos conhecer mais sobre o livro Guarani.",
        "Estou aqui para ajudar com informações sobre Guarani.",
    ]
    return np.random.choice(templates)


# Loads the appropriate template based on the intent and query story point.
def load_template(intent, query_sp):
    if intent == "greeting":
        return greeting_template()
    elif intent == "character_info":
        return character_info_template(query_sp.characters)
    elif intent == "location_info":
        return location_info_template(query_sp.locations)
    elif intent == "plot_info":
        return plot_info_template()
    else:
        return generic_template()


# Creates a vector list for intents based on example phrases.
def create_intent_classifier():
    intent_examples = {
        "greeting": [
            "olá",
            "oi",
            "bom dia",
            "boa tarde",
            "boa noite",
            "como vai?",
            "tudo bem?",
            "eae beleza",
            "Salve",
        ],
        "character_info": [
            "quem é Peri?",
            "me fale sobre Cecília",
            "qual o papel de Dom Antônio?",
            "descreva os personagens principais",
            "quais são as características de Isabel?",
            "Eu quero saber mais sobre o personagem Peri",
        ],
        "location_info": [
            "onde se passa a história?",
            "me conte sobre o cenário",
            "qual a importância do rio Paquequer?",
            "descreva a fazenda",
            "onde fica a casa de Dom Antônio?",
            "em que localidade a história se passa?",
        ],
        "plot_info": [
            "o que acontece no livro?",
            "qual é o enredo?",
            "como termina a história?",
            "quais são os eventos principais?",
            "qual é o conflito central?",
        ],
    }

    # Create vector representations for each intent
    intent_vectors = {}
    for intent, examples in intent_examples.items():
        # Process each example and average their vectors
        vectors = [nlp_pt(clean_pipeline(ex)).vector for ex in examples]
        vectors = np.array(vectors)
        intent_vectors[intent] = np.mean(vectors, axis=0)
    return intent_vectors


# Classifies the intent of a query based on cosine similarity
def classify_intent(query, intent_vectors):
    query_vector = nlp_pt(clean_pipeline(query)).vector
    similarities = {}
    for intent, vector in intent_vectors.items():
        similarities[intent] = cosine_similarity(query_vector, vector)
    return max(similarities.items(), key=lambda x: x[1])[0]


# Finds the top 3 most similar story points
# to the query based on cosine similarity.
def generic_answer(query_sp, story_points):
    candidates = []
    for sp in story_points:
        similarity = cosine_similarity(query_sp.vector, sp.vector)
        if similarity >= SIMILARITY_TRESHOLD:
            sp.similarity = similarity
            candidates.append(sp)
    candidates.sort(key=lambda x: x.similarity, reverse=True)
    return candidates[:3]


# Finds the top 3 most similar story points to the query,
# prioritizing matches with shared characters or locations.
def smart_answer(query_sp, story_points):
    filtered_candidates = []
    query_characters = query_sp.characters
    query_locations = query_sp.locations

    for story_point in story_points:
        # Calculate cosine similarity once
        similarity = cosine_similarity(query_sp.vector, story_point.vector)

        if similarity < SIMILARITY_TRESHOLD:
            continue

        # Calculate weight based on shared characters and locations
        weight = 0
        for entity in query_characters + query_locations:
            if entity in story_point.characters or entity in story_point.locations:
                weight += 1

        story_point.similarity = similarity
        story_point.weight = weight
        filtered_candidates.append(story_point)

    # Sort by weight and then similarity
    filtered_candidates.sort(key=lambda x: (x.weight, x.similarity), reverse=True)
    return filtered_candidates[:3]


# Computes the cosine similarity between two vectors.
def cosine_similarity(vec1, vec2):
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


# Processes a file into paragraph chunks
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().replace("\n\n", " ").replace("\n", " ")
        chapters = re.split(ROMAN_REGEX, content)
    return chapters


# Chatbot default fail response
def say_response_fail():
    print(
        "Desculpe, com base no meu conhecimento \
            não consegui encontrar uma resposta.\n"
    )


# Chatbot default success response
def say_response_success(answer, extra, intent, query_sp):
    template = load_template(intent, query_sp)

    if intent == "greeting":
        print(f"{template}\n\n")
    else:
        print(
            f'{template}\n\n No livro temos um trecho que fala:\n"{answer}"\n\nE ai o que achou?\n\n{extra}\n\n'
        )


# Chatbot default greetings
def say_greetings():
    message = "Olá! sou Embrikenemotron, especialista no livro Guarani.\n \
        Sinta-se à vontade para fazer suas perguntas.\n \
        Para encerrar a sessão, digite 'SAIR'.\n"
    print(message)


def process_chapters(chapters, chunk_size=250, overlap=125):
    paragraphs = []
    step = chunk_size - overlap

    for chptr in chapters:
        words = chptr.split()

        paragraphs.extend(
            " ".join(words[i : i + chunk_size]) for i in range(0, len(words), step)
        )

    return paragraphs


# ======== MAIN =========

if __name__ == "__main__":
    chapters = process_file(FILE_PATH)
    paragraphs = process_chapters(chapters)
    story_points = process_story_points(paragraphs)
    intents = create_intent_classifier()

    user_input = ""
    answer_type = ""
    say_greetings()

    while True:
        user_input = input(">> ")

        print()

        if user_input == "CLEAR":
            print(CLEAR_COMMAND, end="")
            continue

        if user_input == "SAIR":
            print("Bom, até logo!")
            break

        query_sp = process_story_points([user_input])[0]
        intent = classify_intent(user_input, intents)

        if query_sp:
            print(f"DEBUG: Query Text: {query_sp.text}")
            print(f"DEBUG: Characters Extracted: {query_sp.characters}")
            print(f"DEBUG: Locations Extracted: {query_sp.locations}")
            print(f"DEBUG: Intent Classified: {intent}")
        else:
            print("DEBUG: query_sp is empty after processing user input.")

        # First we try a smart answer
        answers = smart_answer(query_sp, story_points)
        answer_type = "smart"
        if not answers:
            # If no smart answer, we try a generic answer
            answers = generic_answer(query_sp, story_points)
            answer_type = "generic"
        if not answers:
            # If no answers at all, we say we failed
            say_response_fail()
            continue

        print(f"DEBUG: Answer type: {answer_type}\n")
        say_response_success(
            answers[0].text,
            f"Similaridade: {answers[0].similarity} Peso: {answers[0].weight}",
            intent,
            query_sp,
        )
