# GROUP: ARTHUR NEUMANN SALERNO, HENRIQUE ALVES SEMMER, VINICIUS TEIDER

import spacy
import unicodedata
import re

# ============ TEXT PREPARATION ==================

guarani_content = ""

with open("guarani.txt", "r", encoding="utf-8") as file:
    guarani_content = file.readlines()

# ====
# REMOVAL OF HEADERS AND IRRELEVANT CONTENT
# ====

# Lines 1 to 116 removed
#    -> Introduction and summary
# Lines 9058 to 9411 removed
#    -> Editing notes
guarani_content = guarani_content[116:9058]

# ====
# Character and break normalization
# Segmentation into 10-line chunks
# ====

chunks = []

count = 1
line_chunk = ""
for line in guarani_content:
    # Skip empty lines
    if line.isspace():
        continue

    # 1. All words to lowercase
    line = line.lower()

    # 2. Unicode normalization
    line = unicodedata.normalize("NFD", line)

    # 3. Standardize quotes and hyphens
    line = line.replace("“", '"').replace("”", '"')
    line = line.replace("‘", "'").replace("–", "-")  # Changed to assign back to line

    # 4. Standardize multiple spaces and tabs to
    # a single space within the line
    line = re.sub(r"\s+", " ", line).strip()

    # 4. Add lines to a 10-line chunk
    if count % 10 == 0:
        chunks.append(line_chunk)
        line_chunk = ""
        count = 1
    else:
        count += 1
        # Add a space so the end of one line doesn't stick
        # to the beginning of the next
        line_chunk += line + " "


# ============ TOKENIZATION AND PROCESSING ==================

# Starting with a smaller model to see if it runs
nlp = spacy.load("pt_core_news_sm")

# Separation into words (tokens)
# And lemmatization (nlp already returns everything)
chunk_doc = nlp(chunks[1])  # Using the first chunk for demonstration

# Removal of stopwords
chunk_doc = [t for t in chunk_doc if not t.is_stop and t.is_alpha]

for token in chunk_doc:
    print(
        token.text,
        token.lemma_,
        # token.pos_,
        # token.tag_,
        # token.dep_,
        # token.shape_,
        # token.is_alpha,
        # token.is_stop,
    )
