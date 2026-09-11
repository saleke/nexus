import re
import spacy

nlp = spacy.load("en_core_web_sm")

def is_clusterable(text: str) -> bool:
    doc = nlp(text)
    words = [token for token in doc if not token.is_punct and not token.is_space]
    if len(words) < 8:
        return False
    
    named_entities = [ent for ent in doc.ents if ent.label_ in {"PROPN", "ORG", "GPE", "PRODUCT"}]
    pos_proper_nouns = [token for token in doc if token.pos_ == "PROPN"]
    
    if len(named_entities) == 0 and len(pos_proper_nouns) == 0:
        return False
        
    return True

def cleanse_text(text: str) -> str:
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
    cleaned = re.sub(r'@\w+', '', cleaned)
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned