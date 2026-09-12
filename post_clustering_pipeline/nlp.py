import re
import spacy

# Load spaCy with parser and lemmatizer disabled for high-throughput tagging & entity detection
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

VALID_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "NORP", "FAC"
}

# Chatting / banality markers indicating low-information casual posts
CHATTER_PATTERNS = [
    r"\blol\b",
    r"\blmao\b",
    r"\bselfie\b",
    r"\bfitcheck\b",
    r"\bfeeling cute\b",
    r"\bhmu\b",
    r"\bdm me\b",
    r"\bbored af\b",
    r"\bgood morning yall\b",
    r"\bgm everyone\b",
]

# Structural markers of ideas, questions, debates, and policy discussions
DEBATE_DISCOURSE_PATTERNS = [
    r"\bwhy\b",
    r"\bshould\b",
    r"\bhowever\b",
    r"\bdespite\b",
    r"\bbecause\b",
    r"\btherefore\b",
    r"\bin contrast\b",
    r"\bwhereas\b",
    r"\balternatively\b",
    r"\bargues?\b",
    r"\bproposes?\b",
    r"\bdebates?\b",
    r"\bcriticizes?\b",
    r"\baccording to\b",
    r"\bconsequence\b",
    r"\beconomic\b",
    r"\bpolicy\b",
]


def is_worth_seeing(text: str) -> tuple[bool, str]:
    """Strict discourse gating: admits events, debates, topics; rejects banality & chatter.
    
    Returns (passes_filter: bool, reason: str).
    """
    if not text or len(text.strip()) == 0:
        return False, "empty_text"

    cleaned = text.strip()
    words = cleaned.split()
    if len(words) < 7:
        return False, "too_short"

    lower = cleaned.lower()

    # 1. Instant rejection of casual personal chatter markers
    for pattern in CHATTER_PATTERNS:
        if re.search(pattern, lower):
            return False, "chatter_marker"

    # 2. First-person dominance check (personal diary/status filter)
    first_person_tokens = re.findall(r"\b(i|me|my|myself|im)\b", lower)
    first_person_ratio = len(first_person_tokens) / max(len(words), 1)
    if len(first_person_tokens) >= 3 and first_person_ratio > 0.22:
        return False, "personal_diary"

    # 3. Check for debate, proposition, or question markers
    has_debate_markers = any(bool(re.search(pat, lower)) for pat in DEBATE_DISCOURSE_PATTERNS)
    has_question = "?" in cleaned

    # 4. Check for real-world entities or proper nouns (news/events)
    doc = nlp(cleaned)
    has_entities = any(ent.label_ in VALID_ENTITY_LABELS for ent in doc.ents)
    has_proper_nouns = any(token.pos_ == "PROPN" for token in doc)

    # Admission rule:
    # A post passes if it has substantive entities/proper nouns OR structured debate/idea markers,
    # provided it was not rejected as personal chatter above.
    if has_entities or has_proper_nouns or has_debate_markers or has_question:
        return True, "valid_discourse"

    return False, "lacks_substance"


def is_clusterable(text: str) -> bool:
    """Backwards-compatible wrapper around is_worth_seeing."""
    passes, _ = is_worth_seeing(text)
    return passes


def cleanse_text(text: str) -> str:
    """Strip URLs, handles, and excessive whitespace from text."""
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', text)
    cleaned = re.sub(r'@\w+', '', cleaned)
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned