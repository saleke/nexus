import re
import spacy

# Load spaCy with parser and lemmatizer disabled for high-throughput tagging & entity detection
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

VALID_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "NORP", "FAC"
}

# Identity-labelling entities: two posts about the SAME "actor" (brand, person,
# org) are likely the same event thread; two posts whose identity entities are
# disjoint are likely DIFFERENT events even when their embeddings are similar
# (e.g. "Apple iPhone launch" vs "Samsung Galaxy launch" both read like
# "phone launch"). GPE/LOC are deliberately excluded: a single natural event
# spans many places (a hurricane hits several states) so geolocation conflict
# is NOT evidence of separate events.
STRONG_ENTITY_LABELS = {
    "PERSON", "ORG", "PRODUCT", "NORP"
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


def _discourse_analysis(text: str) -> tuple[bool, str, list[str]]:
    """Single spaCy pass that both gates the post and extracts strong entities
    (PERSON/ORG/PRODUCT/NORP) as ``LABEL:text`` strings. Reused by the gate and
    by ingestion so one parser run serves both purposes."""
    if not text or len(text.strip()) == 0:
        return False, "empty_text", []

    cleaned = text.strip()
    words = cleaned.split()
    if len(words) < 7:
        return False, "too_short", []

    lower = cleaned.lower()

    # 1. Instant rejection of casual personal chatter markers
    for pattern in CHATTER_PATTERNS:
        if re.search(pattern, lower):
            return False, "chatter_marker", []

    # 2. First-person dominance check (personal diary/status filter)
    first_person_tokens = re.findall(r"\b(i|me|my|myself|im)\b", lower)
    first_person_ratio = len(first_person_tokens) / max(len(words), 1)
    if len(first_person_tokens) >= 3 and first_person_ratio > 0.22:
        return False, "personal_diary", []

    doc = nlp(cleaned)

    strong_entities: list[str] = []
    seen: set[str] = set()
    for ent in doc.ents:
        if ent.label_ in STRONG_ENTITY_LABELS:
            key = f"{ent.label_}:{ent.text.strip().lower()}"
            if key not in seen:
                seen.add(key)
                strong_entities.append(key)

    # 3. Check for debate, proposition, or question markers
    has_debate_markers = any(bool(re.search(pat, lower)) for pat in DEBATE_DISCOURSE_PATTERNS)
    has_question = "?" in cleaned

    # 4. Check for real-world entities or proper nouns (news/events)
    has_entities = any(ent.label_ in VALID_ENTITY_LABELS for ent in doc.ents)
    has_proper_nouns = any(token.pos_ == "PROPN" for token in doc)

    # Admission rule:
    # A post passes if it has substantive entities/proper nouns OR structured debate/idea markers,
    # provided it was not rejected as personal chatter above.
    if has_entities or has_proper_nouns or has_debate_markers or has_question:
        return True, "valid_discourse", strong_entities

    return False, "lacks_substance", []


def is_worth_seeing(text: str) -> tuple[bool, str]:
    """Strict discourse gating: admits events, debates, topics; rejects banality & chatter.
    
    Returns (passes_filter: bool, reason: str).
    """
    passes, reason, _ = _discourse_analysis(text)
    return passes, reason


def analyze_discourse(text: str) -> tuple[bool, str, list[str]]:
    """Full discourse pass: gate verdict, reason, and strong entities
    (``LABEL:text`` strings) for entity-conflict disambiguation downstream."""
    return _discourse_analysis(text)


def extract_strong_entities(text: str) -> list[str]:
    """Backwards- / test-compatible helper: strong entities only."""
    _, _, entities = _discourse_analysis(text)
    return entities


def entities_conflict(post_entities, hub_entities) -> bool:
    """Do two sets of strong entities describe different actors?

    Conflict (the posts likely describe DIFFERENT events) only when BOTH sides
    carry identity entities and the sets are disjoint. Empty side on either
    side means 'no evidence' -> no veto.
    """
    if not post_entities or not hub_entities:
        return False
    ps = set(str(e).strip().lower() for e in post_entities if e)
    hs = set(str(e).strip().lower() for e in hub_entities if e)
    if not ps or not hs:
        return False
    return ps.isdisjoint(hs)


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