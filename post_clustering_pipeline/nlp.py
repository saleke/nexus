import re
import spacy

# Load spaCy with parser and lemmatizer disabled for high-throughput tagging & entity detection
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# Emoji / pictograph strip BEFORE NER + word count. spaCy en_core_web_sm tags
# standalone emoji as ORG/PRODUCT ("🚀" -> ORG:🚀), which used to admit
# emoji-only noise as a "named entity" post. Every pass below operates on the
# stripped text so the gate and the stored entity set see the same tokens the
# embedder sees.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # misc symbols & pictographs + supplemental symbols
    "\U0001F000-\U0001F02F"   # mahjong / domino / dice tiles
    "\U00002600-\U000027BF"   # misc symbols + dingbats (incl. hearts/stars)
    "\U00002190-\U000021FF"   # arrows (headline decoration)
    "\U00002B00-\U00002BFF"   # misc symbols & arrows (incl. the 2B50 star)
    "\uFE0F"                  # variation selector (emoji presentation)
    "\u200D"                  # zero-width joiner (ZWJ sequences)
    "]+"
)


def _strip_emoji(text: str) -> str:
    return _EMOJI_PATTERN.sub(" ", text)

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

# Identity evidence for the merge/tiebreak/split machinery. PERSON is included
# because actor threads are often person-anchored (cricket follows "Bumrah",
# tech follows "Musk"); without PERSON, such posts carry NO identity and ride
# the embedding-majority component at birth, gluing distinct threads together.
# The person-sharing hazard (Musk spanning Tesla vs SpaceX) is neutralized by
# (a) GENERIC_ENTITY_HEADS stripping template junk and (b) the ORG-conflict
# veto in the merge judge, which already separates disjoint-actor hubs.
IDENTITY_LABELS = {"ORG", "PRODUCT", "NORP", "PERSON"}

# Generic collective head nouns that ANY discourse template emits as entities
# ("insiders", "observers", "supporters", "fans", "critics"). They appear in
# every topic (measured: ORG:insiders on 80 posts across all 5 sim topics), so
# they must never count as identity evidence - leaving them in makes every hub
# pair "share identity" and collapses distinct events into one hub.
GENERIC_ENTITY_HEADS = {
    "insiders", "observers", "supporters", "fans", "critics", "skeptics",
    "defenders", "detractors", "pundits", "experts", "analysts", "authorities",
    "officials", "sources", "advisors", "lawmakers", "leaders", "members",
    "followers", "community", "investors",
}


def identity_entity_tokens(entities, *, org_only: bool = False) -> set[str]:
    """Sanitized ORG/PRODUCT/NORP identity tokens (``LABEL:value``) for asks.

    Used by hub-merge overlap, entity-conflict vetoes, assignment tiebreaks,
    and birth's entity split. Drops: PERSON evidence, tokens with no alphabetic
    content, and tokens that reference a generic collective head noun (which
    are shared across topics and would fabricate universal identity overlap).
    """
    out: set[str] = set()
    for e in entities or []:
        e = str(e or "")
        if ":" not in e:
            continue
        label, _, value = e.partition(":")
        if label not in IDENTITY_LABELS:
            continue
        if org_only and label != "ORG":
            continue
        value = value.strip().lower()
        words = value.split()
        if not words or not any(ch.isalpha() for ch in value):
            continue
        if any(w in GENERIC_ENTITY_HEADS for w in words):
            continue
        out.add(f"{label}:{value}")
    return out

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

# Modality: the skeleton of any claim, independent of capitalization or a fixed
# word list ("should", "will", "may"). Universal quantifiers ("every", "all",
# "most", "many", "always") are deliberately NOT admission evidence - they
# appear constantly in substance-free filler ("a lot of stuff happened today",
# "this is basically me every morning") and leaked adversarial noise in testing.
SIGNAL_MODAL_QUANTIFIER_PATTERNS = [
    r"\b(?:should|shouldn't|shouldnt|could|couldn't|couldnt|must|ought to|will|won't|wont|would|wouldn't|wouldnt|may|might)\b",
    r"\b(?:never|always)\b",
]

# Topical signal verbs + nouns (lemma-irrelevant: inflections listed). These
# admit a real event/claim regardless of named entities, capitalization, or a
# whether-word: "Diversity drives innovation, period." clears via "drives"
# even though it has zero entities and no debate marker. Case-insensitive.
SIGNAL_VERBS = {
    "drives", "drive", "driving", "boosted", "boosts", "surges", "surged",
    "surge", "spikes", "spiked", "spike", "plunges", "plunged", "crashes",
    "crashed", "announces", "announced", "unveils", "unveiled", "launches",
    "launched", "warns", "warned", "claims", "claimed", "criticizes",
    "criticized", "praises", "praised", "awards", "awarded", "approves",
    "approved", "rejects", "rejected", "vetoes", "vetoed", "passes", "passed",
    "votes", "voted", "negotiates", "negotiated", "signs", "signed", "strikes",
    "struck", "reports", "reported", "confirms", "confirmed", "denies",
    "denied", "reveals", "revealed", "publishes", "published", "releases",
    "released", "beats", "defeats", "defeated", "wins", "won", "loses", "lost",
    "resigns", "resigned", "retires", "retired", "raises", "raised", "hikes",
    "hiked", "cuts", "cut", "falls", "fell", "jumps", "jumped", "sells",
    "sold", "buys", "bought", "merges", "merged", "acquires", "acquired",
    "invests", "invested", "grows", "grew", "drops", "dropped", "doubles",
    "doubled", "tests", "tested", "studies", "studied", "finds", "found",
    "shows", "showed", "proves", "proved", "suggests", "suggested",
    "recommends", "recommended", "bans", "banned", "legalizes", "legalized",
    "indicts", "indicted", "convicts", "convicted", "blames", "blamed",
    "survives", "survived", "recovers", "recovered", "collapses", "collapsed",
    "closes", "closed", "opens", "opened", "hires", "hired", "fires", "fired",
    "wins", "tops", "leads", "led", "overtakes", "overtook", "calls", "called",
    "plans", "planned", "aims", "vows", "pledges", "pledged", "promises",
    "promised",
}

SIGNAL_NOUNS = {
    "news", "report", "reports", "result", "results", "update", "updates",
    "launch", "verdict", "election", "vote", "votes",
    "ruling", "proposal", "reform", "crisis", "scandal", "alert", "warning",
    "announcement", "interview", "analysis", "study", "surveys", "survey",
    "ban", "lawsuit", "tariff", "verdict", "agreement", "settlement",
    "shortage", "outbreak", "suspension", "ceasefire", "rounds", "round",
    "race", "match", "series", "tournament", "season", "winner", "record",
}

SIGNAL_LEXICON = SIGNAL_VERBS | SIGNAL_NOUNS


def _gate_prechecks(text: str) -> tuple[str, str, str | None]:
    """Pre-parse gate checks that need no spaCy Doc: returns ``(cleaned,
    lowered, fail_reason)`` where ``fail_reason`` is None when parsing is
    required. Mirrors the original ``_discourse_analysis`` early-exit order
    (empty_text -> too_short -> chatter_marker -> personal_diary) exactly, so
    batch and single-doc paths produce identical verdicts."""
    if not text or len(text.strip()) == 0:
        return "", "", "empty_text"

    cleaned = text.strip()
    cleaned = _strip_emoji(cleaned)
    words = cleaned.split()
    if len(words) < 7:
        return cleaned, "", "too_short"

    lower = cleaned.lower()

    # 1. Instant rejection of casual personal chatter markers
    for pattern in CHATTER_PATTERNS:
        if re.search(pattern, lower):
            return cleaned, lower, "chatter_marker"

    # 2. First-person dominance check (personal diary/status filter)
    first_person_tokens = re.findall(r"\b(i|me|my|myself|im)\b", lower)
    first_person_ratio = len(first_person_tokens) / max(len(words), 1)
    if len(first_person_tokens) >= 3 and first_person_ratio > 0.22:
        return cleaned, lower, "personal_diary"

    return cleaned, lower, None


def _gate_classify(cleaned: str, lower: str, doc) -> tuple[bool, str, list[str]]:
    """Admission decision for an already-parsed Doc. Never called unless
    ``_gate_prechecks`` returned a parseable text, so ``lower`` is the lowered
    non-empty cleaned text (deterministically ``cleaned.lower()``)."""
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
    has_modal_quantifier = any(bool(re.search(pat, lower)) for pat in SIGNAL_MODAL_QUANTIFIER_PATTERNS)
    has_question = "?" in cleaned

    # 4. Check for real-world entities or proper nouns (news/events).
    # Lowercase PERSON labels are a known spaCy hallucination on word-salad
    # ("asoidj kajsdlk lkajd" -> PERSON), so PERSON must be capitalized to count;
    # lowercase ORG/GPE/NORP/etc. stay valid ("fed", "china", "bitcoin").
    has_entities = any(
        ent.label_ in VALID_ENTITY_LABELS
        and (ent.text[0].isupper() or ent.label_ != "PERSON")
        for ent in doc.ents
    )
    has_proper_nouns = any(
        token.pos_ == "PROPN" and (token.text[0].isupper() or token.text.isupper())
        for token in doc
    )

    # 5. Topical signal lexicon (verbs/nouns of events, claims, results, deals)
    has_signal_lexicon = any(w in SIGNAL_LEXICON for w in lower.split())

    # Admission rule:
    # A post passes if it carries substantive entities/proper nouns OR a
    # structured claim/question (debate markers, modality, quantification) OR a
    # topical signal verb/noun. Genuine discussion is admitted regardless of
    # capitalization or a fixed phrase list; only truly blank prose - no named
    # thing, no claim structure, no event lexicon - is dropped as chatter.
    if (has_entities or has_proper_nouns or has_debate_markers
            or has_modal_quantifier or has_signal_lexicon or has_question):
        return True, "valid_discourse", strong_entities

    return False, "lacks_substance", []


def _discourse_analysis(text: str) -> tuple[bool, str, list[str]]:
    """Single spaCy pass that both gates the post and extracts strong entities
    (PERSON/ORG/PRODUCT/NORP) as ``LABEL:text`` strings. Reused by the gate and
    by ingestion so one parser run serves both purposes."""
    cleaned, lower, fail = _gate_prechecks(text)
    if fail:
        return False, fail, []
    doc = nlp(cleaned)
    return _gate_classify(cleaned, lower, doc)


def analyze_discourse_batch(texts, batch_size: int = 32) -> list[tuple[bool, str, list[str]]]:
    """Identical verdicts to ``[analyze_discourse(t) for t in texts]`` but the
    spaCy parses run through ``nlp.pipe`` for its internal batching. spaCy
    applies the same pipeline to each document regardless of batching, so every
    (passes, reason, entities) result is deterministic and byte-identical to
    the single-doc path - this is a throughput change only, never a behaviour
    change to the discourse gate."""
    results: list[tuple[bool, str, list[str]]] = [None] * len(texts)  # type: ignore[list-item]
    parse_entries: list[tuple[int, str, str]] = []  # (original index, cleaned, lowered)
    for i, text in enumerate(texts):
        cleaned, lower, fail = _gate_prechecks(text)
        if fail:
            results[i] = (False, fail, [])
        else:
            parse_entries.append((i, cleaned, lower))
    if parse_entries:
        parsed = nlp.pipe([cleaned for _, cleaned, _ in parse_entries], batch_size=batch_size, as_tuples=False)
        for doc, (orig_idx, cleaned, lower) in zip(parsed, parse_entries):
            results[orig_idx] = _gate_classify(cleaned, lower, doc)
    return results


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
    """Do two sets of identity entities describe different actors?

    Conflict (the posts likely describe DIFFERENT events) only when BOTH sides
    carry sanitized ORG/PRODUCT/NORP identity evidence and the sets are
    disjoint. Empty side on either side means 'no evidence' -> no veto.
    Generic collective tokens (insiders/fans/critics...) are stripped first -
    they are shared by every topic and would suppress the vetoes that keep
    distinct events apart.
    """
    ps = identity_entity_tokens(post_entities)
    hs = identity_entity_tokens(hub_entities)
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