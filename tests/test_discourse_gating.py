import pytest
from post_clustering_pipeline.nlp import is_worth_seeing, is_clusterable, cleanse_text


def test_banality_rejection_casual_chatter():
    # Posts containing casual chatter slang must be rejected even if long
    text1 = "Eating at McDonald's with my best friend Sarah today, feeling so bloated lol"
    worth, reason = is_worth_seeing(text1)
    assert worth is False
    assert reason == "chatter_marker"

    text2 = "Just woke up from a nap in Miami, feeling cute might delete later selfie time"
    worth, reason = is_worth_seeing(text2)
    assert worth is False
    assert reason == "chatter_marker"


def test_banality_rejection_personal_diary():
    # 1st person dominance without external substance
    text = "I went to my kitchen and I looked in my pantry and I decided to make dinner for myself"
    worth, reason = is_worth_seeing(text)
    assert worth is False
    assert reason == "personal_diary"


def test_debate_admission_without_named_entities():
    # Philosophical/economic debate or policy questions without named entities/proper nouns MUST pass!
    text1 = "Should governments subsidize renewable energy through direct grants or carbon pricing mechanisms to maximize economic efficiency?"
    worth, reason = is_worth_seeing(text1)
    assert worth is True
    assert reason == "valid_discourse"

    text2 = "Why does economic policy prioritize short-term stimulus over long-term structural productivity reforms despite known inflation risks?"
    worth, reason = is_worth_seeing(text2)
    assert worth is True
    assert reason == "valid_discourse"

    text3 = "Artificial intelligence will transform labor markets; however, worker displacement depends on whether automation outpaces augmentation."
    worth, reason = is_worth_seeing(text3)
    assert worth is True
    assert reason == "valid_discourse"


def test_event_news_admission_with_entities():
    # Real news with proper nouns/entities must pass
    text = "NASA and SpaceX engineers completed the cryogenic propellant loading test at Starbase Texas launch site today."
    worth, reason = is_worth_seeing(text)
    assert worth is True
    assert reason == "valid_discourse"


def test_short_and_empty_text_rejection():
    assert is_worth_seeing("")[0] is False
    assert is_worth_seeing("Too short text")[0] is False
    assert is_clusterable("Short") is False
