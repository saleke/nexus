import pytest

from post_clustering_pipeline.nlp import analyze_discourse, analyze_discourse_batch

CORPUS = [
    "",
    "Short text",
    "Eating at McDonald's with my best friend Sarah today, feeling so bloated lol",
    "Just woke up from a nap in Miami, feeling cute might delete later selfie time",
    "I went to my kitchen and I looked in my pantry and I decided to make dinner for myself",
    "Should governments subsidize renewable energy through direct grants or carbon pricing mechanisms to maximize economic efficiency?",
    "Why does economic policy prioritize short-term stimulus over long-term structural productivity reforms despite known inflation risks?",
    "Artificial intelligence will transform labor markets; however, worker displacement depends on whether automation outpaces augmentation.",
    "NASA and SpaceX engineers completed the cryogenic propellant loading test at Starbase Texas launch site today.",
    "The Fed raised rates by 25 basis points and warned of further tightening, disappointing investors who expected a pause.",
    "A slick new ad for the Pixel phone dropped and honestly the camera looks great, worth a look later",
    "Breaking: a magnitude 6.1 earthquake struck the coast of Chile near Santiago, triggering tsunami advisories along the entire coastline.",
    "Trump and the EU reached a preliminary trade truce, but tariffs on steel remain in place pending the next round of talks.",
    "me me me myself and I never I ever I just I always I do",
    "😎😎😎 just vibing with the boys never better",
    "If the company misses guidance, will the board force a leadership change before Q3 results are announced?",
    "analyst notes point to a sharp decline in copper demand, though supply buffers remain adequate for the next quarter.",
    "tadaa! big reveal coming this friday from the new studio",
    "China's new economic stimulus package boosted copper futures, while the yuan held steady against the dollar.",
    "no news here just a normal tuesday and the weather is fine, the coffee is good, the walk was long",
]


def test_batch_equivalence_single_parse():
    sequential = [analyze_discourse(t) for t in CORPUS]
    batched = analyze_discourse_batch(CORPUS)
    assert len(batched) == len(CORPUS)
    for text, seq, bat in zip(CORPUS, sequential, batched):
        assert seq == bat, f"verdict divergence for {text!r}: {seq} != {bat}"


@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
def test_batch_equivalence_varying_pipe_batch_size(batch_size):
    sequential = [analyze_discourse(t) for t in CORPUS]
    assert analyze_discourse_batch(CORPUS, batch_size=batch_size) == sequential


def test_batch_equivalence_slices():
    for start in range(0, len(CORPUS)):
        for end in range(start + 1, len(CORPUS) + 1):
            slice_ = CORPUS[start:end]
            assert analyze_discourse_batch(slice_) == [analyze_discourse(t) for t in slice_]