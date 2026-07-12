import pytest
from cerebro.structure.segment import adaptive_max_words, chunk_transcript, cohesion_scores
from cerebro.transcript import Segment, Transcript

_PHOTOSYNTHESIS = [
    "Plants convert sunlight into chemical energy through photosynthesis.",
    "Chlorophyll in the leaves absorbs light from the sun efficiently.",
    "Carbon dioxide and water combine inside the chloroplast during this process.",
    "The plant releases oxygen as a byproduct of photosynthesis.",
    "Sunlight intensity directly affects the rate of photosynthesis in leaves.",
    "Chlorophyll gives most plants their characteristic green color.",
]

_DATABASE = [
    "A database index speeds up query performance significantly.",
    "The query planner chooses which index to use for a given query.",
    "Inserting a new row requires updating every index on that table.",
    "A composite index covers queries filtering on multiple columns.",
    "Database administrators monitor index fragmentation over time.",
    "Dropping an unused index can improve write performance on the table.",
]


def _mixed_transcript() -> Transcript:
    segments = [
        Segment(text=t, start=float(i * 5), duration=5.0)
        for i, t in enumerate(_PHOTOSYNTHESIS + _DATABASE)
    ]
    return Transcript(source="x", title="Mixed", segments=segments)


def test_cohesion_dips_at_the_real_topic_boundary():
    transcript = _mixed_transcript()
    scores = cohesion_scores(transcript.segments, window=3)
    assert scores  # long enough transcript to compute real scores

    boundary_index = len(_PHOTOSYNTHESIS)  # where topic actually changes
    boundary_score = scores[boundary_index]
    other_scores = [s for i, s in scores.items() if i != boundary_index]
    # The real topic shift should be at or near the lowest-cohesion point.
    assert boundary_score <= min(other_scores) + 1e-9 or boundary_score == min(scores.values())


def test_chunking_prefers_topic_boundary_over_arbitrary_cut():
    transcript = _mixed_transcript()
    # max_words high enough that the word-count ceiling is never the reason
    # for cutting; min_fraction low enough (the photosynthesis block is only
    # ~55 words) that the boundary condition can fire once it's accumulated.
    chunks = chunk_transcript(transcript, max_words=1000, min_fraction=0.03)
    assert len(chunks) == 2

    first_chunk_text = chunks[0].text.lower()
    second_chunk_text = chunks[1].text.lower()
    assert "photosynthesis" in first_chunk_text
    assert "database" not in first_chunk_text
    assert "index" in second_chunk_text
    assert "photosynthesis" not in second_chunk_text


def test_short_transcript_falls_back_to_word_count_only():
    segs = [Segment(text="one two three", start=0.0, duration=1.0)] * 3
    transcript = Transcript(source="x", title="Short", segments=segs)
    assert cohesion_scores(transcript.segments) == {}
    chunks = chunk_transcript(transcript, max_words=5)
    assert len(chunks) >= 1  # doesn't crash, degrades to pure word-count cutting


def test_max_words_is_never_exceeded_even_on_cohesive_content():
    # All-photosynthesis, repeated three times — a boundary may still fire
    # early on natural sentence-to-sentence variation (that's fine, expected),
    # but no chunk may ever grow past the max_words ceiling regardless.
    segments = [
        Segment(text=t, start=float(i * 5), duration=5.0)
        for i, t in enumerate(_PHOTOSYNTHESIS * 3)
    ]
    transcript = Transcript(source="x", title="Cohesive", segments=segments)
    chunks = chunk_transcript(transcript, max_words=40, min_fraction=0.4)
    assert len(chunks) > 1  # doesn't collapse into one unbounded blob
    for c in chunks:
        assert len(c.text.split()) <= 40


# -- adaptive chunking (cap MAP calls for long sources) --------------------

def test_adaptive_max_words_leaves_short_sources_at_the_base_budget():
    # A short source should never have its budget raised -- its fine
    # granularity must be preserved exactly as before this feature existed.
    assert adaptive_max_words(total_words=1000, base_max_words=1200) == 1200
    assert adaptive_max_words(total_words=5000, base_max_words=1200) == 1200


def test_adaptive_max_words_grows_the_budget_for_long_sources():
    # 30,000 words / 25 calls / 0.4 = ~3000-word budget, well above the base.
    grown = adaptive_max_words(total_words=30000, base_max_words=1200)
    assert grown > 1200
    assert grown == pytest.approx(3000, abs=50)


def test_adaptive_max_words_is_monotonic_in_length():
    prev = 0
    for words in (2000, 10000, 30000, 60000):
        cur = adaptive_max_words(words, base_max_words=1200)
        assert cur >= prev
        prev = cur


def test_adaptive_max_words_handles_degenerate_inputs():
    assert adaptive_max_words(total_words=0, base_max_words=1200) == 1200
    assert adaptive_max_words(total_words=30000, base_max_words=1200, target_calls=0) == 1200


def test_adaptive_chunking_caps_the_call_count_for_a_long_transcript():
    # A long, uniformly cohesive transcript that at the base budget would
    # produce dozens of tiny chunks must instead produce ~target_calls once
    # the adaptive budget is applied.
    words_per_seg = 20
    segs = [
        Segment(text=" ".join(f"w{j}" for j in range(words_per_seg)), start=float(i))
        for i in range(1500)  # ~30,000 words
    ]
    transcript = Transcript(source="x", title="Long", segments=segs)
    total = transcript.word_count

    base_chunks = len(chunk_transcript(transcript, max_words=1200))
    adaptive_chunks = len(chunk_transcript(transcript, max_words=adaptive_max_words(total, 1200)))

    assert adaptive_chunks < base_chunks  # meaningfully fewer calls
    assert adaptive_chunks <= 35  # bounded near the ~25 target (with slack)
