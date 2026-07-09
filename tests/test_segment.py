from cerebro.structure.segment import chunk_transcript, cohesion_scores
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
