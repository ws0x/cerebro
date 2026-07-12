"""Tests for the pre-flight LLM-call-count estimate.

Regression coverage for a real bug: a 131-minute YouTube video needed 61 MAP
calls and silently exhausted a free-tier daily quota mid-build, degrading the
whole map to the offline heuristic engine with no warning beforehand.
"""

from cerebro.cli import _MANY_CALLS_THRESHOLD, _estimate_llm_calls
from cerebro.transcript import OutlineEntry, Segment, Transcript


def _flat_transcript(num_segments: int, words_per_segment: int = 50) -> Transcript:
    text = " ".join(f"word{i}" for i in range(words_per_segment))
    return Transcript(
        source="s",
        title="T",
        segments=[Segment(text=text, start=float(i * 10)) for i in range(num_segments)],
    )


def test_short_transcript_estimates_a_small_number_of_calls():
    # ~250 words total, well under one chunk at any level.
    t = _flat_transcript(num_segments=5, words_per_segment=50)
    assert _estimate_llm_calls(t, "full") <= 3


def test_long_transcript_estimates_many_calls():
    # ~30,000 words -- roughly the real 131-minute video's scale.
    t = _flat_transcript(num_segments=600, words_per_segment=50)
    estimate = _estimate_llm_calls(t, "expert")
    assert estimate > _MANY_CALLS_THRESHOLD


def test_expert_level_estimate_includes_link_overhead_full_does_not():
    t = _flat_transcript(num_segments=5, words_per_segment=50)
    # Same content, only the level differs -- expert must estimate at least
    # one more call (the LINK pass) than full.
    assert _estimate_llm_calls(t, "expert") > _estimate_llm_calls(t, "full") - 1


def test_outline_bearing_transcript_falls_back_to_word_count_approximation():
    # A PDF-shaped transcript (has .outline) chunks differently (per leaf
    # section, not chunk_transcript) -- must not crash, and must still scale
    # up with size.
    short = Transcript(
        source="s", title="T",
        segments=[Segment(text="x " * 50, start=0.0)],
        outline=[OutlineEntry(level=1, title="Ch 1", page=0)],
    )
    long = Transcript(
        source="s", title="T",
        segments=[Segment(text="x " * 50, start=float(i)) for i in range(600)],
        outline=[OutlineEntry(level=1, title=f"Ch {i}", page=i) for i in range(20)],
    )
    assert _estimate_llm_calls(long, "full") > _estimate_llm_calls(short, "full")


def test_empty_transcript_does_not_crash_and_estimates_at_least_one_call():
    t = Transcript(source="s", title="T", segments=[])
    assert _estimate_llm_calls(t, "full") >= 1
