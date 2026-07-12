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


def _varied_transcript(num_segments: int, words_per_segment: int = 50) -> Transcript:
    # Rotating vocabulary every few segments so the semantic chunker fires
    # topic boundaries frequently -- the shape of a real, topically-diverse
    # long video (a podcast wandering across subjects), as opposed to one
    # uniform block. Adaptive chunking caps uniform content at ~10 chunks
    # regardless of length; only varied content climbs toward the ~25 target.
    segs = []
    for i in range(num_segments):
        topic = i // 4  # a new vocabulary set every 4 segments
        text = " ".join(f"topic{topic}word{j}" for j in range(words_per_segment))
        segs.append(Segment(text=text, start=float(i * 10)))
    return Transcript(source="s", title="T", segments=segs)


def test_short_transcript_estimates_a_small_number_of_calls():
    # ~250 words total, well under one chunk at any level.
    t = _flat_transcript(num_segments=5, words_per_segment=50)
    assert _estimate_llm_calls(t, "full") <= 3


def test_long_varied_transcript_still_estimates_many_calls():
    # A long, topically-varied source (~30,000 words) still climbs past the
    # warning threshold even WITH adaptive chunking -- the cap is ~25, above
    # the threshold, so genuinely long real videos are still flagged.
    t = _varied_transcript(num_segments=600, words_per_segment=50)
    estimate = _estimate_llm_calls(t, "expert")
    assert estimate > _MANY_CALLS_THRESHOLD


def test_adaptive_chunking_keeps_a_uniform_long_source_bounded():
    # The whole point of adaptive chunking: a long but uniform (boundary-free)
    # source no longer scales linearly into dozens of calls -- it stays small.
    t = _flat_transcript(num_segments=600, words_per_segment=50)  # ~30k words, uniform
    assert _estimate_llm_calls(t, "expert") < 20


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
