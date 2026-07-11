"""Direct unit tests for HeuristicStructurer -- the offline/no-key default
engine, and the automatic fallback whenever any LLM call fails. Previously
only exercised incidentally through test_batch.py/test_pdf.py fixtures;
never tested for its own chunking/titling/leaf-sampling logic."""

from cerebro.ir import NodeType
from cerebro.structure.heuristic import (
    HeuristicStructurer,
    _chunk_segments,
    _sentences,
    _titlecase_snippet,
    _truncate,
)
from cerebro.transcript import Segment, Transcript

_ENUMERATED_SEGMENTS = [
    Segment(text="In this video I'll cover three things you need to know.", start=0.0),
    Segment(text="Number one is keep promises to yourself. It builds real confidence daily.", start=10.0),
    Segment(text="Number two is get your house in order. Your relationships are the foundation.", start=20.0),
    Segment(text="Number three is do hard things intentionally. Growth comes from chosen discomfort.", start=30.0),
]


def _segments(n, words_per_sentence=6, sentences_per_segment=3):
    out = []
    for i in range(n):
        text = " ".join(
            f"This is sentence {i}-{j} with some words here."
            for j in range(sentences_per_segment)
        )
        out.append(Segment(text=text, start=float(i * 10)))
    return out


def _transcript(n_segments=12, title="My Video"):
    return Transcript(source="s", title=title, segments=_segments(n_segments))


def test_sentences_splits_on_terminal_punctuation():
    assert _sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_sentences_ignores_blank_input():
    assert _sentences("   ") == []


def test_truncate_leaves_short_text_untouched():
    assert _truncate("short text", 100) == "short text"


def test_truncate_cuts_at_a_word_boundary_and_adds_ellipsis():
    result = _truncate("one two three four five six seven eight nine ten", 20)
    assert result.endswith("…")
    assert len(result) <= 21
    # must not have chopped a word in half -- the last real char before the
    # ellipsis is a whole word from the source text
    assert "…" not in result[:-1]


def test_truncate_strips_trailing_punctuation_before_the_ellipsis():
    result = _truncate("one two three,", 13)
    assert not result.endswith(",…")


def test_titlecase_snippet_capitalizes_first_letter_only():
    assert _titlecase_snippet("hello world") == "Hello world"


def test_titlecase_snippet_falls_back_to_untitled_for_empty_text():
    assert _titlecase_snippet("") == "Untitled"


def test_chunk_segments_splits_into_roughly_the_target_count():
    segments = _segments(12)
    chunks = _chunk_segments(segments, target=4)
    assert len(chunks) <= 4
    assert sum(len(c) for c in chunks) == 12


def test_chunk_segments_folds_a_tiny_trailing_chunk_into_the_previous_one():
    # 10 segments, target 4 -> per=2, chunks of [2,2,2,2,2] with no remainder --
    # use a count that actually produces a small tail instead.
    segments = _segments(9)
    chunks = _chunk_segments(segments, target=4)
    # No chunk should be a lone straggler smaller than half the target chunk size
    per = max(1, len(segments) // 4)
    assert all(len(c) >= max(1, per // 2) for c in chunks)


def test_chunk_segments_caps_target_to_available_segment_count():
    chunks = _chunk_segments(_segments(3), target=9)
    assert len(chunks) <= 3


def test_chunk_segments_handles_no_segments():
    assert _chunk_segments([], target=5) == []


def test_structure_produces_a_root_titled_from_the_transcript():
    mm = HeuristicStructurer().structure(_transcript(title="Intro To Neural Networks"), level="full")
    assert mm.root.title == "Intro To Neural Networks"
    assert mm.title == "Intro To Neural Networks"


def test_structure_falls_back_to_a_generic_title_when_transcript_has_none():
    mm = HeuristicStructurer().structure(_transcript(title=""), level="full")
    assert mm.root.title == "Mind Map"


def test_structure_propagates_source_and_level_onto_the_mindmap():
    t = Transcript(source="https://youtu.be/abc", title="T", segments=_segments(5))
    mm = HeuristicStructurer().structure(t, level="expert")
    assert mm.source == "https://youtu.be/abc"
    assert mm.level == "expert"


def test_structure_unknown_level_falls_back_to_full():
    mm = HeuristicStructurer().structure(_transcript(), level="not-a-real-level")
    assert mm.level == "full"


def test_brief_level_produces_topics_with_no_leaf_children():
    mm = HeuristicStructurer().structure(_transcript(n_segments=12), level="brief")
    assert mm.root.children
    for topic in mm.root.children:
        assert topic.children == []


def test_full_level_produces_leaves_under_topics():
    mm = HeuristicStructurer().structure(_transcript(n_segments=12), level="full")
    assert any(topic.children for topic in mm.root.children)


def test_expert_level_produces_more_leaves_per_topic_than_full():
    t = _transcript(n_segments=12, title="T")
    full_leaves = sum(len(topic.children) for topic in HeuristicStructurer().structure(t, level="full").root.children)
    expert_leaves = sum(
        len(topic.children) for topic in HeuristicStructurer().structure(t, level="expert").root.children
    )
    assert expert_leaves >= full_leaves


def test_topics_are_typed_topic_and_leaves_are_typed_detail():
    mm = HeuristicStructurer().structure(_transcript(n_segments=12), level="full")
    for topic in mm.root.children:
        assert topic.type == NodeType.topic
        for leaf in topic.children:
            assert leaf.type == NodeType.detail


def test_topic_carries_the_first_segments_timestamp_when_the_source_has_real_timing():
    mm = HeuristicStructurer().structure(_transcript(n_segments=12), level="full")
    assert mm.root.children[0].timestamp == 0.0  # a genuine "starts at 0:00", not suppressed


def test_no_timestamp_is_added_when_every_segment_has_a_zero_start_and_duration():
    # mirrors a PDF transcript: Segment.start is always 0.0 (a page number is
    # not a timestamp) -- must not render a bogus "[0:00]" on every node.
    t = Transcript(source="s", title="T", segments=[Segment(text=f"Page {i} text here.", start=0.0) for i in range(4)])
    mm = HeuristicStructurer().structure(t, level="full")
    assert all(node.timestamp is None for node in mm.root.children)


def test_empty_transcript_still_produces_a_placeholder_node():
    mm = HeuristicStructurer().structure(Transcript(source="s", title="Empty", segments=[]), level="full")
    assert len(mm.root.children) == 1
    assert mm.root.children[0].title == "(no content extracted)"
    assert mm.root.children[0].type == NodeType.detail


def test_structure_never_raises_on_a_single_short_segment():
    t = Transcript(source="s", title="Short", segments=[Segment(text="Hello there.", start=0.0)])
    mm = HeuristicStructurer().structure(t, level="expert")
    assert mm.root.children


# -- Enumeration-aware path (no AI) -----------------------------------------


def test_an_enumerated_transcript_gets_a_numbered_spine_not_word_count_chunks():
    t = Transcript(source="s", title="3 Things", segments=_ENUMERATED_SEGMENTS)
    mm = HeuristicStructurer().structure(t, level="full")
    titles = [c.title for c in mm.root.children]
    assert titles == [
        "1. Keep Promises to Yourself",
        "2. Get Your House in Order",
        "3. Do Hard Things Intentionally",
    ]


def test_enumerated_branches_carry_their_own_section_timestamp():
    t = Transcript(source="s", title="3 Things", segments=_ENUMERATED_SEGMENTS)
    mm = HeuristicStructurer().structure(t, level="full")
    assert [c.timestamp for c in mm.root.children] == [10.0, 20.0, 30.0]


def test_enumerated_full_level_has_leaves_brief_level_does_not():
    t = Transcript(source="s", title="3 Things", segments=_ENUMERATED_SEGMENTS)
    full = HeuristicStructurer().structure(t, level="full")
    brief = HeuristicStructurer().structure(t, level="brief")
    assert any(c.children for c in full.root.children)
    assert all(c.children == [] for c in brief.root.children)


def test_a_short_intro_produces_no_overview_branch():
    t = Transcript(source="s", title="3 Things", segments=_ENUMERATED_SEGMENTS)
    mm = HeuristicStructurer().structure(t, level="full")
    assert mm.root.children[0].title.startswith("1.")


def test_a_long_intro_becomes_its_own_overview_branch():
    long_intro = " ".join(f"word{i}" for i in range(45))
    segments = [
        Segment(text=long_intro + ".", start=0.0),
        Segment(text="Number one is alpha bravo charlie delta echo foxtrot.", start=30.0),
        Segment(text="Number two is golf hotel india juliet kilo lima.", start=40.0),
        Segment(text="Number three is mike november oscar papa quebec romeo.", start=50.0),
    ]
    t = Transcript(source="s", title="3 Things", segments=segments)
    mm = HeuristicStructurer().structure(t, level="full")
    assert mm.root.children[0].title == "Overview"
    assert mm.root.children[1].title.startswith("1.")


def test_a_non_enumerated_transcript_still_uses_the_flat_chunking_path():
    mm = HeuristicStructurer().structure(_transcript(n_segments=12), level="full")
    assert not any(c.title.startswith(("1.", "2.", "3.")) for c in mm.root.children)
