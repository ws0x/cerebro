from cerebro.cache import Cache
from cerebro.ir import NodeType
from cerebro.llm.providers import MockProvider
from cerebro.structure.document import build_outline_map, build_outline_skeleton
from cerebro.transcript import OutlineEntry, Segment, Transcript


def _transcript(outline, pages):
    return Transcript(
        source="book.pdf",
        title="Book",
        segments=[Segment(text=t, start=float(i)) for i, t in enumerate(pages)],
        outline=[OutlineEntry(*e) for e in outline],
    )


def test_nests_three_levels_by_stack():
    transcript = _transcript(
        outline=[
            (1, "Chapter 1", 0),
            (2, "Section 1.1", 0),
            (3, "Subsection 1.1.1", 1),
            (2, "Section 1.2", 2),
            (1, "Chapter 2", 3),
        ],
        pages=["intro text " * 20, "sec text " * 20, "sub text " * 20, "sec2 text " * 20, "ch2 text " * 20],
    )
    mm = build_outline_skeleton(transcript)

    ch1, ch2 = mm.root.children
    assert ch1.title == "Chapter 1"
    assert ch2.title == "Chapter 2"
    assert [c.title for c in ch1.children] == ["Section 1.1", "Section 1.2"]
    sec11, sec12 = ch1.children
    assert [c.title for c in sec11.children] == ["Subsection 1.1.1"]
    assert sec12.children == []


def test_leaf_notes_are_truncated_page_range_text_with_page_number():
    transcript = _transcript(
        outline=[(1, "Only Heading", 2)],
        pages=["page0", "page1", "this is the real body text for the only heading section"],
    )
    mm = build_outline_skeleton(transcript)
    leaf = mm.root.children[0]
    assert leaf.note.startswith("(p. 3) ")
    assert "real body text" in leaf.note


def test_non_leaf_nodes_have_no_note():
    transcript = _transcript(
        outline=[(1, "Chapter 1", 0), (2, "Section 1.1", 0)],
        pages=["chapter intro then section text"],
    )
    mm = build_outline_skeleton(transcript)
    chapter = mm.root.children[0]
    assert chapter.children  # has a child -> non-leaf
    assert chapter.note is None


def test_empty_outline_produces_placeholder_node():
    transcript = Transcript(source="x.pdf", title="X", segments=[Segment(text="hi")], outline=[])
    mm = build_outline_skeleton(transcript)
    assert mm.root.children[0].type == NodeType.detail
    assert "no structure" in mm.root.children[0].title.lower()


def _two_chapter_transcript():
    return _transcript(
        outline=[(1, "Chapter 1", 0), (1, "Chapter 2", 1)],
        pages=["chapter one full body text here", "chapter two full body text here"],
    )


def test_build_outline_skeleton_honors_requested_level_label():
    transcript = _two_chapter_transcript()
    assert build_outline_skeleton(transcript, level="expert").level == "expert"
    assert build_outline_skeleton(transcript, level="brief").level == "brief"
    assert build_outline_skeleton(transcript).level == "full"  # default unchanged


def test_build_outline_map_brief_level_labels_mindmap_as_brief_not_full():
    transcript = _two_chapter_transcript()
    provider = MockProvider()
    mm = build_outline_map(transcript, provider=provider, cache=Cache(enabled=False), level="brief")
    assert mm.level == "brief"


def test_build_outline_map_without_provider_matches_skeleton():
    transcript = _two_chapter_transcript()
    mm = build_outline_map(transcript, provider=None)
    skeleton = build_outline_skeleton(transcript)
    assert [c.title for c in mm.root.children] == [c.title for c in skeleton.root.children]
    assert mm.root.children[0].note == skeleton.root.children[0].note


def test_build_outline_map_brief_level_skips_ai_even_with_provider():
    transcript = _two_chapter_transcript()
    provider = MockProvider()
    build_outline_map(transcript, provider=provider, cache=Cache(enabled=False), level="brief")
    assert provider.calls == 0


def test_build_outline_map_full_level_enriches_leaves():
    transcript = _two_chapter_transcript()
    provider = MockProvider()
    # synthesize=False isolates the enrichment-call count from the separate
    # synthesis pass (which would add its own call).
    mm = build_outline_map(
        transcript, provider=provider, cache=Cache(enabled=False), level="full", synthesize=False
    )
    leaf = mm.root.children[0]
    assert leaf.note == "A concise summary of this segment."
    assert [c.title for c in leaf.children] == ["Supporting point one", "Supporting point two"]
    assert provider.calls == 2  # one MAP call per leaf


def test_build_outline_map_expert_level_adds_relationships():
    transcript = _two_chapter_transcript()
    provider = MockProvider()
    mm = build_outline_map(transcript, provider=provider, cache=Cache(enabled=False), level="expert")
    assert len(mm.relationships) == 1


def test_build_outline_map_skips_failed_leaf_without_failing_whole_map():
    from cerebro.llm.base import LLMError

    class FlakyProvider(MockProvider):
        def complete_json(self, system, user):
            self.calls += 1
            if "TASK: MAP" in system and "chapter one" in user:
                raise LLMError("boom")
            return super().complete_json(system, user)

    transcript = _two_chapter_transcript()
    provider = FlakyProvider()
    mm = build_outline_map(transcript, provider=provider, cache=Cache(enabled=False), level="full")

    ch1, ch2 = mm.root.children
    assert ch1.note.startswith("(p. 1)")  # failed -> kept its deterministic fallback note
    assert ch1.children == []
    assert ch2.note == "A concise summary of this segment."  # succeeded -> enriched
    assert [c.title for c in ch2.children] == ["Supporting point one", "Supporting point two"]


def test_build_outline_map_raises_when_every_leaf_enrichment_fails():
    # Found via a real Groq-vs-Gemini comparison: when EVERY leaf's LLM call
    # fails (e.g. total rate-limiting), the old behavior silently returned a
    # 100%-fallback map that looked like a normal successful result to any
    # caller just counting nodes -- misreported as an AI-engine success by
    # cli.py (and, for cerebro batch, counted as a genuine item success
    # instead of the honestly-reported failure a video item would get).
    from cerebro.llm.base import LLMError

    import pytest

    class AlwaysFailsProvider(MockProvider):
        def complete_json(self, system, user):
            raise LLMError("boom")

    transcript = _two_chapter_transcript()
    with pytest.raises(LLMError, match="All section-enrichment calls failed"):
        build_outline_map(transcript, provider=AlwaysFailsProvider(), cache=Cache(enabled=False), level="full")


def test_section_boundary_uses_flat_document_order_not_nesting():
    # Chapter 2 starts on page 3; Section 1.2 (page 2) must stop before it
    # even though they're at different nesting depths.
    transcript = _transcript(
        outline=[(1, "Chapter 1", 0), (2, "Section 1.2", 2), (1, "Chapter 2", 3)],
        pages=["p0", "p1", "section one two content", "chapter two content"],
    )
    mm = build_outline_skeleton(transcript)
    sec12 = mm.root.children[0].children[0]
    assert "section one two content" in sec12.note
    assert "chapter two content" not in sec12.note
