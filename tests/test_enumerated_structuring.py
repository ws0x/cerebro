"""The LLMStructurer's author-numbered-list path, driven by MockProvider so
it's deterministic and offline."""

from cerebro.cache import Cache
from cerebro.ir import NodeType
from cerebro.llm.providers import MockProvider
from cerebro.structure.llm import LLMStructurer
from cerebro.transcript import Segment, Transcript


def _list_transcript(intro_words=8):
    intro = "Setup words here now " * (max(1, intro_words // 4))
    segs = [
        Segment(text=intro.strip() + ".", start=0.0),
        Segment(text="Number one is keep promises to yourself. It builds real confidence daily.", start=10.0),
        Segment(text="Number two is get your house in order. Your relationships are the foundation.", start=20.0),
        Segment(text="Number three is do hard things intentionally. Growth comes from chosen discomfort.", start=30.0),
    ]
    return Transcript(source="x", title="3 Non-Negotiables", segments=segs)


def _structure(level="full", transcript=None):
    transcript = transcript or _list_transcript()
    return LLMStructurer(MockProvider(), cache=Cache(enabled=False)).structure(transcript, level=level)


def test_enumerated_spine_is_numbered_in_order():
    mm = _structure()
    branch_titles = [c.title for c in mm.root.children if c.title[0].isdigit()]
    assert branch_titles == [
        "1. Keep Promises to Yourself",
        "2. Get Your House in Order",
        "3. Do Hard Things Intentionally",
    ]


def test_root_keeps_the_real_video_title():
    mm = _structure()
    assert mm.root.title == "3 Non-Negotiables"  # not the LLM's re-summarized "central"


def test_numbered_branches_carry_timestamps():
    mm = _structure()
    numbered = [c for c in mm.root.children if c.title[0].isdigit()]
    assert [c.timestamp for c in numbered] == [10.0, 20.0, 30.0]


def test_full_level_branches_have_notes_and_points():
    mm = _structure(level="full")
    numbered = [c for c in mm.root.children if c.title[0].isdigit()]
    for branch in numbered:
        assert branch.note  # mandatory note
        assert branch.children  # key points as children


def test_brief_level_has_notes_but_no_subpoints():
    # brief = advance organizer: numbered spine + gist note each, NO sub-points.
    # Enforced deterministically (MockProvider's SECTION reply always returns
    # points, so this proves the code drops them at brief, not the prompt).
    mm = _structure(level="brief")
    numbered = [c for c in mm.root.children if c.title[0].isdigit()]
    assert len(numbered) == 3
    assert all(c.note for c in numbered)
    assert all(not c.children for c in numbered)  # spine only, no points


def test_full_level_keeps_subpoints_unlike_brief():
    mm = _structure(level="full")
    numbered = [c for c in mm.root.children if c.title[0].isdigit()]
    assert all(c.children for c in numbered)  # full keeps the points brief drops


def test_long_intro_becomes_an_overview_branch():
    # 40+ intro words -> a leading unnumbered Overview advance-organizer branch.
    long_intro = " ".join(["context"] * 60)
    segs = [
        Segment(text=long_intro + ".", start=0.0),
        Segment(text="Number one is alpha bravo charlie delta echo foxtrot.", start=30.0),
        Segment(text="Number two is golf hotel india juliet kilo lima.", start=40.0),
        Segment(text="Number three is mike november oscar papa quebec romeo.", start=50.0),
    ]
    t = Transcript(source="x", title="Big Intro List", segments=segs)
    mm = _structure(transcript=t)
    assert mm.root.children[0].title == "Overview"
    assert mm.root.children[0].timestamp == 30.0  # sorts before "1."


def test_short_intro_has_no_overview_branch():
    mm = _structure()  # tiny intro
    assert all(c.title != "Overview" for c in mm.root.children)


def test_non_enumerated_transcript_uses_the_generic_path():
    # No author numbering -> the normal map->reduce path (MockProvider's REDUCE
    # returns "Central Topic" with "First/Second main branch").
    segs = [
        Segment(text="Neural networks learn by adjusting weights via gradient descent slowly.", start=0.0),
        Segment(text="Backpropagation sends the error signal backward through the layers.", start=10.0),
        Segment(text="With enough data the model generalizes to brand new unseen inputs.", start=20.0),
    ]
    t = Transcript(source="x", title="How Nets Learn", segments=segs)
    mm = _structure(transcript=t)
    assert not any(c.title[0].isdigit() for c in mm.root.children)  # no numbered spine
    assert any("main branch" in c.title for c in mm.root.children)  # generic REDUCE output


def test_expert_level_adds_relationships():
    mm = _structure(level="expert")
    # MockProvider's LINK returns one relationship (ids 1->3, different branches).
    assert len(mm.relationships) >= 1


def test_section_fill_failure_degrades_to_a_snippet_note_not_a_crash():
    class _FlakyProvider(MockProvider):
        def complete_json(self, system, user):
            if "TASK: SECTION" in system:
                from cerebro.llm.base import LLMError

                raise LLMError("section fill boom")
            return super().complete_json(system, user)

    t = _list_transcript()
    mm = LLMStructurer(_FlakyProvider(), cache=Cache(enabled=False)).structure(t, level="full")
    numbered = [c for c in mm.root.children if c.title[0].isdigit()]
    assert len(numbered) == 3  # spine still built
    assert all(c.note for c in numbered)  # deterministic snippet note kept
    assert all(not c.children for c in numbered)  # no points on failure, but no crash
