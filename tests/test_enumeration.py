from cerebro.structure.enumeration import (
    EnumeratedSection,
    _clean_heading,
    _smart_titlecase,
    detect_enumeration,
)
from cerebro.transcript import Segment, Transcript


def _t(*texts, start_step=5.0):
    segs = [Segment(text=x, start=i * start_step) for i, x in enumerate(texts)]
    return Transcript(source="x", title="Test", segments=segs)


def test_smart_titlecase_matches_template_style():
    assert _smart_titlecase("keep promises to yourself") == "Keep Promises to Yourself"
    assert _smart_titlecase("standards over feelings") == "Standards Over Feelings"
    assert _smart_titlecase("get your house in order") == "Get Your House in Order"
    assert _smart_titlecase("have something bigger than yourself") == "Have Something Bigger Than Yourself"


def test_clean_heading_strips_leading_filler_and_cuts_at_sentence():
    assert _clean_heading("is to have something bigger than yourself. Bodybuilding was") == "Have Something Bigger Than Yourself"
    assert _clean_heading("keep promises to yourself. This one is super simple") == "Keep Promises to Yourself"


def test_detects_number_word_list():
    t = _t(
        "Welcome to the video, let me set this up for you.",
        "Non-negotiable number one is keep promises to yourself, always.",
        "Now non-negotiable number two is get your house in order today too.",
        "And non-negotiable number three is do hard things intentionally.",
    )
    secs = detect_enumeration(t)
    assert [s.number for s in secs] == [1, 2, 3]
    assert [s.heading for s in secs] == [
        "Keep Promises to Yourself",       # comma cut drops ", always"
        "Get Your House in Order Today Too",
        "Do Hard Things Intentionally",
    ]


def test_detects_digit_and_hash_forms():
    t = _t(
        "Intro words here for context and setup.",
        "Tip #1 is drink water every morning without fail.",
        "Tip #2 is sleep eight hours a night consistently.",
        "Tip #3 is walk ten thousand steps daily.",
    )
    secs = detect_enumeration(t)
    assert [s.number for s in secs] == [1, 2, 3]
    assert secs[0].heading == "Drink Water Every Morning Without Fail"


def test_bare_list_noun_requires_a_declaration_marker():
    # "step three is X" should match; "win some[thing] one time" must NOT
    # (the false positive that broke the real video's #1).
    t = _t(
        "To win something one time is hard, but to keep winning takes work.",
        "The first real point, step one is show up every single day.",
        "Next, step two is track your progress carefully over time.",
        "Finally, step three is rest and recover fully each week.",
    )
    secs = detect_enumeration(t)
    assert [s.heading for s in secs] == [
        "Show Up Every Single Day",
        "Track Your Progress Carefully Over Time",
        "Rest and Recover Fully Each Week",
    ]


def test_bare_no_word_does_not_trigger():
    # "no one", "no two ways" must never be read as list cues (only "No." with
    # a period is an abbreviation cue).
    t = _t(
        "There is no one answer and no two ways about it, honestly.",
        "Number one is be consistent with your habits over time.",
        "Number two is measure what actually matters to you.",
        "Number three is adjust your plan as you learn more.",
    )
    secs = detect_enumeration(t)
    assert [s.number for s in secs] == [1, 2, 3]
    assert secs[0].heading == "Be Consistent With Your Habits Over Time"


def test_requires_at_least_three_items():
    t = _t(
        "Number one is show up every day no matter what happens.",
        "Number two is do the work even when it is hard.",
        "The rest of the talk has no more numbered points at all.",
    )
    assert detect_enumeration(t) == []  # only 2 real items -> not a list


def test_non_enumerated_transcript_returns_empty():
    t = _t(
        "Today I want to talk about how neural networks actually learn.",
        "A network adjusts its weights using gradient descent over time.",
        "Backpropagation is how the error signal flows backward through layers.",
        "With enough data and training the model generalizes to new inputs.",
    )
    assert detect_enumeration(t) == []


def test_out_of_order_recap_does_not_corrupt_the_spine():
    # A later "number one again" recap must not restart or duplicate the chain.
    t = _t(
        "Number one is keep your promises to yourself every day.",
        "Number two is get your personal house in order first.",
        "Number three is become a real student of yourself.",
        "So again, number one, keep those promises, that is the foundation.",
    )
    secs = detect_enumeration(t)
    assert [s.number for s in secs] == [1, 2, 3]  # the recap's "number one" is ignored


def test_sections_carry_correct_timestamps_and_indices():
    t = _t(
        "Intro segment zero here.",           # seg 0, t=0
        "Number one is alpha bravo charlie.",  # seg 1, t=5
        "Number two is delta echo foxtrot.",   # seg 2, t=10
        "Number three is golf hotel india.",   # seg 3, t=15
        start_step=5.0,
    )
    secs = detect_enumeration(t)
    assert secs[0].start == 5.0 and secs[0].seg_index == 1
    assert secs[1].start == 10.0 and secs[1].seg_index == 2
    assert secs[2].start == 15.0 and secs[2].seg_index == 3


def test_gate_starts_at_one_not_mid_sequence():
    # A list that only ever says "number three/four/five" (no 1) isn't a
    # recoverable spine -- the greedy chain needs to start at 1.
    t = _t(
        "Number three is something about the middle of a list.",
        "Number four is another mid-list item here for you.",
        "Number five is the final mid-list item we discuss.",
    )
    assert detect_enumeration(t) == []


def test_returns_enumerated_section_dataclass():
    t = _t(
        "Number one is stay disciplined through the hard days.",
        "Number two is keep showing up for the people you love.",
        "Number three is build something bigger than your own ego.",
    )
    secs = detect_enumeration(t)
    assert all(isinstance(s, EnumeratedSection) for s in secs)
    assert secs[0].heading_raw  # raw span preserved for optional LLM polish
