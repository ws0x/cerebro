"""Direct tests for prompts.py -- previously verified only by hand (ad-hoc
python -c scripts) across 4 rounds of edits this session (v3->v7), never as
a permanent regression guard. Locks in: every builder formats cleanly with
no leftover placeholders, the rules that were the subject of real regressions
this session (anchors, numeric anchors, brief-level emptiness) are present in
the actual rendered text, and NODE_TYPES never drifts from the NodeType enum.
"""

import pytest

from cerebro.ir import NodeType
from cerebro.prompts import (
    FOLDER_LABEL_SYSTEM,
    HEADING_POLISH_SYSTEM,
    MAP_SYSTEM,
    NODE_TYPES,
    PROMPT_VERSION,
    cross_link_system,
    link_system,
    reduce_system,
    section_fill_system,
)

_PLACEHOLDERS = ["{node_types}", "{grounding}", "{anchors}", "{notes}", "{depth}", "{limit}", "{points_guidance}"]


def _assert_no_leftover_placeholders(text: str):
    for placeholder in _PLACEHOLDERS:
        assert placeholder not in text, f"unformatted {placeholder!r} leaked into rendered prompt"


def test_prompt_version_is_a_nonempty_string():
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION


def test_node_types_matches_the_nodetype_enum_except_root():
    # root is structural, never assigned by an LLM -- every other member must
    # appear as a whole word in NODE_TYPES, or a new type silently becomes
    # unreachable from every LLM-facing prompt.
    listed = {t.strip() for t in NODE_TYPES.split(",")}
    expected = {member.value for member in NodeType if member != NodeType.root}
    assert listed == expected


def test_map_system_formats_cleanly_and_carries_both_rules():
    _assert_no_leftover_placeholders(MAP_SYSTEM)
    assert "GROUNDING RULE" in MAP_SYSTEM
    assert "ANCHOR RULE" in MAP_SYSTEM
    assert "NUMERIC ANCHORS" in MAP_SYSTEM


def test_map_system_numeric_anchor_rule_is_emphatic_and_non_negotiable():
    # regression guard for the exact bug this rule was written to fix:
    # numbers surviving at brief but silently dropped at expert.
    assert "NON-NEGOTIABLE" in MAP_SYSTEM
    assert "VERBATIM" in MAP_SYSTEM
    assert "digit-for-digit" in MAP_SYSTEM


@pytest.mark.parametrize("level", ["brief", "full", "expert"])
def test_reduce_system_formats_cleanly_for_every_level(level):
    text = reduce_system(level)
    _assert_no_leftover_placeholders(text)
    assert "GROUNDING RULE" in text
    assert "ANCHOR RULE" in text
    assert "DEDUP" in text


def test_reduce_system_falls_back_to_full_for_an_unknown_level():
    assert reduce_system("not-a-real-level") == reduce_system("full")


def test_reduce_system_levels_are_genuinely_different_prompts():
    brief, full, expert = reduce_system("brief"), reduce_system("full"), reduce_system("expert")
    assert len({brief, full, expert}) == 3


def test_reduce_system_brief_asks_for_minimal_nesting_not_a_bigger_tree():
    text = reduce_system("brief")
    assert "minimal nesting" in text
    assert "ADVANCE-ORGANIZER" in text


def test_reduce_system_expert_requires_notes_on_every_branch():
    text = reduce_system("expert")
    assert 'every branch and every sub-branch MUST have a "note"' in text


@pytest.mark.parametrize("limit", [3, 8, 15])
def test_link_system_embeds_the_given_limit(limit):
    text = link_system(limit)
    _assert_no_leftover_placeholders(text)
    assert f"Return at most {limit} of the strongest relationships." in text


def test_link_system_prefers_cross_section_links():
    text = link_system(8)
    assert "DIFFERENT sections" in text


@pytest.mark.parametrize("limit", [3, 8])
def test_cross_link_system_embeds_the_given_limit(limit):
    text = cross_link_system(limit)
    _assert_no_leftover_placeholders(text)
    assert f"Return at most {limit} of the strongest cross-video relationships." in text


def test_cross_link_system_never_links_within_the_same_video():
    assert "Do NOT link nodes within the same video." in cross_link_system(8)


@pytest.mark.parametrize("level", ["brief", "full", "expert"])
def test_section_fill_system_formats_cleanly_for_every_level(level):
    text = section_fill_system(level)
    _assert_no_leftover_placeholders(text)
    assert "GROUNDING RULE" in text
    assert "ANCHOR RULE" in text


def test_section_fill_system_brief_requests_an_empty_points_list():
    # regression guard: brief must stay branches-only even on the
    # enumerated/document structuring path, not just the generic REDUCE path.
    text = section_fill_system("brief")
    assert 'Return an EMPTY "points" list' in text


def test_section_fill_system_levels_are_genuinely_different_prompts():
    brief, full, expert = (section_fill_system(lvl) for lvl in ("brief", "full", "expert"))
    assert len({brief, full, expert}) == 3


def test_section_fill_system_falls_back_to_full_for_an_unknown_level():
    assert section_fill_system("not-a-real-level") == section_fill_system("full")


def test_heading_polish_system_forbids_inventing_headings():
    assert "GROUNDING" in HEADING_POLISH_SYSTEM
    assert "Do not invent a" in HEADING_POLISH_SYSTEM
    assert "the author didn't express" in HEADING_POLISH_SYSTEM


def test_folder_label_system_is_grounded_in_given_names_only():
    assert "GROUNDING RULE" in FOLDER_LABEL_SYSTEM
