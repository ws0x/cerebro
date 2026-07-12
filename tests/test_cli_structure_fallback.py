"""Regression tests for two related bugs:

1. When every LLM call fails and _structure() falls back to HeuristicStructurer,
   the caller must be told, or the final "Map built with {engine}" message (and
   the persisted manifest, and --json output) misattributes a heuristic-only
   map to the LLM engine that actually failed.
2. --engine auto tries every configured provider in the chain before
   degrading to heuristic -- a total failure on the first (a daily quota
   exhaustion, a rate-limit storm) fails over to the next, live-reproduced
   the hard way (Groq exhausted, then Gemini rate-limited, tried by hand).

_structure()/_structure_document() take a LIST of providers (the chain), not
a single one, and return (map, used_heuristic_fallback, used_provider) --
used_provider is whichever one actually built the map, which after failover
isn't necessarily the first in the chain.
"""

from cerebro.cache import Cache
from cerebro.cli import _structure, _structure_document
from cerebro.ir import NodeType
from cerebro.llm.base import LLMError
from cerebro.llm.providers import MockProvider
from cerebro.transcript import OutlineEntry, Segment, Transcript


class _AlwaysFailProvider:
    name = "mock"
    model = "always-fails"

    def complete_json(self, system: str, user: str) -> dict:
        raise LLMError("simulated total outage")


def _transcript():
    # Deliberately NOT phrased as an author-numbered list (no "number N" /
    # ordinal cues) -- this must exercise the generic MAP->REDUCE path, not
    # LLMStructurer's separate enumerated-list path (which has its own,
    # separately-tested total-failure handling in test_enumerated_structuring.py).
    return Transcript(
        source="s",
        title="T",
        segments=[Segment(text=f"This is sentence {i} about the topic at hand.", start=float(i)) for i in range(6)],
    )


def test_a_total_llm_failure_reports_the_fallback_flag():
    mm, used_fallback, used_provider = _structure(_transcript(), "full", [_AlwaysFailProvider()], cache=Cache(enabled=False))
    assert used_fallback is True
    assert used_provider is None
    # the map itself really is the heuristic shape (topic/detail nodes only)
    assert all(c.type == NodeType.topic for c in mm.root.children)


def test_a_successful_llm_run_reports_no_fallback():
    provider = MockProvider()
    mm, used_fallback, used_provider = _structure(_transcript(), "full", [provider], cache=Cache(enabled=False))
    assert used_fallback is False
    assert used_provider is provider
    assert mm.root.children  # a real map was built, not a placeholder


def test_a_total_llm_failure_prints_a_loud_fallback_panel(capsys):
    # Regression guard for the "silent scroll-away" bug found alongside this
    # one: a single plain console line was easy to miss on a long, multi-
    # minute build -- must now be a bordered panel, not just colored text.
    _structure(_transcript(), "full", [_AlwaysFailProvider()], cache=Cache(enabled=False))
    out = capsys.readouterr().out
    assert "Degraded to heuristic" in out
    assert "MAP calls failed" in out


def test_no_provider_at_all_reports_no_fallback():
    # An empty chain means "use heuristic on purpose" (no key configured) --
    # not a fallback from a failure, so this must stay False.
    mm, used_fallback, used_provider = _structure(_transcript(), "full", [], cache=Cache(enabled=False))
    assert used_fallback is False
    assert used_provider is None
    assert mm.root.children


# -- provider failover chain (--engine auto tries every configured provider) --

def test_failover_tries_the_next_provider_after_the_first_fails_entirely():
    healthy = MockProvider()
    mm, used_fallback, used_provider = _structure(
        _transcript(), "full", [_AlwaysFailProvider(), healthy], cache=Cache(enabled=False)
    )
    assert used_fallback is False  # a real LLM map was built -- NOT a heuristic degrade
    assert used_provider is healthy  # by the second provider in the chain
    assert mm.root.children


def test_failover_prints_which_provider_failed_and_which_is_next(capsys):
    _structure(_transcript(), "full", [_AlwaysFailProvider(), MockProvider()], cache=Cache(enabled=False))
    out = capsys.readouterr().out
    assert "mock" in out and "failed" in out
    assert "trying" in out.lower()


def test_failover_degrades_to_heuristic_only_after_every_provider_in_the_chain_fails():
    mm, used_fallback, used_provider = _structure(
        _transcript(), "full", [_AlwaysFailProvider(), _AlwaysFailProvider()], cache=Cache(enabled=False)
    )
    assert used_fallback is True
    assert used_provider is None
    assert all(c.type == NodeType.topic for c in mm.root.children)  # genuinely heuristic


def test_failover_does_not_trigger_when_the_first_provider_succeeds():
    # The whole point: a healthy first provider is used as-is, no wasted
    # attempt at a second one.
    first = MockProvider()
    second = _AlwaysFailProvider()
    mm, used_fallback, used_provider = _structure(_transcript(), "full", [first, second], cache=Cache(enabled=False))
    assert used_provider is first
    assert used_fallback is False


def _outline_transcript():
    # An outline-bearing source (PDF/article) routes through the separate
    # _structure_document path, which has its own (map, used_fallback,
    # used_provider) contract -- must be tested independently of the generic
    # path above.
    return Transcript(
        source="doc.pdf",
        title="Doc",
        segments=[Segment(text="Chapter one body text here.", start=0.0), Segment(text="Chapter two body text here.", start=1.0)],
        outline=[OutlineEntry(1, "Chapter One", 0), OutlineEntry(1, "Chapter Two", 1)],
    )


def test_outline_path_total_llm_failure_reports_the_fallback_flag():
    mm, used_fallback, used_provider = _structure_document(
        _outline_transcript(), "full", [_AlwaysFailProvider()], cache=Cache(enabled=False)
    )
    assert used_fallback is True
    assert used_provider is None
    assert [c.title for c in mm.root.children] == ["Chapter One", "Chapter Two"]  # skeleton, not blank


def test_outline_path_successful_llm_run_reports_no_fallback():
    provider = MockProvider()
    mm, used_fallback, used_provider = _structure_document(
        _outline_transcript(), "full", [provider], cache=Cache(enabled=False)
    )
    assert used_fallback is False
    assert used_provider is provider


def test_outline_path_no_provider_reports_no_fallback():
    mm, used_fallback, used_provider = _structure_document(_outline_transcript(), "full", [], cache=Cache(enabled=False))
    assert used_fallback is False
    assert used_provider is None


def test_outline_path_also_fails_over_to_the_next_provider():
    healthy = MockProvider()
    mm, used_fallback, used_provider = _structure_document(
        _outline_transcript(), "full", [_AlwaysFailProvider(), healthy], cache=Cache(enabled=False)
    )
    assert used_fallback is False
    assert used_provider is healthy
