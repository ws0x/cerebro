"""Regression test for a real bug found via a live Groq-vs-Gemini comparison:
when every LLM call fails and _structure() falls back to HeuristicStructurer,
the caller must be told, or the final "Map built with {engine}" message (and
the persisted manifest, and --json output) misattributes a heuristic-only map
to the LLM engine that actually failed."""

from cerebro.cache import Cache
from cerebro.cli import _structure
from cerebro.ir import NodeType
from cerebro.llm.base import LLMError
from cerebro.llm.providers import MockProvider
from cerebro.transcript import Segment, Transcript


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
    mm, used_fallback = _structure(_transcript(), "full", _AlwaysFailProvider(), cache=Cache(enabled=False))
    assert used_fallback is True
    # the map itself really is the heuristic shape (topic/detail nodes only)
    assert all(c.type == NodeType.topic for c in mm.root.children)


def test_a_successful_llm_run_reports_no_fallback():
    mm, used_fallback = _structure(_transcript(), "full", MockProvider(), cache=Cache(enabled=False))
    assert used_fallback is False
    assert mm.root.children  # a real map was built, not a placeholder


def test_no_provider_at_all_reports_no_fallback():
    # provider=None means "use heuristic on purpose" (no key configured) --
    # not a fallback from a failure, so this must stay False.
    mm, used_fallback = _structure(_transcript(), "full", None, cache=Cache(enabled=False))
    assert used_fallback is False
    assert mm.root.children
