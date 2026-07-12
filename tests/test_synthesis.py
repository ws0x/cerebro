"""Tests for the additive synthesis pass.

Uses a stub provider (no network). The pass must be strictly additive: the
existing spine is never modified, only a 'Key Takeaways' branch is appended.
"""

from __future__ import annotations

from cerebro.ir import MindMap, Node, NodeType
from cerebro.structure.synthesis import add_synthesis


def _structured_map(n_sections=3):
    root = Node(title="Doc", type=NodeType.root)
    for i in range(n_sections):
        s = root.add(f"Chapter {i + 1}", type=NodeType.topic, note=f"section {i + 1}")
        s.add(f"point {i + 1}", type=NodeType.detail)
    return MindMap(title="Doc", root=root, level="full")


class _StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, takeaways):
        self._takeaways = takeaways
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        return {"takeaways": self._takeaways}


class _Cache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


def test_synthesis_appends_a_key_takeaways_branch():
    mm = _structured_map()
    original_children = [c.title for c in mm.root.children]
    provider = _StubProvider(
        takeaways=[
            {"title": "Generalization is the goal", "note": "the whole point", "type": "insight"},
            {"title": "Regularization fights overfitting", "note": "across sections", "type": "insight"},
        ]
    )
    n = add_synthesis(mm, provider, _Cache(), "full")
    assert n == 2

    # spine untouched: the original chapters are all still there, in order,
    # and a new branch is appended at the end.
    assert [c.title for c in mm.root.children][: len(original_children)] == original_children
    assert mm.root.children[-1].title == "Key Takeaways"
    assert mm.root.children[-1].type == NodeType.insight
    takeaway_titles = {c.title for c in mm.root.children[-1].children}
    assert "Generalization is the goal" in takeaway_titles


def test_synthesis_skipped_at_brief():
    mm = _structured_map()
    mm.level = "brief"
    provider = _StubProvider(takeaways=[{"title": "x", "type": "insight"}])
    n = add_synthesis(mm, provider, _Cache(), "brief")
    assert n == 0
    assert provider.calls == 0
    assert all(c.title != "Key Takeaways" for c in mm.root.children)


def test_synthesis_skipped_on_a_too_thin_map():
    root = Node(title="Doc", type=NodeType.root)
    root.add("Only one node", type=NodeType.topic)
    mm = MindMap(title="Doc", root=root, level="full")
    provider = _StubProvider(takeaways=[{"title": "x", "type": "insight"}])
    n = add_synthesis(mm, provider, _Cache(), "full")
    assert n == 0
    assert provider.calls == 0


def test_synthesis_empty_takeaways_adds_no_branch():
    mm = _structured_map()
    provider = _StubProvider(takeaways=[])
    n = add_synthesis(mm, provider, _Cache(), "full")
    assert n == 0
    assert all(c.title != "Key Takeaways" for c in mm.root.children)


def test_synthesis_ignores_malformed_takeaway_entries():
    mm = _structured_map()
    provider = _StubProvider(
        takeaways=[
            {"title": "Good one", "type": "insight"},
            {"note": "no title -- skip"},
            "not even a dict",
            {"title": "   ", "type": "insight"},  # blank title -> skip
        ]
    )
    n = add_synthesis(mm, provider, _Cache(), "full")
    assert n == 1
    assert mm.root.children[-1].children[0].title == "Good one"
