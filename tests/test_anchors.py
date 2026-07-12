"""Tests for anchor verify-and-repair.

Detection is pure/deterministic -- tested directly. Repair uses a stub
provider (no network) so the insert/attachment logic is exercised without a
real LLM.
"""

from __future__ import annotations

from cerebro.ir import MindMap, Node, NodeType
from cerebro.structure.anchors import (
    find_missing_anchors,
    verify_and_repair_anchors,
)


def _map(*leaf_titles, notes=None):
    root = Node(title="Doc", type=NodeType.root)
    branch = root.add("Section", type=NodeType.topic, note=(notes or None))
    for t in leaf_titles:
        branch.add(t, type=NodeType.detail)
    return MindMap(title="Doc", root=root, level="full")


# -- detection -------------------------------------------------------------

def test_missing_number_is_detected():
    source = "The input layer has 784 neurons and 13,000 parameters total."
    mm = _map("Input Layer", "Parameters")
    missing = find_missing_anchors(source, mm)
    assert "784" in missing
    assert "13,000" in missing


def test_present_number_is_not_flagged_even_with_comma_mismatch():
    source = "The network trains on 13,000 examples."
    mm = _map("Trains on 13000 examples")  # map wrote it without the comma
    assert find_missing_anchors(source, mm) == []


def test_bare_single_digit_is_not_treated_as_an_anchor():
    source = "There is 1 key idea and 2 supporting points here."
    mm = _map("Key idea")
    assert find_missing_anchors(source, mm) == []


def test_percentage_and_currency_are_detected():
    source = "Accuracy hit 20% at a cost of $1,500."
    mm = _map("Accuracy", "Cost")
    missing = find_missing_anchors(source, mm)
    assert any("20" in m for m in missing)
    assert any("1,500" in m or "1500" in m for m in missing)


def test_such_as_list_items_are_detected():
    source = "To fight overfitting we use techniques such as dropout, weight decay, and early stopping."
    mm = _map("Regularization")  # the techniques themselves are absent
    missing = find_missing_anchors(source, mm)
    joined = " ".join(missing).lower()
    assert "dropout" in joined
    assert "weight decay" in joined


def test_list_items_already_present_are_not_flagged():
    source = "Techniques such as dropout and weight decay help."
    mm = _map("Dropout", "Weight Decay")
    assert find_missing_anchors(source, mm) == []


def test_multi_word_proper_noun_is_detected():
    source = "As Carl Jung argued, the unconscious shapes behaviour."
    mm = _map("The unconscious")
    missing = find_missing_anchors(source, mm)
    assert "Carl Jung" in missing


def test_proper_noun_already_present_is_not_flagged():
    source = "As Carl Jung argued, the unconscious shapes behaviour."
    mm = _map("Carl Jung on the unconscious")
    assert find_missing_anchors(source, mm) == []


def test_detection_is_capped():
    source = " ".join(f"metric {n}00" for n in range(1, 20))  # many 3-digit numbers
    mm = _map("Nothing relevant")
    assert len(find_missing_anchors(source, mm)) <= 8


# -- repair (stub provider) ------------------------------------------------

class _StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, repairs):
        self._repairs = repairs
        self.calls = 0

    def complete_json(self, system, user):
        self.calls += 1
        return {"repairs": self._repairs}


class _CountingCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


def test_repair_attaches_missing_anchors_as_children():
    source = "Techniques such as dropout, weight decay, and early stopping help."
    mm = _map("Regularization")
    # node ids: 0 = "Section", 1 = "Regularization"
    provider = _StubProvider(
        repairs=[
            {"to": 1, "title": "Dropout", "type": "example"},
            {"to": 1, "title": "Weight Decay", "type": "example"},
        ]
    )
    n = verify_and_repair_anchors(mm, source, provider, _CountingCache(), "full")
    assert n == 2
    titles = {node.title for node in mm.root.walk()}
    assert "Dropout" in titles
    assert "Weight Decay" in titles


def test_no_missing_anchors_means_no_provider_call():
    source = "Techniques such as dropout and weight decay help."
    mm = _map("Dropout", "Weight Decay")
    provider = _StubProvider(repairs=[])
    n = verify_and_repair_anchors(mm, source, provider, _CountingCache(), "full")
    assert n == 0
    assert provider.calls == 0  # nothing missing -> no re-prompt spent


def test_brief_level_skips_repair_entirely():
    source = "The input has 784 neurons and 13,000 parameters."
    mm = _map("Input")
    mm.level = "brief"
    provider = _StubProvider(repairs=[{"to": 0, "title": "784", "type": "detail"}])
    n = verify_and_repair_anchors(mm, source, provider, _CountingCache(), "brief")
    assert n == 0
    assert provider.calls == 0


def test_repair_ignores_out_of_range_node_ids():
    source = "The input has 784 neurons."
    mm = _map("Input")
    provider = _StubProvider(repairs=[{"to": 99, "title": "784", "type": "detail"}])
    n = verify_and_repair_anchors(mm, source, provider, _CountingCache(), "full")
    assert n == 0  # bad id discarded, not crashed
