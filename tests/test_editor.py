import pytest

from cerebro.editor import delete, flatten, node_at, parent_and_index, rename
from cerebro.ir import Node, NodeType


def _sample_tree():
    root = Node(title="Root", type=NodeType.root)
    a = root.add("A")
    a.add("A1")
    a.add("A2")
    root.add("B")
    return root


def test_flatten_visits_every_node_depth_first():
    root = _sample_tree()
    entries = flatten(root)
    assert [e.node.title for e in entries] == ["Root", "A", "A1", "A2", "B"]


def test_flatten_paths_and_depths_are_correct():
    root = _sample_tree()
    entries = {e.node.title: (e.path, e.depth) for e in flatten(root)}
    assert entries["Root"] == ((), 0)
    assert entries["A"] == ((0,), 1)
    assert entries["A1"] == ((0, 0), 2)
    assert entries["A2"] == ((0, 1), 2)
    assert entries["B"] == ((1,), 1)


def test_node_at_resolves_a_path():
    root = _sample_tree()
    assert node_at(root, (0,)).title == "A"
    assert node_at(root, (0, 1)).title == "A2"
    assert node_at(root, ()).title == "Root"


def test_parent_and_index_resolves_correctly():
    root = _sample_tree()
    parent, idx = parent_and_index(root, (0, 1))
    assert parent.title == "A"
    assert idx == 1
    assert parent.children[idx].title == "A2"


def test_rename_changes_only_the_targeted_node():
    root = _sample_tree()
    rename(root, (0, 0), "Renamed A1")
    assert node_at(root, (0, 0)).title == "Renamed A1"
    assert node_at(root, (0, 1)).title == "A2"  # untouched sibling


def test_rename_root_itself():
    root = _sample_tree()
    rename(root, (), "New Root Title")
    assert root.title == "New Root Title"


def test_delete_removes_the_node_and_its_subtree():
    root = _sample_tree()
    delete(root, (0,))  # delete A, taking A1/A2 with it
    assert [e.node.title for e in flatten(root)] == ["Root", "B"]


def test_delete_a_leaf_leaves_siblings_intact():
    root = _sample_tree()
    delete(root, (0, 0))  # delete A1 only
    assert [e.node.title for e in flatten(root)] == ["Root", "A", "A2", "B"]


def test_delete_root_raises():
    root = _sample_tree()
    with pytest.raises(ValueError, match="Cannot delete the root"):
        delete(root, ())


def test_paths_remain_valid_after_a_delete_shifts_sibling_indices():
    # Deleting A's first child (index 0) shifts what was index 1 down to 0 --
    # re-flattening (as the interactive loop always does after any edit)
    # must reflect that, not operate on a stale path.
    root = _sample_tree()
    delete(root, (0, 0))
    entries = flatten(root)
    a2_entry = next(e for e in entries if e.node.title == "A2")
    assert a2_entry.path == (0, 0)  # A2 is now A's first (and only) child
