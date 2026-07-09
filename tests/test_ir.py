from cerebro.ir import MindMap, Node, NodeType


def test_tree_building_and_walk():
    root = Node(title="Root", type=NodeType.root)
    a = root.add("A", type=NodeType.topic)
    a.add("A1", type=NodeType.detail)
    a.add("A2", type=NodeType.detail)
    root.add("B")

    mm = MindMap(title="Root", root=root)
    assert mm.node_count() == 5  # root, A, A1, A2, B
    assert mm.depth() == 3
    titles = [n.title for n in root.walk()]
    assert titles == ["Root", "A", "A1", "A2", "B"]


def test_unique_ids():
    root = Node(title="Root")
    for i in range(50):
        root.add(f"n{i}")
    ids = [n.id for n in root.walk()]
    assert len(ids) == len(set(ids))
