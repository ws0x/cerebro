import xml.etree.ElementTree as ET

from cerebro.convert import mindmap_to_opml
from cerebro.ir import MindMap, Node, NodeType


def _build():
    root = Node(title="Central & <Topic>", type=NodeType.root)
    t = root.add("Branch \"one\"", type=NodeType.topic, timestamp=62.0, note="a note")
    t.add("Leaf", type=NodeType.detail)
    return MindMap(title="Test", root=root)


def test_opml_is_wellformed_and_escaped():
    xml = mindmap_to_opml(_build())
    # Must parse — proves special chars (& < ") were escaped correctly.
    tree = ET.fromstring(xml)
    assert tree.tag == "opml"
    assert tree.attrib["version"] == "2.0"

    body = tree.find("body")
    central = body.find("outline")
    assert central.attrib["text"] == "Central & <Topic>"

    branch = central.find("outline")
    assert branch.attrib["text"] == 'Branch "one"'
    # Timestamp is folded into the note as a [m:ss] prefix.
    assert branch.attrib["_note"].startswith("[1:02]")
    assert "a note" in branch.attrib["_note"]


def test_hierarchy_preserved():
    xml = mindmap_to_opml(_build())
    tree = ET.fromstring(xml)
    leaves = tree.findall(".//outline/outline/outline")
    assert len(leaves) == 1
    assert leaves[0].attrib["text"] == "Leaf"
