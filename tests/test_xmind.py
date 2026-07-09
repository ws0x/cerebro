import json
import zipfile

from cerebro.convert import write_xmind
from cerebro.convert.xmind import mindmap_to_xmind_content
from cerebro.ir import MindMap, Node, NodeType, Relationship


def _map():
    root = Node(title="Central", type=NodeType.root)
    a = root.add("Branch A", type=NodeType.concept, timestamp=62.0, note="a note")
    warn = a.add("Careful here", type=NodeType.warning)
    b = root.add("Branch B", type=NodeType.topic)
    mm = MindMap(title="Test", root=root)
    mm.relationships.append(Relationship(from_id=warn.id, to_id=b.id, label="affects"))
    return mm


def test_content_structure_markers_and_relationships():
    content = mindmap_to_xmind_content(_map())
    assert isinstance(content, list) and len(content) == 1
    sheet = content[0]
    root = sheet["rootTopic"]
    assert root["title"] == "Central"
    assert root["structureClass"].startswith("org.xmind")

    branch_a = root["children"]["attached"][0]
    assert branch_a["markers"] == [{"markerId": "star-blue"}]  # concept
    assert branch_a["notes"]["plain"]["content"].startswith("[1:02]")

    warn = branch_a["children"]["attached"][0]
    assert warn["markers"] == [{"markerId": "symbol-exclam"}]  # warning

    assert len(sheet["relationships"]) == 1
    assert sheet["relationships"][0]["title"] == "affects"


def test_written_file_is_a_valid_xmind_zip(tmp_path):
    path = write_xmind(_map(), tmp_path / "out.xmind")
    assert path.exists()
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"content.json", "metadata.json", "manifest.json"} <= names
        # content.json must be valid JSON with a rootTopic.
        content = json.loads(z.read("content.json"))
        assert content[0]["rootTopic"]["title"] == "Central"
