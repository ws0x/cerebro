import json
import zipfile
from pathlib import Path

import pytest

from cerebro.convert import write_xmind
from cerebro.convert.xmind import mindmap_to_xmind_content
from cerebro.ir import MindMap, Node, NodeType, Relationship

_REFERENCE_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "examples" / "xmind_theme_template" / "DEFAULT_MAP_TEMPLATE.xmind"
)


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


def test_sheet_carries_the_reference_theme():
    sheet = mindmap_to_xmind_content(_map())[0]
    theme = sheet["theme"]
    assert theme["colorThemeId"] == "Dawn-#ffffff-MULTI_LINE_COLORS"
    assert theme["map"]["properties"]["multi-line-colors"] == "#FF6B6B #FF9F69 #97D3B6 #88E2D7 #6FD0F9 #E18BEE"
    assert theme["centralTopic"]["properties"]["fo:font-family"] == "NeverMind"
    # every theme level a real XMind Zen theme defines must be present --
    # a missing one falls back to XMind's own bland application default
    # for that level, quietly breaking the "matches the template" promise
    for level in (
        "map", "centralTopic", "mainTopic", "subTopic", "floatingTopic",
        "summaryTopic", "calloutTopic", "importantTopic", "minorTopic",
        "expiredTopic", "boundary", "zone", "summary", "relationship", "level3",
    ):
        assert level in theme, f"theme is missing the {level!r} level"


def test_root_topic_uses_the_clockwise_structure_class():
    sheet = mindmap_to_xmind_content(_map())[0]
    assert sheet["rootTopic"]["structureClass"] == "org.xmind.ui.map.clockwise"


def test_sheet_carries_the_skeleton_structure_extension():
    sheet = mindmap_to_xmind_content(_map())[0]
    assert sheet["extensions"] == [
        {
            "provider": "org.xmind.ui.skeleton.structure.style",
            "content": {"centralTopic": "org.xmind.ui.map.clockwise"},
        }
    ]


def test_theme_is_present_even_with_no_relationships():
    root = Node(title="Solo", type=NodeType.root)
    root.add("Only Child")
    mm = MindMap(title="Solo", root=root)
    sheet = mindmap_to_xmind_content(mm)[0]
    assert "theme" in sheet
    assert "relationships" not in sheet  # unchanged existing behavior -- no key when there's nothing to carry


def test_theme_round_trips_through_a_real_zip_write_and_read(tmp_path):
    path = write_xmind(_map(), tmp_path / "themed.xmind")
    with zipfile.ZipFile(path) as z:
        content = json.loads(z.read("content.json"))
    assert content[0]["theme"]["colorThemeId"] == "Dawn-#ffffff-MULTI_LINE_COLORS"


@pytest.mark.skipif(not _REFERENCE_TEMPLATE.exists(), reason="reference template backup not present")
def test_embedded_theme_is_byte_identical_to_the_preserved_reference_template():
    # The strongest possible check that the theme was transcribed into
    # xmind.py correctly: compare against the actual reference file kept in
    # the repo (examples/xmind_theme_template/), not a hand-verified sample.
    with zipfile.ZipFile(_REFERENCE_TEMPLATE) as z:
        reference_content = json.loads(z.read("content.json"))
    reference_theme = reference_content[0]["theme"]

    sheet = mindmap_to_xmind_content(_map())[0]
    assert sheet["theme"] == reference_theme
