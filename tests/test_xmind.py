import json
import zipfile
from pathlib import Path

import pytest

from cerebro.convert import write_xmind
from cerebro.convert.xmind import _source_href, mindmap_to_xmind_content
from cerebro.ir import MindMap, Node, NodeType, Relationship

_REFERENCE_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "examples" / "xmind_theme_template" / "DEFAULT_MAP_TEMPLATE.xmind"
)
_TREE_REFERENCE_TEMPLATE = (
    Path(__file__).resolve().parent.parent / "examples" / "xmind_theme_template" / "TREE_MAP_TEMPLATE.xmind"
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


def test_detail_nodes_get_their_own_marker_distinct_from_definition():
    root = Node(title="Central", type=NodeType.root)
    root.add("784 input neurons", type=NodeType.detail)
    root.add("A neuron is a unit that outputs a number", type=NodeType.definition)
    sheet = mindmap_to_xmind_content(MindMap(title="T", root=root))[0]
    detail, definition = sheet["rootTopic"]["children"]["attached"]
    assert detail["markers"] == [{"markerId": "star-orange"}]
    assert definition["markers"] == [{"markerId": "symbol-info"}]
    assert detail["markers"] != definition["markers"]


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


def test_source_href_passes_through_a_web_url():
    assert _source_href("https://youtu.be/abc123") == "https://youtu.be/abc123"
    assert _source_href("http://example.com/article") == "http://example.com/article"


def test_source_href_passes_through_an_existing_file_uri_unchanged():
    uri = "file:///C:/some/existing/path.pdf"
    assert _source_href(uri) == uri


def test_source_href_converts_an_existing_local_path_to_a_file_uri(tmp_path):
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-fake")
    href = _source_href(str(pdf))
    assert href.startswith("file:")
    assert "notes.pdf" in href


def test_source_href_is_none_for_a_nonexistent_local_path():
    assert _source_href("C:/definitely/not/a/real/file.pdf") is None


def test_source_href_is_none_for_no_source():
    assert _source_href(None) is None
    assert _source_href("") is None


def test_root_topic_gets_an_href_when_source_is_a_url():
    mm = MindMap(title="T", root=Node(title="T", type=NodeType.root), source="https://youtu.be/abc123")
    sheet = mindmap_to_xmind_content(mm)[0]
    assert sheet["rootTopic"]["href"] == "https://youtu.be/abc123"


def test_root_topic_has_no_href_when_source_is_none():
    mm = MindMap(title="T", root=Node(title="T", type=NodeType.root), source=None)
    sheet = mindmap_to_xmind_content(mm)[0]
    assert "href" not in sheet["rootTopic"]


def test_non_root_topics_never_get_an_href():
    mm = _map()
    mm.source = "https://youtu.be/abc123"
    sheet = mindmap_to_xmind_content(mm)[0]
    for child in sheet["rootTopic"]["children"]["attached"]:
        assert "href" not in child


def test_href_round_trips_through_write_and_real_zip_read(tmp_path):
    mm = MindMap(title="T", root=Node(title="T", type=NodeType.root), source="https://youtu.be/abc123")
    path = write_xmind(mm, tmp_path / "linked.xmind")
    with zipfile.ZipFile(path) as z:
        content = json.loads(z.read("content.json"))
    assert content[0]["rootTopic"]["href"] == "https://youtu.be/abc123"


def _tree_map():
    root = Node(title="my_project", type=NodeType.root)
    root.add("src", type=NodeType.topic)
    root.add("tests", type=NodeType.topic)
    return MindMap(title="my_project", root=root, level="structure")


def test_a_structure_level_map_gets_the_tree_theme_not_the_video_theme():
    sheet = mindmap_to_xmind_content(_tree_map())[0]
    assert sheet["theme"]["colorThemeId"] == "Hawaii-#FFFFFF-MULTI_LINE_COLORS"
    assert sheet["rootTopic"]["structureClass"] == "org.xmind.ui.logic.right"
    assert sheet["extensions"] == [
        {
            "provider": "org.xmind.ui.skeleton.structure.style",
            "content": {"centralTopic": "org.xmind.ui.logic.right"},
        }
    ]


def test_a_non_structure_map_still_gets_the_video_theme():
    sheet = mindmap_to_xmind_content(_map())[0]  # _map() uses the default level ("full")
    assert sheet["theme"]["colorThemeId"] == "Dawn-#ffffff-MULTI_LINE_COLORS"
    assert sheet["rootTopic"]["structureClass"] == "org.xmind.ui.map.clockwise"


@pytest.mark.parametrize("level", ["brief", "full", "expert", "merged"])
def test_only_the_literal_structure_level_gets_the_tree_theme(level):
    root = Node(title="X", type=NodeType.root)
    mm = MindMap(title="X", root=root, level=level)
    sheet = mindmap_to_xmind_content(mm)[0]
    assert sheet["theme"]["colorThemeId"] == "Dawn-#ffffff-MULTI_LINE_COLORS"


@pytest.mark.skipif(not _TREE_REFERENCE_TEMPLATE.exists(), reason="tree reference template backup not present")
def test_tree_theme_is_byte_identical_to_the_preserved_reference_template():
    with zipfile.ZipFile(_TREE_REFERENCE_TEMPLATE) as z:
        reference_content = json.loads(z.read("content.json"))
    reference_theme = reference_content[0]["theme"]

    sheet = mindmap_to_xmind_content(_tree_map())[0]
    assert sheet["theme"] == reference_theme


def test_tree_theme_round_trips_through_a_real_zip_write_and_read(tmp_path):
    path = write_xmind(_tree_map(), tmp_path / "tree.xmind")
    with zipfile.ZipFile(path) as z:
        content = json.loads(z.read("content.json"))
    assert content[0]["theme"]["colorThemeId"] == "Hawaii-#FFFFFF-MULTI_LINE_COLORS"
