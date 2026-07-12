import json
import zipfile

import pytest

from cerebro.convert.opml import write_opml
from cerebro.convert.xmind import write_xmind
from cerebro.ir import MindMap, Node, NodeType, Relationship
from cerebro.merge import MergeError, merge_maps, read_map, read_opml, read_xmind


def _sample_mindmap(title="My Map"):
    root = Node(title=title, type=NodeType.root)
    child = root.add("Chapter 1", type=NodeType.concept, note="a concept note")
    child.add("Detail A", type=NodeType.detail)
    root.add("Chapter 2", type=NodeType.warning, note="watch out")
    return MindMap(title=title, root=root, source="fake_source.pdf")


def test_read_opml_roundtrips_title_note_and_children(tmp_path):
    mm = _sample_mindmap("PDF Map")
    path = write_opml(mm, tmp_path / "map.opml")
    read_back = read_opml(path)
    assert read_back.title == "PDF Map"
    assert [c.title for c in read_back.root.children] == ["Chapter 1", "Chapter 2"]
    assert read_back.root.children[0].children[0].title == "Detail A"
    assert "a concept note" in read_back.root.children[0].note


def test_read_opml_roundtrips_node_type_via_cerebrotype_attr(tmp_path):
    mm = _sample_mindmap()
    path = write_opml(mm, tmp_path / "map.opml")
    read_back = read_opml(path)
    assert read_back.root.children[0].type == NodeType.concept
    assert read_back.root.children[1].type == NodeType.warning


def test_read_opml_never_carries_relationships(tmp_path):
    # OPML can't represent them on the way out either -- confirms there's
    # nothing to lose on the way back in that wasn't already lost on write.
    root = Node(title="R", type=NodeType.root)
    a = root.add("A")
    b = root.add("B")
    mm = MindMap(title="R", root=root, relationships=[Relationship(from_id=a.id, to_id=b.id, label="relates to")])
    path = write_opml(mm, tmp_path / "map.opml")
    assert read_opml(path).relationships == []


def test_read_xmind_roundtrips_title_note_children_and_type(tmp_path):
    mm = _sample_mindmap("XMind Map")
    path = write_xmind(mm, tmp_path / "map.xmind")
    read_back = read_xmind(path)
    assert read_back.title == "XMind Map"
    assert [c.title for c in read_back.root.children] == ["Chapter 1", "Chapter 2"]
    assert read_back.root.children[0].type == NodeType.concept
    assert read_back.root.children[1].type == NodeType.warning


def test_read_xmind_prefers_the_roots_own_href_over_the_file_path(tmp_path):
    root = Node(title="R", type=NodeType.root)
    root.add("Child")
    mm = MindMap(title="R", root=root, source="https://youtu.be/abc123")
    path = write_xmind(mm, tmp_path / "map.xmind")
    read_back = read_xmind(path)
    # the original video URL survives, not the .xmind file's own filesystem path
    assert read_back.source == "https://youtu.be/abc123"


def test_read_xmind_falls_back_to_the_file_path_when_theres_no_href(tmp_path):
    mm = MindMap(title="R", root=Node(title="R", type=NodeType.root), source=None)
    path = write_xmind(mm, tmp_path / "map.xmind")
    read_back = read_xmind(path)
    assert read_back.source == str(path)


def test_read_xmind_roundtrips_relationships(tmp_path):
    root = Node(title="R", type=NodeType.root)
    a = root.add("A", type=NodeType.concept)
    b = root.add("B", type=NodeType.insight)
    mm = MindMap(title="R", root=root, relationships=[Relationship(from_id=a.id, to_id=b.id, label="relates to")])
    path = write_xmind(mm, tmp_path / "map.xmind")
    read_back = read_xmind(path)
    assert len(read_back.relationships) == 1
    rel = read_back.relationships[0]
    assert rel.label == "relates to"
    # ids must resolve to the actual reconstructed nodes, not just be non-empty strings
    ids_in_tree = {n.id for n in read_back.root.walk()}
    assert rel.from_id in ids_in_tree
    assert rel.to_id in ids_in_tree


def test_read_map_dispatches_by_extension(tmp_path):
    mm = _sample_mindmap()
    opml_path = write_opml(mm, tmp_path / "map.opml")
    xmind_path = write_xmind(mm, tmp_path / "map.xmind")
    assert read_map(opml_path).title == mm.title
    assert read_map(xmind_path).title == mm.title


def test_read_map_rejects_unsupported_extension(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    with pytest.raises(MergeError):
        read_map(bad)


def test_read_opml_rejects_malformed_xml(tmp_path):
    bad = tmp_path / "broken.opml"
    bad.write_text("<opml><body><outline text=", encoding="utf-8")
    with pytest.raises(MergeError):
        read_opml(bad)


def test_read_xmind_rejects_a_non_zip_file(tmp_path):
    bad = tmp_path / "broken.xmind"
    bad.write_bytes(b"not a zip")
    with pytest.raises(MergeError):
        read_xmind(bad)


def test_read_xmind_rejects_an_implausibly_large_content_json(tmp_path, monkeypatch):
    # A crafted .xmind whose content.json entry declares a huge uncompressed
    # size -- the zip's own central directory records this without any
    # decompression needed, so this must be rejected before z.read() is ever
    # called on the entry.
    bad = tmp_path / "bomb.xmind"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        # A real oversized entry (rather than a hand-forged header) so this
        # exercises the actual file_size the zip format records.
        z.writestr("content.json", "[" + ("0" * 200) + "]")

    monkeypatch.setattr("cerebro.merge._MAX_CONTENT_JSON_BYTES", 50)  # lower the bar so the 200+ byte entry trips it
    with pytest.raises(MergeError, match="implausibly large"):
        read_xmind(bad)


def test_read_xmind_rejects_content_json_that_is_a_bare_object(tmp_path):
    bad = tmp_path / "wrong_shape.xmind"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("content.json", json.dumps({"not": "a list of sheets"}))
    with pytest.raises(MergeError):
        read_xmind(bad)


def test_read_xmind_rejects_content_json_that_is_a_bare_string(tmp_path):
    bad = tmp_path / "wrong_shape2.xmind"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("content.json", json.dumps("just a string"))
    with pytest.raises(MergeError):
        read_xmind(bad)


def test_read_xmind_rejects_a_sheet_that_is_not_an_object(tmp_path):
    bad = tmp_path / "wrong_sheet.xmind"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("content.json", json.dumps(["not a sheet object"]))
    with pytest.raises(MergeError):
        read_xmind(bad)


def test_read_xmind_rejects_an_empty_list(tmp_path):
    bad = tmp_path / "empty_list.xmind"
    with zipfile.ZipFile(bad, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("content.json", json.dumps([]))
    with pytest.raises(MergeError):
        read_xmind(bad)


def test_merge_maps_requires_at_least_two(tmp_path):
    mm = _sample_mindmap()
    path = write_opml(mm, tmp_path / "a.opml")
    with pytest.raises(MergeError):
        merge_maps([path])


def test_merge_maps_combines_into_one_root_with_a_branch_per_file(tmp_path):
    a_path = write_opml(_sample_mindmap("Video Map"), tmp_path / "a.opml")
    b_path = write_xmind(_sample_mindmap("PDF Map"), tmp_path / "b.xmind")

    merged = merge_maps([a_path, b_path], title="Combined")
    assert merged.title == "Combined"
    assert merged.root.type == NodeType.root
    assert [c.title for c in merged.root.children] == ["Video Map", "PDF Map"]
    for branch in merged.root.children:
        assert branch.type == NodeType.topic  # demoted from root when merged, same as batch.py


def test_merge_maps_preserves_relationships_per_file(tmp_path):
    root_a = Node(title="A", type=NodeType.root)
    x = root_a.add("X", type=NodeType.concept)
    y = root_a.add("Y", type=NodeType.insight)
    mm_a = MindMap(title="A", root=root_a, relationships=[Relationship(from_id=x.id, to_id=y.id, label="leads to")])
    path_a = write_xmind(mm_a, tmp_path / "a.xmind")
    path_b = write_opml(_sample_mindmap("B"), tmp_path / "b.opml")

    merged = merge_maps([path_a, path_b])
    assert len(merged.relationships) == 1
    assert merged.relationships[0].label == "leads to"


def test_merge_maps_works_with_three_or_more_files(tmp_path):
    paths = [write_opml(_sample_mindmap(f"Map {i}"), tmp_path / f"m{i}.opml") for i in range(4)]
    merged = merge_maps(paths, title="Big Combo")
    assert len(merged.root.children) == 4


def test_xmind_content_json_is_still_well_formed_after_a_merge_write(tmp_path):
    from cerebro.convert.xmind import write_xmind as write_merged_xmind

    a_path = write_opml(_sample_mindmap("A"), tmp_path / "a.opml")
    b_path = write_opml(_sample_mindmap("B"), tmp_path / "b.opml")
    merged = merge_maps([a_path, b_path])
    out = write_merged_xmind(merged, tmp_path / "combined.xmind")
    with zipfile.ZipFile(out) as z:
        data = json.loads(z.read("content.json"))
    assert data[0]["rootTopic"]["title"] == "Merged Map"
    assert len(data[0]["rootTopic"]["children"]["attached"]) == 2
