import zipfile

from cerebro.cli import _export
from cerebro.ir import MindMap, Node, NodeType, Relationship


def _sample_mindmap(with_relationship=False):
    root = Node(title="Root", type=NodeType.root)
    a = root.add("A", type=NodeType.concept)
    b = root.add("B", type=NodeType.insight)
    relationships = [Relationship(from_id=a.id, to_id=b.id, label="relates to")] if with_relationship else []
    return MindMap(title="Root", root=root, relationships=relationships)


def test_export_corrects_a_mismatched_extension_to_match_format(tmp_path):
    # Regression test: an explicit --out whose extension disagrees with the
    # resolved --format (e.g. merge's auto-picked "xmind" when relationships
    # are present, while --out still says .opml from before) must not write
    # the wrong file type into a misleadingly-named file.
    mm = _sample_mindmap()
    mismatched = tmp_path / "my_map.opml"
    written, _ = _export(mm, "xmind", mismatched, "full", 0.1, yes=True)
    assert written.suffix == ".xmind"
    assert written.stem == "my_map"
    assert written.parent == tmp_path
    with zipfile.ZipFile(written) as z:
        assert "content.json" in z.namelist()


def test_export_leaves_a_matching_extension_untouched(tmp_path):
    mm = _sample_mindmap()
    matching = tmp_path / "my_map.xmind"
    written, _ = _export(mm, "xmind", matching, "full", 0.1, yes=True)
    assert written == matching


def test_export_corrects_extension_the_other_direction_too(tmp_path):
    mm = _sample_mindmap()
    mismatched = tmp_path / "my_map.xmind"
    written, _ = _export(mm, "opml", mismatched, "full", 0.1, yes=True)
    assert written.suffix == ".opml"
    assert written.read_text(encoding="utf-8").startswith("<?xml")


def test_export_reports_relationships_dropped_only_for_opml():
    mm = _sample_mindmap(with_relationship=True)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        _, dropped_opml = _export(mm, "opml", Path(d) / "a.opml", "expert", 0.1, yes=True)
        _, dropped_xmind = _export(mm, "xmind", Path(d) / "b.xmind", "expert", 0.1, yes=True)
    assert dropped_opml == 1
    assert dropped_xmind == 0
