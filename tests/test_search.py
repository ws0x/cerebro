import json
import zipfile

from cerebro.search import iter_maps, search_maps


def _write_opml(path, title, note=""):
    path.write_text(
        f'<?xml version="1.0"?><opml version="2.0"><head><title>Root</title></head>'
        f'<body><outline text="{title}" _note="{note}">'
        f'<outline text="Child of {title}" _note="a child note"/>'
        f"</outline></body></opml>",
        encoding="utf-8",
    )


def _write_xmind(path, title, note=""):
    content = [
        {
            "title": "Sheet 1",
            "rootTopic": {
                "id": "root",
                "title": title,
                "notes": {"plain": {"content": note}},
                "children": {"attached": [{"id": "c1", "title": f"Child of {title}"}]},
            },
        }
    ]
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("content.json", json.dumps(content))
        z.writestr("manifest.json", "{}")
        z.writestr("metadata.json", "{}")


def test_iter_maps_finds_opml_and_xmind_only(tmp_path):
    _write_opml(tmp_path / "a.opml", "A")
    _write_xmind(tmp_path / "b.xmind", "B")
    (tmp_path / "notes.txt").write_text("not a map")
    found = iter_maps(tmp_path)
    assert {p.name for p in found} == {"a.opml", "b.xmind"}


def test_iter_maps_recurses_into_subfolders(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_opml(sub / "nested.opml", "Nested")
    found = iter_maps(tmp_path)
    assert any(p.name == "nested.opml" for p in found)


def test_iter_maps_on_missing_dir_returns_empty():
    assert iter_maps(__import__("pathlib").Path("/definitely/does/not/exist")) == []


def test_search_finds_match_in_opml_title(tmp_path):
    _write_opml(tmp_path / "neural.opml", "Backpropagation Basics")
    results = search_maps("backpropagation", tmp_path)
    assert len(results) == 1
    assert results[0].path.name == "neural.opml"
    assert results[0].nodes[0].title == "Backpropagation Basics"


def test_search_finds_match_in_opml_note(tmp_path):
    _write_opml(tmp_path / "caching.opml", "Chapter 1", note="explains consistent hashing in depth")
    results = search_maps("consistent hashing", tmp_path)
    assert len(results) == 1


def test_search_finds_match_in_xmind_title_and_note(tmp_path):
    _write_xmind(tmp_path / "systems.xmind", "Replication", note="covers eventual consistency")
    assert len(search_maps("replication", tmp_path)) == 1
    assert len(search_maps("eventual consistency", tmp_path)) == 1


def test_search_is_case_insensitive_by_default(tmp_path):
    _write_opml(tmp_path / "a.opml", "BackPropagation")
    assert len(search_maps("backpropagation", tmp_path)) == 1
    assert len(search_maps("BACKPROPAGATION", tmp_path)) == 1


def test_search_case_sensitive_mode_respects_case(tmp_path):
    _write_opml(tmp_path / "a.opml", "BackPropagation")
    assert len(search_maps("backpropagation", tmp_path, case_sensitive=True)) == 0
    assert len(search_maps("BackPropagation", tmp_path, case_sensitive=True)) == 1


def test_search_returns_no_results_for_nonmatching_query(tmp_path):
    _write_opml(tmp_path / "a.opml", "Something Else")
    assert search_maps("nonexistent_topic_xyz", tmp_path) == []


def test_search_empty_query_returns_no_results(tmp_path):
    _write_opml(tmp_path / "a.opml", "Anything")
    assert search_maps("", tmp_path) == []


def test_search_caps_matches_per_file(tmp_path):
    path = tmp_path / "many.opml"
    outlines = "".join(f'<outline text="Topic {i}" _note=""/>' for i in range(20))
    path.write_text(
        f'<?xml version="1.0"?><opml version="2.0"><head><title>R</title></head>'
        f'<body><outline text="Root"><outline text="Topic 0"/>{outlines}</outline></body></opml>',
        encoding="utf-8",
    )
    results = search_maps("topic", tmp_path, max_matches_per_file=5)
    assert len(results[0].nodes) == 5


def test_search_skips_a_corrupt_opml_file_without_crashing(tmp_path):
    (tmp_path / "broken.opml").write_text("<opml><body><outline text=", encoding="utf-8")
    _write_opml(tmp_path / "good.opml", "Findable Topic")
    results = search_maps("findable", tmp_path)
    assert len(results) == 1
    assert results[0].path.name == "good.opml"


def test_search_skips_a_corrupt_xmind_file_without_crashing(tmp_path):
    (tmp_path / "broken.xmind").write_bytes(b"not a zip file at all")
    _write_opml(tmp_path / "good.opml", "Findable Topic")
    results = search_maps("findable", tmp_path)
    assert len(results) == 1
