from pathlib import Path

from cerebro.manifest import lookup, record


def test_lookup_returns_none_when_nothing_recorded(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    assert lookup("some_video.mp4", "full", "opml", manifest_path=manifest_path) is None


def test_record_then_lookup_roundtrips(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    out = tmp_path / "my_map.opml"
    record("https://youtu.be/abc123", "full", "opml", "groq:llama-3.3-70b-versatile", out, manifest_path=manifest_path)

    found = lookup("https://youtu.be/abc123", "full", "opml", manifest_path=manifest_path)
    assert found is not None
    assert found["output"] == str(out)
    assert found["engine"] == "groq:llama-3.3-70b-versatile"
    assert "built_at" in found


def test_lookup_distinguishes_by_level(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    out = tmp_path / "my_map.opml"
    record("video.mp4", "full", "opml", "heuristic (offline)", out, manifest_path=manifest_path)
    assert lookup("video.mp4", "expert", "opml", manifest_path=manifest_path) is None
    assert lookup("video.mp4", "full", "opml", manifest_path=manifest_path) is not None


def test_lookup_distinguishes_by_format(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    out = tmp_path / "my_map.opml"
    record("video.mp4", "full", "opml", "heuristic (offline)", out, manifest_path=manifest_path)
    assert lookup("video.mp4", "full", "xmind", manifest_path=manifest_path) is None


def test_local_paths_normalize_regardless_of_how_theyre_written(tmp_path, monkeypatch):
    manifest_path = tmp_path / "map-manifest.json"
    video = tmp_path / "video.mp4"
    video.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    record(str(video), "full", "opml", "heuristic (offline)", tmp_path / "out.opml", manifest_path=manifest_path)
    # same file, referenced relatively instead of absolutely
    found = lookup("video.mp4", "full", "opml", manifest_path=manifest_path)
    assert found is not None


def test_a_second_record_overwrites_the_first_for_the_same_key(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    record("video.mp4", "full", "opml", "heuristic (offline)", tmp_path / "a.opml", manifest_path=manifest_path)
    record("video.mp4", "full", "opml", "groq:llama-3.3-70b-versatile", tmp_path / "b.opml", manifest_path=manifest_path)
    found = lookup("video.mp4", "full", "opml", manifest_path=manifest_path)
    assert found["engine"] == "groq:llama-3.3-70b-versatile"
    assert found["output"] == str(tmp_path / "b.opml")


def test_record_never_raises_when_manifest_dir_is_unwritable(tmp_path, monkeypatch):
    # Simulate a write failure -- record() must swallow it, never crash the build it's piggybacking on.
    bad_path = tmp_path / "nonexistent" / "deeply" / "nested" / "manifest.json"
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(PermissionError("nope")))
    record("video.mp4", "full", "opml", "heuristic (offline)", tmp_path / "out.opml", manifest_path=bad_path)  # must not raise


def test_lookup_returns_none_on_corrupt_manifest_file(tmp_path):
    manifest_path = tmp_path / "map-manifest.json"
    manifest_path.write_text("not valid json{{{", encoding="utf-8")
    assert lookup("video.mp4", "full", "opml", manifest_path=manifest_path) is None
