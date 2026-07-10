from cerebro.cache import Cache
from cerebro.ingest.youtube import _fetch_segments, extract_video_id


def test_extract_video_id_handles_common_url_shapes():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_captions_are_cached_across_calls(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_raw(video_id, languages):
        calls.append(video_id)
        return [{"text": "hello world", "start": 0.0, "duration": 2.0}]

    monkeypatch.setattr("cerebro.ingest.youtube._fetch_segments_raw", fake_fetch_raw)

    cache = Cache(root=tmp_path, enabled=True)

    segments1 = _fetch_segments("abc123", ["en"], cache=cache)
    assert len(calls) == 1
    assert segments1[0].text == "hello world"

    segments2 = _fetch_segments("abc123", ["en"], cache=cache)
    assert len(calls) == 1  # served from cache, no second network call
    assert segments2[0].text == "hello world"


def test_different_video_ids_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_raw(video_id, languages):
        calls.append(video_id)
        return [{"text": f"video {video_id}", "start": 0.0, "duration": 2.0}]

    monkeypatch.setattr("cerebro.ingest.youtube._fetch_segments_raw", fake_fetch_raw)
    cache = Cache(root=tmp_path, enabled=True)

    _fetch_segments("aaa", ["en"], cache=cache)
    _fetch_segments("bbb", ["en"], cache=cache)
    assert calls == ["aaa", "bbb"]


def test_different_languages_do_not_share_a_cache_entry(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_raw(video_id, languages):
        calls.append(tuple(languages))
        return [{"text": "x", "start": 0.0, "duration": 1.0}]

    monkeypatch.setattr("cerebro.ingest.youtube._fetch_segments_raw", fake_fetch_raw)
    cache = Cache(root=tmp_path, enabled=True)

    _fetch_segments("abc123", ["en"], cache=cache)
    _fetch_segments("abc123", ["fr"], cache=cache)
    assert calls == [("en",), ("fr",)]


def test_no_cache_always_fetches_fresh(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_raw(video_id, languages):
        calls.append(video_id)
        return [{"text": "x", "start": 0.0, "duration": 1.0}]

    monkeypatch.setattr("cerebro.ingest.youtube._fetch_segments_raw", fake_fetch_raw)

    _fetch_segments("abc123", ["en"], cache=None)
    _fetch_segments("abc123", ["en"], cache=None)
    assert len(calls) == 2  # no cache given -> no reuse, matches old behavior


def test_disabled_cache_always_fetches_fresh(tmp_path, monkeypatch):
    calls = []

    def fake_fetch_raw(video_id, languages):
        calls.append(video_id)
        return [{"text": "x", "start": 0.0, "duration": 1.0}]

    monkeypatch.setattr("cerebro.ingest.youtube._fetch_segments_raw", fake_fetch_raw)
    cache = Cache(root=tmp_path, enabled=False)

    _fetch_segments("abc123", ["en"], cache=cache)
    _fetch_segments("abc123", ["en"], cache=cache)
    assert len(calls) == 2
