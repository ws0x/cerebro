import requests

from cerebro.cache import Cache
from cerebro.ingest.youtube import _fetch_segments, _fetch_title, _raw_to_segments, extract_video_id, load_youtube

# clean_caption_text() itself (the noise-tag regex) is unit-tested in
# test_captions.py, shared with subtitles.py; the tests here only cover
# _raw_to_segments' own integration of it (dropping now-empty segments).


def test_extract_video_id_handles_common_url_shapes():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s") == "dQw4w9WgXcQ"
    assert extract_video_id("https://youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_raw_to_segments_drops_pure_noise_segments():
    raw = [
        {"text": "hello world", "start": 0.0, "duration": 2.0},
        {"text": "[Music]", "start": 2.0, "duration": 1.0},
        {"text": "let's continue", "start": 3.0, "duration": 2.0},
    ]
    segments = _raw_to_segments(raw)
    assert [s.text for s in segments] == ["hello world", "let's continue"]


def test_raw_to_segments_keeps_partial_text_after_stripping_an_inline_tag():
    raw = [{"text": "before the [Music] chorus starts", "start": 0.0, "duration": 2.0}]
    segments = _raw_to_segments(raw)
    assert segments[0].text == "before the chorus starts"


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


class _FakeResponse:
    def __init__(self, ok=True, status_code=200, json_body=None, json_raises=False):
        self.ok = ok
        self.status_code = status_code
        self._json_body = json_body or {}
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("not valid JSON")
        return self._json_body


def test_fetch_title_success_has_no_warning(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_body={"title": "Real Title"}))
    title, warning = _fetch_title("abc123")
    assert title == "Real Title"
    assert warning is None


def test_fetch_title_network_error_falls_back_to_video_id_and_warns(monkeypatch):
    def raise_it(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "get", raise_it)
    title, warning = _fetch_title("abc123")
    assert title == "abc123"
    assert warning is not None
    assert "abc123" in warning


def test_fetch_title_non_ok_response_falls_back_to_video_id_and_warns(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(ok=False, status_code=404))
    title, warning = _fetch_title("abc123")
    assert title == "abc123"
    assert warning is not None
    assert "404" in warning


def test_fetch_title_unparseable_json_falls_back_to_video_id_and_warns(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_raises=True))
    title, warning = _fetch_title("abc123")
    assert title == "abc123"
    assert warning is not None


def test_load_youtube_surfaces_a_title_fetch_failure_as_a_transcript_warning(monkeypatch):
    monkeypatch.setattr(
        "cerebro.ingest.youtube._fetch_segments_raw",
        lambda video_id, languages: [{"text": "hello", "start": 0.0, "duration": 1.0}],
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(ok=False, status_code=500))

    transcript = load_youtube("https://youtu.be/dQw4w9WgXcQ")

    assert transcript.title == "dQw4w9WgXcQ"  # fell back to the video id
    assert len(transcript.warnings) == 1
    assert "500" in transcript.warnings[0]


def test_load_youtube_has_no_warnings_on_a_clean_run(monkeypatch):
    monkeypatch.setattr(
        "cerebro.ingest.youtube._fetch_segments_raw",
        lambda video_id, languages: [{"text": "hello", "start": 0.0, "duration": 1.0}],
    )
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(json_body={"title": "Real Title"}))

    transcript = load_youtube("https://youtu.be/dQw4w9WgXcQ")

    assert transcript.title == "Real Title"
    assert transcript.warnings == []
