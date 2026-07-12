"""Tests for playlist listing and its error handling.

yt-dlp itself is mocked out (no network calls) by monkeypatching
`yt_dlp.YoutubeDL` — `load_playlist` imports it lazily inside the function
body, but monkeypatching the attribute on the real `yt_dlp` module works
regardless of where the import happens.
"""

from __future__ import annotations

import pytest
from yt_dlp.utils import DownloadError

from cerebro.ingest.playlist import PlaylistIngestError, is_playlist_url, load_playlist


class _FakeYoutubeDL:
    def __init__(self, info=None, error=None):
        self._info = info
        self._error = error

    def __call__(self, opts):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        if self._error is not None:
            raise self._error
        return self._info


def _install_fake(monkeypatch, info=None, error=None):
    monkeypatch.setattr("yt_dlp.YoutubeDL", _FakeYoutubeDL(info=info, error=error))


def test_is_playlist_url_detects_list_param():
    assert is_playlist_url("https://www.youtube.com/watch?v=abc&list=PL123")
    assert is_playlist_url("https://www.youtube.com/playlist?list=PL123")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc")


def test_load_playlist_returns_title_and_items(monkeypatch):
    _install_fake(
        monkeypatch,
        info={
            "title": "My Course",
            "entries": [
                {"id": "vid1", "title": "Lesson 1"},
                {"id": "vid2", "title": "Lesson 2"},
            ],
        },
    )
    result = load_playlist("https://www.youtube.com/playlist?list=PL123")
    assert result.title == "My Course"
    assert result.items == [
        ("Lesson 1", "https://www.youtube.com/watch?v=vid1"),
        ("Lesson 2", "https://www.youtube.com/watch?v=vid2"),
    ]


def test_load_playlist_skips_empty_entries(monkeypatch):
    _install_fake(
        monkeypatch,
        info={"title": "My Course", "entries": [None, {"id": "vid1", "title": "Lesson 1"}]},
    )
    result = load_playlist("https://www.youtube.com/playlist?list=PL123")
    assert len(result.items) == 1


def test_load_playlist_falls_back_to_video_id_as_title(monkeypatch):
    _install_fake(monkeypatch, info={"title": "My Course", "entries": [{"id": "vid1"}]})
    result = load_playlist("https://www.youtube.com/playlist?list=PL123")
    assert result.items == [("vid1", "https://www.youtube.com/watch?v=vid1")]


def test_load_playlist_falls_back_to_default_title_when_missing(monkeypatch):
    _install_fake(monkeypatch, info={"entries": []})
    result = load_playlist("https://www.youtube.com/playlist?list=PL123")
    assert result.title == "Playlist"
    assert result.items == []


def test_load_playlist_wraps_download_error(monkeypatch):
    _install_fake(monkeypatch, error=DownloadError("Private video"))
    with pytest.raises(PlaylistIngestError, match="Private video"):
        load_playlist("https://www.youtube.com/playlist?list=PL123")


def test_load_playlist_raises_on_none_info(monkeypatch):
    _install_fake(monkeypatch, info=None)
    with pytest.raises(PlaylistIngestError, match="no data returned"):
        load_playlist("https://www.youtube.com/playlist?list=PL123")
