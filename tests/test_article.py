"""Real trafilatura extraction against synthetic local HTML -- only the
network fetch (trafilatura.fetch_url) is mocked, same "fake the I/O
boundary, not the logic" pattern test_video.py uses for ffmpeg fixtures.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cerebro.ingest import load_transcript, looks_like_web_url
from cerebro.ingest.article import ArticleIngestError, load_article

_STRUCTURED_HTML = """
<html><head><title>Understanding Distributed Caching</title></head>
<body>
<nav>Home | About | Contact</nav>
<div class="ad">Buy now! Special offer!</div>
<article>
<h1>Understanding Distributed Caching</h1>
<p>A cache is a smaller, faster storage layer that keeps a copy of frequently accessed data
so future requests can be served faster than hitting the underlying slower storage.</p>
<h2>Cache Eviction Policies</h2>
<p>When a cache is full, an eviction policy decides which entry to remove. LRU evicts the
least recently accessed entry, on the assumption recently used data will be used again soon.</p>
<h2>Distributed Caches</h2>
<p>A distributed cache spreads data across many servers, increasing total capacity and letting
the cache survive a single server failing, unlike a single-machine cache.</p>
</article>
<footer>Copyright 2026</footer>
</body></html>
"""

_FLAT_HTML = """
<html><head><title>A Short Note</title></head>
<body>
<article>
<p>This is just a short flowing note with no internal heading structure at all, the kind
of thing you'd find on a simple blog with no sections, just one continuous train of thought.</p>
</article>
</body></html>
"""


def test_looks_like_web_url():
    assert looks_like_web_url("https://example.com/article")
    assert looks_like_web_url("http://example.com/article")
    assert not looks_like_web_url("not a url")
    assert not looks_like_web_url("C:\\some\\path.pdf")


def test_load_article_extracts_title_and_real_heading_structure():
    with patch("trafilatura.fetch_url", return_value=_STRUCTURED_HTML):
        transcript = load_article("https://example.com/caching")
    assert transcript.title == "Understanding Distributed Caching"
    assert [e.title for e in transcript.outline] == ["Cache Eviction Policies", "Distributed Caches"]
    assert transcript.source == "https://example.com/caching"


def test_load_article_outline_pages_point_at_correct_segments():
    with patch("trafilatura.fetch_url", return_value=_STRUCTURED_HTML):
        transcript = load_article("https://example.com/caching")
    for entry in transcript.outline:
        assert 0 <= entry.page < len(transcript.segments)
        assert entry.title in transcript.segments[entry.page].text


def test_load_article_strips_nav_ads_and_footer_boilerplate():
    with patch("trafilatura.fetch_url", return_value=_STRUCTURED_HTML):
        transcript = load_article("https://example.com/caching")
    assert "Buy now" not in transcript.full_text
    assert "Copyright 2026" not in transcript.full_text
    assert "Home | About" not in transcript.full_text


def test_load_article_with_no_real_headings_has_empty_outline():
    with patch("trafilatura.fetch_url", return_value=_FLAT_HTML):
        transcript = load_article("https://example.com/note")
    assert transcript.outline == []
    assert "short flowing note" in transcript.full_text


def test_load_article_raises_on_fetch_failure():
    with patch("trafilatura.fetch_url", return_value=None):
        with pytest.raises(ArticleIngestError, match="Could not fetch"):
            load_article("https://example.com/gone")


def test_load_article_raises_on_network_exception():
    with patch("trafilatura.fetch_url", side_effect=ConnectionError("dns failed")):
        with pytest.raises(ArticleIngestError, match="Could not fetch"):
            load_article("https://example.com/unreachable")


def test_load_article_raises_when_no_extractable_content():
    with patch("trafilatura.fetch_url", return_value="<html><body></body></html>"):
        with pytest.raises(ArticleIngestError, match="Could not extract"):
            load_article("https://example.com/empty")


def test_dispatch_routes_web_urls_to_load_article():
    with patch("trafilatura.fetch_url", return_value=_STRUCTURED_HTML):
        transcript = load_transcript("https://example.com/caching")
    assert transcript.title == "Understanding Distributed Caching"


def test_dispatch_still_routes_youtube_urls_to_youtube_not_article():
    # looks_like_web_url would also match a youtube.com URL -- youtube must win.
    with patch("cerebro.ingest.youtube.load_youtube") as mock_yt:
        mock_yt.return_value = "sentinel"
        result = load_transcript("https://youtube.com/watch?v=abc123")
    assert result == "sentinel"
    mock_yt.assert_called_once()
