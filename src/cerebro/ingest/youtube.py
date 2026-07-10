"""YouTube ingest via captions (fast path, no download).

Uses ``youtube-transcript-api`` for the transcript and YouTube's public oEmbed
endpoint for the title. If captions are unavailable this raises; a future
version falls back to yt-dlp audio download + Whisper.

The library changed its API between 0.6.x and 1.x, so we adapt to whichever is
installed.

Captions are cached (by video id + requested languages) — once a video is
published its captions essentially never change, so refetching them from
YouTube on every single run of the exact same video is pure waste. Only the
raw caption *segments* are cached; the title is refetched each time (cheap,
one HTTP call, and titles occasionally do change).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import requests

from ..transcript import Segment, Transcript

if TYPE_CHECKING:
    from ..cache import Cache

_ID_PATTERNS = [
    re.compile(r"(?:v=|/shorts/|/embed/|youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),
]


def extract_video_id(url: str) -> str:
    for pat in _ID_PATTERNS:
        m = pat.search(url.strip())
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract a YouTube video id from: {url!r}")


def _fetch_title(video_id: str) -> str:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("title", video_id)
    except Exception:
        pass
    return video_id


def _fetch_segments_raw(video_id: str, languages: list[str]) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        # Classic (0.6.x) static API.
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    else:
        # New (1.x) instance API returns a FetchedTranscript.
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    return [dict(item) for item in raw]


def _raw_to_segments(raw: list[dict]) -> list[Segment]:
    return [
        Segment(
            text=item["text"],
            start=float(item.get("start", 0.0)),
            duration=float(item.get("duration", 0.0)),
        )
        for item in raw
        if item.get("text", "").strip()
    ]


def _fetch_segments(video_id: str, languages: list[str], cache: "Cache | None" = None) -> list[Segment]:
    if cache is None or not cache.enabled:
        return _raw_to_segments(_fetch_segments_raw(video_id, languages))

    from ..cache import Cache as _Cache

    key = _Cache.key("youtube_captions", video_id, tuple(languages))
    raw = cache.get(key)
    if raw is None:
        raw = _fetch_segments_raw(video_id, languages)
        cache.set(key, raw)
    return _raw_to_segments(raw)


def load_youtube(url: str, languages: list[str] | None = None, cache: "Cache | None" = None) -> Transcript:
    languages = languages or ["en", "en-US", "en-GB"]
    video_id = extract_video_id(url)
    segments = _fetch_segments(video_id, languages, cache=cache)
    title = _fetch_title(video_id)
    return Transcript(source=url, title=title, segments=segments, language=languages[0])
