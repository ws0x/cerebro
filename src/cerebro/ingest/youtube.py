"""YouTube ingest via captions (fast path, no download).

Uses ``youtube-transcript-api`` for the transcript and YouTube's public oEmbed
endpoint for the title. If captions are unavailable this raises; a future
version falls back to yt-dlp audio download + Whisper.

The library changed its API between 0.6.x and 1.x, so we adapt to whichever is
installed.
"""

from __future__ import annotations

import re

import requests

from ..transcript import Segment, Transcript

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


def _fetch_segments(video_id: str, languages: list[str]) -> list[Segment]:
    from youtube_transcript_api import YouTubeTranscriptApi

    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        # Classic (0.6.x) static API.
        raw = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    else:
        # New (1.x) instance API returns a FetchedTranscript.
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)

    return [
        Segment(
            text=item["text"],
            start=float(item.get("start", 0.0)),
            duration=float(item.get("duration", 0.0)),
        )
        for item in raw
        if item.get("text", "").strip()
    ]


def load_youtube(url: str, languages: list[str] | None = None) -> Transcript:
    languages = languages or ["en", "en-US", "en-GB"]
    video_id = extract_video_id(url)
    segments = _fetch_segments(video_id, languages)
    title = _fetch_title(video_id)
    return Transcript(source=url, title=title, segments=segments, language=languages[0])
