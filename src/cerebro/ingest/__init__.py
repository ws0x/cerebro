"""Ingest layer: any source -> a normalized ``Transcript``.

The public entry point is :func:`load_transcript`, which dispatches on the
source string (YouTube URL, local subtitle/text file, ...).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..transcript import Transcript

if TYPE_CHECKING:
    from ..cache import Cache

_YOUTUBE_RE = re.compile(r"(youtube\.com|youtu\.be)", re.IGNORECASE)
_SUBTITLE_EXTS = {".vtt", ".srt", ".txt"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}


def looks_like_youtube(source: str) -> bool:
    return bool(_YOUTUBE_RE.search(source))


def load_transcript(source: str, cache: "Cache | None" = None) -> Transcript:
    """Load a transcript from a URL or local file path.

    ``cache`` is only used by the local-video path (Whisper transcription is
    the one expensive step in ingest); other sources ignore it.
    """
    if looks_like_youtube(source):
        from .youtube import load_youtube

        return load_youtube(source)

    path = Path(source)
    if path.exists():
        ext = path.suffix.lower()
        if ext in _SUBTITLE_EXTS:
            from .subtitles import load_subtitle_file

            return load_subtitle_file(path)
        if ext in _VIDEO_EXTS:
            from .video import load_video

            return load_video(path, cache=cache)
        raise ValueError(f"Unsupported file type: {ext} ({path})")

    raise ValueError(
        f"Could not interpret source as a YouTube URL or existing file: {source!r}"
    )
