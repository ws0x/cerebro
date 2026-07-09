"""Course-folder discovery: a directory of lesson videos -> ordered sources.

Videos with a sidecar subtitle file (``lesson1.mp4`` + ``lesson1.srt``) use it
directly — fast, free. Videos with no sidecar are still included (flagged
``needs_transcription``) since :mod:`cerebro.ingest.video` can extract an
embedded subtitle track or fall back to Whisper; it's just slower.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SUBTITLE_EXTS = {".vtt", ".srt", ".txt"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(stem: str) -> list:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NUM_RE.split(stem)]


def _prettify(stem: str) -> str:
    return stem.replace("_", " ").replace("-", " ").strip().title() or stem


@dataclass
class CourseFile:
    path: Path
    title: str
    needs_transcription: bool = False


def discover_course_sources(folder: Path) -> list[CourseFile]:
    """Return one lesson source per video/subtitle file found in ``folder``'s
    immediate contents, in natural sort order (so "Lesson 2" precedes "Lesson 10")."""
    folder = Path(folder)
    entries = [f for f in folder.iterdir() if f.is_file()]
    subtitle_by_stem = {f.stem: f for f in entries if f.suffix.lower() in _SUBTITLE_EXTS}
    video_files = [f for f in entries if f.suffix.lower() in _VIDEO_EXTS]

    sources: list[CourseFile] = []
    used_stems: set[str] = set()

    for video in video_files:
        sub = subtitle_by_stem.get(video.stem)
        if sub:
            sources.append(CourseFile(path=sub, title=_prettify(video.stem)))
            used_stems.add(video.stem)
        else:
            sources.append(CourseFile(path=video, title=_prettify(video.stem), needs_transcription=True))

    for stem, sub in subtitle_by_stem.items():
        if stem not in used_stems:
            sources.append(CourseFile(path=sub, title=_prettify(stem)))

    sources.sort(key=lambda c: _natural_key(c.path.stem))
    return sources
