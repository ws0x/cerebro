"""The ``Transcript`` contract — the normalized output of every ingest source.

YouTube, local subtitle tracks, Whisper, etc. all produce this same shape, so
the structurer never needs to know where the words came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    text: str
    start: float = 0.0  # seconds
    duration: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class OutlineEntry:
    """One heading in a source document's real, pre-existing structure (e.g. a
    PDF's bookmarks/TOC, or detected headings). ``page`` is 0-indexed."""

    level: int
    title: str
    page: int


@dataclass
class Transcript:
    source: str
    title: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    # Non-empty only for sources with real, pre-existing structure (PDFs with a
    # TOC/detected headings). When present, the structurer builds the map from
    # this skeleton directly instead of inventing hierarchy from flat text.
    outline: list[OutlineEntry] = field(default_factory=list)
    # Non-fatal, user-facing notices about the ingest itself (e.g. a subtitle
    # file that wasn't valid UTF-8) -- surfaced by the CLI, never silent.
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
