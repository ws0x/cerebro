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
class Transcript:
    source: str
    title: str
    segments: list[Segment] = field(default_factory=list)
    language: str = "en"

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def duration(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
