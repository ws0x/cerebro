"""Local subtitle/text ingest: .srt, .vtt, .txt.

Deterministic and offline — this is what lets us test the whole pipeline
without depending on the network or YouTube captions.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..transcript import Segment, Transcript

# 00:01:02,500  or  00:01:02.500  or  01:02.500
_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_RE = re.compile(r"(.+?)\s*-->\s*(.+)")


def _parse_timestamp(raw: str) -> float:
    m = _TS_RE.search(raw)
    if not m:
        return 0.0
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    millis = int(m.group(4).ljust(3, "0"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _parse_cue_blocks(text: str) -> list[Segment]:
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Skip WEBVTT header / NOTE / STYLE blocks.
        if lines[0].upper().startswith(("WEBVTT", "NOTE", "STYLE")):
            continue
        # A leading numeric index (SRT) is optional.
        if lines[0].strip().isdigit():
            lines = lines[1:]
        if not lines:
            continue
        cue = _CUE_RE.search(lines[0])
        if not cue:
            continue
        start = _parse_timestamp(cue.group(1))
        end = _parse_timestamp(cue.group(2))
        content = " ".join(lines[1:]).strip()
        # Strip simple VTT inline tags like <c> or <00:00:00.000>.
        content = re.sub(r"<[^>]+>", "", content).strip()
        if content:
            segments.append(Segment(text=content, start=start, duration=max(0.0, end - start)))
    return segments


def load_subtitle_file(path: Path) -> Transcript:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()

    if path.suffix.lower() == ".txt" and "-->" not in raw:
        # Plain text: one segment per non-empty line.
        segments = [Segment(text=ln.strip()) for ln in raw.splitlines() if ln.strip()]
    else:
        segments = _parse_cue_blocks(raw)

    return Transcript(source=str(path), title=title, segments=segments)
