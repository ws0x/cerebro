"""Local subtitle/text ingest: .srt, .vtt, .txt.

Deterministic and offline — this is what lets us test the whole pipeline
without depending on the network or YouTube captions.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..transcript import Segment, Transcript
from ._captions import clean_caption_text

# 00:01:02,500  or  00:01:02.500  or  01:02.500
_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_RE = re.compile(r"(.+?)\s*-->\s*(.+)")

# Common legacy Windows encoding tried before giving up -- cp1252 covers the
# vast majority of non-UTF-8 .srt/.vtt exports seen in practice (e.g. Windows
# editors saving "ANSI"). latin-1 always succeeds (every byte 0-255 maps to a
# codepoint), so it's the deterministic last resort rather than lossy
# replacement -- no third-party encoding-detection dependency needed for
# this narrow, well-known case.
_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


def _read_text(path: Path) -> tuple[str, str | None]:
    """Decode a subtitle/text file, returning (text, warning_or_None).

    Previously this always decoded as UTF-8 with errors="replace", which
    silently turned any non-UTF-8 byte into U+FFFD with no signal to the
    caller -- corrupting text that then fed straight into the LLM prompt.
    Now a decode failure is either recovered via a known-common fallback
    encoding, or explicitly reported.
    """
    raw_bytes = path.read_bytes()
    try:
        return raw_bytes.decode("utf-8"), None
    except UnicodeDecodeError:
        pass

    for encoding in _FALLBACK_ENCODINGS:
        try:
            text = raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text, (
            f"{path.name} is not valid UTF-8 -- decoded as {encoding} instead. "
            "Re-save it as UTF-8 if any text looks wrong."
        )

    # Unreachable in practice (latin-1 above never raises), kept as a final
    # safety net rather than letting a decode error crash ingestion outright.
    text = raw_bytes.decode("utf-8", errors="replace")
    return text, (
        f"{path.name} contains bytes that could not be decoded in any supported "
        "encoding; unreadable characters were replaced with �. Re-save it as UTF-8."
    )


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
        # A local file is just as often an export of auto-generated
        # captions (downloaded from YouTube, or another auto-captioning
        # tool) as one fetched live -- same "[Music]"-style noise tags.
        content = clean_caption_text(content)
        if content:
            segments.append(Segment(text=content, start=start, duration=max(0.0, end - start)))
    return segments


def load_subtitle_file(path: Path) -> Transcript:
    path = Path(path)
    raw, warning = _read_text(path)
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()

    if path.suffix.lower() == ".txt" and "-->" not in raw:
        # Plain text: one segment per non-empty line.
        segments = [
            Segment(text=clean_caption_text(ln))
            for ln in raw.splitlines()
            if clean_caption_text(ln)
        ]
    else:
        segments = _parse_cue_blocks(raw)

    warnings = [warning] if warning else []
    return Transcript(source=str(path), title=title, segments=segments, warnings=warnings)
