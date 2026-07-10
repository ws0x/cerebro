"""Shared cleanup for auto-generated caption/subtitle text.

Used by both youtube.py (captions fetched live) and subtitles.py (a local
.srt/.vtt file, which is just as often itself an export of auto-generated
captions -- downloaded from YouTube or produced by some other
auto-captioning tool) and, transitively, video.py's embedded-subtitle path,
which reuses subtitles.py's own cue parser.
"""

from __future__ import annotations

import re

# Auto-generated captions mark non-speech audio events this way --
# "[Music]", "(Applause)", "[laughter]" -- and left in, these leak straight
# into node titles verbatim (most visibly through the heuristic engine,
# which titles a chunk from its own first sentence with no LLM in the loop
# to recognize and skip them). Matched case-insensitively, bracket style
# either way, and only when the *entire* bracketed content is one of these
# known non-speech tags -- never touches an actual spoken sentence that
# happens to be bracketed, or the word "music" used normally in speech.
_NOISE_TAG_RE = re.compile(
    r"[\[(]\s*(?:music|applause|laughter|silence|inaudible|crosstalk|background noise)\s*[\])]",
    re.IGNORECASE,
)


def clean_caption_text(text: str) -> str:
    return re.sub(r"\s+", " ", _NOISE_TAG_RE.sub("", text)).strip()
