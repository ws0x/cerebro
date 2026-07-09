"""Topic-boundary-aware chunking for the MAP stage.

Splitting a transcript purely by word count (the original approach) can slice
a chunk mid-topic, forcing the MAP call to summarize half of one idea and half
of another. Real semantic segmentation (dense embeddings + similarity) would
need a heavy ML dependency this project deliberately keeps out of the base
install (see the Whisper extra). Instead this uses a classic TextTiling-style
approach: lexical cohesion between adjacent windows of segments, via plain
word-overlap (Jaccard similarity) rather than embeddings. It won't catch a
topic shift expressed through synonyms alone, but it reliably catches a
genuine vocabulary shift — which is exactly what "wrong place to cut" looks
like from outside. Fully offline, no network, no model download.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..transcript import Segment, Transcript

_STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on for with and or but "
    "this that these those it its as at by from into about so if then than "
    "we you they he she i my your our their not no do does did have has had "
    "will would can could should may might just also very really".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def cohesion_scores(segments: list[Segment], window: int = 3) -> dict[int, float]:
    """Lexical cohesion at each candidate boundary (the gap just before index i).

    A high score means the ``window`` segments on either side share a lot of
    vocabulary (probably the same topic, not a boundary); a low score means a
    vocabulary shift (probably a topic change). Returns ``{segment_index:
    score}`` for every index where a full window fits on both sides — too
    short a transcript yields an empty dict, and callers should treat that as
    "no boundary information available".
    """
    n = len(segments)
    if n < window * 2:
        return {}
    token_sets = [_tokens(s.text) for s in segments]
    scores: dict[int, float] = {}
    for i in range(window, n - window + 1):
        left: set[str] = set()
        for ts in token_sets[i - window : i]:
            left |= ts
        right: set[str] = set()
        for ts in token_sets[i : i + window]:
            right |= ts
        union = left | right
        scores[i] = len(left & right) / len(union) if union else 1.0
    return scores


# However cohesive a transcript's bottom quartile is relative to itself, a
# point only counts as a real topic boundary if the two sides genuinely share
# less than half their vocabulary — otherwise a single-topic transcript with
# uniformly high (but slightly noisy) cohesion would still produce a "lowest
# quartile" point and get spurious early cuts.
_MAX_BOUNDARY_SCORE = 0.5


def _boundary_threshold(scores: dict[int, float]) -> float:
    """The score below which a candidate point counts as a real boundary.

    Self-calibrated per transcript (bottom quartile of this transcript's own
    score distribution, capped at ``_MAX_BOUNDARY_SCORE``) rather than a fixed
    magic constant alone, since raw Jaccard values vary a lot with vocabulary
    richness and segment length.
    """
    if not scores:
        return 0.0
    values = sorted(scores.values())
    idx = max(0, (len(values) + 3) // 4 - 1)
    return min(values[idx], _MAX_BOUNDARY_SCORE)


@dataclass
class Chunk:
    text: str
    start: float


def chunk_transcript(transcript: Transcript, max_words: int, min_fraction: float = 0.4) -> list[Chunk]:
    """Split into MAP-stage chunks, preferring a detected topic boundary once
    at least ``min_fraction * max_words`` words have accumulated, and always
    cutting at ``max_words`` regardless (a hard ceiling, so one very cohesive
    stretch of talk never becomes one unbounded chunk)."""
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []

    scores = cohesion_scores(segments)
    threshold = _boundary_threshold(scores)
    min_words = int(max_words * min_fraction)

    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_words = 0
    start: float | None = None

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if start is None:
            start = seg.start
        cur.append(text)
        cur_words += len(text.split())

        at_boundary = scores.get(i + 1, 1.0) <= threshold
        hit_ceiling = cur_words >= max_words
        if hit_ceiling or (cur_words >= min_words and at_boundary):
            chunks.append(Chunk(text=" ".join(cur), start=start or 0.0))
            cur, cur_words, start = [], 0, None

    if cur:
        chunks.append(Chunk(text=" ".join(cur), start=start or 0.0))
    return chunks
