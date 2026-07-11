"""Deterministic, offline structurer.

This produces a *structurally correct* map without any AI: it segments the
transcript into time-based topic chunks and derives titles/leaves from the
text. It is intentionally "dumb" — its job is to prove the pipeline and to be
the graceful fallback when no model is available. The LLM structurer replaces
the intelligence while reusing everything downstream (IR -> OPML/XMind).

One exception: author-numbered lists ("7 habits", "reason number one") are
detected via ``detect_enumeration`` (also pure/offline, zero LLM cost) and
used as the map's spine, mirroring what ``LLMStructurer`` does for the same
case — this is the one place the offline engine gets to be genuinely smart
instead of just structurally correct, since recovering an author's own
numbering needs no intelligence, only pattern matching.
"""

from __future__ import annotations

import re

from ..ir import MindMap, Node, NodeType
from ..transcript import Segment, Transcript
from .enumeration import EnumeratedSection, detect_enumeration

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Topic counts per level (capped by available content).
_TARGET_TOPICS = {"brief": 5, "full": 9, "expert": 12}
_LEAVES_PER_TOPIC = {"brief": 0, "full": 3, "expert": 4}

# A pre-#1 intro shorter than this isn't worth its own Overview branch --
# matches the LLM enumerated path's own threshold, for the same reason.
_INTRO_MIN_WORDS = 40


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(",;:. ") + "…"


def _titlecase_snippet(text: str, limit: int = 60) -> str:
    snippet = _truncate(text, limit)
    return snippet[:1].upper() + snippet[1:] if snippet else "Untitled"


def _chunk_segments(segments: list[Segment], target: int) -> list[list[Segment]]:
    if not segments:
        return []
    target = max(1, min(target, len(segments)))
    per = max(1, len(segments) // target)
    chunks = [segments[i : i + per] for i in range(0, len(segments), per)]
    # Fold any tiny trailing chunk back into the previous one.
    if len(chunks) > 1 and len(chunks[-1]) < max(1, per // 2):
        chunks[-2].extend(chunks[-1])
        chunks.pop()
    return chunks


def _has_real_timing(transcript: Transcript) -> bool:
    # A source with no real timing data (e.g. a PDF, where Segment.start is
    # always 0.0 -- a page number is not a timestamp) must not render a bogus
    # "[0:00]" on every node. A real video/subtitle transcript almost always
    # has at least one segment with a genuine nonzero start, which is how the
    # two cases are told apart -- unlike a bare `or None` check, this doesn't
    # also swallow a real video's true first-topic [0:00].
    return any(s.start or s.duration for s in transcript.segments)


def _section_text_spans(
    transcript: Transcript, sections: list[EnumeratedSection]
) -> tuple[str, list[str]]:
    """Slice the transcript into (intro_text, [section_text, ...]) by
    timestamp -- each section owns [its start, the next section's start)."""
    bounds = [s.start for s in sections] + [float("inf")]

    def text_between(lo: float, hi: float) -> str:
        return " ".join(
            s.text.strip() for s in transcript.segments if s.text.strip() and lo <= s.start < hi
        ).strip()

    intro = text_between(float("-inf"), bounds[0])
    section_texts = [text_between(bounds[i], bounds[i + 1]) for i in range(len(sections))]
    return intro, section_texts


def _add_leaves(topic: Node, text: str, n_leaves: int) -> None:
    if not (n_leaves and text):
        return
    sentences = _sentences(text)
    if len(sentences) <= 1:
        return
    # Evenly sample supporting points across the section/chunk.
    body = sentences[1:]
    step = max(1, len(body) // n_leaves)
    for sent in body[::step][:n_leaves]:
        topic.add(_truncate(sent, 90), type=NodeType.detail)


class HeuristicStructurer:
    def structure(self, transcript: Transcript, level: str = "full") -> MindMap:
        level = level if level in _TARGET_TOPICS else "full"
        root = Node(title=transcript.title or "Mind Map", type=NodeType.root)

        sections = detect_enumeration(transcript)
        if sections:
            self._fill_enumerated(root, transcript, sections, level)
        else:
            self._fill_flat(root, transcript, level)

        if not root.children:
            root.add("(no content extracted)", type=NodeType.detail)

        return MindMap(
            title=transcript.title or "Mind Map",
            root=root,
            source=transcript.source,
            level=level,
        )

    def _fill_flat(self, root: Node, transcript: Transcript, level: str) -> None:
        chunks = _chunk_segments(transcript.segments, _TARGET_TOPICS[level])
        n_leaves = _LEAVES_PER_TOPIC[level]
        real_timing = _has_real_timing(transcript)

        for chunk in chunks:
            chunk_text = " ".join(s.text.strip() for s in chunk).strip()
            if not chunk_text:
                continue
            sentences = _sentences(chunk_text)
            head = sentences[0] if sentences else chunk_text

            topic = root.add(
                _titlecase_snippet(head),
                type=NodeType.topic,
                timestamp=chunk[0].start if real_timing else None,
                note=_truncate(chunk_text, 500),
            )
            _add_leaves(topic, chunk_text, n_leaves)

    def _fill_enumerated(
        self, root: Node, transcript: Transcript, sections: list[EnumeratedSection], level: str
    ) -> None:
        n_leaves = _LEAVES_PER_TOPIC[level]
        real_timing = _has_real_timing(transcript)
        intro, section_texts = _section_text_spans(transcript, sections)

        if len(intro.split()) >= _INTRO_MIN_WORDS:
            root.add(
                "Overview",
                type=NodeType.topic,
                timestamp=sections[0].start if real_timing else None,
                note=_truncate(intro, 500),
            )

        for section, text in zip(sections, section_texts):
            heading = section.heading or f"Part {section.number}"
            topic = root.add(
                f"{section.number}. {heading}",
                type=NodeType.topic,
                timestamp=section.start if real_timing else None,
                note=_truncate(text, 500) if text else None,
            )
            _add_leaves(topic, text, n_leaves)
