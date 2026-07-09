"""Deterministic, offline structurer.

This produces a *structurally correct* map without any AI: it segments the
transcript into time-based topic chunks and derives titles/leaves from the
text. It is intentionally "dumb" — its job is to prove the pipeline and to be
the graceful fallback when no model is available. The LLM structurer replaces
the intelligence while reusing everything downstream (IR -> OPML/XMind).
"""

from __future__ import annotations

import re

from ..ir import MindMap, Node, NodeType
from ..transcript import Segment, Transcript

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Topic counts per level (capped by available content).
_TARGET_TOPICS = {"brief": 5, "full": 9, "expert": 12}
_LEAVES_PER_TOPIC = {"brief": 0, "full": 3, "expert": 4}


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


class HeuristicStructurer:
    def structure(self, transcript: Transcript, level: str = "full") -> MindMap:
        level = level if level in _TARGET_TOPICS else "full"
        root = Node(title=transcript.title or "Mind Map", type=NodeType.root)

        chunks = _chunk_segments(transcript.segments, _TARGET_TOPICS[level])
        n_leaves = _LEAVES_PER_TOPIC[level]

        for chunk in chunks:
            chunk_text = " ".join(s.text.strip() for s in chunk).strip()
            if not chunk_text:
                continue
            sentences = _sentences(chunk_text)
            head = sentences[0] if sentences else chunk_text

            topic = root.add(
                _titlecase_snippet(head),
                type=NodeType.topic,
                timestamp=chunk[0].start or None,
                note=_truncate(chunk_text, 500),
            )

            if n_leaves and len(sentences) > 1:
                # Evenly sample supporting points across the chunk.
                body = sentences[1:]
                step = max(1, len(body) // n_leaves)
                for sent in body[::step][:n_leaves]:
                    topic.add(_truncate(sent, 90), type=NodeType.detail)

        if not root.children:
            root.add("(no content extracted)", type=NodeType.detail)

        return MindMap(
            title=transcript.title or "Mind Map",
            root=root,
            source=transcript.source,
            level=level,
        )
