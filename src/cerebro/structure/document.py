"""Outline-aware PDF structurer: TOC/heading skeleton -> MindMap, optionally
enriched by the LLM.

Unlike video (a flat transcript with no real structure -- an LLM must *invent*
a hierarchy), a PDF with a TOC/detected headings already has real structure.
Same judgment call ``foldermap.py`` makes for folder trees: structure is
*known*, so it's used directly; AI is optional *enrichment* of each section's
content, never structure *discovery*. ``build_outline_skeleton`` builds that
skeleton with zero AI (the free/offline default, same posture as
``HeuristicStructurer``); ``build_outline_map`` layers LLM section-summary
extraction and (at expert level) cross-section relationship detection on top,
reusing the *existing* MAP prompt and ``link_relationships`` from
``structure/llm.py`` completely unchanged -- no new prompt, no cache-version
bump, no new relationship-detection logic.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..cache import Cache
from ..ir import MindMap, Node, NodeType
from ..llm.base import LLMError, LLMProvider
from ..prompts import MAP_SYSTEM, PROMPT_VERSION
from ..transcript import Transcript
from .llm import link_relationships

_NOTE_LIMIT = 500  # matches structure/heuristic.py's own truncate limit
_MAX_WORDS = {"brief": 2000, "full": 1400, "expert": 1200}  # matches structure/llm.py's budget


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(",;:. ") + "…"


def _coerce_type(value) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.topic


def _section_text(transcript: Transcript, start_page: int, end_page: int) -> str:
    pages = transcript.segments[start_page:end_page]
    return " ".join(s.text.strip() for s in pages if s.text.strip())


def _build_skeleton(transcript: Transcript) -> tuple[Node, list[tuple[Node, int, str]]]:
    """Nest ``transcript.outline`` into a real ``Node`` tree via a stack keyed
    by heading level. Returns ``(root, leaf_sections)`` where ``leaf_sections``
    pairs each *leaf* node (no children -- non-leaf nodes are represented by
    their children, not a text snippet) with its 1-indexed page number and its
    own full, untruncated page-range text: from its own page up to (not
    including) the next entry's page in flat document order, regardless of
    nesting -- a heading's text genuinely ends wherever the next heading in
    the document begins, however deeply either is nested."""
    outline = transcript.outline
    root = Node(title=transcript.title or "Document", type=NodeType.root)
    if not outline:
        root.add("(no structure detected)", type=NodeType.detail)
        return root, []

    total_pages = len(transcript.segments)
    nodes: list[Node] = []
    stack: list[tuple[int, Node]] = [(0, root)]  # (level, node); root is level 0

    for entry in outline:
        node = Node(title=entry.title, type=NodeType.topic)
        while stack and stack[-1][0] >= entry.level:
            stack.pop()
        parent = stack[-1][1] if stack else root
        parent.children.append(node)
        stack.append((entry.level, node))
        nodes.append(node)

    leaf_sections: list[tuple[Node, int, str]] = []
    for i, (entry, node) in enumerate(zip(outline, nodes)):
        if node.children:
            continue
        end_page = outline[i + 1].page if i + 1 < len(outline) else total_pages
        text = _section_text(transcript, entry.page, max(end_page, entry.page + 1))
        leaf_sections.append((node, entry.page + 1, text))
    return root, leaf_sections


def _apply_fallback_notes(leaf_sections: list[tuple[Node, int, str]]) -> None:
    for node, page, text in leaf_sections:
        if text:
            node.note = f"(p. {page}) {_truncate(text, _NOTE_LIMIT)}"


def build_outline_skeleton(transcript: Transcript) -> MindMap:
    """Deterministic, zero-AI skeleton: each leaf section's note is a
    truncated snippet of its own page-range text."""
    root, leaf_sections = _build_skeleton(transcript)
    _apply_fallback_notes(leaf_sections)
    return MindMap(title=root.title, root=root, source=transcript.source, level="full")


# -- AI section enrichment --------------------------------------------------


def _call(provider: LLMProvider, cache: Cache, task: str, system: str, user: str, *key_parts) -> dict:
    key = Cache.key(provider.name, provider.model, PROMPT_VERSION, task, *key_parts)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = provider.complete_json(system, user)
    cache.set(key, result)
    return result


def _enrich_one(node: Node, text: str, provider: LLMProvider, cache: Cache, level: str) -> None:
    words = text.split()
    cap = _MAX_WORDS.get(level, _MAX_WORDS["full"])
    if len(words) > cap:  # honest tradeoff: an overlong section is truncated,
        text = " ".join(words[:cap])  # not sub-chunked -- keeps this a single call, like a MAP chunk.
    result = _call(provider, cache, "map", MAP_SYSTEM, text, level, text)
    summary = str(result.get("summary", "")).strip()
    if summary:
        node.note = summary
    for point in result.get("points", []) or []:
        if isinstance(point, dict):
            title = str(point.get("title", "")).strip()
            if title:
                node.add(title, type=_coerce_type(point.get("type")))


def _enrich_leaves(
    leaf_sections: list[tuple[Node, int, str]],
    provider: LLMProvider,
    cache: Cache,
    level: str,
    on_event: Callable[..., None],
    max_workers: int = 6,
) -> None:
    todo = [(node, text) for node, _page, text in leaf_sections if text.strip()]
    if not todo:
        return
    on_event("map_start", total=len(todo))
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_enrich_one, node, text, provider, cache, level): node for node, text in todo
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except LLMError as exc:
                on_event("map_error", error=str(exc))
            done += 1
            on_event("map_progress", done=done, total=len(todo))


def build_outline_map(
    transcript: Transcript,
    provider: LLMProvider | None = None,
    cache: Cache | None = None,
    level: str = "full",
    on_event: Callable[..., None] | None = None,
    relationship_limit: int = 8,
) -> MindMap:
    """Build the deterministic skeleton, then -- given a provider, and above
    brief level (brief stays free even with a key configured, matching its
    "shallow, minimal-nesting" spirit) -- enrich every leaf section's note
    into an LLM-extracted summary plus child key points, and at expert level
    detect cross-section relationships via the existing, unchanged
    ``link_relationships``. A failed leaf enrichment call is skipped, not
    fatal, leaving that leaf's deterministic skeleton note in place -- same
    resilience as the video pipeline's MAP stage."""
    on_event = on_event or (lambda *a, **k: None)

    if provider is None or level == "brief":
        mm = build_outline_skeleton(transcript)
        on_event("done", nodes=mm.node_count(), relationships=0)
        return mm

    cache = cache or Cache(enabled=False)
    root, leaf_sections = _build_skeleton(transcript)
    mm = MindMap(title=root.title, root=root, source=transcript.source, level=level)

    # Deterministic fallback note first -- enrichment overwrites it on
    # success; a leaf whose enrichment call fails keeps this instead of
    # being left blank.
    _apply_fallback_notes(leaf_sections)
    _enrich_leaves(leaf_sections, provider, cache, level, on_event)

    if level == "expert":
        link_relationships(
            mm, provider, cache, on_event=on_event, relationship_limit=relationship_limit
        )

    on_event("done", nodes=mm.node_count(), relationships=len(mm.relationships))
    return mm
