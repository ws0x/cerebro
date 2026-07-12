"""Additive synthesis for structured sources.

A PDF's TOC or an author-numbered video keeps its own spine verbatim (the
faithful choice), but that leaves the map with no cross-cutting layer: the
source's actual thesis can end up buried as a leaf under some late section,
and connections that span sections are never surfaced. A live run of an
intro-to-neural-nets PDF put "Generalization is the Real Goal" -- the whole
document's point -- three levels down under section 3.3.

This appends ONE "Key Takeaways" branch synthesizing across sections. It is
strictly additive: the author's structure is never edited, only a new branch
is added at the end. Skipped at brief (an advance organizer is already the
gist) and on maps too thin to have anything cross-cutting to say.
"""

from __future__ import annotations

import json

from ..cache import Cache
from ..ir import MindMap, Node, NodeType
from ..llm.base import LLMError, LLMProvider
from ..prompts import PROMPT_VERSION, SYNTHESIS_SYSTEM

_MIN_NODES = 3  # below this there's nothing cross-cutting to synthesize


def _coerce_type(value) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.insight


def add_synthesis(
    mm: MindMap,
    provider: LLMProvider,
    cache: Cache,
    level: str,
    on_event=None,
) -> int:
    """Append a 'Key Takeaways' branch synthesizing across the map's sections.
    Additive only -- the existing spine is never modified. Returns how many
    takeaways were added (0 if skipped, thin, empty, or the call failed)."""
    on_event = on_event or (lambda *a, **k: None)
    if level == "brief":
        return 0
    nodes = [n for n in mm.root.walk() if n.type != NodeType.root]
    if len(nodes) < _MIN_NODES:
        return 0

    listing = [
        {"id": i, "title": n.title, **({"note": n.note} if n.note else {})}
        for i, n in enumerate(nodes)
    ]
    user = json.dumps({"title": mm.title, "map": listing}, ensure_ascii=False)
    key = Cache.key(provider.name, provider.model, PROMPT_VERSION, "synthesis", mm.level, user)
    result = cache.get(key)
    if result is None:
        on_event("synthesis_start")
        try:
            result = provider.complete_json(SYNTHESIS_SYSTEM, user)
        except LLMError as exc:
            on_event("synthesis_error", error=str(exc))
            return 0
        cache.set(key, result)

    takeaways = [
        t for t in (result.get("takeaways", []) or [])
        if isinstance(t, dict) and str(t.get("title", "")).strip()
    ]
    if not takeaways:
        return 0

    branch = Node(title="Key Takeaways", type=NodeType.insight)
    for t in takeaways:
        note = str(t.get("note", "")).strip() or None
        branch.add(str(t["title"]).strip(), type=_coerce_type(t.get("type")), note=note)
    mm.root.children.append(branch)
    on_event("synthesis_added", count=len(takeaways))
    return len(takeaways)
