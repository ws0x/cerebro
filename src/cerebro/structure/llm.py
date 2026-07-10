"""LLM-backed structurer: transcript → smart MindMap via map → reduce → link.

This is where a map becomes *smart*. The heuristic engine chunks and slices;
this one understands, merges, restructures, and (in expert mode) discovers
cross-branch relationships. It reuses everything downstream (IR → OPML/XMind).

Design notes:
  * MAP calls are independent → run concurrently in a thread pool.
  * Every LLM call is cached by content hash, so re-runs are free and a
    brief→full→expert upgrade reuses the shared MAP results.
  * Failures degrade gracefully: a failed chunk is skipped; only a total wipe-out
    (no map results, or a failed reduce) raises.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..cache import Cache
from ..ir import MindMap, Node, NodeType, Relationship
from ..llm.base import LLMError, LLMProvider
from ..prompts import MAP_SYSTEM, PROMPT_VERSION, cross_link_system, link_system, reduce_system
from ..transcript import Transcript
from .segment import Chunk, chunk_transcript

_MAX_WORDS = {"brief": 2000, "full": 1400, "expert": 1200}


def _coerce_type(value) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.topic


def link_relationships(
    mm: MindMap,
    provider: LLMProvider,
    cache: Cache,
    on_event: Callable[..., None] | None = None,
    cross_video: bool = False,
    relationship_limit: int = 8,
) -> None:
    """Detect and attach cross-branch relationships on an existing MindMap.

    Works on any tree — a single video's own map, or a batch-merged combined
    map spanning many videos — so the exact same call finds within-video
    connections when run per-video, and cross-video connections when run once
    more on the merged result.
    """
    on_event = on_event or (lambda *a, **k: None)
    nodes = [n for n in mm.root.walk() if n.type != NodeType.root]
    if len(nodes) < 3:
        return

    # Map each node ID to its top-level branch (direct child of the root node)
    node_to_branch = {}
    for branch in mm.root.children:
        for node in branch.walk():
            node_to_branch[node.id] = branch

    on_event("link_start")

    if cross_video:
        listing = [
            {"id": i, "title": n.title, "video": node_to_branch[n.id].title}
            for i, n in enumerate(nodes)
            if n.id in node_to_branch
        ]
        system_prompt = cross_link_system(relationship_limit)
    else:
        listing = [{"id": i, "title": n.title} for i, n in enumerate(nodes)]
        system_prompt = link_system(relationship_limit)

    user = json.dumps(listing, ensure_ascii=False)
    key = Cache.key(provider.name, provider.model, PROMPT_VERSION, "link", mm.level, f"cross={cross_video}", f"limit={relationship_limit}", user)
    result = cache.get(key)
    if result is None:
        try:
            result = provider.complete_json(system_prompt, user)
        except LLMError as exc:
            on_event("link_error", error=str(exc))
            return
        cache.set(key, result)
    else:
        on_event("cache_hit", task="link")

    existing_rels = {
        (r.from_id, r.to_id) for r in mm.relationships
    } | {
        (r.to_id, r.from_id) for r in mm.relationships
    }

    for rel in result.get("relationships", []) or []:
        try:
            a, b = int(rel["from"]), int(rel["to"])
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= a < len(nodes) and 0 <= b < len(nodes) and a != b:
            node_a = nodes[a]
            node_b = nodes[b]

            # Enforce that nodes are in different branches/videos
            branch_a = node_to_branch.get(node_a.id)
            branch_b = node_to_branch.get(node_b.id)
            if branch_a is None or branch_b is None or branch_a == branch_b:
                continue

            # Discard duplicate/existing proposals
            if (node_a.id, node_b.id) in existing_rels or (node_b.id, node_a.id) in existing_rels:
                continue

            mm.relationships.append(
                Relationship(from_id=node_a.id, to_id=node_b.id, label=str(rel.get("label", "")).strip())
            )
            existing_rels.add((node_a.id, node_b.id))
            existing_rels.add((node_b.id, node_a.id))


class LLMStructurer:
    def __init__(
        self,
        provider: LLMProvider,
        cache: Cache | None = None,
        max_workers: int = 6,
        on_event: Callable[..., None] | None = None,
        relationship_limit: int = 8,
    ):
        self.provider = provider
        self.cache = cache or Cache(enabled=False)
        self.max_workers = max_workers
        self.on_event = on_event or (lambda *a, **k: None)
        self.relationship_limit = relationship_limit

    # -- cached provider call -------------------------------------------------
    def _call(self, task: str, system: str, user: str, *key_parts) -> dict:
        key = Cache.key(
            self.provider.name, self.provider.model, PROMPT_VERSION, task, *key_parts
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.on_event("cache_hit", task=task)
            return cached
        result = self.provider.complete_json(system, user)
        self.cache.set(key, result)
        return result

    # -- pipeline stages ------------------------------------------------------
    def _map_chunk(self, index: int, chunk: Chunk, level: str) -> dict:
        result = self._call("map", MAP_SYSTEM, chunk.text, level, chunk.text)
        result["t"] = int(chunk.start)
        return result

    def _map(self, chunks: list[Chunk], level: str) -> list[dict]:
        self.on_event("map_start", total=len(chunks))
        results: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._map_chunk, i, c, level): i for i, c in enumerate(chunks)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except LLMError as exc:
                    self.on_event("map_error", index=i, error=str(exc))
                self.on_event("map_progress", done=len(results), total=len(chunks))
        ordered = [results[i] for i in sorted(results)]
        if not ordered:
            raise LLMError("All MAP calls failed; cannot build a map.")
        return ordered

    def _reduce(self, transcript: Transcript, map_results: list[dict], level: str) -> dict:
        self.on_event("reduce_start")
        payload = {
            "title": transcript.title,
            "level": level,
            "segments": map_results,
            "note": "You may include an optional integer 't' (seconds) on any node.",
        }
        user = json.dumps(payload, ensure_ascii=False)
        tree = self._call("reduce", reduce_system(level), user, level, user)
        if "children" not in tree:
            raise LLMError(f"REDUCE returned no tree: {str(tree)[:200]}")
        return tree

    def _build_node(self, data: dict) -> Node:
        node = Node(
            title=str(data.get("title", "Untitled")).strip() or "Untitled",
            type=_coerce_type(data.get("type")),
            note=(data.get("note") or None),
            timestamp=(float(data["t"]) if isinstance(data.get("t"), (int, float)) else None),
        )
        for child in data.get("children", []) or []:
            if isinstance(child, dict):
                node.children.append(self._build_node(child))
        return node

    def _to_ir(self, transcript: Transcript, tree: dict, level: str) -> MindMap:
        root = Node(
            title=str(tree.get("central") or transcript.title or "Mind Map").strip(),
            type=NodeType.root,
        )
        for child in tree.get("children", []) or []:
            if isinstance(child, dict):
                root.children.append(self._build_node(child))
        return MindMap(title=root.title, root=root, source=transcript.source, level=level)

    # -- public API -----------------------------------------------------------
    def structure(self, transcript: Transcript, level: str = "full") -> MindMap:
        level = level if level in _MAX_WORDS else "full"
        chunks = chunk_transcript(transcript, _MAX_WORDS[level])
        if not chunks:
            raise LLMError("Transcript is empty; nothing to map.")
        map_results = self._map(chunks, level)
        tree = self._reduce(transcript, map_results, level)
        mm = self._to_ir(transcript, tree, level)
        if level == "expert":
            link_relationships(
                mm,
                self.provider,
                self.cache,
                on_event=self.on_event,
                relationship_limit=self.relationship_limit,
            )
        self.on_event("done", nodes=mm.node_count(), relationships=len(mm.relationships))
        return mm
