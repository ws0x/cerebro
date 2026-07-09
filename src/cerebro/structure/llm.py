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
from ..prompts import LINK_SYSTEM, MAP_SYSTEM, PROMPT_VERSION, reduce_system
from ..transcript import Transcript
from .segment import Chunk, chunk_transcript

_MAX_WORDS = {"brief": 2000, "full": 1400, "expert": 1200}


def _coerce_type(value) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.topic


class LLMStructurer:
    def __init__(
        self,
        provider: LLMProvider,
        cache: Cache | None = None,
        max_workers: int = 6,
        on_event: Callable[..., None] | None = None,
    ):
        self.provider = provider
        self.cache = cache or Cache(enabled=False)
        self.max_workers = max_workers
        self.on_event = on_event or (lambda *a, **k: None)

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

    def _link(self, mm: MindMap) -> None:
        self.on_event("link_start")
        nodes = [n for n in mm.root.walk() if n.type != NodeType.root]
        if len(nodes) < 3:
            return
        listing = [{"id": i, "title": n.title} for i, n in enumerate(nodes)]
        user = json.dumps(listing, ensure_ascii=False)
        try:
            result = self._call("link", LINK_SYSTEM, user, mm.level, user)
        except LLMError as exc:
            self.on_event("link_error", error=str(exc))
            return
        for rel in result.get("relationships", []) or []:
            try:
                a, b = int(rel["from"]), int(rel["to"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= a < len(nodes) and 0 <= b < len(nodes) and a != b:
                mm.relationships.append(
                    Relationship(
                        from_id=nodes[a].id,
                        to_id=nodes[b].id,
                        label=str(rel.get("label", "")).strip(),
                    )
                )

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
            self._link(mm)
        self.on_event("done", nodes=mm.node_count(), relationships=len(mm.relationships))
        return mm
