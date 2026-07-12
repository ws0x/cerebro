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
from ..prompts import (
    HEADING_POLISH_SYSTEM,
    MAP_SYSTEM,
    PROMPT_VERSION,
    cross_link_system,
    link_system,
    reduce_system,
    section_fill_system,
)
from ..transcript import Transcript
from .anchors import verify_and_repair_anchors
from .enumeration import EnumeratedSection, detect_enumeration
from .segment import Chunk, chunk_transcript
from .synthesis import add_synthesis

_INTRO_MIN_WORDS = 40  # a pre-#1 intro shorter than this isn't worth an Overview branch
_SECTION_NOTE_FALLBACK_CHARS = 400  # deterministic note when a section's LLM fill fails

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

    # Every node's descendant ids, so we can reject a link between a node and
    # its own ancestor/descendant -- that pair is already expressed by the tree
    # edge, so a "relationship" arrow there is redundant. This (not "different
    # branch") is the correct exclusion: for an author-numbered map the most
    # valuable cross-links are the causal claims made *inside* one section
    # ("keep promises" -> "builds confidence"), which a different-branch rule
    # would wrongly discard.
    descendants: dict[str, set[str]] = {}

    def _collect(node: Node) -> set[str]:
        ids: set[str] = set()
        for child in node.children:
            ids.add(child.id)
            ids |= _collect(child)
        descendants[node.id] = ids
        return ids

    _collect(mm.root)

    def _is_hierarchical(id_a: str, id_b: str) -> bool:
        return id_b in descendants.get(id_a, ()) or id_a in descendants.get(id_b, ())

    on_event("link_start")

    def _entry(i: int, n: Node) -> dict:
        # Include the node's note (so LINK can prefer the source's *stated*
        # causal claims over merely-plausible ones) and its section/branch (so
        # the model can see the structure and aim for cross-section links
        # instead of proposing parent->child pairs the filter would reject).
        e: dict = {"id": i, "title": n.title}
        if n.note:
            e["note"] = n.note
        branch = node_to_branch.get(n.id)
        if branch is not None:
            e["section"] = branch.title
        return e

    if cross_video:
        listing = [
            {**_entry(i, n), "video": node_to_branch[n.id].title}
            for i, n in enumerate(nodes)
            if n.id in node_to_branch
        ]
        system_prompt = cross_link_system(relationship_limit)
    else:
        listing = [_entry(i, n) for i, n in enumerate(nodes)]
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

            branch_a = node_to_branch.get(node_a.id)
            branch_b = node_to_branch.get(node_b.id)
            if branch_a is None or branch_b is None:
                continue
            if cross_video:
                # Cross-video linking's whole purpose is connecting *different*
                # sources, so there the different-top-level-branch rule stays.
                if branch_a == branch_b:
                    continue
            elif _is_hierarchical(node_a.id, node_b.id):
                # Within a single map, reject only ancestor/descendant pairs
                # (redundant with the tree); sibling and cross-section links
                # are exactly the non-hierarchical cross-links we want.
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
        synthesize: bool = True,
    ):
        self.provider = provider
        self.cache = cache or Cache(enabled=False)
        self.max_workers = max_workers
        self.on_event = on_event or (lambda *a, **k: None)
        self.relationship_limit = relationship_limit
        # Synthesis applies only to the enumerated (author-numbered) path here
        # -- the free-form flat path already gets REDUCE, which does this job.
        self.synthesize = synthesize

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
        # Prefer the source's real title over the LLM's re-summarized "central":
        # the LLM tends to drop specifics ("7 Non-Negotiables…" → "Bulletproof
        # Mindset"), and the real title is both more faithful and what the user
        # actually recognizes. "central" is only a fallback when there's no title.
        root = Node(
            title=str(transcript.title or tree.get("central") or "Mind Map").strip(),
            type=NodeType.root,
        )
        for child in tree.get("children", []) or []:
            if isinstance(child, dict):
                root.children.append(self._build_node(child))
        if level == "brief":
            # brief = advance organizer: main branches + a gist note each,
            # nothing deeper. The prompt already asks for "minimal nesting",
            # but live testing showed that's unreliable -- a real run
            # produced a *deeper, bigger* tree at brief than the same source
            # got at expert, because the model kept the branch count in line
            # but ignored the nesting instruction. Enforced here in code
            # instead, the same fix already applied to the enumerated path's
            # brief level for the same reason (models routinely ignore shape
            # instructions in the prompt alone).
            for branch in root.children:
                branch.children = []
        return MindMap(title=root.title, root=root, source=transcript.source, level=level)

    # -- enumerated (author-numbered list) path -------------------------------
    def _section_spans(
        self, transcript: Transcript, sections: list[EnumeratedSection]
    ) -> tuple[str, list[str]]:
        """Slice the transcript into (intro_text, [section_text, …]) by
        timestamp -- robust to any segment-filtering differences, since each
        section owns [its start, the next section's start)."""
        bounds = [s.start for s in sections] + [float("inf")]

        def text_between(lo: float, hi: float) -> str:
            return " ".join(
                s.text.strip() for s in transcript.segments if s.text.strip() and lo <= s.start < hi
            ).strip()

        intro = text_between(float("-inf"), bounds[0])
        section_texts = [text_between(bounds[i], bounds[i + 1]) for i in range(len(sections))]
        return intro, section_texts

    def _polish_headings(
        self, transcript: Transcript, sections: list[EnumeratedSection], section_texts: list[str]
    ) -> list[str]:
        """One call to tidy the author's ASR-mangled spoken lead-ins into clean
        parallel headings. Falls back to the deterministic headings on any
        failure or a shape mismatch -- the enumeration detector's own headings
        are already good on their own, this only polishes them."""
        deterministic = [s.heading or f"Part {s.number}" for s in sections]
        payload = [
            {
                "number": s.number,
                "raw_lead_in": s.heading_raw[:120],
                "section_start": " ".join(text.split()[:40]),
            }
            for s, text in zip(sections, section_texts)
        ]
        user = json.dumps(payload, ensure_ascii=False)
        try:
            result = self._call("heading_polish", HEADING_POLISH_SYSTEM, user, user)
        except LLMError:
            return deterministic
        headings = result.get("headings")
        if not isinstance(headings, list) or len(headings) != len(sections):
            return deterministic
        cleaned = []
        for polished, fallback in zip(headings, deterministic):
            polished = str(polished).strip() if polished else ""
            cleaned.append(polished or fallback)
        return cleaned

    def _fill_section(self, heading: str, text: str, level: str) -> tuple[dict, bool]:
        """Content (note + points) for one section whose title is FIXED. On
        failure, a deterministic snippet note keeps the section usable rather
        than blank -- same resilience posture as the document/enrich path.
        Returns (content, succeeded) -- the caller needs to know whether ANY
        section actually got real AI content, or a total failure here (every
        section silently falling back) would look identical to a real map to
        anything counting nodes, and get reported as an AI-engine success."""
        capped = " ".join(text.split()[: _MAX_WORDS[level]])
        user = json.dumps({"section_title": heading, "transcript": capped}, ensure_ascii=False)
        try:
            return self._call("section_fill", section_fill_system(level), user, level, user), True
        except LLMError as exc:
            self.on_event("map_error", error=str(exc))
            snippet = text.strip()[:_SECTION_NOTE_FALLBACK_CHARS]
            return {"note": snippet, "points": []}, False

    def _build_section_branch(
        self, number: int | None, heading: str, start: float, filled: dict, level: str
    ) -> Node:
        prefix = f"{number}. " if number is not None else ""
        branch = Node(
            title=f"{prefix}{heading}".strip(),
            type=NodeType.topic,
            note=(str(filled.get("note")).strip() or None) if filled.get("note") else None,
            timestamp=float(start) if start not in (None, float("-inf")) else None,
        )
        # brief = advance organizer: the numbered spine + a gist note each, no
        # sub-points -- enforced here in code rather than trusted to the prompt,
        # since models routinely ignore "return an empty list".
        if level == "brief":
            return branch
        for point in filled.get("points", []) or []:
            if isinstance(point, dict) and str(point.get("title", "")).strip():
                branch.children.append(
                    Node(title=str(point["title"]).strip(), type=_coerce_type(point.get("type")))
                )
        return branch

    def _structure_enumerated(
        self, transcript: Transcript, sections: list[EnumeratedSection], level: str
    ) -> MindMap:
        """Deterministic numbered spine, LLM only filling each section's content
        -- so the author's own "N things" structure survives verbatim instead of
        being dissolved into renamed thematic clusters."""
        intro, section_texts = self._section_spans(transcript, sections)
        self.on_event("map_start", total=len(sections) + 1)

        headings = self._polish_headings(transcript, sections, section_texts)

        root = Node(title=transcript.title or "Mind Map", type=NodeType.root)
        any_succeeded = False

        # Optional leading advance-organizer branch from the pre-#1 intro/thesis.
        if len(intro.split()) >= _INTRO_MIN_WORDS:
            overview, ok = self._fill_section("Overview", intro, level)
            any_succeeded = any_succeeded or ok
            root.children.append(self._build_section_branch(None, "Overview", sections[0].start, overview, level))
        self.on_event("map_progress", done=1, total=len(sections) + 1)

        done = 1
        results: dict[int, Node] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._fill_section, headings[i], section_texts[i], level): i
                for i in range(len(sections))
            }
            for fut in as_completed(futures):
                i = futures[fut]
                filled, ok = fut.result()
                any_succeeded = any_succeeded or ok
                results[i] = self._build_section_branch(
                    sections[i].number, headings[i], sections[i].start, filled, level
                )
                done += 1
                self.on_event("map_progress", done=done, total=len(sections) + 1)

        if not any_succeeded:
            # Every section silently fell back to a raw snippet -- same
            # all-failed signal _map() raises on, so the caller (cli.py)
            # falls back to the heuristic engine and reports it honestly
            # instead of labeling a 100%-fallback map as an AI success.
            raise LLMError("All section-fill calls failed; cannot build a map.")

        for i in range(len(sections)):
            root.children.append(results[i])

        mm = MindMap(title=root.title, root=root, source=transcript.source, level=level)
        verify_and_repair_anchors(
            mm, transcript.full_text, self.provider, self.cache, level, on_event=self.on_event
        )
        if self.synthesize:
            add_synthesis(mm, self.provider, self.cache, level, on_event=self.on_event)
        if level == "expert":
            link_relationships(
                mm, self.provider, self.cache, on_event=self.on_event, relationship_limit=self.relationship_limit
            )
        self.on_event("done", nodes=mm.node_count(), relationships=len(mm.relationships))
        return mm

    # -- public API -----------------------------------------------------------
    def structure(self, transcript: Transcript, level: str = "full") -> MindMap:
        level = level if level in _MAX_WORDS else "full"

        # A video that is an explicit author-numbered list ("7 non-negotiables",
        # "5 tips") gets its own path: the author already handed us the spine, so
        # we keep it verbatim and numbered instead of asking REDUCE to invent a
        # new thematic grouping -- the speech equivalent of honoring a PDF's TOC.
        sections = detect_enumeration(transcript)
        if sections:
            self.on_event("enumeration_detected", sections=len(sections))
            return self._structure_enumerated(transcript, sections, level)

        chunks = chunk_transcript(transcript, _MAX_WORDS[level])
        if not chunks:
            raise LLMError("Transcript is empty; nothing to map.")
        map_results = self._map(chunks, level)
        tree = self._reduce(transcript, map_results, level)
        mm = self._to_ir(transcript, tree, level)
        verify_and_repair_anchors(
            mm, transcript.full_text, self.provider, self.cache, level, on_event=self.on_event
        )
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
