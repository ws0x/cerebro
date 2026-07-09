"""Batch orchestration: many sources -> one combined MindMap.

Each item is ingested and structured independently (reusing the exact same
single-video pipeline), then merged as a branch under one combined root. A
failing item is recorded and skipped — it never aborts the run, since a single
private/caption-less video in a 40-video playlist shouldn't kill the batch.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from .ingest import load_transcript
from .ir import MindMap, Node, NodeType, Relationship
from .structure.base import Structurer


@dataclass
class BatchItem:
    label: str
    source: str


@dataclass
class BatchOutcome:
    label: str
    mindmap: MindMap | None
    error: str | None
    seconds: float


def run_batch(
    items: list[BatchItem],
    structurer_factory: Callable[[], Structurer],
    level: str,
    title: str,
    max_workers: int = 3,
    on_event: Callable[..., None] | None = None,
) -> tuple[MindMap, list[BatchOutcome]]:
    on_event = on_event or (lambda *a, **k: None)

    def _process(item: BatchItem) -> BatchOutcome:
        t0 = time.perf_counter()
        try:
            transcript = load_transcript(item.source)
            mm = structurer_factory().structure(transcript, level=level)
            return BatchOutcome(item.label, mm, None, time.perf_counter() - t0)
        except Exception as exc:  # an item's failure must not abort the batch
            return BatchOutcome(item.label, None, str(exc), time.perf_counter() - t0)

    outcomes: dict[int, BatchOutcome] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            outcome = fut.result()
            outcomes[i] = outcome
            on_event(
                "item_done",
                completed=len(outcomes),
                total=len(items),
                label=outcome.label,
                ok=outcome.error is None,
                error=outcome.error,
            )

    ordered = [outcomes[i] for i in range(len(items))]

    root = Node(title=title, type=NodeType.root)
    relationships: list[Relationship] = []
    for outcome in ordered:
        if outcome.mindmap is None:
            continue
        branch = outcome.mindmap.root
        branch.type = NodeType.topic
        # Use the batch item's own label (playlist video title / lesson
        # filename) rather than the structurer's guess at a title, so branches
        # match the source listing the user recognizes.
        branch.title = outcome.label
        root.children.append(branch)
        relationships.extend(outcome.mindmap.relationships)

    if not root.children:
        root.add("(no items processed successfully)", type=NodeType.detail)

    combined = MindMap(title=title, root=root, relationships=relationships, level=level)
    return combined, ordered
