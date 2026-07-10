"""Batch orchestration: many sources -> one combined MindMap.

Each item is ingested and structured independently (reusing the exact same
single-video pipeline), then merged as a branch under one combined root. A
failing item is recorded and skipped — it never aborts the run, since a single
private/caption-less video in a 40-video playlist shouldn't kill the batch.

**Incremental reruns.** A batch source's item list and each item's resulting
branch are persisted (~/.cerebro/batch-snapshots/), keyed by the batch source
itself (playlist URL or course-folder path). On a rerun, any item whose exact
source string matches a previous run's is reused wholesale — no transcript
refetch, no re-structuring — since each individual video's own MAP/REDUCE/LINK
calls are already content-hash cached anyway; the real cost this avoids is the
transcript fetch and structuring orchestration itself. Only genuinely new
items (added to the playlist/folder since last time) are processed fresh.
Cross-video linking isn't separately cached — it's one cheap call regardless
of playlist size, and always reruns to reflect the current combined tree.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .ingest import load_transcript
from .ir import MindMap, Node, NodeType, Relationship
from .paths import BATCH_SNAPSHOT_DIR
from .structure.base import Structurer

if TYPE_CHECKING:
    from .cache import Cache

_BATCH_SNAPSHOT_VERSION = "v1"


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


@dataclass
class BatchDiff:
    """What changed since the previous run of this exact batch source."""

    reused: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    previous_built_at: str | None = None

    @property
    def total(self) -> int:
        return len(self.reused) + len(self.added)


def dry_run_batch(
    items: list[BatchItem], level: str, batch_source: str | None, snapshot_dir: str | Path | None = None
) -> tuple[list[str], list[str]]:
    """Returns ``(reused_labels, new_labels)`` for ``items`` without doing any
    work — no transcript fetch, no LLM call, nothing written. Just answers
    "what would run_batch do" by reading the same snapshot run_batch would."""
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else BATCH_SNAPSHOT_DIR
    old_items: dict[str, dict] = {}
    if batch_source:
        snapshot = _load_batch_snapshot(batch_source, {"level": level}, snapshot_dir)
        if snapshot is not None:
            old_items = snapshot["items"]
    reused = [item.label for item in items if item.source in old_items]
    new = [item.label for item in items if item.source not in old_items]
    return reused, new


def _batch_snapshot_path(batch_source: str, snapshot_dir: Path) -> Path:
    key = hashlib.sha256(batch_source.encode("utf-8")).hexdigest()[:24]
    return snapshot_dir / f"{key}.json"


def forget_batch_snapshot(batch_source: str, snapshot_dir: str | Path | None = None) -> bool:
    """Delete the incremental snapshot for ``batch_source``, if one exists.

    ``batch_source`` must match exactly what was passed to ``cerebro batch``
    (the raw playlist URL or folder path string) — the snapshot key is a hash
    of that literal string, unlike the tree snapshot's resolved-path key,
    since a batch source is just as often a URL as a filesystem path. Returns
    whether a snapshot actually existed to delete.
    """
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else BATCH_SNAPSHOT_DIR
    path = _batch_snapshot_path(batch_source, snapshot_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def _load_batch_snapshot(batch_source: str, params: dict, snapshot_dir: Path) -> dict | None:
    path = _batch_snapshot_path(batch_source, snapshot_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != _BATCH_SNAPSHOT_VERSION or data.get("params") != params:
        return None  # incompatible params (different level) -> safest to ignore
    return data


def _save_batch_snapshot(batch_source: str, params: dict, items_data: dict, snapshot_dir: Path) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _BATCH_SNAPSHOT_VERSION,
        "params": params,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": items_data,
        "source": batch_source,  # for `cerebro status` to list *what* is remembered, not just a hash
    }
    _batch_snapshot_path(batch_source, snapshot_dir).write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


def list_batch_snapshots(snapshot_dir: str | Path | None = None) -> list[dict]:
    """Every saved batch snapshot's summary — for `cerebro status`."""
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else BATCH_SNAPSHOT_DIR
    if not snapshot_dir.exists():
        return []
    out = []
    for path in sorted(snapshot_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "source": data.get("source", "(unknown — run before this field existed; rerun to update)"),
            "built_at": data.get("built_at", "?"),
            "items": len(data.get("items", {})),
        })
    return out


def run_batch(
    items: list[BatchItem],
    structurer_factory: Callable[[], Structurer],
    level: str,
    title: str,
    max_workers: int = 3,
    on_event: Callable[..., None] | None = None,
    cache: "Cache | None" = None,
    whisper_model: str = "base",
    incremental: bool = True,
    batch_source: str | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[MindMap, list[BatchOutcome], BatchDiff | None]:
    """Returns ``(combined, outcomes, diff)``.

    ``diff`` is ``None`` when there's nothing to diff against — no
    ``batch_source`` was given, this is the first time this batch source has
    been run, or ``incremental=False`` was requested (a fresh snapshot is
    still saved for next time either way).
    """
    on_event = on_event or (lambda *a, **k: None)
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else BATCH_SNAPSHOT_DIR
    params = {"level": level}

    old_items: dict[str, dict] = {}
    had_prior_snapshot = False
    if incremental and batch_source:
        snapshot = _load_batch_snapshot(batch_source, params, snapshot_dir)
        if snapshot is not None:
            had_prior_snapshot = True
            old_items = snapshot["items"]

    reused_labels: list[str] = []
    added_labels: list[str] = []

    def _process(item: BatchItem) -> BatchOutcome:
        t0 = time.perf_counter()
        old = old_items.get(item.source)
        if old is not None:
            reused_labels.append(item.label)
            branch = Node.model_validate(old["branch"])
            rels = [Relationship.model_validate(r) for r in old.get("relationships", [])]
            mm = MindMap(title=old["label"], root=branch, relationships=rels, level=level)
            return BatchOutcome(item.label, mm, None, time.perf_counter() - t0)

        added_labels.append(item.label)
        try:
            transcript = load_transcript(item.source, whisper_model=whisper_model, cache=cache)
            mm = structurer_factory().structure(transcript, level=level)
            return BatchOutcome(item.label, mm, None, time.perf_counter() - t0)
        except Exception as exc:  # an item's failure must not abort the batch
            return BatchOutcome(item.label, None, str(exc), time.perf_counter() - t0)

    outcomes: dict[int, BatchOutcome] = {}
    items_data: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            outcome = fut.result()
            outcomes[i] = outcome
            if outcome.mindmap is not None:
                item = items[i]
                branch = outcome.mindmap.root
                branch.type = NodeType.topic
                # Use the batch item's own label (playlist video title /
                # lesson filename) rather than the structurer's guess at a
                # title, so branches match the source listing the user
                # recognizes.
                branch.title = outcome.label
                items_data[item.source] = {
                    "label": outcome.label,
                    "branch": branch.model_dump(mode="json"),
                    "relationships": [r.model_dump(mode="json") for r in outcome.mindmap.relationships],
                }
                # Checkpoint after every item, not just once at the very
                # end -- a killed/crashed run (Ctrl+C, network drop, power
                # loss) partway through a long playlist/course folder then
                # loses nothing already completed on the next attempt,
                # instead of refetching and re-structuring everything from
                # scratch just because the run didn't reach its last item.
                # A full overwrite per item is fine at the scale this is for
                # (tens of items, not thousands); a throttled/batched saver
                # would just be complexity nothing here needs yet.
                if batch_source:
                    _save_batch_snapshot(batch_source, params, items_data, snapshot_dir)
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
    for item, outcome in zip(items, ordered):
        if outcome.mindmap is None:
            continue
        root.children.append(outcome.mindmap.root)  # type/title already set above
        relationships.extend(outcome.mindmap.relationships)

    if not root.children:
        root.add("(no items processed successfully)", type=NodeType.detail)

    combined = MindMap(title=title, root=root, relationships=relationships, level=level)

    if batch_source:
        _save_batch_snapshot(batch_source, params, items_data, snapshot_dir)

    if not had_prior_snapshot:
        return combined, ordered, None

    removed_sources = set(old_items) - {item.source for item in items}
    removed = sorted(old_items[src].get("label", src) for src in removed_sources)
    diff = BatchDiff(
        reused=reused_labels,
        added=added_labels,
        removed=removed,
        previous_built_at=snapshot.get("built_at"),
    )
    return combined, ordered, diff
