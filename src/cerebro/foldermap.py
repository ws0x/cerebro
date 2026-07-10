"""Turn a folder's directory structure into a MindMap.

A deliberately separate module from the video pipeline — a folder path means
something completely different here than it does to ``batch`` (which treats a
folder as a course of video lessons). Unlike video, the hierarchy doesn't need
to be *discovered*: the filesystem already gives it to us. What's optional is
*smart labeling* — inferring a folder's purpose from its contents rather than
just using its literal name, via the same LLM provider infrastructure the
video pipeline already uses.

Reuses the existing IR (``MindMap``/``Node``) and every downstream converter
unchanged — this is the payoff of the IR-first architecture.

**Incremental rebuilds.** A previous run's tree is persisted as a snapshot
(``~/.cerebro/tree-snapshots/``), keyed by the resolved folder path: a
Merkle-style signature per directory (its own direct file fingerprints *plus*
its subdirectories' signatures, computed bottom-up) plus any AI label already
assigned. A folder's signature only changes if something changed anywhere
beneath it — a file three levels down changes that file's parent's signature,
and every signature on the path back up to the root, so a mismatch reliably
means "something changed in this subtree," not just "changed right here."
Every directory is still walked each run (cheap — stat calls, not LLM calls),
but an unchanged folder's AI label is reused as-is and never resubmitted, and
the run reports what actually changed instead of silently redoing everything.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .cache import Cache
from .ir import MindMap, Node, NodeType
from .llm.base import LLMError, LLMProvider
from .paths import TREE_SNAPSHOT_DIR
from .prompts import FOLDER_LABEL_SYSTEM, PROMPT_VERSION

# Common noise directories that would bloat a folder map with nothing useful.
# Applied regardless of .gitignore, since e.g. a repo without a committed
# .gitignore can still have a stray node_modules/ nobody wants to see.
# Includes cerebro's own config dir (see paths.CONFIG_DIR) — mapping a home
# directory would otherwise pull in cerebro's own cache/keys as noise, and
# the cache's contents legitimately change between runs, which would make
# repeated maps of the same folder unstable for no reason a user would guess.
_DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", "dist",
    "build", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".idea",
    ".vscode", "target", ".next", ".nuxt", "coverage", ".cache", ".DS_Store",
    ".cerebro",
}

_NOTABLE_PREFIXES = ("readme", "license", "changelog", "contributing")
_NOTABLE_NAMES = {
    "package.json", "pyproject.toml", "dockerfile", "makefile",
    "cargo.toml", "go.mod", "setup.py", "requirements.txt",
}

_SNAPSHOT_VERSION = "v2"  # bump to invalidate all snapshots after a format change


def _is_ignored_name(name: str) -> bool:
    if name in _DEFAULT_IGNORE_DIRS:
        return True
    return name.endswith(".egg-info")


def _is_notable_file(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(_NOTABLE_PREFIXES) or lower in _NOTABLE_NAMES


def _load_gitignore_spec(root: Path):
    gi = root / ".gitignore"
    if not gi.exists():
        return None
    try:
        import pathspec

        lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
        return pathspec.PathSpec.from_lines("gitignore", lines)
    except Exception:
        return None


def _dir_signature(
    file_fingerprints: list[tuple[str, int, float]], subdir_entries: list[tuple[str, str | None]]
) -> str:
    """A Merkle-style signature: depends on this folder's own direct file
    fingerprints *and* each subdirectory's own signature (which recursively
    depends on everything beneath it). A change anywhere in the subtree
    changes every signature on the path back to this folder.
    """
    payload = json.dumps(
        {"files": sorted(file_fingerprints), "dirs": sorted(subdir_entries)}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TreeDiff:
    """What changed between the previous snapshot of a folder and this run."""

    reused: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    previous_built_at: str | None = None

    @property
    def total(self) -> int:
        return len(self.reused) + len(self.added) + len(self.changed)


@dataclass
class _WalkContext:
    old_signatures: dict[str, str]
    old_labels: dict[str, str]
    new_signatures: dict[str, str] = field(default_factory=dict)
    node_by_relpath: dict[str, Node] = field(default_factory=dict)
    reused: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    nodes_needing_labels: list[Node] = field(default_factory=list)


@dataclass
class PendingSnapshot:
    """A build's state, not yet persisted. Saving is deferred until after any
    AI labeling happens (see :func:`finalize_tree_snapshot`) — labeling
    mutates node notes *after* ``build_folder_map`` returns, so saving
    immediately would silently lose every newly-assigned label the moment the
    process exits, and the next run would never benefit from them.
    """

    root: Path
    params: dict
    signatures: dict[str, str]
    node_by_relpath: dict[str, Node]
    snapshot_dir: Path


def finalize_tree_snapshot(pending: PendingSnapshot) -> None:
    """Persist the final state of a build — call this once, after any
    :func:`label_folders` call has finished mutating the tree's notes (or
    immediately, if running in purely heuristic/offline mode with no AI step
    at all — either way, the next run should see whatever the final notes
    actually were)."""
    labels = {relpath: node.note for relpath, node in pending.node_by_relpath.items() if node.note}
    _save_snapshot(pending.root, pending.params, pending.signatures, labels, pending.snapshot_dir)


def _snapshot_path(root: Path, snapshot_dir: Path) -> Path:
    key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:24]
    return snapshot_dir / f"{key}.json"


def list_tree_snapshots(snapshot_dir: str | Path | None = None) -> list[dict]:
    """Every saved tree snapshot's summary — for `cerebro status`."""
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else TREE_SNAPSHOT_DIR
    if not snapshot_dir.exists():
        return []
    out = []
    for path in sorted(snapshot_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "source": data.get("source", "(unknown — mapped before this field existed; rerun to update)"),
            "built_at": data.get("built_at", "?"),
            "folders": len(data.get("signatures", {})),
            "labels": len(data.get("labels", {})),
        })
    return out


def forget_tree_snapshot(root: str | Path, snapshot_dir: str | Path | None = None) -> bool:
    """Delete the incremental snapshot for ``root``, if one exists.

    Doesn't require ``root`` to still exist on disk — forgetting the history
    of a folder you've since deleted or renamed is a legitimate use. Returns
    whether a snapshot actually existed to delete.
    """
    root = Path(root).resolve()
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else TREE_SNAPSHOT_DIR
    path = _snapshot_path(root, snapshot_dir)
    if path.exists():
        path.unlink()
        return True
    return False


def _load_snapshot(root: Path, params: dict, snapshot_dir: Path) -> dict | None:
    path = _snapshot_path(root, snapshot_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") != _SNAPSHOT_VERSION or data.get("params") != params:
        return None  # incompatible build parameters -> safest to ignore, not misapply
    return data


def _save_snapshot(
    root: Path,
    params: dict,
    signatures: dict[str, str],
    labels: dict[str, str],
    snapshot_dir: Path,
) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _SNAPSHOT_VERSION,
        "params": params,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signatures": signatures,
        "labels": labels,
        "source": str(root),  # for `cerebro status` to list *what* is remembered, not just a hash
    }
    _snapshot_path(root, snapshot_dir).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _build_node(
    path: Path, root: Path, spec, max_depth: int, max_files: int, depth: int, ctx: _WalkContext
) -> tuple[Node | None, str | None]:
    """Returns ``(node, signature)``. ``signature`` is ``None`` for files and
    for folders truncated by ``max_depth`` (which weren't fully walked, so no
    trustworthy signature can be computed — they're just never reused)."""
    if _is_ignored_name(path.name):
        return None, None
    if spec is not None:
        try:
            rel = str(path.relative_to(root).as_posix())
            if path.is_dir():
                rel += "/"
            if spec.match_file(rel):
                return None, None
        except ValueError:
            pass

    if path.is_file():
        node_type = NodeType.definition if _is_notable_file(path.name) else NodeType.detail
        return Node(title=path.name, type=node_type), None

    relpath = str(path.relative_to(root).as_posix()) or "."

    try:
        entries = [e for e in path.iterdir() if not _is_ignored_name(e.name)]
    except (PermissionError, OSError):
        return Node(title=path.name, type=NodeType.topic, note="(could not be read)"), None

    if spec is not None:
        filtered = []
        for e in entries:
            try:
                rel = str(e.relative_to(root).as_posix()) + ("/" if e.is_dir() else "")
                if spec.match_file(rel):
                    continue
            except ValueError:
                pass
            filtered.append(e)
        entries = filtered

    file_fps: list[tuple[str, int, float]] = []
    for e in entries:
        if e.is_file():
            try:
                st = e.stat()
                file_fps.append((e.name, st.st_size, st.st_mtime))
            except OSError:
                pass

    node = Node(title=path.name, type=NodeType.topic)
    entries.sort(key=lambda p: (p.is_file(), p.name.lower()))

    signature: str | None
    if depth >= max_depth:
        try:
            total = sum(1 for _ in path.rglob("*"))
        except (PermissionError, OSError):
            total = 0
        if total:
            node.add(f"+{total} more item(s)", type=NodeType.detail)
        signature = None  # not fully walked -> never eligible for reuse
    else:
        children: list[Node] = []
        subdir_sigs: list[tuple[str, str | None]] = []
        for entry in entries:
            child, child_sig = _build_node(entry, root, spec, max_depth, max_files, depth + 1, ctx)
            if child is not None:
                children.append(child)
                if entry.is_dir():
                    subdir_sigs.append((entry.name, child_sig))

        dirs = [c for c in children if c.type == NodeType.topic]
        files = [c for c in children if c.type != NodeType.topic]
        total_files = len(files)
        if total_files > max_files:
            remaining = total_files - max_files
            files = files[:max_files] + [Node(title=f"+{remaining} more file(s)", type=NodeType.detail)]
        node.children = dirs + files
        if dirs or total_files:
            node.note = f"{len(dirs)} folder(s), {total_files} file(s)"
        signature = _dir_signature(file_fps, subdir_sigs)

    if signature is not None:
        ctx.new_signatures[relpath] = signature
    old_signature = ctx.old_signatures.get(relpath)

    ctx.node_by_relpath[relpath] = node

    if signature is not None and old_signature == signature:
        ctx.reused.append(relpath)
        old_label = ctx.old_labels.get(relpath)
        if old_label:
            node.note = old_label  # preserve the AI label instead of the generic count note
    else:
        (ctx.changed if old_signature is not None else ctx.added).append(relpath)
        if depth > 0:  # the root itself is never AI-labeled — matches
            ctx.nodes_needing_labels.append(node)  # label_folders' own default (topic only, root excluded)

    return node, signature


def build_folder_map(
    root: str | Path,
    max_depth: int = 8,
    max_files: int = 20,
    respect_gitignore: bool = True,
    incremental: bool = True,
    snapshot_dir: str | Path | None = None,
) -> tuple[MindMap, TreeDiff | None, list[Node]]:
    """Walk ``root`` and build a MindMap of its directory structure.

    Returns ``(mindmap, diff, nodes_needing_labels, pending_snapshot)``.

    ``diff`` is ``None`` when there's nothing to diff against — either this is
    the first time ``root`` has been mapped, or ``incremental=False`` was
    requested (which still saves a fresh snapshot for next time, it just
    doesn't reuse or report against the old one).

    ``nodes_needing_labels`` is every folder whose subtree changed (or is new)
    since the last snapshot — pass it straight to :func:`label_folders`'s
    ``nodes`` argument so a rerun only spends AI calls on what's actually
    different, never on folders whose label already carried over unchanged.
    On a first-ever run this is simply every folder.

    ``pending_snapshot`` must be passed to :func:`finalize_tree_snapshot`
    once you're done (whether or not you called ``label_folders``) — nothing
    is written to disk until then, since saving before labeling would lose
    every newly-assigned label.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    snapshot_dir = Path(snapshot_dir) if snapshot_dir is not None else TREE_SNAPSHOT_DIR

    params = {
        "max_depth": max_depth,
        "max_files": max_files,
        "respect_gitignore": respect_gitignore,
    }
    snapshot = _load_snapshot(root, params, snapshot_dir) if incremental else None
    had_prior_snapshot = snapshot is not None
    ctx = _WalkContext(
        old_signatures=snapshot["signatures"] if snapshot else {},
        old_labels=snapshot["labels"] if snapshot else {},
    )

    spec = _load_gitignore_spec(root) if respect_gitignore else None
    root_node, _root_sig = _build_node(root, root, spec, max_depth, max_files, depth=0, ctx=ctx)
    root_node.type = NodeType.root
    mm = MindMap(title=root.name or str(root), root=root_node, source=str(root), level="full")

    deleted = sorted(set(ctx.old_signatures) - set(ctx.new_signatures))
    pending = PendingSnapshot(
        root=root,
        params=params,
        signatures=ctx.new_signatures,
        node_by_relpath=ctx.node_by_relpath,
        snapshot_dir=snapshot_dir,
    )

    if not had_prior_snapshot:
        return mm, None, ctx.nodes_needing_labels, pending

    diff = TreeDiff(
        reused=ctx.reused,
        added=ctx.added,
        changed=ctx.changed,
        deleted=deleted,
        previous_built_at=snapshot.get("built_at"),
    )
    return mm, diff, ctx.nodes_needing_labels, pending


def label_folders(
    mm: MindMap,
    provider: LLMProvider,
    cache: Cache,
    nodes: list[Node] | None = None,
    max_workers: int = 6,
    on_event: Callable[..., None] | None = None,
) -> None:
    """Optionally enrich each folder node with an inferred one-line purpose,
    based on its name and immediate contents. The folder's own name is left
    as the node title (still the most useful thing for navigation) — the
    inferred purpose goes in ``note`` instead, so both survive to the output.

    ``nodes``, when given, restricts labeling to exactly those nodes (used
    for incremental rebuilds, where unchanged folders already carry a label
    from a previous run and shouldn't be redundantly relabeled). Defaults to
    every folder in the tree.
    """
    on_event = on_event or (lambda *a, **k: None)
    folders = nodes if nodes is not None else [n for n in mm.root.walk() if n.type == NodeType.topic]
    if not folders:
        return

    on_event("label_start", total=len(folders))

    def _label_one(node: Node) -> None:
        listing = [c.title for c in node.children[:25]]
        payload = {"folder": node.title, "contents": listing}
        user = json.dumps(payload, ensure_ascii=False)
        key = Cache.key(provider.name, provider.model, PROMPT_VERSION, "folder_label", user)
        result = cache.get(key)
        if result is None:
            try:
                result = provider.complete_json(FOLDER_LABEL_SYSTEM, user)
            except LLMError:
                return
            cache.set(key, result)
        label = str(result.get("label", "")).strip()
        if label:
            node.note = label

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_label_one, n) for n in folders]
        done = 0
        for fut in as_completed(futures):
            fut.result()
            done += 1
            on_event("label_progress", done=done, total=len(folders))
