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
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .cache import Cache
from .ir import MindMap, Node, NodeType
from .llm.base import LLMError, LLMProvider
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


def _build_node(path: Path, root: Path, spec, max_depth: int, max_files: int, depth: int) -> Node | None:
    if _is_ignored_name(path.name):
        return None
    if spec is not None:
        try:
            rel = str(path.relative_to(root).as_posix())
            if path.is_dir():
                rel += "/"
            if spec.match_file(rel):
                return None
        except ValueError:
            pass

    if path.is_file():
        node_type = NodeType.definition if _is_notable_file(path.name) else NodeType.detail
        return Node(title=path.name, type=node_type)

    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except (PermissionError, OSError):
        return Node(title=path.name, type=NodeType.topic, note="(could not be read)")

    node = Node(title=path.name, type=NodeType.topic)

    if depth >= max_depth:
        try:
            total = sum(1 for _ in path.rglob("*"))
        except (PermissionError, OSError):
            total = 0
        if total:
            node.add(f"+{total} more item(s)", type=NodeType.detail)
        return node

    children: list[Node] = []
    for entry in entries:
        child = _build_node(entry, root, spec, max_depth, max_files, depth + 1)
        if child is not None:
            children.append(child)

    dirs = [c for c in children if c.type == NodeType.topic]
    files = [c for c in children if c.type != NodeType.topic]
    total_files = len(files)

    if total_files > max_files:
        remaining = total_files - max_files
        files = files[:max_files] + [Node(title=f"+{remaining} more file(s)", type=NodeType.detail)]

    node.children = dirs + files
    if dirs or total_files:
        node.note = f"{len(dirs)} folder(s), {total_files} file(s)"
    return node


def build_folder_map(
    root: str | Path,
    max_depth: int = 8,
    max_files: int = 20,
    respect_gitignore: bool = True,
) -> MindMap:
    """Walk ``root`` and build a MindMap of its directory structure."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    spec = _load_gitignore_spec(root) if respect_gitignore else None
    root_node = _build_node(root, root, spec, max_depth, max_files, depth=0)
    root_node.type = NodeType.root
    return MindMap(title=root.name or str(root), root=root_node, source=str(root), level="full")


def label_folders(
    mm: MindMap,
    provider: LLMProvider,
    cache: Cache,
    max_workers: int = 6,
    on_event: Callable[..., None] | None = None,
) -> None:
    """Optionally enrich each folder node with an inferred one-line purpose,
    based on its name and immediate contents. The folder's own name is left
    as the node title (still the most useful thing for navigation) — the
    inferred purpose goes in ``note`` instead, so both survive to the output.
    """
    on_event = on_event or (lambda *a, **k: None)
    folders = [n for n in mm.root.walk() if n.type == NodeType.topic]
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
