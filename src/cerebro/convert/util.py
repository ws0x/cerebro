"""Shared converter helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..ir import Node


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def note_for(node: Node) -> str:
    """Combine an optional timestamp marker with the node's note into one string."""
    parts: list[str] = []
    if node.timestamp is not None:
        parts.append(f"[{format_timestamp(node.timestamp)}]")
    if node.note:
        parts.append(node.note.strip())
    return " ".join(parts).strip()


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write a file without ever leaving a truncated/corrupt file at `path`.

    `write_fn(tmp_path)` performs the actual write into a temp file created
    in the same directory as `path` (so the final `os.replace` is an atomic
    rename on the same filesystem, not a cross-device copy). If `write_fn`
    raises, or the process is killed mid-write, `path` itself is never
    touched -- either the old file (if any) survives untouched, or nothing
    is created at all.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
