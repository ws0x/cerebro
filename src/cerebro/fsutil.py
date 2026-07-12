"""Generic filesystem helpers shared across converters, batch, and foldermap."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


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
