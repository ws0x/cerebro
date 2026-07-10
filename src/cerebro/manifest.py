"""Local memory of "have I already built a map from this exact source at
this level" for the single-item `cerebro map` command.

Distinct from the LLM response cache (``cache.py``): that avoids re-paying
for individual MAP/REDUCE/LINK calls on a rerun, but nothing previously told
the *user* they'd already done this before -- no way to notice "didn't I map
this last week?" short of remembering or searching the output folder
yourself. This is purely informational, never blocking: reruns are
legitimate (source edited, wanting a different engine's take, etc.), so a
hit here prints a note and lets the build proceed regardless.

Same architectural posture as ``batch.py``'s and ``foldermap.py``'s own
snapshot files -- a small JSON blob under ``~/.cerebro/``, best-effort reads
and writes that never turn into a crash or a blocked build.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .paths import CONFIG_DIR

MAP_MANIFEST_PATH = CONFIG_DIR / "map-manifest.json"


def _normalize_source(source: str) -> str:
    # Local paths are resolved to absolute so "video.mp4" and "./video.mp4"
    # (or the same file reached from a different cwd) hash identically;
    # URLs have no filesystem meaning and are left exactly as given.
    try:
        path = Path(source)
        if path.exists():
            return str(path.resolve())
    except Exception:
        pass
    return source


def _key(source: str, level: str, fmt: str) -> str:
    normalized = f"{_normalize_source(source)}|{level}|{fmt}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def lookup(source: str, level: str, fmt: str, manifest_path: Path | None = None) -> dict | None:
    """The previous build's record for this exact (source, level, fmt), if
    any -- ``{"output": str, "engine": str, "built_at": str}``."""
    manifest_path = manifest_path or MAP_MANIFEST_PATH
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data.get(_key(source, level, fmt))


def record(
    source: str, level: str, fmt: str, engine_label: str, output: Path, manifest_path: Path | None = None
) -> None:
    manifest_path = manifest_path or MAP_MANIFEST_PATH
    try:
        data = {}
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except Exception:
                data = {}
        data[_key(source, level, fmt)] = {
            "source": source,
            "output": str(output),
            "engine": engine_label,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # a manifest write failure must never block a successful build
