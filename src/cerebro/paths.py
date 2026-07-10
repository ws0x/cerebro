"""Well-known filesystem locations for cerebro's global config and default output.

A globally-installed CLI can be invoked from any directory, so anything tied
to the current working directory — a project-local ``.env``, a bare output
filename — silently breaks the moment you're not standing in a specific
folder. These give both a stable, predictable home instead, while a
cwd-local ``.env`` (see :func:`cerebro.llm.config.load_env`) still works too
and takes priority, for repo/dev use with a project-specific key.
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path.home() / ".cerebro"
GLOBAL_ENV_PATH = CONFIG_DIR / ".env"
DEFAULT_OUTPUT_DIR = Path.home() / "cerebro-maps"
CACHE_DIR = CONFIG_DIR / "cache"
TREE_SNAPSHOT_DIR = CONFIG_DIR / "tree-snapshots"
BATCH_SNAPSHOT_DIR = CONFIG_DIR / "batch-snapshots"


def ensure_output_dir() -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR


def load_config() -> dict[str, str | int]:
    import json
    path = CONFIG_DIR / "config.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}
