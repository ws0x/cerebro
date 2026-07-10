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


def load_config(config_dir: Path | None = None) -> dict[str, str | int]:
    # config_dir defaults via a None sentinel, not `= CONFIG_DIR` in the
    # signature — a default bound at def-time would freeze in the value
    # CONFIG_DIR had at import time, silently ignoring the module-level
    # monkeypatch("cerebro.paths.CONFIG_DIR", ...) pattern tests rely on.
    import json
    if config_dir is None:
        config_dir = CONFIG_DIR
    path = config_dir / "config.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_config(data: dict, config_dir: Path | None = None) -> Path:
    import json
    if config_dir is None:
        config_dir = CONFIG_DIR
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
