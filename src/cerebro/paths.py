"""Well-known filesystem locations for cerebro's global config and default output.

A globally-installed CLI can be invoked from any directory, so anything tied
to the current working directory — a project-local ``.env``, a bare output
filename — silently breaks the moment you're not standing in a specific
folder. These give both a stable, predictable home instead, while a
cwd-local ``.env`` (see :func:`cerebro.llm.config.load_env`) still works too
and takes priority, for repo/dev use with a project-specific key.
"""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".cerebro"
GLOBAL_ENV_PATH = CONFIG_DIR / ".env"
CACHE_DIR = CONFIG_DIR / "cache"
TREE_SNAPSHOT_DIR = CONFIG_DIR / "tree-snapshots"
BATCH_SNAPSHOT_DIR = CONFIG_DIR / "batch-snapshots"


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


def _resolve_default_output_dir() -> Path:
    """Precedence: CEREBRO_OUTPUT_DIR env var > "output_dir" persisted in
    ~/.cerebro/config.json (set via `cerebro config set output_dir ...`) >
    ~/cerebro-maps, the generic default every fresh install gets out of the
    box. Previously this hardcoded one person's synced-cloud-drive folder as
    the primary default for every install -- a real path on this machine,
    not a sensible one for anyone else's. Anyone who wants a custom location
    (a cloud-synced drive, a specific project folder) now opts into it via
    config or the env var instead of it being baked into the source."""
    env = os.environ.get("CEREBRO_OUTPUT_DIR")
    if env:
        return Path(env)
    configured = load_config().get("output_dir")
    if configured:
        return Path(configured)
    return Path.home() / "cerebro-maps"


DEFAULT_OUTPUT_DIR = _resolve_default_output_dir()
_FALLBACK_OUTPUT_DIR = Path.home() / "cerebro-maps"


def ensure_output_dir() -> Path:
    """DEFAULT_OUTPUT_DIR may point at a synced cloud-drive letter (or any
    other configured location) that isn't reachable right now (sync client
    not running, different machine, drive letter never existed here) --
    mkdir-ing into it raises, and that's not a reason for the whole command
    to crash. Falls back to ~/cerebro-maps in that case, so a build still
    lands somewhere real instead of failing outright -- but the user should
    still be told, since silently landing somewhere else could otherwise
    look like the file went missing."""
    try:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return DEFAULT_OUTPUT_DIR
    except OSError:
        _FALLBACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from .console import json_mode, qprint

            if not json_mode():
                qprint(
                    f"[yellow]![/] {DEFAULT_OUTPUT_DIR} isn't reachable right now "
                    f"(drive not mounted?) — saving to {_FALLBACK_OUTPUT_DIR} instead."
                )
        except Exception:
            pass
        return _FALLBACK_OUTPUT_DIR
