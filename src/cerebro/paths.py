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
PACING_PATH = CONFIG_DIR / "pacing.json"
QUOTA_PATH = CONFIG_DIR / "quota.json"


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


# NOTE: DEFAULT_OUTPUT_DIR is kept as a module-level name for
# backward-compat (doctor.py imports it), but it is NOT used inside
# ensure_output_dir() itself.  Resolving it once at import time means a
# freshly-installed pipx venv — or a process that ran before config.json was
# written — would cache the wrong path for the lifetime of the process.
# ensure_output_dir() re-resolves on every call so it always honours the
# current config.json and env-var state.
DEFAULT_OUTPUT_DIR = _resolve_default_output_dir()
_FALLBACK_OUTPUT_DIR = Path.home() / "cerebro-maps"


def ensure_output_dir() -> Path:
    """Resolves and creates the correct output directory on every call.

    Re-calls ``_resolve_default_output_dir()`` each time so that changes to
    ``~/.cerebro/config.json`` (e.g. ``cerebro config set output_dir …``) or
    the ``CEREBRO_OUTPUT_DIR`` env var are always honoured — even if this
    process was started before the config file existed (e.g. a freshly
    installed pipx venv used before the first ``config set`` ran).

    Falls back to ``~/cerebro-maps`` if the configured directory isn't
    reachable right now (drive not mounted, sync client offline, etc.) rather
    than crashing the whole command, but warns the user so the unexpected
    landing location isn't silent.
    """
    target = _resolve_default_output_dir()
    fallback = _FALLBACK_OUTPUT_DIR
    try:
        target.mkdir(parents=True, exist_ok=True)
        return target
    except OSError:
        fallback.mkdir(parents=True, exist_ok=True)
        try:
            from .console import json_mode, qprint

            if not json_mode():
                qprint(
                    f"[yellow]![/] {target} isn't reachable right now "
                    f"(drive not mounted?) — saving to {fallback} instead."
                )
        except Exception:
            pass
        return fallback
