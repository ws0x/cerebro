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


def ensure_output_dir() -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR
