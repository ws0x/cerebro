"""Provider resolution + .env loading.

Keys are read from the environment (or a local .env). They are never written to
code or committed. ``resolve_provider`` maps an engine name to a ready provider,
with helpful errors pointing at where to get a free key.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..paths import GLOBAL_ENV_PATH
from .providers import GeminiProvider, GroqProvider, MockProvider


class ConfigError(RuntimeError):
    pass


def read_env_file(path: str | Path) -> dict[str, str]:
    """Parse a .env file into a dict. Returns {} if it doesn't exist."""
    path = Path(path)
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def write_env_file(path: str | Path, data: dict[str, str]) -> None:
    """Write ``data`` as a .env file, one KEY=value per line. Overwrites
    whatever was there — callers that want to preserve existing keys should
    read_env_file() first and merge, which is exactly what `cerebro setup`
    does so a blank answer for one key doesn't erase an already-saved other.

    ``newline="\\n"`` pins the line ending explicitly — Path.write_text()'s
    default text-mode writing silently translates every "\\n" to "\\r\\n" on
    Windows, which would otherwise flip an existing LF .env to CRLF on the
    very first `cerebro setup` run for no functional reason.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in data.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")


def load_env(*extra_paths: str | Path) -> None:
    """Minimal .env loader (no dependency). Only sets vars not already set.

    Checks the current directory's ``.env`` first (project-local override,
    useful when developing cerebro itself or wanting a per-project key), then
    falls back to the global ``~/.cerebro/.env`` — the one that actually
    matters once cerebro is installed globally and run from anywhere.
    """
    candidates = [Path.cwd() / ".env", GLOBAL_ENV_PATH, *(Path(p) for p in extra_paths)]
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        for key, value in read_env_file(path).items():
            os.environ.setdefault(key, value)


def _groq_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def _gemini_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def resolve_provider(engine: str, model: str | None = None):
    """Return a provider for ``engine`` or ``None`` when the caller should fall
    back to the heuristic structurer (engine is 'heuristic', or 'auto' with no
    keys available)."""
    engine = (engine or "auto").lower()

    if engine == "heuristic":
        return None
    if engine == "mock":
        return MockProvider()

    if engine == "auto":
        if _groq_key():
            return GroqProvider(_groq_key(), model or "llama-3.3-70b-versatile")
        if _gemini_key():
            return GeminiProvider(_gemini_key(), model or "gemini-2.5-flash")
        return None  # no keys -> heuristic fallback

    if engine == "groq":
        key = _groq_key()
        if not key:
            raise ConfigError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                "and put it in a .env file or your environment."
            )
        return GroqProvider(key, model or "llama-3.3-70b-versatile")

    if engine == "gemini":
        key = _gemini_key()
        if not key:
            raise ConfigError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in a .env file."
            )
        return GeminiProvider(key, model or "gemini-2.5-flash")

    raise ConfigError(f"Unknown engine: {engine!r} (use auto|groq|gemini|mock|heuristic)")
