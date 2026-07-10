"""Environment diagnostics: what's set up, what's missing, what to do about it.

Read-only except for a throwaway file used to confirm each directory is
actually writable, not just present. Every check degrades independently — a
missing optional piece (Whisper, a second engine's key, network) is reported
as an advisory ("warn"), not a failure; only things that would break every
command outright (a core dependency failing to import, a storage directory
that can't be written to, an unsupported Python version) are hard failures.
Storage paths are injectable so tests never touch the real ``~/.cerebro``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .cache import Cache
from .paths import (
    BATCH_SNAPSHOT_DIR,
    CACHE_DIR,
    CONFIG_DIR,
    DEFAULT_OUTPUT_DIR,
    GLOBAL_ENV_PATH,
    TREE_SNAPSHOT_DIR,
)

Status = str  # "ok" | "warn" | "fail"


@dataclass
class Check:
    group: str
    label: str
    status: Status
    detail: str
    fix: str | None = None


def _check_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".cerebro-doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except Exception as exc:
        return False, f"{path} — {exc}"


def _reachable(host: str, port: int = 443, timeout: float = 2.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def run_diagnostics(
    check_network: bool = True,
    config_dir: Path = CONFIG_DIR,
    cache_dir: Path = CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tree_snapshot_dir: Path = TREE_SNAPSHOT_DIR,
    batch_snapshot_dir: Path = BATCH_SNAPSHOT_DIR,
    global_env_path: Path = GLOBAL_ENV_PATH,
) -> list[Check]:
    checks: list[Check] = []

    # --- Environment ---------------------------------------------------
    py_ok = sys.version_info >= (3, 10)
    checks.append(Check(
        "Environment", "Python version", "ok" if py_ok else "fail",
        sys.version.split()[0],
        None if py_ok else "cerebro needs Python >= 3.10",
    ))
    checks.append(Check("Environment", "cerebro version", "ok", __version__))
    checks.append(Check("Environment", "Platform", "ok", sys.platform))

    # --- Engines ---------------------------------------------------------
    # Assumes load_env() has already run (the CLI's root callback runs it for
    # every invocation) — calling it again here would read the real
    # ~/.cerebro/.env directly, unable to honor an injected global_env_path
    # or a test's monkeypatched environment.
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    checks.append(Check(
        "Engines", "Groq API key", "ok" if groq_key else "warn",
        "found" if groq_key else "not set",
        None if groq_key else "Free key: https://console.groq.com/keys",
    ))
    checks.append(Check(
        "Engines", "Gemini API key", "ok" if gemini_key else "warn",
        "found" if gemini_key else "not set",
        None if gemini_key else "Free key: https://aistudio.google.com/apikey",
    ))
    if not groq_key and not gemini_key:
        checks.append(Check(
            "Engines", "AI structuring", "warn",
            "no key set — map/batch/tree will fall back to the offline heuristic engine",
            "Set GROQ_API_KEY or GEMINI_API_KEY to get AI-structured maps",
        ))
    env_found = [p for p in (Path.cwd() / ".env", global_env_path) if p.exists()]
    checks.append(Check(
        "Engines", ".env file(s)", "ok" if env_found else "warn",
        ", ".join(str(p) for p in env_found) if env_found else "none found",
        None if env_found else f"Create {global_env_path} with your API key(s)",
    ))

    if check_network:
        for host, label, group in (
            ("api.groq.com", "Groq API reachable", "Engines"),
            ("generativelanguage.googleapis.com", "Gemini API reachable", "Engines"),
            ("www.youtube.com", "YouTube reachable", "Sources"),
        ):
            ok = _reachable(host)
            checks.append(Check(
                group, label, "ok" if ok else "warn",
                "reachable" if ok else "unreachable (offline, or blocked by a firewall/proxy)",
            ))

    # --- Local video -----------------------------------------------------
    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(Check(
        "Local video", "ffmpeg", "ok" if ffmpeg_path else "warn",
        ffmpeg_path or "not found on PATH",
        None if ffmpeg_path else "Required for local video files: https://ffmpeg.org/download.html",
    ))
    ffprobe_path = shutil.which("ffprobe")
    checks.append(Check(
        "Local video", "ffprobe", "ok" if ffprobe_path else "warn",
        ffprobe_path or "not found on PATH",
        None if ffprobe_path else "Ships with ffmpeg — the same install fixes this.",
    ))
    whisper_ok = _importable("faster_whisper")
    checks.append(Check(
        "Local video", "faster-whisper", "ok" if whisper_ok else "warn",
        "installed" if whisper_ok else "not installed (only needed for videos with no subtitle track)",
        None if whisper_ok else "pip install cerebro[whisper]",
    ))

    # --- Storage -----------------------------------------------------------
    for label, path in (
        ("Config dir", config_dir),
        ("Cache dir", cache_dir),
        ("Output dir", output_dir),
        ("Tree snapshots dir", tree_snapshot_dir),
        ("Batch snapshots dir", batch_snapshot_dir),
    ):
        ok, detail = _check_writable(path)
        checks.append(Check(
            "Storage", label, "ok" if ok else "fail", detail,
            None if ok else "Check filesystem permissions for this path.",
        ))

    cache = Cache(root=cache_dir)
    count, total_bytes = cache.stats()
    checks.append(Check("Storage", "Response cache", "ok", f"{count} entries"))
    tree_snaps = len(list(tree_snapshot_dir.glob("*.json"))) if tree_snapshot_dir.exists() else 0
    checks.append(Check("Storage", "Tree snapshots", "ok", f"{tree_snaps} folder(s) mapped before"))
    batch_snaps = len(list(batch_snapshot_dir.glob("*.json"))) if batch_snapshot_dir.exists() else 0
    checks.append(Check("Storage", "Batch snapshots", "ok", f"{batch_snaps} playlist/course(s) run before"))

    # --- Dependencies --------------------------------------------------
    for mod, label in (
        ("yt_dlp", "yt-dlp"),
        ("youtube_transcript_api", "youtube-transcript-api"),
        ("questionary", "questionary"),
        ("pathspec", "pathspec"),
        ("pydantic", "pydantic"),
    ):
        ok = _importable(mod)
        checks.append(Check(
            "Dependencies", label, "ok" if ok else "fail",
            "installed" if ok else "missing",
            None if ok else f"pip install {mod.replace('_', '-')}",
        ))

    return checks


def has_failures(checks: list[Check]) -> bool:
    return any(c.status == "fail" for c in checks)
