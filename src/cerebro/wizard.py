"""The guided interactive wizard: source -> level -> engine -> format -> output.

Kept separate from ``cli.py`` because it's a substantial, self-contained piece
of UI. It takes ``do_map``/``do_batch`` as callables rather than importing
them, so there's no import cycle with ``cli.py`` (which imports this module).

Every prompt has two backends: ``questionary`` (arrow-key menus, needs a real
attached console) when one is actually available, and a plain Rich-based
prompt otherwise. The fallback isn't just for piped/non-interactive input —
questionary/prompt_toolkit can fail on legitimate terminals too (e.g. certain
Windows terminal/shell combinations), so every questionary call is wrapped and
falls back automatically on *any* exception. This path is never allowed to
crash the wizard; at worst it gets less pretty.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import questionary
import typer
from questionary import Choice, Style
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from .ingest import looks_like_youtube
from .ingest.playlist import is_playlist_url
from .paths import ensure_output_dir

console = Console()

# Matches Rich's cyan accent used everywhere else in the CLI, so the arrow-key
# menus don't feel like a different tool bolted on.
_QSTYLE = Style(
    [
        ("qmark", "fg:#29b6c9 bold"),
        ("question", "bold"),
        ("answer", "fg:#29b6c9 bold"),
        ("pointer", "fg:#29b6c9 bold"),
        ("highlighted", "fg:#29b6c9 bold"),
        ("selected", "fg:#29b6c9"),
    ]
)

# Piped/pasted input (especially from PowerShell) can carry a leading BOM or
# other zero-width characters that .strip() doesn't touch — left alone, these
# survive into the source path/URL and corrupt Rich's panel width calculation.
_INVISIBLE_RE = re.compile("[﻿​‌‍⁠]")


def _clean(text: str) -> str:
    return _INVISIBLE_RE.sub("", text).strip()


def _cancel() -> None:
    console.print("[dim]Cancelled.[/]")
    raise typer.Exit()


def _has_real_console() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _ask_text(message: str, default: str | None = None) -> str:
    if _has_real_console():
        try:
            result = questionary.text(message, default=default or "", style=_QSTYLE).ask()
            if result is None:
                _cancel()
            return _clean(result)
        except typer.Exit:
            raise
        except Exception:
            pass  # fall through to the Rich prompt below
    kwargs = {"default": default} if default is not None else {}
    return _clean(Prompt.ask(f"[cyan]{message.rstrip(':')}[/]", **kwargs))


def _select(message: str, choices: list[Choice], default: str | None = None) -> str:
    if _has_real_console():
        try:
            result = questionary.select(message, choices=choices, default=default, style=_QSTYLE).ask()
            if result is None:
                _cancel()
            return result
        except typer.Exit:
            raise
        except Exception:
            pass  # fall through to the Rich prompt below

    console.print(f"[cyan]{message.rstrip(':')}[/]")
    values = [c.value for c in choices]
    for c in choices:
        console.print(f"  [bold]{c.value}[/]  {c.title}")
    return Prompt.ask("  →", choices=values, default=default, show_choices=False)


def detect_source_kind(source: str) -> str:
    if is_playlist_url(source):
        return "playlist"
    if looks_like_youtube(source):
        return "youtube"
    if Path(source).is_dir():
        return "folder"
    if Path(source).exists():
        return "file"
    return "unknown"


_KIND_LABEL = {
    "playlist": "YouTube playlist (batch)",
    "youtube": "YouTube video",
    "folder": "local course folder (batch)",
    "file": "local file",
}

_LEVEL_CHOICES = [
    Choice("Brief — main topics only, fastest", value="brief"),
    Choice("Full — subtopics + key points (recommended)", value="full"),
    Choice("Expert — + relationships & cross-links, deepest", value="expert"),
]

_ENGINE_CHOICES = [
    Choice("Auto — Groq/Gemini if a key is set, else offline (recommended)", value="auto"),
    Choice("Groq — free tier, fastest", value="groq"),
    Choice("Gemini — free tier, sticks closer to the source wording", value="gemini"),
    Choice("Heuristic — fully offline, no AI, no key needed", value="heuristic"),
]


def _ask_source() -> tuple[str, str]:
    while True:
        source = _ask_text("Paste a YouTube URL, playlist URL, or local file/folder path:")
        kind = detect_source_kind(source)
        if kind == "unknown":
            console.print(f"[red]✗ Not a recognizable URL or existing path: {source}[/]\n")
            continue
        console.print(f"[green]✓[/] Detected: [bold]{_KIND_LABEL[kind]}[/]")
        return source, kind


def _ask_level() -> str:
    return _select("Processing level:", _LEVEL_CHOICES, default="full")


def _ask_engine() -> str:
    return _select("Engine:", _ENGINE_CHOICES, default="auto")


def _ask_format(level: str) -> str:
    default = "xmind" if level == "expert" else "opml"
    hint = (
        "expert level has relationships — xmind keeps them, opml would drop them"
        if level == "expert"
        else "opml imports everywhere; xmind is native and keeps icons"
    )
    console.print(f"[dim]  {hint}[/]")
    return _select(
        "Output format:",
        [
            Choice("OPML — imports into XMind, Freemind, most outliners", value="opml"),
            Choice("XMind — native file, keeps relationships & markers", value="xmind"),
        ],
        default=default,
    )


def _ask_output(fmt: str) -> Path:
    default = str(ensure_output_dir() / f"mindmap.{fmt}")
    return Path(_ask_text("Output path:", default=default))


def _summary_panel(source, kind, level, engine, fmt, out) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_row("[dim]Source[/]", source)
    table.add_row("[dim]Type[/]", _KIND_LABEL[kind])
    table.add_row("[dim]Level[/]", level)
    table.add_row("[dim]Engine[/]", engine)
    table.add_row("[dim]Format[/]", fmt.upper())
    table.add_row("[dim]Output[/]", str(out))
    return Panel(table, title="[cyan]Ready[/]", border_style="cyan", expand=False)


def run_wizard(
    do_map: Callable[[str, str, str, Path, str, bool, bool], None],
    do_batch: Callable[[str, str, str, Path, str, int, int | None, bool, bool], None],
) -> None:
    console.print(Rule("[bold cyan]Source[/]", style="cyan"))
    console.print("[dim]Ctrl+C to cancel anytime[/]\n")
    source, kind = _ask_source()

    console.print()
    console.print(Rule("[bold cyan]Options[/]", style="cyan"))
    level = _ask_level()
    engine = _ask_engine()
    fmt = _ask_format(level)
    out = _ask_output(fmt)

    while True:
        console.print()
        console.print(Rule(style="cyan"))
        console.print(_summary_panel(source, kind, level, engine, fmt, out))

        choice = _select(
            "Proceed?",
            [
                Choice("Yes, build it", value="go"),
                Choice("Edit an answer", value="edit"),
                Choice("Cancel", value="cancel"),
            ],
            default="go",
        )
        if choice == "cancel":
            _cancel()
        if choice == "go":
            break

        field = _select(
            "What would you like to change?",
            [
                Choice("Source", value="source"),
                Choice("Level", value="level"),
                Choice("Engine", value="engine"),
                Choice("Format", value="format"),
                Choice("Output path", value="output"),
            ],
        )
        console.print()
        if field == "source":
            source, kind = _ask_source()
        elif field == "level":
            level = _ask_level()
        elif field == "engine":
            engine = _ask_engine()
        elif field == "format":
            fmt = _ask_format(level)
        elif field == "output":
            out = _ask_output(fmt)

    console.print()
    if kind in ("playlist", "folder"):
        do_batch(source, level, fmt, out, engine, 3, None, False, True)
    else:
        do_map(source, level, fmt, out, engine, False, True)
