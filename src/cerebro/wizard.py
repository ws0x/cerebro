"""The guided interactive wizard: mode -> source -> level -> engine -> format -> output.

Kept separate from ``cli.py`` because it's a substantial, self-contained piece
of UI. It takes ``do_map``/``do_batch``/``do_tree`` as callables rather than
importing them, so there's no import cycle with ``cli.py`` (which imports
this module).

Every prompt has two backends: ``questionary`` (arrow-key menus, path
autocomplete, needs a real attached console) when one is actually available,
and a plain Rich-based prompt otherwise. The fallback isn't just for
piped/non-interactive input — questionary/prompt_toolkit can fail on
legitimate terminals too (e.g. certain Windows terminal/shell combinations),
so every questionary call is wrapped and falls back automatically on *any*
exception. This path is never allowed to crash the wizard; at worst it gets
less pretty.

**Navigation.** Every step accepts going back to the previous one -- typing
``back`` on a text/path prompt, or picking "← Back" from a menu -- without
losing anything already entered elsewhere; the previous answer for that step
becomes the new prompt's default, ready to edit rather than retype. If the
final build fails partway (bad URL, network blip, an output path that
collides), the already-answered source/level/engine/format/output are never
thrown away either -- see the retry loop at the bottom of ``run_wizard``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import questionary
import typer
from questionary import Choice, Style
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table

from .clipboard import suggest_for_mode
from .console import console, has_real_console
from .ingest import looks_like_youtube
from .ingest.playlist import is_playlist_url
from .paths import ensure_output_dir, load_config, save_config

# Matches Rich's cyan accent used everywhere else in the CLI, so the arrow-key
# menus don't feel like a different tool bolted on.
_QSTYLE = Style(
    [
        ("qmark", "fg:#29b6c9 bold"),
        ("question", "bold"),
        ("answer", "fg:#29b6c9 bold"),
        ("pointer", "fg:#ff2a7f bold"),
        ("highlighted", "fg:#ff2a7f bold"),
        ("selected", "fg:#ff2a7f"),
    ]
)

# Piped/pasted input (especially from PowerShell) can carry a leading BOM or
# other zero-width characters that .strip() doesn't touch — left alone, these
# survive into the source path/URL and corrupt Rich's panel width calculation.
_INVISIBLE_RE = re.compile("[﻿​‌‍⁠]")

# Returned by any ask-step when the user wants the previous step back instead
# of a real answer. A distinct sentinel object (not a string like "back")
# so it can never collide with a legitimately typed value.
_BACK = object()
_BACK_VALUE = "back"  # what a "← Back" menu Choice's value / typed keyword is


def _clean(text: str) -> str:
    return _INVISIBLE_RE.sub("", text).strip()


def _cancel() -> None:
    console.print("[dim]Cancelled.[/]")
    raise typer.Exit()


# Printed right before each individual prompt, not once at the wizard's
# start — a hint that scrolled off-screen three steps ago doesn't help
# anyone. The two variants match what's actually available in each backend:
# arrow-key navigation only applies to the real questionary path, and the
# Rich fallback's choices are already spelled out as literal values.
_HINT_TEXT = "[dim]Enter to confirm · Ctrl+C to cancel[/]"
_HINT_TEXT_BACK = "[dim]Enter to confirm · type 'back' for the previous step · Ctrl+C to cancel[/]"
_HINT_PATH = "[dim]Tab to autocomplete · Enter to confirm · Ctrl+C to cancel[/]"
_HINT_PATH_BACK = "[dim]Tab to autocomplete · Enter to confirm · type 'back' for the previous step · Ctrl+C to cancel[/]"
_HINT_SELECT = "[dim]↑↓ navigate · Enter select · Ctrl+C cancel[/]"
_HINT_FALLBACK = "[dim]Ctrl+C to cancel[/]"
_HINT_FALLBACK_BACK = "[dim]Type 'back' for the previous step · Ctrl+C to cancel[/]"


def _ask_text(message: str, default: str | None = None, allow_back: bool = False) -> str:
    if has_real_console():
        try:
            console.print(_HINT_TEXT_BACK if allow_back else _HINT_TEXT)
            result = questionary.text(message, default=default or "", style=_QSTYLE).ask()
            if result is None:
                _cancel()
            result = _clean(result)
            return _BACK if allow_back and result.lower() == _BACK_VALUE else result
        except typer.Exit:
            raise
        except Exception:
            pass  # fall through to the Rich prompt below
    console.print(_HINT_FALLBACK_BACK if allow_back else _HINT_FALLBACK)
    kwargs = {"default": default} if default is not None else {}
    result = _clean(Prompt.ask(f"[cyan]{message.rstrip(':')}[/]", **kwargs))
    return _BACK if allow_back and result.lower() == _BACK_VALUE else result


def _ask_path(
    message: str,
    default: str | None = None,
    only_directories: bool = False,
    file_filter: Callable[[str], bool] | None = None,
    allow_back: bool = False,
) -> str:
    """Like ``_ask_text``, but backed by ``questionary.path`` on a real
    console — Tab-completion through the filesystem as you type, so typing a
    path separator lists what's actually in that folder instead of asking
    the user to remember or retype it. Falls back to a plain text prompt
    (no completion, same as before) everywhere ``_ask_text`` would."""
    if has_real_console():
        try:
            console.print(_HINT_PATH_BACK if allow_back else _HINT_PATH)
            result = questionary.path(
                message,
                default=default or "",
                only_directories=only_directories,
                file_filter=file_filter,
                style=_QSTYLE,
            ).ask()
            if result is None:
                _cancel()
            result = _clean(result)
            return _BACK if allow_back and result.lower() == _BACK_VALUE else result
        except typer.Exit:
            raise
        except Exception:
            pass  # fall through to the Rich prompt below
    console.print(_HINT_FALLBACK_BACK if allow_back else _HINT_FALLBACK)
    kwargs = {"default": default} if default is not None else {}
    result = _clean(Prompt.ask(f"[cyan]{message.rstrip(':')}[/]", **kwargs))
    return _BACK if allow_back and result.lower() == _BACK_VALUE else result


def _select(message: str, choices: list[Choice], default: str | None = None, allow_back: bool = False) -> str:
    all_choices = [*choices, Choice("← Back", value=_BACK_VALUE)] if allow_back else choices
    if has_real_console():
        try:
            console.print(_HINT_SELECT)
            result = questionary.select(message, choices=all_choices, default=default, style=_QSTYLE).ask()
            if result is None:
                _cancel()
            return _BACK if result == _BACK_VALUE else result
        except typer.Exit:
            raise
        except Exception:
            pass  # fall through to the Rich prompt below

    console.print(_HINT_FALLBACK)
    console.print(f"[cyan]{message.rstrip(':')}[/]")
    values = [c.value for c in all_choices]
    for c in all_choices:
        console.print(f"  [bold]{c.value}[/]  {c.title}")
    result = Prompt.ask("  →", choices=values, default=default, show_choices=False)
    return _BACK if result == _BACK_VALUE else result


# -- The four things cerebro can turn into a mind map -----------------------
# Explicit, not inferred: a folder full of course videos and a folder you
# just want organized as a tree map look identical as a bare path string, so
# guessing which one you meant is exactly the kind of "hidden capability"
# that leaves cerebro tree undiscoverable from the wizard. Asking up front
# instead means every one of cerebro's four real entry points is visible on
# the very first screen, not just the two (map/batch) the old free-text
# source prompt could actually reach.
_MODE_CHOICES = [
    Choice("YouTube video or playlist", value="youtube"),
    Choice("Local video or audio — single file or a folder of lessons", value="local_video"),
    Choice("PDF file", value="pdf"),
    Choice("Folder structure — map how it's organized, not its contents (cerebro tree)", value="tree"),
]

_MODE_LABEL = {
    "youtube": "YouTube video or playlist",
    "local_video": "local video/audio",
    "pdf": "PDF file",
    "tree": "folder structure",
}

_KIND_LABEL = {
    "playlist": "YouTube playlist (batch)",
    "youtube": "YouTube video",
    "folder": "local course folder (batch)",
    "file": "local file",
    "tree": "folder structure map",
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

_TREE_ENGINE_CHOICES = [
    Choice("Heuristic — no AI, folders unlabeled, instant (default)", value="heuristic"),
    Choice("Auto — Groq/Gemini if a key is set, else heuristic", value="auto"),
    Choice("Groq — free tier, AI-labels each folder's purpose", value="groq"),
    Choice("Gemini — free tier, AI-labels each folder's purpose", value="gemini"),
]

_SLUG_RE = re.compile(r"[^\w\- ]+")


def _slug(text: str) -> str:
    name = _SLUG_RE.sub("", text).strip().replace(" ", "_")
    return (name or "mindmap")[:80]


def _ask_mode(default: str | None = None) -> str:
    console.print("[dim]  Every one of these becomes a mind map — pick the one matching what you have.[/]")
    return _select("What are you turning into a mind map?", _MODE_CHOICES, default=default)


def _kind_for(mode: str, source: str) -> str:
    if mode == "youtube":
        return "playlist" if is_playlist_url(source) else "youtube"
    if mode == "tree":
        return "tree"
    if mode == "pdf":
        return "file"
    return "folder" if Path(source).is_dir() else "file"  # local_video


def _ask_source_for_mode(mode: str, default: str = "") -> str:
    suggestion = None if default else suggest_for_mode(mode)
    if suggestion:
        default = suggestion
        console.print(f"[dim]  Found in your clipboard — press Enter to use it, or type something else.[/]")

    while True:
        if mode == "youtube":
            result = _ask_text("Paste a YouTube video or playlist URL:", default=default, allow_back=True)
            if result is _BACK:
                return _BACK
            if not (looks_like_youtube(result) or is_playlist_url(result)):
                console.print(f"[red]✗ Doesn't look like a YouTube URL: {result}[/]\n")
                default = result
                continue
            return result

        if mode == "pdf":
            result = _ask_path(
                "Path to a PDF file:",
                default=default,
                file_filter=lambda p: p.lower().endswith(".pdf") or Path(p).is_dir(),
                allow_back=True,
            )
            if result is _BACK:
                return _BACK
            path = Path(result)
            if not path.exists():
                console.print(f"[red]✗ Not found: {result}[/]\n")
                default = result
                continue
            if path.is_dir():
                console.print(f"[red]✗ That's a folder, not a PDF file: {result}[/]\n")
                default = result
                continue
            if path.suffix.lower() != ".pdf":
                console.print(f"[red]✗ Not a .pdf file: {result}[/]\n")
                default = result
                continue
            return result

        if mode == "tree":
            result = _ask_path("Folder to map:", default=default, only_directories=True, allow_back=True)
            if result is _BACK:
                return _BACK
            path = Path(result)
            if not path.is_dir():
                console.print(f"[red]✗ Not a folder: {result}[/]\n")
                default = result
                continue
            return result

        # local_video
        result = _ask_path("Path to a video/audio file or a folder of lessons:", default=default, allow_back=True)
        if result is _BACK:
            return _BACK
        path = Path(result)
        if not path.exists():
            console.print(f"[red]✗ Not found: {result}[/]\n")
            default = result
            continue
        return result


def _ask_level(default: str = "full", allow_back: bool = False) -> str:
    return _select("Processing level:", _LEVEL_CHOICES, default=default, allow_back=allow_back)


def _ask_engine(default: str = "auto", allow_back: bool = False) -> str:
    return _select("Engine:", _ENGINE_CHOICES, default=default, allow_back=allow_back)


def _ask_tree_engine(default: str = "heuristic", allow_back: bool = False) -> str:
    console.print("[dim]  Structure is already known here — AI only ever labels folder purposes, never invents the tree.[/]")
    return _select("Engine:", _TREE_ENGINE_CHOICES, default=default, allow_back=allow_back)


def _ask_format(level: str, default: str | None = None, allow_back: bool = False) -> str:
    if default is None:
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
        allow_back=allow_back,
    )


def _default_output_path(source: str, kind: str, fmt: str) -> Path:
    """A same-named default beats a fixed "mindmap.<fmt>" every source
    always collides with — the exact trap that turns _export's overwrite
    prompt into a routine annoyance instead of the rare edge case it should
    be. YouTube sources keep the generic name: the real title isn't known
    until after the transcript is fetched, well past this prompt, and
    fetching early just to name a file isn't worth the extra latency."""
    stem = "mindmap"
    try:
        if kind in ("folder", "tree"):
            stem = _slug(Path(source).name)
        elif kind == "file":
            stem = _slug(Path(source).stem)
    except Exception:
        pass
    return ensure_output_dir() / f"{stem}.{fmt}"


def _ask_output(fmt: str, default: str, allow_back: bool = False) -> str:
    return _ask_path("Output path:", default=default, allow_back=allow_back)


def _resync_output_extension(out: Path | None, fmt: str) -> Path | None:
    """Format can change after the output path was already set -- the
    step-loop's own "back" to format-then-forward-to-output, or the confirm
    screen's "Edit an answer" -> Format. Left alone, the path keeps its old
    suffix and the file written at the end has the *wrong* extension for its
    actual on-disk content (e.g. XMind's zip archive saved as ``.opml``,
    which nothing can open correctly). The stem/directory the user chose is
    always kept -- only the suffix is corrected to match what's actually
    about to be written."""
    return out if out is None else out.with_suffix(f".{fmt}")


def _remember_last_answers(level: str, engine: str, fmt: str, tree_engine: str) -> None:
    """Persist the confirmed choices as the new defaults, so the next wizard
    run (and the next bare `map`/`batch`, which read the same config) starts
    from where this one left off instead of always resetting to
    full/auto/opml. Best-effort — a write failure here shouldn't block the
    build the user actually asked for."""
    try:
        config = load_config()
        config.update({"level": level, "engine": engine, "format": fmt, "tree_engine": tree_engine})
        save_config(config)
    except Exception:
        pass


def _summary_panel(mode, source, kind, level, engine, tree_engine, fmt, out) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_row("[dim]Source[/]", source)
    table.add_row("[dim]Type[/]", _KIND_LABEL[kind])
    if mode == "tree":
        table.add_row("[dim]Engine[/]", tree_engine)
    else:
        table.add_row("[dim]Level[/]", level)
        table.add_row("[dim]Engine[/]", engine)
    table.add_row("[dim]Format[/]", fmt.upper())
    table.add_row("[dim]Output[/]", str(out))
    return Panel(table, title="[cyan]Ready[/]", border_style="cyan", expand=False)


def _steps_for_mode(mode: str) -> list[str]:
    if mode == "tree":
        return ["source", "engine", "format", "output"]
    return ["source", "level", "engine", "format", "output"]


def run_wizard(
    do_map: Callable[..., None],
    do_batch: Callable[..., None],
    do_tree: Callable[..., None],
) -> None:
    config = load_config()
    level = str(config.get("level") or "full")
    engine = str(config.get("engine") or "auto")
    fmt = str(config.get("format") or ("xmind" if level == "expert" else "opml"))
    tree_engine = str(config.get("tree_engine") or "heuristic")

    console.print(Rule("[bold cyan]Source[/]", style="cyan"))
    mode = _ask_mode()

    source = ""
    kind: str | None = None
    out: Path | None = None
    steps = _steps_for_mode(mode)
    idx = 0
    printed_options_rule = False

    while idx < len(steps):
        step = steps[idx]

        if step == "source":
            result = _ask_source_for_mode(mode, default=source)
            if result is _BACK:
                old_mode = mode
                mode = _ask_mode(default=mode)
                if mode != old_mode:
                    source, kind, out = "", None, None
                steps = _steps_for_mode(mode)
                idx = 0
                continue
            source = result
            kind = _kind_for(mode, source)
            console.print(f"[green]✓[/] Detected: [bold]{_KIND_LABEL[kind]}[/]")
            if not printed_options_rule:
                console.print()
                console.print(Rule("[bold cyan]Options[/]", style="cyan"))
                printed_options_rule = True
            idx += 1
            continue

        if step == "level":
            result = _ask_level(default=level, allow_back=True)
            if result is _BACK:
                idx -= 1
                continue
            level = result
            idx += 1
            continue

        if step == "engine":
            if mode == "tree":
                result = _ask_tree_engine(default=tree_engine, allow_back=True)
                if result is _BACK:
                    idx -= 1
                    continue
                tree_engine = result
            else:
                result = _ask_engine(default=engine, allow_back=True)
                if result is _BACK:
                    idx -= 1
                    continue
                engine = result
            idx += 1
            continue

        if step == "format":
            result = _ask_format(level, default=fmt, allow_back=True)
            if result is _BACK:
                idx -= 1
                continue
            fmt = result
            out = _resync_output_extension(out, fmt)
            idx += 1
            continue

        if step == "output":
            if out is None:
                out = _default_output_path(source, kind, fmt)
            result = _ask_output(fmt, default=str(out), allow_back=True)
            if result is _BACK:
                idx -= 1
                continue
            out = Path(result)
            idx += 1
            continue

    _EDIT_FIELDS_BY_MODE = {
        "tree": [
            Choice("Source", value="source"),
            Choice("Engine", value="engine"),
            Choice("Format", value="format"),
            Choice("Output path", value="output"),
        ],
        "_default": [
            Choice("Source", value="source"),
            Choice("Level", value="level"),
            Choice("Engine", value="engine"),
            Choice("Format", value="format"),
            Choice("Output path", value="output"),
        ],
    }

    while True:
        console.print()
        console.print(Rule(style="cyan"))
        console.print(_summary_panel(mode, source, kind, level, engine, tree_engine, fmt, out))

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
            _EDIT_FIELDS_BY_MODE.get(mode, _EDIT_FIELDS_BY_MODE["_default"]),
        )
        console.print()
        if field == "source":
            result = _ask_source_for_mode(mode, default=source)
            if result is not _BACK:
                source = result
                kind = _kind_for(mode, source)
        elif field == "level":
            level = _ask_level(default=level)
        elif field == "engine":
            if mode == "tree":
                tree_engine = _ask_tree_engine(default=tree_engine)
            else:
                engine = _ask_engine(default=engine)
        elif field == "format":
            fmt = _ask_format(level, default=fmt)
            out = _resync_output_extension(out, fmt)
        elif field == "output":
            result = _ask_output(fmt, default=str(out))
            if result is not _BACK:
                out = Path(result)

    console.print()
    _remember_last_answers(level, engine, fmt, tree_engine)

    def _build() -> None:
        if mode == "tree":
            do_tree(
                source,
                fmt,
                out,
                tree_engine,
                max_depth=8,
                max_files=20,
                respect_gitignore=True,
                fresh=False,
                no_cache=False,
                preview=True,
                yes=False,
                dry_run=False,
            )
        elif kind in ("playlist", "folder"):
            do_batch(
                source, level, fmt, out, engine,
                workers=3, limit=None, fresh=False, no_cache=False, preview=True,
            )
        else:
            do_map(source, level, fmt, out, engine, no_cache=False, preview=True)

    # A failure here (bad URL that only breaks at fetch time, a network
    # blip, a private/deleted video) must not throw away everything answered
    # above — re-running the whole wizard from scratch over a typo is the
    # exact frustration "back" exists to avoid, just one step later than the
    # Q&A phase. _export's own overwrite prompt (cli.py) already resolves
    # filename collisions in place without unwinding this far; this loop is
    # the safety net for everything else.
    while True:
        try:
            _build()
            return
        except typer.Exit as exc:
            code = exc.exit_code if exc.exit_code is not None else 0
            if code == 0:
                raise  # a deliberate exit somewhere below -- respect it, don't retry
            console.print()
            action = _select(
                "That didn't finish (see above). What now?",
                [
                    Choice("Fix the source and retry", value="source"),
                    Choice("Fix the output path and retry", value="output"),
                    Choice("Retry as-is", value="retry"),
                    Choice("Cancel", value="cancel"),
                ],
                default="source",
            )
            if action == "cancel":
                _cancel()
            elif action == "source":
                result = _ask_source_for_mode(mode, default=source)
                if result is not _BACK:
                    source = result
                    kind = _kind_for(mode, source)
            elif action == "output":
                result = _ask_output(fmt, default=str(out))
                if result is not _BACK:
                    out = Path(result)
            # "retry" (or a no-op back on source/output) just loops and calls _build() again
