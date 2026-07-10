"""Cerebro command-line interface."""

from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def _force_utf8() -> None:
    """Windows consoles often default to a legacy code page (cp1252) that
    cannot encode ✓, em dashes, or emoji, which crashes Rich. Reconfigure the
    standard streams to UTF-8 before anything is printed."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


_force_utf8()

import typer
from rich.panel import Panel
from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as RichProgress
from rich.table import Table

from . import __version__
from .batch import BatchItem, dry_run_batch, forget_batch_snapshot, list_batch_snapshots, run_batch
from .cache import Cache
from .console import console, has_real_console, qprint, quiet_mode, set_ascii, set_high_contrast, set_quiet
from .convert import write_opml, write_xmind
from .doctor import has_failures, run_diagnostics
from .foldermap import (
    build_folder_map,
    finalize_tree_snapshot,
    forget_tree_snapshot,
    label_folders,
    list_tree_snapshots,
)
from .ingest import load_transcript
from .ingest.folder import discover_course_sources
from .ingest.playlist import is_playlist_url, load_playlist
from .llm.base import LLMError
from .llm.config import ConfigError, load_env, read_env_file, resolve_provider, write_env_file
from .paths import CONFIG_DIR, GLOBAL_ENV_PATH, ensure_output_dir, load_config, save_config
from .structure import HeuristicStructurer
from .structure.llm import LLMStructurer, link_relationships
from .ui import print_banner, print_preview
from .wizard import run_wizard

_EPILOG = (
    'Examples: cerebro (wizard)  |  cerebro map "URL" -l expert  |  '
    'cerebro batch "playlist URL" --limit 10  |  cerebro batch ./course_folder --format xmind  |  '
    "cerebro tree ./my_project --engine groq  |  cerebro doctor"
)

app = typer.Typer(
    add_completion=True,
    help="Turn video content into XMind-compatible smart mind maps. Run with no arguments for a guided wizard.",
    epilog=_EPILOG,
)

_HELP_REQUESTED = "--help" in sys.argv[1:]


def _safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
    return (name or "mindmap")[:80]


@contextmanager
def _spinner(description: str):
    """Spinner + description + elapsed time — used instead of bare
    console.status() everywhere, so every long-running step (loading a
    transcript, possibly via a multi-minute Whisper transcription; reading a
    huge playlist; walking a folder) shows elapsed time, not just the ones
    that happen to have a countable N and already got a full progress bar.
    No bar/count columns since these steps have no countable sub-progress —
    that's what RichProgress is still used directly for elsewhere."""
    with RichProgress(
        SpinnerColumn(), TextColumn("[cyan]{task.description}"), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        yield


def _structure(transcript, level, provider, cache, relationship_limit=8):
    """Build the map with the resolved engine, showing live progress and
    falling back to the offline heuristic if an LLM call fails."""
    if provider is None:
        with _spinner(f"Structuring ({level})…"):
            return HeuristicStructurer().structure(transcript, level=level)

    with RichProgress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Thinking…", total=1)

        def on_event(kind, **d):
            if kind == "map_start":
                progress.update(task, description="Mapping segments", total=d["total"], completed=0)
            elif kind == "map_progress":
                progress.update(task, completed=d["done"])
            elif kind == "reduce_start":
                progress.update(task, description="Reducing into a hierarchy", total=1, completed=0)
            elif kind == "link_start":
                progress.update(task, description="Detecting cross-links", total=1, completed=0)

        structurer = LLMStructurer(
            provider, cache, on_event=on_event, relationship_limit=relationship_limit
        )
        try:
            mm = structurer.structure(transcript, level=level)
        except LLMError as exc:
            progress.stop()
            console.print(f"[yellow]! LLM engine failed ({exc}); falling back to heuristic.[/]")
            return HeuristicStructurer().structure(transcript, level=level)
        progress.update(task, completed=progress.tasks[0].total)
        return mm


def _version_callback(value: bool):
    if value:
        console.print(f"cerebro {__version__}")
        raise typer.Exit()


def _no_color_callback(value: bool):
    # Mutated on the one shared Console (see console.py), not a fresh
    # instance — Rich reads .no_color live at render time, so this reaches
    # every module that already imported it, in whatever order they did.
    if value:
        console.no_color = True
    return value


def _ascii_callback(value: bool):
    if value:
        set_ascii(True)
    return value


def _theme_callback(value: str):
    if value == "high-contrast":
        set_high_contrast(True)
    elif value not in ("default", None):
        raise typer.BadParameter("must be 'default' or 'high-contrast'")
    return value


def _quiet_callback(value: bool):
    if value:
        set_quiet(True)
    return value


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        callback=_no_color_callback,
        is_eager=True,
        help="Disable ANSI color. The NO_COLOR env var (https://no-color.org) works too, without this flag.",
    ),
    ascii_: bool = typer.Option(
        False,
        "--ascii",
        callback=_ascii_callback,
        is_eager=True,
        help="Use plain ASCII glyphs instead of emoji/pictographic icons (some terminals and screen readers handle these poorly).",
    ),
    theme: str = typer.Option(
        "default",
        "--theme",
        callback=_theme_callback,
        is_eager=True,
        help="default | high-contrast — high-contrast drops dim/low-emphasis styling in favor of your terminal's own default foreground.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        callback=_quiet_callback,
        is_eager=True,
        help="Suppress the banner and informational status lines (map/batch/tree). Errors, warnings, and the final result still print — this drops decoration, not answers.",
    ),
):
    """Cerebro root."""
    if _HELP_REQUESTED:
        return  # a --help lookup is a reference check, not a real run
    if not quiet_mode():
        print_banner()
    load_env()
    if ctx.invoked_subcommand is None:
        run_wizard(_do_map, _do_batch)


@app.command()
def map(
    source: str = typer.Argument(
        ..., help="YouTube URL or local .srt/.vtt/.txt/.mp4/.mkv/.mov/.webm/.avi/.m4v file."
    ),
    level: str = typer.Option(None, "--level", "-l", help="brief | full | expert"),
    fmt: str = typer.Option(None, "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="auto | groq | gemini | heuristic"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    whisper_model: str = typer.Option(None, "--whisper-model", help="Whisper model size to use for local video transcription."),
    relationship_limit: int = typer.Option(None, "--relationship-limit", "--rel-limit", help="Max number of relationships to detect in expert mode."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite an existing output file without asking."),
):
    """Build a mind map from SOURCE and write it to disk."""
    _do_map(source, level, fmt, out, engine, no_cache, preview, whisper_model, relationship_limit, yes)


def _do_map(
    source: str,
    level: str | None,
    fmt: str | None,
    out: Path | None,
    engine: str | None,
    no_cache: bool,
    preview: bool,
    whisper_model: str | None = None,
    relationship_limit: int | None = None,
    yes: bool = False,
) -> None:
    config = load_config()
    level = level or config.get("level") or "full"
    fmt = fmt or config.get("format") or "opml"
    engine = engine or config.get("engine") or "auto"
    whisper_model = whisper_model or config.get("whisper_model") or "base"

    if relationship_limit is None:
        cfg_lim = config.get("relationship_limit")
        try:
            relationship_limit = int(cfg_lim) if cfg_lim is not None else 8
        except ValueError:
            relationship_limit = 8

    t0 = time.perf_counter()
    cache = Cache(enabled=not no_cache)

    try:
        with _spinner("Loading transcript…"):
            transcript = load_transcript(source, whisper_model=whisper_model, cache=cache)
    except Exception as exc:
        console.print(f"[red]✗ Failed to load transcript: {exc}[/]")
        raise typer.Exit(code=1)
    qprint(
        f"[green]✓[/] Transcript: [bold]{transcript.title}[/] "
        f"— {transcript.word_count:,} words, {len(transcript.segments):,} segments"
    )

    # Resolve the engine (may fall back to the offline heuristic).
    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(code=1)

    engine_label = "heuristic (offline)" if provider is None else f"{provider.name}:{provider.model}"
    if provider is None and engine == "auto":
        qprint("[yellow]![/] No API key found — using the offline heuristic engine.")

    mm = _structure(transcript, level, provider, cache, relationship_limit=relationship_limit)
    qprint(
        f"[green]✓[/] Map built with [bold]{engine_label}[/]: "
        f"{mm.node_count()} nodes, depth {mm.depth()}"
        + (f", {len(mm.relationships)} relationships" if mm.relationships else "")
    )

    if preview:
        console.print()
        # A single map is the primary view for its source, so it gets a more
        # generous cap than batch's (4) or tree's (6) — but expert-level
        # relationship-heavy maps can still nest deep enough to flood the
        # terminal without any cap at all, same problem batch/tree already
        # guard against for their own preview.
        print_preview(mm, max_depth=8)
        console.print()

    _export(mm, fmt, out, level, time.perf_counter() - t0, yes=yes)


@app.command()
def batch(
    source: str = typer.Argument(..., help="YouTube playlist URL or local course-folder path."),
    level: str = typer.Option(None, "--level", "-l", help="brief | full | expert"),
    fmt: str = typer.Option(None, "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="auto | groq | gemini | heuristic"),
    workers: int = typer.Option(3, "--workers", "-w", help="Videos/lessons processed concurrently."),
    limit: int = typer.Option(None, "--limit", help="Process only the first N items."),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore any previous run of this batch; reprocess everything."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    whisper_model: str = typer.Option(None, "--whisper-model", help="Whisper model size to use for local video transcription."),
    relationship_limit: int = typer.Option(None, "--relationship-limit", "--rel-limit", help="Max number of relationships to detect in expert mode."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite an existing output file without asking."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be reused vs. freshly processed, without spending any API calls or writing output."
    ),
):
    """Build one combined mind map from a YouTube playlist or a local course folder.

    Reruns are incremental by default: any video/lesson whose source exactly
    matches a previous run of this same playlist/folder is reused as-is (no
    transcript refetch, no restructuring) — only genuinely new items are
    processed. Use --fresh to ignore that history and reprocess everything.
    """
    _do_batch(source, level, fmt, out, engine, workers, limit, fresh, no_cache, preview, whisper_model, relationship_limit, yes, dry_run)


def _do_batch(
    source: str,
    level: str | None,
    fmt: str | None,
    out: Path | None,
    engine: str | None,
    workers: int,
    limit: int | None,
    fresh: bool,
    no_cache: bool,
    preview: bool,
    whisper_model: str | None = None,
    relationship_limit: int | None = None,
    yes: bool = False,
    dry_run: bool = False,
) -> None:
    config = load_config()
    level = level or config.get("level") or "full"
    fmt = fmt or config.get("format") or "opml"
    engine = engine or config.get("engine") or "auto"
    whisper_model = whisper_model or config.get("whisper_model") or "base"

    if relationship_limit is None:
        cfg_lim = config.get("relationship_limit")
        try:
            relationship_limit = int(cfg_lim) if cfg_lim is not None else 8
        except ValueError:
            relationship_limit = 8

    t0 = time.perf_counter()

    transcribe_count = 0
    if is_playlist_url(source):
        with _spinner("Reading playlist…"):
            info = load_playlist(source)
        items = [BatchItem(label=t, source=u) for t, u in info.items]
        title = info.title
    elif Path(source).is_dir():
        files = discover_course_sources(Path(source))
        items = [BatchItem(label=f.title, source=str(f.path)) for f in files]
        title = Path(source).name
        transcribe_count = sum(1 for f in files if f.needs_transcription)
    else:
        console.print(f"[red]✗[/] Not a YouTube playlist URL or an existing folder: {source}")
        raise typer.Exit(code=1)

    if not items:
        console.print("[red]✗[/] No videos or lessons found to process.")
        raise typer.Exit(code=1)

    total_found = len(items)
    if limit is not None:
        items = items[:limit]

    qprint(f"[green]✓[/] Found [bold]{total_found}[/] item(s) in [bold]{title}[/]")
    if limit is not None and total_found > limit:
        qprint(f"[dim]  Processing first {len(items)} (--limit {limit}).[/]")
    if transcribe_count:
        qprint(
            f"[dim]  {transcribe_count} video(s) have no subtitle file — will extract an "
            "embedded track or transcribe with Whisper (slower).[/]"
        )

    if dry_run:
        reused, new = dry_run_batch(items, level, source if not fresh else None)
        console.print(f"[cyan]Dry run:[/] would reuse [bold]{len(reused)}[/], process [bold]{len(new)}[/] fresh.")
        if new:
            console.print("[dim]  New/changed:[/]")
            for label in new:
                console.print(f"[dim]    • {label}[/]")
        raise typer.Exit()

    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(code=1)

    engine_label = "heuristic (offline)" if provider is None else f"{provider.name}:{provider.model}"
    if provider is None and engine == "auto":
        qprint("[yellow]![/] No API key found — using the offline heuristic engine.")

    cache = Cache(enabled=not no_cache)

    def structurer_factory():
        # Halve the per-video map-call concurrency so total concurrent LLM
        # requests (batch workers × per-video workers) stays bounded — free-tier
        # rate limits don't scale with playlist size.
        return HeuristicStructurer() if provider is None else LLMStructurer(
            provider, cache, max_workers=2, relationship_limit=relationship_limit
        )

    failures: list[tuple[str, str]] = []
    with RichProgress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing with {engine_label}", total=len(items))

        def on_event(kind, **d):
            if kind == "item_done":
                progress.update(task, completed=d["completed"])
                if not d["ok"]:
                    failures.append((d["label"], d["error"]))

        combined, outcomes, diff = run_batch(
            items,
            structurer_factory,
            level,
            title,
            max_workers=workers,
            on_event=on_event,
            cache=cache,
            whisper_model=whisper_model,
            incremental=not fresh,
            batch_source=source,
        )

    # Each video already got its own within-video links (if any) from its own
    # expert-level structuring above; this second pass looks across all of
    # them together, so a concept in lesson 2 can connect to one in lesson 7.
    if level == "expert" and provider is not None and combined.node_count() > 3:
        with _spinner("Finding connections across videos…"):
            link_relationships(
                combined, provider, cache, cross_video=True, relationship_limit=relationship_limit
            )

    ok_count = sum(1 for o in outcomes if o.mindmap is not None)
    qprint(
        f"[green]✓[/] Processed {ok_count}/{len(items)} item(s) with [bold]{engine_label}[/]: "
        f"{combined.node_count()} nodes, depth {combined.depth()}"
        + (f", {len(combined.relationships)} relationships" if combined.relationships else "")
    )
    if diff is not None:
        since = diff.previous_built_at or "an earlier run"
        parts = []
        if diff.added:
            parts.append(f"{len(diff.added)} new")
        if diff.removed:
            parts.append(f"{len(diff.removed)} removed")
        change_desc = ", ".join(parts) if parts else "no changes"
        qprint(
            f"[dim]  ↻ Reused {len(diff.reused)}/{diff.total} item(s) since {since} — {change_desc}.[/]"
        )
    for label, error in failures:
        console.print(f"[yellow]![/] {label}: {error}")  # a real per-item failure, not decoration -- always shown

    if preview:
        console.print()
        print_preview(combined, max_depth=4)
        console.print()

    _export(combined, fmt, out, level, time.perf_counter() - t0, yes=yes)


@app.command()
def tree(
    path: str = typer.Argument(..., help="Local folder to map (not a video/course folder)."),
    fmt: str = typer.Option(None, "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(
        None, "--engine", "-e", help="heuristic (default, free/instant) | groq | gemini — AI-labels folder purposes"
    ),
    max_depth: int = typer.Option(8, "--max-depth", help="Maximum folder nesting depth."),
    max_files: int = typer.Option(20, "--max-files", help="Max files listed per folder before collapsing to a count."),
    no_gitignore: bool = typer.Option(False, "--no-gitignore", help="Don't respect the folder's .gitignore."),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore any previous map of this folder; rebuild everything."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the AI-label response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite an existing output file without asking."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change, without AI-labeling anything or writing output."
    ),
):
    """Map a folder's directory structure — not a video or course folder.

    Reruns are incremental by default: unchanged subfolders (and any AI
    label already assigned to them) are reused from the previous map of this
    exact folder instead of being rewalked and relabeled. Use --fresh to
    ignore that history and rebuild everything.
    """
    _do_tree(path, fmt, out, engine, max_depth, max_files, not no_gitignore, fresh, no_cache, preview, yes, dry_run)


def _do_tree(
    path: str,
    fmt: str | None,
    out: Path | None,
    engine: str | None,
    max_depth: int,
    max_files: int,
    respect_gitignore: bool,
    fresh: bool,
    no_cache: bool,
    preview: bool,
    yes: bool = False,
    dry_run: bool = False,
) -> None:
    config = load_config()
    fmt = fmt or config.get("format") or "opml"
    engine = engine or "heuristic"  # unlike map/batch, AI is opt-in here — the structure is already known

    t0 = time.perf_counter()

    try:
        with _spinner("Walking folder…"):
            mm, diff, nodes_needing_labels, pending_snapshot = build_folder_map(
                path,
                max_depth=max_depth,
                max_files=max_files,
                respect_gitignore=respect_gitignore,
                incremental=not fresh,
            )
    except ValueError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(code=1)

    qprint(f"[green]✓[/] Walked [bold]{path}[/]: {mm.node_count()} nodes, depth {mm.depth()}")
    if diff is not None:
        since = diff.previous_built_at or "an earlier run"
        parts = []
        if diff.added:
            parts.append(f"{len(diff.added)} new")
        if diff.changed:
            parts.append(f"{len(diff.changed)} changed")
        if diff.deleted:
            parts.append(f"{len(diff.deleted)} deleted")
        change_desc = ", ".join(parts) if parts else "no changes"
        qprint(
            f"[dim]  ↻ Reused {len(diff.reused)}/{diff.total} folder(s) since {since} — {change_desc}.[/]"
        )

    if dry_run:
        if engine == "heuristic":
            console.print("[cyan]Dry run:[/] heuristic engine — no AI labeling; nothing written.")
        else:
            console.print(
                f"[cyan]Dry run:[/] would AI-label [bold]{len(nodes_needing_labels)}[/] folder(s) "
                f"with [bold]{engine}[/]; nothing written."
            )
        raise typer.Exit()

    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(code=1)

    if provider is not None:
        if nodes_needing_labels:
            cache = Cache(enabled=not no_cache)
            with RichProgress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Labeling folders", total=1)

                def on_event(kind, **d):
                    if kind == "label_start":
                        progress.update(task, total=d["total"], completed=0)
                    elif kind == "label_progress":
                        progress.update(task, completed=d["done"])

                label_folders(mm, provider, cache, nodes=nodes_needing_labels, on_event=on_event)
            qprint(
                f"[green]✓[/] Labeled {len(nodes_needing_labels)} folder(s) with "
                f"[bold]{provider.name}:{provider.model}[/]"
            )
        else:
            qprint("[dim]  All folders already labeled from a previous run.[/]")
    elif engine != "heuristic":
        qprint("[yellow]![/] No API key found — skipping AI folder labeling.")

    # Saved only now, after any labeling above has finished mutating notes —
    # saving earlier would silently lose every label just assigned.
    finalize_tree_snapshot(pending_snapshot)

    if preview:
        console.print()
        print_preview(mm, max_depth=6)
        console.print()

    _export(mm, fmt, out, "structure", time.perf_counter() - t0, yes=yes)


@app.command()
def setup():
    """Guided setup for API keys — writes ~/.cerebro/.env, no manual editing required.

    Press Enter to skip a key (e.g. to use only one engine, or to stick with
    the fully offline heuristic engine, which needs no key at all). Leaving a
    key blank keeps whatever was already saved for it, if anything.
    """
    from rich.prompt import Prompt

    console.print(
        "[dim]Free keys: Groq -> https://console.groq.com/keys  ·  "
        "Gemini -> https://aistudio.google.com/apikey[/]\n"
    )

    existing = read_env_file(GLOBAL_ENV_PATH)
    # password=True routes through Python's getpass, which on Windows reads
    # the console device directly and hangs indefinitely on piped/redirected
    # stdin instead of raising or falling back — mask only when there's a
    # real attached terminal to mask against.
    mask = has_real_console()
    if not mask:
        console.print("[yellow]![/] No interactive terminal detected — input will be visible, not masked.\n")

    def _ask_key(name: str, label: str) -> None:
        already_set = bool(existing.get(name))
        hint = "already set — Enter to keep" if already_set else "Enter to skip"
        value = Prompt.ask(f"{label} API key [dim]({hint})[/]", password=mask, default="", show_default=False)
        value = value.strip()
        if value:
            existing[name] = value

    _ask_key("GROQ_API_KEY", "Groq")
    _ask_key("GEMINI_API_KEY", "Gemini")

    if not existing:
        console.print(
            "\n[dim]No keys saved. You can still use --engine heuristic "
            "(fully offline, no key needed) any time.[/]"
        )
        raise typer.Exit()

    write_env_file(GLOBAL_ENV_PATH, existing)
    console.print(f"\n[green]✓[/] Saved to {GLOBAL_ENV_PATH}")
    console.print("[dim]Run `cerebro doctor` to verify.[/]")


_STATUS_STYLE = {"ok": ("[green]✓[/]", "green"), "warn": ("[yellow]![/]", "yellow"), "fail": ("[red]✗[/]", "red")}


@app.command()
def doctor(
    network: bool = typer.Option(
        True, "--network/--no-network", help="Check API/YouTube reachability (skip for a faster, offline-only check)."
    ),
):
    """Diagnose your setup: API keys, ffmpeg/Whisper, storage, connectivity.

    Read-only aside from a throwaway file used to confirm each storage
    directory is actually writable. Exits non-zero only on a hard failure —
    a missing optional piece like Whisper or a second engine's key is
    reported as an advisory, not an error.
    """
    with _spinner("Running diagnostics…"):
        checks = run_diagnostics(check_network=network)

    table = Table(box=None, padding=(0, 1, 0, 0), show_header=False)
    table.add_column(width=2)
    table.add_column(style="bold", min_width=22)
    table.add_column()
    last_group = None
    for check in checks:
        if check.group != last_group:
            if last_group is not None:
                table.add_row("", "", "")
            table.add_row("", f"[cyan]{check.group}[/]", "")
            last_group = check.group
        icon, color = _STATUS_STYLE[check.status]
        detail = check.detail
        if check.fix:
            detail += f"\n[dim]  → {check.fix}[/]"
        table.add_row(icon, f"  {check.label}", f"[{color}]{detail}[/]" if check.status != "ok" else detail)

    console.print(Panel(table, title="[cyan]cerebro doctor[/]", border_style="cyan", expand=False))

    ok_count = sum(1 for c in checks if c.status == "ok")
    warn_count = sum(1 for c in checks if c.status == "warn")
    fail_count = sum(1 for c in checks if c.status == "fail")
    summary = f"[green]{ok_count} ok[/]"
    if warn_count:
        summary += f", [yellow]{warn_count} advisory[/]"
    if fail_count:
        summary += f", [red]{fail_count} failing[/]"
    console.print(summary)

    if has_failures(checks):
        raise typer.Exit(code=1)


@app.command()
def status():
    """Show what cerebro remembers: the response cache, plus every folder/playlist with saved incremental history.

    Complements `cerebro doctor` (which checks whether your setup will
    *work*) by answering a different question: what has cerebro already
    *done*, and for what — the thing you need to know before reaching for
    `cerebro forget`.
    """
    cache = Cache()
    count, total_bytes = cache.stats()
    tree_snaps = list_tree_snapshots()
    batch_snaps = list_batch_snapshots()

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Response cache[/]", f"{count} entries, {_human_size(total_bytes)}")
    summary.add_row("[dim]Tree snapshots[/]", f"{len(tree_snaps)} folder(s) with saved history")
    summary.add_row("[dim]Batch snapshots[/]", f"{len(batch_snaps)} playlist/course(s) with saved history")
    console.print(Panel(summary, title="[cyan]cerebro status[/]", border_style="cyan", expand=False))

    if tree_snaps:
        table = Table(title="Folders with saved map history (cerebro tree)", box=None)
        table.add_column("Source", style="bold")
        table.add_column("Built")
        table.add_column("Folders")
        table.add_column("Labeled")
        for snap in tree_snaps:
            table.add_row(snap["source"], snap["built_at"], str(snap["folders"]), str(snap["labels"]))
        console.print(table)

    if batch_snaps:
        table = Table(title="Playlists/courses with saved batch history (cerebro batch)", box=None)
        table.add_column("Source", style="bold")
        table.add_column("Built")
        table.add_column("Items")
        for snap in batch_snaps:
            table.add_row(snap["source"], snap["built_at"], str(snap["items"]))
        console.print(table)

    if not tree_snaps and not batch_snaps:
        console.print("[dim]No incremental history yet — run `cerebro tree` or `cerebro batch` to build some.[/]")


@app.command()
def interactive():
    """Guided wizard: pick a source, level, engine, and format step by step."""
    # print_banner()/load_env() already ran in the _main callback, which fires
    # for every invocation regardless of which subcommand was requested.
    run_wizard(_do_map, _do_batch)


cache_app = typer.Typer(add_completion=False, help="Inspect or clear the response cache.")
app.add_typer(cache_app, name="cache")

config_app = typer.Typer(add_completion=False, help="View or set persisted defaults, instead of hand-editing config.json.")
app.add_typer(config_app, name="config")

# (choices, default) — the single source of truth for what's a valid key/value,
# used by both `config set`'s validation and `config list`'s fallback display.
_CONFIG_KEYS: dict[str, tuple[tuple[str, ...] | None, str]] = {
    "level": (("brief", "full", "expert"), "full"),
    "format": (("opml", "xmind"), "opml"),
    "engine": (("auto", "groq", "gemini", "heuristic"), "auto"),
    "whisper_model": (("tiny", "base", "small", "medium", "large-v2", "large-v3"), "base"),
    "relationship_limit": (None, "8"),
}


@config_app.command("list")
def config_list():
    """Show every persisted default, with cerebro's built-in fallback for anything unset."""
    config = load_config()
    table = Table.grid(padding=(0, 2))
    for key, (_choices, default) in _CONFIG_KEYS.items():
        value = config.get(key)
        display = str(value) if value is not None else f"[dim](unset — default: {default})[/]"
        table.add_row(f"[dim]{key}[/]", display)
    console.print(Panel(table, title="[cyan]Config[/]", border_style="cyan", expand=False))
    console.print(f"[dim]{CONFIG_DIR / 'config.json'}[/]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="level | format | engine | whisper_model | relationship_limit")):
    """Print one config key's current value (persisted, or the built-in default)."""
    if key not in _CONFIG_KEYS:
        console.print(f"[red]✗[/] Unknown config key: {key} (known: {', '.join(_CONFIG_KEYS)})")
        raise typer.Exit(code=1)
    _choices, default = _CONFIG_KEYS[key]
    console.print(str(load_config().get(key, default)))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="level | format | engine | whisper_model | relationship_limit"),
    value: str = typer.Argument(..., help="The new value — must match the key's allowed choices."),
):
    """Persist a default so map/batch/tree don't need the flag every time."""
    if key not in _CONFIG_KEYS:
        console.print(f"[red]✗[/] Unknown config key: {key} (known: {', '.join(_CONFIG_KEYS)})")
        raise typer.Exit(code=1)
    choices, _default = _CONFIG_KEYS[key]
    if choices is not None and value not in choices:
        console.print(f"[red]✗[/] Invalid value {value!r} for {key} (choices: {', '.join(choices)})")
        raise typer.Exit(code=1)
    if key == "relationship_limit" and not value.isdigit():
        console.print(f"[red]✗[/] relationship_limit must be a positive integer, got {value!r}")
        raise typer.Exit(code=1)
    config = load_config()
    config[key] = value
    save_config(config)
    console.print(f"[green]✓[/] {key} = {value}")


@config_app.command("unset")
def config_unset(key: str = typer.Argument(..., help="level | format | engine | whisper_model | relationship_limit")):
    """Remove a persisted default, reverting that key to cerebro's built-in default."""
    config = load_config()
    if key not in config:
        console.print(f"[dim]{key} was already unset.[/]")
        raise typer.Exit()
    del config[key]
    save_config(config)
    default = _CONFIG_KEYS.get(key, (None, "?"))[1]
    console.print(f"[green]✓[/] {key} unset — back to default ({default}).")


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


@cache_app.command("stats")
def cache_stats():
    """Show the cache location, entry count, and total size."""
    cache = Cache()
    count, total_bytes = cache.stats()
    table = Table.grid(padding=(0, 2))
    table.add_row("[dim]Location[/]", str(cache.root))
    table.add_row("[dim]Entries[/]", str(count))
    table.add_row("[dim]Size[/]", _human_size(total_bytes))
    console.print(Panel(table, title="[cyan]Cache[/]", border_style="cyan", expand=False))


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
):
    """Delete all cached responses and transcriptions."""
    cache = Cache()
    count, total_bytes = cache.stats()
    if count == 0:
        console.print("[dim]Cache is already empty.[/]")
        raise typer.Exit()
    if not yes:
        from rich.prompt import Confirm

        if not Confirm.ask(f"Delete {count} cached entries ({_human_size(total_bytes)})?", default=False):
            console.print("[dim]Cancelled.[/]")
            raise typer.Exit()
    removed = cache.clear()
    console.print(f"[green]✓[/] Removed {removed} cached entries.")


forget_app = typer.Typer(add_completion=False, help="Clear one folder's or playlist's incremental history without wiping the whole cache.")
app.add_typer(forget_app, name="forget")


@forget_app.command("tree")
def forget_tree(path: str = typer.Argument(..., help="The folder path exactly as given to `cerebro tree`.")):
    """Forget a folder's map history — the next `cerebro tree PATH` rebuilds it from scratch."""
    if forget_tree_snapshot(path):
        console.print(f"[green]✓[/] Forgot the map history for [bold]{path}[/]. The next run rebuilds it from scratch.")
    else:
        console.print(f"[dim]No saved history for {path} — nothing to forget.[/]")


@forget_app.command("batch")
def forget_batch(
    source: str = typer.Argument(..., help="The playlist URL or course-folder path exactly as given to `cerebro batch`.")
):
    """Forget a playlist/course's batch history — the next `cerebro batch SOURCE` reprocesses everything."""
    if forget_batch_snapshot(source):
        console.print(f"[green]✓[/] Forgot the batch history for [bold]{source}[/]. The next run reprocesses everything.")
    else:
        console.print(f"[dim]No saved history for {source} — nothing to forget.[/]")


def _export(mm, fmt: str, out: Path | None, level: str, elapsed: float, yes: bool = False) -> None:
    if fmt not in ("opml", "xmind"):
        console.print(f"[red]✗[/] Unknown format: {fmt} (use opml or xmind)")
        raise typer.Exit(code=1)
    if fmt == "opml" and mm.relationships:
        console.print(
            f"[yellow]![/] {len(mm.relationships)} relationship(s) dropped — "
            "OPML can't carry cross-links. Use [bold]--format xmind[/] to keep them."
        )

    if out is None:
        out = ensure_output_dir() / f"{_safe_filename(mm.title)}.{fmt}"
    # Applies whether `out` was explicit or auto-generated from the title —
    # re-running the same source without --out resolves to the same default
    # path, so it's just as capable of silently clobbering prior work.
    if out.exists() and not yes:
        from rich.prompt import Confirm

        if not Confirm.ask(f"[yellow]![/] {out} already exists — overwrite?", default=False):
            console.print("[dim]Cancelled — nothing written. Pass --out a different path, or --yes to overwrite.[/]")
            raise typer.Exit(code=1)
    written = write_opml(mm, out) if fmt == "opml" else write_xmind(mm, out)

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Output[/]", f"[bold]{written}[/]")
    summary.add_row("[dim]Format[/]", fmt.upper())
    summary.add_row("[dim]Level[/]", level)
    summary.add_row("[dim]Time[/]", f"{elapsed:.2f}s")
    console.print(Panel(summary, title="[green]Done[/]", border_style="green", expand=False))
    if fmt == "opml":
        qprint(f"[dim]Import into XMind: File → Import → OPML → {written.name}[/]")
    else:
        qprint(f"[dim]Open directly in XMind: {written.name}[/]")


def run() -> None:
    """Entry point with graceful Ctrl+C handling (the wizard advertises this)."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/]")
        raise typer.Exit(code=130)


if __name__ == "__main__":
    run()
