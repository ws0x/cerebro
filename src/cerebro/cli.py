"""Cerebro command-line interface."""

from __future__ import annotations

import json
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
from .console import (
    console,
    has_real_console,
    json_mode,
    qprint,
    quiet_mode,
    set_ascii,
    set_high_contrast,
    set_json,
    set_quiet,
)
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
from .manifest import lookup as manifest_lookup
from .manifest import record as manifest_record
from .merge import MergeError, merge_maps
from .paths import CONFIG_DIR, GLOBAL_ENV_PATH, ensure_output_dir, load_config, save_config
from .search import search_maps
from .structure import HeuristicStructurer
from .structure.document import OutlineAwareStructurer, build_outline_map, build_outline_skeleton
from .structure.llm import LLMStructurer, link_relationships
from .ui import banner, print_banner, print_preview
from .wizard import run_wizard

_EPILOG = (
    'Examples: cerebro (wizard)  |  cerebro map "URL" -l expert  |  '
    'cerebro batch "playlist URL" --limit 10  |  cerebro batch ./course_folder --format xmind  |  '
    "cerebro tree ./my_project --engine groq  |  cerebro doctor  |  cerebro dashboard"
)

app = typer.Typer(
    add_completion=True,
    help="Turn video, audio, and PDF content into XMind-compatible smart mind maps. Run with no arguments for a guided wizard.",
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
        console=console, transient=True, disable=json_mode(),
    ) as progress:
        progress.add_task(description, total=None)
        yield


def _error(message: str, fix: str | None = None, code: int = 1):
    """Report a failure consistently everywhere: a JSON object on stdout
    under --json, otherwise a red X plus an optional actionable next-step —
    the same label/detail/fix shape `cerebro doctor` already established,
    now applied to every command's error paths instead of ad hoc strings.
    """
    if json_mode():
        payload = {"ok": False, "error": message}
        if fix:
            payload["fix"] = fix
        print(json.dumps(payload, ensure_ascii=False))
    else:
        console.print(f"[red]✗[/] {message}")
        if fix:
            console.print(f"[dim]  → {fix}[/]")
    raise typer.Exit(code=code)


def _emit_result(payload: dict) -> None:
    """The --json success counterpart to _error — a no-op otherwise. Uses
    plain print(), not console.print(): Rich's markup parser would try to
    interpret literal "[" characters in a path or title as style tags and
    corrupt the JSON, so this deliberately bypasses Rich entirely."""
    if json_mode():
        print(json.dumps(payload, ensure_ascii=False))


def _structure(transcript, level, provider, cache, relationship_limit=8):
    """Build the map with the resolved engine, showing live progress and
    falling back to the offline heuristic if an LLM call fails."""
    if transcript.outline:
        # A source with real, pre-existing structure (currently: PDFs with a
        # TOC/detected headings) -- the hierarchy is already known, so this
        # skips reduce entirely rather than asking an LLM to reinvent it.
        return _structure_document(transcript, level, provider, cache, relationship_limit)

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
        disable=json_mode(),
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


def _structure_document(transcript, level, provider, cache, relationship_limit=8):
    """Outline-bearing source path (see structure/document.py): the hierarchy
    is already known, so the LLM (if any) only enriches section content and,
    at expert level, detects cross-section relationships. Per-leaf and link
    failures are already handled internally there (the deterministic skeleton
    note is kept on failure), so no top-level fallback is needed here."""
    if provider is None:
        with _spinner(f"Structuring ({level})…"):
            return build_outline_skeleton(transcript, level=level)

    with RichProgress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=json_mode(),
    ) as progress:
        task = progress.add_task("Thinking…", total=1)

        def on_event(kind, **d):
            if kind == "map_start":
                progress.update(task, description="Extracting sections", total=d["total"], completed=0)
            elif kind == "map_progress":
                progress.update(task, completed=d["done"])
            elif kind == "link_start":
                progress.update(task, description="Detecting cross-links", total=1, completed=0)

        mm = build_outline_map(
            transcript,
            provider,
            cache,
            level=level,
            on_event=on_event,
            relationship_limit=relationship_limit,
        )
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


def _json_callback(value: bool):
    if value:
        set_json(True)
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
    as_json: bool = typer.Option(
        False,
        "--json",
        callback=_json_callback,
        is_eager=True,
        help="Print one JSON result object on stdout instead of Rich output (map/batch/tree/doctor). Implies --quiet. Errors become a JSON {\"ok\": false, ...} object too.",
    ),
):
    """Cerebro root."""
    if _HELP_REQUESTED:
        return  # a --help lookup is a reference check, not a real run
    # dashboard renders its own banner as the header of its full-page layout —
    # printing this one first would either flash-and-vanish once the
    # alternate screen buffer takes over, or (no real terminal) duplicate it.
    if not quiet_mode() and ctx.invoked_subcommand != "dashboard":
        print_banner()
    load_env()
    if ctx.invoked_subcommand is None:
        run_wizard(_do_map, _do_batch, _do_tree)


@app.command()
def map(
    source: str = typer.Argument(
        ..., help="YouTube URL or local .srt/.vtt/.txt/.mp4/.mkv/.mov/.webm/.avi/.m4v/.mp3/.wav/.m4a/.flac/.ogg/.aac/.pdf file."
    ),
    level: str = typer.Option(None, "--level", "-l", help="How much structure to extract: brief | full | expert (default: full, or your saved config — see cerebro config)"),
    fmt: str = typer.Option(None, "--format", "-f", help="Output file format: opml | xmind (default: opml, or your saved config)"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="Which engine structures the content: auto | groq | gemini | heuristic (default: auto, or your saved config)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    whisper_model: str = typer.Option(None, "--whisper-model", help="Whisper model size for videos with no subtitle track: tiny | base | small | medium | large-v2 | large-v3 (default: base, or your saved config) -- bigger is slower but more accurate."),
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

    # Purely informational -- a hit here never blocks the build (the source
    # may have changed, or the user may deliberately want a different
    # engine's take), it just answers "didn't I already map this?" before
    # spending the time/LLM calls to find out the hard way.
    previous = manifest_lookup(source, level, fmt)
    if previous and not json_mode():
        qprint(
            f"[dim]  ℹ You already mapped this at '{level}'/{fmt} with {previous['engine']} "
            f"on {previous['built_at'][:10]} → {previous['output']}[/]"
        )

    t0 = time.perf_counter()
    cache = Cache(enabled=not no_cache)

    try:
        with _spinner("Loading transcript…"):
            transcript = load_transcript(source, whisper_model=whisper_model, cache=cache)
    except Exception as exc:
        # No blanket fix hint here -- every ingest module (YouTube, video,
        # PDF) already raises a specific, self-explanatory message ("ffmpeg
        # not found on PATH", "encrypted PDFs are not supported", "File not
        # found: ..."), so a generic "check the URL or path" bolt-on would
        # actively mislead for anything that isn't a path typo.
        _error(f"Failed to load transcript: {exc}")
    qprint(
        f"[green]✓[/] Transcript: [bold]{transcript.title}[/] "
        f"— {transcript.word_count:,} words, {len(transcript.segments):,} segments"
    )

    # Resolve the engine (may fall back to the offline heuristic).
    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        _error(str(exc))  # ConfigError's own message already includes the actionable fix

    engine_label = "heuristic (offline)" if provider is None else f"{provider.name}:{provider.model}"
    if provider is None and engine == "auto":
        qprint("[yellow]![/] No API key found — using the offline heuristic engine.")

    mm = _structure(transcript, level, provider, cache, relationship_limit=relationship_limit)
    qprint(
        f"[green]✓[/] Map built with [bold]{engine_label}[/]: "
        f"{mm.node_count()} nodes, depth {mm.depth()}"
        + (f", {len(mm.relationships)} relationships" if mm.relationships else "")
    )

    if preview and not json_mode():
        console.print()
        # A single map is the primary view for its source, so it gets a more
        # generous cap than batch's (4) or tree's (6) — but expert-level
        # relationship-heavy maps can still nest deep enough to flood the
        # terminal without any cap at all, same problem batch/tree already
        # guard against for their own preview.
        print_preview(mm, max_depth=8)
        console.print()

    elapsed = time.perf_counter() - t0
    written, rel_dropped = _export(mm, fmt, out, level, elapsed, yes=yes)
    manifest_record(source, level, fmt, engine_label, written)
    _emit_result({
        "ok": True,
        "source": source,
        "engine": engine_label,
        "level": level,
        "format": fmt,
        "output": str(written),
        "nodes": mm.node_count(),
        "depth": mm.depth(),
        "relationships": len(mm.relationships),
        "relationships_dropped": rel_dropped,
        "elapsed_seconds": round(elapsed, 2),
        "previously_mapped": previous,
    })


@app.command()
def batch(
    source: str = typer.Argument(..., help="YouTube playlist URL or local course-folder path."),
    level: str = typer.Option(None, "--level", "-l", help="How much structure to extract: brief | full | expert (default: full, or your saved config — see cerebro config)"),
    fmt: str = typer.Option(None, "--format", "-f", help="Output file format: opml | xmind (default: opml, or your saved config)"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="Which engine structures the content: auto | groq | gemini | heuristic (default: auto, or your saved config)"),
    workers: int = typer.Option(3, "--workers", "-w", help="Videos/lessons processed concurrently."),
    limit: int = typer.Option(None, "--limit", help="Process only the first N items."),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore any previous run of this batch; reprocess everything."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    whisper_model: str = typer.Option(None, "--whisper-model", help="Whisper model size for videos with no subtitle track: tiny | base | small | medium | large-v2 | large-v3 (default: base, or your saved config) -- bigger is slower but more accurate."),
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
        _error(
            f"Not a YouTube playlist URL or an existing folder: {source}",
            fix="Pass a YouTube playlist URL or an existing local folder path.",
        )

    if not items:
        _error(
            "No videos or lessons found to process.",
            fix="Check the folder actually contains video/subtitle files, or that the playlist has items.",
        )

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
        if json_mode():
            _emit_result({"ok": True, "dry_run": True, "source": source, "reused": len(reused), "new": new})
        else:
            console.print(f"[cyan]Dry run:[/] would reuse [bold]{len(reused)}[/], process [bold]{len(new)}[/] fresh.")
            if new:
                console.print("[dim]  New/changed:[/]")
                for label in new:
                    console.print(f"[dim]    • {label}[/]")
        raise typer.Exit()

    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        _error(str(exc))

    engine_label = "heuristic (offline)" if provider is None else f"{provider.name}:{provider.model}"
    if provider is None and engine == "auto":
        qprint("[yellow]![/] No API key found — using the offline heuristic engine.")

    cache = Cache(enabled=not no_cache)

    def structurer_factory():
        # Halve the per-video map-call concurrency so total concurrent LLM
        # requests (batch workers × per-video workers) stays bounded — free-tier
        # rate limits don't scale with playlist size. OutlineAwareStructurer
        # routes any item with a real outline (a PDF with a TOC/detected
        # headings, now that course folders can mix PDFs in -- see
        # ingest/folder.py) through build_outline_map instead of flattening
        # it through this same video-oriented path everything else uses.
        return OutlineAwareStructurer(
            provider, cache, relationship_limit=relationship_limit, max_workers=2
        )

    failures: list[tuple[str, str]] = []
    with RichProgress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        disable=json_mode(),
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
    if not json_mode():  # folded into items_failed below instead, under --json
        for label, error in failures:
            console.print(f"[yellow]![/] {label}: {error}")

    if preview and not json_mode():
        console.print()
        print_preview(combined, max_depth=4)
        console.print()

    elapsed = time.perf_counter() - t0
    written, rel_dropped = _export(combined, fmt, out, level, elapsed, yes=yes)
    _emit_result({
        "ok": True,
        "source": source,
        "title": title,
        "engine": engine_label,
        "level": level,
        "format": fmt,
        "output": str(written),
        "nodes": combined.node_count(),
        "depth": combined.depth(),
        "relationships": len(combined.relationships),
        "relationships_dropped": rel_dropped,
        "elapsed_seconds": round(elapsed, 2),
        "items_total": len(items),
        "items_ok": ok_count,
        "items_failed": [{"label": lbl, "error": err} for lbl, err in failures],
        "reused": len(diff.reused) if diff is not None else None,
        "added": len(diff.added) if diff is not None else None,
        "removed": diff.removed if diff is not None else None,
    })


@app.command()
def tree(
    path: str = typer.Argument(..., help="Local folder to map (not a video/course folder)."),
    fmt: str = typer.Option(None, "--format", "-f", help="Output file format: opml | xmind (default: opml, or your saved config)"),
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
        _error(str(exc))

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
        would_label = len(nodes_needing_labels) if engine != "heuristic" else 0
        if json_mode():
            _emit_result({"ok": True, "dry_run": True, "path": path, "would_label": would_label, "engine": engine})
        elif engine == "heuristic":
            console.print("[cyan]Dry run:[/] heuristic engine — no AI labeling; nothing written.")
        else:
            console.print(
                f"[cyan]Dry run:[/] would AI-label [bold]{would_label}[/] folder(s) "
                f"with [bold]{engine}[/]; nothing written."
            )
        raise typer.Exit()

    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        _error(str(exc))

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
                disable=json_mode(),
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

    if preview and not json_mode():
        console.print()
        print_preview(mm, max_depth=6)
        console.print()

    elapsed = time.perf_counter() - t0
    written, _rel_dropped = _export(mm, fmt, out, "structure", elapsed, yes=yes)
    _emit_result({
        "ok": True,
        "path": path,
        "format": fmt,
        "output": str(written),
        "nodes": mm.node_count(),
        "depth": mm.depth(),
        "elapsed_seconds": round(elapsed, 2),
        "reused": len(diff.reused) if diff is not None else None,
        "added": len(diff.added) if diff is not None else None,
        "changed": len(diff.changed) if diff is not None else None,
        "deleted": len(diff.deleted) if diff is not None else None,
        "labeled": len(nodes_needing_labels) if provider is not None else 0,
        "engine": engine,
    })


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

    if json_mode():
        _emit_result({
            "ok": not has_failures(checks),
            "checks": [
                {"group": c.group, "label": c.label, "status": c.status, "detail": c.detail, "fix": c.fix}
                for c in checks
            ],
        })
        if has_failures(checks):
            raise typer.Exit(code=1)
        raise typer.Exit()

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


_SEARCH_NOTE_SNIPPET_LEN = 100


@app.command()
def search(
    query: str = typer.Argument(..., help="Text to look for in every previously-built map's node titles/notes."),
    maps_dir: Path = typer.Option(None, "--dir", "-d", help="Folder to search (default: ~/cerebro-maps, where maps land unless --out pointed elsewhere)."),
    case_sensitive: bool = typer.Option(False, "--case-sensitive", help="Match case exactly instead of ignoring it."),
    limit: int = typer.Option(10, "--limit", help="Max matches shown per file."),
):
    """Search every OPML/XMind map you've already built for a topic.

    Once you've generated a handful of maps, "which one talks about X"
    becomes its own problem this answers without reopening each one by hand.
    """
    search_dir = maps_dir or ensure_output_dir()
    results = search_maps(query, search_dir, case_sensitive=case_sensitive, max_matches_per_file=limit)

    if json_mode():
        _emit_result({
            "ok": True,
            "query": query,
            "dir": str(search_dir),
            "maps_matched": len(results),
            "results": [
                {
                    "path": str(r.path),
                    "matches": [{"title": n.title, "note": n.note} for n in r.nodes],
                }
                for r in results
            ],
        })
        return

    if not results:
        console.print(f"[dim]No matches for \"{query}\" under {search_dir}.[/]")
        return

    total_matches = sum(len(r.nodes) for r in results)
    console.print(
        f"[green]✓[/] {total_matches} match(es) in [bold]{len(results)}[/] map(s) matching [bold]\"{query}\"[/]\n"
    )
    for r in results:
        console.print(f"[bold cyan]{r.path.name}[/] [dim]({r.path})[/]")
        for n in r.nodes:
            console.print(f"  • {n.title}")
            if n.note:
                snippet = n.note.replace("\n", " ").strip()
                if len(snippet) > _SEARCH_NOTE_SNIPPET_LEN:
                    snippet = snippet[:_SEARCH_NOTE_SNIPPET_LEN].rstrip() + "…"
                console.print(f"    [dim]{snippet}[/]")
        console.print()


@app.command()
def merge(
    files: list[Path] = typer.Argument(..., help="Two or more already-built .opml/.xmind files to combine."),
    title: str = typer.Option("Merged Map", "--title", help="Title for the combined map's root."),
    fmt: str = typer.Option(None, "--format", "-f", help="Output format: opml | xmind (default: xmind if any input carries relationships, else opml)."),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the combined map in-terminal."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Overwrite an existing output file without asking."),
):
    """Combine two or more already-built maps into one -- no re-ingestion, no LLM calls.

    Each input file becomes its own top-level branch under a new shared
    root, exactly as batch already does for freshly-built sources -- useful
    for combining, say, a video map and a PDF map on the same topic without
    spending another API call on either one. Each file's own relationships
    (XMind only -- OPML never carried any to begin with) are preserved.
    """
    if len(files) < 2:
        _error("Need at least 2 files to merge.", fix="Pass two or more .opml/.xmind paths.")
    for f in files:
        if not f.exists():
            _error(f"File not found: {f}")

    t0 = time.perf_counter()
    try:
        mm = merge_maps(files, title=title)
    except MergeError as exc:
        _error(str(exc))

    if fmt is None:
        fmt = "xmind" if mm.relationships else "opml"

    qprint(
        f"[green]✓[/] Merged {len(files)} map(s): {mm.node_count()} nodes, depth {mm.depth()}"
        + (f", {len(mm.relationships)} relationships" if mm.relationships else "")
    )

    if preview and not json_mode():
        console.print()
        print_preview(mm, max_depth=6)
        console.print()

    elapsed = time.perf_counter() - t0
    written, rel_dropped = _export(mm, fmt, out, "merged", elapsed, yes=yes)
    _emit_result({
        "ok": True,
        "files": [str(f) for f in files],
        "format": fmt,
        "output": str(written),
        "nodes": mm.node_count(),
        "depth": mm.depth(),
        "relationships": len(mm.relationships),
        "relationships_dropped": rel_dropped,
        "elapsed_seconds": round(elapsed, 2),
    })


def _dashboard_layout():
    from rich.layout import Layout

    checks = run_diagnostics(check_network=False)
    ok = sum(1 for c in checks if c.status == "ok")
    warn = sum(1 for c in checks if c.status == "warn")
    fail = sum(1 for c in checks if c.status == "fail")
    health = Table.grid(padding=(0, 1))
    for c in checks:
        if c.status == "ok":
            continue  # only the things that need a look, so this stays scannable at a glance
        icon, color = _STATUS_STYLE[c.status]
        health.add_row(icon, f"[{color}]{c.label}:[/] {c.detail}")
    if fail == 0 and warn == 0:
        health.add_row("[green]✓[/]", "[green]Everything looks good.[/]")
    health_panel = Panel(
        health,
        title=f"[cyan]Setup[/] — {ok} ok, {warn} advisory, {fail} failing",
        border_style="cyan",
        subtitle="[dim]cerebro doctor for the full picture[/]",
    )

    cache = Cache()
    count, total_bytes = cache.stats()
    tree_snaps = list_tree_snapshots()
    batch_snaps = list_batch_snapshots()
    mem = Table.grid(padding=(0, 2))
    mem.add_row("[dim]Response cache[/]", f"{count} entries, {_human_size(total_bytes)}")
    mem.add_row("[dim]Tree snapshots[/]", f"{len(tree_snaps)} folder(s) mapped")
    mem.add_row("[dim]Batch snapshots[/]", f"{len(batch_snaps)} playlist/course(s) run")
    memory_panel = Panel(
        mem, title="[cyan]Remembered[/]", border_style="cyan", subtitle="[dim]cerebro status for details[/]"
    )

    layout = Layout()
    layout.split_column(
        Layout(banner(), name="header", size=9),
        Layout(name="body"),
        Layout(
            Panel("[dim]Enter to refresh · q + Enter to quit[/]", border_style="dim"), name="footer", size=3
        ),
    )
    layout["body"].split_row(Layout(health_panel, name="health"), Layout(memory_panel, name="memory"))
    return layout


@app.command()
def dashboard():
    """Full-page live overview: setup health + everything cerebro remembers.

    Takes over the whole terminal viewport (the alternate screen buffer —
    the same mechanism `less`/`git diff`/`htop` use) and restores your
    previous terminal content on exit, instead of scrolling more text into
    your history. Enter refreshes; 'q' quits.

    Falls back to a single static render when there's no real attached
    terminal to take over (piped output, CI) rather than switching screens
    or blocking on input that will never come.
    """
    if not has_real_console():
        console.print(_dashboard_layout())
        return

    from rich.prompt import Prompt

    with console.screen() as screen:
        while True:
            screen.update(_dashboard_layout())
            answer = Prompt.ask("", console=console, default="", show_default=False)
            if answer.strip().lower() in ("q", "quit", "exit"):
                break


@app.command()
def interactive():
    """Guided wizard: pick a source, level, engine, and format step by step.

    Identical to running `cerebro` with no arguments — this named form
    exists for discoverability (so it shows up in --help and tab-completion)
    and for scripts/aliases that prefer an explicit subcommand.
    """
    # print_banner()/load_env() already ran in the _main callback, which fires
    # for every invocation regardless of which subcommand was requested.
    run_wizard(_do_map, _do_batch, _do_tree)


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


def _export(mm, fmt: str, out: Path | None, level: str, elapsed: float, yes: bool = False) -> tuple[Path, int]:
    """Writes the map to disk. Returns ``(written_path, relationships_dropped)``
    so callers can fold both into a --json result payload instead of relying
    on this function's own (suppressed, under --json) Rich output."""
    if fmt not in ("opml", "xmind"):
        _error(f"Unknown format: {fmt}", fix="Use opml or xmind.")

    relationships_dropped = len(mm.relationships) if fmt == "opml" and mm.relationships else 0
    if relationships_dropped and not json_mode():
        console.print(
            f"[yellow]![/] {relationships_dropped} relationship(s) dropped — "
            "OPML can't carry cross-links. Use [bold]--format xmind[/] to keep them."
        )

    if out is None:
        out = ensure_output_dir() / f"{_safe_filename(mm.title)}.{fmt}"
    elif out.suffix.lstrip(".").lower() != fmt:
        # The file's actual on-disk content must always match its extension
        # -- an explicit --out whose extension disagrees with the resolved
        # --format (e.g. a saved config default, or an auto-picked format
        # like merge's "xmind if any input has relationships") would
        # otherwise write XMind's zip archive into a file literally named
        # .opml, which nothing can open correctly. The stem/directory the
        # caller chose is always kept -- only the suffix is corrected.
        corrected = out.with_suffix(f".{fmt}")
        if not json_mode():
            console.print(f"[dim]  ↳ Output extension adjusted to match --format {fmt}: {corrected.name}[/]")
        out = corrected
    # Applies whether `out` was explicit or auto-generated from the title —
    # re-running the same source without --out resolves to the same default
    # path, so it's just as capable of silently clobbering prior work.
    if out.exists() and not yes:
        if json_mode():
            # Confirm.ask would block waiting on stdin, which a script
            # consuming JSON off stdout has no way to answer — fail clearly
            # instead of hanging.
            _error(f"{out} already exists.", fix="Pass --yes to overwrite, or --out a different path.")

        from rich.prompt import Prompt

        # A plain y/n here used to mean "no" silently discarded the map that
        # had just been built (often after a real, non-free LLM call) with
        # no way back short of rerunning the whole pipeline from scratch —
        # the single output filename default (see wizard._default_output_path
        # for the wizard's own mitigation) made this a routine, not rare,
        # dead end. Offering a rename in place, looped until it resolves,
        # means the file collision is the only thing that costs a retry —
        # never the map itself.
        while out.exists():
            console.print(f"[yellow]![/] {out} already exists.")
            action = Prompt.ask(
                "  Overwrite, save under a different name, or cancel?",
                choices=["overwrite", "rename", "cancel"],
                default="rename",
            )
            if action == "overwrite":
                break
            if action == "cancel":
                console.print(
                    "[dim]Cancelled — nothing written. The map itself wasn't lost; "
                    "rerun the export with a different --out to save it.[/]"
                )
                raise typer.Exit(code=1)
            new_path = Prompt.ask("  New output path", default=str(out.with_stem(out.stem + "_2")))
            out = Path(new_path)
    written = write_opml(mm, out) if fmt == "opml" else write_xmind(mm, out)

    if not json_mode():
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

    return written, relationships_dropped


def run() -> None:
    """Entry point with graceful Ctrl+C handling (the wizard advertises this)."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/]")
        raise typer.Exit(code=130)


if __name__ == "__main__":
    run()
