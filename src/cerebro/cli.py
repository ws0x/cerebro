"""Cerebro command-line interface."""

from __future__ import annotations

import re
import sys
import time
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
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.progress import Progress as RichProgress
from rich.table import Table

from . import __version__
from .batch import BatchItem, run_batch
from .cache import Cache
from .convert import write_opml, write_xmind
from .ingest import load_transcript
from .ingest.folder import discover_course_sources
from .ingest.playlist import is_playlist_url, load_playlist
from .llm.base import LLMError
from .llm.config import ConfigError, load_env, resolve_provider
from .paths import ensure_output_dir, load_config
from .structure import HeuristicStructurer
from .structure.llm import LLMStructurer, link_relationships
from .ui import print_banner, print_preview
from .wizard import run_wizard

_EPILOG = (
    'Examples: cerebro (wizard)  |  cerebro map "URL" -l expert  |  '
    'cerebro batch "playlist URL" --limit 10  |  cerebro batch ./course_folder --format xmind'
)

app = typer.Typer(
    add_completion=False,
    help="Turn video content into XMind-compatible smart mind maps. Run with no arguments for a guided wizard.",
    epilog=_EPILOG,
)
console = Console()

_HELP_REQUESTED = "--help" in sys.argv[1:]


def _safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
    return (name or "mindmap")[:80]


def _structure(transcript, level, provider, cache, relationship_limit=8):
    """Build the map with the resolved engine, showing live progress and
    falling back to the offline heuristic if an LLM call fails."""
    if provider is None:
        with console.status(f"[cyan]Structuring ({level})…", spinner="dots"):
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


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
):
    """Cerebro root."""
    if _HELP_REQUESTED:
        return  # a --help lookup is a reference check, not a real run
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
):
    """Build a mind map from SOURCE and write it to disk."""
    _do_map(source, level, fmt, out, engine, no_cache, preview, whisper_model, relationship_limit)


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
        with console.status("[cyan]Loading transcript…", spinner="dots"):
            transcript = load_transcript(source, whisper_model=whisper_model, cache=cache)
    except Exception as exc:
        console.print(f"[red]✗ Failed to load transcript: {exc}[/]")
        raise typer.Exit(code=1)
    console.print(
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
        console.print("[yellow]![/] No API key found — using the offline heuristic engine.")

    mm = _structure(transcript, level, provider, cache, relationship_limit=relationship_limit)
    console.print(
        f"[green]✓[/] Map built with [bold]{engine_label}[/]: "
        f"{mm.node_count()} nodes, depth {mm.depth()}"
        + (f", {len(mm.relationships)} relationships" if mm.relationships else "")
    )

    if preview:
        console.print()
        print_preview(mm)
        console.print()

    _export(mm, fmt, out, level, time.perf_counter() - t0)


@app.command()
def batch(
    source: str = typer.Argument(..., help="YouTube playlist URL or local course-folder path."),
    level: str = typer.Option(None, "--level", "-l", help="brief | full | expert"),
    fmt: str = typer.Option(None, "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option(None, "--engine", "-e", help="auto | groq | gemini | heuristic"),
    workers: int = typer.Option(3, "--workers", "-w", help="Videos/lessons processed concurrently."),
    limit: int = typer.Option(None, "--limit", help="Process only the first N items."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
    whisper_model: str = typer.Option(None, "--whisper-model", help="Whisper model size to use for local video transcription."),
    relationship_limit: int = typer.Option(None, "--relationship-limit", "--rel-limit", help="Max number of relationships to detect in expert mode."),
):
    """Build one combined mind map from a YouTube playlist or a local course folder."""
    _do_batch(source, level, fmt, out, engine, workers, limit, no_cache, preview, whisper_model, relationship_limit)


def _do_batch(
    source: str,
    level: str | None,
    fmt: str | None,
    out: Path | None,
    engine: str | None,
    workers: int,
    limit: int | None,
    no_cache: bool,
    preview: bool,
    whisper_model: str | None = None,
    relationship_limit: int | None = None,
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
        with console.status("[cyan]Reading playlist…", spinner="dots"):
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

    console.print(f"[green]✓[/] Found [bold]{total_found}[/] item(s) in [bold]{title}[/]")
    if limit is not None and total_found > limit:
        console.print(f"[dim]  Processing first {len(items)} (--limit {limit}).[/]")
    if transcribe_count:
        console.print(
            f"[dim]  {transcribe_count} video(s) have no subtitle file — will extract an "
            "embedded track or transcribe with Whisper (slower).[/]"
        )

    try:
        provider = resolve_provider(engine)
    except ConfigError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(code=1)

    engine_label = "heuristic (offline)" if provider is None else f"{provider.name}:{provider.model}"
    if provider is None and engine == "auto":
        console.print("[yellow]![/] No API key found — using the offline heuristic engine.")

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

        combined, outcomes = run_batch(
            items, structurer_factory, level, title, max_workers=workers, on_event=on_event, cache=cache, whisper_model=whisper_model
        )

    # Each video already got its own within-video links (if any) from its own
    # expert-level structuring above; this second pass looks across all of
    # them together, so a concept in lesson 2 can connect to one in lesson 7.
    if level == "expert" and provider is not None and combined.node_count() > 3:
        with console.status("[cyan]Finding connections across videos…", spinner="dots"):
            link_relationships(
                combined, provider, cache, cross_video=True, relationship_limit=relationship_limit
            )

    ok_count = sum(1 for o in outcomes if o.mindmap is not None)
    console.print(
        f"[green]✓[/] Processed {ok_count}/{len(items)} item(s) with [bold]{engine_label}[/]: "
        f"{combined.node_count()} nodes, depth {combined.depth()}"
        + (f", {len(combined.relationships)} relationships" if combined.relationships else "")
    )
    for label, error in failures:
        console.print(f"[yellow]![/] {label}: {error}")

    if preview:
        console.print()
        print_preview(combined, max_depth=4)
        console.print()

    _export(combined, fmt, out, level, time.perf_counter() - t0)


@app.command()
def interactive():
    """Guided wizard: pick a source, level, engine, and format step by step."""
    # print_banner()/load_env() already ran in the _main callback, which fires
    # for every invocation regardless of which subcommand was requested.
    run_wizard(_do_map, _do_batch)


cache_app = typer.Typer(add_completion=False, help="Inspect or clear the response cache.")
app.add_typer(cache_app, name="cache")


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


def _export(mm, fmt: str, out: Path | None, level: str, elapsed: float) -> None:
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
    written = write_opml(mm, out) if fmt == "opml" else write_xmind(mm, out)

    summary = Table.grid(padding=(0, 2))
    summary.add_row("[dim]Output[/]", f"[bold]{written}[/]")
    summary.add_row("[dim]Format[/]", fmt.upper())
    summary.add_row("[dim]Level[/]", level)
    summary.add_row("[dim]Time[/]", f"{elapsed:.2f}s")
    console.print(Panel(summary, title="[green]Done[/]", border_style="green", expand=False))
    if fmt == "opml":
        console.print(f"[dim]Import into XMind: File → Import → OPML → {written.name}[/]")
    else:
        console.print(f"[dim]Open directly in XMind: {written.name}[/]")


def run() -> None:
    """Entry point with graceful Ctrl+C handling (the wizard advertises this)."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/]")
        raise typer.Exit(code=130)


if __name__ == "__main__":
    run()
