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
from .structure import HeuristicStructurer
from .structure.llm import LLMStructurer
from .ui import print_banner, print_preview

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn video content into XMind-compatible smart mind maps.",
)
console = Console()


def _safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
    return (name or "mindmap")[:80]


def _structure(transcript, level, provider, no_cache):
    """Build the map with the resolved engine, showing live stages and falling
    back to the offline heuristic if an LLM call fails."""
    if provider is None:
        with console.status(f"[cyan]Structuring ({level})…", spinner="dots"):
            return HeuristicStructurer().structure(transcript, level=level)

    with console.status("[cyan]Thinking…", spinner="dots") as status:
        def on_event(kind, **d):
            if kind == "map_start":
                status.update(f"[cyan]Mapping {d['total']} segment(s)…")
            elif kind == "map_progress":
                status.update(f"[cyan]Mapping {d['done']}/{d['total']} segment(s)…")
            elif kind == "reduce_start":
                status.update("[cyan]Reducing into a hierarchy…")
            elif kind == "link_start":
                status.update("[cyan]Detecting cross-links…")

        structurer = LLMStructurer(provider, Cache(enabled=not no_cache), on_event=on_event)
        try:
            return structurer.structure(transcript, level=level)
        except LLMError as exc:
            status.stop()
            console.print(
                f"[yellow]! LLM engine failed ({exc}); falling back to heuristic.[/]"
            )
            return HeuristicStructurer().structure(transcript, level=level)


def _version_callback(value: bool):
    if value:
        console.print(f"cerebro {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
):
    """Cerebro root."""


@app.command()
def map(
    source: str = typer.Argument(
        ..., help="YouTube URL or local .srt/.vtt/.txt/.mp4/.mkv/.mov/.webm file."
    ),
    level: str = typer.Option("full", "--level", "-l", help="brief | full | expert"),
    fmt: str = typer.Option("opml", "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option("auto", "--engine", "-e", help="auto | groq | gemini | mock | heuristic"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
):
    """Build a mind map from SOURCE and write it to disk."""
    print_banner()
    load_env()
    t0 = time.perf_counter()

    try:
        with console.status("[cyan]Loading transcript…", spinner="dots"):
            transcript = load_transcript(source)
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

    mm = _structure(transcript, level, provider, no_cache)
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
    level: str = typer.Option("full", "--level", "-l", help="brief | full | expert"),
    fmt: str = typer.Option("opml", "--format", "-f", help="opml | xmind"),
    out: Path = typer.Option(None, "--out", "-o", help="Output file path."),
    engine: str = typer.Option("auto", "--engine", "-e", help="auto | groq | gemini | mock | heuristic"),
    workers: int = typer.Option(3, "--workers", "-w", help="Videos/lessons processed concurrently."),
    limit: int = typer.Option(None, "--limit", help="Process only the first N items."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Show the map in-terminal."),
):
    """Build one combined mind map from a YouTube playlist or a local course folder."""
    print_banner()
    load_env()
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
        return HeuristicStructurer() if provider is None else LLMStructurer(provider, cache, max_workers=2)

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
            items, structurer_factory, level, title, max_workers=workers, on_event=on_event
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
        out = Path.cwd() / f"{_safe_filename(mm.title)}.{fmt}"
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


if __name__ == "__main__":
    app()
