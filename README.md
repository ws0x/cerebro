# cerebro

Turn video content into **XMind-compatible smart mind maps** — not flat summaries,
but structured, hierarchical knowledge maps.

Give it a YouTube URL, a local subtitle track, or a course folder; get back a
mind map you can open directly in XMind (via OPML today, native `.xmind` soon).

```
cerebro map "https://youtu.be/VIDEO_ID" --level full
cerebro map examples/intro_to_neural_networks.vtt --level expert
cerebro batch "https://youtube.com/playlist?list=..." --level full --limit 10
cerebro batch path/to/course_folder --format xmind
```

## Architecture

The whole design turns on one decision: **the model never writes a file format.**
Every source is normalized to a `Transcript`; a structurer turns that into a
format-agnostic **IR** (`MindMap`); deterministic converters turn the IR into
OPML / XMind / Markdown. Swap the model or the output format without touching
anything else.

```
source ──▶ ingest ──▶ Transcript ──▶ structure ──▶ MindMap (IR) ──▶ convert ──▶ .opml / .xmind
           (yt /                      (heuristic /                   (deterministic)
            subs)                      LLM)
```

Modules:

| Module | Role |
|---|---|
| `cerebro.ingest` | any source → `Transcript` (YouTube captions, `.srt/.vtt/.txt`) |
| `cerebro.transcript` | the `Transcript` contract |
| `cerebro.structure` | `Transcript` → `MindMap` IR (heuristic now, LLM next) |
| `cerebro.ir` | the `MindMap` intermediate representation |
| `cerebro.convert` | IR → OPML (XMind writer next) |
| `cerebro.ui` | Rich banner, progress, in-terminal map preview |
| `cerebro.cli` | Typer entry point |

## Processing levels

| Level | Depth | Pipeline |
|---|---|---|
| `brief` | main topics only | segment → reduce |
| `full` | subtopics + key points | segment → map → reduce |
| `expert` | + relationships, notes, insights | full + cross-link detection |

## Engines

Set a free key in `.env` (see `.env.example`) and pick an engine:

```
cerebro map <src> --engine groq      # free, fast — console.groq.com/keys
cerebro map <src> --engine gemini    # free — aistudio.google.com/apikey
cerebro map <src> --engine auto      # groq→gemini→heuristic, whatever's available
cerebro map <src> --engine heuristic # offline, no key, deterministic
```

The LLM engine runs **map → reduce → link**: extract each segment, merge into a
smart hierarchy, then (expert level) detect cross-branch relationships. Every
call is content-addressed cached, so re-runs and level upgrades are near-free.
If no key is present or a call fails, it degrades gracefully to the heuristic.

Output formats: `--format opml` (universal, imports everywhere) or
`--format xmind` (native — keeps relationships + markers/icons).

## Batch

`cerebro batch <playlist-url-or-folder>` fans out ingest+structure across every
video/lesson concurrently (`--workers`, default 3; capped inner LLM concurrency
per video to protect free-tier rate limits), then merges each into one combined
map as a branch under a shared root. A failing item (private video, no
captions, missing subtitle) is reported and skipped — never fatal. Course
folders match videos to sidecar subtitle files by filename and report any
video with no match. `--limit N` caps how many items are processed.

## Local video

`.mp4`/`.mkv`/`.mov`/`.webm`/`.avi`/`.m4v` are accepted directly by `map` and
`batch`. Cerebro tries, in order: (1) an embedded text subtitle track
(subrip/ass/webvtt/mov_text), demuxed via ffmpeg — fast, exact; (2) Whisper
transcription of the audio track (`pip install cerebro[whisper]`) — slower,
fully offline, works on anything with speech. Image-based subtitle codecs
(PGS/VobSub, common in some ripped `.mkv`s) aren't supported — they'd need OCR.

## Status

Done and **live-validated end-to-end** (real Groq + Gemini keys, real YouTube
playlist, real ffmpeg video processing): YouTube / subtitle / local-video
ingest (with Whisper fallback) → heuristic **and** LLM (Groq/Gemini)
structurer → **OPML and native `.xmind`** export (relationships + markers) →
**batch** (playlists + course folders, concurrent, partial-failure-safe). Rich
CLI with live progress bar. Content-addressed cache. 19 tests pass — this
covers every input source and output format from the original spec.

## Develop

```
py -3 -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\pytest
```
