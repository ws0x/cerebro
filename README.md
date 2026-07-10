<div align="center">

# cerebro

**Turn video and PDFs into structured knowledge — not just a summary.**

Point it at a YouTube video, a whole playlist, a course folder, a local video
file, or a PDF. Get back a hierarchical, XMind-compatible mind map with real
structure: topics, sub-points, cross-references, and icons — built by an LLM
that *understands* the content, not a transcript-slicer that pretends to.

[![CI](https://github.com/ws0x/cerebro/actions/workflows/ci.yml/badge.svg)](https://github.com/ws0x/cerebro/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)

</div>

---

```
┌───────────────────────────────────────┐
│                                       │
│   ___  ___  ___  ___  ___  ___  ___   │
│  / __|| __|| _ \| __|| _ )| _ \/ _ \  │
│ | (__ | _| |   /| _| | _ \|   / (_) | │
│  \___||___||_|_\|___||___/|_|_\\___/  │
│     content  ->  smart mind maps      │
│                                       │
└─────────────── cerebro ───────────────┘
```

## Table of contents

- [Why cerebro](#why-cerebro)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Two ways to use it](#two-ways-to-use-it)
- [Command reference](#command-reference)
- [Processing levels](#processing-levels)
- [Choosing an engine](#choosing-an-engine)
- [Output formats: OPML vs. XMind](#output-formats-opml-vs-xmind)
- [Batch: playlists & course folders](#batch-playlists--course-folders)
- [Local video: embedded subtitles & Whisper](#local-video-embedded-subtitles--whisper)
- [PDF files](#pdf-files)
- [Folder structure maps (`cerebro tree`)](#folder-structure-maps-cerebro-tree)
- [Caching](#caching)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [How it works](#how-it-works)
- [Development](#development)
- [License](#license)

## Why cerebro

Most "video to notes" tools do one thing: dump a transcript through a
summarizer and hand you a wall of bullet points. That's not a mind map — it's
a shorter wall of text.

cerebro is built around a different idea: a real map-reduce-link pipeline that
reads a transcript the way a person would while taking notes — grouping
related ideas, promoting recurring themes into parents, demoting details into
leaves, and drawing connections *across* branches that a flat summary would
never surface. The output is a genuine hierarchy you can drop straight into
XMind and start editing.

A few things that make it worth trying:

- **Actually smart, not just extractive.** Real map → reduce → link pipeline, with an explicit anti-hallucination grounding rule so it doesn't invent facts your source never said.
- **Free by default.** Works with free-tier [Groq](https://console.groq.com/keys) or [Gemini](https://aistudio.google.com/apikey) keys — no paid API required.
- **Works fully offline too.** No key at all → falls back to a deterministic heuristic engine. No internet for the *video* → local files, embedded subtitles, and Whisper transcription all work with zero network calls.
- **Every real source.** Single YouTube videos, whole playlists, local course folders, local video files (with or without subtitles), and PDF files.
- **Two honest output formats.** Universal OPML (imports everywhere) or native `.xmind` (keeps relationship arrows and icons that OPML physically can't represent).
- **Batch-safe.** A 40-video playlist doesn't die because one video is private — failures are reported per-item, never fatal.
- **Fast.** Concurrent ingestion, concurrent LLM calls, and a content-addressed cache mean re-runs and level upgrades (brief → full → expert) cost almost nothing.

## Requirements

- **Python 3.10+** (only if installing via pip/pipx — the standalone binary needs nothing)
- **[ffmpeg](https://ffmpeg.org/download.html)** on your `PATH` — required for local video files (embedded subtitle extraction and audio extraction for Whisper)
- A free **[Groq](https://console.groq.com/keys)** or **[Gemini](https://aistudio.google.com/apikey)** API key for smart structuring (optional — cerebro works without one, just less "smart")

## Installation

Pick whichever fits how you work.

### Option 1 — pipx (recommended)

The cleanest way to get a global `cerebro` command without touching your
system Python:

```bash
pipx install git+https://github.com/ws0x/cerebro.git
cerebro --version
```

### Option 2 — pip

```bash
pip install git+https://github.com/ws0x/cerebro.git
```

### Option 3 — standalone binary (no Python required)

For Windows users who don't want a Python environment at all. Build it
yourself from source (see [Development](#development)) — it produces a single
`cerebro.exe` you can drop anywhere and run directly. The binary supports
everything **except** Whisper transcription (that path needs the separate
`[whisper]` extra); embedded-subtitle extraction still works since it's just
an `ffmpeg` call.

### Option 4 — from source (for development)

```bash
git clone https://github.com/ws0x/cerebro.git
cd cerebro
python -m venv .venv

# Windows
.venv\Scripts\pip install -e ".[dev]"

# macOS / Linux
.venv/bin/pip install -e ".[dev]"
```

### Want offline Whisper transcription too?

```bash
pip install "cerebro[whisper]"
```

This pulls in `faster-whisper` (a heavier, optional dependency) so cerebro can
transcribe local video that has no subtitle track at all — fully offline,
no API key.

## Quick start

**1. Grab a free API key** (30 seconds, either one works):

| Provider | Get a key | Notes |
|---|---|---|
| Groq | [console.groq.com/keys](https://console.groq.com/keys) | Fastest — Llama 3.3 70B, typically 2–5s per video |
| Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Slower but stays closer to the literal source text |

**2. Save it — where depends on how you installed cerebro:**

```bash
# Running from source / a clone (repo-local, only works in that directory):
cp .env.example .env

# Installed via pipx/pip and run as a global `cerebro` command from anywhere
# (recommended — this is the one that actually matters once installed):
mkdir -p ~/.cerebro && cp .env.example ~/.cerebro/.env
# then edit either file, paste your key into GROQ_API_KEY= or GEMINI_API_KEY=
```

cerebro checks the current directory's `.env` first, then falls back to
`~/.cerebro/.env`. If you set a key in one but not the other and get **"No
API key found — using the offline heuristic engine"** even though you know
you configured one, this is almost always why — check you saved it to the
one cerebro actually reads from wherever you're running it.

**3. Run cerebro with no arguments** — the wizard walks you through the rest:

```bash
cerebro
```

```
─────────────────────────────────── Source ────────────────────────────────────
Ctrl+C to cancel anytime

Paste a YouTube URL, playlist URL, or local file/folder path: examples/intro_to_neural_networks.vtt
✓ Detected: local file

─────────────────────────────────── Options ───────────────────────────────────
Processing level  →  Full — subtopics + key points (recommended)
Engine             →  Auto — Groq/Gemini if a key is set, else offline
Output format       →  OPML — imports into XMind, Freemind, most outliners
Output path (mindmap.opml): my_first_map.opml
───────────────────────────────────────────────────────────────────────────────
┌──────────────────── Ready ────────────────────┐
│ Source  examples/intro_to_neural_networks.vtt │
│ Type    local file                            │
│ Level   full                                  │
│ Engine  auto                                  │
│ Format  OPML                                  │
│ Output  my_first_map.opml                     │
└───────────────────────────────────────────────┘
Proceed?  →  Yes, build it

✓ Transcript: Intro To Neural Networks — 257 words, 10 segments
✓ Map built with groq:llama-3.3-70b-versatile: 6 nodes, depth 4

🧠 Neural Networks
└── ◆ Neural Network Basics
    ├── 🔑 Neural Network Function
    └── ○ Training Process
        ├── ⚠️ Overfitting Prevention
        └── ✨ Data Quality Importance

┌────────── Done ───────────┐
│ Output  my_first_map.opml │
│ Format  OPML              │
│ Level   full              │
│ Time    2.13s             │
└───────────────────────────┘
Import into XMind: File → Import → OPML → my_first_map.opml
```

**4. Open it in XMind:** `File → Import → OPML → my_first_map.opml`. Done.

## Two ways to use it

### The wizard (recommended if you don't want to remember flags)

```bash
cerebro
# or explicitly:
cerebro interactive
```

Paste a source, pick your options with arrow keys, confirm. It detects what
you pasted — single video, playlist, course folder, or local file — and
routes to exactly the same pipeline the flag-driven commands use underneath,
so there's no behavioral difference, only convenience.

If the "Ready" summary isn't right, choose **Edit an answer** instead of
proceeding — it jumps straight to the field you want to change and comes back
to the summary, no need to start over. In a real terminal, every choice
(level, engine, format, proceed/edit/cancel) is an arrow-key menu; if
cerebro can't detect a real interactive terminal (piped input, some
CI/scripting contexts), it automatically falls back to a plain typed-choice
prompt instead of failing.

### Flags (recommended for scripting or repeat use)

```bash
cerebro map "https://youtu.be/dQw4w9WgXcQ" --level expert --format xmind
cerebro batch "https://youtube.com/playlist?list=..." --limit 10
```

Every option below applies to both.

## Command reference

### Global flags

These come before the subcommand (`cerebro --no-color doctor`, not
`cerebro doctor --no-color`) and apply everywhere:

| Flag | Meaning |
|---|---|
| `--no-color` | Disable ANSI color. The [`NO_COLOR`](https://no-color.org) env var works too, without the flag. |
| `--ascii` | Plain ASCII glyphs instead of emoji/pictographic icons — for terminals and screen readers that handle them poorly. |
| `--theme` | `default` \| `high-contrast` — high-contrast drops dim/low-emphasis styling in favor of your terminal's own default foreground. |
| `--version` | Show version and exit. |
| `--install-completion` | Install tab-completion for commands and options in your current shell (bash/zsh/fish/PowerShell). Restart your shell afterward. |
| `--quiet`, `-q` | Suppress the banner and informational status lines (map/batch/tree). Errors, warnings, and the final result still print — this drops decoration, not answers. |
| `--json` | Print one JSON result object on stdout instead of Rich output (`map`/`batch`/`tree`/`doctor`). Implies `--quiet`, and disables all progress bars/spinners too (Rich's live-redraw only erases cleanly on a real terminal — redirected to a file, it leaves stray text behind). Errors become `{"ok": false, "error": "...", "fix": "..."}` instead of exiting with a red X. |

### `cerebro map SOURCE [options]`

Build a mind map from a single source.

`SOURCE` — a YouTube URL, or a local `.srt` / `.vtt` / `.txt` / `.mp4` /
`.mkv` / `.mov` / `.webm` / `.avi` / `.m4v` / `.pdf` file.

| Flag | Default | Meaning |
|---|---|---|
| `--level`, `-l` | `full` | `brief` \| `full` \| `expert` — see [Processing levels](#processing-levels) |
| `--engine`, `-e` | `auto` | `auto` \| `groq` \| `gemini` \| `heuristic` — see [Choosing an engine](#choosing-an-engine) |
| `--format`, `-f` | `opml` | `opml` \| `xmind` — see [Output formats](#output-formats-opml-vs-xmind) |
| `--out`, `-o` | `~/cerebro-maps/<title>.<format>` | Output file path |
| `--no-cache` | off | Disable the response cache for this run |
| `--preview` / `--no-preview` | preview on | Show/hide the in-terminal tree before writing the file |
| `--yes`, `-y` | off | Overwrite an existing output file without asking |
| `--dry-run` | off | *(batch/tree only)* Show what would be reused vs. freshly processed, without spending any API calls or writing output |

If you don't pass `--out`, cerebro writes to a dedicated `~/cerebro-maps/`
folder (created automatically) named after the source's title — not the
current directory, so files don't scatter across wherever you happened to run
the command from.

### `cerebro batch SOURCE [options]`

Build **one combined** mind map from a YouTube playlist or a local course
folder — every item becomes a branch under a shared root.

`SOURCE` — a YouTube playlist URL, or a local folder path.

All `map` flags apply, plus:

| Flag | Default | Meaning |
|---|---|---|
| `--workers`, `-w` | `3` | How many videos/lessons process concurrently |
| `--limit` | *(none)* | Only process the first N items — useful for a big playlist |

### `cerebro tree PATH [options]`

Map a folder's directory structure — see
[Folder structure maps](#folder-structure-maps-cerebro-tree).

| Flag | Default | Meaning |
|---|---|---|
| `--engine`, `-e` | `heuristic` | `heuristic` (free/instant) \| `groq` \| `gemini` — AI folder-purpose labels |
| `--max-depth` | `8` | Maximum folder nesting depth |
| `--max-files` | `20` | Max files listed per folder before collapsing to a count |
| `--no-gitignore` | off | Don't respect the folder's `.gitignore` |
| `--format`, `-f` / `--out`, `-o` / `--no-cache` / `--preview` | *(same as `map`)* | |

### `cerebro interactive`

Launches the guided wizard explicitly (same as running `cerebro` with no
arguments).

### `cerebro cache stats` / `cerebro cache clear`

Inspect or wipe the response cache — see [Caching](#caching).

### `cerebro --version`

Prints the installed version and exits.

## Processing levels

Pick how deep the map should go. Each level is a genuinely different pipeline,
not just a truncated version of the next one up.

| Level | What you get | Pipeline |
|---|---|---|
| `brief` | Main topics only, minimal nesting — a fast overview | segment → reduce |
| `full` | Subtopics and key points, 3–4 levels deep | segment → map → reduce |
| `expert` | Everything in `full`, plus concepts, examples, actionable insights, **and cross-branch relationship arrows** | full + link detection |

Only `expert` produces relationships (the labeled arrows connecting nodes in
different branches — e.g. "Overfitting *prevented by* Regularization"), which
is why `--format xmind` matters most at that level: OPML has no way to
represent them.

## Choosing an engine

| Engine | Cost | Speed | Notes |
|---|---|---|---|
| `auto` *(default)* | free | — | Uses Groq if `GROQ_API_KEY` is set, else Gemini, else falls back to `heuristic` |
| `groq` | free tier | fastest (typically 2–5s) | Llama 3.3 70B via [Groq](https://console.groq.com/keys) |
| `gemini` | free tier | slower (~20–30s) | Gemini 2.5 Flash via [Google AI Studio](https://aistudio.google.com/apikey), stays closer to the literal source wording |
| `heuristic` | free, no key | instant | Fully offline, deterministic sentence-chunking — no AI, structurally correct but not "smart" |

Set whichever key(s) you have in `.env` (copy `.env.example` to get started).
If a live call fails mid-run — bad key, rate limit, network blip — cerebro
**automatically falls back to the heuristic engine** rather than crashing, so
a run never just dies.

## Output formats: OPML vs. XMind

| | OPML | Native `.xmind` |
|---|---|---|
| Hierarchy & content | ✅ full fidelity | ✅ full fidelity |
| Notes & timestamps | ✅ | ✅ |
| Markers / icons (🔑 💡 ⚠️ ✅ ✨) | ❌ | ✅ |
| Relationship arrows (`expert` level) | ❌ | ✅ |
| Opens in | XMind, Freemind, MindNode, Workflowy, most outliners | XMind only (native) |
| Import step | `File → Import → OPML` | Just double-click |

**Rule of thumb:** use `opml` for `brief`/`full` maps or if you want maximum
compatibility; use `xmind` for `expert` maps so you don't lose the
relationships and icons the model worked out. cerebro warns you if you export
an `expert` map as OPML and would be dropping relationships.

## Batch: playlists & course folders

```bash
cerebro batch "https://youtube.com/playlist?list=PL..." --level full --limit 20
cerebro batch ./my_course_folder --format xmind
```

- **YouTube playlists** are listed without downloading anything (fast, via
  `yt-dlp`'s flat extraction), then every video is ingested and structured
  concurrently.
- **Course folders** match each video file to a same-named subtitle file
  (`lesson1.mp4` + `lesson1.srt`) when present. Videos with no sidecar
  subtitle aren't skipped — they're processed via embedded-subtitle
  extraction or Whisper, cerebro just tells you up front that those will be
  slower. PDFs in the same folder need no pairing — each is included as its
  own lesson and keeps its own real structure (see [PDF files](#pdf-files)),
  not flattened into the others.
- **Files/lessons are numbered-sort aware** — "Lesson 2" sorts before "Lesson
  10", not after.
- **One bad item never kills the batch.** A private video, a missing
  transcript, a corrupt file — each failure is caught, reported by name, and
  skipped. You get a combined map from whatever succeeded.
- Every successful item becomes its own top-level branch in the final map,
  titled after the source (playlist video title or lesson filename).

**Reruns are incremental by default.** cerebro remembers which items it
successfully processed for a given playlist/folder (`~/.cerebro/batch-snapshots/`)
and reuses their branches as-is on a rerun — no transcript refetch, no
restructuring — so only genuinely new items (added to the playlist, or that
failed last time and are worth retrying) get processed. Every run reports the
delta:

```
✓ Processed 2/2 item(s) with groq:llama-3.3-70b-versatile: 46 nodes, depth 6
  ↻ Reused 1/2 item(s) since 2026-07-10T04:10:42+00:00 — 1 new.
```

An item that failed last run (private video, rate limit, transient network
error) is never treated as "reused" — it wasn't successfully cached, so it's
retried fresh, same as a genuinely new item. Pass `--fresh` to ignore history
and reprocess everything regardless.

## Local video: embedded subtitles & Whisper

Point `map` or `batch` straight at a `.mp4`/`.mkv`/`.mov`/`.webm`/`.avi`/`.m4v`
file — no manual subtitle extraction needed. cerebro tries, in order:

1. **Embedded text subtitle track** (`subrip`/`ass`/`webvtt`/`mov_text`
   codecs) — demuxed via `ffmpeg`. Fast, exact, no AI involved.
2. **Whisper transcription** of the audio — fully offline, works on anything
   with speech, needs `pip install cerebro[whisper]`.

Image-based subtitle formats (PGS/VobSub, common in some ripped `.mkv` files)
aren't supported — those need OCR, which is out of scope.

Whisper transcriptions are cached, so re-processing the same file (e.g. at a
different level) never re-transcribes.

## PDF files

```bash
cerebro map textbook.pdf
cerebro map textbook.pdf --level expert --format xmind
```

A PDF is unlike a video transcript: it usually already **has** real structure
(chapter/section bookmarks, or at least visually distinct headings) instead of
needing one invented from flat text. cerebro exploits that instead of
throwing it away — the same judgment call `tree` makes for folders (the
structure is *known*; AI only *enriches*, it never has to *discover*):

1. **TOC/bookmarks**, if the PDF has them — used directly, exact.
2. **Font-size heading detection**, if there's no TOC — headings are
   identified by being notably larger than the document's body text and
   short; a repeated identical line across many pages (a running header or
   footer) is filtered out rather than mistaken for real structure. If the
   signal isn't clearly trustworthy (too few candidates, or everything the
   same size), cerebro deliberately reports no structure rather than guess.
3. **No structure found** → falls back to the same map → reduce → link
   pipeline used for video transcripts, unchanged.

When real structure is found, the heading hierarchy becomes the map's real
hierarchy — an LLM (if you have a key configured) is only used to extract each
section's key points and summary, never to invent the outline. `--level brief`
skips AI entirely even with a key configured (skeleton only, near-instant);
`full` adds per-section extraction; `expert` additionally detects cross-section
relationships.

A course folder can freely mix PDF handouts/slides in with video lessons —
`cerebro batch ./course_folder` picks up both, and each PDF keeps its own
real structure inside its branch of the combined map, same as it would via a
standalone `cerebro map`.

**Not supported:** scanned/image-only PDFs (no text layer — would need OCR),
tables and images within a PDF (text only), and password-protected PDFs.

## Folder structure maps (`cerebro tree`)

A completely different thing from `batch` — `batch` treats a folder as a
*course of video lessons*; `tree` treats it as a *directory to map*, no video
involved at all:

```bash
cerebro tree ./my_project
cerebro tree ./my_project --engine groq   # AI-inferred folder purpose labels
```

The hierarchy doesn't need discovering (the filesystem already gives it to
you), so this defaults to `--engine heuristic` — instant, free, fully
offline — unlike `map`/`batch` where AI is the default. Pass `--engine groq`
or `--engine gemini` to opt into a purpose label per folder (e.g. `auth/` →
"Authentication & session handling") inferred from its name and immediate
contents; the folder's real name stays the node title, the inferred purpose
goes in the note, so both are visible.

Noise is filtered automatically: `.git`, `node_modules`, `__pycache__`,
`.venv`, build output, and similar are always skipped, and the folder's own
`.gitignore` is respected too (`--no-gitignore` to disable that). Very large
folders are kept usable with `--max-files` (default 20 per folder, collapses
the rest to a count) and `--max-depth` (default 8).

**Reruns are incremental by default.** cerebro remembers the last map it
built for a given folder (`~/.cerebro/tree-snapshots/`) using a signature per
directory that depends on everything beneath it — a change three levels down
invalidates that folder's signature and every signature back up to the root,
so nothing is ever missed. Unchanged folders keep whatever AI label they
already had and are never resubmitted; only what's actually new or changed
gets relabeled. Every run reports the delta:

```
✓ Walked ./my_project: 214 nodes, depth 6
  ↻ Reused 41/44 folder(s) since 2026-07-10T03:27:46+00:00 — 3 changed.
✓ Labeled 3 folder(s) with groq:llama-3.3-70b-versatile
```

Pass `--fresh` to ignore any previous map and rebuild everything from
scratch (a new snapshot is still saved afterward, so the *next* run can go
back to being incremental).

## Guided key setup (`cerebro setup`)

```bash
cerebro setup
```

Prompts for your Groq/Gemini API keys and writes `~/.cerebro/.env` —
replaces hand-editing that file as the only way to configure a key. Enter
skips a key (e.g. to use only one engine, or to stick with the fully
offline heuristic engine); leaving one blank keeps whatever was already
saved for it. Input is masked when run in a real terminal.

## Persisted defaults (`cerebro config`)

```bash
cerebro config list             # every key, persisted value or built-in default
cerebro config get level
cerebro config set level expert
cerebro config unset level      # revert to the built-in default
```

Persists to `~/.cerebro/config.json` — the same file `map`/`batch`/`tree`
already read defaults from, previously only editable by hand. Keys: `level`,
`format`, `engine`, `whisper_model`, `relationship_limit`. `set` validates
the value against that key's allowed choices before writing.

## Diagnosing your setup (`cerebro doctor`)

```bash
cerebro doctor              # full check, including API/YouTube reachability
cerebro doctor --no-network # skip reachability checks (faster, works offline)
```

Reports on everything that affects whether a run will succeed: which API
keys are set (never their values), whether `ffmpeg`/`ffprobe`/Whisper are
available for local video, whether every storage directory is actually
writable (not just present), cache/snapshot counts, and core dependencies.
Every check degrades independently — a missing optional piece like Whisper
or a second engine's key is reported as an advisory (`!`), not a failure;
`cerebro doctor` only exits non-zero on a hard failure (an unsupported
Python version, a missing core dependency, an unwritable storage path).

## Full-page dashboard (`cerebro dashboard`)

```bash
cerebro dashboard
```

Takes over the whole terminal viewport — the alternate screen buffer, the
same mechanism `less`/`git diff`/`htop` use — instead of scrolling more
text into your history. One screen: setup health (only what needs
attention) side by side with everything cerebro remembers (cache +
snapshots). `Enter` refreshes, `q` quits, and your previous terminal
content is restored on exit. Falls back to a single static render when
there's no real attached terminal (piped output, CI).

## Checking what cerebro remembers (`cerebro status`)

```bash
cerebro status
```

One place to see everything: response cache size, and every folder/playlist
with saved incremental history, each with its source path/URL and when it
was last built — the list you need before reaching for `cerebro forget`.
Complements `cerebro doctor`, which checks whether your setup will *work*
rather than what it's already *done*.

## Forgetting incremental history (`cerebro forget`)

```bash
cerebro forget tree ./my_project        # same path as given to `cerebro tree`
cerebro forget batch "playlist URL"     # same source as given to `cerebro batch`
```

`--fresh` ignores history for one run but still saves a new snapshot
afterward. `forget` instead deletes the saved snapshot outright — the next
run is treated as if that folder/playlist had never been mapped before, with
no effect on the response cache or on any other folder/playlist's history.

## Caching

Every expensive step — YouTube caption fetches, Whisper transcription, a
map/reduce/link LLM call — is cached. Practical effect: re-running the same
video, or upgrading `brief` → `full` → `expert`, reuses everything it can and
only pays for what actually changed. Pass `--no-cache` to force a clean run.
YouTube captions specifically are cached by video ID + language, since a
published video's captions essentially never change — no reason to hit
YouTube again for a video cerebro has already fetched.

The cache lives at `~/.cerebro/cache` — a stable location so it's actually
shared across runs regardless of which directory you're in when you invoke
the globally-installed `cerebro` command:

```bash
cerebro cache stats       # location, entry count, total size
cerebro cache clear       # wipe it (asks for confirmation unless --yes)
```

## Examples

The [`examples/`](examples/) folder has real, live-generated output you can
inspect or import directly — subtitle fixtures, a synthetic course folder, and
`.opml`/`.xmind` files produced by actual Groq and Gemini runs — useful as a
reference for what output quality to expect at each level.

## Troubleshooting

**`ffmpeg not found on PATH`** — install ffmpeg and make sure it's on your
`PATH`. Only needed for local video files; YouTube and subtitle-file sources
don't need it.

**`GROQ_API_KEY not set` / `GEMINI_API_KEY not set`** — you asked for a
specific engine (`--engine groq`) but didn't set that key. Either add it to
`.env`, or use `--engine auto` to let cerebro pick whatever's available (or
fall back to offline).

**No API key found — using the offline heuristic engine, even though I set a
key** — cerebro checks the current directory's `.env` first, then
`~/.cerebro/.env`. This almost always means the key ended up in whichever one
you *aren't* currently running cerebro from — see
[Quick start](#quick-start) step 2. If it's genuinely a heads-up and not a
mistake, it's not an error: the map will still be structurally correct, just
without AI-driven grouping.

**A relationship count of 0 at `expert` level** — relationships only appear
when there are enough distinct branches for the model to meaningfully connect;
very short sources may not produce any.

**`No subtitle track found and faster-whisper is not installed`** — the video
has no embedded subtitles and Whisper isn't installed. Run
`pip install cerebro[whisper]`.

**Import into XMind looks flat / missing arrows** — you exported as `--format
opml`. OPML can't represent relationship arrows or icons; re-export with
`--format xmind`.

## How it works

The core design decision: **the model never writes a file format.** Every
source is normalized into a `Transcript`; a structurer (heuristic or LLM)
turns that into a format-agnostic intermediate representation (`MindMap`);
deterministic converters turn the IR into OPML or XMind. This is what makes
output *reliably* importable — a converter never makes a syntax mistake the
way an LLM asked to hand-write JSON/XML sometimes would.

```
source ──▶ ingest ──▶ Transcript ──▶ structure ──▶ MindMap (IR) ──▶ convert ──▶ .opml / .xmind
        (YouTube /                 (heuristic or                  (deterministic,
         subtitle /                 map→reduce→link                always valid)
         local video)               via Groq/Gemini)
```

Before the MAP stage, a long transcript is split into chunks the model can
digest individually. Splitting purely by word count can slice a chunk
mid-topic — half of one idea, half of another — so cerebro instead detects
genuine topic shifts (a real vocabulary change between adjacent stretches of
talk, not embeddings — that would mean a heavy ML dependency this project
deliberately keeps out of the base install) and prefers to cut there, with the
word-count budget staying only as a hard ceiling.

| Module | Role |
|---|---|
| `cerebro.ingest` | Any source → `Transcript` (YouTube, playlists, subtitles, local video) |
| `cerebro.structure` | `Transcript` → `MindMap` IR (heuristic, or LLM map→reduce→link) |
| `cerebro.structure.segment` | Topic-boundary-aware chunking before the MAP stage (lexical cohesion, no embeddings needed) |
| `cerebro.ir` | The `MindMap` intermediate representation itself |
| `cerebro.convert` | IR → OPML / native XMind |
| `cerebro.batch` | Fan-out + merge for playlists and course folders |
| `cerebro.foldermap` | Directory structure → `MindMap` IR, for `cerebro tree` — a separate concern from the video pipeline entirely |
| `cerebro.llm` | Provider abstraction (Groq / Gemini / mock) |
| `cerebro.cache` | Content-addressed caching |
| `cerebro.paths` | Stable, home-relative locations (`~/.cerebro/.env`, `~/cerebro-maps/`) so a globally-installed CLI works the same from any directory |
| `cerebro.ui` | Rich banner, progress, and the in-terminal map preview |
| `cerebro.wizard` | The guided interactive flow (arrow-key menus with a plain-prompt fallback) |
| `cerebro.cli` | Typer commands (`map`, `batch`, `interactive`) |

## Development

```bash
git clone https://github.com/ws0x/cerebro.git
cd cerebro
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # .venv/bin/pip on macOS/Linux
.venv\Scripts\pytest                    # run the test suite
```

CI runs the full suite on every push across Python 3.11 and 3.13 — see the
badge at the top of this file.

### Building the standalone binary

```bash
.venv\Scripts\pip install pyinstaller
.venv\Scripts\python -m PyInstaller cerebro.spec
# -> dist\cerebro.exe
```

`cerebro.spec` deliberately excludes the Whisper dependency chain
(`faster_whisper`/`ctranslate2`/`av`/`tokenizers`) to keep the binary small
(~23MB instead of ~99MB); users who need offline Whisper transcription should
install via `pip install cerebro[whisper]` instead.

## License

MIT
