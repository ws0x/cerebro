# Cerebro Desktop — Architecture & Roadmap

Planning doc for the Electron desktop app. Decided 2026-07-10; not yet built.
CLI remains the primary interface and is unaffected — this is an additional
front-end, not a replacement.

## Architecture decision

**Electron shell around the existing, unchanged Python core.** No rewrite —
the Python side (42 tests, real LLM integration, working PyInstaller build)
is the asset worth keeping. Wiring:

- Python exposes a **local FastAPI HTTP + WebSocket server**
  (`cerebro/server.py`, new — thin wrapper, no duplicated business logic).
  REST for actions (submit a job, save output); WebSocket for live progress,
  forwarding the exact same `on_event` dicts `LLMStructurer`/`run_batch`
  already emit (`map_start`, `map_progress`, `reduce_start`, `link_start`,
  `done`, errors) instead of routing them into Rich's console.
- Electron's **main process spawns the server** as a child process on app
  launch (a bundled `cerebro-server.exe` built the same way as the existing
  CLI binary — see `cerebro.spec`), manages its lifecycle, picks a free local
  port, and generates a per-session token passed to both the server and the
  renderer via environment variable.
- **React renderer** talks to the server over `http://127.0.0.1:<port>` only
  — bound to localhost, token-gated, no other local process can reach it.
- New optional dependency group: `pip install cerebro[server]` (fastapi +
  uvicorn), same pattern as the existing `[whisper]` extra — keeps the base
  CLI install light for anyone who never touches the desktop app.

**Why not stdio JSON-RPC instead:** simpler to bundle (no port management),
but progress streaming and error handling are meaningfully more awkward, and
this app's core value proposition (live progress, and eventually a real
visual map canvas) benefits from HTTP/WebSocket's request/response +
streaming split. Revisit only if the local-server approach hits a real wall.

## v1 scope (MVP)

**In:** single source only (YouTube URL or local file) → pick level/engine/
format → Start → live progress (WebSocket-driven, real bar not polling) →
result screen (node count, depth, relationship count) → save via a native
Electron save dialog, using the existing `write_opml`/`write_xmind` from
`cerebro.convert` unchanged.

**Explicitly out of v1** (stay CLI-only until proven worth porting):
batch/playlist support, cache stats/clear GUI, `cerebro doctor` GUI. A
visual node-graph canvas (replacing the ASCII tree preview) is the most
interesting v2 direction — Electron makes it possible for the first time —
but is deliberately deferred so v1 stays small enough to actually ship.

## Roadmap (for whoever picks this up next)

1. **Build `cerebro/server.py` alone first.** Prove the API contract with
   `curl`/`websocat` before Electron enters the picture at all — a FastAPI
   server with a bug is much faster to debug standalone than through a GUI.
2. **Scaffold Electron + React** (recommend `electron-vite` — fast HMR, good
   TypeScript support). Get a bare health-check round trip working: main
   process spawns the server, renderer calls `/health`, done.
3. **Wire one real end-to-end case** — a local subtitle file through the
   heuristic engine (no API key, no network, fastest possible thing to
   debug) — source input → options → progress → result, fully working.
4. **Add real LLM engines and YouTube ingest** to the same flow.
5. **Polish**: cyan/pink theming carried over from the CLI banner for visual
   continuity across the "cerebro family," error states, and packaging
   (`electron-builder` with the PyInstaller server as an `extraResources`
   bundle).

## Open questions for whoever starts building

- Exact WebSocket message schema (propose: reuse the `on_event(kind, **d)`
  shape directly as `{"kind": ..., ...d}` JSON frames — avoids inventing a
  second protocol).
- Whether the save dialog offers both OPML and XMind in v1 or just the
  format chosen up front (lean: just the chosen format, matches CLI parity).
- Auto-update strategy for the packaged app — not addressed here, needs its
  own decision when packaging is actually reached in step 5.
