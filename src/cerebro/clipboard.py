"""Best-effort clipboard read for the wizard's source-prompt suggestion.

Never allowed to slow down or crash the wizard: pyperclip has no backend on
some headless Linux setups (no xclip/xsel/wl-clipboard installed), and even
where it works, the clipboard can legitimately hold anything -- a copied
paragraph, binary garbage that decodes strangely, a stale path from an hour
ago. Every failure/mismatch mode here just means "no suggestion", never an
exception the wizard has to handle, and never a wrong guess forced on the
user (it's always just a pre-filled, fully editable default).
"""

from __future__ import annotations

from pathlib import Path

from .ingest import looks_like_web_url, looks_like_youtube
from .ingest.playlist import is_playlist_url

# A real URL or path is always a single short token -- this rules out
# "the user copied a paragraph of notes" before any further checks run.
_MAX_LEN = 500


def read_clipboard_text() -> str | None:
    """Raw clipboard text, cleaned -- or None if unavailable, empty,
    multi-line, or implausibly long to be a URL/path."""
    try:
        import pyperclip

        text = pyperclip.paste()
    except Exception:
        return None
    if not text:
        return None
    text = text.strip()
    # Windows Explorer's "Copy as path" wraps the result in quotes.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text or "\n" in text or len(text) > _MAX_LEN:
        return None
    return text


def _existing_path(text: str) -> Path | None:
    try:
        path = Path(text)
        return path if path.exists() else None
    except Exception:
        return None  # invalid path syntax for this OS, e.g. stray `<>|` chars


def suggest_for_mode(mode: str) -> str | None:
    """A clipboard candidate that plausibly matches what `mode` expects to be
    prompted for next -- or None if the clipboard holds nothing usable.

    ``mode`` is one of the wizard's source kinds: "youtube", "local_video",
    "pdf", "article", "tree". Deliberately conservative -- a false
    suggestion is worse than no suggestion, since it's the very first thing
    the user sees.
    """
    text = read_clipboard_text()
    if text is None:
        return None

    if mode == "youtube":
        return text if (looks_like_youtube(text) or is_playlist_url(text)) else None

    if mode == "article":
        # A YouTube URL is also a web URL -- but that's the youtube mode's
        # signal to claim, not article's, so explicitly exclude it here.
        return text if (looks_like_web_url(text) and not looks_like_youtube(text)) else None

    if mode == "pdf":
        path = _existing_path(text)
        return text if path is not None and path.is_file() and path.suffix.lower() == ".pdf" else None

    if mode == "tree":
        path = _existing_path(text)
        return text if path is not None and path.is_dir() else None

    if mode == "local_video":
        path = _existing_path(text)
        return text if path is not None else None  # file or folder, both valid here

    return None
