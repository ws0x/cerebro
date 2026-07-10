"""Local PDF ingest: TOC-based outline extraction, with a font-size heading
heuristic fallback, and a flat per-page ``Transcript`` for the no-structure case.

PDFs are unlike video transcripts: they usually already have a real hierarchy
(bookmarks/TOC, or at least visually distinct headings) instead of needing one
invented from flat text. Structure is extracted here, deterministically, in
priority order: TOC/bookmarks -> font-size heading heuristic -> none (flat
fallback, same posture as a transcript with no natural structure at all — the
existing map->reduce->link pipeline handles it unchanged).

Scanned/image-only PDFs (no text layer) are not supported — would need OCR,
the same explicit non-goal as image-based (PGS/VobSub) video subtitle codecs.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from ..cache import Cache
from ..transcript import OutlineEntry, Segment, Transcript


class PdfIngestError(RuntimeError):
    pass


# -- font-size heading heuristic (only used when a PDF has no TOC/bookmarks) --
#
# Conservative by design: a wrong "no structure" verdict just falls through to
# the existing flat map->reduce->link pipeline (already good); a wrong "yes
# structure" verdict would corrupt an otherwise-good map with a guessed,
# possibly-wrong skeleton, which is the worse failure mode.
_WORD_RE = re.compile(r"\S+")
_HEADING_SIZE_RATIO = 1.15  # must be notably larger than the doc's body-text size
_MAX_HEADING_WORDS = 12
_MIN_HEADING_COUNT = 3
_MAX_HEADING_LEVELS = 3
# Identical text repeated across this many pages is a running header/footer
# (page decoration), not a real section break — real headings are essentially
# always unique within a document.
_HEADER_FOOTER_REPEAT_THRESHOLD = 3


def _iter_lines(doc):
    """Yield (size, text, page_index) for every text line in the document,
    in document order. ``size`` is the largest span size on that line,
    rounded to the nearest 0.5pt (fonts rarely differ by less than that)."""
    for page_index in range(doc.page_count):
        for block in doc[page_index].get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = round(max(s.get("size", 0.0) for s in spans) * 2) / 2
                yield size, text, page_index


def _detect_headings(doc) -> list:
    """Best-effort heading detection from font size, for PDFs with no TOC.
    Returns the same ``[[level, title, page(1-indexed)], ...]`` shape as
    ``doc.get_toc()``, or ``[]`` if the signal isn't trustworthy."""
    lines = list(_iter_lines(doc))
    if not lines:
        return []

    body_size = Counter(size for size, _text, _page in lines).most_common(1)[0][0]

    candidates = [
        (size, text, page)
        for size, text, page in lines
        if size >= body_size * _HEADING_SIZE_RATIO
        and 1 <= len(_WORD_RE.findall(text)) <= _MAX_HEADING_WORDS
    ]

    repeat_counts = Counter(text for _size, text, _page in candidates)
    candidates = [
        (size, text, page)
        for size, text, page in candidates
        if repeat_counts[text] < _HEADER_FOOTER_REPEAT_THRESHOLD
    ]

    if len(candidates) < _MIN_HEADING_COUNT:
        return []
    if len({page for _size, _text, page in candidates}) < 2:
        return []  # all on one page -- not real document structure

    distinct_sizes = sorted({size for size, _text, _page in candidates}, reverse=True)
    level_by_size = {
        size: min(rank + 1, _MAX_HEADING_LEVELS) for rank, size in enumerate(distinct_sizes)
    }
    return [[level_by_size[size], text, page + 1] for size, text, page in candidates]


def _extract(path: Path) -> dict:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise PdfIngestError(f"Could not open PDF: {path.name} ({exc})") from exc

    try:
        if doc.is_encrypted:
            raise PdfIngestError(
                f"{path.name} is password-protected; encrypted PDFs are not supported."
            )

        pages = [doc[i].get_text() for i in range(doc.page_count)]
        if not any(p.strip() for p in pages):
            raise PdfIngestError(
                f"{path.name} has no extractable text (scanned/image-only PDF); "
                "OCR is not supported."
            )

        title = (doc.metadata or {}).get("title", "").strip()
        toc = doc.get_toc(simple=True)  # [[level, title, page(1-indexed)], ...]
        if not toc:
            toc = _detect_headings(doc)
    finally:
        doc.close()

    return {"title": title, "pages": pages, "toc": toc}


def _cache_key(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return Cache.key("pdf_extract", path.name, stat.st_size, stat.st_mtime)


def load_pdf(path: str | Path, cache: Cache | None = None) -> Transcript:
    path = Path(path)
    if not path.exists():
        raise PdfIngestError(f"File not found: {path}")

    cache = cache or Cache(enabled=False)
    key = _cache_key(path)
    data = cache.get(key) if key else None
    if data is None:
        data = _extract(path)
        if key:
            cache.set(key, data)

    title = data["title"] or path.stem.replace("_", " ").replace("-", " ").strip().title()
    # Segment.start/duration mean "seconds into the source" everywhere else in
    # this codebase (video/YouTube ingest) and downstream code (HeuristicStructurer,
    # LLMStructurer's MAP stage) propagates it straight into Node.timestamp, which
    # every converter renders as [mm:ss]. A PDF page number is not a second count --
    # leaving these at their 0.0 default (rather than start=page index) avoids a
    # page showing up mislabeled as e.g. "[0:04]" in the flat (no-outline) fallback
    # path used both by `map` on a structureless PDF and by `batch` course folders.
    segments = [Segment(text=text) for text in data["pages"]]
    outline = [
        OutlineEntry(level=int(lvl), title=str(t).strip(), page=max(0, int(pg) - 1))
        for lvl, t, pg in data["toc"]
        if str(t).strip()
    ]
    return Transcript(source=str(path), title=title, segments=segments, outline=outline)
