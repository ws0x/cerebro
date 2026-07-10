"""Web article ingest: a blog post/documentation page URL -> a Transcript.

Completes the "any content" story alongside video/audio/PDF -- and, like a
PDF with a real TOC, an article often already has real structure (its own
``<h2>``/``<h3>`` headings) instead of needing one invented from flat text.
The same judgment call ``ingest/pdf.py`` and ``foldermap.py`` already make
applies here: when real headings are found, they become the map's real
hierarchy directly; a single flowing article with no internal heading
structure falls through to the same map -> reduce -> link pipeline used for
a video transcript, unchanged.

Content extraction (stripping nav/ads/comments/footers down to the actual
article body) is delegated to trafilatura rather than hand-rolled --
boilerplate removal is a notoriously fiddly problem this doesn't need to
reinvent.
"""

from __future__ import annotations

import re

from ..transcript import OutlineEntry, Segment, Transcript

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# A single detected heading (or none) isn't a hierarchy worth preserving --
# same "too few candidates, don't guess" caution ingest/pdf.py's font-size
# heading fallback already applies. Below this, the article falls through
# to the flat pipeline instead of a trivial one-section "outline".
_MIN_HEADINGS_FOR_STRUCTURE = 2


class ArticleIngestError(RuntimeError):
    pass


def _title_from_url(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"\.(html?|php|aspx?)$", "", slug, flags=re.IGNORECASE)
    slug = slug.replace("-", " ").replace("_", " ").strip()
    return slug.title() if slug else url


def _parse_markdown_sections(markdown: str, title: str) -> tuple[list[Segment], list[OutlineEntry]]:
    segments: list[Segment] = []
    outline: list[OutlineEntry] = []
    current_lines: list[str] = []
    seen_first_heading = False

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            segments.append(Segment(text=text))

    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if not m:
            current_lines.append(line)
            continue
        level, heading_text = len(m.group(1)), m.group(2).strip()
        # A lone leading H1 that just restates the article title isn't real
        # internal structure -- skip it rather than record a redundant
        # single-entry "outline" of just the title itself.
        if not seen_first_heading and level == 1 and heading_text.lower() == title.strip().lower():
            seen_first_heading = True
            continue
        seen_first_heading = True
        flush()
        current_lines = [heading_text]  # the heading itself is part of its own section's text, same as a PDF page
        outline.append(OutlineEntry(level=level, title=heading_text, page=len(segments)))
    flush()
    return segments, outline


def load_article(url: str) -> Transcript:
    import trafilatura

    try:
        html = trafilatura.fetch_url(url)
    except Exception as exc:
        raise ArticleIngestError(f"Could not fetch URL: {url} ({exc})") from exc
    if not html:
        raise ArticleIngestError(f"Could not fetch URL (no content returned): {url}")

    markdown = trafilatura.extract(html, output_format="markdown", include_formatting=True, favor_recall=True)
    if not markdown or not markdown.strip():
        raise ArticleIngestError(
            f"Could not extract article content from: {url} "
            "(paywalled, JS-rendered with no server-side content, or not article-shaped at all)"
        )

    metadata = trafilatura.extract_metadata(html)
    title = (metadata.title.strip() if metadata and metadata.title else "") or _title_from_url(url)

    segments, outline = _parse_markdown_sections(markdown, title)
    if not segments:
        raise ArticleIngestError(f"No extractable text found at: {url}")
    if len(outline) < _MIN_HEADINGS_FOR_STRUCTURE:
        outline = []

    return Transcript(source=url, title=title, segments=segments, outline=outline)
