"""Integration tests against real PyMuPDF — PDFs are built on the fly with
fitz itself (same "real fixtures, not mocks" philosophy as test_video.py's
real-ffmpeg .mkv fixtures).
"""

from __future__ import annotations

import fitz
import pytest

from cerebro.cache import Cache
from cerebro.convert.util import note_for
from cerebro.ingest import load_transcript
from cerebro.ingest.pdf import PdfIngestError, load_pdf
from cerebro.structure.heuristic import HeuristicStructurer


def _build_pdf_with_toc(tmp_path):
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i} body text about chapter content.")
    doc.set_toc(
        [
            [1, "Chapter 1", 1],
            [2, "Section 1.1", 1],
            [1, "Chapter 2", 3],
        ]
    )
    path = tmp_path / "book.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_load_pdf_extracts_toc_outline(tmp_path):
    path = _build_pdf_with_toc(tmp_path)
    transcript = load_pdf(path)
    assert transcript.title  # falls back to filename-derived title (no metadata title set)
    assert len(transcript.segments) == 4
    assert [(e.level, e.title, e.page) for e in transcript.outline] == [
        (1, "Chapter 1", 0),
        (2, "Section 1.1", 0),
        (1, "Chapter 2", 2),
    ]


def test_ingest_dispatch_routes_pdf_extension(tmp_path):
    path = _build_pdf_with_toc(tmp_path)
    transcript = load_transcript(str(path))
    assert transcript.outline


def test_load_pdf_caches_extraction(tmp_path):
    path = _build_pdf_with_toc(tmp_path)
    cache = Cache(root=tmp_path / "cache")
    t1 = load_pdf(path, cache=cache)
    count_before, _ = cache.stats()
    assert count_before == 1
    t2 = load_pdf(path, cache=cache)
    count_after, _ = cache.stats()
    assert count_after == count_before  # reused, no new entry written
    assert t1.outline == t2.outline


def test_load_pdf_raises_on_no_extractable_text(tmp_path):
    doc = fitz.open()
    doc.new_page()  # blank page, no text at all
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()
    with pytest.raises(PdfIngestError, match="no extractable text"):
        load_pdf(path)


def test_load_pdf_raises_on_encrypted(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "secret content")
    path = tmp_path / "locked.pdf"
    doc.save(str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()
    with pytest.raises(PdfIngestError, match="password-protected"):
        load_pdf(path)


def test_load_pdf_raises_on_missing_file(tmp_path):
    with pytest.raises(PdfIngestError, match="not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_segments_carry_no_fake_timestamp(tmp_path):
    # Segment.start/duration mean "seconds into the source" everywhere else
    # (video/YouTube); a PDF page number must not masquerade as one, or the
    # flat (no-outline) fallback path -- which reuses the unmodified
    # HeuristicStructurer/LLMStructurer -- renders it as a bogus "[0:0N]".
    path = _build_pdf_with_toc(tmp_path)  # 4 pages
    transcript = load_pdf(path)
    assert [(s.start, s.duration) for s in transcript.segments] == [(0.0, 0.0)] * 4


def test_flat_pdf_through_heuristic_structurer_has_no_fake_page_timestamps(tmp_path):
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 500, 700),
            f"Uniform body paragraph number {i} with the exact same font size "
            "everywhere in this document, so no headings are ever detected here.",
            fontsize=11,
        )
    path = tmp_path / "flat.pdf"
    doc.save(str(path))
    doc.close()

    transcript = load_pdf(path)
    assert transcript.outline == []  # confirms this exercises the flat fallback

    mm = HeuristicStructurer().structure(transcript, level="full")
    for node in mm.root.walk():
        assert node.timestamp is None
        assert not note_for(node).startswith("[")  # no stray [mm:ss] marker


def _insert_lines(page, lines, start_y=72, line_height=20):
    y = start_y
    for text, fontsize in lines:
        page.insert_text((72, y), text, fontsize=fontsize)
        y += line_height + fontsize


def test_load_pdf_detects_headings_by_font_size_when_no_toc(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    _insert_lines(
        page,
        [
            ("Introduction", 18),
            ("This chapter introduces the subject matter in plain body text.", 10),
            ("Background", 14),
            ("More body text explaining the background of the topic at hand.", 10),
        ],
    )
    page2 = doc.new_page()
    _insert_lines(
        page2,
        [
            ("Methodology", 18),
            ("Body text describing the methodology used in this study.", 10),
            ("Results", 14),
            ("Body text summarizing the results that were observed.", 10),
        ],
    )
    path = tmp_path / "report.pdf"
    doc.save(str(path))
    doc.close()

    transcript = load_pdf(path)
    titles = [e.title for e in transcript.outline]
    assert titles == ["Introduction", "Background", "Methodology", "Results"]
    levels = {e.title: e.level for e in transcript.outline}
    assert levels["Introduction"] == levels["Methodology"] == 1  # larger size -> level 1
    assert levels["Background"] == levels["Results"] == 2  # smaller heading size -> level 2
    assert [e.page for e in transcript.outline] == [0, 0, 1, 1]


def test_load_pdf_finds_no_structure_in_uniform_text(tmp_path):
    doc = fitz.open()
    for _ in range(3):
        page = doc.new_page()
        _insert_lines(
            page,
            [
                ("This is a plain paragraph of uniform-size body text.", 11),
                ("Another plain paragraph, still the exact same font size.", 11),
            ],
        )
    path = tmp_path / "wall_of_text.pdf"
    doc.save(str(path))
    doc.close()

    transcript = load_pdf(path)
    assert transcript.outline == []


def test_load_pdf_ignores_repeated_running_header_as_noise(tmp_path):
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        _insert_lines(
            page,
            [
                ("MY BOOK TITLE", 16),  # identical on every page -> header, not a heading
                (f"Body paragraph on page {i} with regular text.", 10),
            ],
        )
    path = tmp_path / "running_header.pdf"
    doc.save(str(path))
    doc.close()

    transcript = load_pdf(path)
    assert transcript.outline == []
