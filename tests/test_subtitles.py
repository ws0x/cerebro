from cerebro.ingest.subtitles import load_subtitle_file

_SRT = (
    "1\n00:00:00,000 --> 00:00:01,500\nHello from an SRT file.\n\n"
    "2\n00:00:01,500 --> 00:00:03,000\nThis is the second cue.\n"
)

_VTT = (
    "WEBVTT\n\n"
    "00:00:00.000 --> 00:00:01.500\n<c>Hello from a VTT file.</c>\n\n"
    "00:00:01.500 --> 00:00:03.000\nSecond cue here.\n"
)


def test_load_srt_file(tmp_path):
    path = tmp_path / "lesson.srt"
    path.write_text(_SRT, encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert transcript.title == "Lesson"
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hello from an SRT file."
    assert transcript.segments[1].start == 1.5


def test_load_vtt_file_strips_inline_tags(tmp_path):
    path = tmp_path / "lesson.vtt"
    path.write_text(_VTT, encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert len(transcript.segments) == 2
    assert transcript.segments[0].text == "Hello from a VTT file."  # <c> tags stripped


def test_load_plain_txt_file_one_segment_per_line(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("First line.\nSecond line.\n\nThird line.\n", encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert [s.text for s in transcript.segments] == ["First line.", "Second line.", "Third line."]


def test_load_srt_strips_bracketed_noise_tags(tmp_path):
    srt = (
        "1\n00:00:00,000 --> 00:00:01,500\n[Music]\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\nReal spoken line here.\n\n"
        "3\n00:00:03,000 --> 00:00:04,500\nbefore the [Music] chorus starts\n"
    )
    path = tmp_path / "lesson.srt"
    path.write_text(srt, encoding="utf-8")
    transcript = load_subtitle_file(path)
    # the pure-"[Music]" cue is dropped entirely, not kept as an empty/noisy segment
    assert [s.text for s in transcript.segments] == [
        "Real spoken line here.",
        "before the chorus starts",
    ]


def test_load_plain_txt_strips_bracketed_noise_tags(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("[Music]\nReal line.\n[Applause]\n", encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert [s.text for s in transcript.segments] == ["Real line."]


def test_load_subtitle_file_title_from_filename(tmp_path):
    path = tmp_path / "my_cool_lesson-notes.srt"
    path.write_text(_SRT, encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert transcript.title == "My Cool Lesson Notes"


def test_load_utf8_subtitle_file_has_no_warnings(tmp_path):
    path = tmp_path / "lesson.srt"
    path.write_text(_SRT, encoding="utf-8")
    transcript = load_subtitle_file(path)
    assert transcript.warnings == []


def test_load_cp1252_subtitle_file_decodes_correctly_and_warns(tmp_path):
    # A Windows "ANSI"-saved .srt containing a raw 0x92 byte -- cp1252's
    # curly right single-quote, not valid UTF-8. Previously this was
    # silently mangled to U+FFFD via errors="replace".
    body = b"1\n00:00:00,000 --> 00:00:01,500\nIt\x92s a cp1252 file.\n"
    path = tmp_path / "lesson.srt"
    path.write_bytes(body)

    transcript = load_subtitle_file(path)

    assert transcript.segments[0].text == "It’s a cp1252 file."  # real right single-quote, not U+FFFD
    assert len(transcript.warnings) == 1
    assert "cp1252" in transcript.warnings[0]
    assert path.name in transcript.warnings[0]


def test_load_latin1_subtitle_file_decodes_correctly_and_warns(tmp_path):
    # A raw 0x81 byte is invalid both as UTF-8 and as cp1252 (unassigned in
    # cp1252's table) but decodes fine as latin-1, the deterministic last
    # resort.
    body = b"1\n00:00:00,000 --> 00:00:01,500\nCaf\x81 latin-1 text.\n"
    path = tmp_path / "lesson.srt"
    path.write_bytes(body)

    transcript = load_subtitle_file(path)

    assert "�" not in transcript.segments[0].text  # no lossy replacement
    assert len(transcript.warnings) == 1
    assert "latin-1" in transcript.warnings[0]
