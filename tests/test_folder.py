from cerebro.ingest.folder import discover_course_sources


def test_natural_sort_and_sidecar_matching(tmp_path):
    (tmp_path / "Lesson 2.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8"
    )
    (tmp_path / "Lesson 10.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8"
    )
    (tmp_path / "Lesson 2.mp4").write_bytes(b"")
    (tmp_path / "Lesson 10.mp4").write_bytes(b"")
    (tmp_path / "Lesson 5.mp4").write_bytes(b"")  # no subtitle -> needs_transcription

    sources = discover_course_sources(tmp_path)

    assert [s.path.stem for s in sources] == ["Lesson 2", "Lesson 5", "Lesson 10"]
    by_stem = {s.path.stem: s for s in sources}
    assert by_stem["Lesson 2"].needs_transcription is False
    assert by_stem["Lesson 5"].needs_transcription is True
    assert by_stem["Lesson 5"].path.suffix == ".mp4"  # video itself is the source


def test_standalone_subtitle_without_video_is_included(tmp_path):
    (tmp_path / "notes.txt").write_text("Just text, no video.", encoding="utf-8")
    sources = discover_course_sources(tmp_path)
    assert [s.path.name for s in sources] == ["notes.txt"]
    assert sources[0].needs_transcription is False
