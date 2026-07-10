from pathlib import Path
from unittest.mock import patch

from cerebro.paths import load_config, save_config
from cerebro.wizard import (
    _BACK,
    _ask_source_for_mode,
    _ask_text,
    _clean,
    _default_output_path,
    _kind_for,
    _remember_last_answers,
    _resync_output_extension,
    _select,
    _slug,
    _steps_for_mode,
)
from questionary import Choice


def test_kind_for_youtube_playlist():
    assert _kind_for("youtube", "https://youtube.com/playlist?list=PLxyz") == "playlist"


def test_kind_for_youtube_video():
    assert _kind_for("youtube", "https://youtu.be/dQw4w9WgXcQ") == "youtube"


def test_kind_for_local_video_folder(tmp_path):
    assert _kind_for("local_video", str(tmp_path)) == "folder"


def test_kind_for_local_video_file(tmp_path):
    f = tmp_path / "sub.vtt"
    f.write_text("WEBVTT\n", encoding="utf-8")
    assert _kind_for("local_video", str(f)) == "file"


def test_kind_for_pdf_is_always_file():
    assert _kind_for("pdf", "notes.pdf") == "file"


def test_kind_for_tree_is_always_tree():
    assert _kind_for("tree", "some/folder") == "tree"


def test_clean_strips_bom_and_zero_width_chars():
    assert _clean("﻿https://youtu.be/abc") == "https://youtu.be/abc"
    assert _clean("  path/to​/file.vtt  ") == "path/to/file.vtt"


def test_clean_is_a_noop_on_normal_text():
    assert _clean("https://youtu.be/dQw4w9WgXcQ?si=xyz") == "https://youtu.be/dQw4w9WgXcQ?si=xyz"


def test_remember_last_answers_persists_the_confirmed_choices(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    _remember_last_answers("expert", "groq", "xmind", "gemini")
    cfg = load_config()
    assert cfg["level"] == "expert"
    assert cfg["engine"] == "groq"
    assert cfg["format"] == "xmind"
    assert cfg["tree_engine"] == "gemini"


def test_remember_last_answers_preserves_unrelated_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    save_config({"whisper_model": "small"})
    _remember_last_answers("full", "auto", "opml", "heuristic")
    cfg = load_config()
    assert cfg["whisper_model"] == "small"
    assert cfg["level"] == "full"


def test_remember_last_answers_overwrites_a_previous_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    _remember_last_answers("brief", "heuristic", "opml", "heuristic")
    _remember_last_answers("expert", "groq", "xmind", "gemini")
    cfg = load_config()
    assert cfg["level"] == "expert"
    assert cfg["engine"] == "groq"


def test_steps_for_mode_tree_skips_level():
    assert _steps_for_mode("tree") == ["source", "engine", "format", "output"]


def test_steps_for_mode_content_modes_include_level():
    for mode in ("youtube", "local_video", "pdf"):
        assert _steps_for_mode(mode) == ["source", "level", "engine", "format", "output"]


def test_slug_strips_unsafe_characters_and_spaces():
    assert _slug("My Video: Part 1?") == "My_Video_Part_1"


def test_slug_falls_back_to_mindmap_when_empty():
    assert _slug("???") == "mindmap"


def test_default_output_path_uses_file_stem(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", tmp_path)
    pdf = tmp_path / "Neural Networks.pdf"
    pdf.write_bytes(b"")
    out = _default_output_path(str(pdf), "file", "opml")
    assert out.name == "Neural_Networks.opml"


def test_default_output_path_uses_folder_name_for_batch_and_tree(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", tmp_path)
    folder = tmp_path / "My Course"
    folder.mkdir()
    assert _default_output_path(str(folder), "folder", "xmind").name == "My_Course.xmind"
    assert _default_output_path(str(folder), "tree", "opml").name == "My_Course.opml"


def test_default_output_path_falls_back_to_mindmap_for_youtube(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", tmp_path)
    out = _default_output_path("https://youtu.be/dQw4w9WgXcQ", "youtube", "opml")
    assert out.name == "mindmap.opml"


def test_resync_output_extension_swaps_suffix_keeping_stem_and_dir():
    out = Path("/some/dir/My_Video.opml")
    assert _resync_output_extension(out, "xmind") == Path("/some/dir/My_Video.xmind")


def test_resync_output_extension_is_noop_when_already_matching():
    out = Path("/some/dir/My_Video.xmind")
    assert _resync_output_extension(out, "xmind") == out


def test_resync_output_extension_passes_through_none():
    assert _resync_output_extension(None, "xmind") is None


def test_ask_text_back_keyword_returns_back_sentinel_via_rich_fallback():
    with patch("cerebro.wizard.has_real_console", return_value=False):
        with patch("cerebro.wizard.Prompt.ask", return_value="back"):
            assert _ask_text("Anything:", allow_back=True) is _BACK


def test_ask_text_back_keyword_is_case_insensitive():
    with patch("cerebro.wizard.has_real_console", return_value=False):
        with patch("cerebro.wizard.Prompt.ask", return_value="BACK"):
            assert _ask_text("Anything:", allow_back=True) is _BACK


def test_ask_text_without_allow_back_treats_back_as_a_literal_value():
    with patch("cerebro.wizard.has_real_console", return_value=False):
        with patch("cerebro.wizard.Prompt.ask", return_value="back"):
            assert _ask_text("Anything:", allow_back=False) == "back"


def test_select_back_choice_returns_back_sentinel_via_rich_fallback():
    choices = [Choice("Brief", value="brief"), Choice("Full", value="full")]
    with patch("cerebro.wizard.has_real_console", return_value=False):
        with patch("cerebro.wizard.Prompt.ask", return_value="back"):
            assert _select("Level:", choices, allow_back=True) is _BACK


def test_ask_source_for_mode_uses_clipboard_suggestion_as_default(tmp_path):
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-fake")

    def fake_ask(_message, default=None, **_kwargs):
        return default  # simulates the user just pressing Enter

    with patch("cerebro.wizard.suggest_for_mode", return_value=str(pdf)):
        with patch("cerebro.wizard.has_real_console", return_value=False):
            with patch("cerebro.wizard.Prompt.ask", side_effect=fake_ask):
                result = _ask_source_for_mode("pdf", default="")
    assert result == str(pdf)


def test_ask_source_for_mode_ignores_clipboard_when_a_default_is_already_set(tmp_path):
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-fake")
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-fake")
    with patch("cerebro.wizard.suggest_for_mode", return_value=str(pdf)) as mock_suggest:
        with patch("cerebro.wizard.has_real_console", return_value=False):
            with patch("cerebro.wizard.Prompt.ask", return_value=str(other)):
                _ask_source_for_mode("pdf", default=str(other))
    # a caller-supplied default (e.g. re-showing a previous answer after
    # "back") must win over a clipboard guess -- checking the clipboard is
    # only for the very first ask, not every re-prompt in the retry loop
    mock_suggest.assert_not_called()
