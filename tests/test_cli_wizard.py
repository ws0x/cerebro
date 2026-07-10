from cerebro.paths import load_config, save_config
from cerebro.wizard import _clean, _remember_last_answers, detect_source_kind


def test_detects_playlist_url():
    assert detect_source_kind("https://youtube.com/playlist?list=PLxyz") == "playlist"


def test_detects_youtube_video_url():
    assert detect_source_kind("https://youtu.be/dQw4w9WgXcQ") == "youtube"


def test_detects_existing_folder(tmp_path):
    assert detect_source_kind(str(tmp_path)) == "folder"


def test_detects_existing_file(tmp_path):
    f = tmp_path / "sub.vtt"
    f.write_text("WEBVTT\n", encoding="utf-8")
    assert detect_source_kind(str(f)) == "file"


def test_unknown_for_nonexistent_path():
    assert detect_source_kind("definitely_not_a_real_path_xyz_123") == "unknown"


def test_clean_strips_bom_and_zero_width_chars():
    assert _clean("﻿https://youtu.be/abc") == "https://youtu.be/abc"
    assert _clean("  path/to​/file.vtt  ") == "path/to/file.vtt"


def test_clean_is_a_noop_on_normal_text():
    assert _clean("https://youtu.be/dQw4w9WgXcQ?si=xyz") == "https://youtu.be/dQw4w9WgXcQ?si=xyz"


def test_remember_last_answers_persists_the_confirmed_choices(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    _remember_last_answers("expert", "groq", "xmind")
    cfg = load_config()
    assert cfg["level"] == "expert"
    assert cfg["engine"] == "groq"
    assert cfg["format"] == "xmind"


def test_remember_last_answers_preserves_unrelated_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    save_config({"whisper_model": "small"})
    _remember_last_answers("full", "auto", "opml")
    cfg = load_config()
    assert cfg["whisper_model"] == "small"
    assert cfg["level"] == "full"


def test_remember_last_answers_overwrites_a_previous_choice(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    _remember_last_answers("brief", "heuristic", "opml")
    _remember_last_answers("expert", "groq", "xmind")
    cfg = load_config()
    assert cfg["level"] == "expert"
    assert cfg["engine"] == "groq"
