from unittest.mock import patch

from cerebro.clipboard import read_clipboard_text, suggest_for_mode


def test_read_clipboard_text_strips_and_unquotes():
    with patch("pyperclip.paste", return_value='  "C:\\Users\\me\\file.pdf"  '):
        assert read_clipboard_text() == "C:\\Users\\me\\file.pdf"


def test_read_clipboard_text_rejects_multiline():
    with patch("pyperclip.paste", return_value="line one\nline two"):
        assert read_clipboard_text() is None


def test_read_clipboard_text_rejects_too_long():
    with patch("pyperclip.paste", return_value="x" * 501):
        assert read_clipboard_text() is None


def test_read_clipboard_text_returns_none_on_empty():
    with patch("pyperclip.paste", return_value=""):
        assert read_clipboard_text() is None


def test_read_clipboard_text_never_raises_on_backend_failure():
    with patch("pyperclip.paste", side_effect=Exception("no backend")):
        assert read_clipboard_text() is None


def test_suggest_for_mode_youtube_matches_url():
    with patch("pyperclip.paste", return_value="https://youtube.com/watch?v=abc123"):
        assert suggest_for_mode("youtube") == "https://youtube.com/watch?v=abc123"


def test_suggest_for_mode_youtube_rejects_non_url():
    with patch("pyperclip.paste", return_value="not a url at all"):
        assert suggest_for_mode("youtube") is None


def test_suggest_for_mode_pdf_matches_existing_pdf(tmp_path):
    pdf = tmp_path / "notes.pdf"
    pdf.write_bytes(b"%PDF-fake")
    with patch("pyperclip.paste", return_value=str(pdf)):
        assert suggest_for_mode("pdf") == str(pdf)


def test_suggest_for_mode_pdf_rejects_non_pdf_file(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with patch("pyperclip.paste", return_value=str(txt)):
        assert suggest_for_mode("pdf") is None


def test_suggest_for_mode_pdf_rejects_missing_file(tmp_path):
    with patch("pyperclip.paste", return_value=str(tmp_path / "missing.pdf")):
        assert suggest_for_mode("pdf") is None


def test_suggest_for_mode_tree_matches_existing_dir(tmp_path):
    with patch("pyperclip.paste", return_value=str(tmp_path)):
        assert suggest_for_mode("tree") == str(tmp_path)


def test_suggest_for_mode_tree_rejects_a_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    with patch("pyperclip.paste", return_value=str(f)):
        assert suggest_for_mode("tree") is None


def test_suggest_for_mode_local_video_accepts_file_or_folder(tmp_path):
    f = tmp_path / "lesson.mp4"
    f.write_bytes(b"")
    with patch("pyperclip.paste", return_value=str(f)):
        assert suggest_for_mode("local_video") == str(f)
    with patch("pyperclip.paste", return_value=str(tmp_path)):
        assert suggest_for_mode("local_video") == str(tmp_path)


def test_suggest_for_mode_never_raises_on_weird_path_chars():
    with patch("pyperclip.paste", return_value="C:\\bad<>|path\x00"):
        assert suggest_for_mode("pdf") is None
        assert suggest_for_mode("tree") is None
        assert suggest_for_mode("local_video") is None
