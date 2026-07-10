from cerebro.llm.config import load_env, read_env_file, write_env_file


def test_read_env_file_missing_returns_empty_dict(tmp_path):
    assert read_env_file(tmp_path / "nope.env") == {}


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / ".env"
    write_env_file(path, {"GROQ_API_KEY": "abc123", "GEMINI_API_KEY": "xyz789"})
    assert read_env_file(path) == {"GROQ_API_KEY": "abc123", "GEMINI_API_KEY": "xyz789"}


def test_read_env_file_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# a comment\n\nGROQ_API_KEY=abc\n\n# another\n", encoding="utf-8")
    assert read_env_file(path) == {"GROQ_API_KEY": "abc"}


def test_read_env_file_strips_quotes(tmp_path):
    path = tmp_path / ".env"
    path.write_text('GROQ_API_KEY="abc123"\nGEMINI_API_KEY=\'xyz789\'\n', encoding="utf-8")
    assert read_env_file(path) == {"GROQ_API_KEY": "abc123", "GEMINI_API_KEY": "xyz789"}


def test_write_env_file_uses_lf_not_platform_default(tmp_path):
    # Path.write_text() in default text mode silently translates "\n" to the
    # platform line ending (CRLF on Windows) unless told not to -- verify the
    # raw bytes, not just the parsed round-trip, which wouldn't catch this.
    path = tmp_path / ".env"
    write_env_file(path, {"GROQ_API_KEY": "abc", "GEMINI_API_KEY": "xyz"})
    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.count(b"\n") == 2


def test_write_env_file_overwrites_previous_contents(tmp_path):
    path = tmp_path / ".env"
    write_env_file(path, {"GROQ_API_KEY": "old"})
    write_env_file(path, {"GEMINI_API_KEY": "new"})
    assert read_env_file(path) == {"GEMINI_API_KEY": "new"}


def test_load_env_sets_environ_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    path = tmp_path / "custom.env"
    write_env_file(path, {"SOME_TEST_KEY": "value1"})
    load_env(path)
    import os
    assert os.environ["SOME_TEST_KEY"] == "value1"


def test_load_env_does_not_override_an_already_set_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "already-set")
    path = tmp_path / "custom.env"
    write_env_file(path, {"SOME_TEST_KEY": "from-file"})
    load_env(path)
    import os
    assert os.environ["SOME_TEST_KEY"] == "already-set"
