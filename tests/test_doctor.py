from pathlib import Path

from cerebro.doctor import Check, has_failures, run_diagnostics


def _dirs(tmp_path):
    return dict(
        config_dir=tmp_path / "config",
        cache_dir=tmp_path / "config" / "cache",
        output_dir=tmp_path / "out",
        tree_snapshot_dir=tmp_path / "config" / "tree-snapshots",
        batch_snapshot_dir=tmp_path / "config" / "batch-snapshots",
        global_env_path=tmp_path / "config" / ".env",
    )


def test_reports_python_version_ok(tmp_path):
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    py = next(c for c in checks if c.label == "Python version")
    assert py.status == "ok"
    assert py.group == "Environment"


def test_all_storage_dirs_writable_are_ok(tmp_path):
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    storage = [c for c in checks if c.group == "Storage"]
    assert storage  # non-empty
    assert all(c.status != "fail" for c in storage)
    assert not has_failures(checks)


def test_unwritable_storage_dir_is_a_hard_failure(tmp_path, monkeypatch):
    # A file where a directory is expected makes mkdir(parents=True) fail —
    # a portable way to force a real, non-mocked writability failure.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    dirs = _dirs(tmp_path)
    dirs["output_dir"] = blocker / "sub"

    checks = run_diagnostics(check_network=False, **dirs)
    output_check = next(c for c in checks if c.label == "Output dir")
    assert output_check.status == "fail"
    assert output_check.fix is not None
    assert has_failures(checks)


def test_missing_ffmpeg_is_an_advisory_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.doctor.shutil.which", lambda name: None)
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    ffmpeg = next(c for c in checks if c.label == "ffmpeg")
    assert ffmpeg.status == "warn"  # advisory, never fails the whole run just for this
    assert not has_failures(checks)


def test_missing_core_dependency_is_a_hard_failure(tmp_path, monkeypatch):
    real_import = __import__("importlib").import_module

    def fake_import(name, *a, **k):
        if name == "yt_dlp":
            raise ImportError("simulated missing dependency")
        return real_import(name, *a, **k)

    monkeypatch.setattr("cerebro.doctor.importlib.import_module", fake_import)
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    ytdlp = next(c for c in checks if c.label == "yt-dlp")
    assert ytdlp.status == "fail"
    assert has_failures(checks)


def test_no_api_keys_set_reports_advisory_and_fallback_note(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no cwd .env either
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    groq = next(c for c in checks if c.label == "Groq API key")
    gemini = next(c for c in checks if c.label == "Gemini API key")
    fallback = next(c for c in checks if c.label == "AI structuring")
    assert groq.status == "warn"
    assert gemini.status == "warn"
    assert fallback.status == "warn"
    assert "heuristic" in fallback.detail
    assert not has_failures(checks)  # missing keys never fail the doctor run


def test_api_key_found_is_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-123")
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    groq = next(c for c in checks if c.label == "Groq API key")
    assert groq.status == "ok"
    assert "test-key-123" not in groq.detail  # never echo the key value itself


def test_network_checks_skipped_when_disabled(tmp_path):
    checks = run_diagnostics(check_network=False, **_dirs(tmp_path))
    assert not any("reachable" in c.label for c in checks)


def test_network_checks_present_when_enabled(tmp_path):
    checks = run_diagnostics(check_network=True, **_dirs(tmp_path))
    assert any("reachable" in c.label for c in checks)


def test_cache_and_snapshot_counts_reflect_directory_contents(tmp_path):
    dirs = _dirs(tmp_path)
    dirs["tree_snapshot_dir"].mkdir(parents=True)
    (dirs["tree_snapshot_dir"] / "abc123.json").write_text("{}", encoding="utf-8")
    (dirs["tree_snapshot_dir"] / "def456.json").write_text("{}", encoding="utf-8")

    checks = run_diagnostics(check_network=False, **dirs)
    tree_check = next(c for c in checks if c.label == "Tree snapshots")
    assert "2" in tree_check.detail
