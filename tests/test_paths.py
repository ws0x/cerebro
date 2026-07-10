from pathlib import Path

from cerebro.paths import CONFIG_DIR, DEFAULT_OUTPUT_DIR, GLOBAL_ENV_PATH, ensure_output_dir, load_config


def test_global_env_path_lives_under_config_dir():
    assert GLOBAL_ENV_PATH.parent == CONFIG_DIR
    assert GLOBAL_ENV_PATH.name == ".env"


def test_ensure_output_dir_creates_and_returns_it(tmp_path, monkeypatch):
    target = tmp_path / "cerebro-maps"
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", target)
    assert not target.exists()
    result = ensure_output_dir()
    assert result == target
    assert target.is_dir()


def test_default_output_dir_is_absolute():
    assert DEFAULT_OUTPUT_DIR.is_absolute()


def test_default_output_dir_env_var_overrides_the_hardcoded_default(monkeypatch, tmp_path):
    # Re-import to exercise the module-load-time os.environ.get(...) read --
    # DEFAULT_OUTPUT_DIR itself is a plain constant, not re-evaluated per
    # call, so this only proves the override works at import time, same as
    # any other install would experience it.
    import importlib

    import cerebro.paths as paths_module

    monkeypatch.setenv("CEREBRO_OUTPUT_DIR", str(tmp_path))
    try:
        importlib.reload(paths_module)
        assert paths_module.DEFAULT_OUTPUT_DIR == tmp_path
    finally:
        importlib.reload(paths_module)  # restore the real default for every other test


def test_ensure_output_dir_falls_back_when_preferred_dir_is_unreachable(tmp_path, monkeypatch):
    unreachable = tmp_path / "unreachable"
    fallback = tmp_path / "fallback"
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", unreachable)
    monkeypatch.setattr("cerebro.paths._FALLBACK_OUTPUT_DIR", fallback)

    original_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == unreachable:
            raise OSError("drive not mounted")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    result = ensure_output_dir()
    assert result == fallback
    assert fallback.is_dir()


def test_ensure_output_dir_prefers_the_configured_dir_when_reachable(tmp_path, monkeypatch):
    preferred = tmp_path / "preferred"
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", preferred)
    result = ensure_output_dir()
    assert result == preferred
    assert preferred.is_dir()


def test_ensure_output_dir_fallback_does_not_crash_under_json_mode(tmp_path, monkeypatch):
    from cerebro.console import set_json

    unreachable = tmp_path / "unreachable"
    fallback = tmp_path / "fallback"
    monkeypatch.setattr("cerebro.paths.DEFAULT_OUTPUT_DIR", unreachable)
    monkeypatch.setattr("cerebro.paths._FALLBACK_OUTPUT_DIR", fallback)

    original_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == unreachable:
            raise OSError("drive not mounted")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    set_json(True)
    try:
        result = ensure_output_dir()
    finally:
        set_json(False)
    assert result == fallback


def test_load_config_nonexistent(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    assert load_config() == {}


def test_load_config_valid(tmp_path, monkeypatch):
    import json
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"engine": "gemini", "level": "expert", "relationship_limit": 12}), encoding="utf-8")
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    
    cfg = load_config()
    assert cfg["engine"] == "gemini"
    assert cfg["level"] == "expert"
    assert cfg["relationship_limit"] == 12
