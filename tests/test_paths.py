from cerebro.paths import CONFIG_DIR, DEFAULT_OUTPUT_DIR, GLOBAL_ENV_PATH, ensure_output_dir


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


def test_default_output_dir_is_under_home():
    assert DEFAULT_OUTPUT_DIR.is_absolute()
