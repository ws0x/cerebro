from cerebro.paths import load_config, save_config


def test_load_config_missing_file_returns_empty_dict(tmp_path):
    assert load_config(config_dir=tmp_path) == {}


def test_save_then_load_round_trips(tmp_path):
    save_config({"level": "expert", "relationship_limit": "12"}, config_dir=tmp_path)
    assert load_config(config_dir=tmp_path) == {"level": "expert", "relationship_limit": "12"}


def test_save_overwrites_previous_contents(tmp_path):
    save_config({"level": "brief"}, config_dir=tmp_path)
    save_config({"level": "expert"}, config_dir=tmp_path)
    assert load_config(config_dir=tmp_path) == {"level": "expert"}


def test_load_corrupt_json_returns_empty_dict_not_a_crash(tmp_path):
    config_dir = tmp_path
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text("{not valid json", encoding="utf-8")
    assert load_config(config_dir=config_dir) == {}
