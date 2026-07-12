"""End-to-end CLI-surface tests via Typer's CliRunner.

Every other CLI-adjacent test file calls cli.py's/wizard.py's internal
functions directly with the Typer decorators bypassed -- meaning argument
parsing, flag names, --help text, and exit codes for the actual command
surface a user types had never once been exercised end-to-end. This file
closes that gap.

Commands that write to cerebro's real global state (~/.cerebro/config.json,
cache dir, map manifest) have their underlying path constants monkeypatched
to a tmp directory first -- same "inject the path" philosophy the rest of
the test suite already uses (Cache(root=...), snapshot_dir=..., etc.),
applied here at the module-constant level since these particular constants
are resolved once at import time rather than passed as parameters.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from cerebro.cli import app

runner = CliRunner()

_HELP_INVOCATIONS = [
    ["--help"],
    ["map", "--help"],
    ["batch", "--help"],
    ["tree", "--help"],
    ["setup", "--help"],
    ["doctor", "--help"],
    ["status", "--help"],
    ["search", "--help"],
    ["merge", "--help"],
    ["edit", "--help"],
    ["dashboard", "--help"],
    ["interactive", "--help"],
    ["config", "--help"],
    ["config", "list", "--help"],
    ["config", "get", "--help"],
    ["config", "set", "--help"],
    ["config", "unset", "--help"],
    ["cache", "--help"],
    ["cache", "stats", "--help"],
    ["cache", "clear", "--help"],
    ["forget", "--help"],
    ["forget", "tree", "--help"],
    ["forget", "batch", "--help"],
]


@pytest.mark.parametrize("args", _HELP_INVOCATIONS, ids=lambda a: " ".join(a))
def test_help_exits_cleanly(args):
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_unknown_command_fails_with_nonzero_exit():
    result = runner.invoke(app, ["not-a-real-command"])
    assert result.exit_code != 0


def test_map_missing_source_argument_fails_with_nonzero_exit():
    result = runner.invoke(app, ["map"])
    assert result.exit_code != 0


def test_merge_with_fewer_than_two_files_reports_a_clean_error():
    result = runner.invoke(app, ["merge", "one.opml"])
    assert result.exit_code == 1
    assert "at least 2 files" in result.stdout


def test_status_runs_read_only_and_exits_cleanly():
    # status only reads existing cache/snapshot state -- safe to run against
    # whatever's really on the test machine, it never writes anything.
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


def test_search_with_no_matches_in_an_empty_dir(tmp_path):
    result = runner.invoke(app, ["search", "nonexistent topic", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No matches" in result.stdout


def test_config_set_get_list_unset_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)

    result = runner.invoke(app, ["config", "set", "level", "expert"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "get", "level"])
    assert result.exit_code == 0
    assert "expert" in result.stdout

    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "expert" in result.stdout

    result = runner.invoke(app, ["config", "unset", "level"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "get", "level"])
    assert result.exit_code == 0
    assert "None" in result.stdout or "expert" not in result.stdout


def test_config_set_output_dir_is_a_recognized_key(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.paths.CONFIG_DIR", tmp_path)
    custom = str(tmp_path / "my-maps")

    result = runner.invoke(app, ["config", "set", "output_dir", custom])
    assert result.exit_code == 0

    result = runner.invoke(app, ["config", "get", "output_dir"])
    assert result.exit_code == 0
    # Rich may hard-wrap a long path at the console width, so compare with
    # newlines collapsed rather than requiring an unbroken substring match.
    assert custom in result.stdout.replace("\n", "")


def test_cache_stats_reports_location_and_count(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.cache.CACHE_DIR", tmp_path)
    result = runner.invoke(app, ["cache", "stats"])
    assert result.exit_code == 0
    # Rich wraps a long path across lines inside the panel, so check for the
    # labels rather than an unwrapped substring match on the full tmp_path.
    assert "Location" in result.stdout
    assert "Entries" in result.stdout and "0" in result.stdout


def test_map_end_to_end_with_mock_engine_and_local_fixture(tmp_path, monkeypatch):
    """The flagship test: drives `cerebro map` exactly as a user would --
    real argument parsing, real ingest, real (mocked-provider) structuring,
    real file write -- not a single internal function called directly."""
    monkeypatch.setattr("cerebro.manifest.MAP_MANIFEST_PATH", tmp_path / "map-manifest.json")

    source = tmp_path / "lesson.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nPhotosynthesis converts light into chemical energy.\n",
        encoding="utf-8",
    )
    out = tmp_path / "lesson.opml"

    result = runner.invoke(
        app,
        [
            "map",
            str(source),
            "--engine",
            "mock",
            "--no-cache",
            "--format",
            "opml",
            "--out",
            str(out),
            "--no-preview",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out.exists()
    assert "<opml" in out.read_text(encoding="utf-8")


def test_map_done_panel_shows_which_engine_actually_built_the_map(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.manifest.MAP_MANIFEST_PATH", tmp_path / "map-manifest.json")
    source = tmp_path / "lesson.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nPhotosynthesis converts light into chemical energy.\n",
        encoding="utf-8",
    )
    out = tmp_path / "lesson.opml"

    result = runner.invoke(
        app,
        ["map", str(source), "--engine", "mock", "--no-cache", "--out", str(out), "--no-preview"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Engine" in result.stdout
    assert "mock:mock-1" in result.stdout


def test_map_done_panel_shows_heuristic_engine_too(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.manifest.MAP_MANIFEST_PATH", tmp_path / "map-manifest.json")
    source = tmp_path / "lesson.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nPhotosynthesis converts light into chemical energy.\n",
        encoding="utf-8",
    )
    out = tmp_path / "lesson.opml"

    result = runner.invoke(
        app,
        ["map", str(source), "--engine", "heuristic", "--out", str(out), "--no-preview"],
    )

    assert result.exit_code == 0, result.stdout
    assert "Engine" in result.stdout
    assert "heuristic" in result.stdout


def test_map_warns_before_spending_calls_on_a_long_source(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.manifest.MAP_MANIFEST_PATH", tmp_path / "map-manifest.json")
    # ~30,000 words -- past the many-calls threshold at expert level, same
    # scale as the real 131-minute video that triggered this feature.
    lines = [
        f"{i}\n00:{i:02d}:00,000 --> 00:{i:02d}:05,000\n" + " ".join(f"word{j}" for j in range(50)) + "\n"
        for i in range(600)
    ]
    source = tmp_path / "long_lesson.srt"
    source.write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "long_lesson.opml"

    result = runner.invoke(
        app,
        ["map", str(source), "--engine", "mock", "--level", "expert", "--no-cache", "--out", str(out), "--no-preview"],
    )

    assert result.exit_code == 0, result.stdout
    assert "long source" in result.stdout
    assert "LLM calls" in result.stdout


def test_map_does_not_warn_for_a_short_source(tmp_path, monkeypatch):
    monkeypatch.setattr("cerebro.manifest.MAP_MANIFEST_PATH", tmp_path / "map-manifest.json")
    source = tmp_path / "lesson.srt"
    source.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\nPhotosynthesis converts light into chemical energy.\n",
        encoding="utf-8",
    )
    out = tmp_path / "lesson.opml"

    result = runner.invoke(
        app,
        ["map", str(source), "--engine", "mock", "--no-cache", "--out", str(out), "--no-preview"],
    )

    assert result.exit_code == 0, result.stdout
    assert "long source" not in result.stdout
