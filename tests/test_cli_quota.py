"""CliRunner tests for `cerebro quota`.

Isolates from the real machine's ~/.cerebro/.env and ~/.cerebro/quota.json
the same way test_cli_runner.py isolates config/cache paths -- otherwise
these tests would read whatever real API keys/usage happen to be on the
machine running the suite.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cerebro.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_env_and_quota(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    # config.py did `from ..paths import GLOBAL_ENV_PATH`, binding its own
    # module-level name at import time -- patching cerebro.paths.GLOBAL_ENV_PATH
    # would not reach it, so the patch has to target config.py's own name.
    monkeypatch.setattr("cerebro.llm.config.GLOBAL_ENV_PATH", tmp_path / "nonexistent.env")
    monkeypatch.setattr("cerebro.llm.quota.QUOTA_PATH", tmp_path / "quota.json")
    # load_env() also checks Path.cwd()/".env" -- the repo root has a real
    # (gitignored) .env from early in this project's history with real keys.
    # Without redirecting cwd too, "no keys configured" tests would silently
    # see those real keys instead of the empty environment they're testing.
    monkeypatch.chdir(tmp_path)


def test_quota_with_no_keys_configured_exits_nonzero_with_a_clear_message():
    result = runner.invoke(app, ["quota"])
    assert result.exit_code == 1
    assert "No API keys configured" in result.stdout


def test_quota_json_mode_with_no_keys_reports_ok_false():
    result = runner.invoke(app, ["--json", "quota"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_quota_shows_groq_live_header_data_as_colored_bars(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    (tmp_path / "quota.json").write_text(
        json.dumps(
            {
                "groq": {
                    "model": "llama-3.3-70b-versatile",
                    "limit_requests": 1000,
                    "remaining_requests": 850,
                    "limit_tokens": 12000,
                    "remaining_tokens": 11000,
                    "reset_requests_seconds": 3600,
                    "reset_tokens_seconds": 5,
                    "observed_at": "2026-07-13T09:15:00+00:00",
                    "source": "live_headers",
                    "calls_today": 12,
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["quota"])
    assert result.exit_code == 0
    assert "groq" in result.stdout
    assert "llama-3.3-70b-versatile" in result.stdout
    assert "150" in result.stdout  # 1000 - 850 used
    assert "1,000" in result.stdout or "1000" in result.stdout
    assert "15%" in result.stdout  # 150/1000 used


def test_quota_shows_gemini_local_tracking_with_no_live_data(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    (tmp_path / "quota.json").write_text(
        json.dumps({"gemini": {"model": "gemini-flash-latest", "calls_today": 7, "calls_day_key": "2026-07-13"}}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["quota"])
    assert result.exit_code == 0
    assert "No live quota API" in result.stdout
    assert "7" in result.stdout


def test_quota_gemini_shows_last_known_limit_when_present(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    (tmp_path / "quota.json").write_text(
        json.dumps(
            {
                "gemini": {
                    "model": "gemini-flash-latest",
                    "calls_today": 3,
                    "last_known_limit": {
                        "metric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "value": 50,
                        "hit_at": "2026-07-13T08:40:00+00:00",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["quota"])
    assert result.exit_code == 0
    assert "50" in result.stdout


def test_quota_with_no_recorded_data_at_all_says_so(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    result = runner.invoke(app, ["quota"])
    assert result.exit_code == 0
    assert "No data yet" in result.stdout


def test_quota_json_mode_reports_full_provider_data(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    (tmp_path / "quota.json").write_text(
        json.dumps({"groq": {"model": "llama-3.3-70b-versatile", "limit_requests": 1000, "remaining_requests": 999}}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["--json", "quota"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["providers"]["groq"]["limit_requests"] == 1000


def test_quota_refresh_makes_one_call_per_configured_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    calls = []

    def fake_complete_json(self, system, user):
        calls.append(self.name)
        return {}

    monkeypatch.setattr("cerebro.llm.providers.GroqProvider.complete_json", fake_complete_json)

    result = runner.invoke(app, ["quota", "--refresh"])
    assert result.exit_code == 0
    assert calls == ["groq"]
