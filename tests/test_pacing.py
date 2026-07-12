"""Tests for cross-run rate-limiter pacing persistence (~/.cerebro/pacing.json).

PACING_PATH is patched directly on the ``cerebro.llm.pacing`` module (not on
``cerebro.paths``) since it's a module-level constant computed once at import
time -- patching the origin after import wouldn't be seen here, same footgun
documented in paths.py for CONFIG_DIR-derived defaults.
"""

from __future__ import annotations

import json

import pytest

from cerebro.llm.base import RateLimiter
from cerebro.llm.pacing import load_pacing, record_pacing


@pytest.fixture(autouse=True)
def _pacing_path(tmp_path, monkeypatch):
    path = tmp_path / "pacing.json"
    monkeypatch.setattr("cerebro.llm.pacing.PACING_PATH", path)
    return path


def test_load_pacing_returns_empty_dict_when_file_is_missing():
    assert load_pacing() == {}


def test_load_pacing_returns_empty_dict_on_corrupt_json(_pacing_path):
    _pacing_path.write_text("not json", encoding="utf-8")
    assert load_pacing() == {}


def test_load_pacing_ignores_non_numeric_entries(_pacing_path):
    _pacing_path.write_text(json.dumps({"groq": 3.0, "junk": "nope"}), encoding="utf-8")
    assert load_pacing() == {"groq": 3.0}


def test_record_pacing_persists_the_raised_interval_when_the_limiter_backed_off():
    limiter = RateLimiter(2.0)
    limiter.backoff()  # -> 3.2, hit_limit True
    record_pacing("groq", limiter, default_interval=2.1)
    assert load_pacing()["groq"] == pytest.approx(3.2)


def test_record_pacing_decays_toward_default_on_a_clean_run(_pacing_path):
    _pacing_path.write_text(json.dumps({"groq": 10.0}), encoding="utf-8")
    limiter = RateLimiter(10.0)  # never backed off -- a clean run
    record_pacing("groq", limiter, default_interval=2.1)
    assert load_pacing()["groq"] == pytest.approx(10.0 * 0.85)


def test_record_pacing_decay_is_floored_at_the_default(_pacing_path):
    _pacing_path.write_text(json.dumps({"groq": 2.2}), encoding="utf-8")
    limiter = RateLimiter(2.2)
    record_pacing("groq", limiter, default_interval=2.1)
    # 2.2 * 0.85 < 2.1, so the floor wins.
    assert load_pacing()["groq"] == pytest.approx(2.1)


def test_record_pacing_with_no_prior_value_and_a_clean_run_uses_the_default():
    limiter = RateLimiter(2.1)
    record_pacing("groq", limiter, default_interval=2.1)
    assert load_pacing()["groq"] == pytest.approx(2.1)


def test_record_pacing_only_touches_its_own_provider_key(_pacing_path):
    _pacing_path.write_text(json.dumps({"gemini": 5.0}), encoding="utf-8")
    limiter = RateLimiter(2.1)
    record_pacing("groq", limiter, default_interval=2.1)
    pacing = load_pacing()
    assert pacing["gemini"] == 5.0
    assert "groq" in pacing


def test_new_providers_use_persisted_pacing_as_their_starting_interval(_pacing_path, monkeypatch):
    # Read at construction time (not cached at module-import time), so a
    # freshly written pacing.json is picked up by the very next provider
    # built, with no reload/process-restart needed to see it.
    monkeypatch.delenv("CEREBRO_GROQ_MIN_INTERVAL", raising=False)
    monkeypatch.delenv("CEREBRO_GEMINI_MIN_INTERVAL", raising=False)
    _pacing_path.write_text(json.dumps({"groq": 7.5, "gemini": 9.0}), encoding="utf-8")

    from cerebro.llm.providers import GeminiProvider, GroqProvider

    groq = GroqProvider("fake-key")
    gemini = GeminiProvider("fake-key")
    assert groq._limiter.current_interval == pytest.approx(7.5)
    assert gemini._limiter.current_interval == pytest.approx(9.0)


def test_new_providers_fall_back_to_the_hardcoded_default_with_no_pacing_file(monkeypatch):
    monkeypatch.delenv("CEREBRO_GROQ_MIN_INTERVAL", raising=False)
    monkeypatch.delenv("CEREBRO_GEMINI_MIN_INTERVAL", raising=False)

    from cerebro.llm.providers import GeminiProvider, GroqProvider

    assert GroqProvider("fake-key")._limiter.current_interval == pytest.approx(2.1)
    assert GeminiProvider("fake-key")._limiter.current_interval == pytest.approx(4.5)


def test_an_explicit_env_var_overrides_persisted_pacing(_pacing_path, monkeypatch):
    _pacing_path.write_text(json.dumps({"groq": 7.5}), encoding="utf-8")
    monkeypatch.setenv("CEREBRO_GROQ_MIN_INTERVAL", "1.0")

    from cerebro.llm.providers import GroqProvider

    assert GroqProvider("fake-key")._limiter.current_interval == pytest.approx(1.0)


def test_an_explicit_min_interval_constructor_arg_overrides_persisted_pacing(_pacing_path):
    _pacing_path.write_text(json.dumps({"groq": 7.5}), encoding="utf-8")

    from cerebro.llm.providers import GroqProvider

    assert GroqProvider("fake-key", min_interval=0.5)._limiter.current_interval == pytest.approx(0.5)
