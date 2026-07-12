"""Tests for resolve_provider_chain -- the provider failover chain used by
--engine auto. Keys are set/cleared via monkeypatch.setenv/delenv, never
touching the real environment or a real .env file.
"""

from __future__ import annotations

import pytest

from cerebro.llm.config import ConfigError, resolve_provider_chain
from cerebro.llm.providers import GeminiProvider, GroqProvider


@pytest.fixture(autouse=True)
def _clear_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


def test_auto_with_both_keys_returns_groq_then_gemini(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    chain = resolve_provider_chain("auto")
    assert [type(p) for p in chain] == [GroqProvider, GeminiProvider]


def test_auto_with_only_groq_key_returns_a_single_item_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    chain = resolve_provider_chain("auto")
    assert [type(p) for p in chain] == [GroqProvider]


def test_auto_with_only_gemini_key_returns_a_single_item_chain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    chain = resolve_provider_chain("auto")
    assert [type(p) for p in chain] == [GeminiProvider]


def test_auto_with_no_keys_returns_an_empty_chain():
    assert resolve_provider_chain("auto") == []


def test_explicit_groq_never_includes_gemini_even_if_both_keys_are_set(monkeypatch):
    # An explicit engine choice is respected exactly as asked -- no silent
    # failover chain for it, only 'auto' gets that behavior.
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    chain = resolve_provider_chain("groq")
    assert [type(p) for p in chain] == [GroqProvider]


def test_explicit_gemini_never_includes_groq_even_if_both_keys_are_set(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    chain = resolve_provider_chain("gemini")
    assert [type(p) for p in chain] == [GeminiProvider]


def test_explicit_groq_without_a_key_raises_the_existing_config_error():
    with pytest.raises(ConfigError, match="GROQ_API_KEY not set"):
        resolve_provider_chain("groq")


def test_heuristic_returns_an_empty_chain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g-key")  # even with a key configured
    assert resolve_provider_chain("heuristic") == []


def test_mock_returns_a_single_item_chain():
    from cerebro.llm.providers import MockProvider

    chain = resolve_provider_chain("mock")
    assert len(chain) == 1
    assert isinstance(chain[0], MockProvider)


def test_gemini_default_model_is_the_self_updating_latest_alias(monkeypatch):
    # Regression test: a dated model snapshot ("gemini-2.5-flash") can get
    # retired for NEW API keys/projects specifically (live-observed: a 404
    # "no longer available to new users", even though the model was still
    # listed by the API). "-latest" is Google's self-updating alias, chosen
    # specifically to avoid this class of failure recurring.
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    chain = resolve_provider_chain("gemini")
    assert chain[0].model == "gemini-flash-latest"

    chain_auto = resolve_provider_chain("auto")
    assert chain_auto[0].model == "gemini-flash-latest"
