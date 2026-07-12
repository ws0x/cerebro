"""Tests for the real provider wrappers -- specifically that each one paces
its requests through a RateLimiter before hitting the network. requests.post
is mocked, and the limiter interval is forced to 0 so the tests stay fast;
the RateLimiter's own timing behavior is covered in test_llm_base.py.
"""

from unittest.mock import MagicMock, patch

from cerebro.llm.providers import GeminiProvider, GroqProvider


def _groq_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    return resp


def _gemini_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
    return resp


def test_groq_provider_acquires_the_rate_limiter_before_each_call():
    provider = GroqProvider("fake-key", min_interval=0)
    provider._limiter = MagicMock()
    with patch("requests.post", return_value=_groq_response()):
        assert provider.complete_json("sys", "user") == {"ok": True}
    provider._limiter.acquire.assert_called_once()


def test_gemini_provider_acquires_the_rate_limiter_before_each_call():
    provider = GeminiProvider("fake-key", min_interval=0)
    provider._limiter = MagicMock()
    with patch("requests.post", return_value=_gemini_response()):
        assert provider.complete_json("sys", "user") == {"ok": True}
    provider._limiter.acquire.assert_called_once()


def test_providers_accept_an_explicit_min_interval_override():
    # Construction with a custom interval must not raise and must build a
    # limiter with that interval (0 here keeps the suite fast).
    g = GroqProvider("k", min_interval=0)
    assert g._limiter._min_interval == 0
    gem = GeminiProvider("k", min_interval=3.0)
    assert gem._limiter._min_interval == 3.0
