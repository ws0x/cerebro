"""Tests for the real provider wrappers -- specifically that each one paces
its requests through a RateLimiter before hitting the network. requests.post
is mocked, and the limiter interval is forced to 0 so the tests stay fast;
the RateLimiter's own timing behavior is covered in test_llm_base.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from cerebro.llm.providers import GeminiProvider, GroqProvider


@pytest.fixture(autouse=True)
def _isolate_quota_file(tmp_path, monkeypatch):
    # complete_json() unconditionally records a call attempt + response
    # quota -- without this, every test in this file would silently write
    # fake-key test data over the real ~/.cerebro/quota.json on whatever
    # machine runs the suite.
    monkeypatch.setattr("cerebro.llm.quota.QUOTA_PATH", tmp_path / "quota.json")


def _groq_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.headers = {}
    resp.json.return_value = {"choices": [{"message": {"content": '{"ok": true}'}}]}
    return resp


def _gemini_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.ok = True
    resp.headers = {}
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}
    return resp


def _rate_limited_response():
    resp = MagicMock()
    resp.status_code = 429
    resp.ok = False
    resp.text = "rate limit exceeded"
    resp.headers = {}
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


def test_groq_provider_backs_off_its_own_limiter_on_a_429(monkeypatch):
    # This is the actual wiring check: the provider's real RateLimiter (not a
    # mock) must be the one post_json() is told about, so a 429 observed
    # through this provider slows down *this provider's* future calls.
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = GroqProvider("fake-key", min_interval=2.0)
    responses = [_rate_limited_response(), _groq_response()]
    with patch("requests.post", side_effect=responses):
        assert provider.complete_json("sys", "user") == {"ok": True}
    assert provider._limiter.hit_limit is True
    assert provider._limiter.current_interval == pytest.approx(3.2)


def test_gemini_provider_backs_off_its_own_limiter_on_a_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    provider = GeminiProvider("fake-key", min_interval=2.0)
    responses = [_rate_limited_response(), _gemini_response()]
    with patch("requests.post", side_effect=responses):
        assert provider.complete_json("sys", "user") == {"ok": True}
    assert provider._limiter.hit_limit is True
    assert provider._limiter.current_interval == pytest.approx(3.2)
