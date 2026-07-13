"""Tests for llm/quota.py -- real Groq header parsing + Gemini local tracking.

QUOTA_PATH is patched on the ``cerebro.llm.quota`` module directly (not on
``cerebro.paths``), same reasoning as test_pacing.py: it's a module-level
constant read fresh inside each function via the module's own global, so
patching the origin after import wouldn't be seen by callers that already
imported the name.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from cerebro.llm.quota import (
    load_quota,
    parse_groq_duration,
    record_call_attempt,
    record_response_quota,
)


@pytest.fixture(autouse=True)
def _quota_path(tmp_path, monkeypatch):
    path = tmp_path / "quota.json"
    monkeypatch.setattr("cerebro.llm.quota.QUOTA_PATH", path)
    return path


def _response(status_code=200, headers=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.text = text
    return resp


# -- parse_groq_duration -----------------------------------------------------

def test_parse_groq_duration_minutes_and_seconds():
    assert parse_groq_duration("1m26.4s") == pytest.approx(86.4)


def test_parse_groq_duration_milliseconds():
    assert parse_groq_duration("205ms") == pytest.approx(0.205)


def test_parse_groq_duration_hours_only():
    assert parse_groq_duration("2h") == pytest.approx(7200)


def test_parse_groq_duration_plain_seconds():
    assert parse_groq_duration("10s") == pytest.approx(10.0)


def test_parse_groq_duration_empty_string_returns_none():
    assert parse_groq_duration("") is None


def test_parse_groq_duration_garbage_returns_none():
    assert parse_groq_duration("not a duration") is None


# -- record_response_quota: Groq-style headers -------------------------------

def test_records_groq_headers_on_a_successful_response():
    resp = _response(
        200,
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "999",
            "x-ratelimit-limit-tokens": "12000",
            "x-ratelimit-remaining-tokens": "11959",
            "x-ratelimit-reset-requests": "1m26.4s",
            "x-ratelimit-reset-tokens": "205ms",
        },
    )
    record_response_quota("groq", "llama-3.3-70b-versatile", resp)
    entry = load_quota()["groq"]
    assert entry["model"] == "llama-3.3-70b-versatile"
    assert entry["limit_requests"] == 1000
    assert entry["remaining_requests"] == 999
    assert entry["limit_tokens"] == 12000
    assert entry["remaining_tokens"] == 11959
    assert entry["reset_requests_seconds"] == pytest.approx(86.4)
    assert entry["reset_tokens_seconds"] == pytest.approx(0.205)
    assert entry["source"] == "live_headers"
    assert "observed_at" in entry


def test_records_groq_headers_even_on_a_429_response():
    # Groq's docs: these headers are present on every response, not just
    # successes -- a 429 still tells you exactly where the account stands.
    resp = _response(
        429,
        headers={
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "0",
        },
    )
    record_response_quota("groq", "llama-3.3-70b-versatile", resp)
    entry = load_quota()["groq"]
    assert entry["remaining_requests"] == 0
    assert entry["source"] == "live_headers"


def test_a_later_call_overwrites_earlier_groq_quota_data():
    record_response_quota(
        "groq", "llama-3.3-70b-versatile", _response(200, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999"})
    )
    record_response_quota(
        "groq", "llama-3.3-70b-versatile", _response(200, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "990"})
    )
    assert load_quota()["groq"]["remaining_requests"] == 990


# -- record_response_quota: Gemini-style 429 body ----------------------------

def test_records_gemini_quota_failure_from_the_error_message_text():
    resp = _response(
        429,
        text=(
            "Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 50. Please retry in 34s."
        ),
    )
    record_response_quota("gemini", "gemini-flash-latest", resp)
    entry = load_quota()["gemini"]
    last = entry["last_known_limit"]
    assert last["metric"] == "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    assert last["value"] == 50
    assert "hit_at" in last


def test_gemini_429_without_a_recognizable_quota_message_records_nothing_wrong():
    resp = _response(429, text="some other unrelated error")
    record_response_quota("gemini", "gemini-flash-latest", resp)
    entry = load_quota().get("gemini", {})
    assert "last_known_limit" not in entry


def test_a_non_429_gemini_response_does_not_touch_last_known_limit():
    resp = _response(200, headers={})
    record_response_quota("gemini", "gemini-flash-latest", resp)
    entry = load_quota().get("gemini", {})
    assert "last_known_limit" not in entry


def test_record_response_quota_with_none_response_is_a_noop():
    record_response_quota("groq", "llama-3.3-70b-versatile", None)
    assert load_quota() == {}


def test_recording_for_one_provider_does_not_touch_the_others_entry():
    record_response_quota("groq", "llama-3.3-70b-versatile", _response(200, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999"}))
    record_response_quota("gemini", "gemini-flash-latest", _response(429, text="Quota exceeded for metric: x, limit: 50."))
    data = load_quota()
    assert "groq" in data and "gemini" in data
    assert data["groq"]["source"] == "live_headers"
    assert data["gemini"]["last_known_limit"]["value"] == 50


# -- record_call_attempt -----------------------------------------------------

def test_record_call_attempt_increments_the_daily_counter():
    record_call_attempt("gemini", "gemini-flash-latest")
    record_call_attempt("gemini", "gemini-flash-latest")
    record_call_attempt("gemini", "gemini-flash-latest")
    entry = load_quota()["gemini"]
    assert entry["calls_today"] == 3
    assert entry["model"] == "gemini-flash-latest"


def test_record_call_attempt_resets_on_a_new_day(monkeypatch):
    record_call_attempt("gemini", "gemini-flash-latest")
    record_call_attempt("gemini", "gemini-flash-latest")
    assert load_quota()["gemini"]["calls_today"] == 2

    monkeypatch.setattr("cerebro.llm.quota._today_key", lambda: "2099-01-01")
    record_call_attempt("gemini", "gemini-flash-latest")
    entry = load_quota()["gemini"]
    assert entry["calls_today"] == 1
    assert entry["calls_day_key"] == "2099-01-01"


def test_record_call_attempt_and_response_quota_share_the_same_entry():
    record_call_attempt("groq", "llama-3.3-70b-versatile")
    record_response_quota(
        "groq", "llama-3.3-70b-versatile",
        _response(200, headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999"}),
    )
    entry = load_quota()["groq"]
    assert entry["calls_today"] == 1
    assert entry["limit_requests"] == 1000


def test_load_quota_returns_empty_dict_on_corrupt_file(tmp_path, monkeypatch):
    path = tmp_path / "quota.json"
    path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr("cerebro.llm.quota.QUOTA_PATH", path)
    assert load_quota() == {}
