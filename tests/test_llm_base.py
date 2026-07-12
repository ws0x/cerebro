from unittest.mock import MagicMock, patch

import pytest

from cerebro.llm.base import LLMError, _retry_delay, parse_json, post_json


def _response(status_code, text="", headers=None, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.text = text
    resp.headers = headers or {}
    resp.json.return_value = json_body or {}
    return resp


def test_retry_delay_honors_retry_after_header():
    resp = _response(429, headers={"Retry-After": "5"})
    assert _retry_delay(resp, attempt=0) == 5.0


def test_retry_delay_caps_retry_after_at_max():
    resp = _response(429, headers={"Retry-After": "300"})
    assert _retry_delay(resp, attempt=0) == 30.0


def test_retry_delay_falls_back_to_exponential_backoff_without_header():
    resp = _response(429, headers={})
    assert _retry_delay(resp, attempt=0) == pytest.approx(1.5)
    assert _retry_delay(resp, attempt=1) == pytest.approx(3.0)


def test_retry_delay_falls_back_when_response_is_none():
    # e.g. a network-level RequestException that never got an HTTP response at all
    assert _retry_delay(None, attempt=0) == pytest.approx(1.5)


def test_retry_delay_ignores_a_malformed_retry_after_header():
    resp = _response(429, headers={"Retry-After": "not-a-number"})
    assert _retry_delay(resp, attempt=0) == pytest.approx(1.5)


def test_post_json_succeeds_immediately_on_200():
    with patch("requests.post", return_value=_response(200, json_body={"ok": True})):
        assert post_json("http://x", {}, {}) == {"ok": True}


def test_post_json_retries_a_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    responses = [_response(429, headers={"Retry-After": "1"}), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        assert post_json("http://x", {}, {}, retries=3) == {"ok": True}


def test_post_json_gives_a_rate_limit_specific_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", return_value=_response(429, text="rate limited")):
        with pytest.raises(LLMError, match="Rate limited after 3 attempts"):
            post_json("http://x", {}, {}, retries=3)


def test_post_json_gives_a_generic_error_for_non_rate_limit_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", return_value=_response(500, text="server exploded")):
        with pytest.raises(LLMError, match="Request failed after 3 attempts"):
            post_json("http://x", {}, {}, retries=3)


def test_post_json_retries_a_network_exception_not_just_http_errors(monkeypatch):
    import requests

    monkeypatch.setattr("time.sleep", lambda _: None)
    with patch("requests.post", side_effect=[requests.ConnectionError("dns failed"), _response(200, json_body={"ok": True})]):
        assert post_json("http://x", {}, {}, retries=3) == {"ok": True}


def test_post_json_prints_a_retry_notice(monkeypatch, capsys):
    monkeypatch.setattr("time.sleep", lambda _: None)
    responses = [_response(429, headers={"Retry-After": "2"}), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        post_json("http://x", {}, {}, retries=3)
    out = capsys.readouterr().out
    assert "rate limited" in out.lower()


def test_parse_json_bare_json():
    assert parse_json('{"label": "hello"}') == {"label": "hello"}


def test_parse_json_fenced_with_json_tag():
    content = '```json\n{"label": "hello"}\n```'
    assert parse_json(content) == {"label": "hello"}


def test_parse_json_fenced_without_json_tag():
    content = '```\n{"label": "hello"}\n```'
    assert parse_json(content) == {"label": "hello"}


def test_parse_json_with_surrounding_prose_falls_back_to_brace_scanning():
    content = 'Sure, here is the JSON you asked for:\n{"label": "hello"}\nHope that helps!'
    assert parse_json(content) == {"label": "hello"}


def test_parse_json_finds_the_first_valid_candidate_when_an_earlier_fence_is_broken():
    # First fenced block is truncated/invalid JSON; the model's real answer
    # is the second one -- parse_json must not give up after the first miss.
    content = '```json\n{"broken": \n```\nActually, ```json\n{"label": "hello"}\n```'
    assert parse_json(content) == {"label": "hello"}


def test_parse_json_empty_string_raises():
    with pytest.raises(LLMError, match="Empty model response"):
        parse_json("")


def test_parse_json_whitespace_only_raises():
    with pytest.raises(LLMError, match="Empty model response"):
        parse_json("   \n  ")


def test_parse_json_unparseable_garbage_raises_with_truncated_content_in_message():
    garbage = "not json at all, just plain prose with no braces whatsoever"
    with pytest.raises(LLMError, match="Could not parse JSON"):
        parse_json(garbage)


def test_parse_json_mismatched_braces_raise_rather_than_return_garbage():
    # Has a '{' and a later '}' but the span between them still isn't valid
    # JSON -- must raise LLMError, not silently return something wrong.
    content = "{not valid json} trailing { also not valid }"
    with pytest.raises(LLMError, match="Could not parse JSON"):
        parse_json(content)


def test_parse_json_error_message_truncates_a_very_long_response():
    long_garbage = "x" * 500
    with pytest.raises(LLMError) as exc_info:
        parse_json(long_garbage)
    assert len(str(exc_info.value)) < 300
