import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from cerebro.llm.base import LLMError, RateLimiter, _is_daily_limit, _retry_delay, parse_json, post_json


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


def test_is_daily_limit_detects_groqs_actual_error_phrasing():
    # The exact phrasing live-observed from a real Groq 429 response.
    resp = _response(
        429,
        text='{"error":{"message":"Rate limit reached ... on tokens per day (TPD): Limit 100000, Used 99559"}}',
    )
    assert _is_daily_limit(resp)


def test_is_daily_limit_false_for_an_ordinary_per_minute_429():
    resp = _response(429, text='{"error":"rate limit exceeded, try again in 2s"}')
    assert not _is_daily_limit(resp)


def test_is_daily_limit_false_for_non_429_status():
    resp = _response(500, text="tokens per day")
    assert not _is_daily_limit(resp)


def test_is_daily_limit_false_for_none_response():
    assert not _is_daily_limit(None)


def test_post_json_fails_fast_on_a_daily_limit_without_sleeping(monkeypatch):
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    daily_limit_resp = _response(429, text="tokens per day (TPD): Limit 100000, Used 99559")
    with patch("requests.post", return_value=daily_limit_resp):
        with pytest.raises(LLMError, match="Daily quota exhausted"):
            post_json("http://x", {}, {}, retries=3)
    assert slept == []  # never slept/retried -- failed on the very first attempt


def test_post_json_still_retries_an_ordinary_429_normally(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    responses = [_response(429, text="rate limit exceeded"), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        assert post_json("http://x", {}, {}, retries=3) == {"ok": True}


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


# -- rate limiter ----------------------------------------------------------

def test_rate_limiter_zero_interval_never_waits():
    limiter = RateLimiter(0)
    start = time.monotonic()
    for _ in range(50):
        limiter.acquire()
    assert time.monotonic() - start < 0.1  # effectively instant, disabled


def test_rate_limiter_spaces_sequential_calls_by_the_interval():
    interval = 0.05
    limiter = RateLimiter(interval)
    start = time.monotonic()
    n = 4
    for _ in range(n):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # First acquire is immediate; each subsequent one waits ~interval.
    assert elapsed >= interval * (n - 1) * 0.9


def test_rate_limiter_paces_concurrent_threads_too():
    interval = 0.05
    limiter = RateLimiter(interval)
    n = 6
    start = time.monotonic()

    def worker():
        limiter.acquire()

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.monotonic() - start
    # Even fired all at once, the n reserved slots are spaced `interval` apart,
    # so the last one can't complete before ~(n-1)*interval.
    assert elapsed >= interval * (n - 1) * 0.9


def test_rate_limiter_negative_interval_is_treated_as_disabled():
    limiter = RateLimiter(-5)
    start = time.monotonic()
    for _ in range(20):
        limiter.acquire()
    assert time.monotonic() - start < 0.1


def test_rate_limiter_backoff_raises_the_interval_multiplicatively():
    limiter = RateLimiter(2.0)
    limiter.backoff(factor=1.6)
    assert limiter.current_interval == pytest.approx(3.2)


def test_rate_limiter_backoff_is_cumulative_across_calls():
    limiter = RateLimiter(2.0)
    limiter.backoff(factor=1.6)
    limiter.backoff(factor=1.6)
    assert limiter.current_interval == pytest.approx(2.0 * 1.6 * 1.6)


def test_rate_limiter_backoff_caps_at_the_max_interval():
    limiter = RateLimiter(20.0)
    for _ in range(10):
        limiter.backoff(factor=1.6)
    assert limiter.current_interval == 30.0


def test_rate_limiter_backoff_from_a_zero_starting_interval_still_slows_down():
    # A zero interval means "disabled" for acquire(), but backoff() must still
    # produce a real, positive interval -- 0 * factor would stay 0 forever.
    limiter = RateLimiter(0)
    limiter.backoff(factor=1.6)
    assert limiter.current_interval > 0


def test_rate_limiter_hit_limit_is_false_until_backoff_is_called():
    limiter = RateLimiter(1.0)
    assert limiter.hit_limit is False
    limiter.backoff()
    assert limiter.hit_limit is True


def test_post_json_calls_backoff_on_an_ordinary_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    limiter = RateLimiter(2.0)
    responses = [_response(429, text="rate limit exceeded"), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        post_json("http://x", {}, {}, retries=3, rate_limiter=limiter)
    assert limiter.hit_limit is True
    assert limiter.current_interval == pytest.approx(3.2)


def test_post_json_does_not_backoff_on_a_daily_limit(monkeypatch):
    # A daily-quota 429 fails fast and won't recover within this run -- calling
    # backoff() for it would slow down every remaining call for no reason,
    # since none of them are going to succeed today regardless of pacing.
    slept = []
    monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
    limiter = RateLimiter(2.0)
    daily_limit_resp = _response(429, text="tokens per day (TPD): Limit 100000, Used 99559")
    with patch("requests.post", return_value=daily_limit_resp):
        with pytest.raises(LLMError, match="Daily quota exhausted"):
            post_json("http://x", {}, {}, retries=3, rate_limiter=limiter)
    assert limiter.hit_limit is False
    assert limiter.current_interval == 2.0


def test_post_json_without_a_rate_limiter_still_works_normally(monkeypatch):
    # rate_limiter is optional -- existing callers that don't pass one (or
    # tests) must not break.
    monkeypatch.setattr("time.sleep", lambda _: None)
    responses = [_response(429, text="rate limit exceeded"), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        assert post_json("http://x", {}, {}, retries=3) == {"ok": True}


def test_post_json_acquires_the_limiter_again_before_each_retry(monkeypatch):
    # The actual bug this guards against: a retry that only sleeps its own
    # local delay and never re-takes a slot in the shared queue lets
    # concurrent threads' retries fire uncoordinated, re-colliding with the
    # real rate limit even after backoff() raised the interval. A retry is a
    # genuinely new HTTP request and must be paced exactly like a fresh one.
    monkeypatch.setattr("time.sleep", lambda _: None)
    limiter = MagicMock()
    limiter.hit_limit = False
    responses = [_response(429, text="rate limit exceeded"), _response(200, json_body={"ok": True})]
    with patch("requests.post", side_effect=responses):
        post_json("http://x", {}, {}, retries=3, rate_limiter=limiter)
    limiter.backoff.assert_called_once()
    limiter.acquire.assert_called_once()  # once, for the single retry before success


def test_post_json_does_not_acquire_the_limiter_again_on_a_daily_limit(monkeypatch):
    # A daily-quota 429 fails fast with no retry at all -- there is no retry
    # attempt to pace, so acquire() must not be called an extra time here.
    monkeypatch.setattr("time.sleep", lambda s: None)
    limiter = MagicMock()
    daily_limit_resp = _response(429, text="tokens per day (TPD): Limit 100000, Used 99559")
    with patch("requests.post", return_value=daily_limit_resp):
        with pytest.raises(LLMError, match="Daily quota exhausted"):
            post_json("http://x", {}, {}, retries=3, rate_limiter=limiter)
    limiter.acquire.assert_not_called()
    limiter.backoff.assert_not_called()


def test_post_json_acquires_the_limiter_for_every_retry_across_multiple_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    limiter = MagicMock()
    responses = [
        _response(429, text="rate limit exceeded"),
        _response(429, text="rate limit exceeded"),
        _response(200, json_body={"ok": True}),
    ]
    with patch("requests.post", side_effect=responses):
        post_json("http://x", {}, {}, retries=3, rate_limiter=limiter)
    assert limiter.acquire.call_count == 2  # two retries before the third attempt succeeds


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
