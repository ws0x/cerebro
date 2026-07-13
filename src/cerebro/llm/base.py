"""Provider contract + shared JSON/HTTP helpers."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Protocol

import requests

from ..console import qprint
from .quota import record_response_quota


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries or returns junk."""


_BACKOFF_MAX_INTERVAL = 30.0  # seconds -- same safety cap as _retry_delay's


class RateLimiter:
    """Thread-safe, SELF-CORRECTING minimum-interval limiter.

    Each ``acquire()`` reserves the next available time slot (slots spaced at
    least ``min_interval`` seconds apart) and sleeps until it -- so concurrent
    MAP threads are PACED under a provider's per-minute limit rather than
    bursting all at once and getting 429-stormed. The slot is reserved inside
    the lock but slept for outside it, so threads don't serialize on the lock
    while waiting. A ``min_interval`` of 0 disables it entirely (e.g. tests,
    the offline mock).

    A hardcoded starting interval is a guess -- it can be wrong for a given
    account/tier in either direction. ``backoff()`` is the fix: called from
    post_json the moment ANY thread observes an ordinary 429, it raises the
    interval for EVERY subsequent acquire() across the whole thread pool
    immediately, not just that one call's own retry. A burst that started
    too fast self-corrects mid-run instead of every thread independently
    re-colliding with the same too-optimistic pace."""

    def __init__(self, min_interval: float):
        self._min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._hit_limit = False  # did this run ever need to back off?

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_allowed)
            self._next_allowed = slot + self._min_interval
            wait = slot - now
        if wait > 0:
            time.sleep(wait)

    def backoff(self, factor: float = 1.6) -> None:
        """Slow down for every future acquire() -- the assumed interval was
        too optimistic for this account's real limit. Multiplicative, capped,
        and monotonic within a run (never speeds back up mid-run -- that's
        cross-run persistence's job, see llm/pacing.py)."""
        with self._lock:
            self._hit_limit = True
            base = self._min_interval if self._min_interval > 0 else 1.0
            self._min_interval = min(base * factor, _BACKOFF_MAX_INTERVAL)

    @property
    def current_interval(self) -> float:
        return self._min_interval

    @property
    def hit_limit(self) -> bool:
        return self._hit_limit


class LLMProvider(Protocol):
    name: str
    model: str

    def complete_json(self, system: str, user: str) -> dict: ...


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(content: str) -> dict:
    """Best-effort extraction of a JSON object from a model response.

    Handles bare JSON, ```json fenced blocks, and leading/trailing prose by
    falling back to the first balanced ``{...}`` span.
    """
    content = content.strip()
    if not content:
        raise LLMError("Empty model response")

    for candidate in (content, *(m.group(1) for m in _JSON_FENCE.finditer(content))):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except Exception:
            pass
    raise LLMError(f"Could not parse JSON from response: {content[:200]!r}")


# A daily-quota 429 ("tokens per day (TPD)", "requests per day (RPD)", or a
# bare "per day") cannot recover by retrying within the same run -- it only
# resets tomorrow. Retrying it anyway (as every other 429 correctly does) just
# burns several minutes of guaranteed-failed backoff per remaining call, which
# is exactly what happened live: a 61-chunk video hit a daily cap on chunk 11
# and spent ~6.5 minutes retrying the other 50 chunks before finally giving up.
_DAILY_LIMIT_RE = re.compile(r"\b(?:tokens|requests)\s+per\s+day\b|\bTPD\b|\bRPD\b|\bper[\s-]day\b", re.IGNORECASE)


def _is_daily_limit(resp: requests.Response | None) -> bool:
    if resp is None or resp.status_code != 429:
        return False
    return bool(_DAILY_LIMIT_RE.search(resp.text[:500]))


_MAX_RETRY_DELAY = 30.0  # seconds -- a safety cap regardless of exponential backoff or a server's own Retry-After


def _retry_delay(resp: requests.Response | None, attempt: int) -> float:
    """Honor a `Retry-After` header when the server sends one (common on
    429s) instead of guessing via blind exponential backoff -- capped
    either way, so an aggressive Retry-After value can't silently block a
    build for minutes with nothing to show it's waiting on purpose, not
    just stuck."""
    if resp is not None:
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return min(float(retry_after), _MAX_RETRY_DELAY)
            except ValueError:
                pass
    return min(1.5 * (2**attempt), _MAX_RETRY_DELAY)


def post_json(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = 60,
    retries: int = 3,
    rate_limiter: "RateLimiter | None" = None,
    provider_name: str | None = None,
    model: str | None = None,
) -> dict:
    """POST with exponential backoff; retries 429/5xx and network errors.

    A retry prints a one-line notice (respecting --quiet) so a build that
    goes quiet for a few seconds under free-tier rate limiting reads as
    "waiting on purpose," not stuck -- this used to retry completely
    silently, indistinguishable from a hang. The final failure message
    calls out a rate limit specifically when that's what actually happened,
    with an actionable next step, instead of a bare HTTP status code.

    ``rate_limiter``, when given, gets told about an ordinary (non-daily)
    429 via ``.backoff()`` -- this is what makes the PACING itself adapt
    mid-run: the shared limiter slows down for every subsequent call across
    the whole thread pool the moment any one of them discovers the assumed
    interval was too fast, not just this call's own retry delay.

    ``provider_name``/``model``, when given, feed every real response (success
    or failure alike) to ``record_response_quota`` -- Groq's account-level
    remaining-quota headers are present on every response regardless of
    status, so this is how `cerebro quota` gets genuinely live numbers
    without a dedicated usage-check call.
    """
    last_exc: Exception | None = None
    last_status: int | None = None
    for attempt in range(retries):
        resp: requests.Response | None = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if provider_name:
                record_response_quota(provider_name, model or "", resp)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_status = resp.status_code
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.ok:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except (requests.RequestException, LLMError) as exc:
            last_exc = exc
            if _is_daily_limit(resp):
                raise LLMError(
                    "Daily quota exhausted (HTTP 429, tokens/requests per day). This won't "
                    "recover by retrying -- it resets tomorrow. Switch --engine to another "
                    "provider, or use --engine heuristic to keep going offline right now."
                ) from exc
            if resp is not None and resp.status_code == 429 and rate_limiter is not None:
                rate_limiter.backoff()
            if attempt < retries - 1:
                delay = _retry_delay(resp, attempt)
                reason = "rate limited (429)" if last_status == 429 else f"request failed ({exc})"
                qprint(f"[dim]  ⏳ {reason} — retrying in {delay:.0f}s… ({attempt + 1}/{retries - 1})[/]")
                time.sleep(delay)
                if rate_limiter is not None:
                    # A retry is a genuinely new HTTP request, not a continuation
                    # of the old one -- it must take its own slot in the SAME
                    # shared queue as fresh calls. Without this, concurrent
                    # threads' retries fire the moment their own local delay
                    # expires, independent of each other and of backoff()'s
                    # just-raised interval, and re-collide with the real limit
                    # exactly as before backoff() existed.
                    rate_limiter.acquire()
    if last_status == 429:
        raise LLMError(
            f"Rate limited after {retries} attempts (HTTP 429). Free-tier limits "
            "usually clear within a minute or two -- wait and retry, switch --engine, "
            "or use --engine heuristic to keep going offline right now."
        )
    raise LLMError(f"Request failed after {retries} attempts: {last_exc}")
