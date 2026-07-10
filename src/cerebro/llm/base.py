"""Provider contract + shared JSON/HTTP helpers."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

import requests

from ..console import qprint


class LLMError(RuntimeError):
    """Raised when a provider call fails after retries or returns junk."""


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
) -> dict:
    """POST with exponential backoff; retries 429/5xx and network errors.

    A retry prints a one-line notice (respecting --quiet) so a build that
    goes quiet for a few seconds under free-tier rate limiting reads as
    "waiting on purpose," not stuck -- this used to retry completely
    silently, indistinguishable from a hang. The final failure message
    calls out a rate limit specifically when that's what actually happened,
    with an actionable next step, instead of a bare HTTP status code.
    """
    last_exc: Exception | None = None
    last_status: int | None = None
    for attempt in range(retries):
        resp: requests.Response | None = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                last_status = resp.status_code
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.ok:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except (requests.RequestException, LLMError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                delay = _retry_delay(resp, attempt)
                reason = "rate limited (429)" if last_status == 429 else f"request failed ({exc})"
                qprint(f"[dim]  ⏳ {reason} — retrying in {delay:.0f}s… ({attempt + 1}/{retries - 1})[/]")
                time.sleep(delay)
    if last_status == 429:
        raise LLMError(
            f"Rate limited after {retries} attempts (HTTP 429). Free-tier limits "
            "usually clear within a minute or two -- wait and retry, switch --engine, "
            "or use --engine heuristic to keep going offline right now."
        )
    raise LLMError(f"Request failed after {retries} attempts: {last_exc}")
