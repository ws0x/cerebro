"""Provider contract + shared JSON/HTTP helpers."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

import requests


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


def post_json(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int = 60,
    retries: int = 3,
) -> dict:
    """POST with exponential backoff; retries 429/5xx and network errors."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if not resp.ok:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        except (requests.RequestException, LLMError) as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (2**attempt))
    raise LLMError(f"Request failed after {retries} attempts: {last_exc}")
