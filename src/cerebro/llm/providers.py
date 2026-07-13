"""Concrete LLM backends.

Both real providers are called over plain HTTP (no vendor SDK) to keep the
dependency surface tiny. Each returns a parsed JSON object from a system+user
prompt. ``MockProvider`` returns deterministic structured output so the whole
pipeline and cache are testable with no key and no network.
"""

from __future__ import annotations

import json
import os

from .base import LLMError, RateLimiter, parse_json, post_json
from .pacing import load_pacing
from .quota import record_call_attempt


def _min_interval(env_name: str, default: float) -> float:
    """Per-provider pacing, overridable via env for a different tier/account.
    Defaults are deliberately a touch under each free tier's published
    requests-per-minute so a full run of concurrent MAP calls stays a
    well-behaved client instead of 429-storming."""
    try:
        return max(0.0, float(os.environ.get(env_name, default)))
    except (TypeError, ValueError):
        return default


# Groq free llama-3.3-70b ~30 RPM; Gemini free 2.5-flash is tighter (~10-15
# RPM). Slightly conservative so bursts never cross the line. These are also
# the decay floor a persisted interval nudges back toward on a clean run --
# see llm/pacing.py.
_GROQ_DEFAULT_INTERVAL = 2.1
_GEMINI_DEFAULT_INTERVAL = 4.5
DEFAULT_INTERVALS = {"groq": _GROQ_DEFAULT_INTERVAL, "gemini": _GEMINI_DEFAULT_INTERVAL}


def _starting_interval(provider_name: str, env_name: str, hardcoded_default: float) -> float:
    """A provider's interval learned in a previous run (raised via
    backoff(), or decayed back down after a clean one) is loaded fresh here
    as this run's starting point -- so a run doesn't have to rediscover the
    same real rate limit via another failed call every single time.

    Read at CONSTRUCTION time rather than cached at module-import time: the
    module is typically only imported once per process anyway, but reading
    fresh keeps this simple to test and correct if that ever changes. An
    explicit env var still wins over whatever was learned, since that's a
    deliberate per-account override."""
    persisted = load_pacing().get(provider_name, hardcoded_default)
    return _min_interval(env_name, persisted)


class GroqProvider:
    """Groq — OpenAI-compatible chat completions. Free tier, very fast."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", min_interval: float | None = None):
        self.api_key = api_key
        self.model = model
        self._url = "https://api.groq.com/openai/v1/chat/completions"
        if min_interval is None:
            min_interval = _starting_interval("groq", "CEREBRO_GROQ_MIN_INTERVAL", _GROQ_DEFAULT_INTERVAL)
        self._limiter = RateLimiter(min_interval)

    def complete_json(self, system: str, user: str) -> dict:
        record_call_attempt(self.name, self.model)
        self._limiter.acquire()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = post_json(
            self._url, headers, payload, rate_limiter=self._limiter, provider_name=self.name, model=self.model
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {exc}") from exc
        return parse_json(content)


class GeminiProvider:
    """Google Gemini Flash — free tier, strong JSON adherence."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-flash-latest", min_interval: float | None = None):
        self.api_key = api_key
        self.model = model
        if min_interval is None:
            min_interval = _starting_interval("gemini", "CEREBRO_GEMINI_MIN_INTERVAL", _GEMINI_DEFAULT_INTERVAL)
        self._limiter = RateLimiter(min_interval)

    def complete_json(self, system: str, user: str) -> dict:
        record_call_attempt(self.name, self.model)
        self._limiter.acquire()
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.2},
        }
        data = post_json(
            url,
            {"Content-Type": "application/json"},
            payload,
            rate_limiter=self._limiter,
            provider_name=self.name,
            model=self.model,
        )
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {exc}") from exc
        return parse_json(content)


class MockProvider:
    """Deterministic stand-in. Branches on the ``TASK:`` tag in the system
    prompt so map/reduce/link each return sensible shapes. Counts calls so
    tests can assert cache hits."""

    name = "mock"
    model = "mock-1"

    def __init__(self):
        self.calls = 0

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if "TASK: BATCH MAP" in system:
            # Checked before "TASK: MAP" since that's a substring of this tag.
            payload = json.loads(user)
            return {
                "results": [
                    {
                        "id": seg["id"],
                        "topic": "Key idea from segment",
                        "type": "concept",
                        "summary": "A concise summary of this segment.",
                        "points": [
                            {"title": "Supporting point one", "type": "detail"},
                            {"title": "Supporting point two", "type": "example"},
                        ],
                    }
                    for seg in payload.get("segments", [])
                ]
            }
        if "TASK: MAP" in system:
            return {
                "topic": "Key idea from segment",
                "type": "concept",
                "summary": "A concise summary of this segment.",
                "points": [
                    {"title": "Supporting point one", "type": "detail"},
                    {"title": "Supporting point two", "type": "example"},
                ],
            }
        if "TASK: REDUCE" in system:
            return {
                "central": "Central Topic",
                "children": [
                    {
                        "title": "First main branch",
                        "type": "topic",
                        "note": "branch note",
                        "children": [
                            {"title": "Sub-point A", "type": "concept", "children": []},
                            {"title": "Sub-point B", "type": "example", "children": []},
                        ],
                    },
                    {"title": "Second main branch", "type": "topic", "children": []},
                ],
            }
        if "TASK: LINK" in system:
            return {"relationships": [{"from": 1, "to": 3, "label": "relates to"}]}
        if "TASK: HEADINGS" in system:
            # Return nothing usable, so the enumerated path falls back to its
            # own deterministic (already title-cased) headings. Polishing is a
            # live-LLM nicety; the mock exercises the robust fallback instead.
            return {}
        if "TASK: SECTION" in system:
            return {
                "note": "What the author claims in this section.",
                "points": [
                    {"title": "Section key point", "type": "insight"},
                    {"title": "A concrete example", "type": "example"},
                ],
            }
        return {}
