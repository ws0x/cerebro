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
# RPM). Slightly conservative so bursts never cross the line.
_GROQ_MIN_INTERVAL = _min_interval("CEREBRO_GROQ_MIN_INTERVAL", 2.1)
_GEMINI_MIN_INTERVAL = _min_interval("CEREBRO_GEMINI_MIN_INTERVAL", 4.5)


class GroqProvider:
    """Groq — OpenAI-compatible chat completions. Free tier, very fast."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", min_interval: float | None = None):
        self.api_key = api_key
        self.model = model
        self._url = "https://api.groq.com/openai/v1/chat/completions"
        self._limiter = RateLimiter(_GROQ_MIN_INTERVAL if min_interval is None else min_interval)

    def complete_json(self, system: str, user: str) -> dict:
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
        data = post_json(self._url, headers, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {exc}") from exc
        return parse_json(content)


class GeminiProvider:
    """Google Gemini Flash — free tier, strong JSON adherence."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", min_interval: float | None = None):
        self.api_key = api_key
        self.model = model
        self._limiter = RateLimiter(_GEMINI_MIN_INTERVAL if min_interval is None else min_interval)

    def complete_json(self, system: str, user: str) -> dict:
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
        data = post_json(url, {"Content-Type": "application/json"}, payload)
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
