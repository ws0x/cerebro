"""Concrete LLM backends.

Both real providers are called over plain HTTP (no vendor SDK) to keep the
dependency surface tiny. Each returns a parsed JSON object from a system+user
prompt. ``MockProvider`` returns deterministic structured output so the whole
pipeline and cache are testable with no key and no network.
"""

from __future__ import annotations

from .base import LLMError, parse_json, post_json


class GroqProvider:
    """Groq — OpenAI-compatible chat completions. Free tier, very fast."""

    name = "groq"

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model = model
        self._url = "https://api.groq.com/openai/v1/chat/completions"

    def complete_json(self, system: str, user: str) -> dict:
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

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def complete_json(self, system: str, user: str) -> dict:
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
