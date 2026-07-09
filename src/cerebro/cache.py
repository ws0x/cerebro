"""Content-addressed cache.

Every expensive artifact (a map/reduce/link LLM response) is keyed by a hash of
everything that could change it — provider, model, prompt version, level, and
the input text. Same inputs ⇒ instant cache hit, so re-running a video, or
upgrading brief→full→expert, never repeats work it already did.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, root: str | Path = ".cerebro-cache", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        joined = "\x00".join(str(p) for p in parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self.root / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self.root / f"{key}.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
