"""Cross-run persistence for learned per-provider pacing.

A RateLimiter's starting interval is a per-process guess; ``backoff()``
corrects it live but that correction is lost the moment the process exits --
without persistence, every fresh run re-learns the same lesson via the same
kind of failed 429 the previous run already paid for. This is what makes
provider resolution start each run already calibrated: the interval a
provider last settled on (or backed off to) is loaded as its new starting
``min_interval`` (see providers.py).
"""

from __future__ import annotations

import json

from ..paths import PACING_PATH
from .base import RateLimiter


def load_pacing() -> dict[str, float]:
    """Return {provider_name: learned_interval_seconds}, {} if missing/corrupt."""
    if not PACING_PATH.exists():
        return {}
    try:
        data = json.loads(PACING_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def record_pacing(name: str, limiter: RateLimiter, default_interval: float) -> None:
    """Merge this run's learned interval for ``name`` into the persisted file.

    A run that backed off (``limiter.hit_limit``) persists the raised
    interval directly -- the next run should start already-cautious, not
    rediscover the same limit via another failed call. A run that stayed
    clean nudges the *previous* persisted value back down (x0.85, floored at
    ``default_interval``), so a provider that's been fine for a while
    gradually re-speeds toward the hardcoded default instead of staying
    permanently slow because of one long-past incident.
    """
    pacing = load_pacing()
    if limiter.hit_limit:
        pacing[name] = limiter.current_interval
    else:
        previous = pacing.get(name, default_interval)
        pacing[name] = max(default_interval, previous * 0.85)
    try:
        PACING_PATH.parent.mkdir(parents=True, exist_ok=True)
        PACING_PATH.write_text(json.dumps(pacing, indent=2), encoding="utf-8")
    except OSError:
        pass
