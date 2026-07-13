"""Local tracking + real API-reported quota usage, persisted to
``~/.cerebro/quota.json`` so ``cerebro quota`` can report on it without
needing a fresh call first.

Groq returns real ``x-ratelimit-*`` headers on EVERY response (success or
429) -- confirmed live against the actual API: ``x-ratelimit-limit-requests``
/ ``-remaining-requests`` (a daily request cap) and ``x-ratelimit-limit-
tokens`` / ``-remaining-tokens`` (a per-minute token cap), plus ``-reset-*``
countdowns in a Go-style duration string ("1m26.4s", "205ms"). This is exact,
straight from the account -- not estimated.

Groq ALSO enforces a separate cumulative tokens-per-day budget that none of
those headers expose at all -- confirmed live: a real map run hit "Daily
quota exhausted" while the headers above still showed near-zero usage on
both dimensions they DO cover. It's only visible reactively, in the 429's
own error text ("...on tokens per day (TPD): Limit 100000, Used 99559"),
the moment it's actually hit -- same fundamental limitation as Gemini having
no quota API at all, just for a dimension Groq's headers happen to omit.

Gemini's API has no equivalent (confirmed against its official rate-limits
docs: the only place to see usage is a web dashboard in AI Studio, nothing
programmatic with just an API key). So for Gemini this only tracks what
CEREBRO ITSELF has called today, plus whatever a real 429's error body
happens to reveal about the specific limit it just hit -- Gemini's
RESOURCE_EXHAUSTED message embeds the quota metric name and its numeric
limit in plain text ("Quota exceeded for metric: ...requests, limit: 50").
Never presented as the account's true remaining quota, since it isn't.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from ..paths import QUOTA_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# Matches Groq's Go-style duration strings: "2h", "1m26.4s", "205ms", "10s".
# Returns None (rather than a wrong guess) for anything that doesn't fit.
_DURATION_RE = re.compile(
    r"^(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?(?:(?P<ms>\d+)ms)?$"
)


def parse_groq_duration(text: str) -> float | None:
    """Parse a Groq reset-duration header into seconds, or None if it
    doesn't match the expected shape -- callers must not guess a fallback."""
    text = (text or "").strip()
    if not text:
        return None
    match = _DURATION_RE.match(text)
    if not match or not any(match.groups()):
        return None
    h, m, s, ms = (match.group(g) for g in ("h", "m", "s", "ms"))
    return float(h or 0) * 3600 + float(m or 0) * 60 + float(s or 0) + float(ms or 0) / 1000


# Gemini's 429 message embeds the metric + its numeric daily/per-model limit
# in plain English, e.g.: "Quota exceeded for metric:
# generativelanguage.googleapis.com/generate_content_free_tier_requests,
# limit: 50. Please retry in 34s." -- this is the only place that number is
# ever exposed; there's no separate structured field to read it from instead.
_GEMINI_QUOTA_RE = re.compile(r"metric:\s*([\w./-]+).*?limit:\s*(\d+)", re.IGNORECASE | re.DOTALL)

# Groq's own daily-quota 429 ("...on tokens per day (TPD): Limit 100000, Used
# 99559") reveals a SEPARATE cumulative TOKENS-per-day budget that is not
# exposed by x-ratelimit-limit-tokens at all -- that header is per-MINUTE
# (TPM), not per-day, confirmed live: a real map run hit this daily cap while
# x-ratelimit-remaining-requests/-tokens both still showed near-zero usage.
# It only ever becomes visible reactively, the moment it's actually hit --
# there is no header exposing it proactively, same fundamental limit as
# Gemini's missing quota API, just for a different dimension.
_GROQ_DAILY_RE = re.compile(
    r"on (tokens|requests) per day \((TPD|RPD)\):\s*Limit (\d+),\s*Used (\d+)", re.IGNORECASE
)


def load_quota() -> dict:
    """Return the persisted quota data, {} if missing/corrupt."""
    if not QUOTA_PATH.exists():
        return {}
    try:
        data = json.loads(QUOTA_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    try:
        QUOTA_PATH.parent.mkdir(parents=True, exist_ok=True)
        QUOTA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def record_response_quota(provider_name: str, model: str, resp) -> None:
    """Inspect one real HTTP response for whatever quota info it reveals and
    persist it. Safe to call on every response, success or failure -- a
    no-op if ``resp`` is None or nothing recognizable is present.

    ``resp`` is a ``requests.Response``, typed loosely here to avoid a hard
    dependency on ``requests`` in this module's public signature.
    """
    if resp is None:
        return
    data = load_quota()
    entry = data.setdefault(provider_name, {})
    entry["model"] = model
    headers = resp.headers

    if "x-ratelimit-limit-requests" in headers:
        # Groq-style headers -- present on every response, not just 429s, so
        # this is exact real-time account state, not a snapshot from the
        # moment we last happened to hit a limit. NOTE: these are a per-DAY
        # request cap and a per-MINUTE token cap -- NOT the same dimension as
        # the separate cumulative tokens-per-day budget below, which these
        # headers never reveal at all, even when it's the thing that's
        # actually about to fail a call.
        limit_requests = int(headers["x-ratelimit-limit-requests"])
        entry["limit_requests"] = limit_requests
        entry["remaining_requests"] = int(headers.get("x-ratelimit-remaining-requests", limit_requests))
        if "x-ratelimit-limit-tokens" in headers:
            entry["limit_tokens"] = int(headers["x-ratelimit-limit-tokens"])
        if "x-ratelimit-remaining-tokens" in headers:
            entry["remaining_tokens"] = int(headers["x-ratelimit-remaining-tokens"])
        reset_requests = parse_groq_duration(headers.get("x-ratelimit-reset-requests", ""))
        if reset_requests is not None:
            entry["reset_requests_seconds"] = reset_requests
        reset_tokens = parse_groq_duration(headers.get("x-ratelimit-reset-tokens", ""))
        if reset_tokens is not None:
            entry["reset_tokens_seconds"] = reset_tokens
        entry["observed_at"] = _now_iso()
        entry["source"] = "live_headers"

    if resp.status_code == 429:
        text = resp.text or ""
        daily = _GROQ_DAILY_RE.search(text)
        if daily:
            dimension, code, limit, used = daily.groups()
            entry["daily_budget"] = {
                "metric": f"{dimension} per day ({code})",
                "value": int(limit),
                "used": int(used),
                "hit_at": _now_iso(),
            }
        else:
            match = _GEMINI_QUOTA_RE.search(text)
            if match:
                entry["last_known_limit"] = {
                    "metric": match.group(1),
                    "value": int(match.group(2)),
                    "hit_at": _now_iso(),
                }

    data[provider_name] = entry
    _save(data)


def record_call_attempt(provider_name: str, model: str) -> None:
    """Cerebro's own local call count for today, day-keyed so it resets on
    its own -- the only usage figure available at all for a provider (like
    Gemini) with no live quota API, and a useful cross-check even for Groq
    (a gap between this and the real header count means something else is
    also using the same account)."""
    data = load_quota()
    entry = data.setdefault(provider_name, {})
    entry["model"] = model
    today = _today_key()
    if entry.get("calls_day_key") != today:
        entry["calls_day_key"] = today
        entry["calls_today"] = 0
    entry["calls_today"] = entry.get("calls_today", 0) + 1
    data[provider_name] = entry
    _save(data)
