"""Anchor verify-and-repair: catch concrete hooks the source stated but the
map dropped, and re-attach them via one targeted LLM call.

The MAP/REDUCE compression stages sometimes fold away exactly the details a
learner remembers by -- a specific number, an enumerated list of named
techniques ("dropout, weight decay, early stopping"), a named person/book.
The prompts already call these "NON-NEGOTIABLE", but nothing verified the
guarantee held (a live run of an intro-to-neural-nets transcript dropped every
regularization technique at `full`). This closes that gap:

  1. DETECT deterministically (no LLM): anchors present in the source text but
     absent from the finished map -- numbers, "such as X, Y, Z" example lists,
     and multi-word proper nouns.
  2. REPAIR via ONE targeted re-prompt, and only when step 1 found something.
     The model is the precision backstop: it re-adds only the genuine,
     currently-absent anchors and ignores over-detected noise.

Repair only ever ADDS leaf nodes under the most relevant existing node -- it
never restructures -- so the map stays faithful. Skipped at `brief` (an advance
organizer is deliberately sparse).
"""

from __future__ import annotations

import json
import re

from ..cache import Cache
from ..ir import MindMap, NodeType
from ..llm.base import LLMError, LLMProvider
from ..prompts import ANCHOR_REPAIR_SYSTEM, PROMPT_VERSION

# Meaningful numeric anchors: currency, percentages, decimals, and multi-digit
# or comma-grouped integers. A bare single digit is excluded -- far more often
# enumeration noise ("point one", "step 2") than a statistic worth preserving.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])
    (?:
        \$\s?\d[\d,]*(?:\.\d+)?      # $1,500
      | \d[\d,]*(?:\.\d+)?\s?%       # 20%  /  3.5 %
      | \d+\.\d+                     # 3.14
      | \d[\d,]+                     # 784, 13,000  (2+ chars -> bare single digit excluded)
    )
    """,
    re.VERBOSE,
)

# Enumerated examples the author explicitly flags -- exactly the "dropout,
# weight decay, and early stopping" case. Captures the span right after an
# example cue, up to the next sentence break.
_LIST_CUE_RE = re.compile(r"\b(?:such as|like|including|e\.g\.,?)\s+(.+?)(?:[.!?;]|$)", re.IGNORECASE)
_LIST_SPLIT_RE = re.compile(r",|\band\b|\bor\b", re.IGNORECASE)

# Multi-word proper nouns (2+ consecutive Capitalized words): names, books,
# frameworks, places. Anything already present in the map is filtered out below.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z'’]+(?:\s+[A-Z][A-Za-z'’]+)+")
_PROPER_STOP = {"chapter", "section", "part", "figure", "table", "appendix", "the"}
# Common words that are capitalized only because they START a sentence -- strip
# them off the front of a proper-noun match so "As Carl Jung" → "Carl Jung".
_PROPER_LEAD = {
    "as", "the", "to", "it", "this", "that", "in", "on", "a", "an", "but", "and",
    "so", "we", "you", "they", "he", "she", "i", "if", "when", "then", "for",
    "of", "at", "by", "is", "are", "our", "your", "my", "his", "her", "there",
}

_MAX_ANCHORS = 8  # bound the repair payload; more than this is noise, not loss


def _coerce_type(value) -> NodeType:
    try:
        return NodeType(value)
    except (ValueError, TypeError):
        return NodeType.detail


def _norm(text: str) -> str:
    return re.sub(r"[,\s]+", " ", text.lower()).strip()


def _raw_map_text(mm: MindMap) -> str:
    parts: list[str] = []
    for n in mm.root.walk():
        parts.append(n.title or "")
        if n.note:
            parts.append(n.note)
    return " ".join(parts)


def _num_key(tok: str) -> str:
    return re.sub(r"[^\d.]", "", tok).strip(".")


def _list_items(source: str) -> list[str]:
    items: list[str] = []
    for m in _LIST_CUE_RE.finditer(source):
        span = m.group(1)[:80]
        for part in _LIST_SPLIT_RE.split(span):
            cleaned = re.sub(r"[^\w\s'’-]", " ", part).strip()
            # Cap to 2 words: enumerated example items are noun phrases
            # ("weight decay", "early stopping"), and a 2-word cap trims the
            # trailing prose a run-on list drags in ("weight decay help" ->
            # "weight decay") without losing the anchor itself.
            cleaned = " ".join(cleaned.split()[:2])
            if cleaned and len(cleaned) >= 3:
                items.append(cleaned)
    return items


def _proper_nouns(source: str) -> list[str]:
    out: list[str] = []
    for m in _PROPER_NOUN_RE.finditer(source):
        words = m.group(0).split()
        while words and words[0].lower() in _PROPER_LEAD:
            words.pop(0)  # drop a sentence-initial capitalized function word
        if len(words) < 2:
            continue
        if all(w.lower() in _PROPER_STOP for w in words):
            continue
        out.append(" ".join(words))
    return out


def find_missing_anchors(source: str, mm: MindMap, max_anchors: int = _MAX_ANCHORS) -> list[str]:
    """Concrete anchors present in ``source`` but absent from ``mm`` -- the
    candidate list handed to the repair prompt. High precision on numbers
    (digit-key comparison) and proper nouns; the example-list detector casts a
    slightly wider net that the repair prompt then filters."""
    raw_map = _raw_map_text(mm)
    map_text = _norm(raw_map)
    map_num_keys = {_num_key(m.group(0)) for m in _NUMBER_RE.finditer(raw_map)}

    missing: list[str] = []
    seen: set[str] = set()

    for m in _NUMBER_RE.finditer(source):
        tok = m.group(0).strip()
        key = _num_key(tok)
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in map_num_keys:
            missing.append(tok)

    for term in _list_items(source) + _proper_nouns(source):
        key = _norm(term)
        if not key or len(key) < 3 or key in seen:
            continue
        seen.add(key)
        if key not in map_text:
            missing.append(term)

    return missing[:max_anchors]


def _repair(
    mm: MindMap, missing: list[str], provider: LLMProvider, cache: Cache, on_event
) -> int:
    nodes = [n for n in mm.root.walk() if n.type != NodeType.root]
    if not nodes:
        return 0
    listing = [
        {"id": i, "title": n.title, **({"note": n.note} if n.note else {})}
        for i, n in enumerate(nodes)
    ]
    user = json.dumps({"nodes": listing, "candidates": missing}, ensure_ascii=False)
    key = Cache.key(provider.name, provider.model, PROMPT_VERSION, "anchor_repair", mm.level, user)
    result = cache.get(key)
    if result is None:
        try:
            result = provider.complete_json(ANCHOR_REPAIR_SYSTEM, user)
        except LLMError as exc:
            on_event("anchor_error", error=str(exc))
            return 0
        cache.set(key, result)

    count = 0
    for rep in result.get("repairs", []) or []:
        try:
            to = int(rep["to"])
            title = str(rep.get("title", "")).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if title and 0 <= to < len(nodes):
            nodes[to].add(title, type=_coerce_type(rep.get("type")))
            count += 1
    return count


def verify_and_repair_anchors(
    mm: MindMap,
    source: str,
    provider: LLMProvider,
    cache: Cache,
    level: str,
    on_event=None,
) -> int:
    """Detect source anchors missing from ``mm`` and re-attach the genuine ones.
    Returns how many were repaired. No LLM call unless something is missing;
    skipped entirely at ``brief`` level."""
    on_event = on_event or (lambda *a, **k: None)
    if level == "brief":
        return 0
    missing = find_missing_anchors(source, mm)
    if not missing:
        return 0
    on_event("anchor_check", missing=len(missing))
    repaired = _repair(mm, missing, provider, cache, on_event)
    if repaired:
        on_event("anchor_repaired", count=repaired)
    return repaired
