"""Detect an author's own enumeration in a transcript ("non-negotiable number
one is…", "tip 3", "step five") and recover it as an ordered spine.

WHY THIS EXISTS. A lot of educational video is an explicit numbered list --
"7 habits", "5 mistakes", "three steps". The person watching encodes exactly
that: "there were 7 things." A mind map that dissolves those 7 into 6 renamed
thematic clusters (which is what a free-form REDUCE does, and is correct for
*un*structured talk) fights that memory instead of cueing it. This is the
transcript equivalent of a PDF's table of contents: when the author handed us
the structure, use it directly rather than inventing a new one -- the exact
judgment ``ingest/pdf.py`` already makes for documents, extended to speech.

Pure and offline: no LLM, no network. Returns the sections it's *confident*
about, or an empty list (caller falls back to normal free-form structuring).
False positives are worse than false negatives here -- wrongly imposing a
numbered spine on a video that isn't a list would be a visible failure, so the
gate is deliberately strict (a real ascending 1→N run of at least
``_MIN_ITEMS`` items, an explicit list-noun or "number"/"#" cue required, never
a bare digit).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..transcript import Transcript

_MIN_ITEMS = 3  # fewer than this isn't a "list" worth restructuring around

# Spoken/So-written ordinals 1..20 -> int. Digits handled separately.
_ORDINAL_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}
_ORD_ALT = "|".join(list(_ORDINAL_WORDS) + [str(n) for n in range(1, 21)])

# Nouns that head a countable list. An explicit one of these (or "number"/"#")
# is *required* -- it's what separates "step three" from someone saying "3" in
# passing. Kept broad because listicles use many framings.
_LIST_NOUN = (
    r"non-?negotiables?|tips?|steps?|points?|reasons?|rules?|lessons?|habits?|"
    r"ways?|secrets?|principles?|keys?|things?|laws?|pillars?|traits?|signs?|"
    r"mistakes?|lies?|myths?|questions?|strategies|strategy|factors?|phases?|"
    r"stages?|levels?|commandments?|lessons?|elements?"
)

# Primary, highest-precision: an optional list-noun then "number"/"No."/"#" then
# the ordinal -- "non-negotiable number one", "tip number 3", "#4", "No. 5".
# Note "no\." requires the literal period: a bare "no" is an extremely common
# English word ("no one", "no two ways") and matching it as a list cue is a
# false-positive magnet, so only the abbreviation form counts.
_PAT_NUMBER = re.compile(
    rf"(?:(?:{_LIST_NOUN})\s+)?(?:number|no\.|#)\s*({_ORD_ALT})\b(?:\s+(?:is|are))?[\s:,.\-–—]*([^.!?\n]{{0,120}})",
    re.IGNORECASE,
)
# Secondary: a list-noun *directly* before the ordinal, no "number" -- "step
# three is", "reason 2:". Two guards make this high-precision: the list-noun is
# mandatory (a bare number can never match), AND a declaration marker
# ("is"/"are" or ":"/"-"/"—") must immediately follow the ordinal. Without the
# second guard, "win some[thing] one time" matched ("thing"+"one"), because
# many list-nouns ("thing", "way", "sign") are also everyday words -- requiring
# the declaration means "thing one time" (one→time) can't match while "step
# three is" (three→is) still does.
_PAT_NOUN = re.compile(
    rf"(?:{_LIST_NOUN})\s+({_ORD_ALT})\b(?:\s+(?:is|are)\b|\s*[:–—-])\s*([^.!?\n]{{0,120}})",
    re.IGNORECASE,
)

# Leading connective/filler words stripped off the front of a captured heading
# span before title-casing -- "is to have something bigger" -> "have something
# bigger". Only ever stripped from the *start*, never mid-phrase.
_LEAD_FILLER = {
    "is", "are", "to", "the", "a", "an", "that", "this", "it's", "its",
    "my", "you", "we", "i", "and", "so", "just", "really", "gonna",
    "going", "actually", "basically", "like",
}
# Small words kept lowercase in the middle of a title (never the first word).
# Only genuine articles/coordinating conjunctions/short prepositions -- this
# matches the reference XMind template's own title style, which capitalizes
# longer prepositions ("Over", "Than", "Into") and pronouns ("Your",
# "Yourself") while lowercasing "in"/"as"/"to"/"of".
_TITLE_SMALL = {
    "a", "an", "the", "and", "or", "but", "nor", "for", "of", "on", "in", "to",
    "as", "at", "by", "is", "vs",
}
_WORD_SPLIT = re.compile(r"\s+")
_HEADING_END = re.compile(r"[.!?,;]")


def _smart_titlecase(text: str) -> str:
    words = [w for w in _WORD_SPLIT.split(text.strip()) if w]
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        if i != 0 and lw in _TITLE_SMALL:
            out.append(lw)
        else:
            out.append(lw[:1].upper() + lw[1:])
    return " ".join(out)


def _clean_heading(raw: str, max_words: int = 9) -> str:
    """Best-effort deterministic heading from a raw captured span. The LLM
    enumerated path polishes these further; the offline heuristic engine uses
    them as-is, so they must be presentable on their own."""
    # Cut at the first sentence boundary or comma -- the heading is the clause
    # right after the cue ("keep promises to yourself"), not the whole rest of
    # the sentence ("keep promises to yourself, no matter what, always").
    text = raw.strip()
    m = _HEADING_END.search(text)
    if m and m.start() >= 3:
        text = text[: m.start()]
    words = [w for w in _WORD_SPLIT.split(text) if w]
    # Strip leading filler ("is to ...", "the ...").
    while words and re.sub(r"[^\w']", "", words[0]).lower() in _LEAD_FILLER:
        words.pop(0)
    words = words[:max_words]
    # Trim trailing filler/dangling connectives.
    while words and re.sub(r"[^\w']", "", words[-1]).lower() in _LEAD_FILLER:
        words.pop()
    cleaned = " ".join(words)
    cleaned = re.sub(r"[\s,;:–—-]+$", "", cleaned).strip()
    return _smart_titlecase(cleaned) if cleaned else ""


@dataclass
class EnumeratedSection:
    number: int          # 1-based, as the author numbered it
    heading: str         # cleaned, title-cased author phrasing (no number prefix)
    heading_raw: str     # the raw captured span, for optional LLM polish
    start: float         # seconds into the source where this section begins
    seg_index: int       # index into transcript.segments where it begins


def _offset_index(offsets: list[int], pos: int) -> int:
    """Segment index whose joined-text span contains char ``pos`` (offsets is
    the cumulative start offset of each segment in the joined string)."""
    lo, hi = 0, len(offsets) - 1
    ans = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= pos:
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


def _number_of(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        n = int(token)
        return n if 1 <= n <= 20 else None
    return _ORDINAL_WORDS.get(token)


def detect_enumeration(transcript: Transcript, min_items: int = _MIN_ITEMS) -> list[EnumeratedSection]:
    """Ordered sections if the transcript is confidently an author-numbered
    list, else ``[]``. Numbers are recovered as the greedy ascending 1→N run in
    spoken order, so a later recap ("so number one again…") can't duplicate or
    corrupt the spine."""
    segments = [s for s in transcript.segments if s.text.strip()]
    if len(segments) < min_items:
        return []

    # Join, tracking each segment's char offset and its own start time, so a
    # cue that lands anywhere maps back to a real timestamp/segment.
    parts, offsets, starts = [], [], []
    running = 0
    for s in segments:
        t = s.text.strip()
        offsets.append(running)
        starts.append(s.start)
        parts.append(t)
        running += len(t) + 1  # +1 for the joining space
    joined = " ".join(parts)

    # Collect all candidate matches from both patterns, keyed by position.
    candidates: dict[int, tuple[int, str]] = {}  # pos -> (number, heading_raw)
    for pat in (_PAT_NUMBER, _PAT_NOUN):
        for m in pat.finditer(joined):
            n = _number_of(m.group(1))
            if n is None:
                continue
            pos = m.start()
            # Prefer the match that captured a longer heading tail at a given spot.
            existing = candidates.get(pos)
            if existing is None or len(m.group(2)) > len(existing[1]):
                candidates[pos] = (n, m.group(2))

    if not candidates:
        return []

    # Greedy chain expecting 1, 2, 3, … in spoken order. A match only counts if
    # its number is exactly the next one expected -- this both enforces "starts
    # at 1, ascends by 1" and silently ignores out-of-order recaps.
    ordered_positions = sorted(candidates)
    sections: list[EnumeratedSection] = []
    expected = 1
    for pos in ordered_positions:
        number, raw = candidates[pos]
        if number != expected:
            continue
        heading = _clean_heading(raw)
        seg_i = _offset_index(offsets, pos)
        sections.append(
            EnumeratedSection(
                number=number,
                heading=heading,
                heading_raw=raw.strip(),
                start=starts[seg_i],
                seg_index=seg_i,
            )
        )
        expected += 1

    return sections if len(sections) >= min_items else []
