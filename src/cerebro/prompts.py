"""Prompt templates for the map → reduce → link pipeline.

Templates use ``str.format`` placeholders (``{node_types}``, ``{depth}``) with
literal JSON braces doubled (``{{`` / ``}}``). Bump ``PROMPT_VERSION`` whenever a
prompt changes so the cache invalidates cleanly (it is part of every cache key).
"""

PROMPT_VERSION = "v7"

NODE_TYPES = "topic, concept, definition, example, insight, action, warning, question, detail"

_GROUNDING = """GROUNDING RULE (critical): only use what is explicitly stated in the
input. Do not introduce outside facts, named techniques, examples, or specifics
that are implied by the topic but not actually said. If the input refers to
something generically (e.g. "an activation function"), keep your title equally
generic (e.g. "Activation Function") — do NOT name a specific instance (e.g.
"Sigmoid" or "ReLU") unless that specific name is literally present in the text."""

_ANCHORS = """ANCHOR RULE (preserve memorable hooks): if the input contains any of the
following, you MUST keep it — never drop it, round it, or generalize it into an
abstract umbrella. These concrete anchors are how a learner actually remembers
the abstract point; a map without them is forgettable.
- a direct quotation → keep it close to verbatim, WITH attribution if stated
  (e.g. 'Carl Jung: until you make the unconscious conscious, you will call it
  fate'); type it "example".
- a named book, person, study, framework, or place (e.g. "Man's Search for
  Meaning") → type it "insight".
- a concrete personal story or vivid example (e.g. tearing a muscle before the
  competition and showing up anyway) → a short phrase naming it, type "example".

NUMERIC ANCHORS (a common failure mode — read carefully): any specific number,
count, percentage, statistic, date, price, or measurement is NON-NEGOTIABLE and
must appear VERBATIM, digit-for-digit, exactly as stated (784, not "hundreds";
13,000, not "thousands"; 1,500 calories, not "a calorie deficit"). Never fold a
number into a vaguer paraphrase of the concept it belongs to — write it OUT, in
the title or note of the point it belongs to, or as its own point if it doesn't
cleanly attach to one. Type it "detail". A number is exactly as mandatory as a
quotation — do not keep it at a shallow level of detail and then quietly drop
it once the map gets deeper or more elaborate; a more detailed map must never
contain FEWER concrete numbers than a shallower one would.

Never treat an anchor — numeric or otherwise — as a "duplicate" to merge away."""

_MAP = """You are an expert knowledge cartographer. TASK: MAP.
You receive one segment of a video transcript. Extract its core teaching as a
compact JSON object. Do not summarize the whole video — only this segment.

{grounding}

{anchors}

Write all titles, summaries, and points in the same language as the input transcript. Keep JSON keys in English.

Return ONLY JSON with this shape:
{{
  "topic": "a short noun-phrase title, max 8 words",
  "type": "one of: {node_types}",
  "summary": "1-2 sentence plain summary of the segment, naming any quote/book/story it contains",
  "points": [
    {{"title": "a concrete sub-point, max 12 words (an anchor quote may be longer)", "type": "one of the types above"}}
  ]
}}
Keep points to the 2-5 most important, PLUS any anchors (quotes, named books/people, stories, numbers) — those are always worth a point. Titles must be self-contained (no 'this' or 'it')."""

# Batched variant of MAP: several segments in one call instead of one call
# per segment, to cut request count for long sources. Each segment is
# extracted INDEPENDENTLY of the others (same contract as single-segment
# MAP) -- batching is a request-shape optimization only, never a reason to
# let one segment's content bleed into another's.
_BATCH_MAP = """You are an expert knowledge cartographer. TASK: BATCH MAP.
You receive several segments of a video transcript, each with an id. Extract
EACH segment's core teaching independently as a compact JSON object, exactly
as if it were the only segment given -- do not let one segment's content
influence another's, and do not summarize the whole video.

{grounding}

{anchors}

Write all titles, summaries, and points in the same language as the input transcript. Keep JSON keys in English.

For EACH segment return one result object in this shape:
{{
  "id": <the segment's id, copied exactly>,
  "topic": "a short noun-phrase title, max 8 words",
  "type": "one of: {node_types}",
  "summary": "1-2 sentence plain summary of the segment, naming any quote/book/story it contains",
  "points": [
    {{"title": "a concrete sub-point, max 12 words (an anchor quote may be longer)", "type": "one of the types above"}}
  ]
}}
Keep points to the 2-5 most important per segment, PLUS any anchors (quotes, named books/people, stories, numbers) — those are always worth a point. Titles must be self-contained (no 'this' or 'it').

Return ONLY JSON:
{{"results": [ <one result object per segment, in any order, each carrying its own id> ]}}
If a segment is empty or unusable, omit it from "results" rather than guessing."""

_REDUCE = """You are an expert knowledge cartographer. TASK: REDUCE.
You receive an ordered list of per-segment extractions from one video. Build a
single, smart, hierarchical mind map: merge genuinely duplicate topics, promote
recurring themes to parent nodes, demote details to leaves, and order branches
logically.

{grounding}
This applies to merging too: you may rename a branch to a more general umbrella
term that covers its children, but never invent a more specific term than what
the segments actually contain.

{anchors}

DEDUP: merge only true duplicates (the same idea said twice). Do NOT create two
sibling nodes that say the same thing in different words — pick the clearer one.
But two genuinely distinct ideas that merely sound similar stay separate.

{notes}

Write all titles, notes, and central subject in the same language as the input segments. Keep JSON keys in English.

Return ONLY JSON with this shape (children may nest to any depth):
{{
  "central": "the overall subject, a short noun phrase",
  "children": [
    {{
      "title": "branch title",
      "type": "one of: {node_types}",
      "note": "see the note rule above",
      "children": [ ]
    }}
  ]
}}
Aim for {depth}. Every title must be self-contained and concise."""

_LINK_TEMPLATE = """You are an expert knowledge cartographer. TASK: LINK.
You receive a numbered list of nodes from a finished mind map; each node may
include a short note describing what the source actually claimed about it.
Each node shows the "section" (top-level branch) it belongs to. Identify the
most important NON-hierarchical relationships between them (dependency, cause,
contrast, example-of, prerequisite).

- STRONGLY PREFER links between nodes in DIFFERENT sections — those reveal
  structure the outline doesn't already show.
- You may link two nodes in the SAME section only when the source states an
  explicit cause-and-effect between them (e.g. "keeping promises" → "builds
  confidence").
- NEVER link a node to its own parent or child; that is already shown by the
  tree and will be discarded.

Prefer connections the source EXPLICITLY states (a cause-and-effect the author
actually asserts — read the notes, not just the titles) over connections that
are merely plausible. Do not invent a link that isn't supported by the given
titles and notes.

Write the label in the same language as the node titles. Keep JSON keys in English.

Return ONLY JSON:
{{
  "relationships": [
    {{"from": 0, "to": 1, "label": "short verb phrase"}}
  ]
}}
Return at most {limit} of the strongest relationships. Use the integer ids shown."""

_CROSS_LINK_TEMPLATE = """You are an expert knowledge cartographer. TASK: CROSS-VIDEO LINKING.
You receive a numbered list of nodes from a course/playlist mind map. Each node includes the video/lesson title it belongs to.
Identify the most important connections (dependency, cause, contrast, prerequisite, builds-on) between concepts in DIFFERENT videos.
Do NOT link nodes within the same video.
Only propose a relationship that is actually supported by the titles and context given — do not invent a connection that isn't reasonably implied.

Write the label in the same language as the node titles. Keep JSON keys in English.

Return ONLY JSON:
{{
  "relationships": [
    {{"from": 0, "to": 1, "label": "short verb phrase"}}
  ]
}}
Return at most {limit} of the strongest cross-video relationships. Use the integer ids shown."""


_FOLDER_LABEL = """You are analyzing a software project's folder structure.
Given a folder's name and the names of its immediate subfolders/files, infer
a short, specific purpose label for what this folder is for, aimed at someone
unfamiliar with the codebase.

GROUNDING RULE: base the label only on the folder name and the file/folder
names actually given — do not assume a specific framework, language, or
technology unless a file name literally evidences it (e.g. "pyproject.toml"
implies Python, but do not guess further than that).

Return ONLY JSON:
{"label": "short purpose phrase, max 6 words"}
If the folder's purpose is already fully obvious from its name alone (e.g.
"tests", "docs", "images"), it is fine to return that same word, capitalized."""

FOLDER_LABEL_SYSTEM = _FOLDER_LABEL


def link_system(limit: int) -> str:
    return _LINK_TEMPLATE.format(limit=limit)


def cross_link_system(limit: int) -> str:
    return _CROSS_LINK_TEMPLATE.format(limit=limit)


# Per-level guidance injected into the REDUCE prompt. Each level is a genuinely
# different cognitive job, not just a bigger/smaller tree:
#   brief  = advance organizer (survey before watching / gist after)
#   full   = comprehension map (the author's structure + key points, explained)
#   expert = elaborative map (+ cross-links, anchors, actionable takeaways)
LEVEL_DEPTH = {
    "brief": "a shallow ADVANCE-ORGANIZER map: 4-7 main branches only, minimal nesting — the gist someone surveys before watching or recalls after",
    "full": "a COMPREHENSION map 3-4 levels deep: the source's own structure with the key point under each",
    "expert": "a rich ELABORATIVE map 4+ levels deep: concepts, the concrete anchors (quotes/stories/numbers), and actionable takeaways typed 'action'",
}

# Per-level note requirement, injected into REDUCE. The future-self test: a note
# should make a node intelligible months later without rewatching the source.
_LEVEL_NOTES = {
    "brief": """NOTES: give each of the few main branches a one-line note capturing its gist. Sub-nodes need no note at this level.""",
    "full": """NOTES (required): every branch node MUST have a "note" — one sentence stating what the source actually claims about it, so the node is intelligible months later without rewatching. Leaf key-points may omit it if their title is already self-explanatory.""",
    "expert": """NOTES (required): every branch and every sub-branch MUST have a "note" — one sentence of what the source actually claims (the future-self test). A node that only restates its title in the note is wasted; say something substantive. Mark actionable takeaways with type "action".""",
}

MAP_SYSTEM = _MAP.format(node_types=NODE_TYPES, grounding=_GROUNDING, anchors=_ANCHORS)
BATCH_MAP_SYSTEM = _BATCH_MAP.format(node_types=NODE_TYPES, grounding=_GROUNDING, anchors=_ANCHORS)


# -- Purpose axis -----------------------------------------------------------

# Orthogonal to level (which controls DEPTH). Purpose changes the ORGANIZING
# logic -- the same content is shaped differently for a different reader goal.
# "general" injects nothing, so its prompt is byte-identical to the historical
# text and its cache keys never change: a zero-disruption default. Purpose is
# appended (not woven in) so this is guaranteed.
_PURPOSE = {
    "general": "",
    "learn": (
        "PURPOSE — LEARNING (first exposure): the reader is meeting this material for the "
        "first time. Build understanding from fundamentals up. Keep the concrete examples, "
        "analogies, worked steps, and memorable anchors — they are how a beginner encodes an "
        "abstract idea. Prefer a clear teaching order over the most compressed one."
    ),
    "review": (
        "PURPOSE — REVIEW (already familiar): the reader knows the basics and is refreshing. "
        "Favor the non-obvious — the connections, exceptions, and easy-to-forget details — and "
        "the high-level structure. Compress introductory basics rather than re-explaining them."
    ),
    "present": (
        "PURPOSE — PRESENTING (teaching others): this map will be used to present or teach. "
        "Favor a clear narrative arc: a strong opening framing, a logical flow from branch to "
        "branch, and branch titles that would work as talking points."
    ),
}


def _with_purpose(system: str, purpose: str) -> str:
    """Append the purpose block. For 'general' (or an unknown value) this
    returns ``system`` unchanged -- byte-identical to the historical prompt."""
    block = _PURPOSE.get(purpose or "general", "")
    return f"{system}\n\n{block}" if block else system


def map_system(purpose: str = "general") -> str:
    return _with_purpose(MAP_SYSTEM, purpose)


def batch_map_system(purpose: str = "general") -> str:
    return _with_purpose(BATCH_MAP_SYSTEM, purpose)


def reduce_system(level: str, purpose: str = "general") -> str:
    depth = LEVEL_DEPTH.get(level, LEVEL_DEPTH["full"])
    notes = _LEVEL_NOTES.get(level, _LEVEL_NOTES["full"])
    base = _REDUCE.format(
        node_types=NODE_TYPES, depth=depth, grounding=_GROUNDING, anchors=_ANCHORS, notes=notes
    )
    return _with_purpose(base, purpose)


# -- Enumerated (author-numbered list) path --------------------------------

# One call: clean the author's spoken lead-ins into short, parallel headings.
# Cleanup only, never invention — the author already told us these; we just
# tidy the ASR ("This is one of the biggest cuz it's overarching... be a
# student of yourself" -> "Be a Student of Yourself").
_HEADING_POLISH = """You are an expert knowledge cartographer. TASK: HEADINGS.
You are titling the sections of a video that is an explicit numbered list.
For each section you get its number, the author's raw spoken lead-in (messy
auto-transcribed speech), and the first words of the section. Return a short,
clean, parallel heading for each — ideally an imperative phrase in the author's
own words.

GROUNDING: base each heading ONLY on that section's own text. Do not invent a
heading the author didn't express. Preserve the author's meaning and wording;
just remove filler, false starts, and transcription noise. Max ~7 words each.

Return ONLY JSON, one entry per input section, in the same order:
{"headings": ["Keep Promises to Yourself", "Get Your House in Order", ...]}"""

HEADING_POLISH_SYSTEM = _HEADING_POLISH


# One call per section: given the author's fixed heading + that section's own
# transcript, fill its content (note + key points). The section title is FIXED
# by the author's enumeration and is NOT the model's to change — this is the
# "structure-filler, not structure-inventor" contract.
_SECTION_FILL = """You are an expert knowledge cartographer. TASK: SECTION.
You are filling in one section of a mind map. The section's title is FIXED (it
is the author's own numbered heading) — do not restate or rename it. You receive
that title and the section's transcript.

{grounding}

{anchors}

Return ONLY JSON:
{{
  "note": "one sentence: what the author actually claims in this section (intelligible on its own months later)",
  "points": [
    {{"title": "a concrete sub-point or anchor, self-contained", "type": "one of: {node_types}"}}
  ]
}}
{points_guidance}"""

_SECTION_POINTS_GUIDANCE = {
    "brief": 'Return an EMPTY "points" list — at brief level the section needs only its note.',
    "full": "Return the 2-4 most important sub-points, PLUS any anchors (quotes, named books/people, stories, numbers).",
    "expert": "Return the 3-6 most important sub-points, ALWAYS including every anchor (quote, named book/person, story, number) and marking actionable takeaways with type \"action\".",
}


def section_fill_system(level: str, purpose: str = "general") -> str:
    guidance = _SECTION_POINTS_GUIDANCE.get(level, _SECTION_POINTS_GUIDANCE["full"])
    base = _SECTION_FILL.format(
        node_types=NODE_TYPES, grounding=_GROUNDING, anchors=_ANCHORS, points_guidance=guidance
    )
    return _with_purpose(base, purpose)


# -- Anchor repair ----------------------------------------------------------

# The compression stages (MAP/REDUCE) sometimes fold away the concrete hooks a
# learner remembers by. Deterministic detection finds anchors present in the
# SOURCE but absent from the finished map; this prompt re-attaches the genuine
# ones. The candidate list may include noise (over-detected fragments,
# already-covered items) -- the model is the precision filter, so the prompt
# leans hard on "only genuine, currently-absent anchors; ignore the rest."
_ANCHOR_REPAIR = """You are refining a mind map built from a source document or transcript. TASK: RE-ATTACH DROPPED ANCHORS.

You receive:
- NODES: a numbered list of the map's current nodes (id, title, optional note).
- CANDIDATES: concrete details (numbers, named techniques, named people/books,
  enumerated examples) detected in the SOURCE that may have been dropped from
  the map. Some candidates are noise, already covered, or not real anchors.

For each candidate that is a GENUINE concrete anchor NOT already represented in
the map, return where it should attach and a clean title for it. Attach it to
the single most relevant existing node (by id) as a new child. IGNORE any
candidate that is vague, generic, already present in some node's title or note,
or not actually a concrete anchor.

GROUNDING: never invent detail. A repaired node's title is just the anchor
itself, cleaned up (e.g. "early stopping all" → "Early Stopping"; a number like
"13,000" stays exactly "13,000"). Do not add any fact beyond the candidate.

Write titles in the same language as the node titles. Keep JSON keys in English.

Return ONLY JSON:
{{
  "repairs": [
    {{"to": 3, "title": "Early Stopping", "type": "one of: {node_types}"}}
  ]
}}
Return an empty "repairs" list if no candidate is a genuine missing anchor."""

ANCHOR_REPAIR_SYSTEM = _ANCHOR_REPAIR.format(node_types=NODE_TYPES)


# -- Synthesis (structured sources) -----------------------------------------

# Structured sources (a PDF's TOC, an author-numbered video) keep their own
# spine verbatim -- faithful, but a bare TOC + per-section bullets has no
# cross-cutting layer, so the source's actual thesis can sit buried as a leaf
# under some late section. This ADDITIVE pass appends one "Key Takeaways"
# branch that synthesizes ACROSS sections; it never edits the author's spine.
_SYNTHESIS = """You are an expert knowledge cartographer. TASK: SYNTHESIZE.
You receive a mind map built faithfully from a source's own structure (its
chapters/sections, or its author-numbered list). That structure is FIXED and
you must NOT repeat or restate it.

Produce a short "Key Takeaways" section: the big-picture conclusions and
cross-cutting connections a reader should walk away with — the single most
important thesis of the whole source, plus the points that span MULTIPLE
sections (a cause running across chapters, a recurring principle, the payoff
the sections build toward).

{grounding}

Do NOT summarize section by section. Synthesize ACROSS sections. Aim for 3-5
takeaways. If the map is genuinely too thin to synthesize anything
cross-cutting, return an empty list rather than padding it.

Write in the same language as the node titles. Keep JSON keys in English.

Return ONLY JSON:
{{
  "takeaways": [
    {{"title": "a crisp takeaway, max 12 words", "note": "one sentence on why it matters across the whole source", "type": "one of: {node_types}"}}
  ]
}}"""

SYNTHESIS_SYSTEM = _SYNTHESIS.format(node_types=NODE_TYPES, grounding=_GROUNDING)


def synthesis_system(purpose: str = "general") -> str:
    return _with_purpose(SYNTHESIS_SYSTEM, purpose)
