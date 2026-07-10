"""Prompt templates for the map → reduce → link pipeline.

Templates use ``str.format`` placeholders (``{node_types}``, ``{depth}``) with
literal JSON braces doubled (``{{`` / ``}}``). Bump ``PROMPT_VERSION`` whenever a
prompt changes so the cache invalidates cleanly (it is part of every cache key).
"""

PROMPT_VERSION = "v3"

NODE_TYPES = "topic, concept, definition, example, insight, action, warning, question, detail"

_GROUNDING = """GROUNDING RULE (critical): only use what is explicitly stated in the
input. Do not introduce outside facts, named techniques, examples, or specifics
that are implied by the topic but not actually said. If the input refers to
something generically (e.g. "an activation function"), keep your title equally
generic (e.g. "Activation Function") — do NOT name a specific instance (e.g.
"Sigmoid" or "ReLU") unless that specific name is literally present in the text."""

_MAP = """You are an expert knowledge cartographer. TASK: MAP.
You receive one segment of a video transcript. Extract its core teaching as a
compact JSON object. Do not summarize the whole video — only this segment.

{grounding}

Write all titles, summaries, and points in the same language as the input transcript. Keep JSON keys in English.

Return ONLY JSON with this shape:
{{
  "topic": "a short noun-phrase title, max 8 words",
  "type": "one of: {node_types}",
  "summary": "1-2 sentence plain summary of the segment",
  "points": [
    {{"title": "a concrete sub-point, max 12 words", "type": "one of the types above"}}
  ]
}}
Keep points to the 2-5 most important. Titles must be self-contained (no 'this' or 'it')."""

_REDUCE = """You are an expert knowledge cartographer. TASK: REDUCE.
You receive an ordered list of per-segment extractions from one video. Build a
single, smart, hierarchical mind map: merge duplicate topics, promote recurring
themes to parent nodes, demote details to leaves, and order branches logically
(not merely chronologically).

{grounding}
This applies to merging too: you may rename a branch to a more general umbrella
term that covers its children, but never invent a more specific term than what
the segments actually contain.

Write all titles, notes, and central subject in the same language as the input segments. Keep JSON keys in English.

Return ONLY JSON with this shape (children may nest to any depth):
{{
  "central": "the overall subject, a short noun phrase",
  "children": [
    {{
      "title": "branch title",
      "type": "one of: {node_types}",
      "note": "optional 1-sentence elaboration",
      "children": [ ]
    }}
  ]
}}
Aim for {depth}. Every title must be self-contained and concise."""

_LINK_TEMPLATE = """You are an expert knowledge cartographer. TASK: LINK.
You receive a numbered list of nodes from a finished mind map. Identify the most
important NON-hierarchical relationships between them (dependency, cause,
contrast, example-of, prerequisite). Only link nodes in different branches.
Only propose a relationship that is actually supported by the node titles given
— do not invent a connection that isn't reasonably implied by them.

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


def link_system(limit: int) -> str:
    return _LINK_TEMPLATE.format(limit=limit)


def cross_link_system(limit: int) -> str:
    return _CROSS_LINK_TEMPLATE.format(limit=limit)


# For backward compatibility
LINK_SYSTEM = link_system(8)

# Per-level guidance injected into the REDUCE prompt.
LEVEL_DEPTH = {
    "brief": "a shallow map: 4-6 main branches, minimal nesting",
    "full": "a map 3-4 levels deep with subtopics and key points",
    "expert": "a rich map 4+ levels deep with concepts, examples, and actionable insights",
}

MAP_SYSTEM = _MAP.format(node_types=NODE_TYPES, grounding=_GROUNDING)


def reduce_system(level: str) -> str:
    depth = LEVEL_DEPTH.get(level, LEVEL_DEPTH["full"])
    return _REDUCE.format(node_types=NODE_TYPES, depth=depth, grounding=_GROUNDING)
