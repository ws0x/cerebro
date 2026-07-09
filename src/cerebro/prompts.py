"""Prompt templates for the map → reduce → link pipeline.

Templates use ``str.format`` placeholders (``{node_types}``, ``{depth}``) with
literal JSON braces doubled (``{{`` / ``}}``). Bump ``PROMPT_VERSION`` whenever a
prompt changes so the cache invalidates cleanly (it is part of every cache key).
"""

PROMPT_VERSION = "v1"

NODE_TYPES = "topic, concept, definition, example, insight, action, warning, question, detail"

_MAP = """You are an expert knowledge cartographer. TASK: MAP.
You receive one segment of a video transcript. Extract its core teaching as a
compact JSON object. Do not summarize the whole video — only this segment.

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

LINK_SYSTEM = """You are an expert knowledge cartographer. TASK: LINK.
You receive a numbered list of nodes from a finished mind map. Identify the most
important NON-hierarchical relationships between them (dependency, cause,
contrast, example-of, prerequisite). Only link nodes in different branches.

Return ONLY JSON:
{
  "relationships": [
    {"from": 0, "to": 1, "label": "short verb phrase"}
  ]
}
Return at most 8 of the strongest relationships. Use the integer ids shown."""

# Per-level guidance injected into the REDUCE prompt.
LEVEL_DEPTH = {
    "brief": "a shallow map: 4-6 main branches, minimal nesting",
    "full": "a map 3-4 levels deep with subtopics and key points",
    "expert": "a rich map 4+ levels deep with concepts, examples, and actionable insights",
}

MAP_SYSTEM = _MAP.format(node_types=NODE_TYPES)


def reduce_system(level: str) -> str:
    depth = LEVEL_DEPTH.get(level, LEVEL_DEPTH["full"])
    return _REDUCE.format(node_types=NODE_TYPES, depth=depth)
