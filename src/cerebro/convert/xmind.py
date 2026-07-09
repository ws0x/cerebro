"""IR -> native .xmind (modern XMind / Zen JSON format).

A .xmind file is a ZIP archive containing:
  * content.json  — an array of sheets; each has a rootTopic tree + relationships
  * metadata.json — creator info
  * manifest.json — file listing

Unlike OPML, this format carries cross-link **relationships** and per-node
**markers** (icons), which is the whole reason it exists in Cerebro: it makes
Expert-mode maps render with their full visual structure in XMind.

The conversion is pure data-shaping from the IR — no model involvement — so the
output is valid every time.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

from .. import __version__
from ..ir import MindMap, Node, NodeType
from .util import note_for

# NodeType -> XMind built-in marker id (icons shipped with XMind).
_MARKER = {
    NodeType.concept: "star-blue",
    NodeType.definition: "symbol-info",
    NodeType.example: "symbol-plus",
    NodeType.insight: "star-yellow",
    NodeType.action: "symbol-right",
    NodeType.warning: "symbol-exclam",
    NodeType.question: "symbol-question",
}


def _topic(node: Node) -> dict:
    topic: dict = {"id": node.id, "class": "topic", "title": node.title}

    note = note_for(node)
    if note:
        topic["notes"] = {"plain": {"content": note}}

    marker = _MARKER.get(node.type)
    if marker:
        topic["markers"] = [{"markerId": marker}]

    if node.children:
        topic["children"] = {"attached": [_topic(c) for c in node.children]}

    return topic


def mindmap_to_xmind_content(mm: MindMap) -> list:
    """Build the ``content.json`` structure (a list of sheets)."""
    root_topic = _topic(mm.root)
    root_topic["structureClass"] = "org.xmind.ui.map.unbalanced"

    sheet: dict = {
        "id": uuid.uuid4().hex,
        "class": "sheet",
        "title": mm.title or "Sheet 1",
        "rootTopic": root_topic,
    }

    if mm.relationships:
        sheet["relationships"] = [
            {
                "id": uuid.uuid4().hex,
                "class": "relationship",
                "end1Id": rel.from_id,
                "end2Id": rel.to_id,
                "title": rel.label,
            }
            for rel in mm.relationships
        ]

    return [sheet]


def write_xmind(mm: MindMap, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(mindmap_to_xmind_content(mm), ensure_ascii=False)
    metadata = json.dumps(
        {"creator": {"name": "cerebro", "version": __version__}}, ensure_ascii=False
    )
    manifest = json.dumps(
        {"file-entries": {"content.json": {}, "metadata.json": {}}}, ensure_ascii=False
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("content.json", content)
        z.writestr("metadata.json", metadata)
        z.writestr("manifest.json", manifest)

    return path
