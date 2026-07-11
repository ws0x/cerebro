"""Combine two or more already-built maps (OPML/XMind) into one, without
re-running any ingestion or LLM calls.

Reads each file back into the IR via a small reverse converter -- the
mirror image of ``convert/opml.py``/``convert/xmind.py`` -- then combines
the results the same way ``batch.py`` combines multiple freshly-built
per-source maps: each input file becomes its own top-level branch under a
new shared root. No cross-file relationship synthesis in this pass (that's
a genuinely different, much riskier feature -- see the "cross-source
concept merging" work); relationships are preserved per-file exactly as
each source carried them.

XMind round-trips faithfully for title/note/children/relationships, and for
the 7 node types that have a distinct marker (concept/definition/example/
insight/action/warning/question) -- root/topic/detail share no visual
marker in XMind to begin with, so those three collapse to a plain "topic"
on the way back in, same as they'd render with no icon either way. OPML
round-trips title/note/children/type (via the ``_cerebroType`` attribute
cerebro's own OPML writer already emits) but never relationships -- OPML
can't carry them on the way out either, so there's nothing to read back.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from .convert.xmind import _MARKER
from .ir import MindMap, Node, NodeType, Relationship

_MARKER_TO_TYPE = {marker: node_type for node_type, marker in _MARKER.items()}


class MergeError(Exception):
    pass


def _node_from_opml(el: ET.Element, is_root: bool = False) -> Node:
    node_type = NodeType.root if is_root else NodeType.topic
    type_attr = el.get("_cerebroType")
    if type_attr:
        try:
            node_type = NodeType(type_attr)
        except ValueError:
            pass  # an unrecognized/hand-edited type attribute -- fall back to topic, don't fail the merge
    node = Node(title=el.get("text", ""), type=node_type, note=el.get("_note") or None)
    for child in el.findall("outline"):
        node.children.append(_node_from_opml(child))
    return node


def read_opml(path: Path) -> MindMap:
    try:
        tree = ET.parse(path)
    except Exception as exc:
        raise MergeError(f"Not a valid OPML file: {path} ({exc})") from exc
    root_el = tree.getroot()
    body = root_el.find("body")
    if body is None:
        raise MergeError(f"OPML file has no <body>: {path}")
    outline = body.find("outline")
    if outline is None:
        raise MergeError(f"OPML file has no content: {path}")
    root_node = _node_from_opml(outline, is_root=True)
    title_el = root_el.find("head/title")
    title = (title_el.text if title_el is not None and title_el.text else None) or root_node.title
    return MindMap(title=title, root=root_node, source=str(path))


def _node_from_xmind_topic(topic: dict, is_root: bool = False) -> Node:
    node_type = NodeType.root if is_root else NodeType.topic
    for marker in topic.get("markers") or []:
        mapped = _MARKER_TO_TYPE.get(marker.get("markerId"))
        if mapped:
            node_type = mapped
            break
    note = topic.get("notes", {}).get("plain", {}).get("content") or None
    kwargs = {"title": str(topic.get("title", "")), "type": node_type, "note": note}
    topic_id = topic.get("id")
    if topic_id:  # preserve the original id so relationships below still resolve correctly
        kwargs["id"] = str(topic_id)
    node = Node(**kwargs)
    for child in topic.get("children", {}).get("attached", []):
        node.children.append(_node_from_xmind_topic(child))
    return node


def read_xmind(path: Path) -> MindMap:
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("content.json"))
    except Exception as exc:
        raise MergeError(f"Not a valid XMind file: {path} ({exc})") from exc
    if not data:
        raise MergeError(f"XMind file has no sheets: {path}")
    sheet = data[0]
    root_topic = sheet.get("rootTopic")
    if not root_topic:
        raise MergeError(f"XMind file has no root topic: {path}")
    root_node = _node_from_xmind_topic(root_topic, is_root=True)
    relationships = [
        Relationship(from_id=r["end1Id"], to_id=r["end2Id"], label=r.get("title", ""))
        for r in sheet.get("relationships", [])
        if "end1Id" in r and "end2Id" in r
    ]
    title = sheet.get("title") or root_node.title
    # Prefer the root topic's own href (the original video/PDF/article the map
    # was built from) over the .xmind file's own path -- losing that on a
    # read-modify-write round trip (cerebro edit, or merging this file into
    # another) would silently drop the hyperlink cerebro itself just added.
    source = root_topic.get("href") or str(path)
    return MindMap(title=title, root=root_node, relationships=relationships, source=source)


def read_map(path: Path) -> MindMap:
    suffix = path.suffix.lower()
    if suffix == ".opml":
        return read_opml(path)
    if suffix == ".xmind":
        return read_xmind(path)
    raise MergeError(f"Unsupported file type for merge: {path} (expected .opml or .xmind)")


def merge_maps(paths: list[Path], title: str = "Merged Map") -> MindMap:
    if len(paths) < 2:
        raise MergeError("Need at least 2 maps to merge.")
    root = Node(title=title, type=NodeType.root)
    relationships: list[Relationship] = []
    for path in paths:
        mm = read_map(Path(path))
        branch = mm.root
        branch.type = NodeType.topic
        # The file's own map title as the branch heading -- distinct from
        # the filename, which is often just a generic saved name.
        branch.title = mm.title or Path(path).stem
        root.children.append(branch)
        relationships.extend(mm.relationships)
    return MindMap(title=title, root=root, relationships=relationships, level="merged")
