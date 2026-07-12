"""IR -> OPML 2.0.

OPML carries the full hierarchy + notes and imports natively into XMind,
MindNode, Freemind, Workflowy, and most outliners. It cannot represent
cross-link relationships or markers — those are reserved for the XMind writer.

We build the tree with ElementTree so escaping of ``&``, ``<``, quotes, etc.
is handled correctly and the output is always well-formed XML.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from ..ir import MindMap, Node
from ..fsutil import atomic_write
from .util import note_for


def _build_outline(parent_el: ET.Element, node: Node) -> None:
    attrs = {"text": node.title}
    note = note_for(node)
    if note:
        attrs["_note"] = note
    # Preserve the semantic type so a future re-import / round-trip keeps it.
    if node.type.value not in ("topic", "root"):
        attrs["_cerebroType"] = node.type.value
    el = ET.SubElement(parent_el, "outline", attrs)
    for child in node.children:
        _build_outline(el, child)


def mindmap_to_opml(mm: MindMap) -> str:
    """Render a MindMap to a pretty-printed OPML 2.0 string."""
    opml = ET.Element("opml", version="2.0")

    head = ET.SubElement(opml, "head")
    ET.SubElement(head, "title").text = mm.title
    ET.SubElement(head, "expansionState").text = "0"
    if mm.source:
        ET.SubElement(head, "ownerName").text = mm.generator

    body = ET.SubElement(opml, "body")
    # The single top-level outline becomes XMind's central topic.
    _build_outline(body, mm.root)

    rough = ET.tostring(opml, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8")
    return pretty.decode("utf-8")


def write_opml(mm: MindMap, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda tmp: tmp.write_text(mindmap_to_opml(mm), encoding="utf-8"))
    return path
