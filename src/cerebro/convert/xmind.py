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

Every sheet also carries a fixed visual theme (below), copied verbatim from
a real XMind Zen "Dawn" multi-line-color theme -- a hand-picked reference
map (examples/xmind_theme_template/DEFAULT_MAP_TEMPLATE.xmind) rather than
XMind's own bland application default. It's pure styling data with no
runtime dependency on that file: nothing here reads it, the values are
just copied in, so nothing breaks if that reference file is ever moved or
deleted. See memory/project_cerebro_xmind_theme.md for where this came from.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

from .. import __version__
from ..ir import MindMap, Node, NodeType
from .util import atomic_write, note_for

# NodeType -> XMind built-in marker id (icons shipped with XMind).
_MARKER = {
    NodeType.concept: "star-blue",
    NodeType.definition: "symbol-info",
    NodeType.example: "symbol-plus",
    NodeType.insight: "star-yellow",
    NodeType.action: "symbol-right",
    NodeType.warning: "symbol-exclam",
    NodeType.question: "symbol-question",
    NodeType.detail: "star-orange",
}

# A clockwise radial layout, matching the reference theme -- XMind's own
# auto-layout algorithm for a map with no explicit topic positions (this
# writer never sets any), as opposed to the plain left-to-right "unbalanced"
# tree layout XMind defaults to.
_STRUCTURE_CLASS = "org.xmind.ui.map.clockwise"

_EXTENSIONS = [
    {
        "provider": "org.xmind.ui.skeleton.structure.style",
        "content": {"centralTopic": _STRUCTURE_CLASS},
    }
]

# Copied verbatim from the reference map's own sheet["theme"] -- every
# property here is real XMind Zen theme data (fonts, colors, shapes, line
# styles per topic level), not reverse-engineered. "Dawn" is the theme's own
# name; multi-line-colors cycles a top-level branch's whole subtree through
# the six-color palette below, which is what actually gives each branch its
# own color in XMind -- no per-node color assignment needed in _topic().
_THEME = {
    "map": {
        "id": "f8beccbc-8814-46c1-8523-093074d261ce",
        "properties": {
            "svg:fill": "#ffffff",
            "multi-line-colors": "#FF6B6B #FF9F69 #97D3B6 #88E2D7 #6FD0F9 #E18BEE",
            "color-list": "#FF6B6B #FF9F69 #97D3B6 #88E2D7 #6FD0F9 #E18BEE",
            "line-tapered": "none",
        },
    },
    "centralTopic": {
        "id": "9deff695-46e5-44fa-a6f2-33935d0c20df",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "30pt",
            "fo:font-weight": "800",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#000000",
            "fill-pattern": "none",
            "line-width": "2pt",
            "line-color": "#ADADAD",
            "line-pattern": "solid",
            "border-line-color": "#000000",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.curve",
            "arrow-end-class": "org.xmind.arrowShape.none",
            "alignment-by-level": "inherited",
        },
    },
    "mainTopic": {
        "id": "aa3b3a77-54d3-4ee2-9ccf-4ccbc3b76d8b",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "18pt",
            "fo:font-weight": "500",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "inherited",
            "fill-pattern": "solid",
            "line-width": "2pt",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
            "alignment-by-level": "inherited",
        },
    },
    "subTopic": {
        "id": "f08a6c3b-0c1c-4e8f-a2ca-85e8ebbf0f19",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "inherited",
            "fill-pattern": "none",
            "line-width": "2pt",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
            "alignment-by-level": "inherited",
        },
    },
    "floatingTopic": {
        "id": "7099ff79-5242-4776-95cf-c772447b314e",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "500",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "#EEEBEE",
            "fill-pattern": "solid",
            "line-width": "2pt",
            "line-color": "inherited",
            "line-pattern": "solid",
            "border-line-color": "#EEEBEE",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "org.xmind.arrowShape.none",
            "alignment-by-level": "inherited",
        },
    },
    "summaryTopic": {
        "id": "3202b02e-ab05-42ea-ae7b-48277fadb23f",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "#000000",
            "fill-pattern": "none",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "#000000",
            "border-line-width": "inherited",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
            "alignment-by-level": "inherited",
        },
    },
    "calloutTopic": {
        "id": "575725ac-a562-4f4b-8ba3-0f25f8d96b13",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "#000000",
            "fill-pattern": "solid",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "#000000",
            "border-line-width": "inherited",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
            "alignment-by-level": "inherited",
        },
    },
    "importantTopic": {
        "id": "ccc1db83-f4c5-401b-ae48-12018ab0188b",
        "properties": {
            "fo:font-weight": "bold",
            "svg:fill": "#7F00AC",
            "fill-pattern": "solid",
            "border-line-color": "#7F00AC",
            "border-line-width": "0",
        },
    },
    "minorTopic": {
        "id": "2fd57c12-5f34-4ea7-bcf6-29c2ad0eb340",
        "properties": {
            "fo:font-weight": "bold",
            "svg:fill": "#82004A",
            "fill-pattern": "solid",
            "border-line-color": "#82004A",
            "border-line-width": "0",
        },
    },
    "expiredTopic": {
        "id": "2410370c-bdde-4bfd-8eda-664233ecd24f",
        "properties": {"fo:text-decoration": "line-through", "fill-pattern": "none"},
    },
    "boundary": {
        "id": "f24dd51f-8939-49bc-83b5-2ed196d7ae39",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#9B9B9B",
            "fill-pattern": "solid",
            "line-width": "2",
            "line-color": "#00000066",
            "line-pattern": "dash",
            "shape-class": "org.xmind.boundaryShape.roundedRect",
        },
    },
    "zone": {
        "id": "9b6cf15e-6b11-4f50-9f1d-f1541e75440b",
        "properties": {
            "fo:font-family": "NeverMind, sans-serif, Microsoft YaHei, PingFang SC, Microsoft JhengHei, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol, Noto Color Emoji",
            "fo:font-size": "12",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "#9b9b9b33",
            "fill-pattern": "none",
            "border-line-color": "#00000066",
            "border-line-width": "2pt",
            "border-line-pattern": "solid",
        },
    },
    "summary": {
        "id": "625098ca-1c35-40af-96bb-e3222c31bcaa",
        "properties": {
            "line-width": "2pt",
            "line-color": "#000000",
            "line-pattern": "solid",
            "shape-class": "org.xmind.summaryShape.round",
        },
    },
    "relationship": {
        "id": "c3facac7-a0fe-4b5c-9d21-bc5d2e7fb6b0",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "13pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "line-width": "2",
            "line-color": "#00000066",
            "line-pattern": "dash",
            "shape-class": "org.xmind.relationshipShape.curved",
            "arrow-begin-class": "org.xmind.arrowShape.none",
            "arrow-end-class": "org.xmind.arrowShape.triangle",
        },
    },
    "level3": {
        "id": "c3d8ccf7-116c-428c-8ac5-d20ca165aa25",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "fill-pattern": "solid",
            "line-width": "2pt",
            "line-pattern": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
            "alignment-by-level": "inherited",
        },
    },
    "skeletonThemeId": "8f137afe-a0f3-4f25-b56a-cfd05993ef4b",
    "colorThemeId": "Dawn-#ffffff-MULTI_LINE_COLORS",
}

# -- Folder-structure (cerebro tree) theme ----------------------------------
# A folder tree is a different kind of map from a video/PDF/article content
# map -- there's no narrative to follow, no depth-of-understanding levels,
# just a hierarchy to scan -- so it gets its own visual identity rather than
# reusing the content theme by default. Copied verbatim from a second
# hand-picked reference map (examples/xmind_theme_template/TREE_MAP_TEMPLATE.xmind),
# a real XMind Zen "Hawaii" theme: a right-hand logic-chart layout (reads
# top-to-bottom/left-to-right like an org chart, not radially outward) suited
# to a directory tree's own shape, cooler blue/teal/amber palette, distinct
# from the content theme's warm "Dawn" palette so the two are visually
# distinguishable from across a room. Selected automatically by
# mindmap_to_xmind_content() when mm.level == "structure" (what
# foldermap.py's build_folder_map() sets) -- no caller has to ask for it.
_TREE_STRUCTURE_CLASS = "org.xmind.ui.logic.right"

_TREE_EXTENSIONS = [
    {
        "provider": "org.xmind.ui.skeleton.structure.style",
        "content": {"centralTopic": _TREE_STRUCTURE_CLASS},
    }
]

_TREE_THEME = {
    "map": {
        "id": "02053fa0-1f4b-4859-abb1-e2358d5020f9",
        "properties": {
            "svg:fill": "#FFFFFF",
            "multi-line-colors": "#B7D6E8 #4A94C3 #4B9383 #D29F55",
            "color-list": "#B7D6E8 #4A94C3 #254B85 #4B9383 #D29F55 #F3E6CF",
            "line-tapered": "none",
        },
    },
    "centralTopic": {
        "id": "7d2e2606-7335-4fd6-a701-f4a1e0ccac42",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "28pt",
            "fo:font-weight": "600",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#254B85",
            "fill-pattern": "solid",
            "line-width": "2pt",
            "line-color": "#254B85",
            "line-pattern": "solid",
            "border-line-color": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedfold",
            "arrow-end-class": "org.xmind.arrowShape.none",
            "alignment-by-level": "actived",
        },
    },
    "mainTopic": {
        "id": "97823c85-446e-46cb-9ffd-af88ef26fffb",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "18pt",
            "fo:font-weight": "600",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "inherited",
            "fill-pattern": "solid",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
        },
    },
    "subTopic": {
        "id": "fb8cd5ed-f375-490f-999e-1cd075ffebb2",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "inherited",
            "fill-pattern": "solid",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "inherited",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
        },
    },
    "floatingTopic": {
        "id": "8e2d42b7-6d0f-4d15-a356-158c5ea49b00",
        "properties": {
            "fo:font-family": "Montserrat",
            "fo:font-size": "14pt",
            "fo:font-weight": "normal",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#4A94C3",
            "fill-pattern": "solid",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "solid",
            "border-line-color": "#4A94C3",
            "border-line-width": "0pt",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "org.xmind.arrowShape.none",
        },
    },
    "summaryTopic": {
        "id": "45c724cf-5f9d-457d-b55b-9f32c101772f",
        "properties": {
            "fo:font-family": "Montserrat",
            "fo:font-size": "14pt",
            "fo:font-weight": "400",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#4B9383",
            "fill-pattern": "none",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "#4B9383",
            "border-line-width": "inherited",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.roundedRect",
            "line-class": "org.xmind.branchConnection.roundedElbow",
            "arrow-end-class": "inherited",
        },
    },
    "calloutTopic": {
        "id": "4e165164-dcb1-4456-b831-5e155a955b72",
        "properties": {
            "fo:font-family": "NeverMind",
            "fo:font-size": "14pt",
            "fo:font-weight": "600",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "left",
            "svg:fill": "#4B9383",
            "fill-pattern": "solid",
            "line-width": "inherited",
            "line-color": "inherited",
            "line-pattern": "inherited",
            "border-line-color": "#4B9383",
            "border-line-width": "inherited",
            "border-line-pattern": "inherited",
            "shape-class": "org.xmind.topicShape.ellipse",
            "arrow-end-class": "inherited",
        },
    },
    "importantTopic": {
        "id": "2095e36b-73bd-43a9-91ea-58b16eb1e233",
        "properties": {"svg:fill": "#794ac3", "fill-pattern": "solid", "border-line-color": "#794ac3"},
    },
    "minorTopic": {
        "id": "0b83e1aa-69df-4a1b-b229-eccb3702fd2a",
        "properties": {"svg:fill": "#c34ab2", "fill-pattern": "solid", "border-line-color": "#c34ab2"},
    },
    "expiredTopic": {
        "id": "692e289f-1fb4-409e-90ad-d7feef2c5f17",
        "properties": {"fo:text-decoration": "line-through", "fill-pattern": "none"},
    },
    "boundary": {
        "id": "b8673950-f501-44a5-bb1d-ca8de2434905",
        "properties": {
            "fo:font-family": "'NeverMind','Microsoft YaHei','PingFang SC','Microsoft JhengHei','sans-serif',sans-serif",
            "fo:font-size": "14pt",
            "fo:font-weight": "600",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "svg:fill": "#254B85",
            "fill-pattern": "solid",
            "line-width": "2",
            "line-color": "#254B85",
            "line-pattern": "dash",
            "shape-class": "org.xmind.boundaryShape.roundedRect",
        },
    },
    "summary": {
        "id": "b2f2ff39-56e8-45eb-a204-737f02023097",
        "properties": {
            "line-width": "2pt",
            "line-color": "#254B85",
            "line-pattern": "solid",
            "shape-class": "org.xmind.summaryShape.square",
        },
    },
    "relationship": {
        "id": "9a325766-4c63-4306-9509-ecd066671173",
        "properties": {
            "fo:font-family": "'NeverMind','Microsoft YaHei','PingFang SC','Microsoft JhengHei','sans-serif',sans-serif",
            "fo:font-size": "13pt",
            "fo:font-weight": "600",
            "fo:font-style": "normal",
            "fo:color": "inherited",
            "fo:text-transform": "manual",
            "fo:text-decoration": "none",
            "fo:text-align": "center",
            "line-width": "2",
            "line-color": "#254B85",
            "line-pattern": "dash",
            "shape-class": "org.xmind.relationshipShape.curved",
            "arrow-begin-class": "org.xmind.arrowShape.none",
            "arrow-end-class": "org.xmind.arrowShape.triangle",
        },
    },
    "skeletonThemeId": "a148ee55687bdfc44af2fa5f16",
    "colorThemeId": "Hawaii-#FFFFFF-MULTI_LINE_COLORS",
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


def _source_href(source: str | None) -> str | None:
    """A clickable XMind hyperlink (the topic-level ``href`` field, standard
    since XMind 8/Zen) for the map's own source -- a real web URL as-is, or a
    local file/folder turned into a ``file://`` URI so the map can jump
    straight back to what it was built from. ``None`` (no href at all) for
    anything that isn't a genuine, resolvable link -- e.g. a batch/merge
    combined map has no single source to point at."""
    if not source:
        return None
    if source.startswith(("http://", "https://", "file://")):
        # Already a real URI -- e.g. re-writing a MindMap read back from an
        # existing .xmind, whose source is the href merge.py already recovered.
        return source
    try:
        path = Path(source)
        if path.exists():
            return path.resolve().as_uri()
    except (OSError, ValueError):
        pass
    return None


def mindmap_to_xmind_content(mm: MindMap) -> list:
    """Build the ``content.json`` structure (a list of sheets)."""
    # A folder-structure map (cerebro tree) gets its own theme/layout rather
    # than the video/document content theme -- see _TREE_THEME's own comment
    # for why. mm.level == "structure" is what build_folder_map() sets;
    # nothing else in cerebro produces that value, so this is an unambiguous,
    # zero-configuration signal -- no caller has to ask for the tree theme.
    is_tree = mm.level == "structure"
    structure_class = _TREE_STRUCTURE_CLASS if is_tree else _STRUCTURE_CLASS
    extensions = _TREE_EXTENSIONS if is_tree else _EXTENSIONS
    theme = _TREE_THEME if is_tree else _THEME

    root_topic = _topic(mm.root)
    root_topic["structureClass"] = structure_class
    href = _source_href(mm.source)
    if href:
        root_topic["href"] = href

    sheet: dict = {
        "id": uuid.uuid4().hex,
        "class": "sheet",
        "title": mm.title or "Sheet 1",
        "rootTopic": root_topic,
        "extensions": extensions,
        "theme": theme,
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

    def _write(tmp: Path) -> None:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("content.json", content)
            z.writestr("metadata.json", metadata)
            z.writestr("manifest.json", manifest)

    atomic_write(path, _write)
    return path
