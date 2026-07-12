"""IR -> Markdown, for Obsidian/Notion/plain-outliner users -- a much larger
audience than XMind's, and one OPML/XMind don't reach at all.

Deliberately plain: nested bullets, a note as an indented italic line under
each bullet, no bespoke wikilink scheme. A single map is one document, not a
vault -- ``[[Title]]`` links only pay off with real backlinks across
*multiple* files/headings, and wrapping every bullet in one wouldn't create
that, just noise. Relationships (expert level) get an honest trailing
section instead of a synthetic linking convention: ``**A** → **B** (label)``,
readable on its own and still simple enough to reformat by hand if a reader
does want to wire it into their own vault's linking style.
"""

from __future__ import annotations

from pathlib import Path

from ..ir import MindMap, Node
from .util import atomic_write, note_for

_INDENT = "  "


def _bullets(node: Node, depth: int) -> list[str]:
    lines = [f"{_INDENT * depth}- {node.title}"]
    note = note_for(node)
    if note:
        note_oneline = " ".join(note.split())  # collapse embedded newlines -- a note is one bullet, not a nested block
        lines.append(f"{_INDENT * (depth + 1)}*{note_oneline}*")
    for child in node.children:
        lines.extend(_bullets(child, depth + 1))
    return lines


def mindmap_to_markdown(mm: MindMap) -> str:
    """Render a MindMap to a Markdown string: an H1 title, nested bullets for
    the hierarchy, and (if any) a trailing Relationships section."""
    lines = [f"# {mm.title}", ""]

    for child in mm.root.children:
        lines.extend(_bullets(child, 0))

    if mm.relationships:
        id_to_title = {n.id: n.title for n in mm.root.walk()}
        lines.append("")
        lines.append("## Relationships")
        lines.append("")
        for rel in mm.relationships:
            a = id_to_title.get(rel.from_id, rel.from_id)
            b = id_to_title.get(rel.to_id, rel.to_id)
            suffix = f" ({rel.label})" if rel.label else ""
            lines.append(f"- **{a}** → **{b}**{suffix}")

    return "\n".join(lines) + "\n"


def write_markdown(mm: MindMap, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda tmp: tmp.write_text(mindmap_to_markdown(mm), encoding="utf-8"))
    return path
