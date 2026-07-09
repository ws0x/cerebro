"""Shared converter helpers."""

from __future__ import annotations

from ..ir import Node


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def note_for(node: Node) -> str:
    """Combine an optional timestamp marker with the node's note into one string."""
    parts: list[str] = []
    if node.timestamp is not None:
        parts.append(f"[{format_timestamp(node.timestamp)}]")
    if node.note:
        parts.append(node.note.strip())
    return " ".join(parts).strip()
