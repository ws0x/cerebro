"""Search across every previously-built map (OPML/XMind) for a text query.

No central registry of "every map ever built" exists beyond wherever each
one happened to be written -- ``ensure_output_dir()``'s ``~/cerebro-maps/``
is the de-facto default every map lands in unless ``--out`` pointed
somewhere else, so that's what's scanned by default (the CLI layer exposes
``--dir`` to point this elsewhere). Plain substring matching against every
node's title and note -- no index, no cache -- since re-parsing a folder of
outline files on demand is already fast enough that building and
invalidating an index would be complexity this doesn't need yet.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NodeMatch:
    title: str
    note: str


@dataclass
class MapMatch:
    path: Path
    nodes: list[NodeMatch] = field(default_factory=list)


def iter_maps(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".opml", ".xmind"))


def _nodes_from_opml(path: Path) -> list[tuple[str, str]]:
    try:
        tree = ET.parse(path)
    except Exception:
        return []  # not well-formed XML, or not actually an OPML file -- skip, don't crash the whole search
    out: list[tuple[str, str]] = []

    def walk(el: ET.Element) -> None:
        for child in el.findall("outline"):
            out.append((child.get("text", ""), child.get("_note", "")))
            walk(child)

    body = tree.getroot().find("body")
    if body is not None:
        walk(body)
    return out


def _nodes_from_xmind(path: Path) -> list[tuple[str, str]]:
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("content.json"))
    except Exception:
        return []  # not a valid XMind zip, or missing content.json -- skip, don't crash the whole search
    out: list[tuple[str, str]] = []

    def walk(topic: dict) -> None:
        title = str(topic.get("title", ""))
        note = str(topic.get("notes", {}).get("plain", {}).get("content", ""))
        out.append((title, note))
        for child in topic.get("children", {}).get("attached", []):
            walk(child)

    for sheet in data if isinstance(data, list) else []:
        root_topic = sheet.get("rootTopic")
        if root_topic:
            walk(root_topic)
    return out


def _nodes_for(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".opml":
        return _nodes_from_opml(path)
    if suffix == ".xmind":
        return _nodes_from_xmind(path)
    return []


def search_maps(
    query: str, root: Path, case_sensitive: bool = False, max_matches_per_file: int = 10
) -> list[MapMatch]:
    """Every map under ``root`` with at least one node whose title or note
    contains ``query`` -- capped at ``max_matches_per_file`` matches per
    file so one huge map's results can't drown out everything else."""
    if not query:
        return []
    needle = query if case_sensitive else query.lower()
    results: list[MapMatch] = []
    for path in iter_maps(root):
        matches: list[NodeMatch] = []
        for title, note in _nodes_for(path):
            haystack = f"{title} {note}"
            haystack = haystack if case_sensitive else haystack.lower()
            if needle in haystack:
                matches.append(NodeMatch(title=title, note=note))
                if len(matches) >= max_matches_per_file:
                    break
        if matches:
            results.append(MapMatch(path=path, nodes=matches))
    return results
