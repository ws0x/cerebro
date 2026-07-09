"""The Intermediate Representation (IR).

This is the single contract every output format is built from. The structurer
(heuristic or LLM) produces a ``MindMap``; deterministic converters turn it into
OPML, XMind, Markdown, etc. Nothing downstream ever parses a transcript, and no
model ever writes a file format directly.
"""

from __future__ import annotations

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Semantic role of a node. Drives icons/markers in rich formats (XMind)
    and is otherwise carried along as metadata."""

    root = "root"
    topic = "topic"
    concept = "concept"
    definition = "definition"
    example = "example"
    insight = "insight"
    action = "action"
    warning = "warning"
    question = "question"
    detail = "detail"


class Relationship(BaseModel):
    """A non-hierarchical cross-link between two nodes (XMind's floating arrows).

    Not representable in OPML — carried in the IR so richer converters can use it.
    """

    from_id: str
    to_id: str
    label: str = ""


class Node(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    title: str
    type: NodeType = NodeType.topic
    note: str | None = None
    timestamp: float | None = None  # seconds into the source, if known
    children: list["Node"] = Field(default_factory=list)

    def add(self, title: str, **kwargs) -> "Node":
        """Append a child and return it, for fluent tree building."""
        child = Node(title=title, **kwargs)
        self.children.append(child)
        return child

    def walk(self):
        """Depth-first traversal yielding this node then all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def count(self) -> int:
        return sum(1 for _ in self.walk())

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)


class MindMap(BaseModel):
    title: str
    root: Node
    relationships: list[Relationship] = Field(default_factory=list)
    source: str | None = None  # url or path the map was built from
    level: str = "full"
    generator: str = "cerebro"

    def node_count(self) -> int:
        return self.root.count()

    def depth(self) -> int:
        return self.root.depth()


Node.model_rebuild()
