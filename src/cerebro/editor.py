"""Pure tree-editing helpers for `cerebro edit` -- kept separate from the
interactive terminal loop in cli.py so the actual mutation logic (locate a
node by its position in the tree, rename it, delete it) is unit-testable
without a terminal.

A node's "path" is the sequence of child indices from the root down to it
-- e.g. ``(0, 2)`` is ``root.children[0].children[2]``. A flat tuple, not a
``Node`` reference, so it stays valid to resolve against the current tree
after an edit -- a direct ``Node`` reference into a tree that just had a
sibling deleted could reference a node whose position shifted or that no
longer belongs to a still-consistent tree; a path is always freshly
resolved from the root.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir import Node


@dataclass
class TreeEntry:
    path: tuple[int, ...]
    node: Node
    depth: int


def flatten(node: Node, path: tuple[int, ...] = (), depth: int = 0) -> list[TreeEntry]:
    """Every node in ``node``'s subtree, depth-first, as flat entries -- the
    shape a flat list-based menu (no native tree widget in a terminal) needs
    to represent a hierarchy at all."""
    entries = [TreeEntry(path=path, node=node, depth=depth)]
    for i, child in enumerate(node.children):
        entries.extend(flatten(child, path + (i,), depth + 1))
    return entries


def node_at(root: Node, path: tuple[int, ...]) -> Node:
    node = root
    for i in path:
        node = node.children[i]
    return node


def parent_and_index(root: Node, path: tuple[int, ...]) -> tuple[Node, int]:
    """Only valid for a non-empty path -- the root itself has no parent."""
    parent = root
    for i in path[:-1]:
        parent = parent.children[i]
    return parent, path[-1]


def rename(root: Node, path: tuple[int, ...], new_title: str) -> None:
    node_at(root, path).title = new_title


def delete(root: Node, path: tuple[int, ...]) -> None:
    if not path:
        raise ValueError("Cannot delete the root node.")
    parent, idx = parent_and_index(root, path)
    del parent.children[idx]
