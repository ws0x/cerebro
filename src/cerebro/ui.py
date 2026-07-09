"""Rich UI helpers — banner, step progress, and an in-terminal map preview.

Visual language deliberately echoes vidforge so the tools feel like a family.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from .ir import MindMap, Node, NodeType

console = Console()

_BANNER = r"""
  ___  ___  ___  ___  ___  ___  ___
 / __|| __|| _ \| __|| _ )| _ \/ _ \
| (__ | _| |   /| _| | _ \|   / (_) |
 \___||___||_|_\|___||___/|_|_\\___/
     video  ->  smart mind maps
"""

# Icon per semantic node type, for the terminal preview.
_TYPE_ICON = {
    NodeType.root: "🧠",
    NodeType.topic: "◆",
    NodeType.concept: "○",
    NodeType.definition: "🔑",
    NodeType.example: "💡",
    NodeType.insight: "✨",
    NodeType.action: "✅",
    NodeType.warning: "⚠️",
    NodeType.question: "❓",
    NodeType.detail: "•",
}


def banner() -> Panel:
    text = Text(_BANNER, style="bold cyan")
    return Panel(text, border_style="cyan", expand=False, subtitle="cerebro")


def print_banner() -> None:
    console.print(banner())


def _attach(tree: Tree, node: Node, max_depth: int | None, depth: int = 1) -> None:
    for child in node.children:
        icon = _TYPE_ICON.get(child.type, "•")
        label = Text(f"{icon} {child.title}")
        if child.type == NodeType.detail:
            label.stylize("dim")
        if max_depth is not None and depth >= max_depth and child.children:
            label.append(f"  (+{child.count() - 1} more)", style="dim italic")
            tree.add(label)
        else:
            branch = tree.add(label)
            _attach(branch, child, max_depth, depth + 1)


def map_preview(mm: MindMap, max_depth: int | None = None) -> Tree:
    """Render the mind map as a Rich tree for a pre-export preview.

    ``max_depth`` caps how many levels are expanded — used for batch/playlist
    maps where a full render would flood the terminal.
    """
    root_label = Text(f"🧠 {mm.root.title}", style="bold cyan")
    tree = Tree(root_label)
    _attach(tree, mm.root, max_depth)
    return tree


def print_preview(mm: MindMap, max_depth: int | None = None) -> None:
    console.print(map_preview(mm, max_depth))
