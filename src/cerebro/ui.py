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

_BANNER_ART = (
    r"   ______   ______  ____    ______  ____     ____    ____  " + "\n"
    r"  / ____/  / ____/ / __ \  / ____/ / __ )   / __ \  / __ \ " + "\n"
    r" / /      / __/   / /_/ / / __/   / __  |  / /_/ / / / / / " + "\n"
    r"/ /___   / /___  / _, _/ / /___  / /_/ /  / _, _/ / /_/ /  " + "\n"
    r"\____/  /_____/ /_/ |_| /_____/ /_____/  /_/ |_|  \____/   "
)


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
    from rich.align import Align
    
    text = Text()
    lines = _BANNER_ART.split("\n")
    # Apply vertical gradient coloring
    colors = ["cyan", "cyan", "deep_pink3", "bright_magenta", "bright_magenta"]
    for line, color in zip(lines, colors):
        text.append(line + "\n", style=color)
        
    text.append("\n          video → smart mind maps", style="dim white")
    
    aligned = Align.center(text)
    return Panel(
        aligned,
        title="[bold bright_magenta]cerebro[/bold bright_magenta]",
        subtitle="[dim]whisper + llm[/dim]",
        border_style="deep_pink3",
        expand=True,
    )


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
