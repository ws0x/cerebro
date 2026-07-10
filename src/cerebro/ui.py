"""Rich UI helpers — banner, step progress, and an in-terminal map preview.

Visual language deliberately echoes vidforge so the tools feel like a family.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from .console import ascii_mode, console
from .ir import MindMap, Node, NodeType

_BANNER_ART = (
    r"   ______   ______  ____    ______  ____     ____    ____  " + "\n"
    r"  / ____/  / ____/ / __ \  / ____/ / __ )   / __ \  / __ \ " + "\n"
    r" / /      / __/   / /_/ / / __/   / __  |  / /_/ / / / / / " + "\n"
    r"/ /___   / /___  / _, _/ / /___  / /_/ /  / _, _/ / /_/ /  " + "\n"
    r"\____/  /_____/ /_/ |_| /_____/ /_____/  /_/ |_|  \____/   "
)


# Icon per semantic node type, for the terminal preview. Pictographic emoji
# don't render everywhere (some Windows consoles, screen readers that spell
# out "brain emoji" on every single node) — _TYPE_ICON_ASCII is the fallback
# when ascii_mode() is on, selected via the --ascii flag.
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
_TYPE_ICON_ASCII = {
    NodeType.root: "*",
    NodeType.topic: "#",
    NodeType.concept: "o",
    NodeType.definition: "[D]",
    NodeType.example: "[EX]",
    NodeType.insight: "[!]",
    NodeType.action: "[OK]",
    NodeType.warning: "[WARN]",
    NodeType.question: "[?]",
    NodeType.detail: "-",
}


def _icon(node_type: NodeType) -> str:
    icons = _TYPE_ICON_ASCII if ascii_mode() else _TYPE_ICON
    return icons.get(node_type, "-" if ascii_mode() else "•")


# Width the big wordmark actually needs (art width + panel borders/padding).
# Below this, Rich would wrap each ASCII-art line mid-character into broken
# garbage, so a narrow terminal gets a compact fallback banner instead.
_ART_WIDTH = max(len(line) for line in _BANNER_ART.split("\n"))
_MIN_FULL_BANNER_WIDTH = _ART_WIDTH + 6


def banner() -> Panel:
    from rich.align import Align

    width = console.width
    if width < _MIN_FULL_BANNER_WIDTH:
        return _compact_banner()

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


def _compact_banner() -> Panel:
    from rich.align import Align

    text = Text()
    text.append(f"{_icon(NodeType.root)} cerebro", style="bold bright_magenta")
    text.append("\nvideo → smart mind maps", style="dim white")
    return Panel(
        Align.center(text),
        subtitle="[dim]whisper + llm[/dim]",
        border_style="deep_pink3",
        expand=True,
    )


def print_banner() -> None:
    console.print(banner())


_NOTE_PREVIEW_LEN = 50


def _attach(tree: Tree, node: Node, max_depth: int | None, depth: int = 1) -> None:
    for child in node.children:
        label = Text(f"{_icon(child.type)} {child.title}")
        if child.type == NodeType.detail:
            label.stylize("dim")
        if child.note:
            note = child.note.strip()
            if len(note) > _NOTE_PREVIEW_LEN:
                note = note[:_NOTE_PREVIEW_LEN].rsplit(" ", 1)[0] + "…"
            label.append(f"  — {note}", style="dim italic")
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
    root_label = Text(f"{_icon(NodeType.root)} {mm.root.title}", style="bold cyan")
    tree = Tree(root_label)
    _attach(tree, mm.root, max_depth)
    return tree


def print_preview(mm: MindMap, max_depth: int | None = None) -> None:
    console.print(map_preview(mm, max_depth))
