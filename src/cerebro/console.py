"""Single shared Rich Console for the whole CLI.

Previously cli.py, ui.py, and wizard.py each constructed their own
``Console()`` independently, so a setting applied to one (color, width) never
reached output printed through the others. Rich reads the ``NO_COLOR`` env
var (https://no-color.org) automatically when a Console is constructed, so
setting it before this module's first import — or just mutating
``console.no_color`` afterward, which Rich re-checks at render time, not
construction time — is enough to disable color everywhere.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme

console = Console()

# Overriding these three named styles reaches every `[dim]`/`[deep_pink3]`/
# `[bright_magenta]` markup call site in the app at once (Rich checks a
# pushed theme before falling back to its own style keywords), instead of
# threading a contrast flag through dozens of individual print() calls.
# Both map to bare "bold" rather than a specific hue: "dim" and the pink/
# magenta accent are stylistic flourishes, and the safest universally-legible
# choice is to keep the terminal's own default foreground (already tuned for
# that user's background) and just add weight, rather than guess a color
# that might itself wash out on an unknown background.
_HIGH_CONTRAST_THEME = Theme({"dim": "bold", "deep_pink3": "bold", "bright_magenta": "bold"})

_ascii_mode = False


def set_high_contrast(enabled: bool) -> None:
    if enabled:
        console.push_theme(_HIGH_CONTRAST_THEME)


def set_ascii(enabled: bool) -> None:
    global _ascii_mode
    _ascii_mode = enabled


def ascii_mode() -> bool:
    return _ascii_mode
