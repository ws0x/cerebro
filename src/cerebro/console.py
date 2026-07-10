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

console = Console()
