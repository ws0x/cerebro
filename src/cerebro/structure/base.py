"""The structurer contract."""

from __future__ import annotations

from typing import Protocol

from ..ir import MindMap
from ..transcript import Transcript

# Processing levels, in increasing depth/cost. See README.
LEVELS = ("brief", "full", "expert")


class Structurer(Protocol):
    """Turns a transcript into a MindMap at the requested processing level."""

    def structure(self, transcript: Transcript, level: str = "full") -> MindMap: ...
