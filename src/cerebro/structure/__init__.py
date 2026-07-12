"""Structure layer: ``Transcript`` -> ``MindMap`` IR.

Pluggable strategies. ``HeuristicStructurer`` is deterministic and offline
(proves the pipeline); an LLM-backed structurer swaps in behind the same
interface for genuinely smart maps.
"""

from __future__ import annotations

from .base import Structurer
from .heuristic import HeuristicStructurer

__all__ = ["Structurer", "HeuristicStructurer"]
