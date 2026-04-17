"""
Episodic memory for the LLM-MPC controller.

Tracks the history of (state, action, outcome, surprise) tuples within a
single episode. Old entries are compressed into a running summary to keep
the context window manageable.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryEntry:
    step: int
    semantic_state_text: str
    action: List[float]
    predicted_next_text: Optional[str]
    actual_next_text: str
    surprise_level: str      # 'low' | 'medium' | 'high'
    surprise_cause: str
    strategy_note: str


class EpisodicMemory:
    """
    Maintains a sliding window of recent MemoryEntry objects plus a
    compressed text summary of older entries.

    The compressed summary preserves only surprising events and strategy
    changes — routine low-surprise steps are discarded.
    """

    def __init__(self, window_size: int = 8, compress_threshold: int = 12):
        """
        Args:
            window_size:        Number of recent entries kept in full detail.
            compress_threshold: When total entries exceed this, compress
                                entries outside the window into the summary.
        """
        self.window_size = window_size
        self.compress_threshold = compress_threshold
        self.entries: List[MemoryEntry] = []
        self._summary: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.compress_threshold:
            self._compress()

    def get_context(self) -> str:
        """Return a formatted string suitable for inclusion in an LLM prompt."""
        parts: List[str] = []

        if self._summary:
            parts.append(f"Earlier episode summary:\n{self._summary}")

        if self.entries:
            lines = []
            for e in self.entries[-self.window_size:]:
                line = f"  Step {e.step:3d}: action={[round(a, 3) for a in e.action]}"
                if e.surprise_level != "low":
                    line += f"  [{e.surprise_level.upper()} SURPRISE] {e.surprise_cause}"
                if e.strategy_note:
                    line += f"  → {e.strategy_note}"
                lines.append(line)
            parts.append("Recent steps:\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "No history yet."

    def high_surprise_steps(self) -> List[MemoryEntry]:
        """Return all entries with high surprise level."""
        return [e for e in self.entries if e.surprise_level == "high"]

    def last_strategy_note(self) -> str:
        """Return the most recent non-empty strategy note."""
        for e in reversed(self.entries):
            if e.strategy_note:
                return e.strategy_note
        return ""

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compress(self) -> None:
        """Move entries outside the window into the compressed summary."""
        old = self.entries[: -self.window_size]
        self.entries = self.entries[-self.window_size :]

        notable = [
            e for e in old
            if e.surprise_level in ("medium", "high") or e.strategy_note
        ]
        if not notable:
            return

        new_lines = []
        for e in notable:
            line = f"Step {e.step}: action={[round(a, 3) for a in e.action]}"
            if e.surprise_level != "low":
                line += f" — {e.surprise_level} surprise: {e.surprise_cause}"
            if e.strategy_note:
                line += f" → {e.strategy_note}"
            new_lines.append(line)

        addition = "\n".join(new_lines)
        self._summary = (
            self._summary + "\n" + addition if self._summary else addition
        )
