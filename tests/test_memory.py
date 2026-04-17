"""Unit tests for p2mw.mpc.memory — no MetaWorld install required."""

import pytest

from p2mw.mpc.memory import EpisodicMemory, MemoryEntry


def _entry(step: int, surprise: str = "low", cause: str = "", strategy: str = "") -> MemoryEntry:
    return MemoryEntry(
        step=step,
        semantic_state_text=f"state at step {step}",
        action=[0.1, 0.0, 0.0, 0.5],
        predicted_next_text="predicted",
        actual_next_text="actual",
        surprise_level=surprise,
        surprise_cause=cause,
        strategy_note=strategy,
    )


class TestEpisodicMemory:
    def test_empty_context(self):
        mem = EpisodicMemory()
        assert mem.get_context() == "No history yet."

    def test_add_and_retrieve(self):
        mem = EpisodicMemory()
        mem.add(_entry(0))
        ctx = mem.get_context()
        assert "Step   0" in ctx

    def test_high_surprise_steps(self):
        mem = EpisodicMemory()
        mem.add(_entry(0, "low"))
        mem.add(_entry(1, "high", cause="object slipped"))
        mem.add(_entry(2, "medium"))
        high = mem.high_surprise_steps()
        assert len(high) == 1
        assert high[0].step == 1

    def test_last_strategy_note(self):
        mem = EpisodicMemory()
        mem.add(_entry(0))
        mem.add(_entry(1, strategy="approach from the left"))
        mem.add(_entry(2))
        assert mem.last_strategy_note() == "approach from the left"

    def test_last_strategy_note_empty(self):
        mem = EpisodicMemory()
        mem.add(_entry(0))
        assert mem.last_strategy_note() == ""

    def test_compression_triggers(self):
        mem = EpisodicMemory(window_size=3, compress_threshold=5)
        for i in range(6):
            mem.add(_entry(i, surprise="high", cause=f"cause {i}"))
        # After compression, entries list should be trimmed to window_size
        assert len(mem.entries) <= mem.window_size

    def test_compression_preserves_notable_events(self):
        mem = EpisodicMemory(window_size=3, compress_threshold=5)
        # Add 3 notable + 3 low-surprise (total 6 → triggers compress)
        for i in range(3):
            mem.add(_entry(i, surprise="high", cause=f"surprise {i}"))
        for i in range(3, 6):
            mem.add(_entry(i, surprise="low"))
        # Summary should mention the high-surprise events
        assert "surprise" in mem._summary.lower() or "high" in mem._summary.lower()

    def test_compression_drops_low_surprise_entries(self):
        mem = EpisodicMemory(window_size=3, compress_threshold=5)
        for i in range(6):
            mem.add(_entry(i, surprise="low"))
        # All entries were low-surprise; summary should remain empty
        assert mem._summary == ""

    def test_context_shows_strategy_note(self):
        mem = EpisodicMemory()
        mem.add(_entry(0, surprise="high", strategy="try from the right"))
        ctx = mem.get_context()
        assert "try from the right" in ctx

    def test_context_shows_surprise_label(self):
        mem = EpisodicMemory()
        mem.add(_entry(0, surprise="high", cause="gripper missed"))
        ctx = mem.get_context()
        assert "HIGH SURPRISE" in ctx
