"""
Per-run usage tracker (async-safe) for accumulating model usage such as tokens.

We use ContextVar so concurrent async tasks/runs do not interfere with each other.
Only model adapters that have access to real provider usage (e.g. response.usage)
should record into this tracker.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class UsageTracker:
    """Accumulate usage for a single logical run (e.g. one pipeline execution)."""

    total_tokens: int = 0

    def add_tokens(self, n: Optional[int]) -> None:
        if n is None:
            return
        try:
            n_int = int(n)
        except Exception:
            return
        if n_int > 0:
            self.total_tokens += n_int


_CURRENT_TRACKER: ContextVar[Optional[UsageTracker]] = ContextVar("_CURRENT_TRACKER", default=None)


def get_current_tracker() -> Optional[UsageTracker]:
    """Return current tracker if tracking is enabled in this context."""
    return _CURRENT_TRACKER.get()


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """
    Enable per-run usage tracking for the current async context.

    Example:
        with track_usage() as tracker:
            await pipeline.run()
            print(tracker.total_tokens)
    """
    tracker = UsageTracker()
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_TRACKER.reset(token)


