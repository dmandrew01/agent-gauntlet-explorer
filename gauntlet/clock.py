"""Puzzle clock. Late answers count as losses, and speed is 15% of the score.

The clock keeps a safety margin for the submit round trip and tells each
stage how long it may run.
"""

from __future__ import annotations

import time


class Clock:
    def __init__(self, time_remaining_s: float, max_time_s: float = 0.0, *, margin_s: float | None = None) -> None:
        total = float(time_remaining_s or max_time_s or 60.0)
        self.max_time_s = float(max_time_s or total)
        self.started = time.monotonic()
        if margin_s is None:
            margin_s = max(5.0, 0.08 * self.max_time_s)
        self.margin_s = margin_s
        self.deadline = self.started + total - margin_s

    def sync(self, time_remaining_s: float | None) -> None:
        """Re-anchor on the server's own countdown (never extends past it)."""
        if time_remaining_s is None:
            return
        try:
            server_deadline = time.monotonic() + float(time_remaining_s) - self.margin_s
        except (TypeError, ValueError):
            return
        self.deadline = min(self.deadline, server_deadline)

    def left(self) -> float:
        return self.deadline - time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def used_fraction(self) -> float:
        span = max(1e-6, self.deadline - self.started)
        return min(1.0, self.elapsed() / span)

    def budget(self, fraction: float, floor_s: float = 3.0, cap_s: float | None = None) -> float:
        """Seconds a stage may spend: ``fraction`` of what is left, within bounds."""
        b = max(floor_s, self.left() * fraction)
        if cap_s is not None:
            b = min(b, cap_s)
        return max(0.0, min(b, self.left()))

    def expired(self) -> bool:
        return self.left() <= 0
