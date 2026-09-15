"""Concurrency control.

Image decoding buffers entire bitmaps, so N simultaneous large operations cost
roughly N x (width x height x channels) bytes. Without a cap, a burst of
requests - or one agent looping over a batch - turns into an OOM kill that
takes the whole server down for every client.

The gate is deliberately *blocking with a deadline* rather than fail-fast: an
agent issuing three calls in parallel should have them queue, not error. Only
sustained overload produces a ``too_many_requests`` error.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from ..errors import ConcurrencyLimitError, LimitExceededError


class ConcurrencyGate:
    """A bounded semaphore with an acquisition deadline."""

    __slots__ = ("_limit", "_semaphore", "_timeout")

    def __init__(self, limit: int, *, timeout_seconds: float = 60.0) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._semaphore = threading.BoundedSemaphore(limit)
        self._limit = limit
        self._timeout = timeout_seconds

    @property
    def limit(self) -> int:
        return self._limit

    @contextmanager
    def slot(self) -> Iterator[None]:
        acquired = self._semaphore.acquire(timeout=self._timeout)
        if not acquired:
            raise ConcurrencyLimitError(
                f"server is busy: {self._limit} operations are already running",
                limit=self._limit,
            )
        try:
            yield
        finally:
            self._semaphore.release()


class Deadline:
    """A wall-clock budget checked at step boundaries.

    Cooperative on purpose. A CPU-bound Pillow call cannot be interrupted from
    Python - the only ways to stop it are to kill the thread (which leaks the
    GIL and can corrupt interpreter state) or to run it in a subprocess (which
    would mean shipping images across a process boundary for every operation).
    What this *can* do is stop a long *sequence* of operations from running
    unbounded: a pipeline checks between steps, and the quality search checks
    between encodes, so a call that would otherwise perform a hundred encodes
    gives up after the budget instead.

    That covers the realistic amplification cases (a target-size search across
    twelve variants, a deep pipeline on a large image) without pretending to a
    hard guarantee it cannot provide.
    """

    __slots__ = ("_limit", "_start")

    def __init__(self, seconds: float) -> None:
        self._limit = seconds
        self._start = time.monotonic()

    @property
    def limit(self) -> float:
        return self._limit

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def expired(self) -> bool:
        return self.elapsed > self._limit

    def check(self, context: str = "operation") -> None:
        """Raise if the budget is spent."""
        if self.expired:
            raise LimitExceededError(
                f"{context} exceeded the {self._limit:g}s time budget",
                elapsed_seconds=round(self.elapsed, 2),
                limit_seconds=self._limit,
            )


__all__ = ["ConcurrencyGate", "Deadline"]
