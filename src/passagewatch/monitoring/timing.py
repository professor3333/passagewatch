"""Per-stage wall-clock timing for profiling and job logs.

A :class:`StageTimes` accumulates seconds per named stage. ``stage`` times a block; ``wrap``
times each step of an iterator (the time spent producing each item, such as decoding a
frame), so lazily produced frames are charged to the stage that produces them.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


class StageTimes:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - start)

    def wrap(self, name: str, items: Iterable[T]) -> Iterator[T]:
        iterator = iter(items)
        while True:
            start = time.perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                self.add(name, time.perf_counter() - start)
                return
            self.add(name, time.perf_counter() - start)
            yield item

    def rounded(self, digits: int = 3) -> dict[str, float]:
        return {k: round(v, digits) for k, v in self.seconds.items()}
