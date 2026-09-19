"""Compare pairs of clips several at a time, in separate processes.

Threads are not an option. Run side by side in one process, the FFT gives a wrong answer now and then (measured: about
one comparison in a hundred came back with a different lag), which is the fault that made `correlate` single-threaded
in the first place. Separate processes share nothing, and gave exactly the answers of one-at-a-time in every comparison
tried. They read the audio from the cache files (memory-mapped), so nothing big is copied between processes.
"""
from __future__ import annotations

import functools
import multiprocessing
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np

MAX_WORKERS = 4  # measured: 4 processes give 2.7x, 8 give 2.95x, 16 no more (the FFTs are limited by memory speed)


def default_workers() -> int:
    """Worker processes to use: half the cores, at most MAX_WORKERS. 1 means one comparison at a time."""
    return max(1, min(MAX_WORKERS, (os.cpu_count() or 2) // 2))


@functools.lru_cache(maxsize=128)
def _open(path: str) -> np.ndarray:
    return np.memmap(path, dtype="<f4", mode="r")


def _compare(job: tuple[str, str, int, float]) -> tuple[float, float]:
    """One comparison, run in a worker process: (lag, z) as `correlate` gives them."""
    from .sync import correlate  # here rather than at the top: sync imports this module

    path_a, path_b, fs, min_overlap = job
    return correlate(_open(path_a), _open(path_b), fs, min_overlap=min_overlap)


class Comparer:
    """The comparisons of one sync run: one at a time, or several at a time in worker processes.

    `compute(a, b)` does a comparison in this process. With workers, the caller says which comparisons it will probably
    want next (`top_up`) and they are computed in the background while it works on the current one; `result` then gives
    an answer at once. An answer is the same whichever way it was made, and a comparison that is not wanted after all
    is dropped if it has not begun (`cancel`). If worker processes cannot be started, or die, everything carries on one at a time.
    Pairs are always given with the names in sorted order, as `sync` does.
    """

    def __init__(
        self, paths: dict[str, str], compute: Callable[[str, str], tuple[float, float]], workers: int = 1,
        fs: int = 16000, min_overlap: float = 5.0, on_fallback: Callable[[str], None] | None = None,
    ):
        self.paths, self.compute, self.fs, self.min_overlap = paths, compute, fs, min_overlap
        self.workers = workers if workers >= 2 else 1
        self.on_fallback = on_fallback
        self._pool: ProcessPoolExecutor | None = None
        self._flying: dict[tuple[str, str], Future] = {}
        self._broken = False

    @property
    def parallel(self) -> bool:
        return self.workers >= 2 and not self._broken

    def _in_flight(self) -> int:
        return sum(1 for f in self._flying.values() if not f.done())

    def _fall_back(self, why: str) -> None:
        if not self._broken:
            self._broken = True
            self.cancel()
            if self.on_fallback:
                self.on_fallback(why)

    def _submit(self, pair: tuple[str, str]) -> None:
        try:
            if self._pool is None:  # started when first needed: a small job never pays for it
                self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"))
            job = (self.paths[pair[0]], self.paths[pair[1]], self.fs, self.min_overlap)
            self._flying[pair] = self._pool.submit(_compare, job)
        except (OSError, BrokenProcessPool, RuntimeError) as e:
            self._fall_back(f"could not start worker processes ({e.__class__.__name__})")

    def top_up(self, upcoming: Iterator[tuple[str, str]]) -> None:
        """Start the next pairs from `upcoming` until `workers` comparisons are in flight (not yet finished)."""
        while self.parallel and self._in_flight() < self.workers:
            pair = next(upcoming, None)
            if pair is None:
                return
            if pair not in self._flying:  # one that finished earlier and was not used yet is still good
                self._submit(pair)

    def result(self, a: str, b: str) -> tuple[float, float]:
        """(lag, z) for the pair: from the background if it was started there, else worked out now."""
        future = self._flying.pop((a, b), None)
        if future is not None:
            try:
                return future.result()
            except BrokenProcessPool:
                self._fall_back("the worker processes stopped")
            except Exception:  # noqa: BLE001 - this one comparison failed out there: work it out here, and see
                pass
        return self.compute(a, b)

    def cancel(self) -> None:
        """Drop comparisons that are queued but have not begun: they were guesses that are no longer wanted. Those
        already running or finished are kept, since an answer is an answer and a later search may still ask for it."""
        for pair, future in list(self._flying.items()):
            if future.cancel():
                del self._flying[pair]

    def close(self) -> None:
        self._flying.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
