"""
Process-wide serialisation of PDFium calls.

PDFium is not thread-safe: pypdfium2 documents that its functions must not
run concurrently from different threads, not even on different documents.
The scan API analyses requests in a thread pool, so every function that
opens a PDF with pypdfium2 runs under ``PDFIUM`` (a re-entrant lock, so
guarded functions may call each other). Each of those functions opens and
closes its document inside the guarded call, so no PDFium object outlives
the lock. Pixel analysis of the rendered pages happens afterwards, outside
it, and still runs in parallel.

``PDFIUM.held()`` tells whether the current thread holds the guard; the
tests use it to check that no PDF is opened without it.
"""
from __future__ import annotations

import functools
import threading
from typing import Callable, TypeVar

T = TypeVar("T")


class _Guard:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()

    def __enter__(self) -> "_Guard":
        self._lock.acquire()
        self._local.depth = getattr(self._local, "depth", 0) + 1
        return self

    def __exit__(self, *exc) -> None:
        self._local.depth -= 1
        self._lock.release()

    def held(self) -> bool:
        return getattr(self._local, "depth", 0) > 0


PDFIUM = _Guard()


def exclusive(fn: Callable[..., T]) -> Callable[..., T]:
    """Runs ``fn`` while holding the PDFium guard."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with PDFIUM:
            return fn(*args, **kwargs)

    return wrapper
