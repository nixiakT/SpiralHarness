"""Helpers for releasing sensitive exception graphs at CLI trust boundaries."""

from __future__ import annotations

import traceback


def discard_exception_graph(error: BaseException) -> None:
    """Detach traceback, cause, and context links after clearing inactive frames."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        linked = (current.__cause__, current.__context__)
        pending.extend(item for item in linked if item is not None)
        captured_traceback = current.__traceback__
        if captured_traceback is not None:
            traceback.clear_frames(captured_traceback)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


__all__ = ["discard_exception_graph"]
