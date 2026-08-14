"""Provider-neutral typed failures that preserve safe response accounting."""

from __future__ import annotations

import math

from spiral_harness.execution.contracts import BackendTokenUsage


class BackendResponseRejectedError(RuntimeError):
    """A provider response was unusable after some accounting became known."""

    def __init__(
        self,
        detail: str,
        *,
        usage: BackendTokenUsage | None,
        cost_usd: float | None = None,
    ) -> None:
        super().__init__(detail)
        self.usage = None if usage is None else BackendTokenUsage.model_validate(usage, strict=True)
        if cost_usd is not None and (
            type(cost_usd) is not float or not math.isfinite(cost_usd) or cost_usd < 0.0
        ):
            raise ValueError("cost_usd must be a finite non-negative float or None")
        self.cost_usd = cost_usd


__all__ = ["BackendResponseRejectedError"]
