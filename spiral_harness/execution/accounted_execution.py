"""Load every execution artifact supported by the shared attempt ledger."""

from __future__ import annotations

import spiral_harness.execution.contracts as _contracts
import spiral_harness.execution.pure_contracts as _pure
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository

type AccountedExecution = _contracts.ModelExecution | _pure.PureReferenceExecution


class AccountedExecutionLoadError(ValueError):
    """An execution ref is unsupported or its canonical payload is invalid."""


def load_accounted_execution(
    repository: ArtifactRepository,
    ref: ArtifactRef,
) -> AccountedExecution:
    """Parse a supported execution after its reference shape is verified."""

    if ref.media_type == _contracts.MODEL_EXECUTION_MEDIA_TYPE:
        model_type = _contracts.ModelExecution
        label = "ModelExecution"
    elif ref.media_type == _contracts.PURE_REFERENCE_EXECUTION_MEDIA_TYPE:
        model_type = _pure.PureReferenceExecution
        label = "PURE reference execution"
    else:
        raise AccountedExecutionLoadError("unsupported execution artifact media type")
    try:
        loaded = repository.get_json(ref, model_type)
        return model_type.model_validate(loaded, strict=True)
    except Exception as exc:
        raise AccountedExecutionLoadError(f"execution artifact is not a canonical {label}") from exc


__all__ = [
    "AccountedExecution",
    "AccountedExecutionLoadError",
    "load_accounted_execution",
]
