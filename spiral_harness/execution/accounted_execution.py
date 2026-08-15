"""Load every execution artifact supported by the shared attempt ledger."""

from __future__ import annotations

import spiral_harness.execution.contracts as _contracts
import spiral_harness.execution.native_function_contracts as _native
import spiral_harness.execution.pure_contracts as _pure
import spiral_harness.providers.openai_native_contracts as _native_provider
from spiral_harness.core.models import ArtifactRef
from spiral_harness.storage.protocol import ArtifactRepository

type AccountedExecution = (
    _contracts.ModelExecution | _native.NativeFunctionExecution | _pure.PureReferenceExecution
)


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
    elif ref.media_type == _native.NATIVE_FUNCTION_EXECUTION_MEDIA_TYPE:
        model_type = _native.NativeFunctionExecution
        label = "native function execution"
    elif ref.media_type == _contracts.PURE_REFERENCE_EXECUTION_MEDIA_TYPE:
        model_type = _pure.PureReferenceExecution
        label = "PURE reference execution"
    else:
        raise AccountedExecutionLoadError("unsupported execution artifact media type")
    try:
        if model_type is _native.NativeFunctionExecution:
            execution = _native.load_canonical_native_artifact(repository, ref, model_type)
            request = _native.load_canonical_native_artifact(
                repository,
                execution.request_ref,
                _native_provider.NativeFunctionCallRequest,
            )
            if request != execution.request:
                raise ValueError("native request artifact differs from its execution")
            if execution.response_ref is not None:
                response = _native.load_canonical_native_artifact(
                    repository,
                    execution.response_ref,
                    _native_provider.NativeFunctionCallResponse,
                )
                if response != execution.response:
                    raise ValueError("native response artifact differs from its execution")
            return execution
        loaded = repository.get_json(ref, model_type)
        return model_type.model_validate(loaded, strict=True)
    except Exception as exc:
        raise AccountedExecutionLoadError(f"execution artifact is not a canonical {label}") from exc


__all__ = [
    "AccountedExecution",
    "AccountedExecutionLoadError",
    "load_accounted_execution",
]
