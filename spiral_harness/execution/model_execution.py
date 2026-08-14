"""Pure construction helpers for fixed-model execution records."""

from __future__ import annotations

import spiral_harness.execution.contracts as _contracts


def make_model_execution(
    *,
    spec: _contracts.FrozenModelSpec,
    task: _contracts.CandidateTask,
    request: _contracts.ModelRequest,
    output: str | None,
    status: _contracts.ExecutionStatus,
    usage: _contracts.BackendTokenUsage,
    latency_ms: float,
    execution_fingerprint: str,
    request_sha256: str,
    error: _contracts.ExecutionError | None,
    cost_usd: float | None,
    provider_identity_observation: _contracts.ProviderIdentityObservation | None,
) -> _contracts.ModelExecution:
    """Build one immutable execution from already validated runner inputs."""

    return _contracts.ModelExecution(
        task=task,
        request=request,
        output=output,
        status=status,
        usage=_contracts.ModelUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        ),
        spec=spec,
        provider_identity_observation=provider_identity_observation,
        execution_fingerprint=execution_fingerprint,
        request_sha256=request_sha256,
        error=error,
    )


__all__ = ["make_model_execution"]
