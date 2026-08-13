from __future__ import annotations

from spiral_harness.core.experiment import (
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    ProtocolPartition,
    ProtocolSplit,
)
from spiral_harness.core.models import ArtifactRef
from spiral_harness.evolution.replay_gate_execution import ReplayGateBoundary, execute_replay_gate
from spiral_harness.evolution.replay_study import ReplayStudyExecution


def gate_boundary(execution: ReplayStudyExecution) -> ReplayGateBoundary:
    return ReplayGateBoundary(
        protocol=execution.fixture.protocol,
        protocol_ref=execution.fixture.protocol_ref,
        gate_split_ref=next(
            split.manifest_ref
            for split in execution.fixture.protocol.splits
            if split.partition is ProtocolPartition.GATE
        ),
    )


def foreign_gate_boundary(execution: ReplayStudyExecution) -> ReplayGateBoundary:
    store = execution.fixture.store
    foreign_gate_ref = store.put_json(
        {"fixture": "foreign-baseline-gate", "partition": "gate", "task_ids": ["gate-1"]}
    )
    foreign_splits = tuple(
        ProtocolSplit(partition=split.partition, manifest_ref=foreign_gate_ref)
        if split.partition is ProtocolPartition.GATE
        else split
        for split in execution.fixture.protocol.splits
    )
    foreign_protocol = execution.fixture.protocol.model_copy(update={"splits": foreign_splits})
    foreign_protocol_ref = store.put_json(
        foreign_protocol,
        media_type=PROTOCOL_MANIFEST_MEDIA_TYPE,
    )
    return ReplayGateBoundary(
        protocol=foreign_protocol,
        protocol_ref=foreign_protocol_ref,
        gate_split_ref=foreign_gate_ref,
    )


def replay_gate_closure(
    execution: ReplayStudyExecution,
    boundary: ReplayGateBoundary | None = None,
) -> ArtifactRef:
    return execute_replay_gate(execution, boundary=boundary).closure_ref
