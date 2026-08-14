"""Independent release service for HarnessFaultBench v4 partitions."""

from __future__ import annotations

import hashlib

from spiral_harness.benchmark._harness_fault_cases import (
    _TEMPLATE_BY_PARTITION,
    AUTHORITY_VERSION,
    DEFAULT_HARNESS_FAULT_SPLIT_CONFIG,
    GENERATOR_VERSION,
    PARTITION_OPENING_MEDIA_TYPE,
    PARTITION_ROSTER_MEDIA_TYPE,
    FaultFamily,
    FaultSurface,
    HarnessFaultAuthorityError,
    HarnessFaultPublicCommitment,
    HarnessFaultSplitConfig,
    PartitionCommitment,
    PartitionEvaluationGrant,
    PartitionOpening,
    PartitionRoster,
    PublicSearchTaskView,
    _generate_partition,
    _scenario_root,
    _validate_partition,
    verify_partition_opening,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.experiment import ProtocolPartition
from spiral_harness.storage.artifact_store import ArtifactStore


class HarnessFaultScenarioAuthority:
    """Independent authority. Do not pass this object into a search process."""

    __slots__ = ("_commitment", "_grants", "_search_view")

    def __init__(
        self,
        store: ArtifactStore,
        *,
        exploration_salt: bytes,
        gate_salt: bytes,
        sealed_salt: bytes,
        config: HarnessFaultSplitConfig = DEFAULT_HARNESS_FAULT_SPLIT_CONFIG,
    ) -> None:
        if type(store) is not ArtifactStore:
            raise TypeError("store must be an exact ArtifactStore")
        checked_config = HarnessFaultSplitConfig.model_validate(config, strict=True)
        salts = {
            ProtocolPartition.EXPLORATION: self._salt(exploration_salt, "exploration_salt"),
            ProtocolPartition.GATE: self._salt(gate_salt, "gate_salt"),
            ProtocolPartition.SEALED: self._salt(sealed_salt, "sealed_salt"),
        }
        if len(set(salts.values())) != 3:
            raise ValueError("partition salts must be distinct")
        salt_commitments = {
            partition: hashlib.sha256(salt).hexdigest() for partition, salt in salts.items()
        }
        authority_id = canonical_sha256(
            {
                "authority_version": AUTHORITY_VERSION,
                "generator_version": GENERATOR_VERSION,
                "config": checked_config,
                "salt_commitments": salt_commitments,
            }
        )
        scenarios = {
            partition: _generate_partition(checked_config, partition, salt)
            for partition, salt in salts.items()
        }
        for values in scenarios.values():
            _validate_partition(values)
        self._validate_axis_separation(scenarios)
        rosters = {
            partition: PartitionRoster(
                authority_id=authority_id,
                partition=partition,
                tasks=tuple(item.task for item in values),
            )
            for partition, values in scenarios.items()
        }
        commitments = tuple(
            PartitionCommitment(
                authority_id=authority_id,
                partition=partition,
                config_fingerprint=checked_config.fingerprint,
                template_id=_TEMPLATE_BY_PARTITION[partition],
                family_count=len(FaultFamily),
                surface_count=len(FaultSurface),
                group_count=len(FaultFamily) * checked_config.groups_per_family,
                scenario_count=len(scenarios[partition]),
                salt_commitment=salt_commitments[partition],
                scenario_root=_scenario_root(authority_id, partition, scenarios[partition]),
                roster_root=rosters[partition].root,
            )
            for partition in ProtocolPartition
        )
        self._commitment = HarnessFaultPublicCommitment(
            authority_id=authority_id,
            config=checked_config,
            partitions=commitments,
        )
        self._grants = self._persist_grants(store, salts, scenarios, rosters)
        self._search_view = PublicSearchTaskView(
            public_commitment=self._commitment,
            exploration_tasks=rosters[ProtocolPartition.EXPLORATION].tasks,
        )

    @staticmethod
    def _salt(value: bytes, label: str) -> bytes:
        if not isinstance(value, bytes):
            raise TypeError(f"{label} must be bytes")
        if len(value) < 32:
            raise ValueError(f"{label} must contain at least 32 bytes")
        return bytes(value)

    @staticmethod
    def _validate_axis_separation(scenarios: dict) -> None:
        owners: dict[tuple[str, str], ProtocolPartition] = {}
        for partition, values in scenarios.items():
            for item in values:
                for axis, value in (
                    ("template", item.template_id),
                    ("source", item.source_id),
                    ("group", item.group_id),
                ):
                    previous = owners.setdefault((axis, value), partition)
                    if previous is not partition:
                        raise HarnessFaultAuthorityError(f"{axis} crosses a protocol partition")

    def _persist_grants(
        self,
        store: ArtifactStore,
        salts: dict,
        scenarios: dict,
        rosters: dict,
    ) -> dict[ProtocolPartition, PartitionEvaluationGrant]:
        grants = {}
        for partition in ProtocolPartition:
            roster_ref = store.put_json(rosters[partition], media_type=PARTITION_ROSTER_MEDIA_TYPE)
            opening = PartitionOpening(
                authority_id=self._commitment.authority_id,
                partition=partition,
                config=self._commitment.config,
                salt_hex=salts[partition].hex(),
                scenario_commitments=tuple(
                    item.scenario_commitment for item in scenarios[partition]
                ),
                scenario_root=self._commitment.partition(partition).scenario_root,
                roster_root=rosters[partition].root,
            )
            opening_ref = store.put_json(opening, media_type=PARTITION_OPENING_MEDIA_TYPE)
            grant = PartitionEvaluationGrant(
                public_commitment=self._commitment,
                partition=partition,
                opening_ref=opening_ref,
                roster_ref=roster_ref,
            )
            verify_partition_opening(store, grant)
            grants[partition] = grant
        return grants

    @property
    def public_search_view(self) -> PublicSearchTaskView:
        return self._search_view

    def issue_exploration_grant(self) -> PartitionEvaluationGrant:
        return self._grants[ProtocolPartition.EXPLORATION]

    def issue_gate_grant(self) -> PartitionEvaluationGrant:
        return self._grants[ProtocolPartition.GATE]

    def issue_sealed_grant(self) -> PartitionEvaluationGrant:
        return self._grants[ProtocolPartition.SEALED]


__all__ = ["HarnessFaultScenarioAuthority"]
