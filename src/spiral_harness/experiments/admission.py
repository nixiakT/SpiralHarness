"""Trusted experiment admission of a candidate's content-addressed lineage."""

from __future__ import annotations

from typing import Literal, TypeVar

from pydantic import BaseModel

from spiral_harness.core.canonical import canonical_json_bytes, canonical_sha256
from spiral_harness.core.experiment import (
    EXPERIMENT_MANIFEST_MEDIA_TYPE,
    PROTOCOL_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolManifest,
)
from spiral_harness.core.models import (
    ArtifactRef,
    CandidateMutation,
    HarnessManifest,
    ImmutableModel,
    Sha256,
)
from spiral_harness.execution import CapabilityPolicy
from spiral_harness.harness import HarnessRegistry, HarnessRegistryError
from spiral_harness.storage import ArtifactRepository
from spiral_harness.verification import GateConfig

_ModelT = TypeVar("_ModelT", bound=BaseModel)

ADMISSION_REPORT_MEDIA_TYPE = "application/vnd.spiral-harness.admission-report.v1+json"

_ADMISSION_CHECKS = (
    "canonical_artifacts_verified",
    "candidate_experiment_joined",
    "protocol_seed_planes_matched",
    "frozen_policy_applied",
    "mutation_lineage_recomputed",
    "evidence_joined",
    "evaluation_plan_joined",
    "capability_policy_joined",
)


class CandidateAdmissionError(ValueError):
    """Raised when a candidate does not join its frozen experiment lineage."""


class AdmissionReport(ImmutableModel):
    """Typed proof emitted only after all trusted admission checks succeed."""

    schema_version: Literal["1"] = "1"
    admitted: Literal[True] = True
    candidate_ref: ArtifactRef
    experiment_ref: ArtifactRef
    protocol_ref: ArtifactRef
    parent_harness_ref: ArtifactRef
    child_harness_ref: ArtifactRef
    mutation_ref: ArtifactRef
    evidence_refs: tuple[ArtifactRef, ...]
    evaluation_plan_ref: ArtifactRef
    gate_config_ref: ArtifactRef
    capability_policy_ref: ArtifactRef
    mutation_policy_sha256: Sha256
    checks: tuple[
        Literal["canonical_artifacts_verified"],
        Literal["candidate_experiment_joined"],
        Literal["protocol_seed_planes_matched"],
        Literal["frozen_policy_applied"],
        Literal["mutation_lineage_recomputed"],
        Literal["evidence_joined"],
        Literal["evaluation_plan_joined"],
        Literal["capability_policy_joined"],
    ] = _ADMISSION_CHECKS


class CandidateAdmissionService:
    """Load and validate every artifact that can authorize a candidate.

    The caller supplies the already-frozen experiment reference explicitly.  A
    candidate cannot select a different experiment merely by embedding another
    reference in its own manifest.
    """

    def __init__(self, store: ArtifactRepository) -> None:
        self.store = store

    def admit(
        self,
        *,
        candidate_ref: ArtifactRef,
        experiment_ref: ArtifactRef,
    ) -> AdmissionReport:
        """Return a proof for a valid candidate, otherwise fail without side effects."""

        if experiment_ref.media_type != EXPERIMENT_MANIFEST_MEDIA_TYPE:
            raise CandidateAdmissionError("experiment artifact declares the wrong media type")
        candidate = self._load_canonical(candidate_ref, CandidateManifest, "candidate")
        if candidate.experiment_ref != experiment_ref:
            raise CandidateAdmissionError(
                "candidate experiment_ref does not match the frozen experiment"
            )

        experiment = self._load_canonical(experiment_ref, ExperimentManifest, "experiment")
        if experiment.protocol_ref.media_type != PROTOCOL_MANIFEST_MEDIA_TYPE:
            raise CandidateAdmissionError("protocol artifact declares the wrong media type")
        protocol = self._load_canonical(
            experiment.protocol_ref,
            ProtocolManifest,
            "protocol",
        )

        parent = self._load_canonical(
            candidate.parent_harness_ref,
            HarnessManifest,
            "parent harness",
        )
        child = self._load_canonical(
            candidate.child_harness_ref,
            HarnessManifest,
            "child harness",
        )
        mutation = self._load_canonical(
            candidate.mutation_ref,
            CandidateMutation,
            "candidate mutation",
        )

        self._verify_search_budget(experiment, protocol)
        self._verify_parent_lineage(
            parent_ref=candidate.parent_harness_ref,
            parent=parent,
            seed_ref=experiment.seed_harness_ref,
            protocol=protocol,
        )

        if candidate.evaluation_plan_ref != protocol.gate_config_ref:
            raise CandidateAdmissionError(
                "candidate evaluation_plan_ref does not match protocol gate_config_ref"
            )
        self._load_canonical(
            candidate.evaluation_plan_ref,
            GateConfig,
            "gate configuration",
        )
        self._load_canonical(
            protocol.capability_policy_ref,
            CapabilityPolicy,
            "capability policy",
        )

        if candidate.evidence_refs != mutation.hypothesis.evidence_refs:
            raise CandidateAdmissionError(
                "candidate evidence_refs do not match the mutation hypothesis evidence"
            )
        for evidence_ref in candidate.evidence_refs:
            try:
                self.store.get_bytes(evidence_ref)
            except Exception as exc:
                raise CandidateAdmissionError(
                    f"candidate evidence artifact {evidence_ref.sha256} could not be verified"
                ) from exc

        try:
            self.store.get_bytes(mutation.before.artifact)
        except Exception as exc:
            raise CandidateAdmissionError(
                "candidate before artifact could not be verified"
            ) from exc

        try:
            after_bytes = self.store.get_bytes(mutation.after.artifact)
        except Exception as exc:
            raise CandidateAdmissionError("candidate after artifact could not be verified") from exc

        try:
            recomputed_child = HarnessRegistry(experiment.mutation_policy).apply_mutation(
                parent=parent,
                parent_ref=candidate.parent_harness_ref,
                mutation=mutation,
                artifact_bytes=after_bytes,
                artifact_media_type=mutation.after.artifact.media_type,
            )
        except (HarnessRegistryError, TypeError, ValueError) as exc:
            raise CandidateAdmissionError(
                f"candidate mutation failed the frozen experiment policy: {exc}"
            ) from exc

        if child != recomputed_child:
            raise CandidateAdmissionError(
                "child harness does not equal the manifest recomputed from parent and mutation"
            )

        return AdmissionReport(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
            protocol_ref=experiment.protocol_ref,
            parent_harness_ref=candidate.parent_harness_ref,
            child_harness_ref=candidate.child_harness_ref,
            mutation_ref=candidate.mutation_ref,
            evidence_refs=candidate.evidence_refs,
            evaluation_plan_ref=candidate.evaluation_plan_ref,
            gate_config_ref=protocol.gate_config_ref,
            capability_policy_ref=protocol.capability_policy_ref,
            mutation_policy_sha256=canonical_sha256(experiment.mutation_policy),
        )

    def verify_report(
        self,
        *,
        candidate_ref: ArtifactRef,
        experiment_ref: ArtifactRef,
        report_ref: ArtifactRef,
    ) -> AdmissionReport:
        """Verify that a persisted report is the exact proof for this candidate."""

        if report_ref.media_type != ADMISSION_REPORT_MEDIA_TYPE:
            raise CandidateAdmissionError("admission report declares the wrong media type")
        report = self._load_canonical(report_ref, AdmissionReport, "admission report")
        if report.candidate_ref != candidate_ref:
            raise CandidateAdmissionError("admission report belongs to a different candidate")
        if report.experiment_ref != experiment_ref:
            raise CandidateAdmissionError("admission report belongs to a different experiment")

        expected = self.admit(
            candidate_ref=candidate_ref,
            experiment_ref=experiment_ref,
        )
        if report != expected:
            raise CandidateAdmissionError(
                "admission report does not match the trusted recomputed proof"
            )
        return report

    def _load_canonical(
        self,
        ref: ArtifactRef,
        model_type: type[_ModelT],
        label: str,
    ) -> _ModelT:
        """Load a typed artifact and reject noncanonical schema representations."""

        try:
            payload = self.store.get_bytes(ref)
            loaded = self.store.get_json(ref, model_type)
            canonical = canonical_json_bytes(loaded)
        except Exception as exc:
            raise CandidateAdmissionError(f"{label} artifact could not be verified: {exc}") from exc
        if payload != canonical:
            raise CandidateAdmissionError(
                f"{label} artifact is not in its canonical typed representation"
            )
        return loaded

    def _verify_parent_lineage(
        self,
        *,
        parent_ref: ArtifactRef,
        parent: HarnessManifest,
        seed_ref: ArtifactRef,
        protocol: ProtocolManifest,
    ) -> None:
        """Require an intact, acyclic parent chain ending at the experiment seed."""

        current_ref = parent_ref
        current = parent
        seen = {current_ref.sha256}
        while True:
            self._verify_protocol_planes(protocol, current)
            if current_ref == seed_ref:
                return
            ancestor_ref = current.parent
            if ancestor_ref is None:
                raise CandidateAdmissionError(
                    "candidate parent lineage does not reach the experiment seed harness"
                )
            if ancestor_ref.sha256 in seen:
                raise CandidateAdmissionError("candidate parent lineage contains a cycle")
            seen.add(ancestor_ref.sha256)
            current_ref = ancestor_ref
            current = self._load_canonical(
                ancestor_ref,
                HarnessManifest,
                "parent lineage harness",
            )

    @staticmethod
    def _verify_protocol_planes(
        protocol: ProtocolManifest,
        harness: HarnessManifest,
    ) -> None:
        fields = (
            "model_fingerprint",
            "runtime_fingerprint",
            "trusted_plane_version",
        )
        mismatched = tuple(
            field_name
            for field_name in fields
            if getattr(harness, field_name) != getattr(protocol, field_name)
        )
        if mismatched:
            joined = ", ".join(mismatched)
            raise CandidateAdmissionError(
                f"harness lineage does not match protocol frozen planes: {joined}"
            )

    @staticmethod
    def _verify_search_budget(
        experiment: ExperimentManifest,
        protocol: ProtocolManifest,
    ) -> None:
        fields = (
            "max_tokens",
            "max_tool_calls",
            "max_wall_time_seconds",
            "max_cost_usd",
            "max_evaluations",
        )
        exceeded: list[str] = []
        for field_name in fields:
            protocol_limit = getattr(protocol.budget, field_name)
            if protocol_limit is None:
                continue
            search_limit = getattr(experiment.search_budget, field_name)
            if search_limit is None or search_limit > protocol_limit:
                exceeded.append(field_name)
        if exceeded:
            joined = ", ".join(exceeded)
            raise CandidateAdmissionError(
                f"experiment search budget exceeds the protocol budget: {joined}"
            )


__all__ = [
    "ADMISSION_REPORT_MEDIA_TYPE",
    "AdmissionReport",
    "CandidateAdmissionError",
    "CandidateAdmissionService",
]
