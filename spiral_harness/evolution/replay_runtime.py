"""Deterministic runtime and optimizer adapters for the replay study."""

from __future__ import annotations

from spiral_harness.core.experiment import (
    CANDIDATE_MANIFEST_MEDIA_TYPE,
    CandidateManifest,
    ExperimentManifest,
    ProtocolPartition,
)
from spiral_harness.core.models import (
    CANDIDATE_MUTATION_MEDIA_TYPE,
    HARNESS_MANIFEST_MEDIA_TYPE,
    ArtifactRef,
    CandidateMutation,
    ComponentKind,
    HarnessComponentRef,
    HarnessManifest,
    MutationHypothesis,
)
from spiral_harness.evolution.models import (
    CandidateScreen,
    CandidateScreenFailure,
    CandidateScreenStatus,
    Diagnosis,
    GateAggregateView,
    Nomination,
    PromptProposal,
    ProposalBatch,
    SearchRunManifest,
    StrategyFeedbackView,
)
from spiral_harness.evolution.orchestrator import (
    CandidateMaterialization,
    StrategyArtifactView,
    TrustedStrategyFeedbackContent,
)
from spiral_harness.evolution.replay_setup import (
    PROMPT_COMPONENT_NAME,
    FrozenReplayFixture,
)
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.experiments.baselines import BaselineKind, FeedbackType
from spiral_harness.harness.registry import HarnessRegistry


class ReplayRuntime:
    """Trusted fixture adapter that rejects each complete candidate at screen."""

    def __init__(self, fixture: FrozenReplayFixture) -> None:
        self.fixture = fixture
        self.feedback_call_count = 0
        self.materialization_call_count = 0
        self.screen_call_count = 0
        self.gate_call_count = 0

    def collect_feedback(
        self,
        *,
        search_run_ref: ArtifactRef,
        baseline_kind: BaselineKind,
        round_index: int,
        champion_harness_ref: ArtifactRef,
        prior_gate_aggregate: GateAggregateView | None,
    ) -> object:
        if prior_gate_aggregate is not None:
            raise RuntimeError("one-round replay fixture cannot have prior gate feedback")
        self.feedback_call_count += 1
        feedback = self.fixture.feedback
        values: dict[str, object] = {
            "baseline_kind": baseline_kind,
            "benchmark_metadata_ref": feedback.safe_benchmark_metadata_ref,
        }
        exposed = {FeedbackType.BENCHMARK_METADATA}
        if baseline_kind is not BaselineKind.STATIC:
            values["exploration_inputs_ref"] = feedback.exploration_inputs_ref
            exposed.add(FeedbackType.EXPLORATION_INPUTS)
        if baseline_kind in {
            BaselineKind.PROMPT_ONLY,
            BaselineKind.EVIDENCE_TARGETED,
        }:
            values.update(
                exploration_aggregates_ref=feedback.exploration_aggregates_ref,
                exploration_item_feedback_ref=feedback.exploration_item_feedback_ref,
                exploration_trajectories_ref=feedback.exploration_trajectories_ref,
            )
            exposed.update(
                {
                    FeedbackType.EXPLORATION_AGGREGATES,
                    FeedbackType.EXPLORATION_ITEM_FEEDBACK,
                    FeedbackType.EXPLORATION_TRAJECTORIES,
                }
            )
        if baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            values["diagnostic_evidence_ref"] = feedback.diagnostic_cluster_ref
            exposed.add(FeedbackType.DIAGNOSTIC_EVIDENCE)
        view = StrategyFeedbackView(
            exposed_feedback=tuple(exposed),
            **values,
        )
        run = self.fixture.store.get_json(search_run_ref, SearchRunManifest)
        return self.fixture.feedback_service.attest(
            TrustedStrategyFeedbackContent(
                search_run_ref=search_run_ref,
                experiment_ref=run.experiment_ref,
                benchmark_binding_ref=self.fixture.benchmark_binding_ref,
                exploration_split_ref=next(
                    split.manifest_ref
                    for split in self.fixture.protocol.splits
                    if split.partition is ProtocolPartition.EXPLORATION
                ),
                baseline_kind=baseline_kind,
                search_run_seed=run.search_run_seed,
                round_index=round_index,
                champion_harness_ref=champion_harness_ref,
                prior_gate_aggregate=prior_gate_aggregate,
                view=view,
            )
        )

    def materialize_proposal(
        self,
        *,
        search_run_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposal: PromptProposal,
        proposal_ref: ArtifactRef,
        champion_harness_ref: ArtifactRef,
    ) -> object:
        del feedback_ref
        self.materialization_call_count += 1
        store = self.fixture.store
        if champion_harness_ref.media_type != HARNESS_MANIFEST_MEDIA_TYPE:
            raise ValueError(
                "champion_harness_ref must declare owner media type "
                f"{HARNESS_MANIFEST_MEDIA_TYPE!r}"
            )
        parent = store.get_json(champion_harness_ref, HarnessManifest)
        experiment = store.get_json(self.fixture.experiment_ref, ExperimentManifest)
        hypothesis = store.get_json(proposal.hypothesis_ref, MutationHypothesis)
        before = next(
            component
            for component in parent.components
            if component.name == proposal.target_component_name
        )
        mutation = CandidateMutation(
            target_component=proposal.target_component_name,
            before=before,
            after=HarnessComponentRef(
                name=proposal.target_component_name,
                kind=ComponentKind.PROMPT,
                artifact=proposal.after_prompt_ref,
            ),
            hypothesis=hypothesis,
        )
        mutation_ref = store.put_json(
            mutation,
            media_type=CANDIDATE_MUTATION_MEDIA_TYPE,
        )
        child = HarnessRegistry(experiment.mutation_policy).apply_mutation(
            parent=parent,
            parent_ref=champion_harness_ref,
            mutation=mutation,
            artifact_bytes=store.get_bytes(proposal.after_prompt_ref),
            artifact_media_type=proposal.after_prompt_ref.media_type,
        )
        child_ref = store.put_json(child, media_type=HARNESS_MANIFEST_MEDIA_TYPE)
        candidate_ref = store.put_json(
            CandidateManifest(
                experiment_ref=self.fixture.experiment_ref,
                parent_harness_ref=champion_harness_ref,
                child_harness_ref=child_ref,
                mutation_ref=mutation_ref,
                evidence_refs=hypothesis.evidence_refs,
                evaluation_plan_ref=self.fixture.gate_config_ref,
            ),
            media_type=CANDIDATE_MANIFEST_MEDIA_TYPE,
        )
        return CandidateMaterialization(
            search_run_ref=search_run_ref,
            baseline_kind=proposal.baseline_kind,
            round_index=proposal.round_index,
            proposal_ref=proposal_ref,
            candidate_ref=candidate_ref,
            candidate_harness_ref=child_ref,
        )

    def screen_candidate(
        self,
        *,
        search_run_ref: ArtifactRef,
        feedback_ref: ArtifactRef,
        proposal: PromptProposal,
        proposal_ref: ArtifactRef,
        materialization: CandidateMaterialization,
        champion_harness_ref: ArtifactRef,
    ) -> object:
        del search_run_ref, feedback_ref, champion_harness_ref
        self.screen_call_count += 1
        return CandidateScreen(
            baseline_kind=proposal.baseline_kind,
            round_index=proposal.round_index,
            proposal_ref=proposal_ref,
            candidate_ref=materialization.candidate_ref,
            candidate_harness_ref=materialization.candidate_harness_ref,
            status=CandidateScreenStatus.REJECTED,
            failure_codes=(CandidateScreenFailure.CONSTRAINT_FAILED,),
        )

    def run_gate(
        self,
        *,
        search_run_ref: ArtifactRef,
        nomination: Nomination,
        nomination_ref: ArtifactRef,
        search_tail_ref: ArtifactRef,
    ) -> object:
        del search_run_ref, nomination, nomination_ref, search_tail_ref
        self.gate_call_count += 1
        raise RuntimeError("replay fixture rejects every candidate before nomination")

    def attempt_ledger_for(self, evaluation_ref: ArtifactRef) -> AttemptLedger:
        del evaluation_ref
        raise RuntimeError("replay fixture never creates a trusted screen evaluation")


class ReplayStrategyPlugin:
    """One deterministic optimizer plugin instance scoped to one study cell."""

    def __init__(
        self,
        *,
        baseline_kind: BaselineKind,
        manifest_ref: ArtifactRef,
        implementation_ref: ArtifactRef,
        failure_signature_ref: ArtifactRef,
        evidence_packet_ref: ArtifactRef,
        diagnostic_cluster_ref: ArtifactRef,
    ) -> None:
        self.baseline_kind = baseline_kind
        self._manifest_ref = manifest_ref
        self._implementation_ref = implementation_ref
        self.failure_signature_ref = failure_signature_ref
        self.evidence_packet_ref = evidence_packet_ref
        self.diagnostic_cluster_ref = diagnostic_cluster_ref

    @property
    def manifest_ref(self) -> ArtifactRef:
        return self._manifest_ref

    @property
    def implementation_ref(self) -> ArtifactRef:
        return self._implementation_ref

    def diagnose(
        self,
        *,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        search_run_ref: ArtifactRef,
        round_index: int,
        parent_harness_ref: ArtifactRef,
        artifacts: StrategyArtifactView,
    ) -> object:
        del search_run_ref, parent_harness_ref
        if self.baseline_kind is not BaselineKind.EVIDENCE_TARGETED:
            raise RuntimeError("only evidence-targeted replay cells diagnose")
        if feedback.diagnostic_evidence_ref != self.diagnostic_cluster_ref:
            raise RuntimeError("diagnostic feedback differs from the frozen fixture")
        artifacts.read_json(self.diagnostic_cluster_ref)
        artifacts.read_json(self.failure_signature_ref)
        artifacts.read_json(self.evidence_packet_ref)
        return (
            Diagnosis(
                diagnosis_id=f"fixture-diagnosis-{feedback_ref.sha256[:16]}",
                source_feedback_ref=feedback_ref,
                failure_signature_refs=(self.failure_signature_ref,),
                evidence_packet_refs=(self.evidence_packet_ref,),
                target_component_name=PROMPT_COMPONENT_NAME,
                observed_failure="the fixture trajectory omits an explicit verification step",
                root_cause="the seed prompt does not request independent verification",
                mechanism="replace the prompt with an explicit verify-before-answer instruction",
                predicted_benefit="the candidate should produce more self-checked answers",
                predicted_risk="the additional instruction may add latency",
                falsifier="the candidate trace still omits verification",
            ),
        )

    def propose(
        self,
        *,
        feedback: StrategyFeedbackView,
        feedback_ref: ArtifactRef,
        search_run_ref: ArtifactRef,
        round_index: int,
        parent_harness_ref: ArtifactRef,
        diagnosis_refs: tuple[ArtifactRef, ...],
        artifacts: StrategyArtifactView,
    ) -> object:
        if self.baseline_kind is BaselineKind.PROMPT_ONLY:
            evidence_ref = feedback.exploration_aggregates_ref
            diagnosis_ref = None
        elif self.baseline_kind is BaselineKind.EVIDENCE_TARGETED:
            if len(diagnosis_refs) != 1:
                raise RuntimeError("evidence-targeted fixture requires one diagnosis")
            evidence_ref = diagnosis_refs[0]
            diagnosis_ref = diagnosis_refs[0]
        else:
            raise RuntimeError("only optimizer replay cells invoke the plugin")
        if evidence_ref is None:
            raise RuntimeError("optimizer fixture has no readable evidence")

        # Evidence access precedes both writes in the capability-derived access log.
        artifacts.read_json(evidence_ref)
        parent = HarnessManifest.model_validate(
            artifacts.read_json(parent_harness_ref),
            strict=False,
        )
        before = next(
            component
            for component in parent.components
            if component.name == PROMPT_COMPONENT_NAME and component.kind is ComponentKind.PROMPT
        )
        after_prompt_ref = artifacts.put_prompt(
            "Answer the task, then independently verify the result before returning it. "
            f"Fixture run {search_run_ref.sha256[:16]}."
        )
        hypothesis_ref = artifacts.put_hypothesis(
            MutationHypothesis(
                evidence_refs=(evidence_ref,),
                where="the system prompt verification instruction",
                why="the cited exploration evidence shows missing verification",
                expected_activation="the replacement prompt is active in candidate traces",
                expected_adherence="candidate answers include an independent check",
                expected_behavior="fixture verification failures are reduced",
                expected_benefit="benchmark-score reliability improves",
                protected_slices=("already-correct",),
                falsifier="candidate traces still omit an independent check",
                negative_control="unrelated formatting behavior remains unchanged",
                risks=("additional latency",),
            )
        )
        proposal = PromptProposal(
            proposal_id=(
                f"fixture-{self.baseline_kind.value}-{search_run_ref.sha256[:16]}-{round_index}"
            ),
            baseline_kind=self.baseline_kind,
            round_index=round_index,
            parent_harness_ref=parent_harness_ref,
            target_component_name=PROMPT_COMPONENT_NAME,
            before_prompt_ref=before.artifact,
            after_prompt_ref=after_prompt_ref,
            hypothesis_ref=hypothesis_ref,
            mechanism_family="explicit-self-verification",
            diagnosis_ref=diagnosis_ref,
            proposer_confidence=0.5,
        )
        return ProposalBatch(
            baseline_kind=self.baseline_kind,
            round_index=round_index,
            source_feedback_ref=feedback_ref,
            diagnosis_refs=diagnosis_refs,
            proposals=(proposal,),
        )


__all__ = ["ReplayRuntime", "ReplayStrategyPlugin"]
