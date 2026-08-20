"""Sequential, fail-closed executor for the frozen three-seed BFCL campaign."""

from __future__ import annotations

from pathlib import Path

from spiral_harness.benchmark.bfcl_v4_public_pilot_campaign import (
    BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT,
    BfclV4PublicPilotCampaign,
    BfclV4PublicPilotCampaignReplicate,
    build_bfcl_v4_public_pilot_campaign,
)
from spiral_harness.core.canonical import canonical_sha256
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.native_function_execution import NativeFunctionBackend
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis import (
    compute_bfcl_v4_public_campaign_descriptive_analysis,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_analysis_contracts import (
    BfclV4CampaignPerCallBudget,
    BfclV4CampaignReplicateAnalysisInput,
    BfclV4PublicCampaignAnalysisInput,
    BfclV4PublicCampaignDescriptiveAnalysis,
)
from spiral_harness.experiments.bfcl_v4_public_campaign_executor_contracts import (
    BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE,
    BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE,
    BfclV4CampaignExecutionCheckpoint,
    BfclV4CampaignExecutionFailure,
    BfclV4CampaignExecutionStatus,
    BfclV4CampaignFailureStage,
    BfclV4CampaignVerifiedReplicate,
    BfclV4PublicCampaignExecutionRecord,
    BfclV4PublicCampaignExecutionResult,
    BfclV4PublicCampaignExecutionVerification,
)
from spiral_harness.experiments.bfcl_v4_public_live_config import (
    BfclV4PublicLiveExecutionConfig,
)
from spiral_harness.experiments.bfcl_v4_public_runner import (
    run_bfcl_v4_public_pilot_replicate,
)
from spiral_harness.experiments.bfcl_v4_public_runner_contracts import (
    BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
    BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
    BfclV4PublicPilotRunResult,
    BfclV4PublicPilotRunVerification,
)
from spiral_harness.experiments.bfcl_v4_public_runner_support import (
    load_canonical_model,
    publish_model,
)
from spiral_harness.experiments.bfcl_v4_public_runner_verification import (
    verify_bfcl_v4_public_pilot_result,
)
from spiral_harness.experiments.bfcl_v4_public_selection_contracts import (
    BfclV4JointSelectionDecision,
    BfclV4PublicDescriptiveMetrics,
)
from spiral_harness.storage.protocol import ArtifactRepository


class BfclV4PublicCampaignExecutorError(ValueError):
    """Raised for invalid preflight inputs or a forged terminal evidence graph."""


def _backend_coordinates(backend: NativeFunctionBackend) -> tuple[str, str, str, str]:
    try:
        return (
            backend.fingerprint,
            backend.serializer_fingerprint,
            backend.parser_fingerprint,
            backend.transport_fingerprint,
        )
    except Exception as exc:
        raise BfclV4PublicCampaignExecutorError(
            "native backend identities could not be read"
        ) from exc


def _require_backend_binding(
    backend: NativeFunctionBackend,
    config: BfclV4PublicLiveExecutionConfig,
) -> None:
    expected = (
        config.backend_fingerprint,
        config.serializer_fingerprint,
        config.parser_fingerprint,
        config.transport_fingerprint,
    )
    if _backend_coordinates(backend) != expected:
        raise BfclV4PublicCampaignExecutorError(
            "backend identities differ from the frozen live execution config"
        )


def _load_result(
    repository: ArtifactRepository,
    ref: ArtifactRef,
) -> BfclV4PublicPilotRunResult:
    return load_canonical_model(
        repository,
        ref,
        BfclV4PublicPilotRunResult,
        media_type=BFCL_V4_RUNNER_RESULT_MEDIA_TYPE,
    )


def _verified_binding(
    repository: ArtifactRepository,
    *,
    config: BfclV4PublicLiveExecutionConfig,
    replicate: BfclV4PublicPilotCampaignReplicate,
    result_ref: ArtifactRef,
    verification: BfclV4PublicPilotRunVerification,
    verification_ref: ArtifactRef,
) -> BfclV4CampaignVerifiedReplicate:
    result = _load_result(repository, result_ref)
    expected = (
        replicate.outer_seed_u64,
        replicate.call_plan.fingerprint,
        replicate.call_plan.schedule_content_sha256,
        config.model_spec,
        config.attempt_budget,
        result.fingerprint,
        result.plan_fingerprint,
    )
    observed = (
        result.outer_seed_u64,
        result.plan_fingerprint,
        result.schedule_content_sha256,
        result.model_spec,
        result.attempt_budget,
        verification.result_fingerprint,
        verification.plan_fingerprint,
    )
    if observed != expected:
        raise BfclV4PublicCampaignExecutorError(
            "verified replicate differs from its campaign, config, or verifier"
        )
    return BfclV4CampaignVerifiedReplicate(
        ordinal=replicate.ordinal,
        replicate_id=replicate.replicate_id,
        outer_seed_u64=replicate.outer_seed_u64,
        plan_fingerprint=result.plan_fingerprint,
        schedule_content_sha256=result.schedule_content_sha256,
        attempt_ledger_id=result.attempt_ledger_id,
        model_spec_fingerprint=result.model_spec.fingerprint,
        backend_fingerprint=result.model_spec.backend_fingerprint,
        inference_fingerprint=result.model_spec.inference_fingerprint,
        attempt_budget_fingerprint=result.attempt_budget.fingerprint,
        run_result_ref=result_ref,
        run_result_fingerprint=result.fingerprint,
        verification_ref=verification_ref,
        verification_fingerprint=canonical_sha256(verification),
        closure_ref=result.journal_closure_ref,
        joint_selection_decision_ref=result.joint_selection_decision_ref,
        descriptive_metrics_ref=result.descriptive_metrics_ref,
        provider_attempts_succeeded=result.provider_attempts_succeeded,
        provider_attempts_failed=result.provider_attempts_failed,
        provider_identity_observation_count=result.provider_identity_observation_count,
        provider_declared_identity_consistent=result.provider_declared_identity_consistent,
    )


def _load_and_reverify_binding(
    repository: ArtifactRepository,
    *,
    config: BfclV4PublicLiveExecutionConfig,
    replicate: BfclV4PublicPilotCampaignReplicate,
    binding: BfclV4CampaignVerifiedReplicate,
) -> BfclV4PublicPilotRunResult:
    verification = verify_bfcl_v4_public_pilot_result(repository, binding.run_result_ref)
    stored_verification = load_canonical_model(
        repository,
        binding.verification_ref,
        BfclV4PublicPilotRunVerification,
        media_type=BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE,
    )
    if verification != stored_verification:
        raise BfclV4PublicCampaignExecutorError(
            "stored replicate verification differs from fresh offline replay"
        )
    rebuilt = _verified_binding(
        repository,
        config=config,
        replicate=replicate,
        result_ref=binding.run_result_ref,
        verification=verification,
        verification_ref=binding.verification_ref,
    )
    if rebuilt != binding:
        raise BfclV4PublicCampaignExecutorError(
            "verified replicate binding differs from its result-ref projection"
        )
    return _load_result(repository, binding.run_result_ref)


def _derive_analysis_input(
    repository: ArtifactRepository,
    *,
    campaign: BfclV4PublicPilotCampaign,
    config: BfclV4PublicLiveExecutionConfig,
    bindings: tuple[BfclV4CampaignVerifiedReplicate, ...],
) -> BfclV4PublicCampaignAnalysisInput:
    if len(bindings) != 3:
        raise BfclV4PublicCampaignExecutorError(
            "campaign analysis requires three verified result refs"
        )
    per_call_budget = BfclV4CampaignPerCallBudget(
        max_output_tokens=config.model_spec.inference.max_output_tokens,
        attempt_budget=config.attempt_budget,
        attempt_budget_fingerprint=config.attempt_budget.fingerprint,
    )
    projected = []
    for replicate, binding in zip(campaign.replicates, bindings, strict=True):
        result = _load_and_reverify_binding(
            repository,
            config=config,
            replicate=replicate,
            binding=binding,
        )
        decision = load_canonical_model(
            repository,
            result.joint_selection_decision_ref,
            BfclV4JointSelectionDecision,
            media_type=BFCL_V4_RUNNER_SELECTION_DECISION_MEDIA_TYPE,
        )
        metrics = load_canonical_model(
            repository,
            result.descriptive_metrics_ref,
            BfclV4PublicDescriptiveMetrics,
            media_type=BFCL_V4_RUNNER_METRICS_MEDIA_TYPE,
        )
        if (
            result.journal_closure_ref != binding.closure_ref
            or result.joint_selection_decision_ref != binding.joint_selection_decision_ref
            or result.descriptive_metrics_ref != binding.descriptive_metrics_ref
        ):
            raise BfclV4PublicCampaignExecutorError(
                "analysis evidence refs differ from the verified runner result"
            )
        projected.append(
            BfclV4CampaignReplicateAnalysisInput(
                ordinal=replicate.ordinal,
                replicate_id=replicate.replicate_id,
                outer_seed_u64=replicate.outer_seed_u64,
                plan_fingerprint=result.plan_fingerprint,
                schedule_content_sha256=result.schedule_content_sha256,
                model_spec_fingerprint=result.model_spec.fingerprint,
                backend_fingerprint=result.model_spec.backend_fingerprint,
                inference_fingerprint=result.model_spec.inference_fingerprint,
                attempt_budget=result.attempt_budget,
                attempt_budget_fingerprint=result.attempt_budget.fingerprint,
                per_call_budget_fingerprint=per_call_budget.fingerprint,
                closure_ref=result.journal_closure_ref,
                closure_verification=result.closure_verification,
                joint_selection=decision,
                joint_selection_fingerprint=decision.fingerprint,
                descriptive_metrics=metrics,
                descriptive_metrics_fingerprint=metrics.fingerprint,
                provider_identity_observation_count=(result.provider_identity_observation_count),
                provider_declared_identity_consistent=(
                    result.provider_declared_identity_consistent
                ),
            )
        )
    replicates = tuple(projected)
    return BfclV4PublicCampaignAnalysisInput(
        campaign=campaign,
        campaign_fingerprint=campaign.fingerprint,
        model_spec=config.model_spec,
        model_spec_fingerprint=config.model_spec.fingerprint,
        backend=config.model_spec.backend,
        backend_fingerprint=config.backend_fingerprint,
        inference_fingerprint=config.inference_fingerprint,
        per_call_budget=per_call_budget,
        per_call_budget_fingerprint=per_call_budget.fingerprint,
        ordered_closure_refs=tuple(item.closure_ref for item in replicates),
        replicates=replicates,
        all_call_response_identity_coverage_complete=all(
            item.provider_identity_observation_count == 100 for item in replicates
        ),
    )


class BfclV4PublicCampaignExecutor:
    """Single-use executor; provider failures consume slots, infrastructure failures close."""

    def __init__(
        self,
        repository: ArtifactRepository,
        *,
        checkout: str | Path,
        live_config: BfclV4PublicLiveExecutionConfig,
        backend: NativeFunctionBackend,
    ) -> None:
        self.repository = repository
        self.checkout = Path(checkout)
        self.config = BfclV4PublicLiveExecutionConfig.model_validate(live_config, strict=True)
        if not isinstance(backend, NativeFunctionBackend):
            raise TypeError("backend must implement NativeFunctionBackend")
        _require_backend_binding(backend, self.config)
        self.backend = backend
        self.campaign = build_bfcl_v4_public_pilot_campaign()
        self._started = False

    def run(self) -> BfclV4PublicCampaignExecutionRecord:
        if self._started:
            raise BfclV4PublicCampaignExecutorError("campaign executor is single-use")
        self._started = True
        campaign_ref = publish_model(
            self.repository,
            self.campaign,
            media_type=BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE,
        )
        config_ref = publish_model(
            self.repository,
            self.config,
            media_type=BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE,
        )
        completed: list[BfclV4CampaignVerifiedReplicate] = []
        checkpoints: list[ArtifactRef] = []

        for replicate in self.campaign.replicates:
            stage = BfclV4CampaignFailureStage.REPLICATE_EXECUTION
            unverified_ref: ArtifactRef | None = None
            unverified_fingerprint: str | None = None
            try:
                _require_backend_binding(self.backend, self.config)
                record = run_bfcl_v4_public_pilot_replicate(
                    self.repository,
                    checkout=self.checkout,
                    spec=self.config.model_spec,
                    backend=self.backend,
                    attempt_budget=self.config.attempt_budget,
                    outer_seed_u64=replicate.outer_seed_u64,
                    attempt_ledger_id=(f"{self.campaign.campaign_id}/{replicate.replicate_id}"),
                )
                unverified_ref = record.result_ref
                unverified_fingerprint = record.result.fingerprint
                stage = BfclV4CampaignFailureStage.REPLICATE_VERIFICATION
                verification = verify_bfcl_v4_public_pilot_result(
                    self.repository,
                    record.result_ref,
                )
                verification_ref = publish_model(
                    self.repository,
                    verification,
                    media_type=BFCL_V4_CAMPAIGN_REPLICATE_VERIFICATION_MEDIA_TYPE,
                )
                binding = _verified_binding(
                    self.repository,
                    config=self.config,
                    replicate=replicate,
                    result_ref=record.result_ref,
                    verification=verification,
                    verification_ref=verification_ref,
                )
                _require_backend_binding(self.backend, self.config)
                completed.append(binding)
                unverified_ref = None
                unverified_fingerprint = None
                stage = BfclV4CampaignFailureStage.REPLICATE_CHECKPOINT
                checkpoint = self._checkpoint(
                    campaign_ref=campaign_ref,
                    config_ref=config_ref,
                    completed=tuple(completed),
                    previous_ref=checkpoints[-1] if checkpoints else None,
                )
                checkpoints.append(
                    publish_model(
                        self.repository,
                        checkpoint,
                        media_type=BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE,
                    )
                )
            except Exception:
                return self._incomplete(
                    campaign_ref=campaign_ref,
                    config_ref=config_ref,
                    completed=tuple(completed),
                    checkpoints=tuple(checkpoints),
                    stage=stage,
                    active=replicate,
                    unverified_ref=unverified_ref,
                    unverified_fingerprint=unverified_fingerprint,
                )

        analysis_input: BfclV4PublicCampaignAnalysisInput | None = None
        analysis_input_ref: ArtifactRef | None = None
        try:
            analysis_input = _derive_analysis_input(
                self.repository,
                campaign=self.campaign,
                config=self.config,
                bindings=tuple(completed),
            )
            analysis_input_ref = publish_model(
                self.repository,
                analysis_input,
                media_type=BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE,
            )
            analysis = compute_bfcl_v4_public_campaign_descriptive_analysis(analysis_input)
            analysis_ref = publish_model(
                self.repository,
                analysis,
                media_type=BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
            )
        except Exception:
            return self._incomplete(
                campaign_ref=campaign_ref,
                config_ref=config_ref,
                completed=tuple(completed),
                checkpoints=tuple(checkpoints),
                stage=BfclV4CampaignFailureStage.CAMPAIGN_ANALYSIS,
                active=None,
                analysis_input=analysis_input,
                analysis_input_ref=analysis_input_ref,
            )
        result = self._result(
            status=BfclV4CampaignExecutionStatus.COMPLETE,
            campaign_ref=campaign_ref,
            config_ref=config_ref,
            completed=tuple(completed),
            checkpoints=tuple(checkpoints),
            analysis_input=analysis_input,
            analysis_input_ref=analysis_input_ref,
            analysis=analysis,
            analysis_ref=analysis_ref,
        )
        return self._publish_result(result)

    def _checkpoint(
        self,
        *,
        campaign_ref: ArtifactRef,
        config_ref: ArtifactRef,
        completed: tuple[BfclV4CampaignVerifiedReplicate, ...],
        previous_ref: ArtifactRef | None,
    ) -> BfclV4CampaignExecutionCheckpoint:
        count = len(completed)
        return BfclV4CampaignExecutionCheckpoint(
            campaign_fingerprint=self.campaign.fingerprint,
            campaign_ref=campaign_ref,
            live_execution_config_ref=config_ref,
            live_execution_config_fingerprint=self.config.fingerprint,
            model_spec_fingerprint=self.config.model_spec.fingerprint,
            backend_fingerprint=self.config.backend_fingerprint,
            inference_fingerprint=self.config.inference_fingerprint,
            attempt_budget_fingerprint=self.config.attempt_budget.fingerprint,
            completed_replicates=completed,
            previous_checkpoint_ref=previous_ref,
            completed_replicate_count=count,
            verified_closed_model_calls=count * 100,
            next_replicate_ordinal=count if count < 3 else None,
        )

    def _incomplete(
        self,
        *,
        campaign_ref: ArtifactRef,
        config_ref: ArtifactRef,
        completed: tuple[BfclV4CampaignVerifiedReplicate, ...],
        checkpoints: tuple[ArtifactRef, ...],
        stage: BfclV4CampaignFailureStage,
        active: BfclV4PublicPilotCampaignReplicate | None,
        unverified_ref: ArtifactRef | None = None,
        unverified_fingerprint: str | None = None,
        analysis_input: BfclV4PublicCampaignAnalysisInput | None = None,
        analysis_input_ref: ArtifactRef | None = None,
    ) -> BfclV4PublicCampaignExecutionRecord:
        failure = BfclV4CampaignExecutionFailure(
            stage=stage,
            completed_replicate_count=len(completed),
            active_replicate_ordinal=None if active is None else active.ordinal,
            active_replicate_id=None if active is None else active.replicate_id,
            active_outer_seed_u64=None if active is None else active.outer_seed_u64,
            unverified_run_result_ref=unverified_ref,
            unverified_run_result_fingerprint=unverified_fingerprint,
        )
        result = self._result(
            status=BfclV4CampaignExecutionStatus.INCOMPLETE,
            campaign_ref=campaign_ref,
            config_ref=config_ref,
            completed=completed,
            checkpoints=checkpoints,
            analysis_input=analysis_input,
            analysis_input_ref=analysis_input_ref,
            failure=failure,
        )
        return self._publish_result(result)

    def _result(
        self,
        *,
        status: BfclV4CampaignExecutionStatus,
        campaign_ref: ArtifactRef,
        config_ref: ArtifactRef,
        completed: tuple[BfclV4CampaignVerifiedReplicate, ...],
        checkpoints: tuple[ArtifactRef, ...],
        analysis_input: BfclV4PublicCampaignAnalysisInput | None = None,
        analysis_input_ref: ArtifactRef | None = None,
        analysis: BfclV4PublicCampaignDescriptiveAnalysis | None = None,
        analysis_ref: ArtifactRef | None = None,
        failure: BfclV4CampaignExecutionFailure | None = None,
    ) -> BfclV4PublicCampaignExecutionResult:
        return BfclV4PublicCampaignExecutionResult(
            status=status,
            campaign_fingerprint=self.campaign.fingerprint,
            campaign_ref=campaign_ref,
            live_execution_config_ref=config_ref,
            live_execution_config_fingerprint=self.config.fingerprint,
            model_spec_fingerprint=self.config.model_spec.fingerprint,
            backend_fingerprint=self.config.backend_fingerprint,
            inference_fingerprint=self.config.inference_fingerprint,
            attempt_budget_fingerprint=self.config.attempt_budget.fingerprint,
            completed_replicates=completed,
            checkpoint_refs=checkpoints,
            latest_checkpoint_ref=checkpoints[-1] if checkpoints else None,
            verified_closed_model_calls=len(completed) * 100,
            analysis_input_ref=analysis_input_ref,
            analysis_input_fingerprint=(
                None if analysis_input is None else analysis_input.fingerprint
            ),
            analysis_ref=analysis_ref,
            analysis_fingerprint=None if analysis is None else analysis.fingerprint,
            failure=failure,
        )

    def _publish_result(
        self,
        result: BfclV4PublicCampaignExecutionResult,
    ) -> BfclV4PublicCampaignExecutionRecord:
        result_ref = publish_model(
            self.repository,
            result,
            media_type=BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
        )
        return BfclV4PublicCampaignExecutionRecord(result=result, result_ref=result_ref)


def run_bfcl_v4_public_campaign(
    repository: ArtifactRepository,
    *,
    checkout: str | Path,
    live_config: BfclV4PublicLiveExecutionConfig,
    backend: NativeFunctionBackend,
) -> BfclV4PublicCampaignExecutionRecord:
    """Run the exact three-seed campaign once, without retries, skipping, or network setup."""

    return BfclV4PublicCampaignExecutor(
        repository,
        checkout=checkout,
        live_config=live_config,
        backend=backend,
    ).run()


def verify_bfcl_v4_public_campaign_execution(
    repository: ArtifactRepository,
    result_ref: ArtifactRef,
) -> BfclV4PublicCampaignExecutionVerification:
    """Rebuild every admitted projection from runner result refs and recompute analysis."""

    terminal = load_canonical_model(
        repository,
        result_ref,
        BfclV4PublicCampaignExecutionResult,
        media_type=BFCL_V4_CAMPAIGN_EXECUTION_RESULT_MEDIA_TYPE,
    )
    campaign = load_canonical_model(
        repository,
        terminal.campaign_ref,
        BfclV4PublicPilotCampaign,
        media_type=BFCL_V4_CAMPAIGN_REGISTRATION_MEDIA_TYPE,
    )
    config = load_canonical_model(
        repository,
        terminal.live_execution_config_ref,
        BfclV4PublicLiveExecutionConfig,
        media_type=BFCL_V4_CAMPAIGN_LIVE_CONFIG_MEDIA_TYPE,
    )
    if (
        campaign != build_bfcl_v4_public_pilot_campaign()
        or campaign.fingerprint != BFCL_V4_PUBLIC_PILOT_CAMPAIGN_FINGERPRINT
        or config.fingerprint != terminal.live_execution_config_fingerprint
        or terminal.model_spec_fingerprint != config.model_spec.fingerprint
        or terminal.backend_fingerprint != config.backend_fingerprint
        or terminal.inference_fingerprint != config.inference_fingerprint
        or terminal.attempt_budget_fingerprint != config.attempt_budget.fingerprint
    ):
        raise BfclV4PublicCampaignExecutorError(
            "terminal campaign or execution-config lineage changed"
        )
    for replicate, binding in zip(
        campaign.replicates[: len(terminal.completed_replicates)],
        terminal.completed_replicates,
        strict=True,
    ):
        _load_and_reverify_binding(
            repository,
            config=config,
            replicate=replicate,
            binding=binding,
        )
    previous: ArtifactRef | None = None
    for index, checkpoint_ref in enumerate(terminal.checkpoint_refs):
        checkpoint = load_canonical_model(
            repository,
            checkpoint_ref,
            BfclV4CampaignExecutionCheckpoint,
            media_type=BFCL_V4_CAMPAIGN_CHECKPOINT_MEDIA_TYPE,
        )
        if (
            checkpoint.campaign_ref != terminal.campaign_ref
            or checkpoint.live_execution_config_ref != terminal.live_execution_config_ref
            or checkpoint.completed_replicates != terminal.completed_replicates[: index + 1]
            or checkpoint.previous_checkpoint_ref != previous
        ):
            raise BfclV4PublicCampaignExecutorError("campaign checkpoint chain changed")
        previous = checkpoint_ref
    analysis_matched = False
    if terminal.analysis_input_ref is not None:
        derived = _derive_analysis_input(
            repository,
            campaign=campaign,
            config=config,
            bindings=terminal.completed_replicates,
        )
        stored_input = load_canonical_model(
            repository,
            terminal.analysis_input_ref,
            BfclV4PublicCampaignAnalysisInput,
            media_type=BFCL_V4_CAMPAIGN_ANALYSIS_INPUT_MEDIA_TYPE,
        )
        if stored_input != derived or stored_input.fingerprint != (
            terminal.analysis_input_fingerprint
        ):
            raise BfclV4PublicCampaignExecutorError(
                "stored campaign analysis input differs from result-ref derivation"
            )
        if terminal.analysis_ref is not None:
            recomputed = compute_bfcl_v4_public_campaign_descriptive_analysis(derived)
            stored_analysis = load_canonical_model(
                repository,
                terminal.analysis_ref,
                BfclV4PublicCampaignDescriptiveAnalysis,
                media_type=BFCL_V4_CAMPAIGN_ANALYSIS_MEDIA_TYPE,
            )
            if stored_analysis != recomputed or stored_analysis.fingerprint != (
                terminal.analysis_fingerprint
            ):
                raise BfclV4PublicCampaignExecutorError(
                    "stored campaign analysis differs from exact recomputation"
                )
            analysis_matched = True
    return BfclV4PublicCampaignExecutionVerification(
        execution_result_fingerprint=terminal.fingerprint,
        status=terminal.status,
        verified_replicate_count=len(terminal.completed_replicates),
        verified_model_calls=terminal.verified_closed_model_calls,
        verified_checkpoint_count=len(terminal.checkpoint_refs),
        analysis_recomputed_and_matched=analysis_matched,
    )


__all__ = [
    "BfclV4PublicCampaignExecutor",
    "BfclV4PublicCampaignExecutorError",
    "run_bfcl_v4_public_campaign",
    "verify_bfcl_v4_public_campaign_execution",
]
