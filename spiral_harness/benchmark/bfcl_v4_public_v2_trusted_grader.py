"""Trusted control plane for the BFCL V4 public-development v2 grader."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from spiral_harness.benchmark.bfcl_v4_public_development_v2 import (
    load_bfcl_v4_public_development_v2,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan import (
    build_bfcl_v4_public_development_v2_campaign_plan,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_call_plan_contracts import (
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
    BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256,
    BfclV4PublicDevelopmentV2NodeKind,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_campaign_contracts import (
    BfclV4PublicDevelopmentV2CampaignPlan,
    derive_bfcl_v4_public_development_v2_node_request_lineage,
)
from spiral_harness.benchmark.bfcl_v4_public_development_v2_contracts import (
    BfclV4LoadedPublicDevelopmentV2,
    BfclV4PublicDevelopmentV2Split,
    BfclV4PublicDevelopmentV2Task,
)
from spiral_harness.benchmark.bfcl_v4_public_v2_trusted_grader_contracts import (
    BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256,
    BFCL_V4_PUBLIC_V2_EVALUATION_AUTHORITY_KEY_DOMAIN,
    BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_HMAC_DOMAIN,
    BFCL_V4_PUBLIC_V2_MINIMUM_EVALUATION_AUTHORITY_SECRET_BYTES,
    BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_PROTOCOL,
    BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256,
    BfclV4PublicV2DecisionBarrierEvidence,
    BfclV4PublicV2EvaluationUnlock,
    BfclV4PublicV2TrustedGradeRequest,
    BfclV4PublicV2TrustedGraderReceipt,
)
from spiral_harness.core.canonical import (
    canonical_json_bytes,
    canonical_sha256,
    sha256_bytes,
)
from spiral_harness.core.models import Sha256

_WORKER_UPSTREAM_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
_WORKER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WORKER_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
}


class BfclV4PublicV2TrustedGraderError(RuntimeError):
    """Fail-closed trusted-loader or isolated-worker failure."""


class BfclV4PublicV2EvaluationAuthorizationError(RuntimeError):
    """The global decision barrier lacks authentic HOLDOUT authority."""


def _raw_model_content(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return {name: _raw_model_content(getattr(value, name)) for name in type(value).model_fields}
    if isinstance(value, Mapping):
        return {key: _raw_model_content(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_raw_model_content(item) for item in value)
    if isinstance(value, list):
        return [_raw_model_content(item) for item in value]
    return value


def _checked[ModelT: BaseModel](value: ModelT, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate(_raw_model_content(value), strict=True)
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ValueError("trusted grader contract revalidation failed") from error


def _checkout_and_git(checkout: str | Path) -> tuple[Path, Path]:
    try:
        resolved = Path(checkout).resolve(strict=True)
        git_name = shutil.which("git", path=os.defpath)
        if not resolved.is_dir() or git_name is None:
            raise OSError
        git = Path(git_name).resolve(strict=True)
        completed = subprocess.run(
            [str(git), "-C", str(resolved), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            timeout=20,
        )
        head = completed.stdout.decode("ascii").strip()
    except (OSError, subprocess.TimeoutExpired, UnicodeDecodeError) as error:
        raise BfclV4PublicV2TrustedGraderError(
            "trusted BFCL checkout verification failed"
        ) from error
    if completed.returncode != 0 or completed.stderr or head != _WORKER_UPSTREAM_COMMIT:
        raise BfclV4PublicV2TrustedGraderError("trusted BFCL checkout verification failed")
    return resolved, git


def _strict_worker_output(stdout: bytes) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8", errors="strict")
        if not text.endswith("\n") or "\n" in text[:-1]:
            raise ValueError
        value = json.loads(text[:-1])
        if not isinstance(value, dict) or canonical_json_bytes(value) != stdout[:-1]:
            raise ValueError
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BfclV4PublicV2TrustedGraderError(
            "isolated BFCL grader returned an invalid receipt"
        ) from error
    return value


def _verify_local_worker_sources() -> Path:
    worker = (
        Path(__file__).with_name("_bfcl_v4_public_v2_trusted_grader_worker.py").resolve(strict=True)
    )
    primitives = Path(__file__).with_name("_bfcl_v4_public_grader_worker.py").resolve(strict=True)
    try:
        observed = (
            sha256_bytes(worker.read_bytes()),
            sha256_bytes(primitives.read_bytes()),
        )
    except OSError as error:
        raise BfclV4PublicV2TrustedGraderError(
            "trusted grader worker source binding failed"
        ) from error
    if observed != (
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_SOURCE_SHA256,
        BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_PRIMITIVES_SOURCE_SHA256,
    ):
        raise BfclV4PublicV2TrustedGraderError("trusted grader worker source binding failed")
    return worker


def _authority_key_id(secret: bytes) -> str:
    return hmac.new(
        secret,
        BFCL_V4_PUBLIC_V2_EVALUATION_AUTHORITY_KEY_DOMAIN.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _unlock_authentication_content(
    *,
    barrier: BfclV4PublicV2DecisionBarrierEvidence,
    authority_key_id: str,
) -> dict[str, Any]:
    return {
        "authority_key_id": authority_key_id,
        "barrier_evidence": barrier,
        "barrier_evidence_fingerprint": barrier.fingerprint,
        "domain": BFCL_V4_PUBLIC_V2_EVALUATION_UNLOCK_HMAC_DOMAIN,
    }


def _unlock_tag(
    secret: bytes,
    *,
    barrier: BfclV4PublicV2DecisionBarrierEvidence,
    authority_key_id: str,
) -> str:
    return hmac.new(
        secret,
        canonical_json_bytes(
            _unlock_authentication_content(
                barrier=barrier,
                authority_key_id=authority_key_id,
            )
        ),
        hashlib.sha256,
    ).hexdigest()


def _task_reference_sha256(task_ref: str, task_payload_sha256: str) -> str:
    return canonical_sha256(
        {
            "domain": "spiral-bfcl-v4-public-development-v2-structural-task-reference/v1",
            "campaign_plan_fingerprint": BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT,
            "task_ref": task_ref,
            "task_payload_sha256": task_payload_sha256,
        }
    )


class BfclV4PublicV2TrustedGrader:
    """Trusted control plane holding questions, campaign, and authority state."""

    __slots__ = (
        "_authority_key_id",
        "_campaign",
        "_checkout",
        "_decision_node_references",
        "_evaluation_authority_secret",
        "_git",
        "_loaded_bundle_fingerprint",
        "_nodes_by_id",
        "_semantic_release_fingerprint",
        "_source_sha256",
        "_tasks_by_split",
    )

    def __init__(
        self,
        checkout: str | Path,
        campaign: BfclV4PublicDevelopmentV2CampaignPlan | None = None,
        *,
        evaluation_authority_secret: bytes | None = None,
        semantic_release_fingerprint: Sha256 | None = None,
    ) -> None:
        if (evaluation_authority_secret is None) is not (semantic_release_fingerprint is None):
            raise ValueError(
                "evaluation authority secret and semantic release must be configured together"
            )
        if evaluation_authority_secret is not None and (
            type(evaluation_authority_secret) is not bytes
            or len(evaluation_authority_secret)
            < BFCL_V4_PUBLIC_V2_MINIMUM_EVALUATION_AUTHORITY_SECRET_BYTES
        ):
            raise ValueError("evaluation authority secret is too short")
        if semantic_release_fingerprint is not None and (
            not isinstance(semantic_release_fingerprint, str)
            or _WORKER_SHA256.fullmatch(semantic_release_fingerprint) is None
        ):
            raise ValueError("semantic release fingerprint is invalid")

        resolved, git = _checkout_and_git(checkout)
        prospective_campaign = (
            build_bfcl_v4_public_development_v2_campaign_plan() if campaign is None else campaign
        )
        checked_campaign = _checked(
            prospective_campaign,
            BfclV4PublicDevelopmentV2CampaignPlan,
        )
        if (
            checked_campaign.fingerprint != BFCL_V4_PUBLIC_DEVELOPMENT_V2_CAMPAIGN_FINGERPRINT
            or checked_campaign.node_schedule_content_sha256
            != BFCL_V4_PUBLIC_DEVELOPMENT_V2_NODE_SCHEDULE_SHA256
        ):
            raise ValueError("trusted grader campaign differs from the frozen plan")
        try:
            loaded = _checked(
                load_bfcl_v4_public_development_v2(resolved),
                BfclV4LoadedPublicDevelopmentV2,
            )
        except Exception as error:
            raise BfclV4PublicV2TrustedGraderError(
                "trusted BFCL question bundle verification failed"
            ) from error
        if loaded.manifest.fingerprint != checked_campaign.manifest_fingerprint:
            raise ValueError("trusted question bundle and campaign manifest differ")

        tasks_by_split: dict[
            BfclV4PublicDevelopmentV2Split,
            tuple[BfclV4PublicDevelopmentV2Task, ...],
        ] = {}
        for split in BfclV4PublicDevelopmentV2Split:
            split_tasks = tuple(
                task
                for task, entry in zip(loaded.tasks, loaded.manifest.roster, strict=True)
                if entry.split is split
            )
            expected_count = {
                BfclV4PublicDevelopmentV2Split.FIT: 5,
                BfclV4PublicDevelopmentV2Split.GATE: 4,
                BfclV4PublicDevelopmentV2Split.HOLDOUT: 16,
            }[split]
            if len(split_tasks) != expected_count:
                raise ValueError("trusted question bundle split population changed")
            tasks_by_split[split] = split_tasks

        nodes_by_id = {node.node_id: node for node in checked_campaign.nodes}
        if len(nodes_by_id) != len(checked_campaign.nodes):
            raise ValueError("trusted campaign repeats a node")
        decision_nodes = tuple(
            node
            for node in checked_campaign.nodes
            if node.kind is BfclV4PublicDevelopmentV2NodeKind.GATE_DECISION
        )
        if len(decision_nodes) != 6:
            raise ValueError("trusted campaign lacks the exact decision barrier")

        try:
            source_sha256 = sha256_bytes(Path(__file__).resolve(strict=True).read_bytes())
        except OSError as error:
            raise BfclV4PublicV2TrustedGraderError(
                "trusted grader source binding failed"
            ) from error
        _verify_local_worker_sources()
        self._checkout = resolved
        self._git = git
        self._campaign = checked_campaign
        self._nodes_by_id = nodes_by_id
        self._tasks_by_split = tasks_by_split
        self._loaded_bundle_fingerprint = canonical_sha256(loaded)
        self._source_sha256 = source_sha256
        self._evaluation_authority_secret = evaluation_authority_secret
        self._semantic_release_fingerprint = semantic_release_fingerprint
        self._authority_key_id = (
            None
            if evaluation_authority_secret is None
            else _authority_key_id(evaluation_authority_secret)
        )
        self._decision_node_references = tuple(canonical_sha256(node) for node in decision_nodes)

    @property
    def loaded_question_bundle_fingerprint(self) -> str:
        """Answer-free identity safe for orchestration and auditing."""

        return self._loaded_bundle_fingerprint

    @property
    def campaign_plan_fingerprint(self) -> str:
        return self._campaign.fingerprint

    @property
    def decision_node_references(self) -> tuple[str, ...]:
        """Exact ordered six-node barrier; contains no decisions or scores."""

        return self._decision_node_references

    def issue_evaluation_unlock(
        self,
        evidence: BfclV4PublicV2DecisionBarrierEvidence,
    ) -> BfclV4PublicV2EvaluationUnlock:
        """Authenticate replayed global-barrier evidence inside the trusted plane."""

        if (
            self._evaluation_authority_secret is None
            or self._semantic_release_fingerprint is None
            or self._authority_key_id is None
        ):
            raise BfclV4PublicV2EvaluationAuthorizationError(
                "HOLDOUT evaluation authority is not configured"
            )
        checked = _checked(evidence, BfclV4PublicV2DecisionBarrierEvidence)
        if (
            checked.campaign_plan_fingerprint != self._campaign.fingerprint
            or checked.node_schedule_content_sha256 != self._campaign.node_schedule_content_sha256
            or checked.semantic_release_fingerprint != self._semantic_release_fingerprint
            or checked.decision_node_references != self._decision_node_references
        ):
            raise BfclV4PublicV2EvaluationAuthorizationError(
                "evaluation barrier does not match trusted authority state"
            )
        return BfclV4PublicV2EvaluationUnlock(
            barrier_evidence=checked,
            barrier_evidence_fingerprint=checked.fingerprint,
            authority_key_id=self._authority_key_id,
            authentication_tag_hmac_sha256=_unlock_tag(
                self._evaluation_authority_secret,
                barrier=checked,
                authority_key_id=self._authority_key_id,
            ),
        )

    def _resolve_task(
        self,
        task_ref: str,
    ) -> tuple[BfclV4PublicDevelopmentV2Split, BfclV4PublicDevelopmentV2Task]:
        match = re.fullmatch(r"(fit|gate|holdout)-([0-9]{2})", task_ref)
        if match is None:
            raise ValueError("model-call node has an invalid structural task reference")
        split = BfclV4PublicDevelopmentV2Split(match.group(1))
        ordinal = int(match.group(2))
        tasks = self._tasks_by_split[split]
        if ordinal >= len(tasks):
            raise ValueError("structural task reference is outside its frozen split")
        return split, tasks[ordinal]

    def _authorize_split(
        self,
        split: BfclV4PublicDevelopmentV2Split,
        unlock: BfclV4PublicV2EvaluationUnlock | None,
    ) -> BfclV4PublicV2EvaluationUnlock | None:
        if split is not BfclV4PublicDevelopmentV2Split.HOLDOUT:
            if unlock is not None:
                raise ValueError("non-HOLDOUT grading cannot carry an evaluation unlock")
            return None
        if unlock is None:
            raise BfclV4PublicV2EvaluationAuthorizationError(
                "HOLDOUT grading requires an authenticated global decision barrier"
            )
        if (
            self._evaluation_authority_secret is None
            or self._semantic_release_fingerprint is None
            or self._authority_key_id is None
        ):
            raise BfclV4PublicV2EvaluationAuthorizationError(
                "HOLDOUT evaluation authority is not configured"
            )
        checked = _checked(unlock, BfclV4PublicV2EvaluationUnlock)
        barrier = checked.barrier_evidence
        expected_tag = _unlock_tag(
            self._evaluation_authority_secret,
            barrier=barrier,
            authority_key_id=self._authority_key_id,
        )
        if (
            checked.authority_key_id != self._authority_key_id
            or not hmac.compare_digest(
                checked.authentication_tag_hmac_sha256,
                expected_tag,
            )
            or barrier.campaign_plan_fingerprint != self._campaign.fingerprint
            or barrier.node_schedule_content_sha256 != self._campaign.node_schedule_content_sha256
            or barrier.semantic_release_fingerprint != self._semantic_release_fingerprint
            or barrier.decision_node_references != self._decision_node_references
        ):
            raise BfclV4PublicV2EvaluationAuthorizationError(
                "HOLDOUT evaluation unlock authentication failed"
            )
        return checked

    def _run_worker(
        self,
        *,
        task: BfclV4PublicDevelopmentV2Task,
        request: BfclV4PublicV2TrustedGradeRequest,
    ) -> bool:
        calls = tuple(
            {
                "arguments": json.loads(call.canonical_arguments_json),
                "function_name": call.official_name,
            }
            for call in request.raw_response.tool_calls
        )
        payload = {
            "calls": calls,
            "candidate_payload_sha256": task.candidate_payload_sha256,
            "official_function_names": task.official_function_names,
            "protocol": BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_PROTOCOL,
            "task_id": task.task_id,
        }
        worker = _verify_local_worker_sources()
        python = Path(sys.executable).resolve(strict=True)
        argv = (
            str(python),
            "-I",
            "-B",
            str(worker),
            "--checkout",
            str(self._checkout),
            "--git",
            str(self._git),
        )
        try:
            completed = subprocess.run(
                argv,
                input=canonical_json_bytes(payload),
                env=dict(_WORKER_ENVIRONMENT),
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise BfclV4PublicV2TrustedGraderError(
                "isolated BFCL grader execution failed"
            ) from error
        if completed.returncode != 0 or completed.stderr:
            raise BfclV4PublicV2TrustedGraderError("isolated BFCL grader rejected the invocation")
        output = _strict_worker_output(completed.stdout)
        if set(output) != {
            "answer_data_present",
            "candidate_payload_sha256",
            "checker_diagnostics_present",
            "checker_source_bundle_sha256",
            "correct",
            "protocol",
        }:
            raise BfclV4PublicV2TrustedGraderError("isolated BFCL grader receipt schema changed")
        if (
            output["protocol"] != BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_WORKER_PROTOCOL
            or output["candidate_payload_sha256"] != task.candidate_payload_sha256
            or output["checker_source_bundle_sha256"]
            != BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256
            or output["answer_data_present"] is not False
            or output["checker_diagnostics_present"] is not False
            or not isinstance(output["correct"], bool)
        ):
            raise BfclV4PublicV2TrustedGraderError("isolated BFCL grader receipt binding changed")
        return output["correct"]

    def grade(
        self,
        request: BfclV4PublicV2TrustedGradeRequest,
        *,
        evaluation_unlock: BfclV4PublicV2EvaluationUnlock | None = None,
    ) -> BfclV4PublicV2TrustedGraderReceipt:
        """Grade one exact response and project only a lineage-bound boolean."""

        checked = _checked(request, BfclV4PublicV2TrustedGradeRequest)
        registered_node = self._nodes_by_id.get(checked.node.node_id)
        if registered_node is None or registered_node != checked.node:
            raise ValueError("trusted grade node differs from the frozen campaign")
        expected_lineage = derive_bfcl_v4_public_development_v2_node_request_lineage(
            campaign=self._campaign,
            node_id=registered_node.node_id,
        )
        if checked.request_lineage != expected_lineage:
            raise ValueError("trusted grade lineage differs from the frozen campaign")
        assert registered_node.task_ref is not None
        split, task = self._resolve_task(registered_node.task_ref)
        expected_kinds = {
            BfclV4PublicDevelopmentV2Split.FIT: {
                BfclV4PublicDevelopmentV2NodeKind.PARENT_FIT,
                BfclV4PublicDevelopmentV2NodeKind.CANDIDATE_FIT,
            },
            BfclV4PublicDevelopmentV2Split.GATE: {
                BfclV4PublicDevelopmentV2NodeKind.GATE,
            },
            BfclV4PublicDevelopmentV2Split.HOLDOUT: {
                BfclV4PublicDevelopmentV2NodeKind.HOLDOUT,
                BfclV4PublicDevelopmentV2NodeKind.PURE_AT_B_SAMPLE,
            },
        }[split]
        if registered_node.kind not in expected_kinds:
            raise ValueError("model-call node kind differs from its task split")
        if checked.task_payload_sha256 != task.candidate_payload_sha256:
            raise ValueError("trusted grade request points to another task payload")
        if (
            checked.request.requested_model != self._campaign.execution_profile.model_route
            or checked.request.inference != self._campaign.execution_profile.inference
        ):
            raise ValueError("native request differs from the frozen same-model profile")
        if checked.raw_response.usage.total_tokens > (
            self._campaign.execution_profile.per_call_total_token_ceiling
        ):
            raise ValueError("native response exceeds the frozen total-token ceiling")
        if (
            tuple(tool.official_name for tool in checked.request.task_required_tools)
            != task.official_function_names
        ):
            raise ValueError("native request task tools differ from the frozen task")

        authorized_unlock = self._authorize_split(split, evaluation_unlock)
        correct = self._run_worker(task=task, request=checked)
        return BfclV4PublicV2TrustedGraderReceipt(
            node_reference_sha256=checked.node_reference_sha256,
            campaign_call_slot=registered_node.campaign_call_slot,
            task_ref=registered_node.task_ref,
            task_reference_sha256=_task_reference_sha256(
                registered_node.task_ref,
                task.candidate_payload_sha256,
            ),
            split_role=split,
            request_fingerprint=checked.request_fingerprint,
            response_fingerprint=checked.response_fingerprint,
            evaluation_unlock_fingerprint=(
                None if authorized_unlock is None else authorized_unlock.fingerprint
            ),
            loaded_question_bundle_fingerprint=self._loaded_bundle_fingerprint,
            trusted_authority_fingerprint=BFCL_V4_PUBLIC_V2_TRUSTED_AUTHORITY_FINGERPRINT,
            trusted_grader_implementation_fingerprint=(
                BFCL_V4_PUBLIC_V2_TRUSTED_GRADER_IMPLEMENTATION_FINGERPRINT
            ),
            grader_source_sha256=self._source_sha256,
            checker_source_bundle_sha256=BFCL_V4_PUBLIC_V2_CHECKER_SOURCE_BUNDLE_SHA256,
            correct=correct,
        )


def open_bfcl_v4_public_v2_trusted_grader(
    checkout: str | Path,
    campaign: BfclV4PublicDevelopmentV2CampaignPlan | None = None,
    *,
    evaluation_authority_secret: bytes | None = None,
    semantic_release_fingerprint: Sha256 | None = None,
) -> BfclV4PublicV2TrustedGrader:
    """Open the answer-isolated trusted control plane from a pinned checkout."""

    return BfclV4PublicV2TrustedGrader(
        checkout,
        campaign,
        evaluation_authority_secret=evaluation_authority_secret,
        semantic_release_fingerprint=semantic_release_fingerprint,
    )


def grade_bfcl_v4_public_v2_response(
    grader: BfclV4PublicV2TrustedGrader,
    request: BfclV4PublicV2TrustedGradeRequest,
    *,
    evaluation_unlock: BfclV4PublicV2EvaluationUnlock | None = None,
) -> BfclV4PublicV2TrustedGraderReceipt:
    """Functional facade for orchestration code that owns the trusted grader."""

    if not isinstance(grader, BfclV4PublicV2TrustedGrader):
        raise TypeError("grader must be a BFCL v2 trusted grader")
    return grader.grade(request, evaluation_unlock=evaluation_unlock)


__all__ = [
    "BfclV4PublicV2EvaluationAuthorizationError",
    "BfclV4PublicV2TrustedGrader",
    "BfclV4PublicV2TrustedGraderError",
    "grade_bfcl_v4_public_v2_response",
    "open_bfcl_v4_public_v2_trusted_grader",
]
