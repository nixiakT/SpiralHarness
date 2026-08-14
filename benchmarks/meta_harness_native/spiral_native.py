"""Native Meta-Harness adapter template for controlled Spiral comparisons.

Copy this file into upstream ``reference_examples/text_classification/agents/`` and set
``SPIRAL_META_HARNESS_MODE`` to ``paper``, ``local``, ``score-band``, or ``taxonomy``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from spiral_harness.benchmark import meta_harness_paper as core
from spiral_harness.benchmark import meta_harness_spiral as spiral

from ..llm import LLMCallable
from ..memory_system import MemorySystem, extract_json_field
from .no_memory import PROMPT as NO_MEMORY_PROMPT


class SpiralNative(MemorySystem):
    """Bridge Spiral memories into the upstream native ``MemorySystem`` runtime."""

    def __init__(self, llm: LLMCallable):
        super().__init__(llm)
        self.mode = os.environ.get("SPIRAL_META_HARNESS_MODE", "paper")
        if self.mode == "paper":
            self.paper = core.LabelPrimedQueryMemory()
        elif self.mode == "local":
            self.local = spiral.SpiralLocalEvidenceMemory()
        elif self.mode == "score-band":
            self.score = spiral.SpiralCandidateAdjudicationMemory(local_neighbor_k=12)
        elif self.mode == "taxonomy":
            self.paper = core.LabelPrimedQueryMemory()
            self.draft = core.SpiralDraftVerificationMemory()
            self.local = spiral.SpiralLocalEvidenceMemory()
        else:
            raise ValueError(f"unsupported SPIRAL_META_HARNESS_MODE: {self.mode}")

    def _call_for_answer(self, prompt: str, default: str = "") -> tuple[str, str]:
        response = self.call_llm(prompt)
        return extract_json_field(response, "final_answer", default=default), response

    def predict(self, input: str) -> tuple[str, dict[str, Any]]:
        if self.mode == "paper":
            answer, response = self._call_for_answer(self.paper.build_prompt(input))
            return core.clean_classification_answer(answer), {"full_response": response}

        if self.mode == "local":
            answer, response = self._call_for_answer(self.local.build_prompt(input))
            return core.clean_classification_answer(answer), {"full_response": response}

        if self.mode == "score-band":
            local, _ = self._call_for_answer(self.score.local.build_prompt(input))
            paper, _ = self._call_for_answer(self.score.paper.build_prompt(input))
            query = self.score.local.extract_query(input)
            selected = core.route_by_validation_score_band(
                local_answer=local,
                label_primed_answer=paper,
                ranked_candidates=self.score.local.ranked_label_candidates(query),
            )
            return core.clean_classification_answer(selected), {
                "local_candidate": core.clean_classification_answer(local),
                "label_primed_candidate": core.clean_classification_answer(paper),
            }

        pure, _ = self._call_for_answer(NO_MEMORY_PROMPT.format(input=input))
        verified, _ = self._call_for_answer(
            self.draft.build_verification_prompt(input, pure), default=pure
        )
        paper, _ = self._call_for_answer(self.paper.build_prompt(input))
        local, response = self._call_for_answer(self.local.build_prompt(input))
        selected = core.route_by_taxonomy_specificity(
            model_only_answer=pure,
            draft_verified_answer=verified,
            label_primed_answer=paper,
            local_evidence_answer=local,
        )
        return core.clean_classification_answer(selected), {
            "model_only_candidate": core.clean_classification_answer(pure),
            "draft_verified_candidate": core.clean_classification_answer(verified),
            "label_primed_candidate": core.clean_classification_answer(paper),
            "local_evidence_candidate": core.clean_classification_answer(local),
            "full_response": response,
        }

    def learn_from_batch(self, batch_results: list[dict[str, Any]]) -> None:
        if self.mode == "paper":
            self.paper.learn_from_batch(batch_results)
        elif self.mode == "local":
            self.local.learn_from_batch(batch_results)
        elif self.mode == "score-band":
            self.score.learn_from_batch(batch_results)
        else:
            self.paper.learn_from_batch(batch_results)
            self.draft.learn_from_batch(batch_results)
            self.local.learn_from_batch(batch_results)

    def get_state(self) -> str:
        if self.mode == "paper":
            memories = {"paper": json.loads(self.paper.get_state())}
        elif self.mode == "local":
            memories = {"local": json.loads(self.local.get_state())}
        elif self.mode == "score-band":
            memories = {"score": json.loads(self.score.get_state())}
        else:
            memories = {
                "paper": json.loads(self.paper.get_state()),
                "draft": json.loads(self.draft.get_state()),
                "local": json.loads(self.local.get_state()),
            }
        return json.dumps(
            {"kind": "spiral-meta-harness-native", "mode": self.mode, "memories": memories},
            ensure_ascii=False,
            sort_keys=True,
        )

    def set_state(self, state: str) -> None:
        payload = json.loads(state)
        if payload.get("kind") != "spiral-meta-harness-native" or payload.get("mode") != self.mode:
            raise ValueError("native adapter mode/state mismatch")
        memories = payload["memories"]
        if self.mode == "paper":
            self.paper.set_state(json.dumps(memories["paper"], ensure_ascii=False))
        elif self.mode == "local":
            self.local.set_state(json.dumps(memories["local"], ensure_ascii=False))
        elif self.mode == "score-band":
            self.score.set_state(json.dumps(memories["score"], ensure_ascii=False))
        else:
            self.paper.set_state(json.dumps(memories["paper"], ensure_ascii=False))
            self.draft.set_state(json.dumps(memories["draft"], ensure_ascii=False))
            self.local.set_state(json.dumps(memories["local"], ensure_ascii=False))
