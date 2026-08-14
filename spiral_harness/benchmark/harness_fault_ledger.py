"""Exact live-ledger closure used by the v4 final mechanism verifier."""

from __future__ import annotations

from collections.abc import Iterable

from spiral_harness.benchmark.harness_fault import (
    HarnessFaultGradedBatch,
    HarnessFaultScheduleClosure,
)
from spiral_harness.benchmark.harness_fault_errors import HarnessFaultMechanismError
from spiral_harness.core.models import ArtifactRef
from spiral_harness.execution.attempts import AttemptLedger
from spiral_harness.execution.receipts import (
    EXECUTION_RECEIPT_MEDIA_TYPE,
    ExecutionReceipt,
    replay_trusted_usage,
)
from spiral_harness.execution.schedule import (
    SCHEDULE_PREFLIGHT_MEDIA_TYPE,
    SchedulePreflightCertificate,
)
from spiral_harness.storage.artifact_store import ArtifactStore

type LedgerKey = tuple[str, str, str]


class HarnessFaultLedgerClosureError(HarnessFaultMechanismError):
    """A live writer, receipt set, or preflight failed exact closure."""


def _load(store: ArtifactStore, ref: ArtifactRef, media: str, model: type, label: str):
    try:
        checked = ArtifactRef.model_validate(ref, strict=True)
        if checked.media_type != media:
            raise ValueError("wrong media type")
        loaded = store.get_json(checked, model)
        return model.model_validate(loaded, strict=True)
    except Exception as exc:
        raise HarnessFaultLedgerClosureError(f"{label} artifact cannot be verified") from exc


def index_exact_live_ledgers(
    store: ArtifactStore,
    attempt_ledgers: Iterable[AttemptLedger],
) -> dict[LedgerKey, AttemptLedger]:
    if isinstance(attempt_ledgers, str | bytes | bytearray):
        raise TypeError("attempt_ledgers must be an iterable of exact AttemptLedger values")
    try:
        ledgers = tuple(attempt_ledgers)
    except TypeError as exc:
        raise TypeError("attempt_ledgers must be iterable") from exc
    if len(ledgers) != 3 or len({id(ledger) for ledger in ledgers}) != 3:
        raise HarnessFaultLedgerClosureError(
            "mechanism requires exactly three distinct live ledgers"
        )
    indexed: dict[LedgerKey, AttemptLedger] = {}
    for ledger in ledgers:
        if type(ledger) is not AttemptLedger or ledger.repository is not store:
            raise HarnessFaultLedgerClosureError(
                "mechanism requires this store's exact live ledger writers"
            )
        try:
            state = ledger.state()
        except Exception as exc:
            raise HarnessFaultLedgerClosureError("live ledger cannot be verified") from exc
        key = (state.ledger_id, state.writer_epoch_id, state.budget.fingerprint)
        if key in indexed:
            raise HarnessFaultLedgerClosureError("live ledger writer identities must be unique")
        indexed[key] = ledger
    return indexed


def replay_exact_batch_usage(
    store: ArtifactStore,
    *,
    batch: HarnessFaultGradedBatch,
    closure: HarnessFaultScheduleClosure,
    live_ledgers: dict[LedgerKey, AttemptLedger],
) -> tuple[LedgerKey, ArtifactRef]:
    receipts = tuple(
        _load(
            store,
            receipt_ref,
            EXECUTION_RECEIPT_MEDIA_TYPE,
            ExecutionReceipt,
            "trusted usage receipt",
        )
        for receipt_ref in batch.usage.receipt_refs
    )
    preflight_refs = {receipt.preflight_ref for receipt in receipts}
    if len(preflight_refs) != 1:
        raise HarnessFaultLedgerClosureError(
            "trusted usage receipt set does not share one exact preflight"
        )
    preflight_ref = next(iter(preflight_refs))
    preflight = _load(
        store,
        preflight_ref,
        SCHEDULE_PREFLIGHT_MEDIA_TYPE,
        SchedulePreflightCertificate,
        "trusted usage preflight",
    )
    if preflight.ledger_id is None or preflight.writer_epoch_id is None:
        raise HarnessFaultLedgerClosureError("trusted usage preflight is not ledger-bound")
    ledger_key = (
        preflight.ledger_id,
        preflight.writer_epoch_id,
        preflight.budget_fingerprint,
    )
    ledger = live_ledgers.pop(ledger_key, None)
    if ledger is None:
        raise HarnessFaultLedgerClosureError(
            "trusted usage preflight has no exact live ledger writer"
        )
    try:
        replayed = replay_trusted_usage(
            store,
            schedule=closure.schedule,
            preflight_ref=preflight_ref,
            attempt_ledger=ledger,
            receipt_refs=batch.usage.receipt_refs,
        )
    except Exception as exc:
        raise HarnessFaultLedgerClosureError("final trusted usage replay failed closed") from exc
    if replayed != batch.usage:
        raise HarnessFaultLedgerClosureError(
            "persisted trusted usage differs from final live-ledger replay"
        )
    return ledger_key, replayed.ledger_tail_refs[0]


def reverify_live_ledger_tails(
    ledgers: dict[LedgerKey, AttemptLedger],
    expected_tails: dict[LedgerKey, ArtifactRef],
) -> None:
    for key, ledger in ledgers.items():
        try:
            state = ledger.state()
        except Exception as exc:
            raise HarnessFaultLedgerClosureError("live ledger cannot be reverified") from exc
        if (
            (state.ledger_id, state.writer_epoch_id, state.budget.fingerprint) != key
            or state.tail_ref != expected_tails.get(key)
            or state.pending_reservation_ref is not None
        ):
            raise HarnessFaultLedgerClosureError(
                "live ledger advanced during final trusted usage verification"
            )


__all__ = [
    "HarnessFaultLedgerClosureError",
    "LedgerKey",
    "index_exact_live_ledgers",
    "replay_exact_batch_usage",
    "reverify_live_ledger_tails",
]
