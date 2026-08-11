"""Domain-separated deterministic seeds for reproducible search strategies.

Search randomness is deliberately separate from benchmark rollout seeds.  The
functions here define a small, versioned derivation contract so an independent
search run cannot accidentally reuse a task seed or another strategy's random
stream.
"""

from __future__ import annotations

import hashlib

from spiral_harness.core.canonical import canonical_json_bytes
from spiral_harness.experiments.baselines import BaselineKind

STRATEGY_SEED_DOMAIN = "spiral-harness/evolution/strategy-seed/v1"
RANDOM_VALID_SAMPLE_DOMAIN = "spiral-harness/evolution/random-valid-sample/v1"
UNIFORM_SHUFFLE_ALGORITHM = "sha256-counter-fisher-yates-rejection-v1"


def _require_nonnegative_integer(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _digest_payload(*, domain: str, payload: dict[str, object]) -> bytes:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": domain,
                "payload": payload,
            }
        )
    ).digest()


def derive_strategy_seed(
    *,
    proposal_master_seed: int,
    search_run_seed: int,
    baseline_kind: BaselineKind,
) -> int:
    """Derive a condition-local strategy seed for one independent search run."""

    master = _require_nonnegative_integer(
        proposal_master_seed,
        field_name="proposal_master_seed",
    )
    run = _require_nonnegative_integer(search_run_seed, field_name="search_run_seed")
    if not isinstance(baseline_kind, BaselineKind):
        raise TypeError("baseline_kind must be a BaselineKind")
    digest = _digest_payload(
        domain=STRATEGY_SEED_DOMAIN,
        payload={
            "baseline_kind": baseline_kind.value,
            "proposal_master_seed": master,
            "search_run_seed": run,
        },
    )
    return int.from_bytes(digest, byteorder="big", signed=False)


def derive_random_valid_sample_seed(
    *,
    strategy_seed: int,
    round_index: int,
    catalogue_fingerprint: str,
    eligible_fingerprint: str,
) -> int:
    """Derive one catalogue-sampling seed without sharing another round's stream."""

    seed = _require_nonnegative_integer(strategy_seed, field_name="strategy_seed")
    round_number = _require_nonnegative_integer(round_index, field_name="round_index")
    for field_name, fingerprint in (
        ("catalogue_fingerprint", catalogue_fingerprint),
        ("eligible_fingerprint", eligible_fingerprint),
    ):
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if any(character not in "0123456789abcdef" for character in fingerprint):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    digest = _digest_payload(
        domain=RANDOM_VALID_SAMPLE_DOMAIN,
        payload={
            "catalogue_fingerprint": catalogue_fingerprint,
            "eligible_fingerprint": eligible_fingerprint,
            "round_index": round_number,
            "strategy_seed": seed,
        },
    )
    return int.from_bytes(digest, byteorder="big", signed=False)


def uniform_without_replacement_indices(
    *,
    population_size: int,
    sample_size: int,
    sample_seed: int,
) -> tuple[int, ...]:
    """Return a deterministic uniform-without-replacement prefix.

    Fisher--Yates is driven by SHA-256 counter blocks.  Rejection sampling,
    rather than modulo reduction, avoids index bias.  The returned order is the
    draw order and is therefore intentionally not sorted.
    """

    population = _require_nonnegative_integer(population_size, field_name="population_size")
    requested = _require_nonnegative_integer(sample_size, field_name="sample_size")
    seed = _require_nonnegative_integer(sample_seed, field_name="sample_seed")
    if requested > population:
        raise ValueError("sample_size must not exceed population_size")
    if requested == 0:
        return ()

    values = list(range(population))
    seed_bytes = seed.to_bytes(max(1, (seed.bit_length() + 7) // 8), byteorder="big")
    counter = 0

    def unbiased_index(upper_inclusive: int) -> int:
        nonlocal counter
        width = upper_inclusive + 1
        source_size = 1 << 256
        acceptance_limit = source_size - (source_size % width)
        while True:
            counter_bytes = counter.to_bytes(16, byteorder="big", signed=False)
            counter += 1
            value = int.from_bytes(
                hashlib.sha256(
                    RANDOM_VALID_SAMPLE_DOMAIN.encode("ascii")
                    + b"\x00"
                    + seed_bytes
                    + b"\x00"
                    + counter_bytes
                ).digest(),
                byteorder="big",
                signed=False,
            )
            if value < acceptance_limit:
                return value % width

    # A partial Fisher--Yates shuffle puts the requested random draws first.
    for index in range(requested):
        swap_index = index + unbiased_index(population - index - 1)
        values[index], values[swap_index] = values[swap_index], values[index]
    return tuple(values[:requested])


__all__ = [
    "RANDOM_VALID_SAMPLE_DOMAIN",
    "STRATEGY_SEED_DOMAIN",
    "UNIFORM_SHUFFLE_ALGORITHM",
    "derive_random_valid_sample_seed",
    "derive_strategy_seed",
    "uniform_without_replacement_indices",
]
