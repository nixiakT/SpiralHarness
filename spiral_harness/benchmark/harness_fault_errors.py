"""Shared fail-closed verifier errors for HarnessFaultBench."""


class HarnessFaultMechanismError(ValueError):
    """A batch, ledger, roster, runtime event, or raw artifact failed closure."""


__all__ = ["HarnessFaultMechanismError"]
