# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof's verifier as one zorch ``VerifyChain`` of stage duals.

``verify_shard_chain`` mirrors ``prove_shard_chain`` round for round — one
verifier Round per prover stage, glue included, consuming the prover chain's
message list as the proof object. ``VerifyChain``'s one-message-per-round
check makes the mirror fail loud: a stage or glue step present on one side
and not the other is a structural reject, not a silent Fiat-Shamir desync
(zorch ``docs/stage-composition.md``, "Pipelines as nested chains").

Static configuration (vk, chip metadata) lives on the Round instances and
per-shard values flow on the carry, mirroring the prover's split;
``ShardVerifierCarry`` threads what a later dual reads from an earlier one —
the witness-free dual of ``ShardCarry``.

Only the trace-commit dual checks anything yet; the remaining stage duals are
accept-all placeholders, replaced stage by stage (fractalyze/sp1-zorch#75).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from sp1_zorch.shard_prover.prove_shard import PreambleRound
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from zorch.round import Round, VerifyChain
from zorch.transcript import Transcript


# Pytree like ShardCarry: written stage outputs are array leaves; unwritten
# Optional fields are None (an empty subtree), so the carry crosses a @jit
# boundary as one argument.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=["public_values", "commitment_roots"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardVerifierCarry:
    """What flows between stage duals: each dual writes the fields a later
    one consumes — the same seams as ``ShardCarry``, minus the witness."""

    # Statement input, read by the trace-commit dual (preamble) and the
    # zerocheck dual (constraint evaluation) — on the carry, not the Rounds,
    # exactly as the prover reads ``ShardCarry.public_values``.
    public_values: Array
    # Written by TraceCommitVerifierRound; read by the stacked-open dual
    # (skip-level, two seams later). [prep, main] order, matching SP1's
    # round_evaluation_claims.
    commitment_roots: tuple[Array, Array] | None = None


class TraceCommitVerifierRound(Round):
    """Stage-1 dual of ``TraceCommitRound``: replays the preamble absorb
    stream via ``PreambleRound`` — the same one Round the prover drives —
    with the proof's commitment message, and writes the commitment roots
    onto the carry. No local check: the commitment is validated downstream,
    by the stacked-open dual's Merkle openings against these roots."""

    def __init__(self, *, vk: MachineVerifyingKey, chip_metadata: Array) -> None:
        self._vk = vk
        self._chip_metadata = chip_metadata

    def __call__(
        self, carry: ShardVerifierCarry, msg: Array, transcript: Transcript
    ) -> tuple[ShardVerifierCarry, Transcript, Array]:
        _, transcript, _ = PreambleRound(
            vk=self._vk,
            public_values=carry.public_values,
            commitment=msg,
            chip_metadata=self._chip_metadata,
        )(None, transcript)
        # The prep root is unconditional: SP1's verifier always carries the
        # vk's preprocessed commitment, even though the prover keeps
        # ``prep_region`` optional. The stacked-open dual checking openings
        # against these roots is where a no-prep proof would reconcile.
        carry = replace(
            carry, commitment_roots=(self._vk.preprocessed_commit, msg)
        )
        return carry, transcript, jnp.bool_(True)


class _AcceptAllRound(Round):
    """Placeholder stage dual: passes the carry and transcript through and
    accepts its message unconditionally. Holds the stage's slot so the
    chain's round count mirrors the prover's; replaced by the real dual
    (fractalyze/sp1-zorch#75)."""

    def __call__(
        self, carry: Any, msg: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Array]:
        return carry, transcript, jnp.bool_(True)


def verify_shard_chain(*, vk: MachineVerifyingKey, chip_metadata: Array) -> VerifyChain:
    """The ``VerifyChain`` dual of ``prove_shard_chain``: one verifier Round
    per prover stage, in the prover's order, so the proof's message list
    aligns slot for slot or fails the chain's one-message-per-round check."""
    return VerifyChain(
        [
            TraceCommitVerifierRound(vk=vk, chip_metadata=chip_metadata),
            _AcceptAllRound(),  # LogUp-GKR dual
            _AcceptAllRound(),  # zerocheck dual
            _AcceptAllRound(),  # jagged eval + stacked open dual
        ]
    )
