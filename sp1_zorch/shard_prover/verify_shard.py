# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof's verifier as one zorch ``VerifyChain`` of stage duals.

``verify_shard_chain`` mirrors ``prove_shard_chain`` round for round — one
verifier Round per prover stage, glue included, consuming the prover chain's
message list as the proof object. ``VerifyChain``'s one-message-per-round
check makes the mirror fail loud: a stage or glue step present on one side
and not the other is a structural reject, not a silent Fiat-Shamir desync
(zorch ``docs/stage-composition.md``, "Pipelines as nested chains").

Static configuration (vk, chip metadata, chip set) lives on the Round
instances and per-shard values flow on the carry, mirroring the prover's
split; ``ShardVerifierCarry`` threads what a later dual reads from an
earlier one — the witness-free dual of ``ShardCarry``.

The trace-commit and LogUp-GKR duals are real; the remaining stage duals are
accept-all placeholders, replaced stage by stage (fractalyze/sp1-zorch#75).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import ChipEvaluation, LogupGkrProof
from sp1_zorch.logup_gkr.verifier import verify_logup_gkr
from sp1_zorch.shard_prover.prove_shard import PreambleRound
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from zorch.round import Round, VerifyChain
from zorch.transcript import GrindingTranscript, Transcript


# Pytree like ShardCarry: written stage outputs are array leaves; unwritten
# Optional fields are None (an empty subtree), so the carry crosses a @jit
# boundary as one argument.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "public_values",
        "commitment_roots",
        "gkr_eval_point",
        "gkr_chip_openings",
    ],
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
    # Written by LogupGkrVerifierRound; read by the zerocheck dual (zeta =
    # row tail; claims derivation). The point is the dual's own derivation,
    # the openings are the proof's leaf-checked values.
    gkr_eval_point: Array | None = None
    gkr_chip_openings: Mapping[str, ChipEvaluation] | None = None


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


class LogupGkrVerifierRound(Round):
    """Stage-2 dual of ``LogupGkrRound``: verifies the LogUp-GKR proof via
    ``verify_logup_gkr`` and writes the derived evaluation point plus the
    proof's leaf-checked chip openings onto the carry — the same seams the
    prover Round writes on ``ShardCarry`` for the zerocheck stage."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        chip_names: Sequence[str],
        chip_heights: Mapping[str, int],
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
    ) -> None:
        self._gkr_chips = gkr_chips
        self._chip_names = chip_names
        self._chip_heights = chip_heights
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits

    def __call__(
        self,
        carry: ShardVerifierCarry,
        msg: LogupGkrProof,
        transcript: GrindingTranscript,
    ) -> tuple[ShardVerifierCarry, GrindingTranscript, Array]:
        transcript, eval_point, ok = verify_logup_gkr(
            self._gkr_chips,
            self._chip_names,
            self._chip_heights,
            msg,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
        )
        carry = replace(
            carry, gkr_eval_point=eval_point, gkr_chip_openings=msg.chip_openings
        )
        return carry, transcript, ok


class _AcceptAllRound(Round):
    """Placeholder stage dual: passes the carry and transcript through and
    accepts its message unconditionally. Holds the stage's slot so the
    chain's round count mirrors the prover's; replaced by the real dual
    (fractalyze/sp1-zorch#75)."""

    def __call__(
        self, carry: Any, msg: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Array]:
        return carry, transcript, jnp.bool_(True)


def verify_shard_chain(
    *,
    vk: MachineVerifyingKey,
    chip_metadata: Array,
    gkr_chips: Sequence[GkrChip],
    chip_names: Sequence[str],
    chip_heights: Mapping[str, int],
    num_betas: int,
    num_row_variables: int,
    pow_bits: int = 0,
) -> VerifyChain:
    """The ``VerifyChain`` dual of ``prove_shard_chain``: one verifier Round
    per prover stage, in the prover's order, so the proof's message list
    aligns slot for slot or fails the chain's one-message-per-round check.

    ``chip_names`` and ``chip_heights`` cover every shard chip (the openings
    absorb order and the leaf check's geq thresholds) — the verifier-side
    statement counterpart of the regions the prover Rounds read off the
    carry."""
    return VerifyChain(
        [
            TraceCommitVerifierRound(vk=vk, chip_metadata=chip_metadata),
            LogupGkrVerifierRound(
                gkr_chips,
                chip_names=chip_names,
                chip_heights=chip_heights,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=pow_bits,
            ),
            _AcceptAllRound(),  # zerocheck dual
            _AcceptAllRound(),  # jagged eval + stacked open dual
        ]
    )
