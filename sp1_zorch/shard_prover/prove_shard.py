# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof as one zorch ``ProveChain`` of stage Rounds.

``prove_shard_chain`` sequences the stages of ``docs/shard-pipeline.md`` —
trace commit, LogUp-GKR, zerocheck — as ``zorch.round.Round``s threading one
duplex transcript and a single ``ShardCarry``. The carry holds only what a
later stage reads from an earlier one; static configuration (vk, SMCS, chips)
lives on the Round instances. The jagged evaluation proof appends as a fourth
Round once its zerocheck wiring lands (fractalyze/sp1-zorch#20); proof
assembly consumes the chain's message list (fractalyze/sp1-zorch#21).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from rw_constraints import Chip

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import commit_region
from sp1_zorch.jagged.open import StackedRound
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    LogupGkrProof,
    prove_logup_gkr,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.stage import ZerocheckProof, prove_shard_zerocheck
from zorch.round import ProveChain, Round
from zorch.transcript import Transcript


# Pytree: the two regions (themselves pytrees), public values, and written
# stage outputs are array leaves; unwritten Optional fields are None (an empty
# subtree). Lets the carry cross the chain's @jit boundary as one donatable
# argument.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "main_region",
        "prep_region",
        "public_values",
        "commit_rounds",
        "gkr_eval_point",
        "gkr_chip_openings",
        "zc_sumcheck_point",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardCarry:
    """What flows between stages: the committed regions plus each stage's
    outputs the next one consumes. Stage Rounds return it via ``replace`` —
    a stage writes its own fields and passes the rest through untouched."""

    main_region: JaggedRegion
    prep_region: JaggedRegion | None
    public_values: Array
    # Written by TraceCommitRound; read by the jagged-eval open stage. [prep,
    # main] order, matching SP1's round_evaluation_claims.
    commit_rounds: tuple[StackedRound, ...] | None = None
    # Written by LogupGkrRound; read by ShardZerocheckRound.
    gkr_eval_point: Array | None = None
    gkr_chip_openings: Mapping[str, ChipEvaluation] | None = None
    # Written by ShardZerocheckRound; read by the jagged-eval open stage as its
    # z_row — the accumulated per-round sumcheck challenges, not the GKR zeta.
    zc_sumcheck_point: Array | None = None


def preamble_chip_metadata(
    chip_names: Sequence[str], num_reals: Sequence[int], *, dtype: Any
) -> Array:
    """The preamble's chip-metadata stream as one flat array: chip count, then
    per chip (num_real, name length, name bytes). One flat absorb matches
    SP1's per-value observes byte-for-byte while skipping hundreds of
    single-element transcript calls."""
    metadata: list[int] = [len(chip_names)]
    for name, num_real in zip(chip_names, num_reals, strict=True):
        metadata.append(int(num_real))
        metadata.append(len(name))
        metadata.extend(name.encode("ascii"))
    return jnp.array(metadata, dtype)


class TraceCommitRound(Round):
    """Trace commit plus the shard preamble: commit the main region, then
    absorb SP1's preamble stream (vk, public values, the bound commitment,
    chip metadata). The message is the structure-bound main commitment; the
    prover-side commit data joins the carry once the opening stage that
    reads it lands (fractalyze/sp1-zorch#20)."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        vk: MachineVerifyingKey,
        chip_metadata: Array,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._vk = vk
        self._chip_metadata = chip_metadata

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, Array]:
        bound, main_data = commit_region(
            carry.main_region, self._smcs, log_blowup=self._log_blowup
        )
        transcript = self._vk.observe_into(transcript)
        transcript = transcript.observe(carry.public_values)
        transcript = transcript.observe(bound)
        transcript = transcript.observe(self._chip_metadata)
        # Retain each region's stacked witness for the jagged-eval open. The
        # prep region is bound into the vk at setup, not re-observed here, but
        # the open still reproves it, so commit it for its codeword too. Order
        # is [prep, main], matching SP1's round_evaluation_claims.
        commit_data = []
        if carry.prep_region is not None:
            _, prep_data = commit_region(
                carry.prep_region, self._smcs, log_blowup=self._log_blowup
            )
            commit_data.append(prep_data)
        commit_data.append(main_data)
        carry = replace(
            carry,
            commit_rounds=tuple(
                StackedRound(d.mle, d.codeword, d.digest_layers) for d in commit_data
            ),
        )
        return carry, transcript, bound


class LogupGkrRound(Round):
    """LogUp-GKR stage over ``prove_logup_gkr``; writes the final evaluation
    point and per-chip openings onto the carry for zerocheck."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        witness: Array | None = None,
    ) -> None:
        self._gkr_chips = gkr_chips
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._witness = witness

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, LogupGkrProof]:
        transcript, proof = prove_logup_gkr(
            self._gkr_chips,
            carry.main_region,
            carry.prep_region,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
            witness=self._witness,
        )
        carry = replace(
            carry,
            gkr_eval_point=proof.eval_point,
            gkr_chip_openings=proof.chip_openings,
        )
        return carry, transcript, proof


class ShardZerocheckRound(Round):
    """Zerocheck stage over ``prove_shard_zerocheck``, consuming the GKR
    point and openings off the carry."""

    def __init__(self, chips: Mapping[str, Chip], *, max_log_row_count: int) -> None:
        self._chips = chips
        self._max_log_row_count = max_log_row_count

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, ZerocheckProof]:
        if carry.gkr_eval_point is None or carry.gkr_chip_openings is None:
            raise ValueError(
                "zerocheck needs the LogUp-GKR stage's outputs on the carry; "
                "sequence a LogupGkrRound before this Round"
            )
        transcript, proof = prove_shard_zerocheck(
            self._chips,
            carry.main_region,
            carry.prep_region,
            carry.public_values,
            carry.gkr_eval_point,
            carry.gkr_chip_openings,
            transcript,
            max_log_row_count=self._max_log_row_count,
        )
        carry = replace(carry, zc_sumcheck_point=proof.msgs.challenge)
        return carry, transcript, proof


def prove_shard_chain(
    *,
    smcs: SingleMatrixCommitmentScheme,
    log_blowup: int,
    vk: MachineVerifyingKey,
    chip_metadata: Array,
    gkr_chips: Sequence[GkrChip],
    chips: Mapping[str, Chip],
    num_betas: int,
    num_row_variables: int,
    max_log_row_count: int,
    pow_bits: int = 0,
    witness: Array | None = None,
) -> ProveChain:
    """The SP1 shard chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift
    on it."""
    return ProveChain(
        [
            TraceCommitRound(
                smcs, log_blowup=log_blowup, vk=vk, chip_metadata=chip_metadata
            ),
            LogupGkrRound(
                gkr_chips,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=pow_bits,
                witness=witness,
            ),
            ShardZerocheckRound(chips, max_log_row_count=max_log_row_count),
        ]
    )
