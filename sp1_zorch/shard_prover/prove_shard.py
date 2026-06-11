# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof as one zorch ``ProveChain`` of stage Rounds.

``prove_shard_chain`` sequences the stages of ``docs/shard-pipeline.md`` —
trace commit, LogUp-GKR, zerocheck, jagged evaluation proof — as
``zorch.round.Round``s threading one duplex transcript and a single
``ShardCarry``. The carry holds only what a later stage reads from an earlier
one; static configuration (vk, SMCS, chips) lives on the Round instances.
Proof assembly consumes the chain's message list (fractalyze/sp1-zorch#21).
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
from zk_dtypes import efinfo, koalabearx4_mont

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import commit_region
from sp1_zorch.jagged.open import (
    StackedOpenProof,
    StackedRound,
    stacked_basefold_open,
)
from sp1_zorch.jagged.prover import (
    JaggedEvalInputs,
    JaggedEvalMsg,
    JaggedEvalRound,
    assemble_columns,
)
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    LogupGkrProof,
    prove_logup_gkr,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.stage import ZerocheckProof, prove_shard_zerocheck
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.round import ProveChain, Round
from zorch.transcript import GrindingTranscript, Transcript, sample_challenge
from zorch.utils.bits import log2_ceil_usize


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


class PreambleRound(Round):
    """SP1's shard preamble absorb stream: vk, public values, the main
    commitment, chip metadata. The schedule lives here once — the prover's
    ``TraceCommitRound`` and the byte-match replay's ``preamble_transcript``
    drive this one Round, so an ordering edit cannot land in one Fiat-Shamir
    stream and not the other (the GKR head schedule got the same treatment in
    ``logup_gkr.head``). Carry-agnostic; the message is the observed
    commitment, the stream's one structure-bound value."""

    def __init__(
        self,
        *,
        vk: MachineVerifyingKey,
        public_values: Array,
        commitment: Array,
        chip_metadata: Array,
    ) -> None:
        self._vk = vk
        self._public_values = public_values
        self._commitment = commitment
        self._chip_metadata = chip_metadata

    def __call__(
        self, carry: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Array]:
        transcript = self._vk.observe_into(transcript)
        transcript = transcript.observe(self._public_values)
        transcript = transcript.observe(self._commitment)
        transcript = transcript.observe(self._chip_metadata)
        return carry, transcript, self._commitment


class TraceCommitRound(Round):
    """Trace commit plus the shard preamble: commit the main region, then
    absorb SP1's preamble stream via ``PreambleRound``. The message is the
    structure-bound main commitment; the prover-side commit data joins the
    carry once the opening stage that reads it lands
    (fractalyze/sp1-zorch#20)."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        vk: MachineVerifyingKey,
        chip_metadata: Array,
        jit: bool = False,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._vk = vk
        self._chip_metadata = chip_metadata
        self._jit = jit

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, Array]:
        bound, main_data = commit_region(
            carry.main_region, self._smcs, log_blowup=self._log_blowup, jit=self._jit
        )
        _, transcript, _ = PreambleRound(
            vk=self._vk,
            public_values=carry.public_values,
            commitment=bound,
            chip_metadata=self._chip_metadata,
        )(carry, transcript)
        # Retain each region's stacked witness for the jagged-eval open. The
        # prep region is bound into the vk at setup, not re-observed here, but
        # the open still reproves it, so commit it for its codeword too. Order
        # is [prep, main], matching SP1's round_evaluation_claims.
        commit_data = []
        if carry.prep_region is not None:
            # The prep region stays eager regardless of the knob: it is far
            # below the main region's memory scale, and its different shape
            # would cost a second full-pipeline compile for no benefit.
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
        jit: bool = False,
    ) -> None:
        self._gkr_chips = gkr_chips
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._witness = witness
        self._jit = jit

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
            jit=self._jit,
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


@dataclass(frozen=True)
class ShardJaggedEvalProof:
    """The jagged evaluation proof: the outer/inner sumcheck reducing the
    committed trace to ``D(z_final)``, then the stacked BaseFold open of ``D``
    at that point."""

    eval: JaggedEvalMsg
    open: StackedOpenProof


class ShardJaggedEvalRound(Round):
    """Jagged evaluation proof (SP1 Phase 4): reduce the committed trace to
    ``D(z_final)`` via the outer/inner sumcheck, then open ``D`` at ``z_final``
    with the stacked BaseFold FRI. Reads the zerocheck point, the per-chip GKR
    openings, and the committed stacked witness off the carry."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        num_queries: int,
        pow_bits: int,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits

    def __call__(
        self, carry: ShardCarry, transcript: GrindingTranscript
    ) -> tuple[ShardCarry, GrindingTranscript, ShardJaggedEvalProof]:
        if (
            carry.zc_sumcheck_point is None
            or carry.commit_rounds is None
            or carry.gkr_chip_openings is None
        ):
            raise ValueError(
                "the jagged-eval stage needs the zerocheck point, committed "
                "rounds, and GKR openings on the carry; sequence the commit, "
                "LogUp-GKR, and zerocheck Rounds before it"
            )
        main = carry.main_region
        openings = carry.gkr_chip_openings
        # The jagged eval runs in the extension field; the sumcheck points
        # (z_row) are base-field per-round challenge lists, embedded up to EF
        # where the eval needs them.
        ef = koalabearx4_mont

        # Per-round (row/column counts, real per-column claims) in [prep, main]
        # order — each chip's GKR opening field is its columns' claims — plus
        # each region's raw (unpadded) dense for the combined committed D.
        rc_rounds: list[Sequence[int]] = []
        cc_rounds: list[Sequence[int]] = []
        claims_rounds: list[Array] = []
        denses: list[Array] = []
        prep = carry.prep_region
        regions = ([(prep, "preprocessed")] if prep is not None else []) + [
            (main, "main")
        ]
        for region, claim_field in regions:
            rc_rounds.append(region.row_counts)
            cc_rounds.append(region.column_counts)
            claims_rounds.append(
                jnp.concatenate(
                    [getattr(openings[n], claim_field) for n in region.chip_names]
                )
            )
            denses.append(region.dense[: region.raw_size])

        col_heights, all_claims = assemble_columns(
            rc_rounds, cc_rounds, claims_rounds, dtype=ef
        )

        # The outer Hadamard sumcheck folds D variable by variable, so the
        # combined dense pads to a power of two.
        dense = jnp.concatenate(denses)
        target = 1 << log2_ceil_usize(dense.shape[0])
        dense = jnp.pad(dense, (0, target - dense.shape[0]))

        # z_col is one EF challenge per column variable (SP1 samples it as
        # extension elements, not stacked base squeezes).
        ef_degree = efinfo(ef).degree
        z_col_parts: list[Array] = []
        for _ in range(log2_ceil_usize(len(col_heights))):
            transcript, challenge = sample_challenge(transcript, ef, ef_degree)
            z_col_parts.append(challenge)
        z_col = jnp.stack(z_col_parts) if z_col_parts else jnp.zeros((0,), ef)

        # z_row is the zerocheck sumcheck point in SP1's insert-at-front
        # (reversed) order, embedded base->extension (exact) so the eval's scan
        # carry stays in the extension field.
        inputs = JaggedEvalInputs(
            col_heights=tuple(col_heights),
            all_claims=all_claims,
            z_row=carry.zc_sumcheck_point[::-1].astype(ef),
            z_col=z_col,
            dense=dense,
        )
        _, transcript, eval_msg = JaggedEvalRound(dtype=ef)(inputs, transcript)

        code = BitReversedReedSolomon(
            message_len=1 << main.log_stacking_height,
            blowup=1 << self._log_blowup,
            dtype=main.dense.dtype,
        )
        # The outer sumcheck folds in the base field; the BaseFold open works
        # in the extension field, so embed its folded point and D(z_final)
        # base->extension (exact).
        open_proof, transcript = stacked_basefold_open(
            self._smcs,
            code,
            carry.commit_rounds,
            eval_msg.outer_sumcheck_point.astype(ef),
            eval_msg.dense_eval.astype(ef),
            main.log_stacking_height,
            num_queries=self._num_queries,
            pow_bits=self._pow_bits,
            transcript=transcript,
        )
        return carry, transcript, ShardJaggedEvalProof(eval=eval_msg, open=open_proof)


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
    open_num_queries: int,
    open_pow_bits: int = 0,
    pow_bits: int = 0,
    witness: Array | None = None,
    jit: bool = False,
) -> ProveChain:
    """The SP1 shard chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift
    on it.

    ``jit`` runs every stage that supports it under ``jax.jit`` — the
    trace-commit tail (required at rsp scale, see
    ``sp1_zorch.commit.trace_commit``) and the per-layer GKR prove (see
    ``prove_logup_gkr``). Byte-identical either way."""
    return ProveChain(
        [
            TraceCommitRound(
                smcs,
                log_blowup=log_blowup,
                vk=vk,
                chip_metadata=chip_metadata,
                jit=jit,
            ),
            LogupGkrRound(
                gkr_chips,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=pow_bits,
                witness=witness,
                jit=jit,
            ),
            ShardZerocheckRound(chips, max_log_row_count=max_log_row_count),
            ShardJaggedEvalRound(
                smcs,
                log_blowup=log_blowup,
                num_queries=open_num_queries,
                pow_bits=open_pow_bits,
            ),
        ]
    )
