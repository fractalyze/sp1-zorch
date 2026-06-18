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
import numpy as np
from jax import Array
from rw_constraints import Chip
from zk_dtypes import koalabearx4_mont

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
    sample_z_col,
)
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    LogupGkrProof,
    prove_logup_gkr,
    prove_logup_gkr_body,
    resolve_witness_and_grind,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.stage import ZerocheckProof, prove_shard_zerocheck
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.round import ProveChain, Round
from zorch.transcript import GrindingTranscript, Transcript
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
        "zc_opened_values",
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
    # Written by ShardZerocheckRound; read by the jagged-eval stage as its
    # per-column claims (the trace evaluations at the zerocheck point) and by
    # proof assembly as the wire's ShardOpenedValues.
    zc_opened_values: Mapping[str, ChipEvaluation] | None = None


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
        offload_commit_rounds: bool = False,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._vk = vk
        self._chip_metadata = chip_metadata
        self._jit = jit
        self._offload_commit_rounds = offload_commit_rounds

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
        commit_rounds = tuple(
            StackedRound(d.mle, d.codeword, d.digest_layers) for d in commit_data
        )
        if self._offload_commit_rounds:
            # Park the committed witness (full-blowup codeword + digest tree) on
            # host until the open stage. LogUp-GKR and zerocheck never read it,
            # but as a carry leaf it would pin those device buffers through
            # prove_jagged_pyramid's peak -- ~0.65 GiB over a 32 GiB card on rsp
            # shard17, the OOM that blocks the GPU-vs-GPU baseline
            # (fractalyze/sp1-zorch#55, #124). np.asarray copies each leaf to
            # host (blocking until commit is realized); the pre-offload device
            # tuple plus commit_data are this frame's only remaining references,
            # so the device buffers release on return -- before the GKR pyramid
            # allocates -- rather than staying resident through the chain. The
            # open reloads it (ShardJaggedEvalRound). Pure round-trip:
            # byte-identical, but host-orchestrated only (a host copy cannot run
            # inside a single-@jit trace of the whole chain).
            commit_rounds = jax.tree_util.tree_map(np.asarray, commit_rounds)
        carry = replace(carry, commit_rounds=commit_rounds)
        return carry, transcript, bound


class LogupGkrRound(Round):
    """LogUp-GKR stage over ``prove_logup_gkr``; writes the final evaluation
    point and per-chip openings onto the carry for zerocheck.

    With ``jit=True`` the grind-free body (head challenges, circuit build, the
    rolled pyramid sumcheck, trace openings) runs under one cached outer ``@jit``
    so the warm prove's ~thousands of op-by-op dispatches collapse into a single
    program -- the host-dispatch wall that otherwise leaves the GPU ~idle at
    shard scale (sp1-zorch#119). The grind stays eager so its ``pow_bits > 0``
    host-side PoW verdict is legal; ``@jit`` is byte-transparent, so the proof is
    identical to the eager path either way."""

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
        # Compile the body once so warm chain runs reuse the executable (a fresh
        # `jax.jit` per call would recompile every run). Regions + the post-grind
        # transcript trace as args; chip metadata / counts are closed over as
        # static. `LogUpGkrOutput` is not a registered pytree, so the body
        # returns its two array leaves and `__call__` rebuilds it -- everything
        # else (transcript, round_proofs, openings) already crosses jit.
        self._body = self._compile_body() if jit else None

    def _compile_body(self):
        gkr_chips = self._gkr_chips
        num_betas = self._num_betas
        num_row_variables = self._num_row_variables

        @jax.jit
        def body(main_region, prep_region, transcript, witness):
            transcript, proof = prove_logup_gkr_body(
                gkr_chips,
                main_region,
                prep_region,
                transcript,
                witness,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
            )
            out = proof.circuit_output
            return (
                transcript,
                proof.witness,
                out.numerator,
                out.denominator,
                proof.round_proofs,
                proof.eval_point,
                proof.chip_openings,
            )

        return body

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, LogupGkrProof]:
        if self._body is None:
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
        else:
            # Grind eagerly (its pow_bits > 0 verdict is a host-side bool(ok));
            # the grind-free body then dispatches as one program.
            transcript, witness = resolve_witness_and_grind(
                transcript,
                pow_bits=self._pow_bits,
                witness=self._witness,
                bf_dtype=carry.main_region.dense.dtype,
            )
            (
                transcript,
                witness,
                numerator,
                denominator,
                round_proofs,
                eval_point,
                chip_openings,
            ) = self._body(carry.main_region, carry.prep_region, transcript, witness)
            proof = LogupGkrProof(
                witness=witness,
                circuit_output=LogUpGkrOutput(
                    numerator=numerator, denominator=denominator
                ),
                round_proofs=round_proofs,
                eval_point=eval_point,
                chip_openings=chip_openings,
            )
        carry = replace(
            carry,
            gkr_eval_point=proof.eval_point,
            gkr_chip_openings=proof.chip_openings,
        )
        return carry, transcript, proof


class ShardZerocheckRound(Round):
    """Zerocheck stage over ``prove_shard_zerocheck``, consuming the GKR
    point and openings off the carry. The stage absorbs the per-chip opened
    values itself (``OpenedValuesRound`` in ``zerocheck.stage``); this Round
    threads them onto the carry for the jagged-eval stage's claims and the
    wire's ShardOpenedValues."""

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
        carry = replace(
            carry,
            zc_sumcheck_point=proof.msgs.challenge,
            zc_opened_values=proof.opened_values,
        )
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
    with the stacked BaseFold FRI. Reads the zerocheck point, the per-chip
    opened values at it, and the committed stacked witness off the carry."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        num_queries: int,
        pow_bits: int,
        offload_commit_rounds: bool = False,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits
        self._offload_commit_rounds = offload_commit_rounds

    def __call__(
        self, carry: ShardCarry, transcript: GrindingTranscript
    ) -> tuple[ShardCarry, GrindingTranscript, ShardJaggedEvalProof]:
        if (
            carry.zc_sumcheck_point is None
            or carry.commit_rounds is None
            or carry.zc_opened_values is None
        ):
            raise ValueError(
                "the jagged-eval stage needs the zerocheck point, committed "
                "rounds, and zerocheck opened values on the carry; sequence "
                "the commit, LogUp-GKR, and zerocheck Rounds before it"
            )
        main = carry.main_region
        openings = carry.zc_opened_values
        # The jagged eval runs in the extension field — the upstream sumcheck
        # points are EF challenge lists (one extension sample per variable).
        ef = koalabearx4_mont

        # Per-round (row/column counts, real per-column claims) in [prep, main]
        # order — each chip's opened-values field at the zerocheck point is its
        # columns' claims (SP1's round_evaluation_claims) — plus each region's
        # raw (unpadded) dense for the combined committed D.
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

        transcript, z_col = sample_z_col(transcript, len(col_heights), ef)

        # z_row is the zerocheck sumcheck point in SP1's insert-at-front
        # (reversed) order.
        inputs = JaggedEvalInputs(
            col_heights=tuple(col_heights),
            all_claims=all_claims,
            z_row=carry.zc_sumcheck_point[::-1],
            z_col=z_col,
            dense=dense,
        )
        _, transcript, eval_msg = JaggedEvalRound(dtype=ef)(inputs, transcript)

        code = BitReversedReedSolomon(
            message_len=1 << main.log_stacking_height,
            blowup=1 << self._log_blowup,
            dtype=main.dense.dtype,
        )
        # When TraceCommitRound parked the committed witness on host to free
        # device memory through the GKR + zerocheck stages
        # (fractalyze/sp1-zorch#55/#124), pull it back to the default device for
        # the open. Gated on the same flag so the default path -- which keeps
        # the witness device-resident and stays traceable under one whole-chain
        # @jit -- adds no host->device copy.
        commit_rounds = (
            jax.tree_util.tree_map(jnp.asarray, carry.commit_rounds)
            if self._offload_commit_rounds
            else carry.commit_rounds
        )
        open_proof, transcript = stacked_basefold_open(
            self._smcs,
            code,
            commit_rounds,
            eval_msg.outer_sumcheck_point,
            eval_msg.dense_eval,
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
    offload_commit_rounds: bool = False,
) -> ProveChain:
    """The SP1 shard chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift
    on it.

    ``jit`` stages the heavy per-stage work under ``jax.jit``: the trace-commit
    tail (required at rsp scale, see ``sp1_zorch.commit.trace_commit``) and the
    LogUp-GKR body, whose warm prove is otherwise host-dispatch-bound and leaves
    the GPU ~idle (sp1-zorch#119). Byte-identical either way.

    ``offload_commit_rounds`` parks the committed witness on host between the
    commit and the open so its codeword + digest buffers do not stay device-
    resident through the LogUp-GKR + zerocheck stages -- the ~0.65 GiB that tips
    rsp shard17's full chain over a 32 GiB card (fractalyze/sp1-zorch#55, #124).
    Byte-identical (a pure device<->host round-trip), but requires the
    host-orchestrated ``ProveChain.__call__`` -- it is incompatible with tracing
    the whole chain under one ``@jit`` (a ``device_get`` cannot run in a trace),
    so it defaults off and the eager rsp runnable opts in."""
    return ProveChain(
        [
            TraceCommitRound(
                smcs,
                log_blowup=log_blowup,
                vk=vk,
                chip_metadata=chip_metadata,
                jit=jit,
                offload_commit_rounds=offload_commit_rounds,
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
                offload_commit_rounds=offload_commit_rounds,
            ),
        ]
    )
