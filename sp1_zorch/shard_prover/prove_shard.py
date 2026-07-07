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
from zk_dtypes import koalabearx4_mont

from zorch.pcs.jagged.region import JaggedRegion
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.commit import commit_region
from zorch.pcs.jagged.open import (
    StackedOpenProof,
    StackedRound,
    stacked_basefold_open,
)
from zorch.pcs.jagged.prover import (
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
from sp1_zorch.zerocheck.prover import ZerocheckProof, prove_shard_zerocheck
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
        "commit_commitments",
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
    # Written by TraceCommitRound; read by proof assembly as the jagged proof's
    # original_commitments -- each round's SMCS commitment (pre-structure-binding),
    # [prep, main] order.
    commit_commitments: tuple[Array, ...] | None = None
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
        jit: bool = True,
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
            carry.main_region,
            self._smcs,
            log_blowup=self._log_blowup,
            jit=self._jit,
        )
        _, transcript, _ = PreambleRound(
            vk=self._vk,
            public_values=carry.public_values,
            commitment=bound,
            chip_metadata=self._chip_metadata,
        )(carry, transcript)
        # Retain each region's stacked witness for the jagged-eval open in
        # [prep, main] order (SP1's round_evaluation_claims). Only the mle +
        # digest tree are kept: the open re-encodes the ~6 GB full-blowup
        # codeword, so it never pins device memory through the LogUp-GKR +
        # zerocheck stages — the ~0.65 GiB that tips rsp shard17's full chain
        # over a 32 GiB card (fractalyze/sp1-zorch#55, #124). prep is bound into
        # the vk at setup, not re-observed here, but the open still reproves it.
        commit_data = []
        if carry.prep_region is not None:
            # Same @jit knob as main: committing prep eagerly de-fuses the
            # Merkle fold into ~5k tiny generic-fusion launches (#137).
            _, prep_data = commit_region(
                carry.prep_region,
                self._smcs,
                log_blowup=self._log_blowup,
                jit=self._jit,
            )
            commit_data.append(prep_data)
        commit_data.append(main_data)
        commit_rounds = tuple(
            StackedRound(d.mle, d.digest_layers) for d in commit_data
        )
        # Per-round SMCS commitment for the jagged proof's original_commitments;
        # StackedRound retains only the open witness, so keep it separately.
        commit_commitments = tuple(d.smcs_commitment for d in commit_data)
        carry = replace(
            carry,
            commit_rounds=commit_rounds,
            commit_commitments=commit_commitments,
        )
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
        jit: bool = True,
    ) -> None:
        self._gkr_chips = tuple(gkr_chips)
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._witness = witness
        # Bind the static config to the *class-level* jitted body so every
        # LogupGkrRound shares one `jax.jit` wrapper. JAX's compile cache lives on
        # the wrapper object, so a fresh-per-shard instance (the chain rebuilds
        # with each shard's witness) reuses the compiled executable instead of
        # recompiling. `gkr_chips` is a tuple of frozen GkrChips, so it keys the
        # cache by value -- same machine config, one compile.
        self._body = (
            partial(
                self._jit_body,
                gkr_chips=self._gkr_chips,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
            )
            if jit
            else None
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("gkr_chips", "num_betas", "num_row_variables"))
    def _jit_body(
        main_region,
        prep_region,
        transcript,
        witness,
        *,
        gkr_chips,
        num_betas,
        num_row_variables,
    ):
        # Regions + the post-grind transcript trace as args. `LogUpGkrOutput` is
        # not a registered pytree, so return its two array leaves and let
        # `__call__` rebuild it -- everything else (transcript, round_proofs,
        # openings) already crosses jit.
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
    values itself (``OpenedValuesRound`` in ``zerocheck.prover``); this Round
    threads them onto the carry for the jagged-eval stage's claims and the
    wire's ShardOpenedValues.

    With ``jit=True`` the stage body runs under one cached outer ``@jit``.
    Eagerly, every prove rebuilds the sumcheck engine's ``lax.scan`` bodies,
    and JAX's compile cache (keyed on function identity) misses -- every warm
    prove re-pays the stage compile. Byte-identical either way. ``jit=False``
    stays as a debugging aid; pv-reading constraint circuits are legal under
    ``jit=True`` because the statement rides ``constraint_eval``'s declared
    ``aux_operands`` operand rather than a closure the composite would reject."""

    def __init__(
        self,
        chips: Mapping[str, Chip],
        *,
        max_log_row_count: int,
        jit: bool = True,
    ) -> None:
        self._chips = chips
        self._max_log_row_count = max_log_row_count
        # Class-level jitted body for the same reason as LogupGkrRound: the
        # wrapper (and its compile cache) is shared by every instance, so a
        # rebuilt chain reuses the executable. `chips` keys the cache by the
        # loaded Chip objects' identities (stable for a process-held machine).
        self._body = (
            partial(
                self._jit_body,
                chips=tuple(chips.items()),
                max_log_row_count=max_log_row_count,
            )
            if jit
            else None
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("chips", "max_log_row_count"))
    def _jit_body(
        main_region,
        prep_region,
        public_values,
        eval_point,
        chip_openings,
        transcript,
        *,
        chips,
        max_log_row_count,
    ):
        # ZerocheckProof is not a registered pytree; return its fields (all
        # pytree-crossable) and let `__call__` rebuild it.
        transcript, proof = prove_shard_zerocheck(
            dict(chips),
            main_region,
            prep_region,
            public_values,
            eval_point,
            chip_openings,
            transcript,
            max_log_row_count=max_log_row_count,
        )
        return transcript, (
            proof.batching_challenge,
            proof.gkr_opening_batch_challenge,
            proof.lambda_,
            proof.zeta,
            proof.claimed_sum,
            proof.finals,
            proof.opened_values,
            proof.msgs,
        )

    def __call__(
        self, carry: ShardCarry, transcript: Transcript
    ) -> tuple[ShardCarry, Transcript, ZerocheckProof]:
        if carry.gkr_eval_point is None or carry.gkr_chip_openings is None:
            raise ValueError(
                "zerocheck needs the LogUp-GKR stage's outputs on the carry; "
                "sequence a LogupGkrRound before this Round"
            )
        if self._body is not None:
            transcript, fields = self._body(
                carry.main_region,
                carry.prep_region,
                carry.public_values,
                carry.gkr_eval_point,
                carry.gkr_chip_openings,
                transcript,
            )
            (
                batching_challenge,
                gkr_batch,
                lambda_,
                zeta,
                claimed_sum,
                finals,
                opened_values,
                msgs,
            ) = fields
            proof = ZerocheckProof(
                batching_challenge=batching_challenge,
                gkr_opening_batch_challenge=gkr_batch,
                lambda_=lambda_,
                zeta=zeta,
                claimed_sum=claimed_sum,
                finals=finals,
                opened_values=opened_values,
                msgs=msgs,
            )
        else:
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
    opened values at it, and the committed stacked witness off the carry.

    With ``jit=True`` the stage body (column assembly, the outer/inner
    sumchecks, and the stacked BaseFold open) runs under one cached outer
    ``@jit``. Eagerly, every prove rebuilds the ``fused_region`` marker
    closures and the FRI/grind ``lax.while_loop`` bodies, and JAX's compile
    cache (keyed on function identity) misses -- every warm prove re-pays the
    stage compile. Byte-identical either way."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        num_queries: int,
        pow_bits: int,
        jit: bool = True,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits
        # Class-level jitted body (the LogupGkrRound pattern); `smcs` keys the
        # cache by identity, stable for the chain-held scheme instance.
        self._body = (
            partial(
                self._jit_body,
                smcs=smcs,
                log_blowup=log_blowup,
                num_queries=num_queries,
                pow_bits=pow_bits,
            )
            if jit
            else None
        )

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=("smcs", "log_blowup", "num_queries", "pow_bits"),
    )
    def _jit_body(
        main_region,
        prep_region,
        opened_values,
        zc_sumcheck_point,
        commit_rounds,
        transcript,
        *,
        smcs,
        log_blowup,
        num_queries,
        pow_bits,
    ):
        transcript, eval_msg, open_proof = _jagged_eval_body(
            main_region,
            prep_region,
            opened_values,
            zc_sumcheck_point,
            commit_rounds,
            transcript,
            smcs=smcs,
            log_blowup=log_blowup,
            num_queries=num_queries,
            pow_bits=pow_bits,
        )
        # JaggedEvalMsg is not a registered pytree; return its fields (all
        # arrays) and let `__call__` rebuild it. StackedOpenProof crosses as-is.
        return transcript, (
            eval_msg.outer_sumcheck_claim,
            eval_msg.outer_sumcheck_polys,
            eval_msg.outer_sumcheck_point,
            eval_msg.dense_eval,
            eval_msg.inner_sumcheck_polys,
            eval_msg.inner_point,
            eval_msg.inner_claimed_sum,
        ), open_proof

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
        if self._body is not None:
            transcript, msg_fields, open_proof = self._body(
                carry.main_region,
                carry.prep_region,
                carry.zc_opened_values,
                carry.zc_sumcheck_point,
                carry.commit_rounds,
                transcript,
            )
            eval_msg = JaggedEvalMsg(*msg_fields)
        else:
            transcript, eval_msg, open_proof = _jagged_eval_body(
                carry.main_region,
                carry.prep_region,
                carry.zc_opened_values,
                carry.zc_sumcheck_point,
                carry.commit_rounds,
                transcript,
                smcs=self._smcs,
                log_blowup=self._log_blowup,
                num_queries=self._num_queries,
                pow_bits=self._pow_bits,
            )
        return carry, transcript, ShardJaggedEvalProof(eval=eval_msg, open=open_proof)


def _jagged_eval_body(
    main_region,
    prep_region,
    opened_values,
    zc_sumcheck_point,
    commit_rounds,
    transcript: GrindingTranscript,
    *,
    smcs: SingleMatrixCommitmentScheme,
    log_blowup: int,
    num_queries: int,
    pow_bits: int,
) -> tuple[GrindingTranscript, JaggedEvalMsg, StackedOpenProof]:
    """The jagged-eval stage's traceable body -- the single source both the
    eager path and ``ShardJaggedEvalRound``'s ``@jit`` run."""
    main = main_region
    openings = opened_values
    # The jagged eval runs in the extension field — the upstream sumcheck
    # points are EF challenge lists (one extension sample per variable).
    ef = koalabearx4_mont

    # Per-round (row/column counts, real per-column claims) in [prep, main]
    # order — each chip's opened-values field at the zerocheck point is its
    # columns' claims (SP1's round_evaluation_claims) — plus each region's
    # stacking-aligned dense for the combined committed D.
    rc_rounds: list[Sequence[int]] = []
    cc_rounds: list[Sequence[int]] = []
    claims_rounds: list[Array] = []
    denses: list[Array] = []
    prep = prep_region
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
        # Full region buffer, stacking pad included: col_heights counts each
        # region's pad pair, so the indicator J̃ (and the stacked open) place the
        # next region at the padded offset -- region.dense[:raw_size] would
        # misalign it against J̃.
        denses.append(region.dense)

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
        z_row=zc_sumcheck_point[::-1],
        z_col=z_col,
        dense=dense,
    )
    _, transcript, eval_msg = JaggedEvalRound(dtype=ef)(inputs, transcript)

    code = BitReversedReedSolomon(
        message_len=1 << main.log_stacking_height,
        blowup=1 << log_blowup,
        dtype=main.dense.dtype,
    )
    # commit_rounds carry only mle + digest tree; stacked_basefold_open
    # re-encodes the codeword from the mle. No host<->device round-trip, so the
    # open stays traceable under one whole-chain @jit.
    open_proof, transcript = stacked_basefold_open(
        smcs,
        code,
        commit_rounds,
        eval_msg.outer_sumcheck_point,
        eval_msg.dense_eval,
        main.log_stacking_height,
        num_queries=num_queries,
        pow_bits=pow_bits,
        transcript=transcript,
    )
    return transcript, eval_msg, open_proof


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
    jit: bool = True,
) -> ProveChain:
    """The SP1 shard chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift
    on it.

    ``jit`` stages every stage's heavy body under a cached ``jax.jit``: the
    trace-commit tail (required at rsp scale, see
    ``zorch.pcs.jagged.commit``), the LogUp-GKR body (host-dispatch-bound
    eagerly, sp1-zorch#119), and the zerocheck + jagged-eval bodies — eagerly
    those two rebuild their closure-keyed ``scan``/``while`` bodies each prove,
    so JAX's compile cache misses and every warm prove re-pays the stage
    compile. Byte-identical either way."""
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
            ShardZerocheckRound(
                chips, max_log_row_count=max_log_row_count, jit=jit
            ),
            ShardJaggedEvalRound(
                smcs,
                log_blowup=log_blowup,
                num_queries=open_num_queries,
                pow_bits=open_pow_bits,
                jit=jit,
            ),
        ]
    )
