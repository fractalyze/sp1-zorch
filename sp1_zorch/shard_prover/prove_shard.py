# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof as one zorch ``ProveChain`` of Stages.

``prove_shard_chain`` sequences the stages of ``docs/architecture.md`` —
trace commit, LogUp-GKR, zerocheck, jagged evaluation proof — as Stages
(``zorch.round.Round`` subclasses) threading one duplex transcript and a single
``ShardBridge``. The bridge holds only what a later stage reads from an earlier
one; static configuration (vk, SMCS, chips) lives on the Stage instances.
Proof assembly consumes the chain's message list (fractalyze/sp1-zorch#21).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import frx
import frx.numpy as jnp
from frx import Array
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
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey
from sp1_zorch.zerocheck.jagged import TotalCapClass, pack_flat_arrival
from sp1_zorch.zerocheck.prover import (
    ZerocheckProof,
    chip_traces,
    prove_shard_zerocheck,
)
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.round import ProveChain, Round
from zorch.transcript import GrindingTranscript, Transcript
from zorch.utils.bits import log2_ceil_usize


# Pytree: the two regions (themselves pytrees), public values, and written
# stage outputs are array leaves; unwritten Optional fields are None (an empty
# subtree). Lets the bridge cross the chain's @jit boundary as one donatable
# argument.
@partial(
    frx.tree_util.register_dataclass,
    data_fields=[
        "main_region",
        "prep_region",
        "public_values",
        "commit_digest_layers",
        "commit_commitments",
        "gkr_eval_point",
        "gkr_chip_openings",
        "zc_sumcheck_point",
        "zc_opened_values",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardBridge:
    """What flows between stages: the committed regions plus each stage's
    outputs the next one consumes. Stages return it via ``replace`` —
    a stage writes its own fields and passes the rest through untouched."""

    main_region: JaggedRegion
    prep_region: JaggedRegion | None
    public_values: Array
    # Written by TraceCommitStage; read by the jagged-eval open stage (which
    # rebuilds each StackedRound, recomputing the [S,K] mle from the region dense
    # it already holds) and by proof assembly (the digest-tree root). [prep, main]
    # order, matching SP1's round_evaluation_claims. Only the digest tree is kept
    # here: the mle is a transpose of the region dense, so keeping it too would
    # duplicate the trace on-device through the LogUp-GKR + zerocheck stages.
    commit_digest_layers: tuple[list[Array], ...] | None = None
    # Written by TraceCommitStage; read by proof assembly as the jagged proof's
    # original_commitments -- each round's SMCS commitment (pre-structure-binding),
    # [prep, main] order.
    commit_commitments: tuple[Array, ...] | None = None
    # Written by LogupGkrStage; read by ZerocheckStage.
    gkr_eval_point: Array | None = None
    gkr_chip_openings: Mapping[str, ChipEvaluation] | None = None
    # Written by ZerocheckStage; read by the jagged-eval open stage as its
    # z_row — the accumulated per-round sumcheck challenges, not the GKR zeta.
    zc_sumcheck_point: Array | None = None
    # Written by ZerocheckStage; read by the jagged-eval stage as its
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


class PreambleStage(Round):
    """SP1's shard preamble absorb stream: vk, public values, the main
    commitment, chip metadata. The schedule lives here once — the prover's
    ``TraceCommitStage`` and the byte-match replay's ``preamble_transcript``
    drive this one Stage, so an ordering edit cannot land in one Fiat-Shamir
    stream and not the other (the GKR head schedule got the same treatment in
    ``logup_gkr.head``). Bridge-agnostic; the message is the observed
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
        self, bridge: Any, transcript: Transcript
    ) -> tuple[Any, Transcript, Array]:
        transcript = self._vk.observe_into(transcript)
        transcript = transcript.observe(self._public_values)
        transcript = transcript.observe(self._commitment)
        transcript = transcript.observe(self._chip_metadata)
        return bridge, transcript, self._commitment


class TraceCommitStage(Round):
    """Trace commit plus the shard preamble: commit the main region, then
    absorb SP1's preamble stream via ``PreambleStage``. The message is the
    structure-bound main commitment; the prover-side commit data joins the
    bridge once the opening stage that reads it lands
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
        self, bridge: ShardBridge, transcript: Transcript
    ) -> tuple[ShardBridge, Transcript, Array]:
        bound, main_data = commit_region(
            bridge.main_region,
            self._smcs,
            log_blowup=self._log_blowup,
            jit=self._jit,
        )
        _, transcript, _ = PreambleStage(
            vk=self._vk,
            public_values=bridge.public_values,
            commitment=bound,
            chip_metadata=self._chip_metadata,
        )(bridge, transcript)
        # Keep each region's commit witness for the jagged-eval open, in
        # [prep, main] order (SP1's round_evaluation_claims). prep is bound into
        # the vk at setup, not re-observed here, but the open still reproves it.
        commit_data = []
        if bridge.prep_region is not None:
            # prep uses main's jit knob: an eager commit de-fuses the Merkle
            # fold into many tiny launches.
            _, prep_data = commit_region(
                bridge.prep_region,
                self._smcs,
                log_blowup=self._log_blowup,
                jit=self._jit,
            )
            commit_data.append(prep_data)
        commit_data.append(main_data)
        # Keep only the digest tree; the open recomputes the mle from the region
        # dense (mle == dense.reshape(K, S).T) instead of holding a trace-sized
        # copy through GKR + zerocheck. The mles in commit_data drop at return.
        commit_digest_layers = tuple(d.digest_layers for d in commit_data)
        # Per-round SMCS commitment for the jagged proof's original_commitments;
        # kept separately from the open witness.
        commit_commitments = tuple(d.smcs_commitment for d in commit_data)
        bridge = replace(
            bridge,
            commit_digest_layers=commit_digest_layers,
            commit_commitments=commit_commitments,
        )
        return bridge, transcript, bound


class LogupGkrStage(Round):
    """LogUp-GKR stage over ``prove_logup_gkr``; writes the final evaluation
    point and per-chip openings onto the bridge for zerocheck.

    Eager, not jitted: a whole-body ``@jit`` keeps every pyramid layer live at
    once (XLA owns liveness) and overflows the memory budget on wide shards, and
    the grind's host-side ``pow_bits`` verdict cannot be traced."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        witness: Array | None = None,
    ) -> None:
        self._gkr_chips = tuple(gkr_chips)
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._witness = witness

    def __call__(
        self, bridge: ShardBridge, transcript: Transcript
    ) -> tuple[ShardBridge, Transcript, LogupGkrProof]:
        transcript, proof = prove_logup_gkr(
            self._gkr_chips,
            bridge.main_region,
            bridge.prep_region,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
            witness=self._witness,
        )
        bridge = replace(
            bridge,
            gkr_eval_point=proof.eval_point,
            gkr_chip_openings=proof.chip_openings,
        )
        return bridge, transcript, proof


class ZerocheckStage(Round):
    """Zerocheck stage over ``prove_shard_zerocheck``, consuming the GKR
    point and openings off the bridge. The stage absorbs the per-chip opened
    values itself (``OpenedValuesRound`` in ``zerocheck.prover``); this Stage
    threads them onto the bridge for the jagged-eval stage's claims and the
    wire's ShardOpenedValues.

    The stage body runs under one cached outer ``@jit`` on the total-cap
    contract (fractalyze/sp1-zorch#242): a ``TotalCapClass`` bounds the one
    flat jagged round buffer, the arrival is packed to the class shape in an
    eager prologue, and the shard's real heights ride as one traced int32
    vector, so the body's compile keys on the class and the chip set alone --
    shards that differ only in row counts share one executable (exact heights
    bust the cache: 22 distinct shape signatures across the 25-shard rsp
    block). With no class pinned, the shard's own a-priori-tight class is
    derived (per-shard compile, same body). pv-reading constraint circuits
    are legal because the statement rides ``constraint_eval``'s declared
    ``aux_operands`` operand, not a closure the composite would reject.
    Byte-identical to an eager exact-heights prove, and CPU-executable (the
    former eager-only fallback was a stale fractalyze/frx#168 workaround)."""

    def __init__(
        self,
        chips: Mapping[str, Chip],
        *,
        max_log_row_count: int,
        total_cap_class: TotalCapClass | None = None,
    ) -> None:
        self._chips = chips
        self._max_log_row_count = max_log_row_count
        self._total_cap_class = total_cap_class

    @staticmethod
    @partial(
        frx.jit,
        static_argnames=(
            "chips", "max_log_row_count", "total_cap_class", "chip_names",
            "num_cols", "main_widths", "prep_widths",
        ),
    )
    def _jit_body_totalcap_traced(
        flat_arrival,
        public_values,
        eval_point,
        chip_openings,
        num_reals,
        transcript,
        *,
        chips,
        max_log_row_count,
        total_cap_class,
        chip_names,
        num_cols,
        main_widths,
        prep_widths,
    ):
        # The shard-invariant total-cap body (sp1-zorch#242): the arrival is
        # the ONE class-shaped flat jagged buffer (`pack_flat_arrival`) and
        # the shard's real heights ride in `num_reals` (one traced int32
        # vector); every other per-chip datum is a class-level static. The
        # compile keys on (chips, total_cap_class, the static tuples) alone —
        # shards of one class share the executable, and no per-shard region
        # shape enters the cache key.
        transcript, proof = prove_shard_zerocheck(
            dict(chips),
            None,
            None,
            public_values,
            eval_point,
            chip_openings,
            transcript,
            max_log_row_count=max_log_row_count,
            num_reals=[num_reals[i] for i in range(len(chip_names))],
            total_cap_class=total_cap_class,
            flat_arrival=flat_arrival,
            num_cols=num_cols,
            main_widths=main_widths,
            prep_widths=prep_widths,
            chip_names=chip_names,
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
        self, bridge: ShardBridge, transcript: Transcript
    ) -> tuple[ShardBridge, Transcript, ZerocheckProof]:
        if bridge.gkr_eval_point is None or bridge.gkr_chip_openings is None:
            raise ValueError(
                "zerocheck needs the LogUp-GKR stage's outputs on the bridge; "
                "sequence a LogupGkrStage before this Stage"
            )
        # Shard-invariant flat prologue (sp1-zorch#242): pack the
        # class-shaped flat jagged arrival EAGERLY from the exact-height
        # traces — heights are host ints here, and the pack mirrors the
        # cols*evenpad(h) cumsum the traced body derives, so the layouts
        # agree. No chip pads to the class window (a wide class made that
        # uniform 2W padding overflow int32 element indexing and dwarf the
        # live area); the arrival is live rows + zeros, in the base field.
        names = bridge.main_region.chip_names
        heights_host = [int(h) for h in bridge.main_region.chip_heights]
        traces = chip_traces(
            names, heights_host, bridge.main_region, bridge.prep_region
        )
        # No pinned class: derive this shard's own a-priori-tight class
        # (per-shard compile, same traced body).
        total_cap_class = self._total_cap_class or TotalCapClass.from_heights(
            heights_host, [int(t.shape[0]) for t in traces]
        )
        flat = pack_flat_arrival(traces, heights_host, total_cap_class)
        prep_w = (
            {
                n: int(w)
                for n, w in zip(
                    bridge.prep_region.chip_names,
                    bridge.prep_region.chip_widths,
                )
            }
            if bridge.prep_region is not None
            else {}
        )
        transcript, fields = self._jit_body_totalcap_traced(
            flat,
            bridge.public_values,
            bridge.gkr_eval_point,
            bridge.gkr_chip_openings,
            jnp.asarray(heights_host, jnp.int32),
            transcript,
            chips=tuple(self._chips.items()),
            max_log_row_count=self._max_log_row_count,
            total_cap_class=total_cap_class,
            chip_names=tuple(names),
            num_cols=tuple(int(t.shape[0]) for t in traces),
            main_widths=tuple(int(w) for w in bridge.main_region.chip_widths),
            prep_widths=tuple(prep_w.get(n, 0) for n in names),
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
        bridge = replace(
            bridge,
            zc_sumcheck_point=proof.msgs.challenge,
            zc_opened_values=proof.opened_values,
        )
        return bridge, transcript, proof


@dataclass(frozen=True)
class JaggedPcsProof:
    """The jagged evaluation proof: the outer/inner sumcheck reducing the
    committed trace to ``D(z_final)``, then the stacked BaseFold open of ``D``
    at that point."""

    eval: JaggedEvalMsg
    open: StackedOpenProof


class JaggedPcsStage(Round):
    """Jagged evaluation proof (SP1 Phase 4): reduce the committed trace to
    ``D(z_final)`` via the outer/inner sumcheck, then open ``D`` at ``z_final``
    with the stacked BaseFold FRI. Reads the zerocheck point, the per-chip
    opened values at it, and the committed stacked witness off the bridge.

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
        # Class-level jitted body (the LogupGkrStage pattern); `smcs` keys the
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
        frx.jit,
        static_argnames=("smcs", "log_blowup", "num_queries", "pow_bits"),
    )
    def _jit_body(
        main_region,
        prep_region,
        opened_values,
        zc_sumcheck_point,
        commit_digest_layers,
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
            commit_digest_layers,
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
        self, bridge: ShardBridge, transcript: GrindingTranscript
    ) -> tuple[ShardBridge, GrindingTranscript, JaggedPcsProof]:
        if (
            bridge.zc_sumcheck_point is None
            or bridge.commit_digest_layers is None
            or bridge.zc_opened_values is None
        ):
            raise ValueError(
                "the jagged-eval stage needs the zerocheck point, committed "
                "digest layers, and zerocheck opened values on the bridge; sequence "
                "the commit, LogUp-GKR, and zerocheck Stages before it"
            )
        if self._body is not None:
            transcript, msg_fields, open_proof = self._body(
                bridge.main_region,
                bridge.prep_region,
                bridge.zc_opened_values,
                bridge.zc_sumcheck_point,
                bridge.commit_digest_layers,
                transcript,
            )
            eval_msg = JaggedEvalMsg(*msg_fields)
        else:
            transcript, eval_msg, open_proof = _jagged_eval_body(
                bridge.main_region,
                bridge.prep_region,
                bridge.zc_opened_values,
                bridge.zc_sumcheck_point,
                bridge.commit_digest_layers,
                transcript,
                smcs=self._smcs,
                log_blowup=self._log_blowup,
                num_queries=self._num_queries,
                pow_bits=self._pow_bits,
            )
        return bridge, transcript, JaggedPcsProof(eval=eval_msg, open=open_proof)


def _region_mle(region: JaggedRegion) -> Array:
    """The commit's ``[S, K]`` stacked message for a region, recomputed from its
    dense buffer (``mle == dense.reshape(K, S).T``, S = 2^log_stacking_height) --
    identical to what ``commit_region`` returned. The open reconstructs it here
    instead of the bridge pinning a trace-sized copy through the whole chain
    (fractalyze/sp1-zorch#264)."""
    S = 1 << region.log_stacking_height
    K = region.dense.shape[0] // S
    return region.dense.reshape(K, S).T


def _jagged_eval_body(
    main_region,
    prep_region,
    opened_values,
    zc_sumcheck_point,
    commit_digest_layers,
    transcript: GrindingTranscript,
    *,
    smcs: SingleMatrixCommitmentScheme,
    log_blowup: int,
    num_queries: int,
    pow_bits: int,
) -> tuple[GrindingTranscript, JaggedEvalMsg, StackedOpenProof]:
    """The jagged-eval stage's traceable body -- the single source both the
    eager path and ``JaggedPcsStage``'s ``@jit`` run."""
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
    # Rebuild each StackedRound here: the mle is recomputed from the region dense
    # (the loop above already reads it for `denses`), joined to the carried digest
    # tree, in the same [prep, main] order. stacked_basefold_open re-encodes the
    # codeword from the mle -- no host<->device round-trip, still traceable.
    commit_rounds = tuple(
        StackedRound(_region_mle(region), digests)
        for (region, _), digests in zip(regions, commit_digest_layers, strict=True)
    )
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
    zerocheck_total_cap_class: TotalCapClass | None = None,
) -> ProveChain:
    """The SP1 shard chain. One definition for the stage wiring so the
    benchmark, the byte-match runnables, and proof assembly cannot drift
    on it.

    ``jit`` stages every stage's heavy body under a cached ``frx.jit``: the
    trace-commit tail (required at rsp scale, see
    ``zorch.pcs.jagged.commit``), the LogUp-GKR body (host-dispatch-bound
    eagerly, sp1-zorch#119), and the zerocheck + jagged-eval bodies — eagerly
    those two rebuild their closure-keyed ``scan``/``while`` bodies each prove,
    so JAX's compile cache misses and every warm prove re-pays the stage
    compile. Byte-identical either way."""
    return ProveChain(
        [
            TraceCommitStage(
                smcs,
                log_blowup=log_blowup,
                vk=vk,
                chip_metadata=chip_metadata,
                jit=jit,
            ),
            # GKR is always eager (see LogupGkrStage); only the other stages
            # take the `jit` knob.
            LogupGkrStage(
                gkr_chips,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=pow_bits,
                witness=witness,
            ),
            ZerocheckStage(
                chips,
                max_log_row_count=max_log_row_count,
                total_cap_class=zerocheck_total_cap_class,
            ),
            JaggedPcsStage(
                smcs,
                log_blowup=log_blowup,
                num_queries=open_num_queries,
                pow_bits=open_pow_bits,
                jit=jit,
            ),
        ]
    )
