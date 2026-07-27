# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof as one composite zorch Stage.

``ShardProver`` reduces the public shard statement to the trivial claim over
one duplex transcript. The jagged PCS commits the trace, then three
``ProverStage`` roles — LogUp-GKR, zerocheck, jagged opening — each discharge
the claim the one before produced, so what crosses a seam is a claim both
roles derive rather than a shared mutable carry. The PCS's own prover data
spans its two halves as a local in ``ShardProver.prove``, belonging to no
claim. Static configuration (SMCS, chips, caps) lives on the role instances,
the statement on ``ShardClaim``, and the trace on ``ShardWitness``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
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
    JaggedEvalMsg,
    assemble_col_heights,
    assemble_columns,
    eval_column_arrays,
    eval_round_core,
    sample_z_col,
)
from sp1_zorch.logup_gkr.circuit import GkrCapClass, GkrChip
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
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
)
from zorch.transcript import GrindingTranscript, Transcript
from zorch.utils.bits import log2_ceil_usize


# Pytree: the two regions (themselves pytrees), public values, and written
# stage outputs are array leaves; unwritten Optional fields are None (an empty
# subtree). Lets the witness cross a @jit boundary as one donatable
# argument.
@dataclass(frozen=True)
class ChipMetadata:
    """Which chips this shard holds and how many real rows each one has, in
    SP1's chip order.

    The claim-side half of the trace dimensions: row counts change shard to
    shard, so the statement has to give them, while column counts are fixed by
    each chip's AIR and stay role configuration (`ChipWidths`). Held as values
    rather than as the absorb stream they encode — `preamble_stream` derives
    that — so both roles read the same statement instead of a blob only the
    transcript can consume.
    """

    chip_names: tuple[str, ...]
    num_reals: tuple[int, ...]

    def by_chip(self) -> dict[str, int]:
        return dict(zip(self.chip_names, self.num_reals, strict=True))

    def preamble_stream(self, *, dtype: Any) -> Array:
        """The preamble's chip-metadata stream as one flat array: chip count,
        then per chip (num_real, name length, name bytes). One flat absorb
        matches SP1's per-value observes byte-for-byte while skipping hundreds
        of single-element transcript calls."""
        metadata: list[int] = [len(self.chip_names)]
        for name, num_real in zip(self.chip_names, self.num_reals, strict=True):
            metadata.append(int(num_real))
            metadata.append(len(name))
            metadata.extend(name.encode("ascii"))
        return fnp.array(metadata, dtype)


@dataclass(frozen=True)
class ShardClaim:
    """The public shard statement: what a verifier holds before any proof."""

    vk: MachineVerifyingKey
    public_values: Array
    chip_metadata: ChipMetadata


@partial(
    frx.tree_util.register_dataclass,
    data_fields=["main_region", "prep_region"],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardWitness:
    """The prover's private trace: the committed regions themselves.

    A pytree, so the whole witness crosses a ``@jit`` boundary as one donated
    argument. Its leaves are exactly the regions' dense buffers — a `None`
    prep region is an empty subtree and contributes none.
    """

    main_region: JaggedRegion
    prep_region: JaggedRegion | None = None


@dataclass(frozen=True)
class GkrOutputClaim:
    """What LogUp-GKR reduces its statement to: the input-layer evaluation
    point and the per-chip openings there. Both roles derive it — the prover
    from its own chain, the verifier by replaying the proof."""

    eval_point: Array
    chip_openings: Mapping[str, ChipEvaluation]


@dataclass(frozen=True)
class ZerocheckClaim:
    """Zerocheck's source claim: the public values its constraint evaluation
    folds, plus the LogUp-GKR reduction it is conditional on."""

    public_values: Array
    gkr: GkrOutputClaim
    chip_metadata: ChipMetadata


@dataclass(frozen=True)
class TraceEvaluationClaim:
    """What zerocheck reduces to and the jagged opening discharges: the trace
    evaluates to `opened_values` at `point`."""

    point: Array
    opened_values: Mapping[str, ChipEvaluation]


@dataclass(frozen=True, kw_only=True)
class BoundRoots:
    """The structure-bound Merkle roots an opening is checked against.

    Both roles derive these — the verifier from the vk and the proof's
    commitment — so they are claim data. The raw `SmcsCommitments` the prover
    also holds are the same two arrays before structure binding, so only a
    distinct type stops the two being passed to each other's slot; the fields
    are keyword-only for the same reason.
    """

    preprocessed: Array
    main: Array

    def in_round_order(self, has_preprocessed: bool) -> list[Array]:
        """The roots the opening binds against, in SP1's round order."""
        return [self.preprocessed, self.main] if has_preprocessed else [self.main]


@dataclass(frozen=True, kw_only=True)
class SmcsCommitments:
    """Per-round SMCS commitments before structure binding — the wire's
    ``original_commitments``.

    Prover output the verifier cannot derive, so this is proof data, not claim
    data. `preprocessed` is absent when the shard commits no preprocessed
    region, which is why the arity varies where `BoundRoots` is fixed: SP1's
    verifier always carries the vk's preprocessed commitment even when the
    prover has no preprocessed region to commit.
    """

    preprocessed: Array | None = None
    main: Array

    def in_round_order(self) -> list[Array]:
        return (
            [self.preprocessed, self.main]
            if self.preprocessed is not None
            else [self.main]
        )


@dataclass(frozen=True)
class JaggedOpeningClaim:
    """The opening statement: the committed trace evaluates to
    `evaluation.opened_values` at `evaluation.point`, under `roots`."""

    evaluation: TraceEvaluationClaim
    roots: BoundRoots
    chip_metadata: ChipMetadata


@dataclass(frozen=True)
class JaggedCommitData:
    """What the jagged PCS's commit half retains for its open half.

    Fiat-Shamir splits the two halves across the whole shard proof — the
    commitment must bind the transcript before LogUp-GKR draws a challenge,
    and the open cannot run until zerocheck produces the point to open at.
    This is the scheme's own prover data bridging that gap, per round in
    [prep, main] order; it is not a value any claim carries.
    """

    digest_layers: tuple[list[Array], ...]
    commitments: SmcsCommitments


@dataclass(frozen=True)
class JaggedOpeningWitness:
    """The shard's trace plus the prover data the commit half produced."""

    trace: ShardWitness
    commit_data: JaggedCommitData


@dataclass(frozen=True)
class ShardProof:
    """The wire sections: the trace commitment, then one reduction proof per
    Stage."""

    commitment: Array
    gkr: LogupGkrProof
    zerocheck: ZerocheckProof
    jagged: JaggedPcsProof


def absorb_preamble(
    transcript: Transcript,
    *,
    vk: MachineVerifyingKey,
    public_values: Array,
    commitment: Array,
    chip_metadata: Array,
) -> Transcript:
    """SP1's shard preamble absorb stream: vk, public values, the main
    commitment, chip metadata.

    A transcript-only schedule operation, so it is one shared function both
    roles call rather than a stage: the prover, the verifier dual, and the
    byte-match replay's ``preamble_transcript`` run this single definition, and
    an ordering edit cannot land in one Fiat-Shamir stream and not the other
    (the GKR head schedule has the same treatment in ``logup_gkr.head``).
    """
    transcript = vk.observe_into(transcript)
    transcript = transcript.observe(public_values)
    transcript = transcript.observe(commitment)
    return transcript.observe(chip_metadata)


def bind_commitment(
    transcript: Transcript, claim: ShardClaim, commitment: Array
) -> tuple[Transcript, BoundRoots]:
    """Bind the committed trace into the stream and name the roots it is
    opened against — what both composites do between the PCS's two halves.

    The prover has just committed and the verifier has just read the
    commitment off the wire; from here their transcripts must agree, so both
    reach that state through this one function rather than two copies of the
    same two steps.

    The prep root is unconditional: SP1's verifier always carries the vk's
    preprocessed commitment, even though the prover keeps ``prep_region``
    optional. The stacked-open dual checking openings against these roots is
    where a no-prep proof would reconcile.
    """
    transcript = absorb_preamble(
        transcript,
        vk=claim.vk,
        public_values=claim.public_values,
        commitment=commitment,
        chip_metadata=claim.chip_metadata.preamble_stream(
            dtype=claim.public_values.dtype
        ),
    )
    return transcript, BoundRoots(
        preprocessed=claim.vk.preprocessed_commit, main=commitment
    )


class LogupGkrProver(
    ProverStage[ShardClaim, ShardWitness, GkrOutputClaim, LogupGkrProof]
):
    """LogUp-GKR stage over ``prove_logup_gkr``; writes the final
    evaluation point and per-chip openings as the claim zerocheck consumes.

    Eager orchestration, not one ``@jit`` body: a whole-body jit keeps
    every pyramid layer live at once (OOM on wide shards) and the grind's
    host-side ``pow_bits`` verdict cannot be traced. Every traced zone
    underneath keys its compile on (chip set, ``GkrCapClass``) — shards of
    one class share every executable; no pinned class means
    the shard's own tight class (per-shard compile, same body)."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        witness: Array | None = None,
        gkr_cap_class: GkrCapClass | None = None,
    ) -> None:
        self._gkr_chips = tuple(gkr_chips)
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._witness = witness
        self._gkr_cap_class = gkr_cap_class

    def prove(
        self,
        claim: ShardClaim,
        witness: ShardWitness,
        transcript: Transcript,
    ) -> ProveResult[GkrOutputClaim, LogupGkrProof]:
        del claim  # the GKR statement is the chip set, fixed on the role
        transcript, proof = prove_logup_gkr(
            self._gkr_chips,
            witness.main_region,
            witness.prep_region,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
            witness=self._witness,
            cap_class=self._gkr_cap_class,
        )
        return ProveResult(
            GkrOutputClaim(proof.eval_point, proof.chip_openings),
            proof,
            transcript,
        )


class ZerocheckProver(
    ProverStage[ZerocheckClaim, ShardWitness, TraceEvaluationClaim, ZerocheckProof]
):
    """Zerocheck stage over ``prove_shard_zerocheck``, consuming the GKR
    point and openings off its source claim. The stage absorbs the per-chip opened
    values itself (``OpenedValuesRound`` in ``zerocheck.prover``); this Stage
    surfaces them in its reduced claim for the jagged opening and the
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
            "chips",
            "max_log_row_count",
            "total_cap_class",
            "chip_names",
            "num_cols",
            "main_widths",
            "prep_widths",
        ),
    )
    def _jit_body_totalcap_traced(
        flat_arrival: Array,
        public_values: Array,
        eval_point: Array,
        chip_openings: Mapping[str, ChipEvaluation],
        num_reals: Array,
        transcript: Transcript,
        *,
        chips: tuple[tuple[str, Chip], ...],
        max_log_row_count: int,
        total_cap_class: TotalCapClass,
        chip_names: tuple[str, ...],
        num_cols: tuple[int, ...],
        main_widths: tuple[int, ...],
        prep_widths: tuple[int, ...],
    ) -> tuple[Transcript, tuple[Any, ...]]:
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

    def prove(
        self,
        claim: ZerocheckClaim,
        witness: ShardWitness,
        transcript: Transcript,
    ) -> ProveResult[TraceEvaluationClaim, ZerocheckProof]:
        # Shard-invariant flat prologue (sp1-zorch#242): pack the
        # class-shaped flat jagged arrival EAGERLY from the exact-height
        # traces — heights are host ints here, and the pack mirrors the
        # cols*evenpad(h) cumsum the traced body derives, so the layouts
        # agree. No chip pads to the class window (a wide class made that
        # uniform 2W padding overflow int32 element indexing and dwarf the
        # live area); the arrival is live rows + zeros, in the base field.
        names = witness.main_region.chip_names
        heights_host = [int(h) for h in witness.main_region.chip_heights]
        traces = chip_traces(
            names, heights_host, witness.main_region, witness.prep_region
        )
        # No pinned class: derive this shard's own a-priori-tight class
        # (per-shard compile, same traced body).
        total_cap_class = self._total_cap_class or TotalCapClass.from_heights(
            heights_host, [int(t.shape[0]) for t in traces]
        )
        flat = pack_flat_arrival(
            traces, heights_host, total_cap_class, self._max_log_row_count
        )
        prep_w = (
            {
                n: int(w)
                for n, w in zip(
                    witness.prep_region.chip_names,
                    witness.prep_region.chip_widths,
                )
            }
            if witness.prep_region is not None
            else {}
        )
        transcript, fields = self._jit_body_totalcap_traced(
            flat,
            claim.public_values,
            claim.gkr.eval_point,
            claim.gkr.chip_openings,
            fnp.asarray(heights_host, fnp.int32),
            transcript,
            chips=tuple(self._chips.items()),
            max_log_row_count=self._max_log_row_count,
            total_cap_class=total_cap_class,
            chip_names=tuple(names),
            num_cols=tuple(int(t.shape[0]) for t in traces),
            main_widths=tuple(int(w) for w in witness.main_region.chip_widths),
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
        return ProveResult(
            TraceEvaluationClaim(proof.msgs.challenge, proof.opened_values),
            proof,
            transcript,
        )


@dataclass(frozen=True)
class JaggedPcsProof:
    """The jagged evaluation proof: the outer/inner sumcheck reducing the
    committed trace to ``D(z_final)``, then the stacked BaseFold open of ``D``
    at that point."""

    eval: JaggedEvalMsg
    open: StackedOpenProof
    original_commitments: SmcsCommitments


@partial(frx.jit, static_argnames=("rc_rounds", "cc_rounds", "target", "dtype"))
def _jagged_pack_jit(
    denses: list[Array],
    claims_chips: list[list[Array]],
    *,
    rc_rounds: tuple[tuple[int, ...], ...],
    cc_rounds: tuple[tuple[int, ...], ...],
    target: int,
    dtype: Any,
) -> tuple[Array, Array]:
    """Pack zone: the tier-padded combined dense and the ordered claim buffer
    as one fused executable — eagerly these are several full-buffer copies per
    prove. Keyed per region shape tuple (a cheap concat/pad graph)."""
    claims_rounds = [fnp.concatenate(chips) for chips in claims_chips]
    _, all_claims = assemble_columns(
        list(rc_rounds), list(cc_rounds), claims_rounds, dtype=dtype
    )
    dense = fnp.concatenate(denses)
    return fnp.pad(dense, (0, target - dense.shape[0])), all_claims


@partial(frx.jit, static_argnames=("num_columns", "dtype"))
def _jagged_eval_jit(
    offsets: Array,
    merged: Array,
    all_claims: Array,
    dense: Array,
    zc_sumcheck_point: Array,
    transcript: GrindingTranscript,
    *,
    num_columns: int,
    dtype: Any,
) -> tuple[GrindingTranscript, JaggedEvalMsg]:
    """The eval half (outer/inner sumcheck) as one shard-invariant ``@jit``
    zone: per-shard column heights ride only as the VALUES of the traced
    ``offsets``/``merged`` arrays and ``dense`` arrives pre-padded to its
    power-of-two tier, so the compile keys on the layout class alone
    (chip set + area tier) — shards differing only in heights share the
    executable."""
    transcript, z_col = sample_z_col(transcript, num_columns, dtype)
    weights = expand_eq_to_hypercube(z_col, fnp.ones((), dtype))[:num_columns]
    # z_row is the zerocheck sumcheck point in SP1's insert-at-front
    # (reversed) order.
    eval_msg, transcript = eval_round_core(
        offsets,
        merged,
        weights,
        all_claims,
        dense,
        zc_sumcheck_point[::-1],
        z_col,
        transcript,
        dtype=dtype,
    )
    return transcript, eval_msg


class JaggedPcsProver(
    ProverStage[JaggedOpeningClaim, JaggedOpeningWitness, TrivialClaim, JaggedPcsProof]
):
    """The jagged PCS, whose two halves bracket the shard proof.

    ``commit`` packs and Merkle-commits the trace regions; ``prove`` is the
    open — reduce the committed trace to ``D(z_final)`` via the outer/inner
    sumcheck, then open ``D`` at ``z_final`` with the stacked BaseFold FRI,
    reading the zerocheck point and the per-chip opened values off its claim.
    Only the open reduces a claim, so only the open is the Stage role; the
    commit is the scheme's other half, and ``JaggedCommitData`` is what it
    hands forward.

    Eager orchestration over shard-invariant jitted zones (the LogupGkrStage
    pattern, sp1-zorch#274): the prologue folds per-shard heights into traced
    array values and pads the combined dense to its power-of-two tier, so the
    eval zone's compile keys on the layout class alone; the stacked open runs
    zorch's zoned ``stacked_basefold_open`` (dominant fold zone K-independent).
    Byte-identical to the eager path."""

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
        self._jit = jit

    def commit(self, witness: ShardWitness) -> tuple[Array, JaggedCommitData]:
        """Commit the trace regions; returns the bound main commitment and the
        prover data the open replays.

        No transcript argument: committing is not a transcript operation. The
        composite absorbs the returned commitment through ``absorb_preamble``,
        so the Fiat-Shamir binding has one visible home rather than hiding
        inside the scheme.
        """
        bound, main_data = commit_region(
            witness.main_region,
            self._smcs,
            log_blowup=self._log_blowup,
            jit=self._jit,
        )
        # Per-round in [prep, main] order (SP1's round_evaluation_claims). prep
        # is bound into the vk at setup, not re-observed here, but the open
        # still reproves it.
        commit_data = []
        if witness.prep_region is not None:
            # prep uses main's jit knob: an eager commit de-fuses the Merkle
            # fold into many tiny launches.
            _, prep_data = commit_region(
                witness.prep_region,
                self._smcs,
                log_blowup=self._log_blowup,
                jit=self._jit,
            )
            commit_data.append(prep_data)
        commit_data.append(main_data)
        # Keep only the digest tree; the open recomputes the mle from the region
        # dense (mle == dense.reshape(K, S).T) instead of holding a trace-sized
        # copy through GKR + zerocheck. The mles in commit_data drop at return.
        smcs = [d.smcs_commitment for d in commit_data]
        return bound, JaggedCommitData(
            digest_layers=tuple(d.digest_layers for d in commit_data),
            commitments=SmcsCommitments(
                main=smcs[-1], preprocessed=smcs[0] if len(smcs) > 1 else None
            ),
        )

    def prove(
        self,
        claim: JaggedOpeningClaim,
        witness: JaggedOpeningWitness,
        transcript: GrindingTranscript,
    ) -> ProveResult[TrivialClaim, JaggedPcsProof]:
        main = witness.trace.main_region
        openings = claim.evaluation.opened_values
        zc_point = claim.evaluation.point
        # The jagged eval runs in the extension field — the upstream sumcheck
        # points are EF challenge lists (one extension sample per variable).
        ef = koalabearx4_mont

        # Per-round (row/column counts, real per-column claims) in [prep, main]
        # order — each chip's opened-values field at the zerocheck point is its
        # columns' claims (SP1's round_evaluation_claims) — plus each region's
        # stacking-aligned dense for the combined committed D.
        rc_rounds: list[Sequence[int]] = []
        cc_rounds: list[Sequence[int]] = []
        claims_chips: list[list[Array]] = []
        denses: list[Array] = []
        prep = witness.trace.prep_region
        regions = ([(prep, "preprocessed")] if prep is not None else []) + [
            (main, "main")
        ]
        for region, claim_field in regions:
            rc_rounds.append(region.row_counts)
            cc_rounds.append(region.column_counts)
            claims_chips.append(
                [getattr(openings[n], claim_field) for n in region.chip_names]
            )
            # Full region buffer, stacking pad included: col_heights counts each
            # region's pad pair, so the indicator J̃ (and the stacked open) place
            # the next region at the padded offset -- region.dense[:raw_size]
            # would misalign it against J̃.
            denses.append(region.dense)

        col_heights = assemble_col_heights(rc_rounds, cc_rounds)
        # Heights become traced-array VALUES here, off the eval zone's
        # compile key.
        offsets, merged = eval_column_arrays(col_heights, dtype=ef)
        # The combined dense pads to its power-of-two tier: raw region lengths
        # vary within a class, the padded tier does not — only the padded form
        # may cross into the eval zone.
        target = 1 << log2_ceil_usize(sum(int(d.shape[0]) for d in denses))

        if self._jit:
            dense, all_claims = _jagged_pack_jit(
                denses,
                claims_chips,
                rc_rounds=tuple(tuple(rc) for rc in rc_rounds),
                cc_rounds=tuple(tuple(cc) for cc in cc_rounds),
                target=target,
                dtype=ef,
            )
            transcript, eval_msg = _jagged_eval_jit(
                offsets,
                merged,
                all_claims,
                dense,
                zc_point,
                transcript,
                num_columns=len(col_heights),
                dtype=ef,
            )
            # Free the eval leg's buffers (padded dense is GiB-scale) before
            # the open allocates its [N, K] round codewords.
            del dense, offsets, merged, all_claims
        else:
            claims_rounds = [fnp.concatenate(chips) for chips in claims_chips]
            _, all_claims = assemble_columns(
                rc_rounds, cc_rounds, claims_rounds, dtype=ef
            )
            dense = fnp.concatenate(denses)
            dense = fnp.pad(dense, (0, target - dense.shape[0]))
            transcript, z_col = sample_z_col(transcript, len(col_heights), ef)
            weights = expand_eq_to_hypercube(z_col, fnp.ones((), ef))[
                : len(col_heights)
            ]
            eval_msg, transcript = eval_round_core(
                offsets,
                merged,
                weights,
                all_claims,
                dense,
                zc_point[::-1],
                z_col,
                transcript,
                dtype=ef,
            )

        code = BitReversedReedSolomon(
            message_len=1 << main.log_stacking_height,
            blowup=1 << self._log_blowup,
            dtype=main.dense.dtype,
        )
        # Rebuild each StackedRound from the region's [K, S] block view (no
        # copy), joined to the carried digest tree, in [prep, main] order.
        commit_rounds = tuple(
            StackedRound(region.block, digests)
            for (region, _), digests in zip(
                regions, witness.commit_data.digest_layers, strict=True
            )
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
        return ProveResult(
            TrivialClaim(),
            JaggedPcsProof(
                eval=eval_msg,
                open=open_proof,
                original_commitments=witness.commit_data.commitments,
            ),
            transcript,
        )


class ShardProver(ProverStage[ShardClaim, ShardWitness, TrivialClaim, ShardProof]):
    """The SP1 shard prover: the jagged PCS commit, then three Stages.

    A composite role, so the wiring has one definition and the benchmark, the
    byte-match runnables, and proof assembly cannot drift on it. Three Stages
    reduce the shard statement to the trivial claim — LogUp-GKR, zerocheck,
    jagged opening — each one's reduced claim the next one's source claim.
    They are bracketed by the PCS: ``opening.commit`` binds the trace up front
    and ``opening.prove`` discharges it at the end, with ``JaggedCommitData``
    held here in between because it belongs to neither claim.

    Reduces to the trivial claim: the jagged opening is terminal, so a shard
    proof is a complete argument rather than one link in a chain.

    ``jit`` stages every heavy body under a cached ``frx.jit``: the
    trace-commit tail (required at rsp scale), the zerocheck body, and the
    jagged-eval sumcheck zone (its stacked open always runs zorch's zoned
    jits) — eagerly the sumcheck bodies rebuild their closure-keyed
    ``scan``/``while`` bodies each prove, so JAX's compile cache misses and
    every warm prove re-pays that compile. LogUp-GKR is always eager
    orchestration over class-keyed inner zones. Byte-identical either way.
    """

    def __init__(
        self,
        *,
        smcs: SingleMatrixCommitmentScheme,
        log_blowup: int,
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
        gkr_cap_class: GkrCapClass | None = None,
    ) -> None:
        self.gkr = LogupGkrProver(
            gkr_chips,
            num_betas=num_betas,
            num_row_variables=num_row_variables,
            pow_bits=pow_bits,
            witness=witness,
            gkr_cap_class=gkr_cap_class,
        )
        self.zerocheck = ZerocheckProver(
            chips,
            max_log_row_count=max_log_row_count,
            total_cap_class=zerocheck_total_cap_class,
        )
        self.opening = JaggedPcsProver(
            smcs,
            log_blowup=log_blowup,
            num_queries=open_num_queries,
            pow_bits=open_pow_bits,
            jit=jit,
        )

    def prove(
        self,
        claim: ShardClaim,
        witness: ShardWitness,
        transcript: GrindingTranscript,
    ) -> ProveResult[TrivialClaim, ShardProof]:
        commitment, commit_data = self.opening.commit(witness)
        transcript, roots = bind_commitment(transcript, claim, commitment)
        gkr = self.gkr.prove(claim, witness, transcript)
        zerocheck = self.zerocheck.prove(
            ZerocheckClaim(claim.public_values, gkr.reduced_claim, claim.chip_metadata),
            witness,
            gkr.transcript,
        )
        opening = self.opening.prove(
            JaggedOpeningClaim(zerocheck.reduced_claim, roots, claim.chip_metadata),
            JaggedOpeningWitness(witness, commit_data),
            zerocheck.transcript,
        )
        return ProveResult(
            TrivialClaim(),
            ShardProof(
                commitment=commitment,
                gkr=gkr.reduction_proof,
                zerocheck=zerocheck.reduction_proof,
                jagged=opening.reduction_proof,
            ),
            opening.transcript,
        )
