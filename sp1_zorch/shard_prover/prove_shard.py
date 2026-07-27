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

from zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.logup_gkr.circuit import GkrCapClass, GkrChip
from sp1_zorch.logup_gkr.prover import (
    ChipEvaluation,
    LogupGkrProof,
    prove_logup_gkr,
)
from sp1_zorch.jagged_pcs.prover import JaggedPcsProof, JaggedPcsProver
from sp1_zorch.shard_prover.types import (
    BoundRoots,
    ChipMetadata,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    MachineVerifyingKey,
    ShardWitness,
    TraceEvaluationClaim,
)
from sp1_zorch.zerocheck.jagged import TotalCapClass, pack_flat_arrival
from sp1_zorch.zerocheck.prover import (
    ZerocheckProof,
    chip_traces,
    prove_shard_zerocheck,
)
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
)
from zorch.transcript import GrindingTranscript, Transcript


# Pytree: the two regions (themselves pytrees), public values, and written
# stage outputs are array leaves; unwritten Optional fields are None (an empty
# subtree). Lets the witness cross a @jit boundary as one donatable
# argument.
@dataclass(frozen=True)
class ShardClaim:
    """Some trace of this shape is a valid execution of the shard.

    Spelled out: there exists a trace holding `chip_metadata`'s chips at its
    row counts, whose preprocessed part is the one `vk` commits to, on which
    every chip's AIR constraints vanish and whose LogUp bus balances against
    `public_values`. Nothing here names that trace — it is existentially
    quantified, and the prover exhibits one by committing to it, which is why
    the commitment is proof data rather than a field of the statement.
    """

    vk: MachineVerifyingKey
    public_values: Array
    chip_metadata: ChipMetadata


@dataclass(frozen=True)
class GkrOutputClaim:
    """The trace's LogUp columns take `chip_openings` at `eval_point`.

    What LogUp-GKR reduces the bus-balance statement to: checking a whole
    logarithmic-derivative argument becomes checking a handful of column
    values at one point. Both roles derive it — the prover from its own layer
    chain, the verifier by replaying the proof — so neither has to be trusted
    for it.
    """

    eval_point: Array
    chip_openings: Mapping[str, ChipEvaluation]


@dataclass(frozen=True)
class ZerocheckClaim:
    """Every chip's AIR constraints vanish on the trace — conditionally on
    `gkr`, whose column openings the constraint sum folds in.

    Conditional because zerocheck never re-proves the LogUp leg: it inherits
    `gkr` as a hypothesis and discharges only the constraint half, so the two
    together are what pin the trace. `public_values` supplies the operands the
    PV-reading constraint circuits index.
    """

    public_values: Array
    gkr: GkrOutputClaim
    chip_metadata: ChipMetadata


@dataclass(frozen=True)
class ShardProof:
    """What a verifier needs to check a `ShardClaim` without the trace.

    The commitment fixes which trace is being talked about; the three
    reduction proofs then carry the verifier along the same chain the prover
    walked, one section per Stage, ending at the trivial claim.
    """

    commitment: Array  # structure-bound main root; see JaggedCommitData
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
        pow_witness: Array | None = None,
        gkr_cap_class: GkrCapClass | None = None,
    ) -> None:
        self._gkr_chips = tuple(gkr_chips)
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._pow_witness = pow_witness
        self._gkr_cap_class = gkr_cap_class

    def prove(
        self,
        claim: ShardClaim,
        witness: ShardWitness,
        transcript: Transcript,
    ) -> ProveResult[GkrOutputClaim, LogupGkrProof]:
        # The verifier dual reads the row counts off the claim while the
        # prover has them in the witness's regions; a claim that disagrees
        # would otherwise only surface later, as a transcript divergence.
        assert claim.chip_metadata.by_chip() == {
            n: int(h)
            for n, h in zip(
                witness.main_region.chip_names,
                witness.main_region.chip_heights,
                strict=True,
            )
        }, "claim's chip metadata does not match the witness's main region"
        transcript, proof = prove_logup_gkr(
            self._gkr_chips,
            witness,
            transcript,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
            pow_witness=self._pow_witness,
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
        pow_witness: Array | None = None,
        jit: bool = True,
        zerocheck_total_cap_class: TotalCapClass | None = None,
        gkr_cap_class: GkrCapClass | None = None,
    ) -> None:
        self.gkr = LogupGkrProver(
            gkr_chips,
            num_betas=num_betas,
            num_row_variables=num_row_variables,
            pow_bits=pow_bits,
            pow_witness=pow_witness,
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
