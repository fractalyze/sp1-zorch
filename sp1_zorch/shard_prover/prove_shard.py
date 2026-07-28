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

from frx import Array
from rw_constraints import Chip
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.stage import (
    ProveResult,
    ProverStage,
    TrivialClaim,
)
from zorch.transcript import Transcript

from sp1_zorch.jagged_pcs.prover import JaggedPcsProver
from sp1_zorch.logup_gkr.circuit import GkrCapClass, GkrChip
from sp1_zorch.logup_gkr.prover import LogupGkrProver
from sp1_zorch.types import (
    BoundRoots,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    MachineVerifyingKey,
    ShardClaim,
    ShardProof,
    ShardWitness,
    ZerocheckClaim,
)
from sp1_zorch.zerocheck.jagged import TotalCapClass
from sp1_zorch.zerocheck.prover import (
    ZerocheckProver,
)


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
        transcript: Transcript,
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
