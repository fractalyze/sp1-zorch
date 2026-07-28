# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof's verifier as one composite zorch Stage.

``ShardVerifier`` mirrors ``ShardProver`` role for role — one
``VerifierStage`` per ``ProverStage``, glue included — consuming the named
sections of ``ShardProof`` and reducing to the same trivial claim. The seams
are the Stages' reduced claims, so one cannot read something an earlier Stage
never derived; a section present on one side and not the other is a missing
attribute rather than a silent Fiat-Shamir desync.

Static configuration (chip set, column counts, caps) lives on the role
instances and
the statement on ``ShardClaim``, mirroring the prover's split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_constraints import Chip
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.stage import (
    TrivialClaim,
    VerifierStage,
    VerifyResult,
)
from zorch.transcript import Transcript

from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.shard_prover.prove_shard import bind_commitment
from sp1_zorch.shard_prover.types import (
    ChipWidths,
    JaggedOpeningClaim,
    ShardClaim,
    ShardProof,
    ZerocheckClaim,
)
from sp1_zorch.zerocheck.verifier import ZerocheckVerifier
from sp1_zorch.logup_gkr.verifier import LogupGkrVerifier
from sp1_zorch.jagged_pcs.verifier import JaggedPcsVerifier


class ShardVerifier(VerifierStage[ShardClaim, TrivialClaim, ShardProof]):
    """The SP1 shard verifier: one dual per prover role, in the prover's
    order, so the two Fiat-Shamir streams stay in lockstep.

    Mirrors ``ShardProver`` role for role, and reduces to the same trivial
    claim — the jagged opening is terminal on both sides. Each dual's reduced
    claim is the next one's source claim, so one cannot silently read a seam
    an earlier Stage never wrote.

    ``verify_public_values`` runs the LogUp-GKR output-layer bus-balance leg
    (the public-values digest vs the circuit cumulative sum); a structural test
    over a synthetic shard with no real public-values bus sets it False.
    """

    def __init__(
        self,
        *,
        smcs: SingleMatrixCommitmentScheme,
        log_blowup: int,
        gkr_chips: Sequence[GkrChip],
        chips: Mapping[str, Chip],
        chip_names: Sequence[str],
        chip_widths: Mapping[str, ChipWidths],
        num_betas: int,
        num_row_variables: int,
        max_log_row_count: int,
        log_stacking_height: int,
        open_num_queries: int,
        open_pow_bits: int = 0,
        pow_bits: int = 0,
        verify_public_values: bool = True,
    ) -> None:
        self.gkr = LogupGkrVerifier(
            gkr_chips,
            chip_names=chip_names,
            num_betas=num_betas,
            num_row_variables=num_row_variables,
            pow_bits=pow_bits,
            verify_public_values=verify_public_values,
        )
        self.zerocheck = ZerocheckVerifier(
            chips,
            chip_names=chip_names,
            chip_widths=chip_widths,
            max_log_row_count=max_log_row_count,
        )
        self.opening = JaggedPcsVerifier(
            smcs,
            log_blowup=log_blowup,
            num_queries=open_num_queries,
            pow_bits=open_pow_bits,
            chip_names=chip_names,
            chip_widths=chip_widths,
            log_stacking_height=log_stacking_height,
            max_log_row_count=max_log_row_count,
        )

    def verify(
        self,
        claim: ShardClaim,
        reduction_proof: ShardProof,
        transcript: Transcript,
    ) -> VerifyResult[TrivialClaim]:
        transcript, roots = bind_commitment(
            transcript, claim, reduction_proof.commitment
        )
        gkr = self.gkr.verify(claim, reduction_proof.gkr, transcript)
        zerocheck = self.zerocheck.verify(
            ZerocheckClaim(claim.public_values, gkr.reduced_claim, claim.chip_metadata),
            reduction_proof.zerocheck,
            gkr.transcript,
        )
        opening = self.opening.verify(
            JaggedOpeningClaim(zerocheck.reduced_claim, roots, claim.chip_metadata),
            reduction_proof.jagged,
            zerocheck.transcript,
        )
        return VerifyResult(
            TrivialClaim(),
            opening.transcript,
            gkr.ok & zerocheck.ok & opening.ok,
        )
