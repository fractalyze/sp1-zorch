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

import frx.numpy as fnp
from rw_constraints import Chip
from zorch.pcs.jagged.region import structure_counts
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.prover import assemble_columns, sample_z_col
from zorch.pcs.jagged.verifier import (
    stacked_basefold_verify,
    verify_jagged_eval_msg,
)
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import LogupGkrProof
from sp1_zorch.logup_gkr.verifier import verify_logup_gkr
from sp1_zorch.shard_prover.prove_shard import (
    GkrOutputClaim,
    JaggedOpeningClaim,
    ShardClaim,
    ShardProof,
    TraceEvaluationClaim,
    ZerocheckClaim,
    bind_commitment,
    JaggedPcsProof,
)
from sp1_zorch.shard_prover.types import ChipWidths
from sp1_zorch.zerocheck.prover import ZerocheckProof
from sp1_zorch.zerocheck.verifier import verify_shard_zerocheck
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.stage import (
    TrivialClaim,
    VerifierStage,
    VerifyResult,
)
from zorch.transcript import GrindingTranscript, Transcript
from zorch.utils.bits import log2_ceil_usize


class LogupGkrVerifier(VerifierStage[ShardClaim, GkrOutputClaim, LogupGkrProof]):
    """Stage-2 dual of ``LogupGkrStage``: verifies the LogUp-GKR proof via
    ``verify_logup_gkr`` and writes the derived evaluation point plus the
    proof's leaf-checked chip openings as its reduced claim — the same seams the
    prover role reduces to for the zerocheck stage."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        chip_names: Sequence[str],
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        verify_public_values: bool = True,
    ) -> None:
        self._gkr_chips = gkr_chips
        self._chip_names = chip_names
        self._num_betas = num_betas
        self._num_row_variables = num_row_variables
        self._pow_bits = pow_bits
        self._verify_public_values = verify_public_values

    def verify(
        self,
        claim: ShardClaim,
        reduction_proof: LogupGkrProof,
        transcript: GrindingTranscript,
    ) -> VerifyResult[GkrOutputClaim]:
        msg = reduction_proof
        transcript, eval_point, ok = verify_logup_gkr(
            self._gkr_chips,
            self._chip_names,
            claim.chip_metadata.by_chip(),
            msg,
            transcript,
            claim.public_values if self._verify_public_values else None,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
        )
        return VerifyResult(
            GkrOutputClaim(eval_point, msg.chip_openings), transcript, ok
        )


class ZerocheckVerifier(
    VerifierStage[ZerocheckClaim, TraceEvaluationClaim, ZerocheckProof]
):
    """Stage-3 dual of ``ZerocheckStage``: verifies the zerocheck proof
    via ``verify_shard_zerocheck``, consuming the GKR point and openings off
    its source claim, and reduces to the dual's own sumcheck point plus the
    proof's oracle-checked opened values — the same seams the prover
    role reduces to for the jagged-eval stage.

    The proof's opened values are checked against the statement's column
    counts
    before anything consumes them (SP1's ``verify_opening_shape`` inside
    ``verify_zerocheck``, ``crates/hypercube/src/verifier/shard.rs``) — the
    verifier absorbs the proof's opened values, so a shape lie never
    desyncs Fiat-Shamir and only a statement check rejects it. Downstream
    later duals reading those opened values may trust their shapes."""

    def __init__(
        self,
        chips: Mapping[str, Chip],
        *,
        chip_names: Sequence[str],
        chip_widths: Mapping[str, ChipWidths],
        max_log_row_count: int,
    ) -> None:
        self._chips = chips
        self._chip_names = chip_names
        self._chip_widths = chip_widths
        self._max_log_row_count = max_log_row_count

    def verify(
        self,
        claim: ZerocheckClaim,
        reduction_proof: ZerocheckProof,
        transcript: Transcript,
    ) -> VerifyResult[TraceEvaluationClaim]:
        msg = reduction_proof
        opened = msg.opened_values
        for n in self._chip_names:
            widths = self._chip_widths[n]
            if int(opened[n].main.shape[0]) != widths.main:
                raise ValueError(
                    f"chip {n!r}: need one main claim per statement column "
                    f"({widths.main}), got {int(opened[n].main.shape[0])}"
                )
            prep_open = opened[n].preprocessed
            if widths.prep is not None:
                if prep_open is None or int(prep_open.shape[0]) != widths.prep:
                    got = "none" if prep_open is None else int(prep_open.shape[0])
                    raise ValueError(
                        f"chip {n!r}: need one preprocessed claim per "
                        f"statement column ({widths.prep}), got {got}"
                    )
            elif prep_open is not None:
                raise ValueError(
                    f"chip {n!r}: the statement has no preprocessed trace, "
                    f"but the proof opens {int(prep_open.shape[0])} "
                    f"preprocessed columns"
                )
        transcript, point, ok = verify_shard_zerocheck(
            self._chips,
            self._chip_names,
            claim.chip_metadata.by_chip(),
            claim.public_values,
            claim.gkr.eval_point,
            claim.gkr.chip_openings,
            msg,
            transcript,
            max_log_row_count=self._max_log_row_count,
        )
        return VerifyResult(
            TraceEvaluationClaim(point, msg.opened_values), transcript, ok
        )


class JaggedPcsVerifier(
    VerifierStage[JaggedOpeningClaim, TrivialClaim, JaggedPcsProof]
):
    """Stage-4 dual of ``JaggedPcsStage``: rebuilds the column manifest
    and per-column claims from the statement plus the oracle-checked
    opened values, samples ``z_col`` itself, verifies the outer/inner
    sumchecks via ``verify_jagged_eval_msg``, and closes the chain with
    ``stacked_basefold_verify`` against the skip-level commitment
    roots.

    The column manifest is built entirely from the statement; the
    opened values only supply the claims, their shapes already
    checked against the same statement by the zerocheck dual. A statement
    with no preprocessed chip states that no preprocessed round exists, so
    a proof carrying one is a structural reject."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        num_queries: int,
        pow_bits: int,
        chip_names: Sequence[str],
        chip_widths: Mapping[str, ChipWidths],
        log_stacking_height: int,
        max_log_row_count: int,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits
        self._chip_names = chip_names
        self._chip_widths = chip_widths
        self._log_stacking_height = log_stacking_height
        self._max_log_row_count = max_log_row_count

    def verify(
        self,
        claim: JaggedOpeningClaim,
        reduction_proof: JaggedPcsProof,
        transcript: GrindingTranscript,
    ) -> VerifyResult[TrivialClaim]:
        msg = reduction_proof
        opened = claim.evaluation.opened_values
        zc_point = claim.evaluation.point
        ef = zc_point.dtype
        chip_widths = self._chip_widths
        row_counts = claim.chip_metadata.by_chip()

        # [prep, main] manifests from the statement, mirroring the prover's
        # region walk. Column counts are AIR-static role config; the row counts
        # are per-shard and come off the claim.
        prep_names = [n for n in self._chip_names if chip_widths[n].prep is not None]
        # `prep_names` was filtered on exactly this, so each width is present.
        preps = {n: w for n in prep_names if (w := chip_widths[n].prep) is not None}
        regions: list[tuple[list[str], list[int], list[int], str]] = []
        if prep_names:
            regions.append(
                (
                    prep_names,
                    [row_counts[n] for n in prep_names],
                    [preps[n] for n in prep_names],
                    "preprocessed",
                )
            )
        regions.append(
            (
                list(self._chip_names),
                [row_counts[n] for n in self._chip_names],
                [chip_widths[n].main for n in self._chip_names],
                "main",
            )
        )

        S = 1 << self._log_stacking_height
        rc_rounds, cc_rounds, claims_rounds = [], [], []
        round_widths: list[int] = []
        raw_total = 0
        for names, heights, widths, claim_field in regions:
            rc, cc, area, aligned = structure_counts(
                heights,
                widths,
                log_stacking_height=self._log_stacking_height,
                max_log_row_count=self._max_log_row_count,
            )
            rc_rounds.append(rc)
            cc_rounds.append(cc)
            claims_rounds.append(
                fnp.concatenate([getattr(opened[n], claim_field) for n in names])
            )
            round_widths.append(aligned >> self._log_stacking_height)
            raw_total += area

        col_heights, all_claims = assemble_columns(
            rc_rounds, cc_rounds, claims_rounds, dtype=ef
        )

        # The prover pads the concatenated raw packed dense to a power of
        # two; the round count is a statement fact, so a mis-sized outer
        # transcript is a structural reject.
        num_outer = log2_ceil_usize(raw_total)
        if msg.eval.outer_sumcheck_polys.shape[0] != num_outer:
            raise ValueError(
                f"need one outer round per packed-dense variable "
                f"({num_outer}), got {msg.eval.outer_sumcheck_polys.shape[0]}"
            )

        # z_col is the dual's own sampling, through the same shared rule.
        transcript, z_col = sample_z_col(transcript, len(col_heights), ef)

        transcript, z_final, ok_eval = verify_jagged_eval_msg(
            col_heights,
            all_claims,
            zc_point[::-1],
            z_col,
            msg.eval,
            transcript,
            dtype=ef,
        )

        bf = claim.roots.main.dtype
        code = BitReversedReedSolomon(
            message_len=S, blowup=1 << self._log_blowup, dtype=bf
        )

        # The soundness anchor: each round's shape-bound proof commitment,
        # rebound with the statement-derived structure counts, must be the
        # preamble-observed commitment (SP1's table-sizes
        # check) — only then do the open's Merkle checks against the proof
        # commitments bind the openings to the statement.
        statement_roots = claim.roots.in_round_order(bool(prep_names))
        if len(msg.open.component_commitments) != len(statement_roots):
            raise ValueError(
                f"need one committed round per statement region "
                f"({len(statement_roots)}), got "
                f"{len(msg.open.component_commitments)}"
            )
        ok_bind = fnp.bool_(True)
        for component, root, rc, cc in zip(
            msg.open.component_commitments, statement_roots, rc_rounds, cc_rounds
        ):
            rebound = self._smcs.bind_structure(
                component, fnp.array(rc, dtype=bf), fnp.array(cc, dtype=bf)
            )
            ok_bind = ok_bind & fnp.array_equal(rebound, root)

        transcript, ok_open = stacked_basefold_verify(
            self._smcs,
            code,
            round_widths,
            z_final,
            msg.eval.dense_eval,
            self._log_stacking_height,
            msg.open,
            transcript,
            num_queries=self._num_queries,
            pow_bits=self._pow_bits,
        )
        return VerifyResult(TrivialClaim(), transcript, ok_eval & ok_bind & ok_open)


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
        transcript: GrindingTranscript,
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
