# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The jagged-PCS opening's verifier role — dual of `prover.JaggedPcsProver`.

Lives beside the prover it mirrors because the two are the halves of one claim
reduction, not two unrelated stages (`zorch.stage`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import frx.numpy as fnp
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.prover import assemble_columns, sample_z_col
from zorch.pcs.jagged.region import structure_counts
from zorch.pcs.jagged.verifier import (
    stacked_basefold_verify,
    verify_jagged_eval_msg,
)
from zorch.stage import TrivialClaim, VerifierStage, VerifyResult
from zorch.transcript import Transcript
from zorch.utils.bits import log2_ceil_usize

from sp1_zorch.jagged_pcs.types import JaggedPcsProof
from sp1_zorch.shard_prover.types import ChipWidths, JaggedOpeningClaim


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
        transcript: Transcript,
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
