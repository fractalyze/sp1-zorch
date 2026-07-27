# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof's verifier as one composite zorch Stage.

``ShardVerifier`` mirrors ``ShardProver`` phase for phase — one
``VerifierStage`` role per prover role, glue included — consuming the named
sections of ``ShardProof`` and reducing to the same trivial claim. The seams
are the phases' reduced claims, so a phase cannot read something an earlier
phase never derived; a section present on one side and not the other is a
missing attribute rather than a silent Fiat-Shamir desync.

Static configuration (chip set, shapes, caps) lives on the role instances and
the statement on ``ShardClaim``, mirroring the prover's split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import frx.numpy as fnp
from frx import Array
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
    CommitmentRoots,
    GkrOutputClaim,
    JaggedOpeningClaim,
    ShardClaim,
    ShardProof,
    TraceEvaluationClaim,
    ZerocheckClaim,
    absorb_preamble,
    JaggedPcsProof,
)
from sp1_zorch.shard_prover.types import ChipShape
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


# Written phase outputs are array leaves; unwritten
# Optional fields are None (an empty subtree), so it crosses a @jit
# boundary as one argument.
class TraceCommitAbsorber:
    """Dual of the prover's committer: replays the preamble absorb stream via
    ``absorb_preamble`` — the same one function the prover calls — with the
    proof's commitment, and names the commitment roots. No local check: the
    commitment is validated downstream, by the stacked-open dual's Merkle
    openings against these roots.
    """

    def absorb(
        self, claim: ShardClaim, commitment: Array, transcript: Transcript
    ) -> tuple[Transcript, CommitmentRoots]:
        """Absorb the preamble and derive the commitment roots.

        The dual of the prover's committer: no claim is reduced here, so it is
        not a stage — it binds the commitment into the stream and names the
        roots a later opening checks against.

        The prep root is unconditional: SP1's verifier always carries the vk's
        preprocessed commitment, even though the prover keeps ``prep_region``
        optional. The stacked-open dual checking openings against these roots
        is where a no-prep proof would reconcile.
        """
        transcript = absorb_preamble(
            transcript,
            vk=claim.vk,
            public_values=claim.public_values,
            commitment=commitment,
            chip_metadata=claim.chip_metadata,
        )
        return transcript, CommitmentRoots(claim.vk.preprocessed_commit, commitment)


class LogupGkrVerifier(
    VerifierStage[ShardClaim, GkrOutputClaim, LogupGkrProof]
):
    """Stage-2 dual of ``LogupGkrStage``: verifies the LogUp-GKR proof via
    ``verify_logup_gkr`` and writes the derived evaluation point plus the
    proof's leaf-checked chip openings as its reduced claim — the same seams the
    prover role reduces to for the zerocheck stage."""

    def __init__(
        self,
        gkr_chips: Sequence[GkrChip],
        *,
        chip_names: Sequence[str],
        chip_heights: Mapping[str, int],
        num_betas: int,
        num_row_variables: int,
        pow_bits: int = 0,
        verify_public_values: bool = True,
    ) -> None:
        self._gkr_chips = gkr_chips
        self._chip_names = chip_names
        self._chip_heights = chip_heights
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
            self._chip_heights,
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

    The proof's opened values are checked against the statement shapes
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
        chip_shapes: Mapping[str, ChipShape],
        max_log_row_count: int,
    ) -> None:
        self._chips = chips
        self._chip_names = chip_names
        self._chip_shapes = chip_shapes
        self._chip_heights = {n: s.main.height for n, s in chip_shapes.items()}
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
            shape = self._chip_shapes[n]
            if int(opened[n].main.shape[0]) != shape.main.width:
                raise ValueError(
                    f"chip {n!r}: need one main claim per statement column "
                    f"({shape.main.width}), got {int(opened[n].main.shape[0])}"
                )
            prep_open = opened[n].preprocessed
            if shape.prep is not None:
                if prep_open is None or int(prep_open.shape[0]) != shape.prep.width:
                    got = "none" if prep_open is None else int(prep_open.shape[0])
                    raise ValueError(
                        f"chip {n!r}: need one preprocessed claim per "
                        f"statement column ({shape.prep.width}), got {got}"
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
            self._chip_heights,
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

    The column manifest is built entirely from the statement shapes; the
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
        chip_shapes: Mapping[str, ChipShape],
        log_stacking_height: int,
        max_log_row_count: int,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits
        self._chip_names = chip_names
        self._chip_shapes = chip_shapes
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
        shapes = self._chip_shapes

        # [prep, main] manifests from the statement, mirroring the prover's
        # region walk.
        prep_names = [n for n in self._chip_names if shapes[n].prep is not None]
        regions: list[tuple[list[str], list[int], list[int], str]] = []
        if prep_names:
            regions.append(
                (
                    prep_names,
                    [shapes[n].prep.height for n in prep_names],
                    [shapes[n].prep.width for n in prep_names],
                    "preprocessed",
                )
            )
        regions.append(
            (
                list(self._chip_names),
                [shapes[n].main.height for n in self._chip_names],
                [shapes[n].main.width for n in self._chip_names],
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
        statement_roots = claim.roots.as_statement(bool(prep_names))
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
        return VerifyResult(
            TrivialClaim(), transcript, ok_eval & ok_bind & ok_open
        )


class ShardVerifier(VerifierStage[ShardClaim, TrivialClaim, ShardProof]):
    """The SP1 shard verifier: one dual per prover phase, in the prover's
    order, so the two Fiat-Shamir streams stay in lockstep.

    Mirrors ``ShardProver`` phase for phase, and reduces to the same trivial
    claim — the jagged opening is terminal on both sides. Each dual's reduced
    claim is the next one's source claim, so a phase cannot silently read a
    seam an earlier phase never wrote.

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
        chip_shapes: Mapping[str, ChipShape],
        num_betas: int,
        num_row_variables: int,
        max_log_row_count: int,
        log_stacking_height: int,
        open_num_queries: int,
        open_pow_bits: int = 0,
        pow_bits: int = 0,
        verify_public_values: bool = True,
    ) -> None:
        self.absorber = TraceCommitAbsorber()
        self.gkr = LogupGkrVerifier(
            gkr_chips,
            chip_names=chip_names,
            chip_heights={n: s.main.height for n, s in chip_shapes.items()},
            num_betas=num_betas,
            num_row_variables=num_row_variables,
            pow_bits=pow_bits,
            verify_public_values=verify_public_values,
        )
        self.zerocheck = ZerocheckVerifier(
            chips,
            chip_names=chip_names,
            chip_shapes=chip_shapes,
            max_log_row_count=max_log_row_count,
        )
        self.opening = JaggedPcsVerifier(
            smcs,
            log_blowup=log_blowup,
            num_queries=open_num_queries,
            pow_bits=open_pow_bits,
            chip_names=chip_names,
            chip_shapes=chip_shapes,
            log_stacking_height=log_stacking_height,
            max_log_row_count=max_log_row_count,
        )

    def verify(
        self,
        claim: ShardClaim,
        reduction_proof: ShardProof,
        transcript: GrindingTranscript,
    ) -> VerifyResult[TrivialClaim]:
        transcript, roots = self.absorber.absorb(
            claim, reduction_proof.commitment, transcript
        )
        gkr = self.gkr.verify(claim, reduction_proof.gkr, transcript)
        zerocheck = self.zerocheck.verify(
            ZerocheckClaim(claim.public_values, gkr.reduced_claim),
            reduction_proof.zerocheck,
            gkr.transcript,
        )
        opening = self.opening.verify(
            JaggedOpeningClaim(zerocheck.reduced_claim, roots),
            reduction_proof.jagged,
            zerocheck.transcript,
        )
        return VerifyResult(
            TrivialClaim(),
            opening.transcript,
            gkr.ok & zerocheck.ok & opening.ok,
        )
