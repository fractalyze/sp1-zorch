# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The shard proof's verifier as one zorch ``VerifyChain`` of stage duals.

``verify_shard_chain`` mirrors ``prove_shard_chain`` round for round — one
verifier Round per prover stage, glue included, consuming the prover chain's
message list as the proof object. ``VerifyChain``'s one-message-per-round
check makes the mirror fail loud: a stage or glue step present on one side
and not the other is a structural reject, not a silent Fiat-Shamir desync
(zorch ``docs/stage-composition.md``, "Pipelines as nested chains").

Static configuration (vk, chip metadata, chip set) lives on the Round
instances and per-shard values flow on the carry, mirroring the prover's
split; ``ShardVerifierCarry`` threads what a later dual reads from an
earlier one — the witness-free dual of ``ShardCarry``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from rw_constraints import Chip
from zorch.pcs.jagged.region import structure_counts
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.prover import assemble_columns, sample_z_col
from zorch.pcs.jagged.verifier import (
    stacked_basefold_verify,
    verify_jagged_eval_msg,
)
from sp1_zorch.logup_gkr.circuit import GkrChip
from sp1_zorch.logup_gkr.prover import ChipEvaluation, LogupGkrProof
from sp1_zorch.logup_gkr.verifier import verify_logup_gkr
from sp1_zorch.shard_prover.prove_shard import (
    PreambleRound,
    ShardJaggedEvalProof,
)
from sp1_zorch.shard_prover.types import ChipShape, MachineVerifyingKey
from sp1_zorch.zerocheck.prover import ZerocheckProof
from sp1_zorch.zerocheck.verifier import verify_shard_zerocheck
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.round import Round, VerifyChain
from zorch.transcript import GrindingTranscript, Transcript
from zorch.utils.bits import log2_ceil_usize


# Pytree like ShardCarry: written stage outputs are array leaves; unwritten
# Optional fields are None (an empty subtree), so the carry crosses a @jit
# boundary as one argument.
@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "public_values",
        "commitment_roots",
        "gkr_eval_point",
        "gkr_chip_openings",
        "zc_sumcheck_point",
        "zc_opened_values",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class ShardVerifierCarry:
    """What flows between stage duals: each dual writes the fields a later
    one consumes — the same seams as ``ShardCarry``, minus the witness."""

    # Statement input, read by the trace-commit dual (preamble) and the
    # zerocheck dual (constraint evaluation) — on the carry, not the Rounds,
    # exactly as the prover reads ``ShardCarry.public_values``.
    public_values: Array
    # Written by TraceCommitVerifierRound; read by the stacked-open dual
    # (skip-level, two seams later). [prep, main] order, matching SP1's
    # round_evaluation_claims.
    commitment_roots: tuple[Array, Array] | None = None
    # Written by LogupGkrVerifierRound; read by the zerocheck dual (zeta =
    # row tail; claims derivation). The point is the dual's own derivation,
    # the openings are the proof's leaf-checked values.
    gkr_eval_point: Array | None = None
    gkr_chip_openings: Mapping[str, ChipEvaluation] | None = None
    # Written by ShardZerocheckVerifierRound; read by the jagged-eval dual as
    # its z_row (the dual's own sampled challenges) and its per-column claims
    # (the proof's opened values, shape- and oracle-checked by the zerocheck
    # dual).
    zc_sumcheck_point: Array | None = None
    zc_opened_values: Mapping[str, ChipEvaluation] | None = None


class TraceCommitVerifierRound(Round):
    """Stage-1 dual of ``TraceCommitRound``: replays the preamble absorb
    stream via ``PreambleRound`` — the same one Round the prover drives —
    with the proof's commitment message, and writes the commitment roots
    onto the carry. No local check: the commitment is validated downstream,
    by the stacked-open dual's Merkle openings against these roots."""

    def __init__(self, *, vk: MachineVerifyingKey, chip_metadata: Array) -> None:
        self._vk = vk
        self._chip_metadata = chip_metadata

    def __call__(
        self, carry: ShardVerifierCarry, msg: Array, transcript: Transcript
    ) -> tuple[ShardVerifierCarry, Transcript, Array]:
        _, transcript, _ = PreambleRound(
            vk=self._vk,
            public_values=carry.public_values,
            commitment=msg,
            chip_metadata=self._chip_metadata,
        )(None, transcript)
        # The prep root is unconditional: SP1's verifier always carries the
        # vk's preprocessed commitment, even though the prover keeps
        # ``prep_region`` optional. The stacked-open dual checking openings
        # against these roots is where a no-prep proof would reconcile.
        carry = replace(
            carry, commitment_roots=(self._vk.preprocessed_commit, msg)
        )
        return carry, transcript, jnp.bool_(True)


class LogupGkrVerifierRound(Round):
    """Stage-2 dual of ``LogupGkrRound``: verifies the LogUp-GKR proof via
    ``verify_logup_gkr`` and writes the derived evaluation point plus the
    proof's leaf-checked chip openings onto the carry — the same seams the
    prover Round writes on ``ShardCarry`` for the zerocheck stage."""

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

    def __call__(
        self,
        carry: ShardVerifierCarry,
        msg: LogupGkrProof,
        transcript: GrindingTranscript,
    ) -> tuple[ShardVerifierCarry, GrindingTranscript, Array]:
        transcript, eval_point, ok = verify_logup_gkr(
            self._gkr_chips,
            self._chip_names,
            self._chip_heights,
            msg,
            transcript,
            carry.public_values if self._verify_public_values else None,
            num_betas=self._num_betas,
            num_row_variables=self._num_row_variables,
            pow_bits=self._pow_bits,
        )
        carry = replace(
            carry, gkr_eval_point=eval_point, gkr_chip_openings=msg.chip_openings
        )
        return carry, transcript, ok


class ShardZerocheckVerifierRound(Round):
    """Stage-3 dual of ``ShardZerocheckRound``: verifies the zerocheck proof
    via ``verify_shard_zerocheck``, consuming the GKR point and openings off
    the carry, and writes the dual's own sumcheck point plus the proof's
    oracle-checked opened values onto the carry — the same seams the prover
    Round writes on ``ShardCarry`` for the jagged-eval stage.

    The proof's opened values are checked against the statement shapes
    before anything consumes them (SP1's ``verify_opening_shape`` inside
    ``verify_zerocheck``, ``crates/hypercube/src/verifier/shard.rs``) — the
    verifier absorbs the proof's opened values, so a shape lie never
    desyncs Fiat-Shamir and only a statement check rejects it. Downstream
    duals reading the carry's opened values may trust their shapes."""

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

    def __call__(
        self,
        carry: ShardVerifierCarry,
        msg: ZerocheckProof,
        transcript: Transcript,
    ) -> tuple[ShardVerifierCarry, Transcript, Array]:
        if carry.gkr_eval_point is None or carry.gkr_chip_openings is None:
            raise ValueError(
                "the zerocheck dual needs the LogUp-GKR stage's outputs on "
                "the carry; sequence a LogupGkrVerifierRound before this Round"
            )
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
            carry.public_values,
            carry.gkr_eval_point,
            carry.gkr_chip_openings,
            msg,
            transcript,
            max_log_row_count=self._max_log_row_count,
        )
        carry = replace(
            carry, zc_sumcheck_point=point, zc_opened_values=msg.opened_values
        )
        return carry, transcript, ok


class ShardJaggedEvalVerifierRound(Round):
    """Stage-4 dual of ``ShardJaggedEvalRound``: rebuilds the column manifest
    and per-column claims from the statement plus the carry's oracle-checked
    opened values, samples ``z_col`` itself, verifies the outer/inner
    sumchecks via ``verify_jagged_eval_msg``, and closes the chain with
    ``stacked_basefold_verify`` against the carry's skip-level commitment
    roots.

    The column manifest is built entirely from the statement shapes; the
    carry's opened values only supply the claims, their shapes already
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

    def __call__(
        self,
        carry: ShardVerifierCarry,
        msg: ShardJaggedEvalProof,
        transcript: GrindingTranscript,
    ) -> tuple[ShardVerifierCarry, GrindingTranscript, Array]:
        if (
            carry.zc_sumcheck_point is None
            or carry.zc_opened_values is None
            or carry.commitment_roots is None
        ):
            raise ValueError(
                "the jagged-eval dual needs the zerocheck point, opened "
                "values, and commitment roots on the carry; sequence the "
                "trace-commit and zerocheck duals before this Round"
            )
        opened = carry.zc_opened_values
        ef = carry.zc_sumcheck_point.dtype
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
                jnp.concatenate([getattr(opened[n], claim_field) for n in names])
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
            carry.zc_sumcheck_point[::-1],
            z_col,
            msg.eval,
            transcript,
            dtype=ef,
        )

        bf = carry.commitment_roots[1].dtype
        code = BitReversedReedSolomon(
            message_len=S, blowup=1 << self._log_blowup, dtype=bf
        )

        # The soundness anchor: each round's shape-bound proof commitment,
        # rebound with the statement-derived structure counts, must be the
        # preamble-observed commitment off the carry (SP1's table-sizes
        # check) — only then do the open's Merkle checks against the proof
        # commitments bind the openings to the statement.
        statement_roots = (
            list(carry.commitment_roots)
            if prep_names
            else [carry.commitment_roots[1]]
        )
        if len(msg.open.component_commitments) != len(statement_roots):
            raise ValueError(
                f"need one committed round per statement region "
                f"({len(statement_roots)}), got "
                f"{len(msg.open.component_commitments)}"
            )
        ok_bind = jnp.bool_(True)
        for component, root, rc, cc in zip(
            msg.open.component_commitments, statement_roots, rc_rounds, cc_rounds
        ):
            rebound = self._smcs.bind_structure(
                component, jnp.array(rc, dtype=bf), jnp.array(cc, dtype=bf)
            )
            ok_bind = ok_bind & jnp.array_equal(rebound, root)

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
        return carry, transcript, ok_eval & ok_bind & ok_open


def verify_shard_chain(
    *,
    smcs: SingleMatrixCommitmentScheme,
    log_blowup: int,
    vk: MachineVerifyingKey,
    chip_metadata: Array,
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
) -> VerifyChain:
    """The ``VerifyChain`` dual of ``prove_shard_chain``: one verifier Round
    per prover stage, in the prover's order, so the proof's message list
    aligns slot for slot or fails the chain's one-message-per-round check.

    ``chip_names`` and ``chip_shapes`` cover every shard chip (the openings
    absorb order, the opening-shape check, the leaf and oracle checks' geq
    thresholds, the jagged column manifest) — the verifier-side statement
    counterpart of the regions the prover Rounds read off the carry.
    ``log_stacking_height`` and the ``open_*`` parameters mirror the
    prover's stage-4 configuration.

    ``verify_public_values`` runs the LogUp-GKR output-layer bus-balance leg
    (the public-values digest vs the circuit cumulative sum); a structural
    test over a synthetic shard with no real public-values bus sets it False."""
    chip_heights = {n: s.main.height for n, s in chip_shapes.items()}
    return VerifyChain(
        [
            TraceCommitVerifierRound(vk=vk, chip_metadata=chip_metadata),
            LogupGkrVerifierRound(
                gkr_chips,
                chip_names=chip_names,
                chip_heights=chip_heights,
                num_betas=num_betas,
                num_row_variables=num_row_variables,
                pow_bits=pow_bits,
                verify_public_values=verify_public_values,
            ),
            ShardZerocheckVerifierRound(
                chips,
                chip_names=chip_names,
                chip_shapes=chip_shapes,
                max_log_row_count=max_log_row_count,
            ),
            ShardJaggedEvalVerifierRound(
                smcs,
                log_blowup=log_blowup,
                num_queries=open_num_queries,
                pow_bits=open_pow_bits,
                chip_names=chip_names,
                chip_shapes=chip_shapes,
                log_stacking_height=log_stacking_height,
                max_log_row_count=max_log_row_count,
            ),
        ]
    )
