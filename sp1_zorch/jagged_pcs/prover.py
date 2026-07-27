# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The jagged PCS as an sp1-zorch role: commit the trace, open it at a point.

Lives here rather than in zorch because the body reads SP1 shard structure —
`JaggedRegion` chip names and the preprocessed/main split — which zorch's
implementation-agnostic rule keeps out of its own scheme code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import frx
import frx.numpy as fnp
from frx import Array
from zk_dtypes import koalabearx4_mont

from zorch.coding.reed_solomon import BitReversedReedSolomon
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
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import GrindingTranscript
from zorch.utils.bits import log2_ceil_usize

from sp1_zorch.shard_prover.types import (
    JaggedCommitData,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    ShardWitness,
    SmcsCommitments,
)


@dataclass(frozen=True)
class JaggedPcsProof:
    """Discharges a `JaggedOpeningClaim`, leaving nothing to prove.

    Two legs: the outer/inner sumcheck reducing the committed trace to a
    single value ``D(z_final)``, then the stacked BaseFold open showing that
    value really is the commitment's, at that point.
    """

    eval: JaggedEvalMsg
    open: StackedOpenProof
    smcs_commitments: SmcsCommitments


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
                smcs_commitments=witness.commit_data.commitments,
            ),
            transcript,
        )
