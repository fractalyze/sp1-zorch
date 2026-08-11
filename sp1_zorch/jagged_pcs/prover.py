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
from zk_dtypes import efinfo
from zk_dtypes import koalabearx4_mont as EF
from zorch.coding.reed_solomon import BitReversedReedSolomon
from zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.pcs.jagged.commit import commit_region
from zorch.pcs.jagged.open import (
    StackedRound,
    stacked_basefold_open,
)
from zorch.pcs.jagged.prover import (
    JaggedEvalMsg,
    assemble_col_heights,
    assemble_columns,
    eval_column_arrays,
    eval_round_core,
    inner_sumcheck_core,
    outer_sumcheck,
    outer_sumcheck_claim,
    sample_z_col,
)
from zorch.poly.eq import expand_eq_to_hypercube
from zorch.poly.univariate import eval_coeffs
from zorch.stage import ProveResult, ProverStage, TrivialClaim
from zorch.transcript import Transcript, reinterpret_challenge
from zorch.utils.bits import log2_ceil_usize

from sp1_zorch.types import (
    JaggedCommitData,
    JaggedOpeningClaim,
    JaggedOpeningWitness,
    JaggedPcsProof,
    ShardWitness,
    SmcsCommitments,
)


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
    transcript: Transcript,
    *,
    num_columns: int,
    dtype: Any,
) -> tuple[Transcript, JaggedEvalMsg]:
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


@dataclass(frozen=True)
class JaggedCapClass:
    """Capacity class of the jagged open's eval scan (sp1-zorch#334) — the
    jagged analogue of zerocheck's ``TotalCapClass`` and LogUp-GKR's
    ``GkrCapClass``.

    ``scan_cap`` bounds the dense rows one outer-sumcheck dispatch touches: a
    padded dense larger than ``scan_cap`` runs the chunked scan (contiguous
    ``scan_cap``-row chunks, per-round partial sums combined across chunks),
    anything that fits runs the monolithic ``_jagged_eval_jit`` unchanged.
    Chunk starts and the fixed challenges ride as traced values, so the chunk
    zones compile once per (layout class, ``scan_cap``) and are reused across
    chunks and across shards of the class. Field arithmetic is exact, so the
    chunked scan is byte-identical to the monolithic one."""

    scan_cap: int

    def __post_init__(self) -> None:
        if self.scan_cap < 2 or self.scan_cap & (self.scan_cap - 1):
            raise ValueError(
                f"scan_cap must be a power of two >= 2, got {self.scan_cap}"
            )

    @classmethod
    def for_tier(cls, log_area_tier: int) -> "JaggedCapClass":
        """The tier/8 cap: on the keccak layout class (2^29 tier) the #334
        projection puts the eval arena at 3.35-4.35 GiB here (vs 13.0 GiB
        monolithic); the GPU BufferAssignment dump is the binding judge."""
        return cls(1 << max(log_area_tier - 3, 1))


def _indicator_chunk(
    offsets: Array, z_row: Array, z_col: Array, start: Array, size: int
) -> Array:
    """J̃ over the dense index window ``[start, start + size)``.

    zorch's ``partial_eval_core`` evaluates the outer indicator over the FIRST
    ``size`` indices only; the window form instead shifts the decoded prefix
    sums by ``-start`` (int32, negatives allowed), which maps window query
    ``j`` to global query ``start + j``: the searchsorted count compares
    ``t - start <= j  <=>  t <= start + j``, so the owning column, local row,
    height mask, and eq factors all equal the global scan's values on the
    window. ``start`` rides as a traced value, so one compile serves every
    chunk. No ``zorch.jagged_indicator`` fused-region marker here: the marker
    contract carries no window operand, so the chunk form inlines the
    byte-identical decomposition."""
    dtype = z_row.dtype
    n_d = offsets.shape[1]
    # Decode the MSB-first prefix-bit tensor: the canonical bit lives in int32
    # limb 0 (an EF tensor bitcasts to a trailing limb axis, base does not).
    limbs = frx.lax.bitcast_convert_type(offsets, fnp.int32)
    bit_vals = limbs[..., 0] if limbs.ndim > offsets.ndim else limbs
    powers = fnp.left_shift(fnp.int32(1), n_d - 1 - fnp.arange(n_d, dtype=fnp.int32))
    prefix = fnp.sum(bit_vals * powers, axis=1) - start

    col_eq = expand_eq_to_hypercube(z_col, fnp.ones([], dtype=dtype))
    one = fnp.ones([], dtype=dtype)
    n_r = z_row.shape[0]
    row_len = fnp.left_shift(fnp.int32(1), n_r)

    # c_idx = owning column = (#prefix entries <= j) - 1: a vectorized binary
    # search of a static step count (searchsorted side="right"; extra steps
    # are no-ops once lo == hi). The tail j past the shifted t_L lands at the
    # last index, where the height mask zeros it.
    i_idx = fnp.arange(size, dtype=fnp.int32)
    n = prefix.shape[0]
    lo = fnp.zeros(i_idx.shape, fnp.int32)
    hi = fnp.full(i_idx.shape, n, fnp.int32)
    for _ in range(log2_ceil_usize(n) + 2):
        mid = (lo + hi) // 2
        val = prefix[fnp.minimum(mid, n - 1)]
        go_right = (mid < n) & (val <= i_idx)
        lo = fnp.where(go_right, mid + 1, lo)
        hi = fnp.where(go_right, hi, mid)
    c_idx = lo - 1
    t_c = prefix[c_idx]
    h = prefix[c_idx + 1] - t_c  # column height (0 for padding columns)
    local = i_idx - t_c
    # min(h, row_len): the row eq covers 2^n_r rows, so a taller-than-capacity
    # column truncates — same window as the monolithic form.
    mask = local < fnp.minimum(h, row_len)

    # eq(z_row, local) per element, MSB-first — the per-row eq factor without
    # a 2^n_r gather table.
    row_vals = fnp.ones(i_idx.shape, dtype)
    for bit_pos in range(n_r):
        bit = ((local >> (n_r - 1 - bit_pos)) & 1).astype(dtype)
        z_k = z_row[bit_pos]
        row_vals = row_vals * (bit * z_k + (one - bit) * (one - z_k))
    val = col_eq[c_idx] * row_vals
    return fnp.where(mask, val, fnp.zeros([], dtype=dtype))


def _outer_chunk_states(
    offsets: Array,
    dense: Array,
    z_row: Array,
    z_col: Array,
    start: Array,
    folds: Array,
    *,
    chunk_rows: int,
) -> tuple[Array, Array]:
    """One chunk of the outer-sumcheck state pair at round ``len(folds) + 1``:
    the dense slice and its indicator window, re-folded LSB-first through the
    challenges already fixed. Chunks are contiguous power-of-two windows, so
    every even/odd fold pair falls inside one chunk and the chunk-local fold
    equals the global fold's slice."""
    a = frx.lax.dynamic_slice_in_dim(dense, start, chunk_rows, axis=0)
    b = _indicator_chunk(offsets, z_row, z_col, start, chunk_rows)
    for alpha in folds:
        a = a[0::2] + alpha * (a[1::2] - a[0::2])
        b = b[0::2] + alpha * (b[1::2] - b[0::2])
    return a, b


def _outer_chunk_partials(
    offsets: Array,
    dense: Array,
    z_row: Array,
    z_col: Array,
    start: Array,
    folds: Array,
    *,
    chunk_rows: int,
) -> tuple[Array, Array]:
    """One chunk's contribution to the round's ``(s0, s_inf)`` pair sums —
    exact field addition makes the cross-chunk total bit-identical to the
    monolithic scan's full-buffer sums."""
    a, b = _outer_chunk_states(
        offsets, dense, z_row, z_col, start, folds, chunk_rows=chunk_rows
    )
    p0a, p1a = a[0::2], a[1::2]
    p0b, p1b = b[0::2], b[1::2]
    return fnp.sum(p0a * p0b), fnp.sum((p1a - p0a) * (p1b - p0b))


@partial(frx.jit, static_argnames=("chunk_rows",))
def _jagged_outer_chunk_partials_jit(
    offsets: Array,
    dense: Array,
    z_row: Array,
    z_col: Array,
    start: Array,
    folds: Array,
    *,
    chunk_rows: int,
) -> tuple[Array, Array]:
    """Chunk partial-sums zone: ``start`` and ``folds`` are traced VALUES, so
    the compile keys on (layout class, ``chunk_rows``, fold depth) — one
    executable per recompute round, reused across chunks and shards."""
    return _outer_chunk_partials(
        offsets, dense, z_row, z_col, start, folds, chunk_rows=chunk_rows
    )


@partial(frx.jit, static_argnames=("chunk_rows",))
def _jagged_outer_chunk_fold_jit(
    offsets: Array,
    dense: Array,
    z_row: Array,
    z_col: Array,
    start: Array,
    folds: Array,
    *,
    chunk_rows: int,
) -> tuple[Array, Array]:
    """Chunk materialize zone — same key structure as the partials zone; the
    concatenation of its outputs across chunks is the monolithic fold state."""
    return _outer_chunk_states(
        offsets, dense, z_row, z_col, start, folds, chunk_rows=chunk_rows
    )


@frx.jit
def _jagged_outer_tail_jit(
    state_a: Array, state_b: Array, claim: Array, transcript: Transcript
) -> tuple[Array, Array, Array, Transcript]:
    """The residual monolithic rounds over the materialized state pair —
    zorch's ``outer_sumcheck`` continued from the running claim."""
    return outer_sumcheck(state_a, state_b, claim, transcript)


def _inner_core(
    merged: Array,
    z_col: Array,
    z_row: Array,
    z_final: Array,
    transcript: Transcript,
    *,
    num_columns: int,
    dtype: Any,
) -> tuple[Array, Array, Array, Transcript]:
    weights = expand_eq_to_hypercube(z_col, fnp.ones((), dtype))[:num_columns]
    return inner_sumcheck_core(
        merged,
        weights,
        z_row,
        z_final,
        transcript,
        dtype=dtype,
        num_bits=merged.shape[1] // 2,
    )


@partial(frx.jit, static_argnames=("num_columns", "dtype"))
def _jagged_inner_jit(
    merged: Array,
    z_col: Array,
    z_row: Array,
    z_final: Array,
    transcript: Transcript,
    *,
    num_columns: int,
    dtype: Any,
) -> tuple[Array, Array, Array, Transcript]:
    """The inner branching-program sumcheck as its own zone (the chunked path
    splits ``_jagged_eval_jit``'s single body); keys on the (L, n_d) class."""
    return _inner_core(
        merged, z_col, z_row, z_final, transcript, num_columns=num_columns, dtype=dtype
    )


def _chunked_eval(
    offsets: Array,
    merged: Array,
    all_claims: Array,
    dense: Array,
    zc_sumcheck_point: Array,
    transcript: Transcript,
    *,
    num_columns: int,
    scan_cap: int,
    dtype: Any,
    jit: bool,
) -> tuple[Transcript, JaggedEvalMsg]:
    """The eval half with the outer scan chunked to ``scan_cap`` rows per
    dispatch (sp1-zorch#334) — eager orchestration over the chunk zones, the
    LogupGkrProver pattern.

    Recompute-until-fit: materializing the fold state after round 1 pins
    ``target`` EF elements regardless of chunk width, so rounds ``1..k`` are
    instead re-folded per chunk from the original dense/indicator with only
    the partial pair sums crossing the zone boundary, and the state pair
    materializes once its total is ``scan_cap / 2`` elements (the chunk fold
    depth floors at one element for deep chunking). Exact field arithmetic
    makes every piece — cross-chunk partial sums, re-folded states, the
    transcript stream — bit-identical to the monolithic zone."""
    target = int(dense.shape[0])
    n_rounds = log2_ceil_usize(target)
    log_chunk = log2_ceil_usize(scan_cap)
    num_chunks = target // scan_cap
    k = min(n_rounds - log_chunk + 1, log_chunk)
    ef_limbs = efinfo(dtype).degree
    two = fnp.array(2, dtype)

    transcript, z_col = sample_z_col(transcript, num_columns, dtype)
    claim = outer_sumcheck_claim(all_claims, z_col)
    z_row = zc_sumcheck_point[::-1]

    partials = _jagged_outer_chunk_partials_jit if jit else _outer_chunk_partials
    fold = _jagged_outer_chunk_fold_jit if jit else _outer_chunk_states
    tail = _jagged_outer_tail_jit if jit else outer_sumcheck
    inner = _jagged_inner_jit if jit else _inner_core

    def starts() -> list[Array]:
        return [fnp.asarray(c * scan_cap, fnp.int32) for c in range(num_chunks)]

    cur = claim
    chunk_polys: list[Array] = []
    challenges: list[Array] = []
    for _ in range(k):
        fixed = fnp.stack(challenges) if challenges else fnp.zeros((0,), dtype)
        s0: Array | None = None
        s_inf: Array | None = None
        for start in starts():
            p0, p_inf = partials(
                offsets, dense, z_row, z_col, start, fixed, chunk_rows=scan_cap
            )
            s0 = p0 if s0 is None else s0 + p0
            s_inf = p_inf if s_inf is None else s_inf + p_inf
        coef = fnp.stack([s0, cur - two * s0 - s_inf, s_inf])
        # Fused absorb+squeeze — byte for byte the monolithic round's op.
        transcript, raw = transcript.observe_and_sample(coef, ef_limbs)
        alpha = reinterpret_challenge(raw, dtype)
        cur = eval_coeffs(coef, alpha)
        chunk_polys.append(coef)
        challenges.append(alpha)

    fixed = fnp.stack(challenges)
    a_parts: list[Array] = []
    b_parts: list[Array] = []
    for start in starts():
        a_c, b_c = fold(offsets, dense, z_row, z_col, start, fixed, chunk_rows=scan_cap)
        a_parts.append(a_c)
        b_parts.append(b_c)
    state_a, state_b = fnp.concatenate(a_parts), fnp.concatenate(b_parts)
    del a_parts, b_parts

    tail_polys, z_tail, dense_eval, transcript = tail(state_a, state_b, cur, transcript)
    del state_a, state_b
    outer_polys = fnp.concatenate([fnp.stack(chunk_polys), tail_polys])
    # The proof point is the full challenge list reversed (SP1's
    # insert-at-front); the tail already carries its own reversed suffix.
    z_final = fnp.concatenate([z_tail, fnp.stack(challenges[::-1])])

    inner_polys, inner_point, inner_claimed_sum, transcript = inner(
        merged, z_col, z_row, z_final, transcript, num_columns=num_columns, dtype=dtype
    )
    return transcript, JaggedEvalMsg(
        outer_sumcheck_claim=claim,
        outer_sumcheck_polys=outer_polys,
        outer_sumcheck_point=z_final,
        dense_eval=dense_eval,
        inner_sumcheck_polys=inner_polys,
        inner_point=inner_point,
        inner_claimed_sum=inner_claimed_sum,
    )


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

    Eager orchestration over shard-invariant jitted zones (the LogupGkrProver
    pattern, sp1-zorch#274): the prologue folds per-shard heights into traced
    array values and pads the combined dense to its power-of-two tier, so the
    eval zone's compile keys on the layout class alone; the stacked open runs
    zorch's zoned ``stacked_basefold_open`` (dominant fold zone K-independent).
    Byte-identical to the eager path.

    ``jagged_cap_class`` gates the chunked outer scan (sp1-zorch#334): unset —
    the default — the monolithic zones above run unchanged; set, a padded
    dense larger than ``scan_cap`` runs ``_chunked_eval`` instead, byte-
    identically."""

    def __init__(
        self,
        smcs: SingleMatrixCommitmentScheme,
        *,
        log_blowup: int,
        num_queries: int,
        pow_bits: int,
        jit: bool = True,
        jagged_cap_class: JaggedCapClass | None = None,
    ) -> None:
        self._smcs = smcs
        self._log_blowup = log_blowup
        self._num_queries = num_queries
        self._pow_bits = pow_bits
        self._jit = jit
        self._cap_class = jagged_cap_class

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
        transcript: Transcript,
    ) -> ProveResult[TrivialClaim, JaggedPcsProof]:
        main = witness.trace.main_region
        openings = claim.evaluation.opened_values
        zc_point = claim.evaluation.point
        # The jagged eval runs in the extension field — the upstream sumcheck
        # points are EF challenge lists (one extension sample per variable).
        ef = EF

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
        else:
            claims_rounds = [fnp.concatenate(chips) for chips in claims_chips]
            _, all_claims = assemble_columns(
                rc_rounds, cc_rounds, claims_rounds, dtype=ef
            )
            dense = fnp.concatenate(denses)
            dense = fnp.pad(dense, (0, target - dense.shape[0]))

        if self._cap_class is not None and self._cap_class.scan_cap < target:
            # The monolithic eval zone's temp arena scales with the padded
            # tier (13 GiB on the keccak class, sp1-zorch#334); the class-gated
            # chunk loop bounds it by scan_cap instead. Byte-identical.
            transcript, eval_msg = _chunked_eval(
                offsets,
                merged,
                all_claims,
                dense,
                zc_point,
                transcript,
                num_columns=len(col_heights),
                scan_cap=self._cap_class.scan_cap,
                dtype=ef,
                jit=self._jit,
            )
        elif self._jit:
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
        else:
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
        # Free the eval leg's buffers (padded dense is GiB-scale) before the
        # open allocates its [N, K] round codewords.
        del dense, offsets, merged, all_claims

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
