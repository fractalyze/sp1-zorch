# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bincode serializer for the SP1 shard-proof wire format.

Produces byte buffers compatible with Rust's ``bincode::deserialize`` under
bincode's default (legacy) config: little-endian, fixed 8-byte ``u64`` length
prefixes, no varint.

KoalaBear's serde impl emits **canonical** u32, never the Montgomery raw form
the device arrays carry — ``_field_bytes`` converts via
``lax.convert_element_type(..., uint32)``. Extension-field elements flatten to
their base-field limbs before conversion.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from zk_dtypes import efinfo

from sp1_zorch.shard_prover.types import ChipOpenedValues, MachineVerifyingKey

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sp1_zorch.jagged.open import Opening, StackedOpenProof
    from sp1_zorch.jagged.prover import JaggedEvalMsg
    from sp1_zorch.logup_gkr.prover import LogupGkrProof
    from sp1_zorch.shard_prover.prove_shard import ShardCarry, ShardJaggedEvalProof
    from sp1_zorch.zerocheck.stage import ZerocheckProof


def _u64(v: int) -> bytes:
    return struct.pack("<Q", int(v))


def _usize(v: int) -> bytes:
    return _u64(v)


def _vec_prefix(length: int) -> bytes:
    return _u64(length)


def _field_bytes(arr: Array) -> bytes:
    """Canonical LE bytes for any BF/EF field array (any shape)."""
    a = jnp.atleast_1d(arr)
    if a.dtype.itemsize > 4:
        a = lax.bitcast_convert_type(a, efinfo(a.dtype).base_field_dtype)
    return np.asarray(lax.convert_element_type(jnp.ravel(a), np.uint32)).tobytes()


def _eval_poly_at(coeffs_row: Array, alpha: Array) -> Array:
    """Evaluate a univariate polynomial (coefficient form) at alpha via Horner."""
    result = jnp.zeros((), dtype=coeffs_row.dtype)
    for i in range(int(coeffs_row.shape[0]) - 1, -1, -1):
        result = result * alpha + coeffs_row[i]
    return result


def _encode_tensor(arr: Array, dimensions: list[int]) -> bytes:
    """Encode ``Tensor<T>``: ``{storage: Vec<T>, dimensions: Vec<usize>}``."""
    flat = jnp.ravel(arr)
    n = int(flat.shape[0])
    return (
        _vec_prefix(n)
        + _field_bytes(flat)
        + _vec_prefix(len(dimensions))
        + b"".join(_usize(d) for d in dimensions)
    )


def _encode_point(arr: Array) -> bytes:
    """Encode ``Point<T> = {values: Buffer<T>}`` = ``Vec<T>``."""
    flat = jnp.atleast_1d(arr)
    return _vec_prefix(int(flat.shape[0])) + _field_bytes(flat)


def _encode_partial_sumcheck_proof(
    round_polys: Array, claimed_sum: Array, point: Array
) -> bytes:
    """Encode ``PartialSumcheckProof<EF>``: ``{univariate_polys: Vec<Vec<EF>>,
    claimed_sum: EF, point_and_eval: (Point<EF>, EF)}``. The wire's eval is
    the last round polynomial at ``point[0]`` — the final fold's value, with
    ``point[0]`` the last challenge in SP1's insert-at-front point order."""
    n_rounds = int(round_polys.shape[0])
    n_coeffs = int(round_polys.shape[1])

    parts = [_vec_prefix(n_rounds)]
    for r in range(n_rounds):
        parts.append(_vec_prefix(n_coeffs))
        parts.append(_field_bytes(round_polys[r]))

    parts.append(_field_bytes(claimed_sum))
    parts.append(_encode_point(point))
    parts.append(_field_bytes(_eval_poly_at(round_polys[-1], point[0])))
    return b"".join(parts)


def _encode_logup_gkr_proof(proof, max_log_row_count: int) -> bytes:
    """Encode ``LogupGkrProof<F, EF>`` (rust field order: circuit_output,
    round_proofs, logup_evaluations, witness).

    ``proof`` is ``sp1_zorch.logup_gkr.prover.LogupGkrProof``; the wire's
    per-layer ``point_and_eval`` reads each round proof's ``point``, retained
    by zorch at prove time.
    """
    parts = []

    n_num = int(jnp.atleast_1d(proof.circuit_output.numerator).shape[0])
    parts.append(_encode_tensor(proof.circuit_output.numerator, [n_num, 1]))
    n_den = int(jnp.atleast_1d(proof.circuit_output.denominator).shape[0])
    parts.append(_encode_tensor(proof.circuit_output.denominator, [n_den, 1]))

    parts.append(_vec_prefix(len(proof.round_proofs)))
    for rp in proof.round_proofs:
        parts.append(_field_bytes(rp.numerator_0))
        parts.append(_field_bytes(rp.numerator_1))
        parts.append(_field_bytes(rp.denominator_0))
        parts.append(_field_bytes(rp.denominator_1))
        parts.append(_encode_partial_sumcheck_proof(rp.round_polys, rp.claim, rp.point))

    # SP1's eval_point has exactly max_log_row_count dims after all GKR
    # rounds. The prover-side point may overshoot — trim to the tail.
    gkr_point = proof.eval_point
    if gkr_point.shape[0] > max_log_row_count:
        gkr_point = gkr_point[-max_log_row_count:]
    parts.append(_encode_point(gkr_point))

    chip_map = proof.chip_openings
    parts.append(_vec_prefix(len(chip_map)))
    for name in sorted(chip_map):  # BTreeMap: ascending key order
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        ce = chip_map[name]
        n_main = int(jnp.atleast_1d(ce.main).shape[0])
        parts.append(_encode_tensor(ce.main, [n_main]))
        if ce.preprocessed is not None:
            parts.append(b"\x01")
            n_prep = int(jnp.atleast_1d(ce.preprocessed).shape[0])
            parts.append(_encode_tensor(ce.preprocessed, [n_prep]))
        else:
            parts.append(b"\x00")

    parts.append(_field_bytes(proof.witness))
    return b"".join(parts)


def _encode_digest(arr) -> bytes:
    """Encode ``GC::Digest = [F; 8]`` = 8 × canonical u32."""
    if hasattr(arr, "dtype"):
        return _field_bytes(arr)[:32]
    return struct.pack(f"<{len(arr)}I", *[int(x) for x in arr])[:32]


def _encode_chip_opened_values(cov: ChipOpenedValues, max_log_row_count: int) -> bytes:
    """Encode ``ChipOpenedValues<F, EF>``. A chip without a preprocessed
    trace serializes an EMPTY ``Vec`` — unlike the GKR chip openings, whose
    missing prep is an ``Option`` tag byte."""
    parts = []

    if cov.preprocessed_evals is not None:
        n = int(cov.preprocessed_evals.shape[0])
        parts.append(_vec_prefix(n))
        parts.append(_field_bytes(cov.preprocessed_evals))
    else:
        parts.append(_vec_prefix(0))

    n = int(cov.main_evals.shape[0])
    parts.append(_vec_prefix(n))
    parts.append(_field_bytes(cov.main_evals))

    n_bits = max_log_row_count + 1
    degree_bits = [(cov.degree >> bit) & 1 for bit in range(n_bits - 1, -1, -1)]
    parts.append(_vec_prefix(n_bits))
    parts.append(struct.pack(f"<{n_bits}I", *degree_bits))

    return b"".join(parts)


def _encode_shard_opened_values(
    chip_opened_values, chip_names, max_log_row_count: int
) -> bytes:
    """Encode ``ShardOpenedValues<F, EF> = {chips: BTreeMap<String,
    ChipOpenedValues>}`` — ascending chip-name order."""
    sorted_pairs = sorted(zip(chip_names, chip_opened_values, strict=True))
    parts = [_vec_prefix(len(sorted_pairs))]
    for name, cov in sorted_pairs:
        name_bytes = name.encode("utf-8")
        parts.append(_vec_prefix(len(name_bytes)))
        parts.append(name_bytes)
        parts.append(_encode_chip_opened_values(cov, max_log_row_count))
    return b"".join(parts)


def _pack_batch_openings(opening: Opening, root_digest: Array) -> bytes:
    """Encode ``MerkleTreeOpeningAndProof<GC>`` from one vmapped SMCS batch
    opening: the opened rows as ``Tensor<F>`` with dimensions ``[num_queries,
    width]``, then the proof — the **raw** Merkle root (the sibling paths
    reconstruct it, not the separator-bound commitment the transcript
    observes), depth, width, and the sibling digests as a ``Tensor`` with
    dimensions ``[num_queries, depth]`` (query-major)."""
    rows, paths = opening
    num_queries, width = (int(s) for s in rows.shape)
    depth = len(paths)

    parts = [_encode_tensor(rows, [num_queries, width])]

    parts.append(_encode_digest(root_digest))
    parts.append(_usize(depth))
    parts.append(_usize(width))
    parts.append(_vec_prefix(num_queries * depth))
    # ``paths`` is level-major, one (Q, digest) array per tree level; the
    # wire wants every query's full path contiguously.
    if depth > 0:
        parts.append(_field_bytes(jnp.stack(paths, axis=1)))
    parts.append(_vec_prefix(2))
    parts.append(_usize(num_queries))
    parts.append(_usize(depth))
    return b"".join(parts)


def _encode_basefold_proof(
    open_proof: StackedOpenProof, component_raw_roots: Sequence[Array]
) -> bytes:
    """Encode ``BasefoldProof<GC>``.

    ``component_raw_roots`` are the commit-time raw Merkle roots of the
    committed rounds, in the same order as ``component_openings`` — the
    proof retains only the fold layers' raw roots (``fri_raw_roots``), so
    the commit stage supplies the component ones.
    """
    parts = []

    # Vec<(EF, EF)>: one pair-count prefix, then the (s(0), s(1)) pairs
    # contiguous — exactly the (num_vars, 2) array's row-major bytes.
    msgs = open_proof.univariate_messages
    parts.append(_vec_prefix(int(msgs.shape[0])))
    parts.append(_field_bytes(msgs))

    # Vec<Digest>: each row is exactly [F; 8], so the stacked array's bytes
    # are the digests back to back.
    fri_commitments = open_proof.fri_commitments
    parts.append(_vec_prefix(int(fri_commitments.shape[0])))
    parts.append(_field_bytes(fri_commitments))

    comp = open_proof.component_openings
    parts.append(_vec_prefix(len(comp)))
    for opening, raw_root in zip(comp, component_raw_roots, strict=True):
        parts.append(_pack_batch_openings(opening, raw_root))

    query = open_proof.query_openings
    parts.append(_vec_prefix(len(query)))
    for i, opening in enumerate(query):
        parts.append(_pack_batch_openings(opening, open_proof.fri_raw_roots[i]))

    parts.append(_field_bytes(open_proof.final_poly))
    parts.append(_field_bytes(open_proof.pow_witness))
    return b"".join(parts)


def _encode_evaluation_proof(
    eval_msg: JaggedEvalMsg,
    open_proof: StackedOpenProof,
    component_raw_roots: Sequence[Array],
    row_counts_and_column_counts: Sequence[Sequence[tuple[int, int]]],
    max_log_row_count: int,
) -> bytes:
    """Encode ``JaggedPcsProof<GC, StackedBasefoldProof<GC>>``.

    ``row_counts_and_column_counts`` is the per-committed-round ``(row_count,
    column_count)`` chip layout (``JaggedRegion`` order, stacking dummies
    included). ``component_raw_roots`` doubles as the wire's
    ``original_commitments`` — both are the rounds' pre-binding Merkle roots.
    ``log_m`` is the outer sumcheck's round count, read off the proof.
    """
    parts = [_encode_basefold_proof(open_proof, component_raw_roots)]

    parts.append(_vec_prefix(len(open_proof.batch_evals)))
    for evals in open_proof.batch_evals:
        n_evals = int(jnp.atleast_1d(evals).shape[0])
        parts.append(_encode_tensor(evals, [n_evals]))

    outer_polys = eval_msg.outer_sumcheck_polys
    parts.append(
        _encode_partial_sumcheck_proof(
            outer_polys,
            eval_msg.outer_sumcheck_claim,
            eval_msg.outer_sumcheck_point,
        )
    )

    parts.append(
        _encode_partial_sumcheck_proof(
            eval_msg.inner_sumcheck_polys,
            eval_msg.inner_claimed_sum,
            eval_msg.inner_point,
        )
    )

    parts.append(_vec_prefix(len(row_counts_and_column_counts)))
    for round_counts in row_counts_and_column_counts:
        parts.append(_vec_prefix(len(round_counts)))
        for row_count, column_count in round_counts:
            parts.append(_usize(row_count))
            parts.append(_usize(column_count))

    parts.append(_vec_prefix(len(component_raw_roots)))
    for root in component_raw_roots:
        parts.append(_encode_digest(root))

    parts.append(_field_bytes(eval_msg.dense_eval))
    parts.append(_usize(max_log_row_count))
    parts.append(_usize(int(outer_polys.shape[0])))
    return b"".join(parts)


def encode_vk(vk: MachineVerifyingKey) -> bytes:
    """Encode ``MachineVerifyingKey<SP1GlobalContext>`` to bincode.

    Serde field order is pc_start, initial_global_cumulative_sum (SepticDigest
    = x then y), preprocessed_commit, enable_untrusted_programs — NOT the
    transcript ``observe_into`` order, which leads with the commit.
    """
    return b"".join(
        [
            _field_bytes(vk.pc_start),
            _field_bytes(vk.cum_sum_x),
            _field_bytes(vk.cum_sum_y),
            _field_bytes(vk.preprocessed_commit),
            struct.pack("<I", int(vk.enable_untrusted)),
        ]
    )


def chip_opened_values(carry: ShardCarry) -> list[ChipOpenedValues]:
    """Bridge the carry's zerocheck opened values to the wire's per-chip
    shape. The split off the final folded traces is the zerocheck stage's
    (``zerocheck.stage.split_opened_values`` — one view shared with the
    transcript absorbs and the jagged-eval claims); ``degree`` is the chip's
    live row count — the height whose bits the wire spells out.
    """
    if carry.zc_opened_values is None:
        raise ValueError(
            "the carry holds no zerocheck opened values; run the chain "
            "through ShardZerocheckRound before assembling the wire"
        )
    main = carry.main_region
    values = []
    for i, name in enumerate(main.chip_names):
        ev = carry.zc_opened_values[name]
        values.append(
            ChipOpenedValues(
                preprocessed_evals=ev.preprocessed,
                main_evals=ev.main,
                degree=int(main.chip_heights[i]),
            )
        )
    return values


def encode_shard_proof(
    carry: ShardCarry,
    commitment: Array,
    gkr_proof: LogupGkrProof,
    zerocheck_proof: ZerocheckProof,
    jagged_proof: ShardJaggedEvalProof,
    *,
    max_log_row_count: int,
) -> bytes:
    """Encode ``ShardProof<SP1GlobalContext, SP1PcsProofInner>`` to bincode.

    ``carry`` is the prove_shard chain's final carry (committed regions +
    stacked witnesses); the remaining arguments are the chain's messages in
    stage order. Serde field order: public values, main commitment,
    LogUp-GKR proof, zerocheck partial sumcheck, shard opened values,
    evaluation proof.
    """
    parts = [_encode_point(carry.public_values)]

    parts.append(_field_bytes(commitment))

    parts.append(_encode_logup_gkr_proof(gkr_proof, max_log_row_count))

    # The wire's zerocheck point is SP1's insert-at-front order — the
    # accumulated round challenges reversed, the same z_row order the
    # jagged-eval stage consumes.
    parts.append(
        _encode_partial_sumcheck_proof(
            zerocheck_proof.msgs.round_poly,
            zerocheck_proof.claimed_sum,
            zerocheck_proof.msgs.challenge[::-1],
        )
    )

    parts.append(
        _encode_shard_opened_values(
            chip_opened_values(carry),
            list(carry.main_region.chip_names),
            max_log_row_count,
        )
    )

    # Committed-round order is [prep, main] — the order TraceCommitRound
    # wrote the carry's StackedRounds in.
    regions = [
        region
        for region in (carry.prep_region, carry.main_region)
        if region is not None
    ]
    component_raw_roots = [
        stacked.digest_layers[-1][0] for stacked in carry.commit_rounds
    ]
    row_column_counts = [
        list(zip(region.row_counts, region.column_counts, strict=True))
        for region in regions
    ]
    parts.append(
        _encode_evaluation_proof(
            jagged_proof.eval,
            jagged_proof.open,
            component_raw_roots,
            row_column_counts,
            max_log_row_count=max_log_row_count,
        )
    )
    return b"".join(parts)
