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

import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from zk_dtypes import efinfo

from sp1_zorch.shard_prover.types import ChipOpenedValues, MachineVerifyingKey


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
    round_polys: Array, claimed_sum: Array, point: Array, final_eval: Array
) -> bytes:
    """Encode ``PartialSumcheckProof<EF>``: ``{univariate_polys: Vec<Vec<EF>>,
    claimed_sum: EF, point_and_eval: (Point<EF>, EF)}``."""
    n_rounds = int(round_polys.shape[0])
    n_coeffs = int(round_polys.shape[1])

    parts = [_vec_prefix(n_rounds)]
    for r in range(n_rounds):
        parts.append(_vec_prefix(n_coeffs))
        parts.append(_field_bytes(round_polys[r]))

    parts.append(_field_bytes(claimed_sum))
    parts.append(_encode_point(point))
    parts.append(_field_bytes(final_eval))
    return b"".join(parts)


def _encode_logup_gkr_proof(proof, layer_points, max_log_row_count: int) -> bytes:
    """Encode ``LogupGkrProof<F, EF>`` (rust field order: circuit_output,
    round_proofs, logup_evaluations, witness).

    ``proof`` is ``sp1_zorch.logup_gkr.prover.LogupGkrProof``. The wire's
    per-layer ``point_and_eval`` is not retained on ``JaggedLayerProof`` (the
    verifier re-derives the challenges from the transcript), so the layer
    sumcheck points arrive separately as ``layer_points``, one per round
    proof, supplied by the assembly's transcript replay.
    """
    parts = []

    n_num = int(jnp.atleast_1d(proof.circuit_output.numerator).shape[0])
    parts.append(_encode_tensor(proof.circuit_output.numerator, [n_num, 1]))
    n_den = int(jnp.atleast_1d(proof.circuit_output.denominator).shape[0])
    parts.append(_encode_tensor(proof.circuit_output.denominator, [n_den, 1]))

    parts.append(_vec_prefix(len(proof.round_proofs)))
    for rp, point in zip(proof.round_proofs, layer_points, strict=True):
        parts.append(_field_bytes(rp.numerator_0))
        parts.append(_field_bytes(rp.numerator_1))
        parts.append(_field_bytes(rp.denominator_0))
        parts.append(_field_bytes(rp.denominator_1))
        final_eval = _eval_poly_at(rp.round_polys[-1], point[0])
        parts.append(
            _encode_partial_sumcheck_proof(rp.round_polys, rp.claim, point, final_eval)
        )

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
