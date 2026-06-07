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

from sp1_zorch.shard_prover.types import MachineVerifyingKey


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


def _encode_digest(arr) -> bytes:
    """Encode ``GC::Digest = [F; 8]`` = 8 × canonical u32."""
    if hasattr(arr, "dtype"):
        return _field_bytes(arr)[:32]
    return struct.pack(f"<{len(arr)}I", *[int(x) for x in arr])[:32]


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
