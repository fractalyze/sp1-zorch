# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bincode primitives: hand-computed goldens for the SP1 wire layout.

Anchors the layout bincode's default config defines (little-endian, 8-byte
length prefixes, no varint) and the field rule SP1's serde impl fixes:
KoalaBear serializes as canonical u32, never Montgomery raw; extension
fields flatten to their base-field limbs.
"""

import struct

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.shard_prover.serialize import (
    _encode_digest,
    _encode_point,
    _encode_tensor,
    _eval_poly_at,
    _field_bytes,
    _u64,
    _usize,
    _vec_prefix,
    encode_vk,
)
from sp1_zorch.shard_prover.types import MachineVerifyingKey


class BincodePrimitivesTest(absltest.TestCase):
    def test_u64_usize_vec_prefix_little_endian_8_bytes(self) -> None:
        self.assertEqual(_u64(5), b"\x05" + b"\x00" * 7)
        self.assertEqual(_u64(0x0102030405060708), bytes(range(8, 0, -1)))
        self.assertEqual(_usize(7), _u64(7))
        self.assertEqual(_vec_prefix(3), _u64(3))

    def test_field_bytes_emits_canonical_u32_not_montgomery(self) -> None:
        # Values 1,2,3 at the Python boundary are canonical; the wire must
        # carry canonical u32 LE regardless of the Montgomery storage form.
        arr = jnp.array([1, 2, 3], dtype=F)
        self.assertEqual(_field_bytes(arr), struct.pack("<3I", 1, 2, 3))

    def test_field_bytes_scalar_and_2d_flatten(self) -> None:
        self.assertEqual(_field_bytes(jnp.array(9, dtype=F)), struct.pack("<I", 9))
        arr = jnp.arange(6, dtype=F).reshape(2, 3)
        self.assertEqual(_field_bytes(arr), struct.pack("<6I", *range(6)))

    def test_field_bytes_extension_field_flattens_to_base_limbs(self) -> None:
        # One EF element = 4 base limbs on the wire, canonical u32 each.
        arr = jnp.array([1, 2, 3, 4], dtype=F).view(EF)
        self.assertEqual(_field_bytes(arr), struct.pack("<4I", 1, 2, 3, 4))

    def test_encode_tensor_storage_then_dimensions(self) -> None:
        # Tensor<T> = {storage: Vec<T>, dimensions: Vec<usize>}.
        arr = jnp.arange(6, dtype=F).reshape(2, 3)
        expected = (
            _u64(6)
            + struct.pack("<6I", *range(6))
            + _u64(2)
            + _u64(2)
            + _u64(3)
        )
        self.assertEqual(_encode_tensor(arr, [2, 3]), expected)

    def test_encode_point_is_len_prefixed_vec(self) -> None:
        arr = jnp.array([7, 8], dtype=F)
        self.assertEqual(_encode_point(arr), _u64(2) + struct.pack("<2I", 7, 8))
        # Scalar promotes to a 1-vector.
        self.assertEqual(
            _encode_point(jnp.array(7, dtype=F)), _u64(1) + struct.pack("<I", 7)
        )

    def test_eval_poly_at_horner(self) -> None:
        # p(x) = 2 + 3x + 5x^2 at x=4 -> 2 + 12 + 80 = 94.
        coeffs = jnp.array([2, 3, 5], dtype=F)
        alpha = jnp.array(4, dtype=F)
        self.assertEqual(int(_eval_poly_at(coeffs, alpha)), 94)


class EncodeDigestTest(absltest.TestCase):
    def test_field_array_digest_is_32_canonical_bytes(self) -> None:
        arr = jnp.arange(1, 9, dtype=F)  # [F; 8]
        self.assertEqual(_encode_digest(arr), struct.pack("<8I", *range(1, 9)))

    def test_plain_int_sequence_digest(self) -> None:
        self.assertEqual(
            _encode_digest(list(range(1, 9))), struct.pack("<8I", *range(1, 9))
        )


class EncodeVkTest(absltest.TestCase):
    def test_wire_order_differs_from_transcript_observe_order(self) -> None:
        # Serde field order is pc_start, cumulative sum, preprocessed_commit,
        # enable_untrusted — NOT the observe_into transcript order (which
        # leads with the commit). A swap would still verify locally but be
        # rejected by sp1_verify_shard's deserializer.
        vk = MachineVerifyingKey(
            preprocessed_commit=jnp.arange(1, 9, dtype=F),
            pc_start=jnp.array([9, 10, 11], dtype=F),
            cum_sum_x=jnp.arange(1, 8, dtype=F),
            cum_sum_y=jnp.arange(8, 15, dtype=F),
            enable_untrusted=0,
        )
        expected = (
            struct.pack("<3I", 9, 10, 11)
            + struct.pack("<7I", *range(1, 8))
            + struct.pack("<7I", *range(8, 15))
            + struct.pack("<8I", *range(1, 9))
            + struct.pack("<I", 0)
        )
        self.assertEqual(encode_vk(vk), expected)


if __name__ == "__main__":
    absltest.main()
