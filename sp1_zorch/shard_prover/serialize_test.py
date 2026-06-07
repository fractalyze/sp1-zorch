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

from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.logup_gkr.jagged_prover import JaggedLayerProof

from sp1_zorch.logup_gkr.prover import ChipEvaluation, LogupGkrProof
from sp1_zorch.shard_prover.serialize import (
    _encode_chip_opened_values,
    _encode_digest,
    _encode_logup_gkr_proof,
    _encode_partial_sumcheck_proof,
    _encode_point,
    _encode_shard_opened_values,
    _encode_tensor,
    _eval_poly_at,
    _field_bytes,
    _u64,
    _usize,
    _vec_prefix,
    encode_vk,
)
from sp1_zorch.shard_prover.types import ChipOpenedValues, MachineVerifyingKey


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


class EncodePartialSumcheckProofTest(absltest.TestCase):
    def test_round_polys_then_claim_then_point_and_eval(self) -> None:
        # PartialSumcheckProof<EF> = {univariate_polys: Vec<Vec<EF>>,
        # claimed_sum: EF, point_and_eval: (Point<EF>, EF)} — each round poly
        # is its own length-prefixed Vec.
        round_polys = jnp.arange(6, dtype=F).reshape(2, 3)
        claimed_sum = jnp.array(7, dtype=F)
        point = jnp.array([8, 9], dtype=F)
        final_eval = jnp.array(10, dtype=F)
        expected = (
            _u64(2)
            + _u64(3)
            + struct.pack("<3I", 0, 1, 2)
            + _u64(3)
            + struct.pack("<3I", 3, 4, 5)
            + struct.pack("<I", 7)
            + _u64(2)
            + struct.pack("<2I", 8, 9)
            + struct.pack("<I", 10)
        )
        self.assertEqual(
            _encode_partial_sumcheck_proof(round_polys, claimed_sum, point, final_eval),
            expected,
        )


def _gkr_proof() -> LogupGkrProof:
    """One-layer synthetic proof; values chosen so every wire chunk is
    recognizable in the golden."""
    rp = JaggedLayerProof(
        lam=jnp.array(5, dtype=F),
        claim=jnp.array(6, dtype=F),
        round_polys=jnp.array([[1, 2, 3]], dtype=F),
        numerator_0=jnp.array(7, dtype=F),
        numerator_1=jnp.array(8, dtype=F),
        denominator_0=jnp.array(9, dtype=F),
        denominator_1=jnp.array(10, dtype=F),
    )
    return LogupGkrProof(
        witness=jnp.array(15, dtype=F),
        circuit_output=LogUpGkrOutput(
            numerator=jnp.array([1, 2], dtype=F),
            denominator=jnp.array([3, 4], dtype=F),
        ),
        round_proofs=[rp],
        eval_point=jnp.array([11, 12], dtype=F),
        chip_openings={
            # Two chips deliberately out of order: the wire is a BTreeMap,
            # so "add" must serialize before "cpu" regardless of dict order.
            "cpu": ChipEvaluation(main=jnp.array([13, 14], dtype=F), preprocessed=None),
            "add": ChipEvaluation(
                main=jnp.array([20], dtype=F),
                preprocessed=jnp.array([21], dtype=F),
            ),
        },
    )


class EncodeLogupGkrProofTest(absltest.TestCase):
    def test_full_structure_golden(self) -> None:
        proof = _gkr_proof()
        layer_points = [jnp.array([16], dtype=F)]
        # final_eval = p(16) for p = 1 + 2x + 3x^2 -> 801.
        partial_sumcheck = (
            _u64(1)
            + _u64(3)
            + struct.pack("<3I", 1, 2, 3)
            + struct.pack("<I", 6)
            + _u64(1)
            + struct.pack("<I", 16)
            + struct.pack("<I", 801)
        )
        expected = (
            # circuit_output: numerator/denominator as [n, 1] tensors
            _u64(2) + struct.pack("<2I", 1, 2) + _u64(2) + _u64(2) + _u64(1)
            + _u64(2) + struct.pack("<2I", 3, 4) + _u64(2) + _u64(2) + _u64(1)
            # round_proofs
            + _u64(1)
            + struct.pack("<4I", 7, 8, 9, 10)
            + partial_sumcheck
            # logup_evaluations: point + BTreeMap("add" first)
            + _u64(2) + struct.pack("<2I", 11, 12)
            + _u64(2)
            + _u64(3) + b"add"
            + _u64(1) + struct.pack("<I", 20) + _u64(1) + _u64(1)
            + b"\x01" + _u64(1) + struct.pack("<I", 21) + _u64(1) + _u64(1)
            + _u64(3) + b"cpu"
            + _u64(2) + struct.pack("<2I", 13, 14) + _u64(1) + _u64(2)
            + b"\x00"
            # witness
            + struct.pack("<I", 15)
        )
        self.assertEqual(
            _encode_logup_gkr_proof(proof, layer_points, max_log_row_count=2),
            expected,
        )

    def test_eval_point_trims_to_max_log_row_count_tail(self) -> None:
        proof = _gkr_proof()
        layer_points = [jnp.array([16], dtype=F)]
        full = _encode_logup_gkr_proof(proof, layer_points, max_log_row_count=2)
        trimmed = _encode_logup_gkr_proof(proof, layer_points, max_log_row_count=1)
        # Point [11, 12] trims to its tail [12]: one 4-byte element fewer
        # (the u64 prefix stays 8 bytes, only its value drops).
        self.assertEqual(len(full) - len(trimmed), 4)
        self.assertIn(_u64(1) + struct.pack("<I", 12), trimmed)


class EncodeOpenedValuesTest(absltest.TestCase):
    def test_chip_opened_values_with_prep_and_degree_bits(self) -> None:
        cov = ChipOpenedValues(
            preprocessed_evals=jnp.array([1, 2], dtype=F),
            main_evals=jnp.array([3, 4, 5], dtype=F),
            degree=4,
        )
        # degree bits are height decomposed MSB-first over
        # max_log_row_count + 1 positions: 4 = 0b0100 over 4 bits.
        expected = (
            _u64(2)
            + struct.pack("<2I", 1, 2)
            + _u64(3)
            + struct.pack("<3I", 3, 4, 5)
            + _u64(4)
            + struct.pack("<4I", 0, 1, 0, 0)
        )
        self.assertEqual(_encode_chip_opened_values(cov, max_log_row_count=3), expected)

    def test_missing_prep_is_an_empty_vec_not_an_option(self) -> None:
        # Unlike the GKR chip openings (Option<Tensor>, 0x00/0x01 tag), a
        # chip with no preprocessed trace serializes an EMPTY Vec here.
        cov = ChipOpenedValues(
            preprocessed_evals=None,
            main_evals=jnp.array([3], dtype=F),
            degree=1,
        )
        expected = (
            _u64(0) + _u64(1) + struct.pack("<I", 3) + _u64(2) + struct.pack("<2I", 0, 1)
        )
        self.assertEqual(_encode_chip_opened_values(cov, max_log_row_count=1), expected)

    def test_shard_opened_values_sorts_chips_btreemap_order(self) -> None:
        cov_a = ChipOpenedValues(
            preprocessed_evals=None, main_evals=jnp.array([1], dtype=F), degree=1
        )
        cov_b = ChipOpenedValues(
            preprocessed_evals=None, main_evals=jnp.array([2], dtype=F), degree=1
        )
        out = _encode_shard_opened_values([cov_b, cov_a], ["cpu", "add"], 1)
        expected = (
            _u64(2)
            + _u64(3)
            + b"add"
            + _encode_chip_opened_values(cov_a, 1)
            + _u64(3)
            + b"cpu"
            + _encode_chip_opened_values(cov_b, 1)
        )
        self.assertEqual(out, expected)


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
