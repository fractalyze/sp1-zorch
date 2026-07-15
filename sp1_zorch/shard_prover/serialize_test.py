# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Bincode primitives: hand-computed goldens for the SP1 wire layout.

Anchors the layout bincode's default config defines (little-endian, 8-byte
length prefixes, no varint) and the field rule SP1's serde impl fixes:
KoalaBear serializes as canonical u32, never Montgomery raw; extension
fields flatten to their base-field limbs.
"""

import struct

import frx.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from zorch.logup_gkr.circuit import LogUpGkrOutput
from zorch.logup_gkr.jagged_prover import JaggedLayerProof
from zorch.sumcheck.prover import RoundMsg

from zorch.pcs.jagged.region import JaggedRegion
from zorch.pcs.jagged.prover import JaggedEvalMsg

from zorch.pcs.jagged.open import StackedOpenProof
from sp1_zorch.logup_gkr.prover import ChipEvaluation, LogupGkrProof
from sp1_zorch.shard_prover.prove_shard import ShardBridge, ShardJaggedEvalProof
from sp1_zorch.shard_prover.serialize import (
    _encode_basefold_proof,
    _encode_chip_opened_values,
    _encode_digest,
    _encode_evaluation_proof,
    _encode_logup_gkr_proof,
    _encode_partial_sumcheck_proof,
    _encode_point,
    _encode_shard_opened_values,
    _encode_tensor,
    _eval_poly_at,
    _field_bytes,
    _pack_batch_openings,
    _u64,
    _usize,
    _vec_prefix,
    chip_opened_values,
    encode_shard_proof,
    encode_vk,
)
from sp1_zorch.shard_prover.types import ChipOpenedValues, MachineVerifyingKey
from sp1_zorch.zerocheck.prover import ZerocheckProof


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
        expected = _u64(6) + struct.pack("<6I", *range(6)) + _u64(2) + _u64(2) + _u64(3)
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
    def test_round_polys_then_claim_then_point_and_derived_eval(self) -> None:
        # PartialSumcheckProof<EF> = {univariate_polys: Vec<Vec<EF>>,
        # claimed_sum: EF, point_and_eval: (Point<EF>, EF)} — each round poly
        # is its own length-prefixed Vec, and the eval is the last round
        # poly at point[0]: 3 + 4*8 + 5*64 = 355.
        round_polys = jnp.arange(6, dtype=F).reshape(2, 3)
        claimed_sum = jnp.array(7, dtype=F)
        point = jnp.array([8, 9], dtype=F)
        expected = (
            _u64(2)
            + _u64(3)
            + struct.pack("<3I", 0, 1, 2)
            + _u64(3)
            + struct.pack("<3I", 3, 4, 5)
            + struct.pack("<I", 7)
            + _u64(2)
            + struct.pack("<2I", 8, 9)
            + struct.pack("<I", 355)
        )
        self.assertEqual(
            _encode_partial_sumcheck_proof(round_polys, claimed_sum, point),
            expected,
        )


def _gkr_proof() -> LogupGkrProof:
    """One-layer synthetic proof; values chosen so every wire chunk is
    recognizable in the golden."""
    rp = JaggedLayerProof(
        lam=jnp.array(5, dtype=F),
        claim=jnp.array(6, dtype=F),
        round_polys=jnp.array([[1, 2, 3]], dtype=F),
        point=jnp.array([16], dtype=F),
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
            _u64(2)
            + struct.pack("<2I", 1, 2)
            + _u64(2)
            + _u64(2)
            + _u64(1)
            + _u64(2)
            + struct.pack("<2I", 3, 4)
            + _u64(2)
            + _u64(2)
            + _u64(1)
            # round_proofs
            + _u64(1)
            + struct.pack("<4I", 7, 8, 9, 10)
            + partial_sumcheck
            # logup_evaluations: point + BTreeMap("add" first)
            + _u64(2)
            + struct.pack("<2I", 11, 12)
            + _u64(2)
            + _u64(3)
            + b"add"
            + _u64(1)
            + struct.pack("<I", 20)
            + _u64(1)
            + _u64(1)
            + b"\x01"
            + _u64(1)
            + struct.pack("<I", 21)
            + _u64(1)
            + _u64(1)
            + _u64(3)
            + b"cpu"
            + _u64(2)
            + struct.pack("<2I", 13, 14)
            + _u64(1)
            + _u64(2)
            + b"\x00"
            # witness
            + struct.pack("<I", 15)
        )
        self.assertEqual(
            _encode_logup_gkr_proof(proof, max_log_row_count=2),
            expected,
        )

    def test_eval_point_trims_to_max_log_row_count_tail(self) -> None:
        proof = _gkr_proof()
        full = _encode_logup_gkr_proof(proof, max_log_row_count=2)
        trimmed = _encode_logup_gkr_proof(proof, max_log_row_count=1)
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
            _u64(0)
            + _u64(1)
            + struct.pack("<I", 3)
            + _u64(2)
            + struct.pack("<2I", 0, 1)
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


def _opening(base: int):
    """A 2-query, width-3, depth-2 vmapped SMCS batch opening whose values are
    distinct offsets of ``base`` so each wire chunk is recognizable."""
    rows = jnp.arange(base, base + 6, dtype=F).reshape(2, 3)
    paths = [
        jnp.arange(base + 10, base + 26, dtype=F).reshape(2, 8),
        jnp.arange(base + 30, base + 46, dtype=F).reshape(2, 8),
    ]
    return rows, paths


def _open_proof(num_rounds: int = 1) -> StackedOpenProof:
    """One-fold-round synthetic stacked open over ``num_rounds`` committed
    rounds."""
    return StackedOpenProof(
        component_commitments=[
            jnp.arange(70 + 10 * r, 78 + 10 * r, dtype=F) for r in range(num_rounds)
        ],
        fri_raw_roots=jnp.arange(80, 88, dtype=F).reshape(1, 8),
        fri_commitments=jnp.arange(90, 98, dtype=F).reshape(1, 8),
        univariate_messages=jnp.arange(1, 9, dtype=F).view(EF).reshape(1, 2),
        final_poly=jnp.array([61, 62, 63, 64], dtype=F).view(EF),
        pow_witness=jnp.array(77, dtype=F),
        batch_evals=[
            jnp.array([31 + 10 * r, 32 + 10 * r], dtype=F) for r in range(num_rounds)
        ],
        component_openings=[_opening(200 + 200 * r) for r in range(num_rounds)],
        query_openings=[_opening(300)],
    )


def _eval_msg() -> JaggedEvalMsg:
    return JaggedEvalMsg(
        outer_sumcheck_claim=jnp.array(5, dtype=F),
        outer_sumcheck_polys=jnp.array([[1, 2, 3]], dtype=F),
        outer_sumcheck_point=jnp.array([4], dtype=F),
        dense_eval=jnp.array(20, dtype=F),
        inner_sumcheck_polys=jnp.array([[6, 7, 8]], dtype=F),
        inner_point=jnp.array([9], dtype=F),
        inner_claimed_sum=jnp.array(10, dtype=F),
    )


class PackBatchOpeningsTest(absltest.TestCase):
    def test_values_tensor_then_root_depth_width_and_querymajor_digests(self) -> None:
        rows, paths = _opening(0)
        root = jnp.arange(100, 108, dtype=F)
        expected = (
            # values: Tensor<F> {storage: 6 elems, dimensions: [2, 3]}
            _u64(6)
            + struct.pack("<6I", *range(6))
            + _u64(2)
            + _u64(2)
            + _u64(3)
            # proof: raw root, depth, width
            + struct.pack("<8I", *range(100, 108))
            + _u64(2)
            + _u64(3)
            # digests: Tensor {4 digests, dimensions [2, 2]} — query-major,
            # so query 0's full path (level 0 then level 1) precedes query 1's.
            + _u64(4)
            + struct.pack("<8I", *range(10, 18))
            + struct.pack("<8I", *range(30, 38))
            + struct.pack("<8I", *range(18, 26))
            + struct.pack("<8I", *range(38, 46))
            + _u64(2)
            + _u64(2)
            + _u64(2)
        )
        self.assertEqual(_pack_batch_openings((rows, paths), root), expected)


class EncodeBasefoldProofTest(absltest.TestCase):
    def test_field_order_and_root_routing(self) -> None:
        # The component opening packs with the commit-time raw root passed in;
        # the query opening packs with the proof's own fri_raw_roots entry —
        # NOT the bound fri_commitments digest the commitments Vec carries.
        proof = _open_proof()
        component_raw_roots = [jnp.arange(100, 108, dtype=F)]
        expected = (
            _u64(1)
            + struct.pack("<8I", *range(1, 9))  # (s(0), s(1)) message pair
            + _u64(1)
            + struct.pack("<8I", *range(90, 98))  # bound fold-layer root
            + _u64(1)
            + _pack_batch_openings(proof.component_openings[0], component_raw_roots[0])
            + _u64(1)
            + _pack_batch_openings(proof.query_openings[0], proof.fri_raw_roots[0])
            + struct.pack("<4I", 61, 62, 63, 64)
            + struct.pack("<I", 77)
        )
        self.assertEqual(_encode_basefold_proof(proof, component_raw_roots), expected)


class EncodeEvaluationProofTest(absltest.TestCase):
    def test_field_order_final_evals_and_derived_log_m(self) -> None:
        open_proof = _open_proof()
        component_raw_roots = [jnp.arange(100, 108, dtype=F)]
        # original_commitments are the SMCS commitments, distinct from the raw
        # roots the batch openings reconstruct.
        component_commitments = [jnp.arange(200, 208, dtype=F)]
        eval_msg = _eval_msg()
        rc = [[(2, 3), (4, 5)], [(6, 7)]]
        expected = (
            _encode_basefold_proof(open_proof, component_raw_roots)
            # batch evaluations: one Tensor per committed round
            + _u64(1)
            + _u64(2)
            + struct.pack("<2I", 31, 32)
            + _u64(1)
            + _u64(2)
            # outer sumcheck; final_eval = 1 + 2*4 + 3*16 = 57 at point[0]
            + _u64(1)
            + _u64(3)
            + struct.pack("<3I", 1, 2, 3)
            + struct.pack("<I", 5)
            + _u64(1)
            + struct.pack("<I", 4)
            + struct.pack("<I", 57)
            # inner sumcheck; final_eval = 6 + 7*9 + 8*81 = 717
            + _u64(1)
            + _u64(3)
            + struct.pack("<3I", 6, 7, 8)
            + struct.pack("<I", 10)
            + _u64(1)
            + struct.pack("<I", 9)
            + struct.pack("<I", 717)
            # per-round (row_count, column_count) layout
            + _u64(2)
            + _u64(2)
            + _u64(2)
            + _u64(3)
            + _u64(4)
            + _u64(5)
            + _u64(1)
            + _u64(6)
            + _u64(7)
            # original commitments = the SMCS commitments (not the raw roots)
            + _u64(1)
            + struct.pack("<8I", *range(200, 208))
            + struct.pack("<I", 20)
            + _u64(22)  # max_log_row_count
            + _u64(1)  # log_m, read off the outer round count
        )
        self.assertEqual(
            _encode_evaluation_proof(
                eval_msg,
                open_proof,
                component_raw_roots,
                component_commitments,
                rc,
                max_log_row_count=22,
            ),
            expected,
        )


def _digest_layers(root_base: int) -> list:
    # The assembly reads only the raw root (the root layer's single row); the
    # bridge holds the digest tree, not the mle -- the open recomputes the mle
    # from the region dense (fractalyze/sp1-zorch#264).
    return [
        jnp.zeros((2, 8), dtype=F),
        jnp.arange(root_base, root_base + 8, dtype=F).reshape(1, 8),
    ]


def _bridge() -> ShardBridge:
    # "alpha": 2 main cols x 3 rows + 1 prep col; "lookup": 1 main col x 2
    # rows, no prep. Trailing two counts per region are the stacking dummies.
    main_region = JaggedRegion(
        dense=jnp.zeros(8, dtype=F),
        chip_starts=(0, 6, 8),
        row_counts=(3, 2, 4, 1),
        column_counts=(2, 1, 1, 1),
        log_stacking_height=2,
        chip_names=("alpha", "lookup"),
    )
    prep_region = JaggedRegion(
        dense=jnp.zeros(3, dtype=F),
        chip_starts=(0, 3),
        row_counts=(3, 4, 1),
        column_counts=(1, 1, 1),
        log_stacking_height=2,
        chip_names=("alpha",),
    )
    return ShardBridge(
        main_region=main_region,
        prep_region=prep_region,
        public_values=jnp.arange(1, 6, dtype=F),
        commit_digest_layers=(_digest_layers(100), _digest_layers(400)),
        # SMCS commitments (original_commitments), distinct from the raw roots.
        commit_commitments=(
            jnp.arange(200, 208, dtype=F),
            jnp.arange(500, 508, dtype=F),
        ),
        zc_opened_values=_opened_values(),
    )


def _opened_values() -> dict[str, ChipEvaluation]:
    # The stage's split of _zerocheck_proof's finals ([main | prep] stacks,
    # evaluation at position 0).
    return {
        "alpha": ChipEvaluation(
            main=jnp.array([31, 32], dtype=F),
            preprocessed=jnp.array([33], dtype=F),
        ),
        "lookup": ChipEvaluation(main=jnp.array([41], dtype=F), preprocessed=None),
    }


def _zerocheck_proof() -> ZerocheckProof:
    return ZerocheckProof(
        batching_challenge=jnp.array(1, dtype=F),
        gkr_opening_batch_challenge=jnp.array(2, dtype=F),
        lambda_=jnp.array(3, dtype=F),
        zeta=jnp.array([4], dtype=F),
        claimed_sum=jnp.array(9, dtype=F),
        finals=[
            # [main | prep] column stacks; the evaluation sits at position 0.
            jnp.array([[31, 0], [32, 0], [33, 0]], dtype=F),
            jnp.array([[41, 0]], dtype=F),
        ],
        opened_values=_opened_values(),
        msgs=RoundMsg(
            round_poly=jnp.array([[1, 2, 3], [4, 5, 6]], dtype=F),
            challenge=jnp.array([7, 8], dtype=F),
        ),
    )


class ChipOpenedValuesTest(absltest.TestCase):
    def test_bridges_carry_openings_with_live_row_degree(self) -> None:
        # The split itself is the zerocheck stage's
        # (zerocheck.prover.split_opened_values, pinned in prover_test); this
        # bridge adds the wire's degree off the chip heights.
        values = chip_opened_values(_bridge())
        self.assertLen(values, 2)

        alpha, lookup = values
        self.assertEqual(alpha.main_evals.tolist(), [31, 32])
        self.assertEqual(alpha.preprocessed_evals.tolist(), [33])
        self.assertEqual(alpha.degree, 3)

        self.assertEqual(lookup.main_evals.tolist(), [41])
        self.assertIsNone(lookup.preprocessed_evals)
        self.assertEqual(lookup.degree, 2)

    def test_rejects_a_bridge_without_opened_values(self) -> None:
        bridge = ShardBridge(
            main_region=_bridge().main_region,
            prep_region=None,
            public_values=jnp.arange(1, 6, dtype=F),
        )
        with self.assertRaisesRegex(ValueError, "opened values"):
            chip_opened_values(bridge)


class EncodeShardProofTest(absltest.TestCase):
    """Wiring golden for the full-proof assembly: the component encoders
    carry their own byte goldens above, so this pins the bridging decisions —
    wire field order, the zerocheck point reversal, the finals split, the
    raw-root extraction off the bridge's stacked witnesses, and the per-round
    layout with the stacking dummies included."""

    def test_wire_order_and_bridged_values(self) -> None:
        bridge = _bridge()
        zerocheck = _zerocheck_proof()
        gkr = _gkr_proof()
        jagged = ShardJaggedEvalProof(eval=_eval_msg(), open=_open_proof(num_rounds=2))
        commitment = jnp.arange(50, 58, dtype=F)

        covs = [
            ChipOpenedValues(
                preprocessed_evals=jnp.array([33], dtype=F),
                main_evals=jnp.array([31, 32], dtype=F),
                degree=3,
            ),
            ChipOpenedValues(
                preprocessed_evals=None,
                main_evals=jnp.array([41], dtype=F),
                degree=2,
            ),
        ]
        expected = (
            _u64(5)
            + _field_bytes(bridge.public_values)
            + _field_bytes(commitment)
            + _encode_logup_gkr_proof(gkr, 3)
            # The zerocheck point is the challenge list reversed: [7, 8] -> [8, 7].
            + _encode_partial_sumcheck_proof(
                zerocheck.msgs.round_poly,
                zerocheck.claimed_sum,
                jnp.array([8, 7], dtype=F),
            )
            + _encode_shard_opened_values(covs, ["alpha", "lookup"], 3)
            + _encode_evaluation_proof(
                jagged.eval,
                jagged.open,
                # Raw roots off the bridge's stacked witnesses, [prep, main].
                [jnp.arange(100, 108, dtype=F), jnp.arange(400, 408, dtype=F)],
                # original_commitments = the bridge's SMCS commitments, [prep, main].
                [jnp.arange(200, 208, dtype=F), jnp.arange(500, 508, dtype=F)],
                # Region layouts with the stacking dummies included.
                [[(3, 1), (4, 1), (1, 1)], [(3, 2), (2, 1), (4, 1), (1, 1)]],
                max_log_row_count=3,
            )
        )
        self.assertEqual(
            encode_shard_proof(
                bridge,
                commitment,
                gkr,
                zerocheck,
                jagged,
                max_log_row_count=3,
            ),
            expected,
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
