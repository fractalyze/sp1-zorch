# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""First-layer layout plumbing the rsp byte-match relies on.

These tests pin the LAYOUT — slot accounting, even/odd row routing, padding
placement, chip/interaction ordering — with expected values computed through
the same field ops. The value-level check is the byte-match against the SP1
reference dump, which runs against the rsp capture separately (GPU)."""

from types import SimpleNamespace

import jax.numpy as jnp
from absl.testing import absltest
from rw_constraints import Interaction, VirtualPairCol
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    GkrChip,
    build_gkr_chips,
    generate_circuit_layers,
    generate_first_layer,
    sp1_col_h,
    sp1_next_row_counts,
)
from sp1_zorch.shard_prover.chip_loader import make_chip_stub

ALPHA = jnp.array(7, F)
BETAS = jnp.array([5, 11], F)


def _interaction(mult_col: int, val_col: int, *, kind: int = 3, is_send: bool = True):
    return Interaction(
        values=(VirtualPairCol.single_main(val_col),),
        multiplicity=VirtualPairCol.single_main(mult_col),
        kind=kind,
        is_send=is_send,
    )


def _region(*chips, names):
    return JaggedRegion.from_chips(
        list(chips),
        log_stacking_height=3,
        max_log_row_count=5,
        chip_names=names,
    )


def _main(height: int, width: int = 2, offset: int = 0):
    return (
        jnp.arange(offset, offset + height * width, dtype=jnp.uint32)
        .reshape(height, width)
        .view(F)
    )


def _expected_vals(trace, inter, prep=None):
    """(mult, fingerprint) per row by direct formula."""
    mult = inter.multiplicity.apply_batch(prep, trace)
    if not inter.is_send:
        mult = -mult
    fp = jnp.broadcast_to(ALPHA + BETAS[0] * inter.kind, (trace.shape[0],))
    for i, vpc in enumerate(inter.values):
        fp = fp + BETAS[i + 1] * vpc.apply_batch(prep, trace)
    return mult, fp


class Sp1ColHTest(absltest.TestCase):
    def test_matches_populate_last_circuit_layer(self) -> None:
        self.assertEqual(sp1_col_h(0), 2)
        self.assertEqual(sp1_col_h(8), 2)
        self.assertEqual(sp1_col_h(9), 3)
        self.assertEqual(sp1_col_h(150704), 37676)


class GenerateFirstLayerTest(absltest.TestCase):
    def test_even_odd_routing_and_slot_padding(self) -> None:
        main = _main(6)
        inter = _interaction(0, 1)
        layer = generate_first_layer(
            [GkrChip("A", (inter,))], _region(main, names=("A",)), None, ALPHA, BETAS
        )
        mult, fp = _expected_vals(main, inter)

        # col_h(6) = 2 -> 4 slots; (6+1)//2 = 3 real pairs + 1 neutral pad.
        self.assertEqual(layer.row_counts, (4,))
        self.assertTrue(bool(jnp.all(layer.numerator_0[:3] == mult[0::2])))
        self.assertTrue(bool(jnp.all(layer.numerator_1[:3] == mult[1::2])))
        self.assertTrue(bool(jnp.all(layer.denominator_0[:3] == fp[0::2])))
        self.assertTrue(bool(jnp.all(layer.denominator_1[:3] == fp[1::2])))
        self.assertTrue(bool(layer.numerator_0[3] == jnp.array(0, F)))
        self.assertTrue(bool(layer.numerator_1[3] == jnp.array(0, F)))
        self.assertTrue(bool(layer.denominator_0[3] == jnp.array(1, F)))
        self.assertTrue(bool(layer.denominator_1[3] == jnp.array(1, F)))

    def test_receive_negates_multiplicity(self) -> None:
        main = _main(8)
        recv = _interaction(0, 1, is_send=False)
        layer = generate_first_layer(
            [GkrChip("A", (recv,))], _region(main, names=("A",)), None, ALPHA, BETAS
        )
        mult = recv.multiplicity.apply_batch(None, main)
        self.assertTrue(bool(jnp.all(layer.numerator_0 == -mult[0::2])))

    def test_interactions_pad_to_power_of_two(self) -> None:
        main_a, main_b = _main(8), _main(4, offset=100)
        chips = [
            GkrChip("A", (_interaction(0, 1), _interaction(1, 0, kind=5))),
            GkrChip("B", (_interaction(0, 1, kind=7),)),
        ]
        layer = generate_first_layer(
            chips, _region(main_a, main_b, names=("A", "B")), None, ALPHA, BETAS
        )
        # 3 real interactions pad to 4; the pad slot is 2 * sp1_col_h(0) = 4
        # rows of the neutral fraction.
        self.assertEqual(layer.row_counts, (4, 4, 4, 4))
        self.assertEqual(layer.num_interaction_variables, 2)
        lo = layer.start_indices[3]
        self.assertTrue(bool(jnp.all(layer.numerator_0[lo:] == jnp.array(0, F))))
        self.assertTrue(bool(jnp.all(layer.denominator_0[lo:] == jnp.array(1, F))))
        self.assertTrue(bool(jnp.all(layer.denominator_1[lo:] == jnp.array(1, F))))

    def test_zero_interaction_chip_contributes_nothing(self) -> None:
        main_a, main_b = _main(8), _main(4, offset=100)
        chips = [GkrChip("A", ()), GkrChip("B", (_interaction(0, 1),))]
        layer = generate_first_layer(
            chips, _region(main_a, main_b, names=("A", "B")), None, ALPHA, BETAS
        )
        mult, _ = _expected_vals(main_b, chips[1].interactions[0])
        # col_h clamps at max(real_h, 8): h=4 still reserves 4 slots.
        self.assertEqual(layer.row_counts, (4,))
        self.assertTrue(bool(jnp.all(layer.numerator_0[:2] == mult[0::2])))
        self.assertTrue(bool(jnp.all(layer.numerator_0[2:] == jnp.array(0, F))))

    def test_prep_column_feeds_fingerprint_and_trims_to_main_height(self) -> None:
        main = _main(4)
        prep = _main(8, width=1, offset=200)  # keygen-height prep, taller
        inter = Interaction(
            values=(VirtualPairCol(constant=0, column_weights=((0, True, 1),)),),
            multiplicity=VirtualPairCol.single_main(0),
            kind=2,
            is_send=True,
        )
        layer = generate_first_layer(
            [GkrChip("A", (inter,))],
            _region(main, names=("A",)),
            _region(prep, names=("A",)),
            ALPHA,
            BETAS,
        )
        _, fp = _expected_vals(main, inter, prep=prep[:4])
        # h=4 -> 4 slots, 2 real pairs; the rest is neutral padding.
        self.assertTrue(bool(jnp.all(layer.denominator_0[:2] == fp[0::2])))
        self.assertTrue(bool(jnp.all(layer.denominator_1[:2] == fp[1::2])))

    def test_rejects_odd_real_height(self) -> None:
        # An odd height makes the n1 side one slot short of n0's — the SP1
        # reference never produces one, so fail loud instead of building a
        # silently ragged layer.
        main = _main(5)
        with self.assertRaises(ValueError):
            generate_first_layer(
                [GkrChip("A", (_interaction(0, 1),))],
                _region(main, names=("A",)),
                None,
                ALPHA,
                BETAS,
            )


class Sp1NextRowCountsTest(absltest.TestCase):
    def test_matches_sp1_jagged_mle_schedule(self) -> None:
        # ceil(rc / 4) * 2: halves multiples of 4, rounds the fold up to
        # even otherwise, saturates at 2.
        self.assertEqual(sp1_next_row_counts((4, 8, 150704)), (2, 4, 75352))
        self.assertEqual(sp1_next_row_counts((10,)), (6,))
        self.assertEqual(sp1_next_row_counts((2,)), (2,))


class GenerateCircuitLayersTest(absltest.TestCase):
    def _first_layer(self):
        # col_h(24) = 6 -> 12 slots; col_h(4) clamps at 8 -> 4 slots.
        main_a, main_b = _main(24), _main(4, offset=100)
        chips = [
            GkrChip("A", (_interaction(0, 1),)),
            GkrChip("B", (_interaction(0, 1, kind=5),)),
        ]
        return generate_first_layer(
            chips, _region(main_a, main_b, names=("A", "B")), None, ALPHA, BETAS
        )

    def test_layer_shapes_follow_schedule(self) -> None:
        layers = generate_circuit_layers(self._first_layer(), 4)
        self.assertEqual(
            [layer.row_counts for layer in layers],
            [(12, 4), (6, 2), (4, 2), (2, 2)],
        )

    def test_saturated_segment_repads_neutral(self) -> None:
        # B saturates at rc=2 after the first step; each later fold collapses
        # it to one real fraction and re-pads the 1-child slot with (n=0, d=1).
        layers = generate_circuit_layers(self._first_layer(), 4)
        last = layers[-1]
        b_pad = last.start_indices[1] + 1
        self.assertTrue(bool(last.numerator_0[b_pad] == jnp.array(0, F)))
        self.assertTrue(bool(last.numerator_1[b_pad] == jnp.array(0, F)))
        self.assertTrue(bool(last.denominator_0[b_pad] == jnp.array(1, F)))
        self.assertTrue(bool(last.denominator_1[b_pad] == jnp.array(1, F)))

    def test_depth_one_keeps_only_the_first_layer(self) -> None:
        first = self._first_layer()
        layers = generate_circuit_layers(first, 1)
        self.assertEqual(len(layers), 1)
        self.assertIs(layers[0], first)

    def test_rejects_nonpositive_depth(self) -> None:
        with self.assertRaises(ValueError):
            generate_circuit_layers(self._first_layer(), 0)


class BuildGkrChipsTest(absltest.TestCase):
    def _chip_with_infos(self, name, infos):
        chip = make_chip_stub(name, 2)
        chip._interaction_info = infos
        return chip

    def test_sends_sorted_then_receives_sorted(self) -> None:
        s0 = _interaction(0, 1, kind=1)
        s1 = _interaction(0, 1, kind=2)
        r0 = _interaction(0, 1, kind=3, is_send=False)
        chip = self._chip_with_infos(
            "A",
            {
                "f_r0": SimpleNamespace(kind="receive", sp1_index=0, interaction=r0),
                "f_s1": SimpleNamespace(kind="send", sp1_index=1, interaction=s1),
                "f_s0": SimpleNamespace(kind="send", sp1_index=0, interaction=s0),
            },
        )
        (gkr,) = build_gkr_chips({"A": chip}, ("A",))
        self.assertEqual(gkr.interactions, (s0, s1, r0))

    def test_zero_interaction_chip_kept(self) -> None:
        chip = self._chip_with_infos("A", {})
        (gkr,) = build_gkr_chips({"A": chip}, ("A",))
        self.assertEqual(gkr.interactions, ())

    def test_stub_chip_skipped(self) -> None:
        stub = self._chip_with_infos(
            "A", {"f": SimpleNamespace(kind="send", sp1_index=0, interaction=None)}
        )
        self.assertEqual(build_gkr_chips({"A": stub}, ("A",)), ())


if __name__ == "__main__":
    absltest.main()
