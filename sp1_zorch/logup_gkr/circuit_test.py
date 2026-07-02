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
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.logup_gkr.circuit import (
    GkrChip,
    _chip_view,
    build_gkr_chips,
    generate_first_layer,
    sp1_col_h,
    sp1_next_row_counts,
    sp1_schedules,
)
from sp1_zorch.shard_prover.chip_loader import make_chip_stub
from zorch.logup_gkr.circuit import (
    JaggedGkrLayer,
    jagged_layer_transition,
    scan_build_jagged_pyramid,
)
from zorch.utils.bits import log2_ceil_usize


def generate_circuit_layers(
    first_layer: JaggedGkrLayer, num_row_variables: int
) -> list[JaggedGkrLayer]:
    """Eager reference pyramid: fold the first layer to the floor through SP1's
    fixed-depth schedule -- ``num_row_variables - 1`` transitions
    (``max_log_row_count - 1`` on a core shard), one ``jagged_layer_transition``
    per step. The reference the fused ``scan_build_jagged_pyramid`` is checked
    against.
    """
    if num_row_variables < 1:
        raise ValueError(f"num_row_variables must be >= 1, got {num_row_variables}")
    layers = [first_layer]
    counts = first_layer.row_counts
    for _ in range(num_row_variables - 1):
        counts = sp1_next_row_counts(counts)
        layers.append(jagged_layer_transition(layers[-1], counts))
    return layers

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


class ScanBuildJaggedPyramidWiringTest(absltest.TestCase):
    """Lever A (sp1-zorch#143): building the pyramid via zorch's fused
    `scan_build_jagged_pyramid` under the SP1 schedule must be byte-identical to
    the eager `generate_circuit_layers` reference loop."""

    def _first_layer(self, alpha, betas):
        # col_h(24) = 6 -> 12 slots; col_h(4) clamps at 8 -> 4 slots.
        main_a, main_b = _main(24), _main(4, offset=100)
        chips = [
            GkrChip("A", (_interaction(0, 1),)),
            GkrChip("B", (_interaction(0, 1, kind=5),)),
        ]
        return generate_first_layer(
            chips, _region(main_a, main_b, names=("A", "B")), None, alpha, betas
        )

    def _assert_layers_byte_equal(self, got, want):
        self.assertEqual(len(got), len(want))
        for i, (g, w) in enumerate(zip(got, want)):
            self.assertEqual(g.row_counts, w.row_counts, msg=f"layer {i} row_counts")
            for plane in (
                "numerator_0",
                "numerator_1",
                "denominator_0",
                "denominator_1",
            ):
                gp, wp = getattr(g, plane), getattr(w, plane)
                self.assertEqual(gp.dtype, wp.dtype, msg=f"layer {i} {plane} dtype")
                self.assertEqual(gp.shape, wp.shape, msg=f"layer {i} {plane} shape")
                self.assertTrue(bool(jnp.all(gp == wp)), msg=f"layer {i} {plane}")

    def test_schedule_is_the_eager_loop_count_sequence(self) -> None:
        # sp1_schedules is exactly the per-transition out_row_counts
        # generate_circuit_layers folds through (the [1:] of its layer shapes).
        first = self._first_layer(ALPHA, BETAS)
        self.assertEqual(sp1_schedules(first.row_counts, 4), [(6, 2), (4, 2), (2, 2)])

    def test_depth_one_has_no_transitions(self) -> None:
        first = self._first_layer(ALPHA, BETAS)
        self.assertEqual(sp1_schedules(first.row_counts, 1), [])

    def test_rejects_nonpositive_depth(self) -> None:
        # A sub-1 depth must fail loud, not silently yield an empty schedule.
        first = self._first_layer(ALPHA, BETAS)
        with self.assertRaises(ValueError):
            sp1_schedules(first.row_counts, 0)

    def test_scan_build_matches_eager_same_field(self) -> None:
        # Base-field alpha keeps numerator and denominator in one field, so
        # scan_build takes its pure-scan path (no base->EF carve-out).
        first = self._first_layer(ALPHA, BETAS)
        scanned = scan_build_jagged_pyramid(first, sp1_schedules(first.row_counts, 4))
        self._assert_layers_byte_equal(scanned, generate_circuit_layers(first, 4))

    def test_scan_build_matches_eager_mixed_field(self) -> None:
        # The real prover's first layer carries base-field multiplicities under
        # extension-field denominators (alpha is an EF head challenge); scan_build
        # carves the base->EF promoting first transition out eagerly. Pin that
        # carve-out path byte-identical to the eager loop too.
        first = self._first_layer(ALPHA.astype(EF), BETAS.astype(EF))
        scanned = scan_build_jagged_pyramid(first, sp1_schedules(first.row_counts, 4))
        self._assert_layers_byte_equal(scanned, generate_circuit_layers(first, 4))


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


def _reference_first_layer(chips, main_region, prep_region, alpha, betas):
    """Pre-vectorization per-interaction build (the SP1-byte-matched path):
    ``apply_batch`` per interaction, even/odd split + fold-neutral pad, then
    power-of-two interaction padding. The oracle for the batched build."""
    total = sum(len(c.interactions) for c in chips)
    padded = 1 << log2_ceil_usize(total)
    m_idx = {n: i for i, n in enumerate(main_region.chip_names)}
    p_idx = (
        {n: i for i, n in enumerate(prep_region.chip_names)}
        if prep_region is not None
        else {}
    )
    bf, ef = main_region.dense.dtype, alpha.dtype
    n0, n1, d0, d1, rc = [], [], [], [], []
    for chip in chips:
        if not chip.interactions:
            continue
        main = _chip_view(main_region, m_idx[chip.name])
        h = main.shape[0]
        prep = _chip_view(prep_region, p_idx[chip.name])[:h] if chip.name in p_idx else None
        slot = 2 * sp1_col_h(h)
        pad = slot - h // 2
        pn, pd = jnp.zeros(pad, dtype=bf), jnp.ones(pad, dtype=ef)
        for inter in chip.interactions:
            mult = inter.multiplicity.apply_batch(prep, main)
            if not inter.is_send:
                mult = -mult
            fp = jnp.broadcast_to(alpha + betas[0] * inter.kind, (h,))
            for i, v in enumerate(inter.values):
                fp = fp + betas[i + 1] * v.apply_batch(prep, main)
            n0.append(jnp.concatenate([mult[0::2], pn]))
            n1.append(jnp.concatenate([mult[1::2], pn]))
            d0.append(jnp.concatenate([fp[0::2], pd]))
            d1.append(jnp.concatenate([fp[1::2], pd]))
            rc.append(slot)
    pad_slot = 2 * sp1_col_h(0)
    if (npad := padded - total) > 0:
        t = npad * pad_slot
        n0.append(jnp.zeros(t, dtype=bf))
        n1.append(jnp.zeros(t, dtype=bf))
        d0.append(jnp.ones(t, dtype=ef))
        d1.append(jnp.ones(t, dtype=ef))
        rc.extend([pad_slot] * npad)
    return (
        jnp.concatenate(n0),
        jnp.concatenate(n1),
        jnp.concatenate(d0),
        jnp.concatenate(d1),
        tuple(rc),
    )


# Betas long enough to fingerprint up to three value columns (betas[0]..[3]).
BETAS3 = jnp.array([5, 11, 13, 2], F)


def _vpc(constant, *terms):
    """A VirtualPairCol from ``(col, is_prep, weight)`` term tuples."""
    return VirtualPairCol(constant=constant, column_weights=tuple(terms))


class FirstLayerBatchedEquivalenceTest(absltest.TestCase):
    """The batched build must be byte-identical to the per-interaction
    reference across the cases the matmul / segment_sum path newly exercises:
    multi-term and constant affine forms, duplicate columns, multi-value
    fingerprints, send/receive mixes, prep columns, power-of-two padding, and
    interaction counts past the old 128 chunk size."""

    def _assert_equiv(self, chips, main_region, prep_region, betas):
        got = generate_first_layer(chips, main_region, prep_region, ALPHA, betas)
        rn0, rn1, rd0, rd1, rrc = _reference_first_layer(
            chips, main_region, prep_region, ALPHA, betas
        )
        self.assertEqual(got.row_counts, rrc)
        for name, ref in (
            ("numerator_0", rn0),
            ("numerator_1", rn1),
            ("denominator_0", rd0),
            ("denominator_1", rd1),
        ):
            self.assertTrue(
                bool(jnp.all(getattr(got, name) == ref)), f"{name} diverged"
            )

    def test_multi_term_constant_and_duplicate_columns(self) -> None:
        main = _main(8, width=4)
        chips = [
            GkrChip(
                "A",
                (
                    # constant + two distinct columns
                    Interaction(
                        values=(_vpc(3, (0, False, 2), (2, False, 5)),),
                        multiplicity=_vpc(1, (1, False, 4)),
                        kind=3,
                        is_send=True,
                    ),
                    # same column twice -> weights accumulate
                    Interaction(
                        values=(_vpc(0, (3, False, 7), (3, False, 9)),),
                        multiplicity=_vpc(0, (0, False, 1), (0, False, 1)),
                        kind=5,
                        is_send=False,
                    ),
                ),
            )
        ]
        self._assert_equiv(chips, _region(main, names=("A",)), None, BETAS)

    def test_multi_value_fingerprint_and_send_receive_mix(self) -> None:
        main = _main(10, width=4)
        chips = [
            GkrChip(
                "A",
                (
                    Interaction(
                        values=(
                            VirtualPairCol.single_main(0),
                            VirtualPairCol.single_main(1),
                            VirtualPairCol.single_main(2),
                        ),
                        multiplicity=VirtualPairCol.single_main(3),
                        kind=7,
                        is_send=True,
                    ),
                    Interaction(
                        values=(
                            VirtualPairCol.single_main(2),
                            VirtualPairCol.single_main(0),
                        ),
                        multiplicity=VirtualPairCol.single_main(1),
                        kind=4,
                        is_send=False,
                    ),
                ),
            )
        ]
        self._assert_equiv(chips, _region(main, names=("A",)), None, BETAS3)

    def test_prep_and_padding_interactions(self) -> None:
        main = _main(4, width=2)
        prep = _main(8, width=2, offset=200)
        chips = [
            GkrChip(
                "A",
                (
                    Interaction(
                        values=(_vpc(0, (0, True, 1), (1, False, 3)),),
                        multiplicity=_vpc(0, (0, True, 2)),
                        kind=2,
                        is_send=True,
                    ),
                ),
            ),
            GkrChip("B", (_interaction(0, 1, kind=9),)),
        ]
        # 2 interactions already power-of-two; add a chip to force padding to 4.
        self._assert_equiv(
            chips,
            _region(main, _main(6, width=2, offset=50), names=("A", "B")),
            _region(prep, names=("A",)),
            BETAS,
        )

    def test_prep_region_chip_without_prep_columns(self) -> None:
        # A chip can sit in the prep region while no interaction reads a prep
        # column, and its prep view can be a different (here shorter) height.
        # The build must not fold prep into the column space then -- regression
        # for the unconditional [main|prep] concat (CI: prove_shard_test).
        main = _main(6, width=3)
        prep = _main(3, width=2, offset=99)  # present, unused, mismatched height
        chips = [GkrChip("A", (_interaction(0, 1), _interaction(1, 2, kind=4)))]
        self._assert_equiv(
            chips, _region(main, names=("A",)), _region(prep, names=("A",)), BETAS
        )

    def test_prep_column_form_with_no_prep_region(self) -> None:
        # A form may name a prep column on a chip with no prep region; both the
        # batched build and apply_batch treat it as a zero contribution.
        main = _main(8, width=2)
        inter = Interaction(
            values=(_vpc(0, (0, True, 5), (0, False, 1)),),
            multiplicity=VirtualPairCol.single_main(1),
            kind=2,
            is_send=True,
        )
        self._assert_equiv([GkrChip("A", (inter,))], _region(main, names=("A",)), None, BETAS)

    def test_many_interactions_past_old_chunk_size(self) -> None:
        # The old build chunked at 128 interactions; the batched build is one
        # matmul, so cross 128 to pin there is no residual chunk assumption.
        main = _main(8, width=4)
        inters = tuple(
            _interaction(i % 4, (i + 1) % 4, kind=(i % 6) + 1, is_send=(i % 2 == 0))
            for i in range(200)
        )
        self._assert_equiv([GkrChip("A", inters)], _region(main, names=("A",)), None, BETAS)


if __name__ == "__main__":
    absltest.main()
