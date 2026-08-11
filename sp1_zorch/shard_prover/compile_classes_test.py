# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Compile-class derivation + pin resolution, on synthetic regions.

``resolve_classes`` is the ONE manifest/flag resolution both the staged prove
harness and the warm cache cover key on, so its precedence and field-by-field
override semantics are pinned here — a drift would silently split the classes
a warm fills from the classes a prove requests."""

from types import SimpleNamespace
from typing import Any, cast

from absl.testing import absltest

from sp1_zorch.logup_gkr.circuit import GkrCapClass
from sp1_zorch.shard_prover import compile_classes as cc
from sp1_zorch.zerocheck.jagged import TotalCapClass


def _region(
    names: Any,
    widths: Any,
    heights: Any,
    dense_rows: int,
    log_stack: int,
    column_counts: Any,
) -> SimpleNamespace:
    return SimpleNamespace(
        chip_names=tuple(names),
        chip_widths=list(widths),
        chip_heights=list(heights),
        dense=SimpleNamespace(shape=(dense_rows,)),
        log_stacking_height=log_stack,
        column_counts=list(column_counts),
    )


def _gkr_chip(name: str, n_interactions: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, interactions=[object()] * n_interactions)


class TightClassesTest(absltest.TestCase):
    def test_prep_widths_join_by_name_and_heights_evenpad(self) -> None:
        main = _region(("a", "b"), (3, 5), (4, 6), 64, 2, (3, 5))
        # Prep carries chip "b" only: its 2 columns join b's main width.
        prep = _region(("b",), (2,), (6,), 16, 2, (2,))
        gkr_chips = (_gkr_chip("a", 1), _gkr_chip("b", 2))
        tc, gkr, slot = cc.tight_classes(main, prep, ("a", "b"), [3, 6], gkr_chips)
        # area = Σ cols * evenpad(num_real): a = 3*4, b = (5+2)*6.
        self.assertEqual(tc.area_cap, 3 * 4 + 7 * 6)
        # GKR heights come from the REGION heights, even-padded.
        self.assertEqual(gkr.chip_heights, (4, 6))
        self.assertEqual(slot, gkr.resolved_slot_cap(cast(Any, gkr_chips), ("a", "b")))

    def test_no_prep_region(self) -> None:
        main = _region(("a",), (2,), (4,), 32, 2, (2,))
        tc, gkr, _ = cc.tight_classes(main, None, ("a",), [4], (_gkr_chip("a", 1),))
        self.assertEqual(tc.area_cap, 2 * 4)
        self.assertEqual(gkr.chip_heights, (4,))


class JaggedClassTest(absltest.TestCase):
    def test_derives_l_nd_k_rlc(self) -> None:
        main = _region(("a",), (3,), (4,), 1 << 6, 2, (2, 1))
        prep = _region(("p",), (1,), (2,), 1 << 4, 2, (1,))
        j = cc.jagged_class(main, prep)
        self.assertEqual(j["L"], 4)  # (2+1) main + 1 prep columns
        self.assertEqual(j["K"], [(1 << 4) >> 2, (1 << 6) >> 2])  # prep, main
        total = (1 << 6) + (1 << 4)
        self.assertEqual(j["n_d"], (total - 1).bit_length() + 1)
        self.assertEqual(j["rlc_bits"], (sum(j["K"]) - 1).bit_length())

    def test_no_prep_region(self) -> None:
        main = _region(("a",), (3,), (4,), 1 << 6, 3, (3,))
        j = cc.jagged_class(main, None)
        self.assertEqual(j["L"], 3)
        self.assertEqual(j["K"], [(1 << 6) >> 3])


class ResolveClassesTest(absltest.TestCase):
    _ORDER = ("a", "b")
    _OWN_TC = TotalCapClass(area_cap=100)
    _OWN_GKR = GkrCapClass((4, 6), 20)

    def _resolve(self, **kw: Any) -> tuple:
        return cc.resolve_classes(self._ORDER, self._OWN_TC, self._OWN_GKR, **kw)

    def test_no_pins_returns_own(self) -> None:
        tc, gkr = self._resolve()
        self.assertEqual(tc, self._OWN_TC)
        self.assertEqual(gkr, self._OWN_GKR)

    def test_flag_specs_override_own(self) -> None:
        tc, gkr = self._resolve(
            zc_spec={"area_cap": 400},
            gkr_spec={"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36},
        )
        self.assertEqual(tc, TotalCapClass(area_cap=400))
        self.assertEqual(gkr, GkrCapClass((8, 10), 36))

    def test_gkr_spec_without_slot_cap_leaves_none(self) -> None:
        # None defers the pyramid capacity to the chips-derived total at
        # prove time.
        _, gkr = self._resolve(gkr_spec={"chip_heights": {"a": 8, "b": 10}})
        self.assertIsNone(gkr.slot_cap)

    def test_manifest_entry_overrides_flag_specs(self) -> None:
        tc, gkr = self._resolve(
            zc_spec={"area_cap": 400},
            gkr_spec={"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36},
            manifest_entry={
                "area_cap": 500,
                "gkr": {"a": 12, "b": 14},
                "gkr_slot_cap": 52,
            },
        )
        self.assertEqual(tc, TotalCapClass(area_cap=500))
        self.assertEqual(gkr, GkrCapClass((12, 14), 52))

    def test_partial_manifest_entry_pins_only_named_fields(self) -> None:
        # area-only entry: zerocheck pinned by the entry, GKR falls through
        # to the flag spec.
        tc, gkr = self._resolve(
            gkr_spec={"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36},
            manifest_entry={"area_cap": 500},
        )
        self.assertEqual(tc, TotalCapClass(area_cap=500))
        self.assertEqual(gkr, GkrCapClass((8, 10), 36))

    def test_partial_manifest_entry_falls_through_to_own(self) -> None:
        tc, gkr = self._resolve(manifest_entry={"gkr": {"a": 12, "b": 14}})
        self.assertEqual(tc, self._OWN_TC)
        # An entry "gkr" without "gkr_slot_cap" defers the slot cap.
        self.assertEqual(gkr, GkrCapClass((12, 14), None))

    def test_gkr_heights_follow_order_not_entry_key_order(self) -> None:
        _, gkr = self._resolve(manifest_entry={"gkr": {"b": 14, "a": 12}})
        self.assertEqual(gkr.chip_heights, (12, 14))

    def test_values_coerced_to_int(self) -> None:
        tc, gkr = self._resolve(
            manifest_entry={
                "area_cap": "500",
                "gkr": {"a": "12", "b": "14"},
                "gkr_slot_cap": "52",
            }
        )
        self.assertEqual(tc.area_cap, 500)
        self.assertEqual(gkr, GkrCapClass((12, 14), 52))


if __name__ == "__main__":
    absltest.main()
