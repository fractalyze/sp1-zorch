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


def _tc(area: int, gkr: GkrCapClass) -> TotalCapClass:
    """The zerocheck class `resolve_classes` returns: the resolved area plus
    the RESOLVED GKR class's per-chip heights, which double as the zerocheck's
    per-chip round-window caps (both bound a chip's real height)."""
    return TotalCapClass(area_cap=area, chip_height_caps=gkr.chip_heights)


class ResolveClassesTest(absltest.TestCase):
    _ORDER = ("a", "b")
    _OWN_TC = TotalCapClass(area_cap=100)
    _OWN_GKR = GkrCapClass((4, 6), 20)

    def _resolve(self, **kw: Any) -> tuple:
        return cc.resolve_classes(self._ORDER, self._OWN_TC, self._OWN_GKR, **kw)

    def test_no_pins_returns_own(self) -> None:
        tc, gkr = self._resolve()
        self.assertEqual(tc, _tc(self._OWN_TC.area_cap, self._OWN_GKR))
        self.assertEqual(gkr, self._OWN_GKR)

    def test_flag_specs_override_own(self) -> None:
        tc, gkr = self._resolve(
            zc_spec={"area_cap": 400},
            gkr_spec={"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36},
        )
        self.assertEqual(tc, _tc(400, gkr))
        self.assertEqual(gkr, GkrCapClass((8, 10), 36))

    def test_zerocheck_window_caps_track_the_resolved_gkr_class(self) -> None:
        # The zerocheck's per-chip round windows narrow off these caps, so
        # they must follow the GKR class the SAME resolution picked — a stale
        # pairing would either under-bound a chip's window (loud at prove
        # dispatch) or leave a wider window than the class allows.
        for kw in (
            {},
            {"gkr_spec": {"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36}},
            {"manifest_entry": {"area_cap": 500, "gkr": {"a": 12, "b": 14}}},
        ):
            with self.subTest(str(kw)):
                tc, gkr = self._resolve(**kw)
                self.assertEqual(tc.chip_height_caps, gkr.chip_heights)

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
        self.assertEqual(tc, _tc(500, gkr))
        self.assertEqual(gkr, GkrCapClass((12, 14), 52))

    def test_partial_manifest_entry_pins_only_named_fields(self) -> None:
        # area-only entry: zerocheck pinned by the entry, GKR falls through
        # to the flag spec.
        tc, gkr = self._resolve(
            gkr_spec={"chip_heights": {"a": 8, "b": 10}, "slot_cap": 36},
            manifest_entry={"area_cap": 500},
        )
        self.assertEqual(tc, _tc(500, gkr))
        self.assertEqual(gkr, GkrCapClass((8, 10), 36))

    def test_partial_manifest_entry_falls_through_to_own(self) -> None:
        tc, gkr = self._resolve(manifest_entry={"gkr": {"a": 12, "b": 14}})
        self.assertEqual(tc, _tc(self._OWN_TC.area_cap, gkr))
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


class QuantizePolicyTest(absltest.TestCase):
    """Golden vectors for the mirrored quantization constants — identical
    literals in zkvm-prover's ``test_manifest.py`` (``zkvm_sp1.manifest``
    owns the schema; a drift on either side fails that side's test)."""

    def test_area_golden_vectors(self) -> None:
        # Block A's core33 tight max and block B's (+1088 cells) land in ONE
        # bucket — the cross-block cap-jitter recompile closes by
        # construction.
        self.assertEqual(cc.quantize_area(401893824), 402096128)
        self.assertEqual(cc.quantize_area(401894912), 402096128)
        self.assertEqual(cc.quantize_area(400776800), 400982016)

    def test_slot_golden_vector(self) -> None:
        self.assertEqual(cc.quantize_slot(93165272), 96894976)

    def test_height_golden_vectors(self) -> None:
        # The three clamp arms of the per-chip slack: relative (mid chip),
        # absolute bound (big chip), absolute floor (tiny chip).
        self.assertEqual(cc.quantize_height(77824), 89088)
        self.assertEqual(cc.quantize_height(1688832), 1730560)
        self.assertEqual(cc.quantize_height(2240), 5120)
        self.assertEqual(cc.quantize_height(32), 3072)

    def test_quantize_covers_its_input(self) -> None:
        for x in (0, 1, 32767, 32768, 401893824):
            self.assertGreaterEqual(cc.quantize_area(x), x)


class ClassNameTest(absltest.TestCase):
    def test_slug_is_count_plus_sig8(self) -> None:
        self.assertEqual(cc.class_name(("Cpu", "Add")), "2ch-57ec70cf")
        self.assertEqual(cc.class_name(("KeccakPermute",)), "1ch-bf79035f")

    def test_order_independent(self) -> None:
        self.assertEqual(cc.class_name(("a", "b")), cc.class_name(("b", "a")))


class ManifestEntryForTest(absltest.TestCase):
    """The read-side class match, mirroring
    ``zkvm_sp1.manifest.entry_for_shard`` (zkvm-prover#176)."""

    _CORE = {"area_cap": 500, "gkr": {"a": 8, "b": 8}, "gkr_slot_cap": 32}

    def test_class_keyed_entry_matches_by_chip_set(self) -> None:
        manifest = {"2ch-7e18f737": self._CORE}
        self.assertIs(
            cc.manifest_entry_for(manifest, ("a", "b"), name="shard40"),
            manifest["2ch-7e18f737"],
        )

    def test_name_hit_requires_chip_set_agreement(self) -> None:
        # Cross-block name collision: the named entry is a foreign class,
        # the class-keyed entry must serve instead.
        manifest = {
            "shard40": {"area_cap": 9, "gkr": {"keccak": 2}},
            "2ch-7e18f737": self._CORE,
        }
        self.assertIs(
            cc.manifest_entry_for(manifest, ("a", "b"), name="shard40"),
            manifest["2ch-7e18f737"],
        )

    def test_matching_name_wins(self) -> None:
        manifest = {"shard3": self._CORE, "other": dict(self._CORE, area_cap=999)}
        self.assertIs(
            cc.manifest_entry_for(manifest, ("a", "b"), name="shard3"),
            manifest["shard3"],
        )

    def test_superset_entry_never_covers_a_smaller_shard(self) -> None:
        manifest = {"shard15": {"area_cap": 999, "gkr": {"a": 8, "b": 8, "sha": 8}}}
        self.assertIsNone(cc.manifest_entry_for(manifest, ("a", "b"), name="shard15"))

    def test_largest_area_cap_wins_among_same_set_entries(self) -> None:
        manifest = {
            "shard9": {"area_cap": 46, "gkr": {"a": 8}},
            "shard3": {"area_cap": 403, "gkr": {"a": 8}},
            "shard5": {"area_cap": 339, "gkr": {"a": 8}},
        }
        self.assertIs(cc.manifest_entry_for(manifest, ("a",)), manifest["shard3"])

    def test_area_only_named_entry_is_trusted(self) -> None:
        manifest = {"shard2": {"area_cap": 11}}
        self.assertIs(
            cc.manifest_entry_for(manifest, ("a", "b"), name="shard2"),
            manifest["shard2"],
        )

    def test_unknown_chip_set_matches_nothing(self) -> None:
        self.assertIsNone(cc.manifest_entry_for({}, ("a", "b"), name="shard0"))
        manifest = {"shard0": {"area_cap": 1, "gkr": {"z": 8}}}
        self.assertIsNone(cc.manifest_entry_for(manifest, ("a", "b"), name="shard0"))


if __name__ == "__main__":
    absltest.main()
