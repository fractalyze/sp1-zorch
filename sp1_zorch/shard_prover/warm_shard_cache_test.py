# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Per-class warm selection: the greedy shard cover of a dump's class-keyed
compile keys (``select_warm_shards``) and its RAISE-on-gap cover contract
(``check_warm_cover``), on synthetic analyze fixtures. The per-shard jagged
pack zone is outside the contract (see ``_COVER_KINDS``)."""

from collections.abc import Sequence
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized

from sp1_zorch.shard_prover import warm_shard_cache as wsc

_JAGGED = {"L": 12, "n_d": 20, "K": [4], "rlc_bits": 2}


def _cls(
    order: Sequence[str],
    area: int,
    heights: dict[str, int] | None = None,
    jagged: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One shard's analyze record (the ``_analyze`` classes-dict shape)."""
    order = list(order)
    heights = dict(heights) if heights else {n: 64 for n in order}
    return {
        "order": order,
        "area_cap": int(area),
        "gkr_heights": heights,
        "gkr_slot_bound": sum(heights.values()),
        "jagged": dict(jagged if jagged is not None else _JAGGED),
    }


def _groups(classes: dict[str, Any]) -> dict[tuple[str, ...], list[str]]:
    groups: dict[tuple[str, ...], list[str]] = {}
    for name, c in classes.items():
        groups.setdefault(tuple(c["order"]), []).append(name)
    return groups


def _manifest(classes: dict[str, Any]) -> dict[str, Any]:
    """The real group manifest for the fixture, via the production planner."""
    return wsc._plan(classes, _groups(classes))["manifest"]


class SelectWarmShardsTest(absltest.TestCase):
    def test_tight_group_selects_single_area_max_rep(self) -> None:
        classes = {
            "shard0": _cls(("a", "b"), 400),
            "shard1": _cls(("a", "b"), 402),
            "shard2": _cls(("a", "b"), 399),
        }
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        self.assertEqual(sel, ["shard1"])

    def test_loose_group_selects_one_per_distinct_area(self) -> None:
        # 100/400 < group_area_ratio: each shard keeps its own zerocheck
        # class; the two 400s share one, area-tie broken by shard number.
        classes = {
            "shard0": _cls(("a", "b"), 400),
            "shard1": _cls(("a", "b"), 100),
            "shard2": _cls(("a", "b"), 400),
        }
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        self.assertEqual(sel, ["shard0", "shard1"])

    def test_distinct_chip_sets_each_covered(self) -> None:
        classes = {
            "shard0": _cls(("a",), 100),
            "shard1": _cls(("b",), 100),
        }
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        self.assertEqual(sel, ["shard0", "shard1"])

    def test_gkr_gap_adds_its_carrier(self) -> None:
        # One pinned zerocheck class, but a manifest with divergent GKR pins:
        # the second GKR class's carrier must ride along.
        classes = {
            "shard0": _cls(("a", "b"), 402),
            "shard1": _cls(("a", "b"), 402),
        }
        manifest = {
            "shard0": {"area_cap": 402, "gkr": {"a": 64, "b": 64}, "gkr_slot_cap": 128},
            "shard1": {"area_cap": 402, "gkr": {"a": 96, "b": 64}, "gkr_slot_cap": 160},
        }
        sel = wsc.select_warm_shards(classes, manifest)
        self.assertEqual(sel, ["shard0", "shard1"])

    def test_jagged_gap_adds_its_carrier(self) -> None:
        # Same zerocheck + GKR pins (tight group), distinct jagged K: the
        # rider's open-zone compile is real, so the rider stays selected.
        classes = {
            "shard0": _cls(("a", "b"), 402),
            "shard1": _cls(
                ("a", "b"), 400, jagged={"L": 12, "n_d": 20, "K": [8], "rlc_bits": 3}
            ),
        }
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        self.assertEqual(sel, ["shard0", "shard1"])

    def test_pack_zone_is_outside_the_cover(self) -> None:
        # Two shards can share every class key yet differ in exact row counts,
        # which key the per-shard ``_jagged_pack_jit`` zone. That zone is
        # outside the cover contract — its key never reaches the analyze
        # record — so the selection still collapses to one representative and
        # the rider pays its pack compile cold on first prove.
        classes = {"shard0": _cls(("a", "b"), 402), "shard1": _cls(("a", "b"), 402)}
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        self.assertEqual(sel, ["shard0"])

    def test_per_class_false_returns_all_shards(self) -> None:
        classes = {f"shard{i}": _cls(("a", "b"), 400) for i in range(4)}
        sel = wsc.select_warm_shards(classes, {}, per_class=False)
        self.assertEqual(sel, ["shard0", "shard1", "shard2", "shard3"])

    def test_selection_banner_reports_k_of_n(self) -> None:
        classes = {
            "shard0": _cls(("a", "b"), 400),
            "shard1": _cls(("a", "b"), 402),
            "shard2": _cls(("a", "b"), 399),
            "shard3": _cls(("keccak",), 90),
            "shard4": _cls(("keccak",), 88),
        }
        sel = wsc.select_warm_shards(classes, _manifest(classes))
        banner = wsc._selection_banner(sel, len(classes))
        self.assertEqual(
            banner,
            "warming 2 of 5 shards (compile-key cover): ['shard1', 'shard3']",
        )


class CheckWarmCoverTest(parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "chipset",
            {"shard0": _cls(("a",), 100), "shard1": _cls(("b",), 100)},
        ),
        (
            "zerocheck",
            {"shard0": _cls(("a",), 100), "shard1": _cls(("a",), 200)},
        ),
        (
            "gkr",
            {
                "shard0": _cls(("a",), 100, heights={"a": 64}),
                "shard1": _cls(("a",), 100, heights={"a": 128}),
            },
        ),
        (
            "jagged",
            {
                "shard0": _cls(("a",), 100),
                "shard1": _cls(
                    ("a",), 100, jagged={"L": 9, "n_d": 20, "K": [4], "rlc_bits": 2}
                ),
            },
        ),
    )
    def test_raises_on_gap(self, classes: dict[str, Any]) -> None:
        with self.assertRaisesRegex(ValueError, "misses compile keys"):
            wsc.check_warm_cover(["shard0"], classes, {})

    def test_passes_on_full_selection(self) -> None:
        classes = {"shard0": _cls(("a",), 100), "shard1": _cls(("b",), 200)}
        wsc.check_warm_cover(["shard0", "shard1"], classes, {})


if __name__ == "__main__":
    absltest.main()
