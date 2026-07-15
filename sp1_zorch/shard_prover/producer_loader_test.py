# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Producer ingest: fixture-loader parity and ingest-time failure modes."""

from pathlib import Path

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.shard_prover.chip_loader import (
    load_sp1_chips,
    rw_name_to_sp1,
    sp1_name_to_rw,
)
from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard
from sp1_zorch.shard_prover.producer_loader import regions_from_producer
from sp1_zorch.shard_prover.replay import shard_regions
from sp1_zorch.shard_prover.types import MachineVerifyingKey

_VK_TEXT = """\
preprocessed_commit=[1, 2, 3, 4, 5, 6, 7, 8]
pc_start=[9, 10, 11]
cum_sum_x=[1, 2, 3, 4, 5, 6, 7]
cum_sum_y=[8, 9, 10, 11, 12, 13, 14]
enable_untrusted=[0]
"""


def _write_chip(trace_dir: Path, name: str, num_real: int, width: int) -> np.ndarray:
    """Write one chip's .meta/.bin; returns the raw u32 matrix — the same
    bits the producer would hand as a device uint32 array."""
    (trace_dir / f"{name}.meta").write_text(f"num_real={num_real}\nwidth={width}\n")
    raw = np.arange(num_real * width, dtype=np.uint32).reshape(num_real, width)
    raw.tofile(trace_dir / f"{name}.bin")
    return raw


def _write_dump(root: Path) -> dict[str, np.ndarray]:
    """A synthetic dump the producer parity check replays: one chip at the
    rw manifest width (real constraints attach), two stubs whose rw / SP1
    sorted orders disagree (UType / Uint256MulMod — the collation flip the
    ingest must undo), a prep trace, and 187 public values."""
    trace_dir = root / "gpu_traces"
    prep_dir = trace_dir / "preprocessed"
    prep_dir.mkdir(parents=True)
    add_width = load_sp1_chips(chip_names=["add"])["add"].num_cols
    raws = {
        "Add": _write_chip(trace_dir, "Add", num_real=4, width=add_width),
        "UType": _write_chip(trace_dir, "UType", num_real=2, width=5),
        "Uint256MulMod": _write_chip(trace_dir, "Uint256MulMod", num_real=3, width=6),
    }
    _write_chip(prep_dir, "Program", num_real=2, width=4)
    pv = np.arange(100, 287, dtype=np.uint32)
    pv.tofile(trace_dir / "public_values.bin")
    raws["public_values"] = pv
    (root / "gpu_vk.txt").write_text(_VK_TEXT)
    return raws


def _bits(a) -> np.ndarray:
    """Montgomery u32 bitpatterns — the parity comparison unit (dtype-blind,
    so a value-level __eq__ on the field dtype can't mask a bit divergence)."""
    return np.asarray(a.view(jnp.uint32))


class RegionsFromProducerParityTest(absltest.TestCase):
    """The producer entry must assemble bit-identical regions and equal
    shard metadata to the fixture-loader path for the same trace bits —
    the fractalyze/sp1-zorch#200 unit gate."""

    def setUp(self):
        super().setUp()
        root = Path(self.create_tempdir().full_path)
        raws = _write_dump(root)
        pv_raw = raws.pop("public_values")

        self.fixture_shard = load_fixture_shard(root)
        self.f_main, self.f_prep = shard_regions(self.fixture_shard)

        # The producer bundle for the same shard: rw-named uint32 device
        # arrays in sorted rw-name order, carrying riscv-witness's per-zkVM
        # registry prefix exactly as a live bundle does ("sp1_add"), built
        # from the dump writer's raw bits, independent of the loader's
        # arrays — so bit-parity below compares two ingests, not one array
        # with itself.
        producer = {
            "sp1_" + sp1_name_to_rw(name): jnp.array(raw)
            for name, raw in raws.items()
        }
        self.producer_order = tuple(sorted(producer))
        self.producer_chips = {n: producer[n] for n in self.producer_order}
        num_reals = {n: int(a.shape[0]) for n, a in producer.items()}
        self.p_main, self.p_prep, self.p_shard = regions_from_producer(
            self.producer_chips,
            num_reals=num_reals,
            public_values=jnp.array(pv_raw),
            vk=self.fixture_shard.vk,
            preprocessed=self.fixture_shard.preprocessed_traces,
        )

    def _assert_region_parity(self, fixture_region, producer_region):
        self.assertEqual(fixture_region.chip_names, producer_region.chip_names)
        self.assertEqual(fixture_region.chip_starts, producer_region.chip_starts)
        self.assertEqual(fixture_region.row_counts, producer_region.row_counts)
        self.assertEqual(fixture_region.column_counts, producer_region.column_counts)
        self.assertEqual(
            fixture_region.log_stacking_height,
            producer_region.log_stacking_height,
        )
        self.assertEqual(fixture_region.dense.dtype, producer_region.dense.dtype)
        np.testing.assert_array_equal(
            _bits(fixture_region.dense), _bits(producer_region.dense)
        )

    def test_main_region_parity(self):
        self._assert_region_parity(self.f_main, self.p_main)

    def test_prep_region_parity(self):
        self.assertIsNotNone(self.f_prep)
        self.assertIsNotNone(self.p_prep)
        self._assert_region_parity(self.f_prep, self.p_prep)

    def test_public_values_view_parity(self):
        pv = self.p_shard.main_trace_data.public_values
        self.assertEqual(pv.dtype, F)
        np.testing.assert_array_equal(
            _bits(self.fixture_shard.main_trace_data.public_values), _bits(pv)
        )

    def test_chip_order_and_num_reals_parity(self):
        f_traces = self.fixture_shard.main_trace_data.traces
        p_traces = self.p_shard.main_trace_data.traces
        self.assertEqual(f_traces.chip_order, p_traces.chip_order)
        for name in f_traces.chip_order:
            self.assertEqual(
                f_traces.per_chip[name].num_real, p_traces.per_chip[name].num_real
            )

    def test_chip_resolution_parity(self):
        f_chips = self.fixture_shard.main_trace_data.chips
        p_chips = self.p_shard.main_trace_data.chips
        self.assertEqual(set(f_chips), set(p_chips))
        for name in f_chips:
            self.assertEqual(f_chips[name].num_cols, p_chips[name].num_cols)
            self.assertEqual(
                f_chips[name].constraint_names, p_chips[name].constraint_names
            )
        # Add matched the manifest width, so real constraints attached on
        # both paths (a stub-only shard would pass resolution parity
        # vacuously).
        self.assertNotEmpty(p_chips["Add"].constraint_names)

    def test_producer_order_differs_but_maps_to_chip_order(self):
        # Guard that the fixture exercises the rw/SP1 collation flip: the
        # bundle's sorted-rw order maps to a NON-sorted SP1 sequence, and
        # ingest re-sorts it into the fixture path's chip_order.
        mapped = tuple(rw_name_to_sp1(n) for n in self.producer_order)
        order = self.p_shard.main_trace_data.traces.chip_order
        self.assertNotEqual(mapped, order)
        self.assertEqual(tuple(sorted(mapped)), order)


class RegionsFromProducerErrorsTest(absltest.TestCase):
    """Ingest-time failure modes: torn num_reals and unmappable rw names."""

    _VK = MachineVerifyingKey(
        preprocessed_commit=jnp.zeros(8, F),
        pc_start=jnp.zeros(3, F),
        cum_sum_x=jnp.zeros(7, F),
        cum_sum_y=jnp.zeros(7, F),
        enable_untrusted=0,
    )

    def _ingest(self, chips, num_reals):
        return regions_from_producer(
            chips,
            num_reals=num_reals,
            public_values=jnp.zeros(187, dtype=jnp.uint32),
            vk=self._VK,
        )

    def test_num_reals_height_mismatch_fails_at_ingest(self):
        chips = {"add": jnp.ones((4, 3), dtype=jnp.uint32)}
        with self.assertRaisesRegex(ValueError, "num_reals"):
            self._ingest(chips, {"add": 3})

    def test_unknown_rw_chip_name_fails(self):
        chips = {"no_such_chip": jnp.ones((2, 3), dtype=jnp.uint32)}
        with self.assertRaisesRegex(ValueError, "unknown rw chip name"):
            self._ingest(chips, {"no_such_chip": 2})


class RwNameToSp1Test(absltest.TestCase):
    def test_irregular_names_come_from_the_table(self):
        self.assertEqual(rw_name_to_sp1("byte_lookup"), "Byte")
        self.assertEqual(rw_name_to_sp1("utype"), "UType")
        self.assertEqual(rw_name_to_sp1("memory_global_final"), "MemoryGlobalFinalize")

    def test_single_token_names_invert_lower(self):
        self.assertEqual(rw_name_to_sp1("add"), "Add")
        self.assertEqual(rw_name_to_sp1("poseidon2"), "Poseidon2")

    def test_registry_prefix_is_stripped(self):
        self.assertEqual(rw_name_to_sp1("sp1_add"), "Add")
        self.assertEqual(rw_name_to_sp1("sp1_memory_global_final"), "MemoryGlobalFinalize")

    def test_round_trip_through_forward_map(self):
        for rw in ("add", "byte_lookup", "utype", "uint256_mul", "divrem"):
            self.assertEqual(sp1_name_to_rw(rw_name_to_sp1(rw)), rw)

    def test_snake_case_outside_the_table_is_uninvertible(self):
        with self.assertRaisesRegex(ValueError, "unknown rw chip name"):
            rw_name_to_sp1("definitely_not_a_chip")


if __name__ == "__main__":
    absltest.main()
