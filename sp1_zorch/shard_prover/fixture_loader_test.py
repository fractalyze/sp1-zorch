# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Fixture loader: synthetic GPU-dump round-trips and rw-chip attachment."""

from pathlib import Path

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.shard_prover.chip_loader import load_sp1_chips
from sp1_zorch.shard_prover.fixture_loader import (
    check_match,
    load_fixture_shard,
    read_dump,
)

_VK_TEXT = """\
preprocessed_commit=[1, 2, 3, 4, 5, 6, 7, 8]
pc_start=[9, 10, 11]
cum_sum_x=[1, 2, 3, 4, 5, 6, 7]
cum_sum_y=[8, 9, 10, 11, 12, 13, 14]
enable_untrusted=[0]
"""


def _write_chip(trace_dir: Path, name: str, num_real: int, width: int) -> np.ndarray:
    """Write one chip's .meta (+ .bin when non-empty); returns the raw u32s."""
    (trace_dir / f"{name}.meta").write_text(f"num_real={num_real}\nwidth={width}\n")
    raw = np.arange(num_real * width, dtype=np.uint32)
    if num_real > 0:
        raw.tofile(trace_dir / f"{name}.bin")
    return raw


def _write_dump(root: Path, trace_subdir: str = "gpu_traces") -> dict[str, np.ndarray]:
    """Build a minimal two-chip dump with prep traces, PVs, and a VK."""
    trace_dir = root / trace_subdir
    prep_dir = trace_dir / "preprocessed"
    prep_dir.mkdir(parents=True)
    raws = {
        "Add": _write_chip(trace_dir, "Add", num_real=4, width=3),
        "DivRem": _write_chip(trace_dir, "DivRem", num_real=0, width=5),
        "prep/Program": _write_chip(prep_dir, "Program", num_real=2, width=4),
    }
    pv = np.arange(100, 107, dtype=np.uint32)
    pv.tofile(trace_dir / "public_values.bin")
    raws["public_values"] = pv
    (root / "gpu_vk.txt").write_text(_VK_TEXT)
    return raws


class CheckMatchTest(absltest.TestCase):
    def test_equal_arrays_match(self):
        self.assertTrue(check_match("eq", fnp.ones((2,), F), fnp.ones((2,), F)))

    def test_shape_divergence_is_a_mismatch(self):
        # (1,) vs (1, 1) broadcast as all-equal; the harness must still
        # report a mismatch.
        self.assertFalse(
            check_match("shape", fnp.ones((1,), F), fnp.ones((1, 1), F))
        )


class ReadDumpTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = Path(self.create_tempdir().full_path)
        self.raws = _write_dump(self.root)
        self.dump = read_dump(self.root)

    def test_main_trace_is_montgomery_u32_view(self):
        add = self.dump.traces["Add"]
        self.assertEqual(add.shape, (4, 3))
        self.assertEqual(add.dtype, F)
        expected = fnp.array(self.raws["Add"].reshape(4, 3)).view(F)
        self.assertTrue(bool(fnp.all(add == expected)))
        self.assertEqual(self.dump.num_reals, {"Add": 4, "DivRem": 0})

    def test_zero_height_chip_loads_as_empty(self):
        self.assertEqual(self.dump.traces["DivRem"].shape, (0, 5))
        self.assertEqual(self.dump.traces["DivRem"].dtype, F)

    def test_preprocessed_trace_loaded(self):
        prog = self.dump.preprocessed["Program"]
        self.assertEqual(prog.shape, (2, 4))
        expected = fnp.array(self.raws["prep/Program"].reshape(2, 4)).view(F)
        self.assertTrue(bool(fnp.all(prog == expected)))

    def test_public_values(self):
        pv = self.dump.public_values
        self.assertEqual(pv.shape, (7,))
        expected = fnp.array(self.raws["public_values"]).view(F)
        self.assertTrue(bool(fnp.all(pv == expected)))

    def test_vk_canonical_ints(self):
        vk = self.dump.vk
        self.assertTrue(
            bool(
                fnp.all(
                    vk.preprocessed_commit
                    == fnp.arange(1, 9, dtype=fnp.int32).astype(F)
                )
            )
        )
        self.assertEqual(vk.pc_start.shape, (3,))
        self.assertEqual(vk.cum_sum_x.shape, (7,))
        self.assertEqual(vk.cum_sum_y.shape, (7,))
        self.assertEqual(vk.enable_untrusted, 0)

    def test_autodetects_trace_subdir(self):
        alt = Path(self.create_tempdir().full_path)
        _write_dump(alt, trace_subdir="traces")
        dump = read_dump(alt)
        self.assertIn("Add", dump.traces)

    def test_tolerates_whitespace_and_blank_lines(self):
        """Whitespace/blank-line tolerance — but malformed non-empty lines
        must still fail loudly so dump-format drift surfaces."""
        alt = Path(self.create_tempdir().full_path)
        raws = _write_dump(alt)
        (alt / "gpu_traces" / "Add.meta").write_text("num_real = 4\n\nwidth = 3\n")
        (alt / "gpu_vk.txt").write_text(_VK_TEXT.replace("\npc_start", "\n\npc_start"))
        dump = read_dump(alt)
        self.assertEqual(dump.traces["Add"].shape, (4, 3))
        self.assertTrue(
            bool(
                fnp.all(dump.public_values == fnp.array(raws["public_values"]).view(F))
            )
        )
        self.assertEqual(dump.vk.pc_start.shape, (3,))

    def test_malformed_vk_line_fails_loudly(self):
        alt = Path(self.create_tempdir().full_path)
        _write_dump(alt)
        (alt / "gpu_vk.txt").write_text(_VK_TEXT + "not a key value line\n")
        with self.assertRaises(ValueError):
            read_dump(alt)


class LoadFixtureShardTest(absltest.TestCase):
    """Chips with a manifest-matching width get rw constraints; the rest
    fall back to constraint-less stubs so the shard keeps its chip set."""

    def _shard_with_add_width(self, width: int):
        root = Path(self.create_tempdir().full_path)
        trace_dir = root / "gpu_traces"
        trace_dir.mkdir(parents=True)
        _write_chip(trace_dir, "Add", num_real=2, width=width)
        _write_chip(trace_dir, "Zzz", num_real=1, width=4)
        np.zeros(7, dtype=np.uint32).tofile(trace_dir / "public_values.bin")
        (root / "gpu_vk.txt").write_text(_VK_TEXT)
        return load_fixture_shard(root)

    def test_matching_width_attaches_rw_constraints(self):
        manifest_width = load_sp1_chips(chip_names=["add"])["add"].num_cols
        shard = self._shard_with_add_width(manifest_width)
        add = shard.main_trace_data.chips["Add"]
        self.assertNotEmpty(add.constraint_names)
        self.assertEqual(add.num_cols, manifest_width)

    def test_width_mismatch_falls_back_to_stub(self):
        manifest_width = load_sp1_chips(chip_names=["add"])["add"].num_cols
        shard = self._shard_with_add_width(manifest_width + 1)
        add = shard.main_trace_data.chips["Add"]
        self.assertEmpty(add.constraint_names)
        self.assertEqual(add.num_cols, manifest_width + 1)

    def test_unknown_chip_falls_back_to_stub(self):
        manifest_width = load_sp1_chips(chip_names=["add"])["add"].num_cols
        shard = self._shard_with_add_width(manifest_width)
        zzz = shard.main_trace_data.chips["Zzz"]
        self.assertEmpty(zzz.constraint_names)
        self.assertEqual(zzz.num_cols, 4)

    def test_shard_carries_vk_prep_and_ordered_traces(self):
        manifest_width = load_sp1_chips(chip_names=["add"])["add"].num_cols
        shard = self._shard_with_add_width(manifest_width)
        self.assertEqual(shard.vk.enable_untrusted, 0)
        self.assertEqual(shard.preprocessed_traces, {})
        traces = shard.main_trace_data.traces
        self.assertEqual(traces.chip_order, ("Add", "Zzz"))
        self.assertEqual(traces.per_chip["Add"].num_real, 2)


if __name__ == "__main__":
    absltest.main()
