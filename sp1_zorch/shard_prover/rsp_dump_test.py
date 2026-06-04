# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""rsp eth-block dump integration: loader output matches the dump's own
metadata. Manual — the capture (~1.5 GB/shard) lives outside the repo, so
the shard directory comes from ``SP1_ZORCH_RSP_SHARD``:

    bazel test //sp1_zorch/shard_prover:rsp_dump_test \\
        --test_env=SP1_ZORCH_RSP_SHARD=/path/to/rsp_dump/shardN
"""

import os
from pathlib import Path

from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.shard_prover.fixture_loader import load_fixture_shard

_ENV = "SP1_ZORCH_RSP_SHARD"


def _rsp_shard_dir() -> Path:
    """Requiring the env var keeps machine-local dump paths out of the tree;
    a missing value is an error, not a skip — this target only runs when
    asked for explicitly (``manual``), so silence would hide a typo."""
    value = os.environ.get(_ENV)
    if not value:
        raise RuntimeError(f"{_ENV} must point at an rsp shard dump directory")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"{_ENV}={value} is not a directory")
    return path


class RspDumpTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.shard_dir = _rsp_shard_dir()
        cls.shard = load_fixture_shard(cls.shard_dir)

    def test_chip_set_matches_dump_metadata(self):
        metas = {p.stem for p in (self.shard_dir / "gpu_traces").glob("*.meta")}
        self.assertEqual(set(self.shard.main_trace_data.traces.chip_order), metas)

    def test_trace_dims_match_each_meta(self):
        traces = self.shard.main_trace_data.traces
        for name in traces.chip_order:
            meta_path = self.shard_dir / "gpu_traces" / f"{name}.meta"
            meta = dict(
                line.split("=") for line in meta_path.read_text().strip().split("\n")
            )
            chip = traces.per_chip[name]
            self.assertEqual(
                chip.array.shape,
                (int(meta["num_real"]), int(meta["width"])),
                msg=name,
            )
            self.assertEqual(chip.num_real, int(meta["num_real"]), msg=name)
            self.assertEqual(chip.array.dtype, F, msg=name)

    def test_public_values_are_sp1_max_num_pvs(self):
        self.assertEqual(self.shard.main_trace_data.public_values.shape, (187,))

    def test_vk_and_prep_loaded(self):
        self.assertEqual(self.shard.vk.preprocessed_commit.shape, (8,))
        self.assertNotEmpty(self.shard.preprocessed_traces)

    def test_every_chip_resolves_constraints_or_stub(self):
        chips = self.shard.main_trace_data.chips
        with_constraints = [n for n, c in chips.items() if c.constraint_names]
        # A real shard must attach actual AIR constraints for the bulk of
        # its chips — an all-stub shard means the name/width mapping broke.
        self.assertGreater(len(with_constraints), len(chips) // 2)


if __name__ == "__main__":
    absltest.main()
