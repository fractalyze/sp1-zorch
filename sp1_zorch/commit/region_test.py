# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""JaggedRegion packing: hand-computed goldens for SP1's structure-count
convention (per-chip entries + trailing ``(max_height, leftover)`` /
``(num_added_cols - 1, 1)`` pad pair)."""

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.region import JaggedRegion


def _chip(h: int, w: int, start: int = 0):
    return jnp.arange(start, start + h * w, dtype=jnp.uint32).reshape(h, w).view(F)


class JaggedRegionTest(absltest.TestCase):
    """log_stacking_height=4 (S=16), max_log_row_count=5 (max_height=32)."""

    def _region(self):
        return JaggedRegion.from_chips(
            [_chip(4, 3), _chip(2, 5, start=100)],
            log_stacking_height=4,
            max_log_row_count=5,
            chip_names=("a", "b"),
        )

    def test_structure_counts_encode_trailing_pad(self):
        region = self._region()
        # total_area 22 -> aligned 32 -> 10 pad vals -> 1 pad col, leftover 10.
        self.assertEqual(region.row_counts, (4, 2, 32, 10))
        self.assertEqual(region.column_counts, (3, 5, 0, 1))
        self.assertEqual(region.num_chips, 2)
        self.assertEqual(region.chip_heights, (4, 2))
        self.assertEqual(region.chip_widths, (3, 5))
        self.assertEqual(region.chip_starts, (0, 12, 22))
        self.assertEqual(region.raw_size, 22)

    def test_dense_is_column_major_per_chip_with_zero_pad(self):
        region = self._region()
        self.assertEqual(region.dense.shape, (32,))
        want = jnp.concatenate(
            [
                _chip(4, 3).T.reshape(-1),
                _chip(2, 5, start=100).T.reshape(-1),
                jnp.zeros(10, dtype=F),
            ]
        )
        self.assertTrue(bool(jnp.all(region.dense == want)))

    def test_zero_height_chip_kept_in_counts_not_dense(self):
        region = JaggedRegion.from_chips(
            [_chip(4, 3), _chip(0, 7), _chip(2, 5)],
            log_stacking_height=4,
            max_log_row_count=5,
        )
        self.assertEqual(region.chip_heights, (4, 0, 2))
        self.assertEqual(region.chip_widths, (3, 7, 5))
        self.assertEqual(region.dense.shape, (32,))  # area unchanged by empty chip
        self.assertEqual(region.chip_starts, (0, 12, 12, 22))

    def test_pad_spans_multiple_columns(self):
        # One 1x1 chip, S=16, max_height=4: 15 pad vals -> 4 pad cols
        # (3 full of height 4, leftover 3).
        region = JaggedRegion.from_chips(
            [_chip(1, 1)], log_stacking_height=4, max_log_row_count=2
        )
        self.assertEqual(region.row_counts, (1, 4, 3))
        self.assertEqual(region.column_counts, (1, 3, 1))

    def test_minimum_one_full_stack(self):
        # Area 1 with S=16 pads to one full stack of 16.
        region = JaggedRegion.from_chips(
            [_chip(1, 1)], log_stacking_height=4, max_log_row_count=5
        )
        self.assertEqual(region.dense.shape, (16,))

    def test_overheight_chip_raises(self):
        with self.assertRaises(ValueError):
            JaggedRegion.from_chips(
                [_chip(8, 1)], log_stacking_height=4, max_log_row_count=2
            )

    def test_non_2d_chip_raises(self):
        for shape in ((2, 3, 4), (6,)):
            with self.assertRaisesRegex(ValueError, "2-D"):
                JaggedRegion.from_chips(
                    [jnp.zeros(shape, dtype=F)],
                    log_stacking_height=4,
                    max_log_row_count=5,
                )

    def test_mixed_chip_dtypes_raise(self):
        # uint32 x int32 would silently promote through the concat; the
        # commitment preimage must never change dtype under the packer.
        with self.assertRaisesRegex(ValueError, "dtype"):
            JaggedRegion.from_chips(
                [
                    jnp.zeros((2, 3), dtype=jnp.uint32),
                    jnp.zeros((2, 3), dtype=jnp.int32),
                ],
                log_stacking_height=4,
                max_log_row_count=5,
            )

    def test_empty_chip_list_raises(self):
        with self.assertRaises(ValueError):
            JaggedRegion.from_chips([], log_stacking_height=4, max_log_row_count=5)


if __name__ == "__main__":
    absltest.main()
