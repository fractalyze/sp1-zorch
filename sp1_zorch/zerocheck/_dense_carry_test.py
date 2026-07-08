# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Isolated byte-equality of the row-major dense-carry addressing vs the
per-chip blocks it replaces — the crux of sp1-zorch#242, validated before the
engine consumes it. One `num_cols` class: chips of distinct heights stacked
along the row axis of a `[Σ w_i, num_cols]` buffer."""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont
from sp1_zorch.zerocheck._dense_carry import (
    dense_capacity, row_offsets, seg_view, seg_scatter, pad4,
)

BF, EF = koalabear_mont, koalabearx4_mont


def _rand_ef(seed, shape):
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=tuple(shape) + (4,), dtype=np.int64)
    return jnp.array(ints, dtype=BF).view(EF).reshape(shape)


def _u32(a):
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)


def _row_pad(block, w):
    """Zero-extend a `[h, num_cols]` row-major block to `[w, num_cols]` along the
    ROW axis — the in-tree stand-in for zorch's removed `zero_extend` (#412)."""
    return jnp.pad(block, ((0, w - block.shape[0]), (0, 0)))


def _split_rows(block):
    """The row pairing the engine's fold applies to a row-major buffer (axis-0
    stride-2), the dual of `split_pairs`' last-axis split on the old column-major
    buffer."""
    return block[0::2], block[1::2]


class DenseCarryAddressingTest(absltest.TestCase):
    def setUp(self):
        # One class: uniform num_cols, chips of distinct heights (one empty).
        self.num_cols = 3
        self.heights = (5, 0, 3)
        self.widths = tuple(pad4(h) for h in self.heights)  # (8, 0, 4)
        # Per-chip row-major blocks [w_i, num_cols], live prefix then zero tail.
        self.blocks = [
            _row_pad(_rand_ef(i, (h, self.num_cols)), w)
            for i, (h, w) in enumerate(zip(self.heights, self.widths))
        ]
        self.dense = jnp.concatenate(self.blocks, axis=0)  # [Σ w_i, num_cols]
        self.rc = jnp.asarray(self.heights, jnp.int32)
        # Non-empty chips only — a w==0 filler chip has no rows to window.
        self.chips = [(i, w) for i, w in enumerate(self.widths) if w]

    def test_capacity_and_offsets_match_host_prefix_sum(self):
        self.assertEqual(dense_capacity(self.heights), sum(self.widths))
        host = [0]
        for w in self.widths:
            host.append(host[-1] + w)
        got = row_offsets(self.rc)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(host[:-1], np.int32))

    def test_seg_view_matches_per_chip_block(self):
        offs = row_offsets(self.rc)
        for i, w in self.chips:
            got = seg_view(self.dense, offs[i], w)
            np.testing.assert_array_equal(_u32(got), _u32(self.blocks[i]))
            gp0, gp1 = _split_rows(got)
            rp0, rp1 = _split_rows(self.blocks[i])
            np.testing.assert_array_equal(_u32(gp0), _u32(rp0))
            np.testing.assert_array_equal(_u32(gp1), _u32(rp1))

    def test_scatter_roundtrips(self):
        offs = row_offsets(self.rc)
        dense = self.dense
        for i, w in self.chips:
            block = seg_view(dense, offs[i], w) + jnp.ones((), EF)
            dense = seg_scatter(dense, offs[i], block)
        for i, w in self.chips:
            got = seg_view(dense, offs[i], w)
            np.testing.assert_array_equal(_u32(got), _u32(self.blocks[i] + jnp.ones((), EF)))


if __name__ == "__main__":
    absltest.main()
