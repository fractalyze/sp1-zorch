# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Isolated byte-equality of the tight dense-carry addressing vs the per-chip
buffers it replaces — the crux of sp1-zorch#242, validated before the engine
consumes it."""
from __future__ import annotations
import jax, jax.numpy as jnp, numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont, koalabearx4_mont
from zorch.sumcheck.prover import split_pairs, zero_extend
from sp1_zorch.zerocheck._dense_carry import (
    dense_capacity, round0_offsets, seg_gather, seg_scatter, pad4,
)

BF, EF = koalabear_mont, koalabearx4_mont

def _rand_ef(seed, shape):
    ints = np.random.default_rng(seed).integers(1, 1 << 30, size=tuple(shape) + (4,), dtype=np.int64)
    return jnp.array(ints, dtype=BF).view(EF).reshape(shape)

def _u32(a):
    return np.asarray(jax.lax.bitcast_convert_type(a, jnp.uint32)).reshape(-1)

class DenseCarryAddressingTest(absltest.TestCase):
    def setUp(self):
        self.num_cols = (2, 3, 1)
        self.heights = (5, 0, 3)
        self.widths = tuple(pad4(h) for h in self.heights)  # (8, 0, 4)
        self.traces = [_rand_ef(i, (nc, h)) for i, (nc, h) in enumerate(zip(self.num_cols, self.heights))]
        self.bufs = [zero_extend(t, w).astype(EF) for t, w in zip(self.traces, self.widths)]
        self.dense = jnp.concatenate([b.reshape(-1) for b in self.bufs])
        self.rc = jnp.asarray(self.heights, jnp.int32)
        # Non-empty chips only — the w==0 filler chip has no segment to gather.
        self.chips = [
            (i, nc, w)
            for i, (nc, w) in enumerate(zip(self.num_cols, self.widths))
            if w
        ]

    def test_capacity_and_offsets_match_host_prefix_sum(self):
        self.assertEqual(dense_capacity(self.num_cols, self.heights),
                         sum(nc * w for nc, w in zip(self.num_cols, self.widths)))
        host = [0]
        for nc, w in zip(self.num_cols, self.widths):
            host.append(host[-1] + nc * w)
        got = round0_offsets(self.num_cols, self.rc)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(host[:-1], np.int32))

    def test_seg_gather_matches_per_chip_buffer(self):
        offs = round0_offsets(self.num_cols, self.rc)
        for i, nc, w in self.chips:
            got = seg_gather(self.dense, offs[i], nc, w)
            np.testing.assert_array_equal(_u32(got), _u32(self.bufs[i]))
            gp0, gp1 = split_pairs(got)
            rp0, rp1 = split_pairs(self.bufs[i])
            np.testing.assert_array_equal(_u32(gp0), _u32(rp0))
            np.testing.assert_array_equal(_u32(gp1), _u32(rp1))

    def test_scatter_roundtrips(self):
        offs = round0_offsets(self.num_cols, self.rc)
        dense = self.dense
        for i, nc, w in self.chips:
            block = seg_gather(dense, offs[i], nc, w) + jnp.ones((), EF)
            dense = seg_scatter(dense, offs[i], block)
        for i, nc, w in self.chips:
            got = seg_gather(dense, offs[i], nc, w)
            np.testing.assert_array_equal(_u32(got), _u32(self.bufs[i] + jnp.ones((), EF)))

if __name__ == "__main__":
    absltest.main()
