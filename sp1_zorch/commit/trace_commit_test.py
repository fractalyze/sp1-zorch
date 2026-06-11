# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Trace commit structure: determinism and structure-hash binding. The
byte-match against the SP1 reference dump lives in trace_commit_rsp_test."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.commit.trace_commit import commit_region
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    return SingleMatrixCommitmentScheme(
        Sponge(perm, SpongeParams(rate=8, out=8)),
        Compression(perm, CompressionParams(arity=2, chunk=8)),
    )


def _region(heights=(4, 2)):
    chips = [
        jnp.arange(100 * i, 100 * i + h * 3, dtype=jnp.uint32).reshape(h, 3).view(F)
        for i, h in enumerate(heights)
    ]
    return JaggedRegion.from_chips(chips, log_stacking_height=3, max_log_row_count=4)


class CommitRegionTest(absltest.TestCase):
    def test_commitment_shape_and_determinism(self):
        smcs = _smcs()
        c1, data1 = commit_region(_region(), smcs, log_blowup=2)
        c2, _ = commit_region(_region(), smcs, log_blowup=2)
        self.assertEqual(c1.shape, (8,))
        self.assertEqual(c1.dtype, F)
        self.assertTrue(bool(jnp.all(c1 == c2)))
        # Prover data keeps what the opening stage needs.
        self.assertEqual(data1.dense.shape, _region().dense.shape)
        self.assertNotEmpty(data1.digest_layers)

    def test_jit_matches_eager(self):
        """The @jit zone exists for memory, not semantics — the commitment and
        every retained prover-data leaf must be byte-identical to eager."""
        smcs = _smcs()
        eager = commit_region(_region(), smcs, log_blowup=2)
        jitted = commit_region(_region(), smcs, log_blowup=2, jit=True)
        for le, lj in zip(jax.tree.leaves(eager), jax.tree.leaves(jitted), strict=True):
            np.testing.assert_array_equal(le, lj)

    def test_structure_binding_separates_same_dense(self):
        """Two regions with identical dense bytes but different chip splits
        must commit differently — that's what the structure hash is for."""
        smcs = _smcs()
        flat = jnp.arange(24, dtype=jnp.uint32).view(F)
        a = JaggedRegion.from_chips(
            [flat.reshape(8, 3).view(F)], log_stacking_height=3, max_log_row_count=4
        )
        b = JaggedRegion.from_chips(
            [flat[:12].reshape(4, 3).view(F), flat[12:].reshape(4, 3).view(F)],
            log_stacking_height=3,
            max_log_row_count=4,
        )
        ca, _ = commit_region(a, smcs, log_blowup=2)
        cb, _ = commit_region(b, smcs, log_blowup=2)
        self.assertFalse(bool(jnp.all(ca == cb)))

    def test_unaligned_dense_raises(self):
        # Bypasses from_chips (which pads by construction) to hit the guard.
        bad = JaggedRegion(
            dense=jnp.zeros(10, dtype=F),
            chip_starts=(0, 10),
            row_counts=(2, 8, 2),
            column_counts=(5, 0, 1),
            log_stacking_height=3,
        )
        with self.assertRaises(ValueError):
            commit_region(bad, _smcs(), log_blowup=2)

    def test_blowup_changes_commitment(self):
        smcs = _smcs()
        c2, _ = commit_region(_region(), smcs, log_blowup=2)
        c1, _ = commit_region(_region(), smcs, log_blowup=1)
        self.assertFalse(bool(jnp.all(c1 == c2)))


if __name__ == "__main__":
    absltest.main()
