"""Bootstrap wiring proof: sp1-zorch consumes zorch's engine with its own params.

This is the first slice's verification — it builds zorch's Sponge / Compression /
MerkleTree over zorch's agnostic Poseidon2 engine, parameterized by sp1-zorch's
OWN koalabear16 instance (named params are a consumer concern, not zorch API),
and reproduces the SP1-form koalabear16 Merkle root over a 4x8 matrix. Passing
under `bazel test` proves the cross-module dependency (sp1_zorch -> zorch), the
pinned toolchain, and that sp1-zorch's params drive zorch's engine correctly.
The SP1-specific SMCS that layers on top lands in the next slice.
"""

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import koalabear_mont as F

from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

# SP1-form golden (vendored-kernel parameterization):
# PaddingFreeSponge<_,16,8,8> leaves + TruncatedPermutation<_,2,8,16> over
# arange(32) reshaped to 4x8. Matches zorch/commit/merkle_test.py.
_SP1_MERKLE_ROOT_4X8 = jnp.array(
    [
        709053809,
        548310247,
        1460186906,
        135994348,
        1863522735,
        953629012,
        708601688,
        648442714,
    ],
    dtype=F,
)


def _kb16_tree():
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return MerkleTree(sponge, comp)


class ZorchWiringTest(absltest.TestCase):
    def test_zorch_merkle_commit_reachable_and_correct(self) -> None:
        tree = _kb16_tree()
        raw_root, _ = tree.commit(jnp.arange(32, dtype=F).reshape(4, 8))
        self.assertTrue(bool(jnp.array_equal(raw_root, _SP1_MERKLE_ROOT_4X8)))


if __name__ == "__main__":
    absltest.main()
