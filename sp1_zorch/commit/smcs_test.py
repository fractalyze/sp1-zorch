"""SMCS.commit — Merkle root + SP1 domain separator.

The raw Merkle root is anchored to the Plonky3 koalabear16 golden (the vector
from zorch's merkle_test); the domain-separated commitment is checked two ways —
an independent recomputation of SP1's separator formula, and a pinned regression
vector. SP1 byte-match equivalence against the reference prover is the FFI slice.
"""

import jax.numpy as jnp
from zk_dtypes import koalabear_mont as F
from zk_dtypes import koalabearx4_mont as EF

from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams

# Plonky3 koalabear16 root over arange(32).reshape(4, 8) — before the separator.
_PLONKY3_RAW_ROOT_4X8 = jnp.array(
    [1670701318, 437280557, 23464423, 637192971,
     1642004034, 359231982, 157670030, 587973557],
    dtype=F,
)
# commit() over the same matrix: compress([root, sponge([log_height=2, width=8])]).
_SMCS_COMMIT_4X8 = jnp.array(
    [758018295, 1781457694, 199952559, 105804,
     757812367, 897983307, 503747739, 1584629093],
    dtype=F,
)


def _smcs():
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return sponge, comp, SingleMatrixCommitmentScheme(sponge, comp)


def test_commit_applies_sp1_domain_separator():
    sponge, comp, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    committed = smcs.commit(matrix)

    raw_root, _ = MerkleTree(sponge, comp).commit(matrix)
    assert jnp.array_equal(raw_root, _PLONKY3_RAW_ROOT_4X8)

    # Independent recomputation of SP1's separator over the golden root.
    params = sponge.hash(jnp.array([2, 8], dtype=F))  # [log_height, width]
    expected = comp.compress(jnp.stack([raw_root, params]))
    assert jnp.array_equal(committed, expected)
    assert jnp.array_equal(committed, _SMCS_COMMIT_4X8)


def test_commit_shape_and_determinism():
    _, _, smcs = _smcs()
    m = jnp.arange(32, dtype=F).reshape(4, 8)
    c1, c2 = smcs.commit(m), smcs.commit(m)
    assert c1.shape == (8,)
    assert jnp.array_equal(c1, c2)


def test_single_row_commit_log_height_zero():
    _, _, smcs = _smcs()
    # height 1 -> log_height 0; raw root is the lone leaf digest, then bound.
    c = smcs.commit(jnp.arange(8, dtype=F).reshape(1, 8))
    assert c.shape == (8,)


def test_non_power_of_two_height_raises():
    _, _, smcs = _smcs()
    try:
        smcs.commit(jnp.arange(24, dtype=F).reshape(3, 8))  # height 3
        assert False, "expected ValueError for non-power-of-two height"
    except ValueError:
        pass


def test_extension_field_matrix_not_yet_supported():
    # EF commit (base-field reinterpretation of EF rows) is the FFI byte-match
    # slice; until then commit rejects EF up front rather than faulting in the
    # base-field permutation.
    _, _, smcs = _smcs()
    try:
        smcs.commit(jnp.zeros((4, 4), dtype=EF))
        assert False, "expected NotImplementedError for extension-field matrix"
    except NotImplementedError:
        pass


if __name__ == "__main__":
    test_commit_applies_sp1_domain_separator()
    test_commit_shape_and_determinism()
    test_single_row_commit_log_height_zero()
    test_non_power_of_two_height_raises()
    test_extension_field_matrix_not_yet_supported()
    print("ok")
