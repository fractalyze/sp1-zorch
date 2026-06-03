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
from zorch.hash.poseidon2.poseidon2 import Poseidon2
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
    committed, _ = smcs.commit(matrix)

    raw_root, _ = MerkleTree(sponge, comp).commit(matrix)
    assert jnp.array_equal(raw_root, _PLONKY3_RAW_ROOT_4X8)

    # Independent recomputation of SP1's separator over the golden root.
    params = sponge.hash(jnp.array([2, 8], dtype=F))  # [log_height, width]
    expected = comp.compress(jnp.stack([raw_root, params]))
    assert jnp.array_equal(committed, expected)
    assert jnp.array_equal(committed, _SMCS_COMMIT_4X8)


def test_commit_returns_commitment_and_prover_layers():
    # commit hands back the layered digest tree (zorch's stateless prover data)
    # alongside the commitment, so open_batch can produce sibling paths without
    # recommitting. Layers run leaf-digests -> ... -> root.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4 -> 3 layers
    committed, digest_layers = smcs.commit(matrix)
    assert committed.shape == (8,)
    assert [layer.shape for layer in digest_layers] == [(4, 8), (2, 8), (1, 8)]
    assert jnp.array_equal(committed, _SMCS_COMMIT_4X8)


def test_open_batch_returns_rows_and_sibling_path():
    # open_batch returns the queried rows plus, per Merkle layer, the sibling
    # digest of each query's node — log_height layers, batched over the queries.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # height 4 -> log_height 2
    _, layers = smcs.commit(matrix)
    indices = jnp.array([0, 3])

    rows, proofs = smcs.open_batch(indices, matrix, layers)

    assert rows.shape == (2, 8)
    assert jnp.array_equal(rows[0], matrix[0])
    assert jnp.array_equal(rows[1], matrix[3])

    assert len(proofs) == 2  # log_height
    # Level 0 sibling is the other leaf digest; level 1 sibling is the opposite
    # subtree root. Selected by flipping the low bit of the per-level index.
    assert jnp.array_equal(proofs[0], layers[0][indices ^ 1])
    assert jnp.array_equal(proofs[1], layers[1][(indices >> 1) ^ 1])


def _proof_for(proofs, q):
    """Slice the per-level batched siblings down to query ``q``'s single path."""
    return [proofs[level][q] for level in range(len(proofs))]


def test_open_verify_roundtrip_ok_for_every_index():
    # Opening any row and folding its sibling path back up must reconstruct the
    # bound commitment exactly -> VERIFY_OK. Structural, no golden needed.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    commitment, layers = smcs.commit(matrix)
    indices = jnp.array([0, 1, 2, 3])
    rows, proofs = smcs.open_batch(indices, matrix, layers)
    for q, idx in enumerate(range(4)):
        code = smcs.verify_batch(commitment, (4, 8), idx, rows[q], _proof_for(proofs, q))
        assert int(code) == SingleMatrixCommitmentScheme.VERIFY_OK


def test_verify_rejects_tampered_row():
    # A corrupted opened row re-hashes to a different leaf -> the rebound root
    # cannot match the commitment.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    commitment, layers = smcs.commit(matrix)
    rows, proofs = smcs.open_batch(jnp.array([1]), matrix, layers)
    tampered = rows[0] + jnp.ones(8, dtype=F)
    code = smcs.verify_batch(commitment, (4, 8), 1, tampered, _proof_for(proofs, 0))
    assert int(code) == SingleMatrixCommitmentScheme.VERIFY_ERR_ROOT_MISMATCH


def test_verify_rejects_index_out_of_bounds():
    # index >= height is caught even with a correctly-sized proof.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)
    commitment, layers = smcs.commit(matrix)
    rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
    code = smcs.verify_batch(commitment, (4, 8), 4, rows[0], _proof_for(proofs, 0))
    assert int(code) == SingleMatrixCommitmentScheme.VERIFY_ERR_INDEX_OUT_OF_BOUNDS


def test_verify_rejects_wrong_proof_length():
    # A proof whose length != log_height cannot authenticate any leaf.
    _, _, smcs = _smcs()
    matrix = jnp.arange(32, dtype=F).reshape(4, 8)  # log_height 2
    commitment, layers = smcs.commit(matrix)
    rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
    short_proof = _proof_for(proofs, 0)[:1]  # 1 sibling, expected 2
    code = smcs.verify_batch(commitment, (4, 8), 0, rows[0], short_proof)
    assert int(code) == SingleMatrixCommitmentScheme.VERIFY_ERR_WRONG_HEIGHT


def test_heap_digests_layout():
    # The layered tree flattens to SP1's heap buffer: root at index 0, then each
    # level top-down, leaves last. For 2^H leaves the buffer has 2^(H+1)-1 nodes.
    _, _, smcs = _smcs()
    matrix = jnp.arange(64, dtype=F).reshape(8, 8)  # height 8 -> 15-node heap
    _, layers = smcs.commit(matrix)
    heap = smcs.heap_digests(layers)
    assert heap.shape == (15, 8)
    assert jnp.array_equal(heap[0], layers[-1][0])  # root at index 0
    assert jnp.array_equal(heap[7:], layers[0])  # leaves at indices 2^3-1 .. end


def test_prove_openings_matches_open_batch_siblings():
    # The heap-indexed path kernel must select exactly the siblings that the
    # layered open_batch reads — same authentication path, two representations.
    _, _, smcs = _smcs()
    matrix = jnp.arange(64, dtype=F).reshape(8, 8)  # height 8, tree_height 3
    _, layers = smcs.commit(matrix)
    heap = smcs.heap_digests(layers)
    indices = jnp.array([0, 3, 5])

    paths = smcs.prove_openings_at_indices(heap, indices, 3)
    assert paths.shape == (3, 3, 8)  # (Q, tree_height, digest_elems)

    _, proofs = smcs.open_batch(indices, matrix, layers)
    for level in range(3):
        assert jnp.array_equal(paths[:, level, :], proofs[level])


def test_open_verify_single_row_roundtrip():
    # height 1 boundary: log_height 0 -> empty proof; verify folds nothing and
    # just rebinds the lone leaf digest.
    _, _, smcs = _smcs()
    matrix = jnp.arange(8, dtype=F).reshape(1, 8)
    commitment, layers = smcs.commit(matrix)
    rows, proofs = smcs.open_batch(jnp.array([0]), matrix, layers)
    assert proofs == []
    code = smcs.verify_batch(commitment, (1, 8), 0, rows[0], _proof_for(proofs, 0))
    assert int(code) == SingleMatrixCommitmentScheme.VERIFY_OK


def test_commit_shape_and_determinism():
    _, _, smcs = _smcs()
    m = jnp.arange(32, dtype=F).reshape(4, 8)
    (c1, _), (c2, _) = smcs.commit(m), smcs.commit(m)
    assert c1.shape == (8,)
    assert jnp.array_equal(c1, c2)


def test_single_row_commit_log_height_zero():
    _, _, smcs = _smcs()
    # height 1 -> log_height 0; raw root is the lone leaf digest, then bound.
    c, _ = smcs.commit(jnp.arange(8, dtype=F).reshape(1, 8))
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
    test_commit_returns_commitment_and_prover_layers()
    test_open_batch_returns_rows_and_sibling_path()
    test_open_verify_roundtrip_ok_for_every_index()
    test_verify_rejects_tampered_row()
    test_verify_rejects_index_out_of_bounds()
    test_verify_rejects_wrong_proof_length()
    test_heap_digests_layout()
    test_prove_openings_matches_open_batch_siblings()
    test_open_verify_single_row_roundtrip()
    test_commit_shape_and_determinism()
    test_single_row_commit_log_height_zero()
    test_non_power_of_two_height_raises()
    test_extension_field_matrix_not_yet_supported()
    print("ok")
