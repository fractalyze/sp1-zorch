"""SMCS GPU conformance vs the SP1 reference kernel (Mont u32, exact).

Two layers of guarantee, both skip-gated on CUDA + ``SP1_JAX_FFI_LIB``:

1. **GPU jit == eager** for the full commit (tree layers, commitment) plus the
   open/verify roundtrip. The jit path runs zkx's ``zorch.merkle_commit``
   expander (SP1-style ``poseidon2_sponge_hash`` / ``poseidon2_merkle_compress``
   kernels); eager runs the plain block lowering that the CPU goldens anchor.
   This is the regression gate for the expander — it catches multi-chunk
   absorb bugs (leaf width > sponge rate) and constant-folding bugs that the
   width-8 CPU goldens cannot see.
2. **Heap/path arithmetic vs SP1's actual kernel output**:
   ``prove_openings_at_indices`` walked over the digest heap produced by SP1's
   ``sp1_zkx_merkle_commit_kb31`` byte-matches an independent derivation of
   SP1's sibling walk (sibling of node j is ``((j-1) ^ 1) + 1``). This holds
   regardless of hash constants — it pins our heap layout + gather to the
   reference kernel's.

A full digest-tree byte-match against the vendored FFI is deliberately absent:
``libsp1_gpu_jax_ffi`` implements SP1's *legacy* Poseidon2 (powers-of-two
internal diagonal, R⁻¹-twisted internal diffusion), while sp1-zorch's params
are the honest Plonky3 koalabear16 — different permutations by design. The
live byte-match is gated on an honest SP1 FFI rebuild, tracked on
fractalyze/sp1-zorch#2.
"""

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import ShapeDtypeStruct
from zk_dtypes import koalabear_mont as F

from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme, VerifyCode
from sp1_zorch.poseidon2.koalabear16 import koalabear16_params
from zorch.hash.compression import Compression, CompressionParams
from zorch.hash.poseidon2.poseidon2 import Poseidon2
from zorch.hash.sponge import Sponge, SpongeParams
from zorch.utils.bits import log2_strict_usize

# KoalaBear Poseidon2 digest width; must match DIGEST_WIDTH in the FFI kernel.
_DIGEST_WIDTH = 8
_FFI_TARGET = "sp1_zkx_merkle_commit_kb31"

_ffi_registered = False


def _cuda_available() -> bool:
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:
        return False


def setUpModule() -> None:
    if "SP1_JAX_FFI_LIB" not in os.environ:
        raise absltest.SkipTest(
            "SP1_JAX_FFI_LIB not set; the SP1 FFI library is not vendored here"
        )
    if not _cuda_available():
        raise absltest.SkipTest("no CUDA device")


def _sp1_merkle_commit(matrix: jax.Array) -> jax.Array:
    """SP1's reference commit: ``(height, width)`` Mont matrix -> ``(2N-1, 8)``
    u32 digest heap (root at row 0, leaf ``m`` at row ``2^H - 1 + m``)."""
    global _ffi_registered
    if not _ffi_registered:
        lib = ctypes.CDLL(os.environ["SP1_JAX_FFI_LIB"])
        # Platform must be "CUDA" (uppercase): the lowercase key lands in a
        # deferred queue the zkx plugin never drains.
        jax.ffi.register_ffi_target(
            _FFI_TARGET, jax.ffi.pycapsule(getattr(lib, _FFI_TARGET)), platform="CUDA"
        )
        _ffi_registered = True
    height = log2_strict_usize(matrix.shape[0])
    # The kernel reads ``width`` consecutive u32s per leaf in column-major
    # trace storage: hand it the transpose made contiguous.
    col_major = jnp.array(matrix.view(jnp.uint32).T)
    count = ((1 << (height + 1)) - 1) * _DIGEST_WIDTH
    flat = jax.ffi.ffi_call(_FFI_TARGET, ShapeDtypeStruct((count,), jnp.uint32))(
        col_major, height=height
    )
    return flat.reshape(-1, _DIGEST_WIDTH)


def _smcs() -> SingleMatrixCommitmentScheme:
    perm = Poseidon2(koalabear16_params())
    sponge = Sponge(perm, SpongeParams(rate=8, out=8))
    comp = Compression(perm, CompressionParams(arity=2, chunk=8))
    return SingleMatrixCommitmentScheme(sponge, comp)


def _matrix(height: int, width: int) -> jax.Array:
    """Deterministic distinct field elements, well below the modulus."""
    return jnp.arange(height * width, dtype=F).reshape(height, width)


def _u32(a: jax.Array) -> np.ndarray:
    return np.asarray(jnp.asarray(a).view(jnp.uint32))


class SmcsGpuConformanceTest(absltest.TestCase):
    def _assert_jit_commit_matches_eager(self, height: int, width: int) -> None:
        smcs = _smcs()
        matrix = _matrix(height, width)
        commitment_j, layers_j = jax.jit(smcs.commit)(matrix)
        commitment_e, layers_e = smcs.commit(matrix)
        self.assertTrue(
            np.array_equal(_u32(commitment_j), _u32(commitment_e)),
            f"jit commitment diverges from eager for {height}x{width}: "
            f"jit={_u32(commitment_j).tolist()} eager={_u32(commitment_e).tolist()}",
        )
        for lvl, (lj, le) in enumerate(zip(layers_j, layers_e)):
            self.assertTrue(
                np.array_equal(_u32(lj), _u32(le)),
                f"jit digest layer {lvl} diverges from eager for {height}x{width}",
            )

    def test_jit_commit_matches_eager_multi_chunk_leaf(self) -> None:
        # width 16 > rate 8: two absorb chunks per leaf — the case the
        # width-8 CPU goldens cannot exercise.
        self._assert_jit_commit_matches_eager(16, 16)

    def test_jit_commit_matches_eager_single_chunk_leaf(self) -> None:
        self._assert_jit_commit_matches_eager(256, 8)

    def test_verify_roundtrip_on_gpu(self) -> None:
        height, width = 16, 16
        smcs = _smcs()
        matrix = _matrix(height, width)
        commitment, layers = jax.jit(smcs.commit)(matrix)
        indices = jnp.array([2, 9], dtype=jnp.int32)
        rows, proofs = smcs.open_batch(indices, matrix, layers)
        for q in range(indices.shape[0]):
            proof = [level[q] for level in proofs]
            code = smcs.verify_batch(
                commitment, (height, width), int(indices[q]), rows[q], proof
            )
            self.assertEqual(int(code), VerifyCode.OK)

        bad_row = rows[0].at[0].add(1)
        code = smcs.verify_batch(
            commitment,
            (height, width),
            int(indices[0]),
            bad_row,
            [level[0] for level in proofs],
        )
        self.assertEqual(int(code), VerifyCode.ROOT_MISMATCH)

    def test_prove_openings_match_sp1_reference_heap_walk(self) -> None:
        # The digest heap comes from SP1's OWN kernel, so this pins our heap
        # indexing + path gather to the reference layout — independent of the
        # hash constants (see the module docstring).
        height, width = 256, 16
        matrix = _matrix(height, width)
        ref = np.asarray(_sp1_merkle_commit(matrix))  # (2N-1, 8) u32, heap order
        tree_height = log2_strict_usize(height)
        indices = jnp.array([0, 1, 5, height - 1], dtype=jnp.int32)
        paths = _smcs().prove_openings_at_indices(
            jnp.asarray(ref).view(F), indices, tree_height
        )

        got = _u32(paths)  # (Q, tree_height, 8)
        for q, leaf in enumerate(np.asarray(indices)):
            node = (1 << tree_height) - 1 + int(leaf)
            for level in range(tree_height):
                sibling = ((node - 1) ^ 1) + 1
                self.assertTrue(
                    np.array_equal(got[q, level], ref[sibling]),
                    f"query {q} level {level}: sibling diverges from SP1's "
                    f"heap walk (node {node}, sibling {sibling})",
                )
                node = (node - 1) >> 1


if __name__ == "__main__":
    absltest.main()
