"""SP1 single-matrix commitment (SMCS) over zorch's Merkle blocks.

Semantically SP1's ``CudaTcsProver::commit_tensors`` for one matrix of
power-of-two height: hash each row to a leaf, fold sibling pairs to a Merkle
root (zorch's ``MerkleTree`` over ``Sponge`` + ``Compression``), then apply
SP1's domain separator (the ``single_layer.rs`` convention) binding the matrix
shape into the root:

    commit = compress([merkle_root, sponge([log_height, width])])

The domain separator is SP1-specific and lives here, not in zorch — zorch's
``MerkleTree`` is deliberately scheme-agnostic and adds no separator. Open/verify
and the SP1 heap proof layout land in the next slice.
"""

from __future__ import annotations

import jax.numpy as jnp
import zk_dtypes
from jax import Array

from zorch.commit.merkle import MerkleTree
from zorch.hash.compression import Compression
from zorch.hash.sponge import Sponge
from zorch.utils.bits import log2_strict_usize


def _is_extension_field(dtype) -> bool:
    """True if ``dtype`` is an extension field (``efinfo`` resolves it)."""
    try:
        zk_dtypes.efinfo(dtype)
        return True
    except ValueError:
        return False


class SingleMatrixCommitmentScheme:
    """SP1's single-matrix commitment, built on zorch's agnostic Merkle blocks.

    Holds the leaf ``Sponge`` and the 2-to-1 ``Compression`` (both also drive the
    internal ``MerkleTree``); ``digest_elems`` is the compressor chunk size.
    """

    def __init__(self, sponge: Sponge, compressor: Compression):
        # Keep sponge + compressor too: the domain separator calls them directly
        # and zorch's MerkleTree does not expose its internals.
        self._tree = MerkleTree(sponge, compressor)
        self._sponge = sponge
        self._compressor = compressor
        self.digest_elems = compressor.chunk

    def commit(self, matrix: Array) -> Array:
        """Commit a base-field ``(height, width)`` matrix (power-of-two height).

        Returns the ``(digest_elems,)`` commitment with SP1's domain separator
        applied.

        Extension-field matrices are rejected for now: SP1 commits each EF row as
        ``width * degree`` base-field elements, and that reinterpretation isn't
        wired through zorch's blocks yet (it's the FFI byte-match slice). Without
        the guard an EF matrix faults in the base-field permutation.
        """
        if _is_extension_field(matrix.dtype):
            raise NotImplementedError(
                "extension-field matrices are not yet supported; pass a base-field "
                "matrix (EF commit is the FFI byte-match slice, fractalyze/zorch#37)"
            )
        raw_root, _ = self._tree.commit(matrix)
        return self._bind_root(raw_root, matrix)

    def _bind_root(self, raw_root: Array, matrix: Array) -> Array:
        height, width = matrix.shape
        log_height = log2_strict_usize(height)  # power-of-two enforced by commit
        # Base-field matrix (commit guards EF), so width is the base-field width.
        params = self._sponge.hash(jnp.array([log_height, width], dtype=matrix.dtype))
        return self._compressor.compress(jnp.stack([raw_root, params]))
