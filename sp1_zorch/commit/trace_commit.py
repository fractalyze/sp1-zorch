# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are
RS-encoded (zorch ``ReedSolomon``) into a ``[S * blowup, K]`` codeword in
SP1's bit-reversed row order, Merkle-committed via the SMCS, then bound to
the region's row/column structure. Mirrors sp1-hypercube's jagged commit
(the prover half of the basefold commitment SP1 core uses).

``jit=True`` runs the bit-reverse -> Merkle pipeline as one @jit zone: the
eager form keeps every intermediate (~6 GB codeword copies at rsp scale)
live at once and OOMs a 32 GB device, while one fused graph lets XLA
release each buffer after its last use. The RS encode stays eager on BOTH
paths: inside a jit module the zkx GPU NTT emitter returns wrong codewords
at >= 2^21 codeword length (fractalyze/zkx#624), while the standalone
eager op is correct — and the encode's two live buffers are not the
memory cliff. No input is donated — the dense buffer outlives the commit
(the jagged-eval stage reads it again), so there is nothing safe to
donate. Output is byte-identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from sp1_zorch.commit.region import JaggedRegion
from sp1_zorch.commit.smcs import SingleMatrixCommitmentScheme
from zorch.coding.reed_solomon import ReedSolomon


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "dense",
        "mle",
        "codeword",
        "digest_layers",
        "row_counts",
        "column_counts",
        "smcs_commitment",
    ],
    meta_fields=[],
)
@dataclass(frozen=True)
class TraceCommitData:
    """Prover-side retained state for the opening stage.

    The row/column count arrays live here because the structure hash bound
    them; the verifier-side rebind needs the exact device values. ``mle`` and
    ``codeword`` are the stacked open's per-region witness — the ``[S, K]``
    message matrix and the committed ``[S*blowup, K]`` bit-reversed leaves —
    so the opening stage reproves at the eval point without recommitting.
    """

    dense: Array
    mle: Array
    codeword: Array
    digest_layers: list[Array]
    row_counts: Array
    column_counts: Array
    smcs_commitment: Array  # shape-bound SMCS root, before structure binding


def _leaf_commit(
    codeword: Array, *, smcs: SingleMatrixCommitmentScheme
) -> tuple[Array, list[Array], Array]:
    """The codeword-scale commit tail, shared by the eager and @jit paths.

    Only the big-buffer chain lives here; the small tails (structure binding,
    the mle transpose) stay in ``commit_region`` so the zone's outputs are
    exactly the buffers it exists to manage.
    """
    # SP1's codeword layout is bit-reversed (FRI fold pairs adjacent). zkx's
    # fft cannot emit bit-reversed output (its bit-reverse is input-side
    # DIT), so this is a separate permutation today; the zorch-level order
    # convention is settled with the opening machinery
    # (fractalyze/sp1-zorch#20).
    codeword = lax.bit_reverse(codeword, dimensions=(1,))

    # Codeword rows are the Merkle leaves; SMCS binds (log_height, width).
    commitment, digest_layers = smcs.commit(codeword.T)
    return codeword.T, digest_layers, commitment


# ``smcs`` is a static arg keyed by object identity (the scheme defines no
# __eq__/__hash__): every call site must reuse one instance per process, or
# each fresh instance silently recompiles the full poseidon2/Merkle pipeline.
_leaf_commit_jit = jax.jit(_leaf_commit, static_argnames=("smcs",))


def commit_region(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    *,
    log_blowup: int,
    jit: bool = False,
) -> tuple[Array, TraceCommitData]:
    """Commit a packed region; returns ``(bound_commitment, prover_data)``.

    ``jit`` runs the commit tail as one fused graph — required at rsp scale
    on a 32 GB device (see the module docstring). Byte-identical either way.
    """
    S = 1 << region.log_stacking_height
    dense = region.dense
    if dense.shape[0] % S != 0:
        raise ValueError(
            f"dense size {dense.shape[0]} must be a multiple of the stacking "
            f"height {S} (from_chips pads to it)"
        )
    K = dense.shape[0] // S

    # Row k of [K, S] is stacked column k of the dense MLE; RS-encode each.
    # Eager on purpose, never inside the jit zone — fractalyze/zkx#624 (see
    # the module docstring).
    rs = ReedSolomon(message_len=S, blowup=1 << log_blowup, dtype=dense.dtype)
    codeword = rs.encode(dense.reshape(K, S))

    tail = _leaf_commit_jit if jit else _leaf_commit
    codeword_t, digest_layers, commitment = tail(codeword, smcs=smcs)
    row_counts = jnp.array(region.row_counts, dtype=dense.dtype)
    column_counts = jnp.array(region.column_counts, dtype=dense.dtype)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    return bound, TraceCommitData(
        dense=dense,
        # Column k of the [S, K] message matrix is dense block k (the open
        # evaluates each column at the stack point); the leaves are the
        # committed [S*blowup, K] codeword Merkle-bound just above.
        mle=dense.reshape(K, S).T,
        codeword=codeword_t,
        digest_layers=digest_layers,
        row_counts=row_counts,
        column_counts=column_counts,
        smcs_commitment=commitment,
    )
