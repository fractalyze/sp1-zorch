# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 trace commit: stacked RS-encode of a jagged region + SMCS commit.

The dense buffer becomes one ``[S, K]`` stacked MLE whose columns are
RS-encoded (zorch ``ReedSolomon``) into a ``[S * blowup, K]`` codeword in
SP1's bit-reversed row order, Merkle-committed via the SMCS, then bound to
the region's row/column structure. Mirrors sp1-hypercube's jagged commit
(the prover half of the basefold commitment SP1 core uses).

The pipeline runs eagerly for now: a single donated @jit zone is what the
rsp-scale main region needs to fit on a 32 GB device (the eager form keeps
multiple ~6 GB codeword copies live), but under the pinned jax the zorch
composites compile through the slow no-CompositeOp fallback and one fused
pipeline graph blows past test timeouts. The jit zone lands when the jax
pin picks up CompositeOp (jax#164) — tracked on fractalyze/sp1-zorch#17.
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


def commit_region(
    region: JaggedRegion,
    smcs: SingleMatrixCommitmentScheme,
    *,
    log_blowup: int,
) -> tuple[Array, TraceCommitData]:
    """Commit a packed region; returns ``(bound_commitment, prover_data)``."""
    S = 1 << region.log_stacking_height
    dense = region.dense
    if dense.shape[0] % S != 0:
        raise ValueError(
            f"dense size {dense.shape[0]} must be a multiple of the stacking "
            f"height {S} (from_chips pads to it)"
        )
    K = dense.shape[0] // S

    # Row k of [K, S] is stacked column k of the dense MLE; RS-encode each.
    rs = ReedSolomon(message_len=S, blowup=1 << log_blowup, dtype=dense.dtype)
    codeword = rs.encode(dense.reshape(K, S))
    # SP1's codeword layout is bit-reversed (FRI fold pairs adjacent). zkx's
    # fft cannot emit bit-reversed output (its bit-reverse is input-side
    # DIT), so this is a separate permutation today; the zorch-level order
    # convention is settled with the opening machinery
    # (fractalyze/sp1-zorch#20).
    codeword = lax.bit_reverse(codeword, dimensions=(1,))

    # Codeword rows are the Merkle leaves; SMCS binds (log_height, width).
    commitment, digest_layers = smcs.commit(codeword.T)

    row_counts = jnp.array(region.row_counts, dtype=dense.dtype)
    column_counts = jnp.array(region.column_counts, dtype=dense.dtype)
    bound = smcs.bind_structure(commitment, row_counts, column_counts)
    return bound, TraceCommitData(
        dense=dense,
        # Column k of the [S, K] message matrix is dense block k (the open
        # evaluates each column at the stack point); the leaves are the
        # committed [S*blowup, K] codeword Merkle-bound just above.
        mle=dense.reshape(K, S).T,
        codeword=codeword.T,
        digest_layers=digest_layers,
        row_counts=row_counts,
        column_counts=column_counts,
        smcs_commitment=commitment,
    )
