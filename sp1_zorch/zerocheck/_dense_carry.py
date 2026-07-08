# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Tight dense-stacked round-buffer addressing for the jagged zerocheck carry
(fractalyze/sp1-zorch#242). Chip segments are stored column-major, each padded
to a static width `w_i` (multiple of 4), concatenated at RUNTIME offsets derived
from `row_counts`. The runtime offsets unpin the stage compile from the shard's
heights (the static buffer shape is one `capacity`), so a single executable
serves any shard of a chip set. `constraint_eval` and the `variant="zerocheck"`
round marker are unchanged — they still see per-chip `(num_cols_i, w_i)` views;
only the storage behind those views is dense."""
from __future__ import annotations
import jax.numpy as jnp
from jax import Array, lax


def pad4(h: int | Array) -> int | Array:
    """SP1's round-0 alignment: real height up to the next multiple of 4. Works
    on a Python int (host/AOT capacity) or elementwise on an int32 Array (the
    Python-int literals stay weakly typed, so an int32 input stays int32)."""
    return ((h + 3) // 4) * 4


def dense_capacity(num_cols: tuple[int, ...], row_counts: tuple[int, ...]) -> int:
    """Static packed length Σ_i num_cols_i * pad4(row_counts_i). In the AOT path
    the row_counts are the a-priori cap heights, so this is the static capacity
    the stage compile keys on."""
    return sum(nc * pad4(h) for nc, h in zip(num_cols, row_counts, strict=True))


def round0_offsets(num_cols: tuple[int, ...], row_counts: Array) -> Array:
    """Runtime i32[num_chips] exclusive-prefix offsets into the dense buffer:
    concat([0, cumsum(num_cols_i * pad4(row_counts_i))[:-1]]). Computed once
    outside the scan from the traced `row_counts`."""
    i32 = jnp.int32
    seg = jnp.asarray(num_cols, i32) * pad4(row_counts.astype(i32))
    # Prepend the exclusive-prefix zero with a bare jnp.pad (one pad op that
    # lowers cleanly on the GPU emitter) rather than a manual concatenate+zeros.
    return jnp.pad(jnp.cumsum(seg)[:-1], (1, 0))


def seg_gather(dense: Array, off_i: Array, num_cols_i: int, w_i: int) -> Array:
    """Chip i's (num_cols_i, w_i) column-major view at runtime offset `off_i`.
    Advanced-index gather with the static index grid `col*w_i + row`; the result
    shape is static so downstream constraint_eval compiles once."""
    i32 = jnp.int32
    grid = (jnp.arange(num_cols_i, dtype=i32)[:, None] * i32(w_i)
            + jnp.arange(w_i, dtype=i32)[None, :])
    return dense[off_i.astype(i32) + grid]


def seg_scatter(dense: Array, off_i: Array, block_i: Array) -> Array:
    """Write chip i's folded (num_cols_i, w_i) block back at `off_i`."""
    return lax.dynamic_update_slice(dense, block_i.reshape(-1), (off_i.astype(jnp.int32),))
