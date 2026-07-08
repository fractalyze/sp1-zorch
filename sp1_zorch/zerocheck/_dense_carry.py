# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Row-major dense-stacked round-buffer addressing for the jagged zerocheck
carry (fractalyze/sp1-zorch#242).

Within a `num_cols` class the chips share one row-major buffer
`[Σ_i pad4(rows_i), num_cols]`: chip `i`'s block is its `[w_i, num_cols]` window
at row offset `off_i`, `w_i = pad4(rows_i)`. The window is exactly the
`zorch.constraint_eval` `start_offset` / `window_rows` contract — an axis-0 row
offset with a static row height — so a chip reads its segment in place from the
shared buffer (`trace=dense, start_offset=off_i, window_rows=w_i`) with no copy,
and the driver's own fold addresses the same segment identically. Runtime
offsets derive from the traced `row_counts`, so the stage compile keys on
(chip set, row capacity), not the shard's heights.

Row-major, not the earlier column-major flat spike: the constraint_eval emitter
windows axis 0 of a rank-2 `[rows, cols]` trace (Phase 2.2), so a chip's view is
a plain `dense[off_i:off_i+w_i, :]` slice / `dynamic_update_slice`, never a
strided `col*w+row` gather.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array, lax


def pad4(h: int | Array) -> int | Array:
    """SP1's round-0 alignment: real height up to the next multiple of 4. Works
    on a Python int (host/AOT capacity) or elementwise on an int32 Array (the
    Python-int literals stay weakly typed, so an int32 input stays int32)."""
    return ((h + 3) // 4) * 4


def dense_capacity(row_counts: tuple[int, ...]) -> int:
    """Static stacked row count Σ_i pad4(row_counts_i) — the row dimension of a
    class's `[Σ w_i, num_cols]` buffer (num_cols is the class's fixed column
    count, applied by the caller). In the AOT path the row_counts are the
    a-priori cap heights, so this is the static row capacity the stage compile
    keys on."""
    return sum(pad4(h) for h in row_counts)


def row_offsets(row_counts: Array) -> Array:
    """Runtime i32[num_chips] exclusive-prefix ROW offsets into the class buffer:
    concat([0, cumsum(pad4(row_counts_i))[:-1]]). Computed once outside the scan
    from the traced `row_counts`; each is a chip's `start_offset` for
    `constraint_eval` and for `seg_view` / `seg_scatter`."""
    i32 = jnp.int32
    seg = pad4(row_counts.astype(i32))
    # Prepend the exclusive-prefix zero with a bare jnp.pad (one pad op that
    # lowers cleanly on the GPU emitter) rather than a manual concatenate+zeros.
    return jnp.pad(jnp.cumsum(seg)[:-1], (1, 0))


def seg_view(dense: Array, off_i: Array, w_i: int) -> Array:
    """Chip i's static `[w_i, num_cols]` row-major window at runtime row offset
    `off_i` — the same axis-0 window `constraint_eval`'s `start_offset` reads, so
    the driver and the marked kernel address the segment identically. The result
    shape is static (`w_i` is closed over), so downstream ops compile once."""
    return lax.dynamic_slice_in_dim(dense, off_i.astype(jnp.int32), w_i, axis=0)


def seg_scatter(dense: Array, off_i: Array, block_i: Array) -> Array:
    """Write chip i's `[w_i, num_cols]` block back at row offset `off_i` (column
    offset 0 — a class buffer's columns are the whole chip)."""
    return lax.dynamic_update_slice(
        dense, block_i, (off_i.astype(jnp.int32), jnp.asarray(0, jnp.int32))
    )
