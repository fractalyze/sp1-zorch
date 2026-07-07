# Copyright 2026 The sp1-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""SP1 zerocheck's batching-coefficient layouts, and the marked constraint fold.

Zerocheck = constraint evaluation + sumcheck reduction. This module owns the
SP1-specific coefficient layouts — the constraint-fold RLC and the GKR
opening-batch weights — and hands the eval + fold to zorch's agnostic
`constraint_eval` combinator so the K constraints never materialize and the
region lowers toward one kernel. The chip's per-row constraint evaluation
(`eval_fn`) is SP1 glue's to supply; its body is riscv-witness's, not
sp1-zorch's. Running the sumcheck reduction over the folded result is out of
this module's scope.

The RLC coefficients are descending powers of the challenge because that is the
order SP1's reference prover folds constraints in (`shifted_powers(1, alpha,
K)[::-1]`); the layout is SP1's, so it lives here rather than in zorch — the
generic ascending power chain is zorch's `poly.univariate.powers`, which both
layouts reorder.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from zorch.constraint_eval import constraint_eval
from zorch.poly.univariate import powers


# Reference: SP1's `chip_powers_of_alpha` constraint-fold coefficients —
# https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs#L520-L524
def rlc_coeffs(alpha: Array, num_constraints: int) -> Array:
    """SP1's RLC coefficients for `num_constraints` constraints: descending
    powers of `alpha`, ``[alpha**(K-1), ..., alpha, 1]``. Empty for ``K = 0``
    — a lookup-only chip (SP1's Byte / Program / Range) has no transition
    constraints and folds to nothing."""
    if num_constraints == 0:
        return jnp.zeros((0,), alpha.dtype)
    return powers(alpha, num_constraints)[::-1]


# Reference: SP1's GKR opening-batch weights, column j carrying beta**(j+1)
# (`gkr_opening_batch_randomness.powers().skip(1)`) —
# https://github.com/fractalyze/sp1/blob/640d8b80c/crates/hypercube/src/prover/shard.rs#L541-L549
def gkr_powers(beta: Array, num_cols: int) -> Array:
    """SP1's GKR column-batching weights ``[beta, beta**2, ..., beta**num_cols]``
    — zorch's ascending chain shifted by one; empty for a zero-column batch,
    mirroring ``rlc_coeffs``' ``K = 0``."""
    return powers(beta, num_cols + 1)[1:]


def constraint_rlc(
    eval_fn: Callable[[Array, Array], Array],
    rows: Array,
    alpha: Array,
    num_constraints: int,
    public_values: Array,
) -> Array:
    """Fold a chip's K constraints over `rows` under `alpha` as one
    `zorch.constraint_eval`. `eval_fn(rows, public_values)` produces `[..., K]`;
    the result drops the K axis. Equivalent to
    ``eval_fn(rows, public_values) @ rlc_coeffs(alpha, K)``, but marked so the
    recognizing emitter lowers it to one kernel. `public_values` rides as a
    declared `aux_operands` operand so the 2-ary eval reads the statement
    without closing over it."""
    return constraint_eval(
        eval_fn,
        rows,
        rlc_coeffs(alpha, num_constraints),
        aux_operands=(public_values,),
    )
